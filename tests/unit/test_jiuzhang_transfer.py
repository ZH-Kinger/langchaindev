"""九章（北京 B200）迁移链：单跳直连，杭州 OSS → /root/nas。

与泰国那条的实质差异（这些差异本身就是测试要盯的点）：
  · 单跳 —— 不经新加坡中转
  · 目的盘是 GPFS **不是 FUSE** —— 不加泰国那套 `--parallel 1 --part-size 5Gi` 降速 flag
  · 单段状态机
"""
import base64
import re

import pytest

from core.jiuzhang_transfer import engine, orchestrator as o
from core.ssh_transfer.paths import SshPathError


# ── 路径：复用泰国链的白名单（安全边界只能有一份实现）────────────────────────

@pytest.mark.parametrize("bad", [
    "oss://b/a$(touch /tmp/pwn)/", "oss://b/a`id`/", "oss://b/../etc/",
    "oss://b/a b/", "oss://b/a;rm -rf x/",
])
def test_injection_blocked_by_shared_whitelist(bad):
    """这条链同样通往生产机，注入即 RCE。白名单是唯一防线。"""
    with pytest.raises(SshPathError):
        o.build_plan(bad)


def test_dest_dir_is_single_source_of_truth():
    """传输器写哪、校验查哪，必须算出同一个串。"""
    p = o.build_plan("oss://wuji-bucket-hangzhou/a/b/")
    assert engine.dest_dir(p.source_prefix, p.dest_rel()) == "/root/nas/a/b/"
    assert engine.dest_dir("a/b/", "c/") == "/root/nas/c/"


def test_job_id_idempotent_same_day():
    """同一天同一对路径 → 同一个 job（防连点重复起任务）；换路径则不同。"""
    p = o.build_plan("oss://my-bucket/x/")
    assert o._job_id(p) == o._job_id(p)
    assert o._job_id(p) != o._job_id(o.build_plan("oss://my-bucket/y/"))
    assert o._job_id(p).startswith("jz-")


def test_short_bucket_name_rejected():
    """OSS 桶名下限 3 位。（写上面那条测试时用了 `b1`，被白名单挡下——
    说明这条约束是真在生效的，顺手锁住。）"""
    with pytest.raises(SshPathError, match="桶名非法"):
        o.build_plan("oss://b1/x/")


# ── ossutil flags：GPFS 不需要 FUSE 那套降速 ────────────────────────────────

def test_flags_do_not_carry_fuse_workaround():
    """`--part-size 5Gi` / `--parallel 1` 是 ossfs2(FUSE) 只能顺序写的补丁。
    九章是 GPFS，加上去只会白白牺牲并发。"""
    f = engine._cp_flags()
    assert "--part-size" not in f
    assert "--parallel 1" not in f
    assert "--parallel" in f and "--job" in f


def test_flags_use_singular_job_flag():
    """ossutil 2.x 是 `--job`（单数）；`--jobs` 不存在，写了会 unknown flag。"""
    f = engine._cp_flags()
    assert "--job " in f and "--jobs" not in f


def test_flags_force_and_update_present():
    """`-f` 必须给：detached 跑时交互提示会永久挂住。`-u` 让续跑不重下。"""
    f = engine._cp_flags()
    assert " -u" in f and " -f" in f


def test_flags_pin_endpoint_not_remote_config():
    """显式给 endpoint —— 不吃九章 ~/.ossutilconfig 的默认值（人工维护，改了会静默 403）。"""
    assert "oss-cn-hangzhou" in engine._cp_flags()


# ── 下发脚本：base64 传递，$HOME 要能展开 ──────────────────────────────────

def _script_for(**kw):
    cap = {}
    real = engine.run
    engine.run = lambda s, **k: cap.setdefault("s", s) and None or (0, "LAUNCHED", "")
    try:
        engine.start_pull("jz-t", source_bucket="bkt", source_prefix="a/b/", **kw)
    finally:
        engine.run = real
    return cap["s"]


def test_inner_command_passed_as_base64():
    """内层命令 base64 传递 —— 代码库明令禁止手写嵌套引号（泰国链栽过两次）。"""
    s = _script_for()
    m = re.search(r"echo (\S+) \| base64 -d", s)
    assert m, "内层命令没走 base64"
    inner = base64.b64decode(m.group(1)).decode()
    assert "ossutil cp -r oss://bkt/a/b/" in inner


def test_checkpoint_dir_is_shell_expandable():
    """断点目录含 $HOME，**不能被 shlex.quote 单引号包住** —— 否则会建出一个字面量
    叫 `$HOME` 的目录，断点续传永远命中不了。"""
    s = _script_for()
    inner = base64.b64decode(re.search(r"echo (\S+) \| base64 -d", s).group(1)).decode()
    assert '--checkpoint-dir "$JD/ckpt"' in inner
    assert "'$HOME" not in inner
    assert "export JD" in s, "内层 bash -c 是子进程，不 export 拿不到 $JD"


def test_launch_result_must_be_confirmed():
    """既无 LAUNCHED 也无 ALREADY_RUNNING = 结果不可知 → 当失败。
    「不知道起没起」比「以为起了其实没起」好排查得多。"""
    real = engine.run
    engine.run = lambda s, **k: (0, "no markers here", "")
    try:
        with pytest.raises(engine.JiuzhangError, match="无法确认"):
            engine.start_pull("jz-t", source_bucket="b", source_prefix="a/")
    finally:
        engine.run = real


# ── 审批门 ──────────────────────────────────────────────────────────────────

def test_unknown_size_forces_approval():
    """估算失败 fail-safe 当作需审批 —— 不放行未知大小的迁移。"""
    assert o.needs_approval(0, size_known=False) is True
    assert o.needs_approval(2 * 1024 ** 4) is True
    assert o.needs_approval(1024 ** 3) is False


# ── 轮询：rc 语义与失败判定 ────────────────────────────────────────────────

@pytest.mark.parametrize("rc,expect", [(0, "DONE"), (24, "DONE"), (1, "FAILED"), (2, "FAILED")])
def test_rc_semantics(monkeypatch, rc, expect):
    """0 成功；24 = 源文件传输中消失，非致命（与泰国链一致）。"""
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, f"ALIVE=0\nRC={rc}\nLOG=1", ""))
    assert engine.poll("jz-t")["status"] == expect


def test_ssh_failure_is_not_task_failure(monkeypatch):
    """SSH 不通 ≠ 任务失败。必须抛出去让上层保持在途 —— 误判失败会触发重传。"""
    monkeypatch.setattr(engine, "run", lambda s, **k: (255, "", "connection refused"))
    with pytest.raises(engine.JiuzhangError):
        engine.poll("jz-t")


def test_dead_without_rc_is_failure(monkeypatch):
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, "ALIVE=0\nRC=NONE\nLOG=1", ""))
    st = engine.poll("jz-t")
    assert st["status"] == "FAILED" and "退出码" in st["error"]


# ── 进度解析 ────────────────────────────────────────────────────────────────

def test_progress_counts_skipped(monkeypatch):
    """续跑时 `-u` 跳过的量记在 skipped 里；只算 done 会让进度偏低、ETA 偏长。"""
    line = "Copy done:(10 files,2.00 GiB) skipped:(5 files,1.00 GiB), 50%, avg 100.0 MiB/s"
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, line, ""))
    p = engine.progress("jz-t")
    assert p["bytes_done"] == 3 * 1024 ** 3
    assert p["objects_done"] == 15
    assert p["speed_bps"] == 100 * 1024 ** 2


def test_progress_never_returns_pct(monkeypatch):
    """刻意不返回百分比：ossutil 在 Scanning 阶段的分母是「已扫到的量」、会虚高。"""
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, "done:(1 files,1.00 GiB), 99%", ""))
    assert "pct" not in engine.progress("jz-t")


# ── 估算 ────────────────────────────────────────────────────────────────────

def test_estimate_regex_anchored(monkeypatch):
    """正则必须锚到具体那行 —— 松散写法会先命中表头、跨行吞进 object count，
    把 22MB 读成 3B，直接绕过审批门（泰国链踩过）。"""
    out = ("total object count: 3\n"
           "total object sum size: 23068672\n")
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, out, ""))
    assert engine.estimate_source("b", "p/") == (23068672, 3, True)


def test_estimate_failure_reports_not_ok(monkeypatch):
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, "no du output", ""))
    assert engine.estimate_source("b", "p/") == (0, 0, False)


# ── 意图互斥：两条链都含「迁移」，判据必须能分开 ──────────────────────────

@pytest.mark.parametrize("text,jz,xw", [
    # 九章：判据要求出现「九章/jiuzhang/b200」
    ("数据迁移(九章b200)", True, False),
    ("九章b200", True, False),
    ("迁移到九章", True, False),
    ("九章集群搬运", True, False),
    # 曦望（泰国）：正式名与旧称都要认 —— 改名不能让历史话术失效
    ("数据迁移(曦望)", False, True),
    ("曦望迁移", False, True),
    ("数据迁移（泰国H200）", False, True),
    ("迁移到泰国", False, True),
    # 泛化话术两条都不该抢
    ("跨云迁移", False, False),
    ("桶间迁移", False, False),
])
def test_intents_are_mutually_exclusive(text, jz, xw):
    """**同时命中就是 bug** —— messages.py 里靠书写顺序先到先得，
    两条都命中意味着改一次顺序就会静默换掉目的机房。"""
    from core.feishu_bot import messages as m
    assert m._is_jiuzhang_transfer_intent(text) is jz, text
    assert m._is_ssh_transfer_intent(text) is xw, text
    assert not (m._is_jiuzhang_transfer_intent(text) and m._is_ssh_transfer_intent(text)), \
        f"「{text}」两条链都命中，目的机房会取决于代码书写顺序"


def test_fast_job_not_mistaken_for_launch_failure(monkeypatch):
    """小任务可能比下发校验里的 `sleep 3` 还快跑完（实测 66 MiB / 11 对象 = 0.9 秒）。

    那时进程已正常退出、`kill -0` 必然失败。只按存活判定会把**已经成功**误报成
    「起来即死」——而小任务正是大家用来验证链路的那种，这个 bug 会让人以为链路是坏的。
    """
    monkeypatch.setattr(engine, "run", lambda s, **k: (0, "ALREADY_DONE rc=0", ""))
    engine.start_pull("jz-t", source_bucket="bkt", source_prefix="a/")   # 不抛即通过


def test_launch_script_checks_rc_before_liveness():
    """脚本里 rc 判定必须排在 kill -0 之前。"""
    s = _script_for()
    # 脚本里有两处 kill -0（幂等检查、下发后校验），取**最后**那处比
    rc_check = s.rindex('if [ -f "$JD/pull.rc" ]')
    last_kill = s.rindex("kill -0")
    assert rc_check < last_kill, "下发后校验必须先看 rc、再看进程存活"
