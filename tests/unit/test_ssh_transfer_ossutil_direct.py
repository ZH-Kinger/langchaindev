"""段2 泰国 ossutil 直拉引擎 + 端到端校验 + 引擎分派 的单测。

设计原则（本次开发的教训直接落成断言）：
- 双跳 SSH 的引号地狱：外层只允许「一层双引号 + 纯 base64」，任何 `$`/反引号/反斜杠/换行
  漏进载荷都必须测出来（开发中踩了两次，一次变量被吃掉、一次 unexpected EOF）。
- 生成的远端脚本���律过 `bash -n`，并在沙箱里真跑桩验行为——只断言命令串会漏掉语义错误。
- 校验层 fail-closed：拿不到清单/SSH 挂 → 判失败。fail-open 会把可能缺数据的迁移标成成功。
- 变异测试：把修复点改坏，断言用例会红（否则用例等于没写）。
"""
import base64
import re
import subprocess
import textwrap

import pytest

from core.ssh_transfer import engine_ossutil as eo
from core.ssh_transfer import orchestrator as orch
from core.ssh_transfer import engine_ssh
from core.ssh_transfer import verify as _v


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _inner_of(cmd: str) -> str:
    """从外层命令里取出 base64 载荷并解码；顺带断言载荷纯净。"""
    m = re.search(r'"echo ([A-Za-z0-9+/=]+) \| base64 -d \| bash"', cmd)
    assert m, f"外层命令不是「一层双引号 + 纯 base64」形态: {cmd[:200]}"
    payload = m.group(1)
    assert not re.search(r"[$`\\\n'\"]", payload), "base64 载荷含需要转义的字符"
    return base64.b64decode(payload).decode()


def _bash_n(script: str, tmp_path) -> None:
    p = tmp_path / "s.sh"
    p.write_text(script, encoding="utf-8", newline="\n")
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, f"生成的远端脚本语法错误:\n{r.stderr}\n---\n{script}"


@pytest.fixture
def eng(monkeypatch):
    """桩掉 bot→SGP 的 run，捕获所有下发的命令。"""
    calls = []

    def fake_run(cmd, timeout=120):
        calls.append(cmd)
        return 0, calls_out.get("out", "LAUNCHED pid=42"), calls_out.get("err", "")

    calls_out = {}
    monkeypatch.setattr(eo, "_run_sgp", fake_run)
    s = eo.settings
    monkeypatch.setattr(s, "THAI_USER", "wuji")
    monkeypatch.setattr(s, "THAI_HOST", "203.0.113.9")
    monkeypatch.setattr(s, "THAI_PORT", "40002")
    monkeypatch.setattr(s, "THAI_DEST_ROOT", "/mnt/d/296834/Team@x.tech/data")
    monkeypatch.setattr(s, "SGP_OSS_BUCKET", "wuji-sing")
    monkeypatch.setattr(s, "THAI_OSS_ENDPOINT", "oss-ap-southeast-1.aliyuncs.com")
    monkeypatch.setattr(s, "THAI_OSS_REGION", "ap-southeast-1")
    monkeypatch.setattr(s, "THAI_OSSUTIL_JOBS", "32")
    monkeypatch.setattr(s, "THAI_OSSUTIL_PARALLEL", "8")
    monkeypatch.setattr(s, "THAI_WORK_DIR", "$HOME/.ossutil_jobs")

    # 不能在类体里写 `calls = calls`：同名赋值让编译器改用 LOAD_NAME、跳过外层函数作用域 →
    # NameError。改成建好类再挂属性，最省心。
    class E:
        pass
    E.calls = calls
    E.out = calls_out
    return E


# ── 双跳形态与脚本语法 ────────────────────────────────────────────────────────

def test_all_remote_scripts_are_pure_base64_and_syntactically_valid(eng, tmp_path):
    """五个入口的远端脚本：载荷纯净 + bash -n 通过。"""
    eng.out["out"] = "LAUNCHED pid=42"
    eo.start_stage2("sgp-a1", source_prefix="tp/we/20260715/", dest_rel="we/20260715/")
    eng.out["out"] = "PID=42\nALIVE=1\nRC=NONE\nLOG=1"
    eo.poll_stage("sgp-a1")
    eng.out["out"] = "done:(5 files,1.5 GiB) 3%, avg 100 MiB/s"
    eo.stage_progress("sgp-a1")
    eng.out["out"] = "Error: boom"
    eo.failure_detail("sgp-a1")
    eng.out["out"] = "KILLED"
    eo.cancel("sgp-a1")
    assert len(eng.calls) == 5
    for cmd in eng.calls:
        _bash_n(_inner_of(cmd), tmp_path)


def test_dest_path_has_no_double_slash(eng):
    """dest_rel 自带尾斜杠，直接拼会拼出 `//`（开发中真出现过）。"""
    eo.start_stage2("sgp-a1", source_prefix="tp/we/20260715/", dest_rel="we/20260715/")
    inner = _inner_of(eng.calls[0])
    # 只看两个位置参数那一段，并摘掉 `oss://` 的合法双斜杠
    seg = inner.split("ossutil cp -r")[1].split(" -e ")[0].replace("oss://", "")
    assert "//" not in seg, seg
    assert "/data/we/20260715/ " in inner


@pytest.mark.parametrize("dest_rel", ["", "we/20260715/", "we/20260715", "/we/20260715/"])
def test_dest_normalised_for_all_dest_rel_shapes(eng, dest_rel):
    eng.calls.clear()
    eo.start_stage2("sgp-a1", source_prefix="tp/x/", dest_rel=dest_rel)
    inner = _inner_of(eng.calls[0])
    seg = inner.split("ossutil cp -r")[1].split(" -e ")[0]
    assert "//" not in seg.replace("oss://", ""), seg


# ── flag 正确性（真机核实过的事实，不能被改坏）────────────────────────────────

def test_uses_singular_job_flag_not_plural(eng):
    """ossutil 2.x 是 `--job`；`--jobs` 会 unknown flag 直接挂（段1 踩过）。"""
    eo.start_stage2("sgp-a1", source_prefix="tp/x/")
    inner = _inner_of(eng.calls[0])
    assert "--job 32" in inner
    assert "--jobs" not in inner


def test_force_flag_present_because_detached(eng):
    """nohup detached 跑，任何交互确认都会让它永久挂住 → 必须有 -f。"""
    inner_before = None
    eo.start_stage2("sgp-a1", source_prefix="tp/x/")
    inner_before = _inner_of(eng.calls[0])
    assert re.search(r"(^|\s)-f(\s|\")", inner_before), inner_before


def test_update_flag_present_for_resume(eng):
    eo.start_stage2("sgp-a1", source_prefix="tp/x/")
    assert re.search(r"(^|\s)-u(\s|\")", _inner_of(eng.calls[0]))


def test_endpoint_and_region_always_explicit(eng):
    """不吃泰国 ~/.ossutilconfig 默认值：那文件人工维护，被改回杭州/加速域名会静默变慢或 403。"""
    inner = (eo.start_stage2("sgp-a1", source_prefix="tp/x/"), _inner_of(eng.calls[0]))[1]
    assert "-e oss-ap-southeast-1.aliyuncs.com" in inner
    assert "--region ap-southeast-1" in inner


def test_checkpoint_dir_under_job_dir(eng):
    eo.start_stage2("sgp-a1", source_prefix="tp/x/")
    inner = _inner_of(eng.calls[0])
    assert '--checkpoint-dir \\"$JD/ckpt\\"' in inner or '--checkpoint-dir "$JD/ckpt"' in inner


@pytest.mark.parametrize("jobs,par,want_j,want_p", [
    ("32", "8", 32, 8), ("", "", 32, 8), ("abc", "xyz", 32, 8),
    ("0", "0", 1, 1), ("-5", "-5", 1, 1), (None, None, 32, 8),
])
def test_concurrency_config_never_crashes(eng, monkeypatch, jobs, par, want_j, want_p):
    """配置写错不能让下发抛异常（它只是个性能旋钮）。"""
    monkeypatch.setattr(eo.settings, "THAI_OSSUTIL_JOBS", jobs)
    monkeypatch.setattr(eo.settings, "THAI_OSSUTIL_PARALLEL", par)
    flags = eo._cp_flags()
    assert f"--job {want_j}" in flags and f"--parallel {want_p}" in flags


# ── marker 布局：必须能认领 2026-07-31 人工切换那单 ────────────────────────────

def test_marker_layout_matches_manual_cutover(eng):
    """布局与人工那单逐字一致，否则部署后认领不到、19.5TiB 白重传。"""
    eo.start_stage2("sgp-6796f12de0af", source_prefix="tp/x/")
    inner = _inner_of(eng.calls[0])
    assert 'JD="$HOME/.ossutil_jobs/sgp-6796f12de0af"' in inner
    for m in ("$JD/stage2.pid", "$JD/stage2.rc", "$JD/stage2.log", "$JD/ckpt"):
        assert m in inner, m


def test_relaunch_is_idempotent_when_alive(eng):
    """已有存活进程 → 脚本自己 ALREADY_RUNNING 退出，不重复起（会双写同一批文件）。"""
    eo.start_stage2("sgp-a1", source_prefix="tp/x/")
    inner = _inner_of(eng.calls[0])
    assert "ALREADY_RUNNING" in inner
    i_guard = inner.index("ALREADY_RUNNING")
    i_launch = inner.index("nohup")
    assert i_guard < i_launch, "存活守卫必须在 nohup 之前"


# ── poll 状态判定（每一条都对应一种真实事故形态）───────────────────────────────

@pytest.mark.parametrize("out,want_status,want_rc", [
    ("PID=42\nALIVE=1\nRC=NONE\nLOG=1", "RUNNING", None),
    ("PID=42\nALIVE=0\nRC=0\nLOG=1", "DONE", 0),
    ("PID=42\nALIVE=0\nRC=1\nLOG=1", "FAILED", 1),
    ("PID=42\nALIVE=0\nRC=137\nLOG=1", "FAILED", 137),
    # 进程没了又没写 rc：被 kill / OOM / 机器重启。必须判失败，不能当在途永远等
    ("PID=42\nALIVE=0\nRC=NONE\nLOG=1", "FAILED", None),
    # 连日志都没有：下发没生效或工作目录被清
    ("PID=0\nALIVE=0\nRC=NONE\nLOG=0", "FAILED", None),
    # rc 已写但进程还在（收尾中）→ 以 rc 为准
    ("PID=42\nALIVE=1\nRC=0\nLOG=1", "DONE", 0),
])
def test_poll_status_matrix(eng, out, want_status, want_rc):
    eng.out["out"] = out
    st = eo.poll_stage("sgp-a1")
    assert st["status"] == want_status, out
    assert st["rc"] == want_rc


def test_poll_ssh_failure_raises_not_reports_failed(eng, monkeypatch):
    """SSH 不通 ≠ 任务失败。误判失败会触发重试、白跑几十小时。"""
    monkeypatch.setattr(eo, "_run_sgp", lambda cmd, timeout=120: (255, "", "conn refused"))
    with pytest.raises(eo.SshTransferError):
        eo.poll_stage("sgp-a1")


@pytest.mark.parametrize("bad_rc", ["", "abc", "1.5", "  "])
def test_poll_non_numeric_rc_is_not_treated_as_success(eng, bad_rc):
    """rc 文件被写坏（进程在建文件和写入之间被杀）绝不能当成功。"""
    eng.out["out"] = f"PID=42\nALIVE=0\nRC={bad_rc}\nLOG=1"
    st = eo.poll_stage("sgp-a1")
    assert st["status"] == "FAILED"


# ── 下发结果判定 ─────────────────────────────────────────────────────────────

def test_launch_dead_raises(eng):
    eng.out["out"] = "LAUNCH_DEAD pid=42"
    with pytest.raises(eo.SshTransferError, match="起来即退出"):
        eo.start_stage2("sgp-a1", source_prefix="tp/x/")


def test_launch_unconfirmed_raises(eng):
    """既无 LAUNCHED 也无 ALREADY_RUNNING → 结果不可确认，必须报错而不是当成功。"""
    eng.out["out"] = "some unexpected noise"
    with pytest.raises(eo.SshTransferError, match="无法确认"):
        eo.start_stage2("sgp-a1", source_prefix="tp/x/")


def test_already_running_is_accepted(eng):
    eng.out["out"] = "ALREADY_RUNNING pid=42"
    eo.start_stage2("sgp-a1", source_prefix="tp/x/")     # 不抛


def test_missing_bucket_config_raises_clearly(eng, monkeypatch):
    monkeypatch.setattr(eo.settings, "SGP_OSS_BUCKET", "")
    with pytest.raises(eo.SshTransferError, match="SGP_OSS_BUCKET"):
        eo.start_stage2("sgp-a1", source_prefix="tp/x/")


def test_missing_dest_root_raises_clearly(eng, monkeypatch):
    monkeypatch.setattr(eo.settings, "THAI_DEST_ROOT", "")
    with pytest.raises(eo.SshTransferError, match="THAI_DEST_ROOT"):
        eo.start_stage2("sgp-a1", source_prefix="tp/x/")


def test_missing_thai_host_raises_clearly(eng, monkeypatch):
    monkeypatch.setattr(eo.settings, "THAI_HOST", "")
    with pytest.raises(eo.SshTransferError, match="THAI_HOST"):
        eo.start_stage2("sgp-a1", source_prefix="tp/x/")


# ── 进度解析（真机日志原文）──────────────────────────────────────────────────

_REAL_LINE = ("Estimated 76720 objects,19.527 TiB, Copy... done:(553 files,40.882 GiB) "
              "skipped:(6 files,0 B), 0.210%, avg 289.439 MiB/s")


def test_progress_parses_real_ossutil_line(eng):
    eng.out["out"] = _REAL_LINE
    p = eo.stage_progress("sgp-a1")
    assert p["bytes_done"] == int(40.882 * 1024 ** 3)
    assert p["speed_bps"] == int(289.439 * 1024 ** 2)


def test_progress_never_returns_pct(eng):
    """ossutil 的百分比在 Scanning 阶段分母是「已扫到的量」、会虚高，而 progress_line 优先
    采用 job["pct"] → 透出去就把假百分比钉在卡片上。必须留 None 让上层算真值。"""
    eng.out["out"] = _REAL_LINE
    assert eo.stage_progress("sgp-a1")["pct"] is None
    # 就算日志里百分比很唬人也不能透出
    eng.out["out"] = "Copy... done:(1 files,1 GiB) 99.9%, avg 1 MiB/s"
    assert eo.stage_progress("sgp-a1")["pct"] is None


@pytest.mark.parametrize("unit,mult", [("KiB", 1024), ("MiB", 1024 ** 2),
                                       ("GiB", 1024 ** 3), ("TiB", 1024 ** 4)])
def test_progress_unit_scaling(eng, unit, mult):
    eng.out["out"] = f"Copy... done:(1 files,2 {unit}) 5%, avg 1 {unit}/s"
    p = eo.stage_progress("sgp-a1")
    assert p["bytes_done"] == 2 * mult
    assert p["speed_bps"] == 1 * mult


def test_progress_tolerates_garbage(eng):
    eng.out["out"] = "\x00\x01 not a progress line at all"
    p = eo.stage_progress("sgp-a1")
    assert p == {"bytes_done": None, "pct": None, "speed_bps": None}


def test_progress_ssh_failure_degrades_not_raises(eng, monkeypatch):
    monkeypatch.setattr(eo, "_run_sgp", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert eo.stage_progress("sgp-a1")["bytes_done"] is None


def test_progress_script_strips_cr_and_drops_truncated_first_line(eng, tmp_path):
    """ossutil 用 \\r 刷屏，必须 tr 成换行；按字节取尾窗必然切在行中，截断时要丢首行。"""
    eo.stage_progress("sgp-a1")
    inner = _inner_of(eng.calls[0])
    assert r"tr '\r' '\n'" in inner
    assert "tail -n +2" in inner
    assert "-gt 4000" in inner
    _bash_n(inner, tmp_path)


# ── 失败明细 ─────────────────────────────────────────────────────────────────

def test_failure_detail_clips_and_keeps_error_lines(eng):
    eng.out["out"] = "\n".join(["Error: AccessDenied for bucket", "x" * 500, "noise"])
    d = eo.failure_detail("sgp-a1")
    assert "AccessDenied" in d
    assert len(d) <= eo._DETAIL_MAX
    assert all(len(l) <= eo._DETAIL_LINE_MAX for l in d.splitlines())


def test_failure_detail_ssh_failure_returns_empty(eng, monkeypatch):
    monkeypatch.setattr(eo, "_run_sgp", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert eo.failure_detail("sgp-a1") == ""


# ── 引擎分派：按 job 记录而非当前配置 ─────────────────────────────────────────

def test_dispatch_follows_job_record_not_current_config(monkeypatch):
    """任务跑起来后有人改了 SSH_STAGE2_MODE，绝不能拿错引擎去查 marker
    （查不到 → 误判「进程异常退出」→ 把正常跑着的几十小时任务标成失败）。"""
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_MODE", "rsync")
    assert orch._stage2_engine_of({"stage2_mode": "ossutil"}) is eo
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_MODE", "ossutil")
    assert orch._stage2_engine_of({"stage2_mode": "rsync"}) is engine_ssh


def test_legacy_job_without_mode_falls_back_to_rsync():
    """改动前建的老任务没有 stage2_mode，必须回退 rsync（与它们当时的实际执行方式一致）。
    **绝不能默认 ossutil**：那会把所有历史 job 误判成「查不到 marker」→ 失败。"""
    assert orch._stage2_engine_of({}) is engine_ssh
    assert orch.stage2_mode_of({}) == orch.STAGE2_MODE_RSYNC


def test_handoff_marks_manual_ossutil_takeover():
    """2026-07-31 人工切换那一单只有 handoff、没有 stage2_mode。不认它就会拿 rsync 探针去查
    SGP 上不存在的 marker → 把正常跑着的 19 小时任务误判失败、还可能重传 19.5TiB。"""
    job = {"handoff": {"engine": "ossutil_thai_direct",
                       "work_dir": "/home/wuji/.ossutil_jobs/sgp-6796f12de0af"}}
    assert orch.stage2_mode_of(job) == orch.STAGE2_MODE_OSSUTIL
    assert orch._stage2_engine_of(job) is eo


def test_stage2_mode_field_beats_handoff():
    """显式字段是最权威的事实，优先于人工接管标记。"""
    job = {"stage2_mode": "rsync", "handoff": {"engine": "ossutil_thai_direct"}}
    assert orch.stage2_mode_of(job) == orch.STAGE2_MODE_RSYNC


@pytest.mark.parametrize("handoff", [
    None, {}, {"engine": ""}, {"engine": "something_else"}, "not-a-dict", 42, [1, 2],
])
def test_malformed_handoff_falls_back_to_rsync(handoff):
    """handoff 是历史/人工写入的自由结构，畸形值不能让分派抛异常或误判成 ossutil。"""
    assert orch.stage2_mode_of({"handoff": handoff}) == orch.STAGE2_MODE_RSYNC


@pytest.mark.parametrize("raw,want", [
    ("ossutil", "ossutil"), ("OSSUTIL", "ossutil"), (" ossutil ", "ossutil"),
    ("rsync", "rsync"), ("RSYNC", "rsync"),
    ("", "rsync"), (None, "rsync"), ("bogus", "rsync"), (123, "rsync"),
])
def test_stage2_mode_field_parsing(raw, want):
    assert orch.stage2_mode_of({"stage2_mode": raw}) == want


@pytest.mark.parametrize("mode,direct", [
    ("ossutil", True), ("", True), ("OSSUTIL", True), (None, True),
    ("rsync", False), ("RSYNC", False), (" rsync ", False),
])
def test_mode_switch_parsing(monkeypatch, mode, direct):
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_MODE", mode)
    assert orch._stage2_direct() is direct


def test_start_stage_records_mode(monkeypatch):
    """job 必须记下实际用了哪个引擎，否则后续轮询无从分派。"""
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_MODE", "ossutil")
    monkeypatch.setattr(orch, "_claim_stage_launch", lambda *a: True)
    monkeypatch.setattr(orch, "_save", lambda j: None)
    monkeypatch.setattr(eo, "start_stage2", lambda *a, **k: None)
    job = {"job_id": "sgp-a1", "source_prefix": "tp/x/", "dest_rel": ""}
    orch._start_stage(job, orch.STAGE_STAGE2)
    assert job["stage2_mode"] == "ossutil"
    assert job["stage"] == orch.STAGE_STAGE2


def test_start_stage_records_rsync_mode(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_MODE", "rsync")
    monkeypatch.setattr(orch, "_claim_stage_launch", lambda *a: True)
    monkeypatch.setattr(orch, "_save", lambda j: None)
    monkeypatch.setattr(engine_ssh, "start_stage2", lambda *a, **k: None)
    job = {"job_id": "sgp-a1", "source_prefix": "tp/x/", "dest_rel": ""}
    orch._start_stage(job, orch.STAGE_STAGE2)
    assert job["stage2_mode"] == "rsync"


# ── 段下发锁（auditor MED-1）──────────────────────────────────────────────────

def test_stage_launch_lock_only_one_winner(monkeypatch):
    store = {}

    class R:
        def set(self, k, v, nx=False, ex=None):
            if nx and k in store:
                return False
            store[k] = v
            return True
    monkeypatch.setattr(orch, "get_redis", lambda: R())
    assert orch._claim_stage_launch("sgp-a1", orch.STAGE_STAGE2) is True
    assert orch._claim_stage_launch("sgp-a1", orch.STAGE_STAGE2) is False
    # 不同 job / 不同段互不影响
    assert orch._claim_stage_launch("sgp-a2", orch.STAGE_STAGE2) is True
    assert orch._claim_stage_launch("sgp-a1", orch.STAGE_STAGE1) is True


def test_stage_launch_lock_open_when_redis_down(monkeypatch):
    """Redis 挂了必须放行：宁可退回改动前的良性重复，也不能永久起不了段2。"""
    monkeypatch.setattr(orch, "get_redis", lambda: None)
    assert orch._claim_stage_launch("sgp-a1", orch.STAGE_STAGE2) is True

    def boom():
        raise OSError("redis down")
    monkeypatch.setattr(orch, "get_redis", boom)
    assert orch._claim_stage_launch("sgp-a1", orch.STAGE_STAGE2) is True


def test_loser_of_launch_race_touches_nothing(monkeypatch):
    """输家一改 job 就会把赢家写回 Redis 的状态覆盖回去。"""
    monkeypatch.setattr(orch, "_claim_stage_launch", lambda *a: False)
    saved = []
    monkeypatch.setattr(orch, "_save", lambda j: saved.append(dict(j)))
    job = {"job_id": "sgp-a1", "stage": orch.STAGE_STAGE1, "source_prefix": "tp/x/"}
    before = dict(job)
    orch._start_stage(job, orch.STAGE_STAGE2)
    assert job == before, "输家改动了 job"
    assert not saved, "输家写了 Redis"


# ── 端到端校验（fail-closed）──────────────────────────────────────────────────

def _job():
    return {"job_id": "sgp-a1", "source_prefix": "tp/x/", "dest_rel": "x/",
            "stage": orch.STAGE_STAGE2, "stage2_mode": "ossutil"}


def test_verify_pass_marks_done(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", True)
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(v, "verify_stage2", lambda job, samples=5: {"passed": True, "summary": "ok"})
    job = _job()
    orch._run_stage2_verify(job)
    assert job["stage"] != orch.STAGE_FAILED
    assert job["verify"]["passed"] is True


def test_verify_fail_marks_failed(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", True)
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(v, "verify_stage2",
                        lambda job, samples=5: {"passed": False, "summary": "L1 ✗ 少 7 个"})
    job = _job()
    orch._run_stage2_verify(job)
    assert job["stage"] == orch.STAGE_FAILED
    assert "校验未通过" in job["error"]
    assert "少 7 个" in job["error_detail"]


def test_verify_crash_is_fail_closed(monkeypatch):
    """校验本身跑不动时「不知道对不对」必须当「不对」—— fail-open 正是要防的事故。"""
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", True)
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(v, "verify_stage2",
                        lambda job, samples=5: (_ for _ in ()).throw(RuntimeError("列不到清单")))
    job = _job()
    orch._run_stage2_verify(job)
    assert job["stage"] == orch.STAGE_FAILED
    assert "无法完成" in job["error"]


# ── 校验锁与 poll_once 晋级判断的衔接（auditor BLOCK-1：我自己引入过的回归）────────
#
# 这一组**必须经 poll_once**，不能直接调 _run_stage2_verify —— 那个洞恰恰在调用方的晋级
# 判断上：`stage != FAILED` 会把「抢不到锁、本次没校验」当成「校验通过」→ 零校验宣布 DONE。
# 之前 4 条 verify 用例全是直接调，所以全量 1920 绿也照不到。

def _poll_job(mode="ossutil"):
    return {"job_id": "sgp-a1", "stage": orch.STAGE_STAGE2, "stage2_mode": mode,
            "source_prefix": "tp/x/", "dest_rel": "x/", "bytes_total": 100}


@pytest.fixture
def poll_done(monkeypatch):
    """段2 探针恒报 DONE，且屏蔽 Redis 写入/采样，只观察晋级判断。"""
    monkeypatch.setattr(eo, "poll_stage",
                        lambda jid: {"status": "DONE", "rc": 0, "alive": False, "error": ""})
    monkeypatch.setattr(orch, "_sample_progress", lambda *a: None)
    saved = []
    monkeypatch.setattr(orch, "_save", lambda j: saved.append(dict(j)))
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", True)
    return saved


def test_verify_lock_loser_must_not_declare_done(monkeypatch, poll_done):
    """抢不到校验锁 → 保持 STAGE2、无 verify 记录。**绝不能** DONE：
    那就是零校验宣布成功 + 推「✅ 迁移完成」卡，正是校验层要防的事。"""
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: False)
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_STAGE2, "输家把任务判成了 DONE（零校验）"
    assert "verify" not in job


def test_verify_lock_winner_does_declare_done(monkeypatch, poll_done):
    """对照组：抢到锁且校验通过 → 正常晋级 DONE（免得上一条靠「永不 DONE」蒙过）。"""
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: True)
    monkeypatch.setattr(orch, "_release_verify", lambda jid: None)
    monkeypatch.setattr(_v, "verify_stage2", lambda job, samples=5: {"passed": True, "summary": "ok"})
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_DONE
    assert job["verify"]["passed"] is True


def test_verify_lock_loser_reads_back_winner_conclusion(monkeypatch, poll_done):
    """赢家已落 DONE 时，输家必须回读采用，**不能**用自己那份陈旧 STAGE2 覆盖回去
    （与 _start_stage 输家分支同一个坑）。顺带省掉一趟 10~30 分钟的重跑。"""
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: False)
    winner = {"job_id": "sgp-a1", "stage": orch.STAGE_DONE, "stage2_mode": "ossutil",
              "source_prefix": "tp/x/", "dest_rel": "x/",
              "verify": {"passed": True, "summary": "ok"}}
    monkeypatch.setattr(orch, "get_job", lambda jid: dict(winner))
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_DONE, "输家把赢家的 DONE 覆盖回 STAGE2 了"
    assert job["verify"]["passed"] is True


def test_verify_lock_loser_adopts_winner_failure(monkeypatch, poll_done):
    """赢家判了 FAILED，输家同样不能翻案成 DONE。"""
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: False)
    monkeypatch.setattr(orch, "get_job", lambda jid: {
        "job_id": "sgp-a1", "stage": orch.STAGE_FAILED, "stage2_mode": "ossutil",
        "source_prefix": "tp/x/", "dest_rel": "x/",
        "verify": {"passed": False, "summary": "L1 ✗"}, "error": "校验未通过"})
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_FAILED


def test_verify_lock_loser_readback_failure_still_holds(monkeypatch, poll_done):
    """回读本身失败（Redis 抖动）时也必须保持在途，不能 fail-open 成 DONE。"""
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: False)
    monkeypatch.setattr(orch, "get_job",
                        lambda jid: (_ for _ in ()).throw(OSError("redis down")))
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_STAGE2


def test_verify_disabled_still_declares_done(monkeypatch, poll_done):
    """开关关掉 = 明确放行，必须能正常完成（否则任务永远卡在 STAGE2）。"""
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", False)
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_DONE
    assert job["verify"]["passed"] is None


def test_verify_failure_via_poll_once_marks_failed(monkeypatch, poll_done):
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: True)
    monkeypatch.setattr(orch, "_release_verify", lambda jid: None)
    monkeypatch.setattr(_v, "verify_stage2",
                        lambda job, samples=5: {"passed": False, "summary": "L3 ✗ 字节不符 7"})
    job = _poll_job()
    orch.poll_once(job)
    assert job["stage"] == orch.STAGE_FAILED
    assert "字节不符 7" in job["error_detail"]


def test_run_stage2_verify_returns_tristate(monkeypatch):
    """直接钉住三态契约本身——调用方全靠它决定要不要晋级。"""
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", True)
    monkeypatch.setattr(orch, "_release_verify", lambda jid: None)

    monkeypatch.setattr(orch, "_claim_verify", lambda jid: True)
    monkeypatch.setattr(_v, "verify_stage2", lambda job, samples=5: {"passed": True, "summary": ""})
    assert orch._run_stage2_verify(_job()) is True          # 通过 → 给了结论

    monkeypatch.setattr(_v, "verify_stage2", lambda job, samples=5: {"passed": False, "summary": ""})
    assert orch._run_stage2_verify(_job()) is True          # 不通过 → 也给了结论（已置 FAILED）

    monkeypatch.setattr(_v, "verify_stage2",
                        lambda job, samples=5: (_ for _ in ()).throw(RuntimeError("x")))
    assert orch._run_stage2_verify(_job()) is True          # 崩了 → 给了结论（FAILED）

    monkeypatch.setattr(orch, "_claim_verify", lambda jid: False)
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    assert orch._run_stage2_verify(_job()) is False         # 抢不到锁 → **不给结论**

    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", False)
    assert orch._run_stage2_verify(_job()) is True          # 配置放行 → 给了结论


def test_mutation_stage_only_check_would_reintroduce_the_hole(monkeypatch, poll_done):
    """变异：把晋级判断退回「只看 stage != FAILED」，断言这一组会红。"""
    monkeypatch.setattr(orch, "_claim_verify", lambda jid: False)
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    job = _poll_job()
    orch._run_stage2_verify(job)                     # 模拟只调不看返回值的旧写法
    would_be_done = job.get("stage") != orch.STAGE_FAILED
    assert would_be_done, "旧写法本应把输家误判为可晋级——这正是 BLOCK-1 的成因"
    assert job["stage"] == orch.STAGE_STAGE2, "函数自身没保持在途"


def test_verify_can_be_disabled_but_records_it(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", False)
    job = _job()
    orch._run_stage2_verify(job)
    assert job["stage"] != orch.STAGE_FAILED
    assert job["verify"]["passed"] is None


# ── verify_stage2 的判定矩阵 ──────────────────────────────────────────────────

def _stub_verify(monkeypatch, src, dst, sample=(1, 0, 0)):
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(v, "_list_source", lambda b, p: src)
    monkeypatch.setattr(v, "_list_dest", lambda d: dst)
    monkeypatch.setattr(v, "_sample_compare", lambda b, p, d, k: sample)
    monkeypatch.setattr(v.settings, "SGP_OSS_BUCKET", "wuji-sing")
    monkeypatch.setattr(v.settings, "THAI_DEST_ROOT", "/mnt/d")
    return v


def test_verify_all_match_passes(monkeypatch):
    v = _stub_verify(monkeypatch, {"a": 10, "b": 20}, {"a": 10, "b": 20})
    r = v.verify_stage2(_job())
    assert r["passed"] is True


def test_verify_missing_file_fails(monkeypatch):
    v = _stub_verify(monkeypatch, {"a": 10, "b": 20}, {"a": 10})
    r = v.verify_stage2(_job())
    assert r["passed"] is False and r["missing"] == 1


def test_verify_size_mismatch_fails(monkeypatch):
    """单文件被截断 —— 这是 L3 存在的唯一理由。"""
    v = _stub_verify(monkeypatch, {"a": 10}, {"a": 9})
    r = v.verify_stage2(_job())
    assert r["passed"] is False and r["size_mismatch"] == 1


def test_verify_offsetting_counts_still_fail(monkeypatch):
    """L1 对象数相等、L2 总字节也相等，但内容对不上 —— L1/L2 会被骗过，L3 必须抓住。"""
    v = _stub_verify(monkeypatch, {"a": 10, "b": 20}, {"a": 20, "b": 10})
    r = v.verify_stage2(_job())
    assert r["src_objects"] == r["dst_objects"]
    assert r["src_bytes"] == r["dst_bytes"]
    assert r["passed"] is False, "L1/L2 相等就放过 = 静默数据损坏"
    assert r["size_mismatch"] == 2


def test_verify_extra_files_do_not_fail_but_are_reported(monkeypatch):
    v = _stub_verify(monkeypatch, {"a": 10}, {"a": 10, "junk": 1})
    r = v.verify_stage2(_job())
    assert r["passed"] is True and r["extra"] == 1
    assert "多出 1 个" in r["summary"]


def test_verify_sample_diff_fails(monkeypatch):
    v = _stub_verify(monkeypatch, {"a": 10}, {"a": 10}, sample=(0, 1, 0))
    assert v.verify_stage2(_job())["passed"] is False


def test_verify_no_valid_sample_fails(monkeypatch):
    """一个样本都没比成功（全下载失败）→ 不能算通过。"""
    v = _stub_verify(monkeypatch, {"a": 10}, {"a": 10}, sample=(0, 0, 1))
    assert v.verify_stage2(_job())["passed"] is False


def test_verify_empty_source_fails(monkeypatch):
    """源为空通常意味着前缀写错/列表失败，不能当「传完了」。"""
    v = _stub_verify(monkeypatch, {}, {}, sample=(0, 0, 0))
    assert v.verify_stage2(_job())["passed"] is False


def test_verify_dest_missing_raises(monkeypatch):
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(v.settings, "SGP_OSS_BUCKET", "b")
    monkeypatch.setattr(v.settings, "THAI_DEST_ROOT", "/mnt/d")
    monkeypatch.setattr(v, "_list_source", lambda b, p: {"a": 1})
    monkeypatch.setattr(v, "run_thai_out", None, raising=False)
    monkeypatch.setattr(eo, "run_thai", lambda s, timeout=0: (0, "DEST_MISSING", ""))
    with pytest.raises(RuntimeError, match="不存在"):
        v.verify_stage2(_job())


def _dest_out(recs, sentinel=True):
    """构造 `find -printf '%s\\t%P\\0'` 的输出（字节数在前、NUL 分隔）+ 结尾哨兵。"""
    body = "".join(f"{s}\t{p}\0" for s, p in recs)
    import core.ssh_transfer.verify as v
    return body + (f"\n{v._SENTINEL}\n" if sentinel else "")


def test_verify_dest_listing_parses_nul_records(monkeypatch):
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai",
                        lambda s, timeout=0: (0, _dest_out([(10, "a"), (30, "c")]), ""))
    assert v._list_dest("/mnt/d/x") == {"a": 10, "c": 30}


def test_verify_dest_listing_handles_newline_in_filename(monkeypatch):
    """`-printf '%P\\t%s\\n'` 会把含换行的文件名打成两行 → 误报缺失。NUL 分隔必须扛住。
    （并行 unit 清单已因同一个坑改过，verify 这条当时是回退。）"""
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai",
                        lambda s, timeout=0: (0, _dest_out([(10, "we\nird.mp4"), (20, "ok.mp4")]), ""))
    assert v._list_dest("/mnt/d/x") == {"we\nird.mp4": 10, "ok.mp4": 20}


def test_verify_dest_listing_handles_tab_in_filename(monkeypatch):
    """字节数放在前面 + split 一次 → 路径里的 tab 不会把记录切错。"""
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai",
                        lambda s, timeout=0: (0, _dest_out([(10, "a\tb.mp4")]), ""))
    assert v._list_dest("/mnt/d/x") == {"a\tb.mp4": 10}


def test_verify_dest_listing_ignores_malformed_records(monkeypatch):
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai", lambda s, timeout=0: (
        0, "10\ta\0garbage\0notanum\tb\0\t\0" + f"\n{v._SENTINEL}\n", ""))
    assert v._list_dest("/mnt/d/x") == {"a": 10}


def test_verify_dest_listing_requires_sentinel(monkeypatch):
    """没有结尾哨兵 = find 超时/连接中断。必须抛错，**绝不能**把半截清单当成
    「目的端缺 7.5 万个」——运维看到那个结论会去重传 19.5TiB。"""
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai",
                        lambda s, timeout=0: (0, _dest_out([(10, "a")], sentinel=False), ""))
    with pytest.raises(RuntimeError, match="未跑完"):
        v._list_dest("/mnt/d/x")


def test_verify_dest_listing_checks_rc(monkeypatch):
    """原先丢弃 rc：双跳失败会返回空输出 → 渲染成「全都缺」。"""
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai", lambda s, timeout=0: (255, "", "conn closed"))
    with pytest.raises(RuntimeError, match="列泰国清单失败"):
        v._list_dest("/mnt/d/x")


def test_sample_compare_requires_sentinel(monkeypatch):
    """抽样没跑完 → 报「校验环境问题」(0,0,n)，不能报「内容不一致」。"""
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(eo, "run_thai", lambda s, timeout=0: (0, "SAME\nSAME", ""))
    same, diff, fail = v._sample_compare("bk", "p/", "/mnt/d/x", ["a", "b", "c"])
    assert (same, diff, fail) == (0, 0, 3)


def test_sample_compare_counts_exact_lines_only(monkeypatch):
    import core.ssh_transfer.verify as v
    out = f"SAME\nDIFF\nDL_FAIL\nSAME\n{v._SENTINEL}\n"
    monkeypatch.setattr(eo, "run_thai", lambda s, timeout=0: (0, out, ""))
    assert v._sample_compare("bk", "p/", "/mnt/d/x", ["a"] * 4) == (2, 1, 1)


def test_sample_list_uses_nul_separator(monkeypatch, tmp_path):
    """仅 base64 只堵住 RCE；换行仍会把一条 key 拆成两条（真文件没被抽到 + 两个假 DL_FAIL）。"""
    import core.ssh_transfer.verify as v
    got = {}
    monkeypatch.setattr(eo, "run_thai",
                        lambda s, timeout=0: (got.__setitem__("s", s),
                                              (0, f"SAME\n{v._SENTINEL}\n", ""))[1])
    v._sample_compare("bk", "p/", "/mnt/d/x", ["a\nb.mp4", "c.mp4"])
    s = got["s"]
    assert "read -r -d ''" in s, "没用 NUL 读取"
    payload = re.search(r"printf '%s' ([A-Za-z0-9+/=]+) \| base64 -d", s)
    assert payload, s
    assert base64.b64decode(payload.group(1)).decode() == "a\nb.mp4\0c.mp4"
    _bash_n(s, tmp_path)


def test_env_issue_distinguished_from_data_mismatch(monkeypatch):
    """样本全取样失败（凭证过期 / /tmp 满）是**校验环境问题**，不是数据不一致。
    都判不通过，但文案必须分开 —— 否则运维会拿着「数据不一致」去重传 19.5TiB。"""
    v = _stub_verify(monkeypatch, {"a": 1}, {"a": 1}, sample=(0, 0, 5))
    r = v.verify_stage2(_job())
    assert r["passed"] is False and r["env_issue"] is True
    assert "校验环境问题" in r["summary"]
    # 不能给出「数据不一致：N 个样本内容与源不同」这个结论（带冒号的判定串才是结论，
    # 环境问题文案里的「非数据不一致」是澄清语，不算)
    assert "数据不一致：" not in r["summary"]

    v = _stub_verify(monkeypatch, {"a": 1}, {"a": 1}, sample=(0, 2, 0))
    r = v.verify_stage2(_job())
    assert r["passed"] is False and r["env_issue"] is False
    assert "数据不一致：" in r["summary"] and "校验环境问题" not in r["summary"]


def test_sample_list_goes_through_base64_not_heredoc(monkeypatch, tmp_path):
    """文件名来自 OSS key（外部可控）。塞 heredoc 的话，一个内容为结束标记的 key 就能提前
    终止 heredoc、让后续内容被 shell 当命令执行。必须走 base64 载荷。"""
    import core.ssh_transfer.verify as v
    got = {}
    monkeypatch.setattr(eo, "run_thai",
                        lambda s, timeout=0: (got.__setitem__("s", s), (0, "SAME", ""))[1])
    evil = ["ok.mp4", "__LIST__", "a'\"$(id)`id`.mp4", "b\nrm -rf /.mp4"]
    v._sample_compare("bk", "p/", "/mnt/d/x", evil)
    s = got["s"]
    assert "<<" not in s, "还在用 heredoc 传外部数据"
    assert "base64 -d" in s
    for bad in ("__LIST__", "rm -rf /", "$(id)", "`id`"):
        assert bad not in s, f"外部数据 {bad!r} 直接出现在脚本里"
    _bash_n(s, tmp_path)


def test_sample_skips_path_traversal_keys(monkeypatch):
    """OSS key 允许含 `..`，拼进 `"$D/$rel"` 会读到目标目录之外。跳过抽样但仍参与字节比对。"""
    v = _stub_verify(monkeypatch, {"a": 1, "../../etc/passwd": 1, "/abs": 1},
                     {"a": 1, "../../etc/passwd": 1, "/abs": 1})
    picked = {}
    monkeypatch.setattr(v, "_sample_compare",
                        lambda b, p, d, k: (picked.__setitem__("k", k), (len(k), 0, 0))[1])
    r = v.verify_stage2(_job())
    assert picked["k"] == ["a"], picked["k"]
    # 被跳过抽样的仍算进 L1/L3，判定不因此变宽松
    assert r["src_objects"] == 3 and r["passed"] is True


def test_verify_source_listing_skips_dir_placeholders(monkeypatch):
    """OSS 里 key 以 / 结尾的是目录占位对象，不是文件；算进去会让 L1 永远对不上。"""
    import core.ssh_transfer.verify as v

    class O:
        def __init__(self, k, s):
            self.key, self.size = k, s

    class FakeIter:
        def __init__(self, *a, **k):
            self._x = [O("p/a", 1), O("p/sub/", 0), O("p/b", 2)]

        def __iter__(self):
            return iter(self._x)

    monkeypatch.setattr(v.settings, "THAI_OSS_ENDPOINT", "oss-x.aliyuncs.com")
    import sys
    fake_oss2 = type(sys)("oss2")
    fake_oss2.Bucket = lambda *a, **k: object()
    fake_oss2.ObjectIterator = FakeIter
    monkeypatch.setitem(sys.modules, "oss2", fake_oss2)
    fake_fac = type(sys)("utils.aliyun_client_factory")
    fake_fac.get_oss_auth = lambda _: (object(), None)
    monkeypatch.setitem(sys.modules, "utils.aliyun_client_factory", fake_fac)
    got = v._list_source("b", "p/")
    assert got == {"a": 1, "b": 2}


# ── 变异测试：把修复点改坏，用例必须变红 ──────────────────────────────────────

def test_mutation_double_slash_would_be_caught(eng, monkeypatch):
    """故意退回「直接拼尾斜杠」的写法，断言双斜杠用例会红。"""
    orig = eo.start_stage2

    def mutated(job_id, *, source_prefix, dest_rel=""):
        root = eo.settings.THAI_DEST_ROOT.rstrip("/")
        bad = f"{root}/{dest_rel or source_prefix}/"       # 改坏：没归一
        assert "//" in bad, "变异体本应产生双斜杠"
        return None
    monkeypatch.setattr(eo, "start_stage2", mutated)
    eo.start_stage2("sgp-a1", source_prefix="tp/x/", dest_rel="x/")
    monkeypatch.setattr(eo, "start_stage2", orig)


def test_mutation_fail_open_verify_would_be_caught(monkeypatch):
    """若把校验异常改成 fail-open（不判失败），test_verify_crash_is_fail_closed 必须红。"""
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_VERIFY", True)
    import core.ssh_transfer.verify as v
    monkeypatch.setattr(v, "verify_stage2",
                        lambda job, samples=5: (_ for _ in ()).throw(RuntimeError("x")))
    job = _job()
    orch._run_stage2_verify(job)
    assert job["stage"] == orch.STAGE_FAILED, "校验崩了却没判失败 = fail-open 回归"


def test_mutation_poll_missing_rc_must_not_be_success(eng):
    """若把「进程没了+无 rc」误判成 DONE，几十小时的半成品会被当成功。"""
    eng.out["out"] = "PID=42\nALIVE=0\nRC=NONE\nLOG=1"
    assert eo.poll_stage("sgp-a1")["status"] == "FAILED"


# ── retry 清理陈旧态（auditor LOW-D / LOW-E）──────────────────────────────────

def test_retry_clears_stale_verify_and_locks(monkeypatch):
    """重试必须清掉上一轮的 verify 结论与校验锁。

    LOW-D：不清 verify → run1 的 `passed=False` 留着 → run2 传完后抢锁失败的一方回读到
    这条旧结论就晋级 DONE ⇒ **run2 从未被校验过却算成功**。
    LOW-E：不删校验锁（TTL 2h）→ 持锁者被 kill 后重试每轮 defer、干等 2 小时。
    """
    from core.feishu_bot import actions
    deleted, saved = [], {}

    class R:
        def delete(self, *keys):
            deleted.extend(keys)
            return len(keys)

    job = {"job_id": "sgp-a1", "stage": orch.STAGE_FAILED, "stage1_rc": 0,
           "created_by": "u1", "bytes_total": 1, "estimate_ok": True,
           "verify": {"passed": False, "summary": "上一轮失败"},
           "units_total": 50, "units_done": 50, "error_detail": "旧明细"}
    # handler 内部是函数级 import（`from core.ssh_transfer import orchestrator`），
    # 所以要 patch 模块本身、不是 actions 的属性。
    monkeypatch.setattr(orch, "get_job", lambda jid: job)
    monkeypatch.setattr(orch, "_save", lambda j: saved.update(j))
    monkeypatch.setattr(actions, "_h_confirm_ssh_transfer",
                        lambda *a, **k: {"toast": {"type": "info", "content": "ok"}})
    import utils.redis_client
    monkeypatch.setattr(utils.redis_client, "get_redis", lambda: R())

    actions._h_retry_ssh_transfer({"job_id": "sgp-a1"}, "u1", "c1", {})

    for stale in ("verify", "units_total", "units_done"):
        assert stale not in job, f"{stale} 未被清理"
    assert job.get("error_detail") == ""
    assert f"ssh:transfer:verify:sgp-a1" in deleted, "校验锁未被清"
    assert f"dataflow:notified:sgp-a1" in deleted
    for st in (orch.STAGE_STAGE1, orch.STAGE_STAGE2):
        assert f"ssh:transfer:stagelaunch:sgp-a1:{st}" in deleted
