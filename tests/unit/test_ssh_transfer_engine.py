"""#51 SSH 迁移链 —— engine_ssh 命令生成 / marker 解析 / rc 语义 / 私钥&hostkey 守卫。

paramiko 未安装：engine_ssh 在函数内 lazy import paramiko，导入模块本身不需要它。
- 命令类测试全程 mock `engine_ssh.run`（捕获命令串），从不触及 `_client`/paramiko。
- 私钥/hostkey 守卫在触碰 paramiko 属性前就 raise，注入一个空的 stub paramiko 即可跑通。
"""
import sys
import types

import pytest

from core.ssh_transfer import engine_ssh
from core.ssh_transfer.engine_ssh import STAGE1, STAGE2, SshTransferError


@pytest.fixture
def capture_run(monkeypatch):
    """mock engine_ssh.run，捕获最后一次命令串；返回 (0,'','')。"""
    box = {"cmds": []}

    def fake_run(cmd, *, timeout=30):
        box["cmds"].append(cmd)
        return (0, "", "")

    monkeypatch.setattr(engine_ssh, "run", fake_run)
    box["last"] = lambda: box["cmds"][-1]
    return box


@pytest.fixture
def fake_paramiko(monkeypatch):
    """注入空 stub paramiko —— 供仅需「import paramiko 成功」的守卫路径。"""
    mod = types.ModuleType("paramiko")
    monkeypatch.setitem(sys.modules, "paramiko", mod)
    return mod


# ── _rc_ok ────────────────────────────────────────────────────────────────────

def test_rc_ok_stage1_only_zero():
    assert engine_ssh._rc_ok(STAGE1, 0) is True
    assert engine_ssh._rc_ok(STAGE1, 24) is False
    assert engine_ssh._rc_ok(STAGE1, 1) is False


def test_rc_ok_stage2_zero_and_24():
    assert engine_ssh._rc_ok(STAGE2, 0) is True
    assert engine_ssh._rc_ok(STAGE2, 24) is True
    assert engine_ssh._rc_ok(STAGE2, 23) is False
    assert engine_ssh._rc_ok(STAGE2, 255) is False


# ── start_stage1 命令生成 ─────────────────────────────────────────────────────

def test_start_stage1_command(capture_run):
    engine_ssh.start_stage1("sgp-abc123", source_bucket="wuji-data-tran",
                            source_prefix="team/data/")
    cmd = capture_run["last"]()
    assert "ossutil cp" in cmd
    assert "oss://wuji-data-tran/team/data/" in cmd
    assert "--jobs" in cmd
    assert "-u" in cmd
    assert "--checkpoint-dir" in cmd
    assert "nohup" in cmd
    assert "echo $?" in cmd          # rc marker
    assert "echo $!" in cmd          # pid marker
    assert "stage1.rc" in cmd
    assert "stage1.pid" in cmd
    assert "stage1.log" in cmd


def test_start_stage1_dst_is_sgp_mount(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "SGP_OSS_MOUNT", "/mnt/sgp_oss")
    engine_ssh.start_stage1("sgp-x", source_bucket="b", source_prefix="p/")
    cmd = capture_run["last"]()
    assert "/mnt/sgp_oss/p/" in cmd


# ── _launch 分组结构（真机 bug 回归：& 优先级低于 &&，pid 写入必须在 mkdir 之后） ──
#
# 旧写法 `mkdir && rm && nohup ... & echo $!>pid` 会把整段 mkdir&&rm&&nohup 一起丢后台，
# echo $!>pid 不等 mkdir 完成就先在前台跑 → pid 目录未建、写失败 rc=1 → 起任务报错。
# 修复用 `{ ...; }` 组起「后台起任务 + 写 pid」，确保在 mkdir&&rm 之后才执行：
#   mkdir -p JOB && rm -f RC && { nohup bash -c 'INNER' > LOG 2>&1 & echo $! > PID; }

def _launch_via(engine, capture_run, stage):
    """经公开入口触发 _launch，返回捕获到的 remote 命令串。"""
    if stage == STAGE1:
        engine.start_stage1("sgp-abc123", source_bucket="wuji-data-tran",
                            source_prefix="team/data/")
    else:
        engine.start_stage2("sgp-abc123", source_prefix="team/data/")
    return capture_run["last"]()


@pytest.mark.parametrize("stage", [STAGE1, STAGE2])
def test_launch_has_grouping_structure(capture_run, monkeypatch, stage):
    """remote 串含分组结构：`{ nohup ... & echo $! > <pid>; }`。"""
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "")
    remote = _launch_via(engine_ssh, capture_run, stage)
    # 组开头/组内后台起任务/写 pid/组结尾
    assert "{ nohup" in remote
    assert "& echo $!" in remote
    assert f"> {engine_ssh._marker('sgp-abc123', stage, 'pid')}" in remote
    assert remote.rstrip().endswith("; }")


@pytest.mark.parametrize("stage", [STAGE1, STAGE2])
def test_launch_ordering_mkdir_before_pid(capture_run, monkeypatch, stage):
    """顺序断言：mkdir 出现在 echo $! 之前（pid 写入不再抢先于建目录）。"""
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "")
    remote = _launch_via(engine_ssh, capture_run, stage)
    assert remote.index("mkdir") < remote.index("echo $!")


@pytest.mark.parametrize("stage", [STAGE1, STAGE2])
def test_launch_pid_write_inside_group_after_and(capture_run, monkeypatch, stage):
    """结构正确性：`echo $! > <pid>` 在 `{ }` 组内、组在 `&&` 之后。

    断言 `&& {` 子串存在且 `echo $!` 出现在 `&& {` 之后 —— 证明 pid 写入不再
    先于 mkdir（旧写法 pid 写在最外层前台、不受 && 约束）。
    """
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "")
    remote = _launch_via(engine_ssh, capture_run, stage)
    assert "&& {" in remote
    assert remote.index("&& {") < remote.index("echo $!")


@pytest.mark.parametrize("stage", [STAGE1, STAGE2])
def test_launch_mkdir_rm_precede_group(capture_run, monkeypatch, stage):
    """mkdir -p 在最前、&& rm -f 其次，且都在分组 `{ nohup` 之前。"""
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "")
    remote = _launch_via(engine_ssh, capture_run, stage)
    assert remote.index("mkdir -p") < remote.index("&& rm -f")
    assert remote.index("&& rm -f") < remote.index("{ nohup")
    # rc marker 在 rm -f 后被清理、且写在 inner 里（echo $? > rc）
    rc_marker = engine_ssh._marker("sgp-abc123", stage, "rc")
    assert f"rm -f {rc_marker}" in remote
    assert f"echo $? > {rc_marker}" in remote


def test_launch_inner_is_shlex_quoted_wrapper(capture_run, monkeypatch):
    """注入防护未回归：work_cmd 整体经 `bash -c '<inner>'` 单引号包裹。

    分组只在 inner 外层加 `{ }`，不改 inner 的 shlex.quote —— 即便含空格/元字符
    的路径混进 work_cmd 也整体被引用、不会破 bash -c 参数边界。
    """
    import shlex
    monkeypatch.setattr(engine_ssh.settings, "SGP_OSS_MOUNT", "/mnt/sgp_oss")
    monkeypatch.setattr(engine_ssh.settings, "SGP_OSSUTIL_JOBS", 30)
    # 含空格 + shell 元字符（正常经 paths 白名单挡住，此处直调 engine 验兜底引用）
    prefix = "team/da ta; rm -rf x/"
    engine_ssh.start_stage1("sgp-abc123", source_bucket="b", source_prefix=prefix)
    remote = capture_run["last"]()
    src = f"oss://b/{prefix}"
    # src 被 shlex.quote 后（含空格→整体加单引号），再随 inner 一起被 bash -c 引用；
    # 断言 quote 后的 src 片段出现（证明未裸拼），且 `bash -c '` 包裹存在。
    assert shlex.quote(src) in remote
    assert "bash -c '" in remote
    # 危险片段 `; rm -rf x` 被夹在引号内、其后紧跟收尾引号或路径，不在顶层裸露
    assert "rm -rf x" in remote  # 存在但被引用（下一断言证明未破坏分组结构）
    assert remote.rstrip().endswith("; }")


def test_launch_raises_on_nonzero_rc(monkeypatch):
    """起任务 SSH 返回非 0 → SshTransferError（rc!=0 说明前台 pid 写入/建目录失败）。

    这正是旧 bug 的现场：分组前 echo $!>pid 抢跑失败 → rc=1 → 起任务报错。
    """
    monkeypatch.setattr(engine_ssh, "run",
                        lambda cmd, *, timeout=30: (1, "", "mkdir: cannot create"))
    with pytest.raises(SshTransferError) as ei:
        engine_ssh.start_stage1("sgp-x", source_bucket="b", source_prefix="p/")
    assert "stage1" in str(ei.value)


# ── start_stage2 命令生成 ─────────────────────────────────────────────────────

def test_start_stage2_command_basic(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "")
    engine_ssh.start_stage2("sgp-abc", source_prefix="team/data/")
    cmd = capture_run["last"]()
    assert "rsync" in cmd
    assert "-a" in cmd
    assert "--info=progress2" in cmd
    assert "mkdir -p" in cmd
    user = engine_ssh.settings.THAI_USER
    host = engine_ssh.settings.THAI_HOST
    assert f"{user}@{host}:" in cmd
    assert "--rsync-path=sudo rsync" not in cmd
    assert "--bwlimit=" not in cmd


def test_start_stage2_sudo_true_adds_rsync_path(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "true")
    engine_ssh.start_stage2("sgp-abc", source_prefix="p/")
    assert "--rsync-path=sudo rsync" in capture_run["last"]()


def test_start_stage2_sudo_false_no_rsync_path(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    engine_ssh.start_stage2("sgp-abc", source_prefix="p/")
    assert "--rsync-path=sudo rsync" not in capture_run["last"]()


def test_start_stage2_bwlimit_when_set(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "50m")
    engine_ssh.start_stage2("sgp-abc", source_prefix="p/")
    assert "--bwlimit=50m" in capture_run["last"]()


def test_start_stage2_missing_dest_root_raises(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_DEST_ROOT", "")
    with pytest.raises(SshTransferError):
        engine_ssh.start_stage2("sgp-abc", source_prefix="p/")


# ── poll_stage 解析 ───────────────────────────────────────────────────────────

def _run_returning(monkeypatch, out):
    monkeypatch.setattr(engine_ssh, "run", lambda cmd, *, timeout=30: (0, out, ""))


def test_poll_stage_rc0_done(monkeypatch):
    _run_returning(monkeypatch, "RC=0\n")
    st = engine_ssh.poll_stage("sgp-x", STAGE1)
    assert st["status"] == "DONE"
    assert st["rc"] == 0
    assert st["alive"] is False


def test_poll_stage_stage2_rc23_failed(monkeypatch):
    _run_returning(monkeypatch, "RC=23\n")
    st = engine_ssh.poll_stage("sgp-x", STAGE2)
    assert st["status"] == "FAILED"
    assert st["rc"] == 23
    assert st["error"]


def test_poll_stage_stage2_rc24_done(monkeypatch):
    _run_returning(monkeypatch, "RC=24\n")
    st = engine_ssh.poll_stage("sgp-x", STAGE2)
    assert st["status"] == "DONE"


def test_poll_stage_alive_running(monkeypatch):
    _run_returning(monkeypatch, "ALIVE\n")
    st = engine_ssh.poll_stage("sgp-x", STAGE1)
    assert st["status"] == "RUNNING"
    assert st["alive"] is True


def test_poll_stage_dead_failed(monkeypatch):
    _run_returning(monkeypatch, "DEAD\n")
    st = engine_ssh.poll_stage("sgp-x", STAGE1)
    assert st["status"] == "FAILED"
    assert st["alive"] is False


def test_poll_stage_garbage_rc_defaults_fail(monkeypatch):
    """RC= 后跟非数字 → rc 兜底 1 → stage1 FAILED。"""
    _run_returning(monkeypatch, "RC=oops\n")
    st = engine_ssh.poll_stage("sgp-x", STAGE1)
    assert st["status"] == "FAILED"
    assert st["rc"] == 1


# ── estimate_source 解析 ──────────────────────────────────────────────────────

def test_estimate_source_parses_du(monkeypatch):
    """解析出 size → 3 元组 ok=True。"""
    out = "total object count: 1,234\ntotal sum size: 5,678,900\n"
    _run_returning(monkeypatch, out)
    assert engine_ssh.estimate_source("bkt", "a/") == (5678900, 1234, True)


def test_estimate_source_zero_size_but_parsed_ok_true(monkeypatch):
    """空前缀 size=0 但确实解析到大小行 → ok=True（不当作未知）。"""
    out = "total object count: 0\ntotal sum size: 0\n"
    _run_returning(monkeypatch, out)
    assert engine_ssh.estimate_source("bkt", "a/") == (0, 0, True)


def test_estimate_source_unparseable_not_ok(monkeypatch):
    """没匹配 size 行 → (0,0,False)（未知，上层 fail-safe 当需审批）。"""
    _run_returning(monkeypatch, "nothing useful here\n")
    assert engine_ssh.estimate_source("bkt", "a/") == (0, 0, False)


def test_estimate_source_empty_not_ok(monkeypatch):
    _run_returning(monkeypatch, "")
    assert engine_ssh.estimate_source("bkt", "a/") == (0, 0, False)


# ── 私钥 / host key 守卫（stub paramiko，raise 在触碰属性前） ──────────────────

def test_load_private_key_missing_enc(fake_paramiko, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "SGP_SSH_KEY_ENC", "")
    with pytest.raises(SshTransferError):
        engine_ssh._load_private_key()


def test_add_host_key_missing(fake_paramiko, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "SGP_SSH_HOST_KEY", "")
    with pytest.raises(SshTransferError):
        engine_ssh._add_host_key(object())


def test_add_host_key_malformed(fake_paramiko, monkeypatch):
    """只有一段（无 base64 部分）→ 报错。"""
    monkeypatch.setattr(engine_ssh.settings, "SGP_SSH_HOST_KEY", "ssh-ed25519")
    with pytest.raises(SshTransferError):
        engine_ssh._add_host_key(object())


def test_load_private_key_non_pem_decrypt_result(fake_paramiko, monkeypatch):
    """decrypt 返回非 PEM（明文透传/坏密文）→ SshTransferError，不误报'格式非法'。"""
    monkeypatch.setattr(engine_ssh.settings, "SGP_SSH_KEY_ENC", "some-ciphertext")
    import utils.crypto as crypto
    monkeypatch.setattr(crypto, "decrypt", lambda s: "not a pem private key")
    with pytest.raises(SshTransferError) as ei:
        engine_ssh._load_private_key()
    assert "PEM" in str(ei.value)


def test_load_private_key_empty_decrypt_result(fake_paramiko, monkeypatch):
    """decrypt 返回空串 → SshTransferError。"""
    monkeypatch.setattr(engine_ssh.settings, "SGP_SSH_KEY_ENC", "some-ciphertext")
    import utils.crypto as crypto
    monkeypatch.setattr(crypto, "decrypt", lambda s: "")
    with pytest.raises(SshTransferError):
        engine_ssh._load_private_key()
