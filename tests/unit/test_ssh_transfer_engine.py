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
