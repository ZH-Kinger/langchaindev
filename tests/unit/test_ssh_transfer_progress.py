"""#51 续 —— 进度/速度解析。

engine_ssh.stage_progress（rsync --info=progress2 / ossutil 速率）+ _speed_bps 单位换算 +
start_stage2 用 dest_rel；orchestrator._sample_progress（字节差算速率）+ progress_line + _fmt_eta。

paramiko 未装：全程 mock（stage_progress 桩 tail_log，start_stage2 桩 run）。Redis 不涉及。
"""
import pytest

from core.ssh_transfer import engine_ssh, orchestrator as orch
from core.ssh_transfer.engine_ssh import STAGE1, STAGE2


# ── engine.stage_progress：段2 rsync progress2 ─────────────────────────────────

def test_stage_progress_rsync_bytes_pct_speed(monkeypatch):
    monkeypatch.setattr(engine_ssh, "tail_log",
                        lambda jid, st, lines=3: "\n  1,234,567  73%  45.67MB/s  0:12:34\n")
    prog = engine_ssh.stage_progress("sgp-x", STAGE2)
    assert prog["bytes_done"] == 1234567
    assert prog["pct"] == 73
    assert prog["speed_bps"] == int(45.67 * 1024 ** 2)


def test_stage_progress_rsync_takes_last_match(monkeypatch):
    """多行进度取最后一次匹配（最新）。"""
    log = ("  1,000  10%  1.00MB/s  0:00:10\n"
           "  9,000  90%  2.00MB/s  0:00:01\n")
    monkeypatch.setattr(engine_ssh, "tail_log", lambda jid, st, lines=3: log)
    prog = engine_ssh.stage_progress("sgp-x", STAGE2)
    assert prog["bytes_done"] == 9000
    assert prog["pct"] == 90
    assert prog["speed_bps"] == int(2.0 * 1024 ** 2)


# ── engine.stage_progress：段1 ossutil 只稳取速率 ──────────────────────────────

def test_stage_progress_ossutil_rate_only(monkeypatch):
    monkeypatch.setattr(engine_ssh, "tail_log",
                        lambda jid, st, lines=3: "uploading ... 123.4 MiB/s\n")
    prog = engine_ssh.stage_progress("sgp-x", STAGE1)
    assert prog["speed_bps"] == int(123.4 * 1024 ** 2)
    assert prog["bytes_done"] is None       # 段1 不取字节
    assert prog["pct"] is None


def test_stage_progress_no_match_all_none(monkeypatch):
    monkeypatch.setattr(engine_ssh, "tail_log", lambda jid, st, lines=3: "nothing useful\n")
    assert engine_ssh.stage_progress("sgp-x", STAGE2) == {
        "bytes_done": None, "pct": None, "speed_bps": None}
    assert engine_ssh.stage_progress("sgp-x", STAGE1) == {
        "bytes_done": None, "pct": None, "speed_bps": None}


# ── _speed_bps 单位换算（K/M/G/T、带/不带 i） ───────────────────────────────────

@pytest.mark.parametrize("unit,mult", [
    ("", 1), ("K", 1024), ("M", 1024 ** 2), ("G", 1024 ** 3), ("T", 1024 ** 4),
    ("k", 1024), ("m", 1024 ** 2),   # 大小写不敏感
])
def test_speed_bps_units(unit, mult):
    assert engine_ssh._speed_bps("1", unit) == mult
    assert engine_ssh._speed_bps("2.5", unit) == int(2.5 * mult)


@pytest.mark.parametrize("txt", ["45.67MB/s", "45.67MiB/s", "45.67 MB/s", "45.67 MiB/s"])
def test_rate_i_and_noni_equal(monkeypatch, txt):
    """`MiB/s` 与 `MB/s` 换算一致（正则吃掉 i，单位组只留 M）。"""
    monkeypatch.setattr(engine_ssh, "tail_log", lambda jid, st, lines=3: f"x {txt}\n")
    prog = engine_ssh.stage_progress("sgp-x", STAGE1)
    assert prog["speed_bps"] == int(45.67 * 1024 ** 2)


# ── start_stage2 用 dest_rel（命令含 mkdir -p .../test/ + rsync target .../test/） ─

@pytest.fixture
def capture_run(monkeypatch):
    box = {"cmds": []}

    def fake_run(cmd, *, timeout=30):
        box["cmds"].append(cmd)
        return (0, "", "")

    monkeypatch.setattr(engine_ssh, "run", fake_run)
    box["last"] = lambda: box["cmds"][-1]
    return box


def test_start_stage2_uses_dest_rel(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_DEST_ROOT", "/mnt/data/x")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "")
    engine_ssh.start_stage2("sgp-x", source_prefix="ossutil_output/", dest_rel="test/")
    cmd = capture_run["last"]()
    assert "mkdir -p" in cmd
    # 目标（mkdir + rsync target）落在 <root>/<dest_rel>，不是源前缀
    assert "/mnt/data/x/test/" in cmd
    assert "/mnt/data/x/ossutil_output/" not in cmd
    # 源仍是 SGP 挂载盘上的源前缀
    assert "ossutil_output/" in cmd


def test_start_stage2_empty_dest_rel_mirrors_source(capture_run, monkeypatch):
    monkeypatch.setattr(engine_ssh.settings, "THAI_DEST_ROOT", "/mnt/data/x")
    engine_ssh.start_stage2("sgp-x", source_prefix="ossutil_output/", dest_rel="")
    cmd = capture_run["last"]()
    assert "/mnt/data/x/ossutil_output/" in cmd       # 空 dest_rel 镜像源前缀


# ── orchestrator._sample_progress ──────────────────────────────────────────────

def test_sample_progress_uses_direct_speed(monkeypatch):
    """日志直接给速率 → 直接用。"""
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": None, "pct": None, "speed_bps": 5000})
    job = {"job_id": "sgp-x"}
    orch._sample_progress(job, STAGE2)
    assert job["speed_bps"] == 5000


def test_sample_progress_computes_speed_from_byte_diff(monkeypatch):
    """日志给字节无速率 → 用相邻两次采样差算速率。"""
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 2000, "pct": None, "speed_bps": None})
    monkeypatch.setattr(orch.time, "time", lambda: 110.0)
    job = {"job_id": "sgp-x", "_bd_sample": 1000, "_bd_ts": 100.0}
    orch._sample_progress(job, STAGE1)
    assert job["bytes_done"] == 2000
    assert job["_bd_sample"] == 2000
    assert job["_bd_ts"] == 110.0
    assert job["speed_bps"] == int((2000 - 1000) / (110.0 - 100.0))   # 100


def test_sample_progress_first_sample_no_speed(monkeypatch):
    """首次采样（无 prev）+ 无速率 → 记字节但不算速率。"""
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 1000, "pct": None, "speed_bps": None})
    monkeypatch.setattr(orch.time, "time", lambda: 100.0)
    job = {"job_id": "sgp-x"}
    orch._sample_progress(job, STAGE1)
    assert job["bytes_done"] == 1000
    assert "speed_bps" not in job


def test_sample_progress_sets_pct(monkeypatch):
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 500, "pct": 42, "speed_bps": 10})
    job = {"job_id": "sgp-x"}
    orch._sample_progress(job, STAGE2)
    assert job["pct"] == 42
    assert job["bytes_done"] == 500
    assert job["speed_bps"] == 10


def test_sample_progress_engine_error_noop(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ssh down")
    monkeypatch.setattr(orch.engine_ssh, "stage_progress", boom)
    job = {"job_id": "sgp-x"}
    orch._sample_progress(job, STAGE1)
    assert "bytes_done" not in job and "speed_bps" not in job


# ── orchestrator.progress_line ─────────────────────────────────────────────────

def test_progress_line_full():
    """有 total + 速率 → 已传X/Y(%)·速率·剩余ETA。"""
    job = {"bytes_done": 500 * 1024 ** 2, "bytes_total": 1024 ** 3, "speed_bps": 100 * 1024 ** 2}
    line = orch.progress_line(job)
    assert "已传" in line and "/" in line
    assert "%" in line
    assert "速率" in line
    assert "剩余约" in line


def test_progress_line_total_no_speed_no_eta():
    """有 total 无速率 → 只显示已传、无速率无剩余（spd=0 不算 ETA、不除零）。"""
    job = {"bytes_done": 100, "bytes_total": 200}
    line = orch.progress_line(job)
    assert "已传" in line
    assert "速率" not in line
    assert "剩余" not in line


def test_progress_line_speed_zero_no_divide():
    """speed_bps=0 → 不出速率行、不算 ETA（不 ZeroDivisionError）。"""
    job = {"bytes_done": 100, "bytes_total": 200, "speed_bps": 0}
    line = orch.progress_line(job)   # 不抛
    assert "速率" not in line and "剩余" not in line


def test_progress_line_only_bytes_done_no_total():
    job = {"bytes_done": 100}
    line = orch.progress_line(job)
    assert "已传" in line
    assert "/" not in line           # 无 total → 不出 X/Y


def test_progress_line_done_no_eta():
    """已传=总量（完成）→ 有速率但无剩余（bd<bt 为假）。"""
    job = {"bytes_done": 200, "bytes_total": 200, "speed_bps": 100}
    line = orch.progress_line(job)
    assert "剩余" not in line


def test_progress_line_empty():
    assert orch.progress_line({}) == "进度采集中…"


# ── orchestrator._fmt_eta 分档 ─────────────────────────────────────────────────

def test_fmt_eta_seconds():
    assert orch._fmt_eta(30) == "30s"


def test_fmt_eta_minutes():
    assert orch._fmt_eta(90) == "1m30s"


def test_fmt_eta_hours():
    assert orch._fmt_eta(3661) == "1h1m"


def test_fmt_eta_negative_clamped():
    assert orch._fmt_eta(-5) == "0s"
