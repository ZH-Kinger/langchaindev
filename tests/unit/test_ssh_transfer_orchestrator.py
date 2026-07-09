"""#51 SSH 迁移链 —— orchestrator 状态机 / job 幂等 / 审批 / 轮询推进。

engine_ssh 全程 mock（不连 SSH、不 import paramiko）。Redis 用 conftest 的 fakeredis。
"""
import pytest

from core.ssh_transfer import paths, orchestrator as orch
from core.ssh_transfer.orchestrator import (
    STAGE_NEW, STAGE_STAGE1, STAGE_STAGE2, STAGE_DONE, STAGE_FAILED,
)


@pytest.fixture
def plan():
    return paths.build_plan("oss://wuji-data-tran/team/data/")


@pytest.fixture
def plan2():
    return paths.build_plan("oss://wuji-data-tran/other/set/")


# ── _job_id ───────────────────────────────────────────────────────────────────

def test_job_id_deterministic_same_plan_same_day(plan):
    assert orch._job_id(plan) == orch._job_id(plan)


def test_job_id_has_sgp_prefix(plan):
    assert orch._job_id(plan).startswith("sgp-")


def test_job_id_differs_by_prefix(plan, plan2):
    assert orch._job_id(plan) != orch._job_id(plan2)


# ── create_job_record ─────────────────────────────────────────────────────────

def test_create_first_build_new_record(plan):
    job = orch.create_job_record(plan, open_id="u1", bytes_total=123, objects_total=4)
    assert job["stage"] == STAGE_NEW
    assert job["created_by"] == "u1"
    assert job["source_bucket"] == "wuji-data-tran"
    assert job["source_prefix"] == "team/data/"
    assert job["stage1_rc"] is None
    assert job["launched"] is False
    assert job["bytes_total"] == 123 and job["objects_total"] == 4
    assert job["estimate_ok"] is True          # 默认 size_known=True
    # 真落库
    assert orch.get_job(job["job_id"]) == job


def test_create_records_estimate_unknown(plan):
    """size_known=False → estimate_ok 落盘 False（上层据此强制审批）。"""
    job = orch.create_job_record(plan, open_id="u1", bytes_total=0, size_known=False)
    assert job["estimate_ok"] is False
    assert orch.get_job(job["job_id"])["estimate_ok"] is False


def test_create_idempotent_returns_existing(plan):
    first = orch.create_job_record(plan, open_id="u1")
    # 推进到非终态
    first["stage"] = STAGE_STAGE1
    orch._save(first)
    again = orch.create_job_record(plan, open_id="u1")
    assert again["stage"] == STAGE_STAGE1          # 返回旧记录，未重置为 NEW
    assert again["job_id"] == first["job_id"]


def test_create_backfills_empty_created_by(plan):
    """幂等分支：空 created_by 遇真 open_id → 回填并落库。"""
    orch.create_job_record(plan, open_id="")       # 先建空
    job_id = orch._job_id(plan)
    assert orch.get_job(job_id)["created_by"] == ""
    again = orch.create_job_record(plan, open_id="u9")
    assert again["created_by"] == "u9"
    assert orch.get_job(job_id)["created_by"] == "u9"   # 落库了


def test_create_does_not_overwrite_nonempty_created_by(plan):
    orch.create_job_record(plan, open_id="owner")
    again = orch.create_job_record(plan, open_id="intruder")
    assert again["created_by"] == "owner"


def test_create_empty_open_id_does_not_write_empty(plan):
    orch.create_job_record(plan, open_id="owner")
    again = orch.create_job_record(plan, open_id="")
    assert again["created_by"] == "owner"           # 不被空覆盖


def test_create_failed_goes_to_new_build(plan):
    """已 FAILED 的记录 → 走新建（stage 复位 NEW、created_by=本次 open_id）。"""
    first = orch.create_job_record(plan, open_id="u1")
    first["stage"] = STAGE_FAILED
    first["error"] = "boom"
    orch._save(first)
    again = orch.create_job_record(plan, open_id="u2")
    assert again["stage"] == STAGE_NEW
    assert again["created_by"] == "u2"
    assert again["error"] == ""


# ── estimate_source（orchestrator 包一层，3 元组） ────────────────────────────

def test_orch_estimate_source_passthrough(monkeypatch, plan):
    monkeypatch.setattr(orch.engine_ssh, "estimate_source",
                        lambda b, p: (999, 7, True))
    assert orch.estimate_source(plan) == (999, 7, True)


def test_orch_estimate_source_exception_not_ok(monkeypatch, plan):
    """引擎抛错（SSH 不通）→ (0,0,False)，交给 needs_approval fail-safe。"""
    def boom(b, p):
        raise RuntimeError("ssh down")
    monkeypatch.setattr(orch.engine_ssh, "estimate_source", boom)
    assert orch.estimate_source(plan) == (0, 0, False)


# ── needs_approval ────────────────────────────────────────────────────────────

def test_needs_approval_default_1tb_boundary(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_TRANSFER_APPROVAL_TB", 1)
    tb = 1024 ** 4
    assert orch.needs_approval(tb, size_known=True) is False    # 恰阈值不触发
    assert orch.needs_approval(tb + 1, size_known=True) is True  # 超一字节触发
    assert orch.needs_approval(0, size_known=True) is False


def test_needs_approval_size_unknown_forces_true(monkeypatch):
    """size_known=False → 恒需审批（fail-safe，不放行未知大小的迁移），与字节数无关。"""
    monkeypatch.setattr(orch.settings, "SSH_TRANSFER_APPROVAL_TB", 1)
    assert orch.needs_approval(0, size_known=False) is True
    assert orch.needs_approval(1, size_known=False) is True
    assert orch.needs_approval(1024 ** 4, size_known=False) is True


def test_needs_approval_default_size_known_true(monkeypatch):
    """size_known 默认 True（不传时按阈值判定）。"""
    monkeypatch.setattr(orch.settings, "SSH_TRANSFER_APPROVAL_TB", 1)
    assert orch.needs_approval(1024 ** 4) is False
    assert orch.needs_approval(1024 ** 4 + 1) is True


def test_needs_approval_custom_threshold(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_TRANSFER_APPROVAL_TB", 2)
    tb2 = 2 * 1024 ** 4
    assert orch.needs_approval(tb2, size_known=True) is False
    assert orch.needs_approval(tb2 + 1, size_known=True) is True


def test_needs_approval_bad_setting_falls_back_to_1tb(monkeypatch):
    monkeypatch.setattr(orch.settings, "SSH_TRANSFER_APPROVAL_TB", "not-a-number")
    assert orch.needs_approval(1024 ** 4 + 1, size_known=True) is True
    assert orch.needs_approval(1024 ** 4, size_known=True) is False


# ── poll_once（mock engine_ssh.poll_stage / start_stage2） ────────────────────

@pytest.fixture
def new_job(plan):
    return orch.create_job_record(plan, open_id="u1")


def test_poll_once_stage1_done_starts_stage2(monkeypatch, new_job):
    from unittest.mock import MagicMock
    new_job["stage"] = STAGE_STAGE1
    orch._save(new_job)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "DONE", "rc": 0})
    start2 = MagicMock()
    monkeypatch.setattr(orch.engine_ssh, "start_stage2", start2)
    monkeypatch.setattr(orch.engine_ssh, "start_stage1",
                        lambda *a, **k: pytest.fail("stage1 不该再起"))
    out = orch.poll_once(new_job)
    assert out["stage1_rc"] == 0
    assert out["stage"] == STAGE_STAGE2
    assert start2.call_count == 1


def test_poll_once_stage1_failed_no_stage2(monkeypatch, new_job):
    new_job["stage"] = STAGE_STAGE1
    orch._save(new_job)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "FAILED", "rc": 1, "error": "段1 退出码 1"})
    monkeypatch.setattr(orch.engine_ssh, "start_stage2",
                        lambda *a, **k: pytest.fail("段1失败不该起段2"))
    out = orch.poll_once(new_job)
    assert out["stage"] == STAGE_FAILED
    assert out["error"] == "段1 退出码 1"
    assert out["finished_ts"] > 0


def test_poll_once_stage2_done_finishes(monkeypatch, new_job):
    new_job["stage"] = STAGE_STAGE2
    new_job["stage1_rc"] = 0
    orch._save(new_job)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "DONE", "rc": 0})
    out = orch.poll_once(new_job)
    assert out["stage"] == STAGE_DONE
    assert out["finished_ts"] > 0


def test_poll_once_running_stays(monkeypatch, new_job):
    new_job["stage"] = STAGE_STAGE1
    orch._save(new_job)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "RUNNING", "rc": None, "alive": True})
    out = orch.poll_once(new_job)
    assert out["stage"] == STAGE_STAGE1


def test_poll_once_non_active_stage_noop(monkeypatch, new_job):
    new_job["stage"] = STAGE_DONE
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda *a, **k: pytest.fail("终态不该 poll"))
    out = orch.poll_once(new_job)
    assert out["stage"] == STAGE_DONE


def test_poll_once_engine_error_keeps_inflight(monkeypatch, new_job):
    new_job["stage"] = STAGE_STAGE1
    orch._save(new_job)

    def boom(*a, **k):
        raise RuntimeError("ssh down")
    monkeypatch.setattr(orch.engine_ssh, "poll_stage", boom)
    out = orch.poll_once(new_job)
    assert out["stage"] == STAGE_STAGE1   # 轮询失败保持在途，不误判失败


# ── refresh ───────────────────────────────────────────────────────────────────

def test_refresh_inflight_repolls(monkeypatch, new_job):
    new_job["stage"] = STAGE_STAGE2
    new_job["stage1_rc"] = 0
    orch._save(new_job)
    calls = []
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: calls.append(st) or {"status": "DONE", "rc": 0})
    out = orch.refresh(new_job["job_id"])
    assert calls, "在途应重查"
    assert out["stage"] == STAGE_DONE


def test_refresh_terminal_does_not_repoll(monkeypatch, new_job):
    new_job["stage"] = STAGE_DONE
    orch._save(new_job)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda *a, **k: pytest.fail("终态不该重查"))
    out = orch.refresh(new_job["job_id"])
    assert out["stage"] == STAGE_DONE


def test_refresh_missing_job_returns_none():
    assert orch.refresh("sgp-doesnotexist") is None


# ── run_to_completion（mock engine start_stage1/2 + poll_stage） ──────────────

def _mock_engine(monkeypatch, poll_results):
    calls = {"stage1": 0, "stage2": 0}
    monkeypatch.setattr(orch.engine_ssh, "start_stage1",
                        lambda jid, **kw: calls.__setitem__("stage1", calls["stage1"] + 1))
    monkeypatch.setattr(orch.engine_ssh, "start_stage2",
                        lambda jid, **kw: calls.__setitem__("stage2", calls["stage2"] + 1))
    it = iter(poll_results)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage", lambda jid, st: next(it))
    return calls


def test_run_to_completion_two_stages(monkeypatch, new_job):
    calls = _mock_engine(monkeypatch, [
        {"status": "DONE", "rc": 0},   # 段1
        {"status": "DONE", "rc": 0},   # 段2
    ])
    seen = []
    out = orch.run_to_completion(new_job, on_update=lambda j: seen.append(j["stage"]),
                                 poll_interval=0)
    assert out["stage"] == STAGE_DONE
    assert out["stage1_rc"] == 0
    assert calls == {"stage1": 1, "stage2": 1}
    # on_update 至少见到 段1起 / 段2起 / DONE 三次推进
    assert STAGE_STAGE1 in seen and STAGE_STAGE2 in seen and STAGE_DONE in seen


def test_run_to_completion_stage1_failed_no_stage2(monkeypatch, new_job):
    calls = _mock_engine(monkeypatch, [
        {"status": "FAILED", "rc": 1, "error": "段1 退出码 1"},
    ])
    seen = []
    out = orch.run_to_completion(new_job, on_update=lambda j: seen.append(j["stage"]),
                                 poll_interval=0)
    assert out["stage"] == STAGE_FAILED
    assert calls["stage1"] == 1
    assert calls["stage2"] == 0
    assert STAGE_FAILED in seen


def test_run_to_completion_start_stage1_raises(monkeypatch, new_job):
    def boom(jid, **kw):
        raise orch.engine_ssh.SshTransferError("SGP 连接失败")
    monkeypatch.setattr(orch.engine_ssh, "start_stage1", boom)
    monkeypatch.setattr(orch.engine_ssh, "start_stage2",
                        lambda *a, **k: pytest.fail("起段1即失败不该到段2"))
    seen = []
    out = orch.run_to_completion(new_job, on_update=lambda j: seen.append(j["stage"]),
                                 poll_interval=0)
    assert out["stage"] == STAGE_FAILED
    assert "SGP 连接失败" in out["error"]
    assert seen == [STAGE_FAILED]


def test_run_to_completion_retry_resumes_stage2(monkeypatch, new_job):
    """retry：stage1_rc==0 的 job 直接从段2 起，不再调 start_stage1。"""
    new_job["stage1_rc"] = 0
    new_job["stage"] = STAGE_NEW
    orch._save(new_job)
    calls = _mock_engine(monkeypatch, [
        {"status": "DONE", "rc": 0},   # 段2
    ])
    out = orch.run_to_completion(new_job, poll_interval=0)
    assert out["stage"] == STAGE_DONE
    assert calls["stage1"] == 0
    assert calls["stage2"] == 1
