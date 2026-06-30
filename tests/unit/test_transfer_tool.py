"""tools/transfer/transfer.py: manage_transfer 工具行为测试。"""
import json
import pytest


@pytest.fixture
def cfg(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "TRANSFER_BUCKET_MAP_RAW",
                        json.dumps({"tos://src-bucket": "dst-oss-bucket"}))
    monkeypatch.setattr(settings, "TRANSFER_APPROVAL_TB", 1.0)
    monkeypatch.setattr(settings, "TRANSFER_OVERWRITE_DEFAULT", "no")
    return settings


@pytest.fixture
def no_estimate(monkeypatch):
    """探测大小可控：默认 0，便于按需置超/低阈值。"""
    from core.transfer import orchestrator
    box = {"bytes": 0, "objs": 0}
    monkeypatch.setattr(orchestrator, "estimate_source", lambda plan: (box["bytes"], box["objs"]))
    return box


def test_plan_dry_run_no_job(cfg, no_estimate):
    from tools.transfer.transfer import manage_transfer
    out = manage_transfer("plan", source="tos://src-bucket/a/")
    assert "迁移计划" in out and "tos→oss" in out


def test_status_missing_job(cfg):
    from tools.transfer.transfer import manage_transfer
    out = manage_transfer("status", job_id="tr-nope")
    assert "未找到" in out


def test_apply_over_threshold_pushes_admin_card(cfg, no_estimate, monkeypatch):
    """超阈值 apply：不自动启动，推确认卡给管理员。"""
    no_estimate["bytes"] = 2 * 1024 ** 4   # 2TB
    pushed = {}
    import tools.transfer.transfer as t
    monkeypatch.setattr(t, "_push_admin_confirm", lambda job: pushed.setdefault("job", job) or True)
    # 不应启动后台线程
    started = {"n": 0}
    import threading
    real_thread = threading.Thread
    monkeypatch.setattr(threading, "Thread",
                        lambda *a, **k: started.__setitem__("n", started["n"] + 1) or real_thread(target=lambda: None))

    out = t.manage_transfer("apply", source="tos://src-bucket/a/", open_id="ou_1")
    assert "超过审批阈值" in out
    assert pushed.get("job") is not None        # 确认卡已推
    assert started["n"] == 0                     # 未启动迁移线程


def test_apply_under_threshold_starts(cfg, no_estimate, monkeypatch):
    """阈值内 apply：启动后台执行。"""
    no_estimate["bytes"] = 100 * 1024 ** 3      # 0.1TB
    import tools.transfer.transfer as t
    from core.transfer import orchestrator
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    out = t.manage_transfer("apply", source="tos://src-bucket/a/", open_id="ou_1")
    assert "已提交迁移任务" in out


def test_unsupported_direction(cfg, no_estimate, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "TRANSFER_BUCKET_MAP_RAW", json.dumps({"oss://b": "t"}))
    from tools.transfer.transfer import manage_transfer
    out = manage_transfer("plan", source="oss://b/a/")
    assert "尚未实现" in out or "暂未支持" in out or "tos_mig" in out
