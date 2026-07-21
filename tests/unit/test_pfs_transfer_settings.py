"""PFS 跨云直传 —— settings 默认 + validate() 告警（LOW-1）+ 对账 spec。"""
import json

from config.settings import settings


# ══════════════════════════════════════════════════════════════════════════════
# 配置默认
# ═════════════════════���════════════════════════════════════════════════════════

def test_settings_defaults_present():
    assert hasattr(settings, "PFS_TRANSFER_ENABLED")
    assert hasattr(settings, "PFS_STAGING_MAP_RAW")
    assert hasattr(settings, "PFS_TRANSFER_APPROVAL_TB")
    assert hasattr(settings, "PFS_TRANSFER_STAGING_CLEANUP")
    assert hasattr(settings, "PFS_TRANSFER_CHAT_ID")


# ═══════════════════════════════��══════════════════════════════════════════════
# validate()：启用 + 空 map → 告警；未启用/有 map → 不告警（LOW-1）
# ════════��═════════════════════════════════════════════════════════════════════

def _fields(missing):
    return [f for f, _impact in missing]


def test_validate_enabled_empty_map_warns(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_ENABLED", True)
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", "{}")
    assert "PFS_STAGING_MAP" in _fields(settings.validate())


def test_validate_enabled_default_string_warns(monkeypatch):
    """默认 '{}' 是非空串（_REQUIRED_FIELDS 的 not val 抓不到），仍须告警。"""
    monkeypatch.setattr(settings, "PFS_TRANSFER_ENABLED", True)
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", '{}')
    assert "PFS_STAGING_MAP" in _fields(settings.validate())


def test_validate_enabled_bad_json_warns(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_ENABLED", True)
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", "{not json")
    assert "PFS_STAGING_MAP" in _fields(settings.validate())


def test_validate_disabled_no_warn(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_ENABLED", False)
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", "{}")
    assert "PFS_STAGING_MAP" not in _fields(settings.validate())


def test_validate_enabled_with_map_no_warn(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_ENABLED", True)
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW",
                        json.dumps({"vepfs://fs-x": {"region": "cn-beijing", "tos_bucket": "b"}}))
    assert "PFS_STAGING_MAP" not in _fields(settings.validate())


# ══════════════════════════════════════════════════════════════════════════════
# 对账 spec：pfs 链在 _dataflow_reconcile_specs、active 含三段、契约齐
# ══════════════════════════════════════════════════════════════════════════════

def test_reconcile_specs_include_pfs():
    from core.dsw_scheduler import _dataflow_reconcile_specs
    specs = _dataflow_reconcile_specs()
    names = [s["name"] for s in specs]
    assert "pfs" in names


def test_reconcile_pfs_active_three_stages():
    from core.dsw_scheduler import _dataflow_reconcile_specs
    from core.pfs_transfer import orchestrator as o
    spec = next(s for s in _dataflow_reconcile_specs() if s["name"] == "pfs")
    assert spec["active"] == {o.STAGE_SINKING, o.STAGE_CROSSING, o.STAGE_PREHEATING}


def test_reconcile_pfs_contract_complete():
    """reconcile 依赖的契约：orchestrator 有 _KEY_PREFIX/get_job/refresh/_save/STAGE_DONE/
    STAGE_FAILED；cards 有 result_card；chat 可调用。"""
    from core.dsw_scheduler import _dataflow_reconcile_specs
    spec = next(s for s in _dataflow_reconcile_specs() if s["name"] == "pfs")
    o, cards = spec["o"], spec["cards"]
    for attr in ("_KEY_PREFIX", "get_job", "refresh", "_save", "STAGE_DONE", "STAGE_FAILED"):
        assert hasattr(o, attr), f"orchestrator 缺 {attr}"
    assert hasattr(cards, "result_card")
    assert callable(spec["chat"])
    assert spec["chat"]() is not None or spec["chat"]() == ""   # 可调用、不抛
