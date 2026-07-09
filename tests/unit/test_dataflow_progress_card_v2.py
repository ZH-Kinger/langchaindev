"""#46-A2：CPFS/vePFS 新增 progress_card_v2（schema 2.0 纯展示），confirm 默认回 2.0、retry 回 1.0。

飞书 200830：2.0 确认卡不能原地替换成 1.0 卡（静默失败，确认到完成间无反馈）。修法照 transfer #33：
  确认卡(2.0) → progress_card_v2(2.0，同家族替换)；重试(1.0 结果卡) → progress_card(1.0)。
"""
import importlib
import pytest


# (cards 模块, confirm handler, retry handler, orchestrator 模块)
_CHAINS = [
    ("core.cpfs_dataflow.cards", "_h_confirm_cpfs_dataflow", "_h_retry_cpfs_dataflow",
     "core.cpfs_dataflow.orchestrator"),
    ("core.vepfs_dataflow.cards", "_h_confirm_vepfs_dataflow", "_h_retry_vepfs_dataflow",
     "core.vepfs_dataflow.orchestrator"),
]
_IDS = ["cpfs", "vepfs"]


def _mod(path):
    return importlib.import_module(path)


class _SyncThread:
    def __init__(self, target=None, daemon=None, **k): self._t = target
    def start(self):
        if self._t:
            self._t()


def _full_job(orch, stage=None):
    # 含 1.0 progress_card 所需字段（fs_id/directory/sub_path），真机 job 恒有
    return {"job_id": "j-v2", "stage": stage or orch.STAGE_NEW, "created_by": "ou_1",
            "operation_label": "沉降", "fs_id": "fs-1", "directory": "/cwr/o/",
            "cpfs_dir": "/cwr/o/", "oss_bucket": "bk", "oss_prefix": "e/",
            "sub_path": "/wzh/", "tos_bucket": "bk", "tos_prefix": "e/"}


# ── progress_card_v2 纯渲染 ────────────────────────────────────────────────────

@pytest.mark.parametrize("cards_path", [c[0] for c in _CHAINS], ids=_IDS)
def test_progress_card_v2_is_schema_2_0_no_button(cards_path):
    cards = _mod(cards_path)
    job = {"job_id": "j-x", "stage": "RUNNING", "operation_label": "预热",
           "cpfs_dir": "/a/b/", "oss_bucket": "bk", "oss_prefix": "p/",
           "sub_path": "/a/b/", "tos_bucket": "bk", "tos_prefix": "p/"}
    d = cards.progress_card_v2(job)
    assert d["schema"] == "2.0"
    assert "button" not in str(d) and "form" not in str(d)     # 纯展示、无按钮
    assert "j-x" in str(d)


@pytest.mark.parametrize("cards_path", [c[0] for c in _CHAINS], ids=_IDS)
def test_progress_card_v2_missing_optional_fields_no_keyerror(cards_path):
    """job 只有 job_id/stage（缺 operation_label/桶/前缀等可选字段）→ 不抛 KeyError。"""
    cards = _mod(cards_path)
    d = cards.progress_card_v2({"job_id": "j-min", "stage": "RUNNING"})
    assert d["schema"] == "2.0"
    assert "j-min" in str(d)


# ── confirm 默认 reply_v2=True → 2.0；retry(reply_v2=False) → 1.0 ────────────────

@pytest.mark.parametrize("cards_path,confirm_name,retry_name,orch_path", _CHAINS, ids=_IDS)
def test_confirm_default_returns_schema_2_0(monkeypatch, cards_path, confirm_name,
                                            retry_name, orch_path):
    from core.feishu_bot import actions
    orch = _mod(orch_path)
    job = _full_job(orch)
    store = {job["job_id"]: job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(orch, "run_to_completion", lambda j, **k: j)
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    r = getattr(actions, confirm_name)({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert r["toast"]["type"] == "success"
    data = r["card"]["data"]
    assert data["schema"] == "2.0"                 # 与 2.0 确认卡同家族
    assert "button" not in str(data) and "form" not in str(data)


@pytest.mark.parametrize("cards_path,confirm_name,retry_name,orch_path", _CHAINS, ids=_IDS)
def test_retry_returns_schema_1_0(monkeypatch, cards_path, confirm_name, retry_name, orch_path):
    """retry 由 1.0 结果卡触发 → reply_v2=False → 回 1.0 progress_card（无 schema 键）。"""
    from core.feishu_bot import actions
    orch = _mod(orch_path)
    job = _full_job(orch, stage=orch.STAGE_FAILED)
    job["error"] = "boom"
    store = {job["job_id"]: job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(orch, "run_to_completion", lambda j, **k: j)
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    r = getattr(actions, retry_name)({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert r["toast"]["type"] == "success"
    assert r["card"]["type"] == "raw"
    assert "schema" not in r["card"]["data"]       # 老式 1.0 卡（card() 不写 schema 键）
