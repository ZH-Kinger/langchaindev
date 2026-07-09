"""HIGH#3（#37）：查询进度不再在 3s 回调里 refresh（poll 云端数十秒→超时→飞书重投→
被 15s 去重吞成 {}→原地卡永远刷不出）。

新行为：四个 `_h_query_*` + `_h_query_progress_by_id` sync 只读 `get_job` 秒回当前卡，
未完成时起后台线程 `_async_refresh_and_push` 真正 refresh 后推更新卡（阶段推进才推、
终态经 `_claim_dataflow_notify` 去重）。
"""
import importlib

import pytest

from core.feishu_bot import actions


# 四条查询链：(handler 名, orchestrator 路径, cards 路径, 在途 stage)
_QUERY_CHAINS = [
    ("_h_query_transfer_progress", "core.transfer.orchestrator",       "core.transfer.cards",       "CROSSING"),
    ("_h_query_cpfs_progress",     "core.cpfs_dataflow.orchestrator",  "core.cpfs_dataflow.cards",  "RUNNING"),
    ("_h_query_vepfs_progress",    "core.vepfs_dataflow.orchestrator", "core.vepfs_dataflow.cards", "RUNNING"),
    ("_h_query_bucket_transfer",   "core.bucket_transfer.orchestrator","core.bucket_transfer.cards","RUNNING"),
]
_IDS = [c[0] for c in _QUERY_CHAINS]


def _mod(path):
    return importlib.import_module(path)


class _NoStartThread:
    """捕获后台线程构造但不运行 —— 用来证明 sync 路径本身不 refresh/不 poll 云端。"""
    def __init__(self, target=None, args=(), daemon=None, **k):
        self.target, self.args = target, args
        _NoStartThread.made.append(self)

    def start(self):
        pass


@pytest.fixture(autouse=True)
def _reset_threads():
    _NoStartThread.made = []
    yield
    _NoStartThread.made = []


# ── sync 路径：在途 → 秒回当前卡、不在 sync 内 refresh ──────────────────────────

@pytest.mark.parametrize("handler,orch_path,cards_path,running", _QUERY_CHAINS, ids=_IDS)
def test_query_running_returns_current_card_no_sync_refresh(monkeypatch, handler, orch_path,
                                                            cards_path, running):
    orch, cards = _mod(orch_path), _mod(cards_path)
    monkeypatch.setattr(actions.threading, "Thread", _NoStartThread)
    monkeypatch.setattr(orch, "get_job", lambda jid: {"job_id": jid, "stage": running})

    def _no_refresh(jid):
        raise AssertionError("sync 路径不得 refresh（会 poll 云端数十秒→超 3s）")
    monkeypatch.setattr(orch, "refresh", _no_refresh)
    monkeypatch.setattr(cards, "progress_card", lambda j: {"PROGRESS": j["stage"]})
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["stage"]})

    out = getattr(actions, handler)({"job_id": "j-run"}, "ou", "chat", {})
    # 立即回当前(进度)卡
    assert out["card"]["type"] == "raw" and out["card"]["data"] == {"PROGRESS": running}
    assert "后台刷新中" in out["toast"]["content"]
    # 起了一条后台线程，且目标是 _async_refresh_and_push（真正的 refresh 在那里）
    assert len(_NoStartThread.made) == 1
    assert _NoStartThread.made[0].target is actions._async_refresh_and_push


@pytest.mark.parametrize("handler,orch_path,cards_path,running", _QUERY_CHAINS, ids=_IDS)
def test_query_terminal_returns_result_no_thread(monkeypatch, handler, orch_path, cards_path, running):
    orch, cards = _mod(orch_path), _mod(cards_path)
    monkeypatch.setattr(actions.threading, "Thread", _NoStartThread)
    monkeypatch.setattr(orch, "get_job", lambda jid: {"job_id": jid, "stage": orch.STAGE_DONE})
    monkeypatch.setattr(cards, "progress_card", lambda j: {"PROGRESS": j["stage"]})
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["stage"]})

    out = getattr(actions, handler)({"job_id": "j-done"}, "ou", "chat", {})
    assert out["card"]["data"] == {"RESULT": orch.STAGE_DONE}
    assert "后台刷新中" not in out["toast"]["content"]
    assert _NoStartThread.made == []           # 终态不起后台线程


@pytest.mark.parametrize("handler,orch_path,cards_path,running", _QUERY_CHAINS, ids=_IDS)
def test_query_missing_job_error_toast(monkeypatch, handler, orch_path, cards_path, running):
    orch = _mod(orch_path)
    monkeypatch.setattr(actions.threading, "Thread", _NoStartThread)
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    out = getattr(actions, handler)({"job_id": "nope"}, "ou", "chat", {})
    assert out["toast"]["type"] == "error"
    assert "card" not in out
    assert _NoStartThread.made == []


# ── _async_refresh_and_push：阶段变化/终态去重 ─────────────────────────────────

class _FakeOrch:
    STAGE_DONE = "DONE"
    STAGE_FAILED = "FAILED"

    def __init__(self, prev, new):
        self._prev, self._new = prev, new
        self.refreshed = []

    def get_job(self, jid):
        return {"job_id": jid, "stage": self._prev} if self._prev else None

    def refresh(self, jid):
        self.refreshed.append(jid)
        return {"job_id": jid, "stage": self._new} if self._new else None


@pytest.fixture
def push_capture(monkeypatch):
    import core.dsw_scheduler as s
    sent = []
    monkeypatch.setattr(s, "_send_card", lambda o, c, card: sent.append((c, card)))
    return s, sent


def _prog(j):
    return {"PROGRESS": j["stage"]}


def _res(j):
    return {"RESULT": j["stage"]}


def test_async_no_stage_change_no_push(monkeypatch, push_capture):
    s, sent = push_capture
    orch = _FakeOrch(prev="RUNNING", new="RUNNING")
    actions._async_refresh_and_push(orch, "j1", _prog, _res, "oc_chat")
    assert orch.refreshed == ["j1"]     # 后台真的 refresh 了
    assert sent == []                   # 无变化 → 不刷屏


def test_async_stage_advance_pushes_progress(monkeypatch, push_capture):
    s, sent = push_capture
    orch = _FakeOrch(prev="SINKING", new="CROSSING")
    actions._async_refresh_and_push(orch, "j2", _prog, _res, "oc_chat")
    assert sent == [("oc_chat", {"PROGRESS": "CROSSING"})]


def test_async_terminal_pushes_result_when_gate_open(monkeypatch, push_capture):
    s, sent = push_capture
    monkeypatch.setattr(s, "_claim_dataflow_notify", lambda jid: True)   # 抢到闸门
    orch = _FakeOrch(prev="RUNNING", new="DONE")
    actions._async_refresh_and_push(orch, "j3", _prog, _res, "oc_chat")
    assert sent == [("oc_chat", {"RESULT": "DONE"})]


def test_async_terminal_skips_when_gate_taken(monkeypatch, push_capture):
    s, sent = push_capture
    monkeypatch.setattr(s, "_claim_dataflow_notify", lambda jid: False)  # 在线线程/对账已推
    orch = _FakeOrch(prev="RUNNING", new="FAILED")
    actions._async_refresh_and_push(orch, "j4", _prog, _res, "oc_chat")
    assert sent == []


def test_async_refresh_none_no_push(monkeypatch, push_capture):
    s, sent = push_capture
    orch = _FakeOrch(prev="RUNNING", new=None)   # refresh 返回 None（任务消失）
    actions._async_refresh_and_push(orch, "j5", _prog, _res, "oc_chat")
    assert sent == []


def test_async_terminal_pushes_even_if_prev_already_terminal(monkeypatch, push_capture):
    """prev 已是终态、refresh 仍终态（stage 未变）：仍走闸门推结果卡（终态不受“无变化”短路影响）。"""
    s, sent = push_capture
    monkeypatch.setattr(s, "_claim_dataflow_notify", lambda jid: True)
    orch = _FakeOrch(prev="DONE", new="DONE")
    actions._async_refresh_and_push(orch, "j6", _prog, _res, "oc_chat")
    assert sent == [("oc_chat", {"RESULT": "DONE"})]


# ── #41 _async_refresh_and_push 推送目标优先 created_by ──────────────────────────

class _OrchCreatedBy:
    STAGE_DONE = "DONE"
    STAGE_FAILED = "FAILED"

    def __init__(self, prev, new, created_by):
        self._prev, self._new, self._cb = prev, new, created_by

    def get_job(self, jid):
        return {"job_id": jid, "stage": self._prev}

    def refresh(self, jid):
        j = {"job_id": jid, "stage": self._new}
        if self._cb is not None:
            j["created_by"] = self._cb
        return j


@pytest.fixture
def push_capture_target(monkeypatch):
    """捕获 _send_card 完整三参，用于断言首参=推送目标（created_by）。"""
    import core.dsw_scheduler as s
    sent = []
    monkeypatch.setattr(s, "_send_card", lambda o, c, card: sent.append((o, c, card)))
    return s, sent


def test_async_terminal_target_is_created_by(monkeypatch, push_capture_target):
    s, sent = push_capture_target
    monkeypatch.setattr(s, "_claim_dataflow_notify", lambda jid: True)
    orch = _OrchCreatedBy(prev="RUNNING", new="DONE", created_by="ou_creator")
    actions._async_refresh_and_push(orch, "j-t", _prog, _res, "oc_chat")
    assert sent and sent[0][0] == "ou_creator"      # 终态推给发起人
    assert sent[0][1] == "oc_chat"


def test_async_progress_target_is_created_by(monkeypatch, push_capture_target):
    s, sent = push_capture_target
    orch = _OrchCreatedBy(prev="SINKING", new="CROSSING", created_by="ou_creator")
    actions._async_refresh_and_push(orch, "j-p", _prog, _res, "oc_chat")
    assert sent and sent[0][0] == "ou_creator"      # 进度也推给发起人
    assert sent[0][2] == {"PROGRESS": "CROSSING"}


def test_async_target_falls_back_empty_when_no_created_by(monkeypatch, push_capture_target):
    s, sent = push_capture_target
    monkeypatch.setattr(s, "_claim_dataflow_notify", lambda jid: True)
    orch = _OrchCreatedBy(prev="RUNNING", new="DONE", created_by=None)
    actions._async_refresh_and_push(orch, "j-e", _prog, _res, "oc_chat")
    assert sent and sent[0][0] == ""                 # 降级：空目标→配置频道


# ── _h_query_progress_by_id：前缀校验 + 秒回 toast + 后台推送 ─────────────────────

class _SyncThread:
    def __init__(self, target=None, daemon=None, **k): self._t = target
    def start(self): self._t()


def test_by_id_illegal_prefix_error_toast(monkeypatch):
    out = actions._h_query_progress_by_id({}, "ou", "oc", {"job_id": "weird-123"})
    assert out["toast"]["type"] == "error"
    assert "vepfs-" in out["toast"]["content"] and "tr-" in out["toast"]["content"]


def test_by_id_empty_is_noop():
    assert actions._h_query_progress_by_id({}, "ou", "oc", {}) == {}


@pytest.mark.parametrize("jid,orch_path,cards_path", [
    ("tr-abc",    "core.transfer.orchestrator",       "core.transfer.cards"),
    ("cpfs-abc",  "core.cpfs_dataflow.orchestrator",  "core.cpfs_dataflow.cards"),
    ("vepfs-abc", "core.vepfs_dataflow.orchestrator", "core.vepfs_dataflow.cards"),
])
def test_by_id_legal_prefix_toast_and_bg_push(monkeypatch, jid, orch_path, cards_path):
    orch, cards = _mod(orch_path), _mod(cards_path)
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)   # 同步跑 _bg
    monkeypatch.setattr(orch, "refresh", lambda j: {"job_id": j, "stage": orch.STAGE_DONE})
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["job_id"]})
    monkeypatch.setattr(cards, "progress_card", lambda j: {"PROGRESS": j["job_id"]})
    import core.dsw_scheduler as s
    pushed = []
    monkeypatch.setattr(s, "_send_card", lambda o, c, card: pushed.append(card))

    out = actions._h_query_progress_by_id({}, "ou", "oc", {"job_id": jid})
    assert out["toast"]["type"] == "success" and "正在查询" in out["toast"]["content"]
    assert "card" not in out                       # sync 不原地替换
    assert pushed == [{"RESULT": jid}]             # 后台推终态结果卡


def test_by_id_dedup_second_click_swallowed(monkeypatch):
    """同一 (open_id, job_id) 8s 内第二次点击 → info“查询中…”，不重复起线程。"""
    made = []
    monkeypatch.setattr(actions.threading, "Thread",
                        lambda target=None, daemon=None, **k: made.append(target)
                        or type("T", (), {"start": lambda self: None})())
    r1 = actions._h_query_progress_by_id({}, "ou_dedup", "oc", {"job_id": "tr-dd"})
    r2 = actions._h_query_progress_by_id({}, "ou_dedup", "oc", {"job_id": "tr-dd"})
    assert "正在查询" in r1["toast"]["content"]
    assert r2["toast"]["type"] == "info" and "查询中" in r2["toast"]["content"]
    assert len(made) == 1                          # 第二次未起后台线程
