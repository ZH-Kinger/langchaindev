"""#50 查询按 ID 推结果卡与自动完成通知重复（真机：两张一样的完成卡）。

覆盖 core/feishu_bot/actions.py::_h_query_progress_by_id 的后台 `_bg`：
  终态推结果卡前先 `dsw_scheduler._claim_dataflow_notify(jid)` NX 闸门，抢到才 `_send_card(result_card)`、
  抢不到只 `_send_text` 文本回执；非终态照推 progress_card（不碰闸门）。与在线线程/对账/文本查询
  `_push_terminal_result_card` 对齐——全链路对同一 job_id 终态结果卡「至多推一张」。

断言点：
  1) 终态 + claim True → `_send_card` 推 result_card（首参 open_id）、不发文本；
  2) 终态 + claim False（已被别路径推过）→ 不调 `_send_card`、改 `_send_text` 回执含 jid；
  3) 非终态（CROSSING/RUNNING）→ 照推 progress_card、不调 `_claim_dataflow_notify`；
  4) 三前缀（vepfs-/cpfs-/tr-）都走该逻辑；refresh 返 None → `_send_text` 未找到；
  5) 回归：同一次点击双投递被 `cpfs:query:{open_id}:{jid}` 8s NX 锁挡（第二次只回 toast「查询中…」不进 _bg）。
"""
import importlib
import json

import pytest

from core.feishu_bot import actions

# jid 前缀 → (orchestrator 模块, cards 模块)。_bg 内部按前缀 import 这两个模块。
CHAINS = {
    "vepfs-abc": ("core.vepfs_dataflow.orchestrator", "core.vepfs_dataflow.cards"),
    "cpfs-abc":  ("core.cpfs_dataflow.orchestrator", "core.cpfs_dataflow.cards"),
    "tr-abc":    ("core.transfer.orchestrator", "core.transfer.cards"),
}
# 非终态 stage：tr- 是 CROSSING，cpfs/vepfs 是 RUNNING（三 orchestrator 常量值即字面量）。
INFLIGHT = {"vepfs-abc": "RUNNING", "cpfs-abc": "RUNNING", "tr-abc": "CROSSING"}
ALL_JIDS = list(CHAINS)


class _SyncThread:
    """同步跑后台线程体，便于断言 _bg 推送行为（不真起线程）。"""
    def __init__(self, target=None, daemon=None, **k):
        self._t = target

    def start(self):
        self._t()


@pytest.fixture
def gate(monkeypatch):
    """桩掉 _bg 依赖：同步线程 + NX 闸门 + _send_card/_send_text + 三链 refresh/result_card/progress_card。

    返回 state：
      cards  = [(open_id, chat, card), ...]  经 _send_card 推出的卡（result/progress 哨兵）
      texts  = [(open_id, chat, text), ...]  经 _send_text 推出的文本
      claim  = {"ret": bool, "calls": [jid,...]}  控制/记录 _claim_dataflow_notify
      jobs   = {jid: job|None}  refresh(jid) 的返回；缺键 → None（覆盖 refresh 返 None）
    """
    import core.dsw_scheduler as sched

    state = {
        "cards": [],
        "texts": [],
        "claim": {"ret": True, "calls": []},
        "jobs": {},
    }
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    def _claim(jid):
        state["claim"]["calls"].append(jid)
        return state["claim"]["ret"]

    monkeypatch.setattr(sched, "_claim_dataflow_notify", _claim)
    monkeypatch.setattr(sched, "_send_card",
                        lambda oid, chat, card: state["cards"].append((oid, chat, card)))
    monkeypatch.setattr(sched, "_send_text",
                        lambda oid, chat, text: state["texts"].append((oid, chat, text)))

    for orch_path, cards_path in CHAINS.values():
        orch = importlib.import_module(orch_path)
        cmod = importlib.import_module(cards_path)
        monkeypatch.setattr(orch, "refresh", lambda jid: state["jobs"].get(jid))
        # 哨兵卡：区分 result / progress，并回带 jid 以便断言是「哪个任务的哪种卡」
        monkeypatch.setattr(cmod, "result_card", lambda j: {"RESULT": j["job_id"]})
        monkeypatch.setattr(cmod, "progress_card", lambda j: {"PROGRESS": j["job_id"]})
    return state


def _call(jid, open_id="ou_me", chat_id="oc_x"):
    """chat_id 非空 → _bg 里 chat=chat_id，不触各 _cfg_*_chat()（隔离）。"""
    return actions._h_query_progress_by_id({}, open_id, chat_id, {"job_id": jid})


# ── 1) 终态 + 抢到闸门 → 推 result_card（首参 open_id）、不发文本 ──────────────────
@pytest.mark.parametrize("jid", ALL_JIDS)
@pytest.mark.parametrize("stage", ["DONE", "FAILED"])
def test_terminal_claim_won_pushes_result_card(gate, jid, stage):
    gate["claim"]["ret"] = True
    gate["jobs"][jid] = {"job_id": jid, "stage": stage}
    resp = _call(jid)

    assert resp["toast"]["type"] == "success" and "正在查询" in resp["toast"]["content"]
    assert "card" not in resp                                   # sync 秒回 toast，不原地替换
    # 抢到闸门 → 推一张 result_card，首参=open_id，卡是该任务的结果卡
    assert gate["claim"]["calls"] == [jid]
    assert len(gate["cards"]) == 1
    oid, chat, card = gate["cards"][0]
    assert oid == "ou_me" and chat == "oc_x"
    assert card == {"RESULT": jid}
    assert gate["texts"] == []                                  # 不发文本回执


# ── 2) 终态 + 闸门被占（别路径已推）→ 不推卡、改文本回执含 jid ─────────────────────
@pytest.mark.parametrize("jid", ALL_JIDS)
@pytest.mark.parametrize("stage", ["DONE", "FAILED"])
def test_terminal_claim_lost_sends_text_only(gate, jid, stage):
    gate["claim"]["ret"] = False
    gate["jobs"][jid] = {"job_id": jid, "stage": stage}
    _call(jid)

    assert gate["claim"]["calls"] == [jid]                      # 抢了闸门但没抢到
    assert gate["cards"] == []                                  # 关键：不重复推结果卡
    assert len(gate["texts"]) == 1
    oid, chat, text = gate["texts"][0]
    assert oid == "ou_me" and chat == "oc_x"
    # 文本回执带 job_id + 结果语义（阶段/完成），不锁死措辞（无 error → 「已完成（阶段 …）」）
    assert jid in text and stage in text


# ── 2b) 终态 FAILED 带 error + 闸门被占 → 文本回执含 jid + 失败原因 ─────────────────
@pytest.mark.parametrize("jid", ALL_JIDS)
def test_terminal_claim_lost_failed_text_has_error(gate, jid):
    gate["claim"]["ret"] = False
    gate["jobs"][jid] = {"job_id": jid, "stage": "FAILED", "error": "boom-xyz"}
    _call(jid)

    assert gate["cards"] == []
    assert len(gate["texts"]) == 1
    _, _, text = gate["texts"][0]
    assert jid in text and "boom-xyz" in text                   # 失败原因入回执


# ── 2c) Low-1：抢到闸门后 _send_card 抛异常 → 放开闸门(delete notified 键)、不冒泡 ────
def test_send_card_failure_releases_gate(gate, monkeypatch):
    import core.dsw_scheduler as sched
    from utils.redis_client import get_redis

    jid = "tr-abc"
    gate["claim"]["ret"] = True
    gate["jobs"][jid] = {"job_id": jid, "stage": "DONE"}
    # 模拟闸门已被 claim 抢到（桩 _claim 不真写键，这里手动占位，验证异常时被放开）
    get_redis().set(f"dataflow:notified:{jid}", 1)

    def _boom(*a, **k):
        raise RuntimeError("feishu down")
    monkeypatch.setattr(sched, "_send_card", _boom)

    resp = _call(jid)                                           # 异常被内层 try 吞，不冒泡
    assert resp["toast"]["type"] == "success"
    assert gate["claim"]["calls"] == [jid]                      # 确实抢到过闸门
    # 推送失败 → delete(dataflow:notified:{jid}) 放开闸门，避免自动完成通知被永久抑制
    assert get_redis().get(f"dataflow:notified:{jid}") is None
    assert gate["texts"] == []                                  # 不发文本回执（走的是 claim 成功分支）


# ── 3) 非终态 → 照推 progress_card，绝不碰 NX 闸门 ────────────────────────────────
@pytest.mark.parametrize("jid", ALL_JIDS)
def test_inflight_pushes_progress_without_gate(gate, jid):
    gate["jobs"][jid] = {"job_id": jid, "stage": INFLIGHT[jid]}
    _call(jid)

    assert gate["claim"]["calls"] == []                         # 进度卡不去重，从不抢闸门
    assert len(gate["cards"]) == 1
    oid, chat, card = gate["cards"][0]
    assert oid == "ou_me" and chat == "oc_x"
    assert card == {"PROGRESS": jid}                            # 进度卡（非结果卡）
    assert gate["texts"] == []


# ── 4) refresh 返 None（任务不存在/已过期）→ 文本「未找到」，不推卡、不碰闸门 ──────────
@pytest.mark.parametrize("jid", ALL_JIDS)
def test_refresh_none_sends_not_found(gate, jid):
    # gate["jobs"] 不含 jid → refresh 返 None
    _call(jid)

    assert gate["claim"]["calls"] == []
    assert gate["cards"] == []
    assert len(gate["texts"]) == 1
    oid, chat, text = gate["texts"][0]
    assert oid == "ou_me" and jid in text and "未找到" in text


# ── FAILED 终态用真实 result_card：含各链重试按钮（抢到闸门时确实推的是可重试的失败卡）──
def test_failed_real_result_card_has_retry_button(monkeypatch):
    """不桩 result_card，验 tr- FAILED 抢到闸门时推的真实结果卡带 retry_transfer 按钮。"""
    import core.dsw_scheduler as sched
    from core.transfer import orchestrator as o
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    monkeypatch.setattr(sched, "_claim_dataflow_notify", lambda jid: True)
    pushed = []
    monkeypatch.setattr(sched, "_send_card", lambda oid, chat, card: pushed.append(card))
    monkeypatch.setattr(sched, "_send_text", lambda *a: None)
    job = {"job_id": "tr-fail", "stage": o.STAGE_FAILED, "direction": "tos→oss",
           "source": "tos://s/a/", "dest": "oss://d/a/", "error": "boom"}
    monkeypatch.setattr(o, "refresh", lambda jid: job)
    actions._h_query_progress_by_id({}, "ou_me", "oc_x", {"job_id": "tr-fail"})
    assert pushed and "retry_transfer" in json.dumps(pushed[0], ensure_ascii=False)


# ── 5) 回归：同一次点击双投递被 cpfs:query 8s NX 锁挡（第二次不进 _bg）──────────────
def test_double_delivery_blocked_by_query_lock(gate):
    """飞书对同一点击双投递（老式回调 + 2.0 事件）→ 第二条被 (open_id, jid) 8s NX 锁挡，
    只回「查询中…」toast、不再进 _bg（不二次推卡）。"""
    jid = "tr-abc"
    gate["claim"]["ret"] = True
    gate["jobs"][jid] = {"job_id": jid, "stage": "DONE"}

    first = _call(jid)
    assert first["toast"]["type"] == "success"
    assert len(gate["cards"]) == 1                              # 第一次进 _bg 推了一张

    second = _call(jid)                                         # 同 open_id + 同 jid
    assert second["toast"]["type"] == "info" and "查询中" in second["toast"]["content"]
    assert "card" not in second
    # 第二次未进 _bg：卡数不变、闸门只被抢过一次
    assert len(gate["cards"]) == 1
    assert gate["claim"]["calls"] == [jid]
    # 不同 jid 不被误挡（另一把锁）
    gate["jobs"]["tr-other"] = {"job_id": "tr-other", "stage": "DONE"}
    third = _call("tr-other")
    assert third["toast"]["type"] == "success"
    assert len(gate["cards"]) == 2


# ── 前缀/空值回归（与既有 test_cpfs_feishu 对齐，防跑偏）─────────────────────────
def test_bad_prefix_returns_error_toast(gate):
    resp = _call("xyz-1")
    assert resp["toast"]["type"] == "error"
    assert gate["cards"] == [] and gate["texts"] == []


def test_empty_jid_is_noop(gate):
    assert actions._h_query_progress_by_id({}, "ou_me", "oc_x", {}) == {}
    assert gate["cards"] == [] and gate["texts"] == []
