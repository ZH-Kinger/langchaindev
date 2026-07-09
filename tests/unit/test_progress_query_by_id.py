"""#49 跨云迁移(tr-)/CPFS/vePFS 按任务ID查进度 —— 文本查询实时化(refresh 自愈)。

覆盖 core/feishu_bot/messages.py::_handle_progress_query 三分支从「只 get_job 读旧 Redis」
改为「o.refresh(jid) or o.get_job(jid)」后的行为：
  1) 三种前缀都调用对应 orchestrator.refresh（本次核心保证）；in-flight/终态两态文案。
  2) refresh 返 None 回退 get_job；get_job 也 None → 「未找到任务」；refresh 内部吞 poll 异常仍返可用 job。
  3) tr- 消息补 bytes_done/bytes_total 实时进度（有 done 显示 "done / total"，无则只 total）。
  4) transfer.cards.progress_card_v2 info_md 含 job_id + 「查询进度」提示行，仍是 schema 2.0 无按钮/表单。
  5) 无 ID 的进度查询仍弹 query_input_card（回归）。
"""
import importlib
import json

import pytest

from core.feishu_bot import messages, messaging

# jid → 该前缀走的 orchestrator 模块路径（messages 内是 `from core.X import orchestrator as o`）
VEPFS_JID = "vepfs-abcdef"
CPFS_JID = "cpfs-abcdef"
TR_JID = "tr-abcdef"
MODPATH = {
    VEPFS_JID: "core.vepfs_dataflow.orchestrator",
    CPFS_JID: "core.cpfs_dataflow.orchestrator",
    TR_JID: "core.transfer.orchestrator",
}
INFLIGHT_STAGE = {VEPFS_JID: "RUNNING", CPFS_JID: "RUNNING", TR_JID: "CROSSING"}

# jid 前缀 → chain 名（_push_terminal_result_card 的第一参）+ 该 chain 的配置频道桩返回值
CHAIN_OF = {VEPFS_JID: "vepfs", CPFS_JID: "cpfs", TR_JID: "transfer"}
CHAT_OF = {"vepfs": "oc_vepfs", "cpfs": "oc_cpfs", "transfer": "oc_tr"}
CARDS_MOD = {
    "vepfs": "core.vepfs_dataflow.cards",
    "cpfs": "core.cpfs_dataflow.cards",
    "transfer": "core.transfer.cards",
}
RETRY_ACTION = {
    "vepfs": "retry_vepfs_dataflow",
    "cpfs": "retry_cpfs_dataflow",
    "transfer": "retry_transfer",
}


@pytest.fixture
def cap(monkeypatch):
    """捕获 _handle_progress_query 的文本/卡片回复。"""
    texts, cards = [], []
    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, text: texts.append(text))
    monkeypatch.setattr(messaging, "_feishu_reply_card", lambda mid, card: cards.append(card))
    return {"texts": texts, "cards": cards}


@pytest.fixture
def push_spy(monkeypatch):
    """桩掉终态补推链路：NX 闸门(默认放行) + _send_card + 三链配置频道 + 三链 result_card(哨兵)。"""
    import importlib as _il
    import core.dsw_scheduler as sched
    from core.feishu_bot import actions

    sent = []
    claim = {"ret": True, "calls": []}
    monkeypatch.setattr(sched, "_claim_dataflow_notify",
                        lambda jid: (claim["calls"].append(jid), claim["ret"])[1])
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append((o, c, card)))
    monkeypatch.setattr(actions, "_cfg_vepfs_chat", lambda: "oc_vepfs")
    monkeypatch.setattr(actions, "_cfg_cpfs_chat", lambda: "oc_cpfs")
    monkeypatch.setattr(actions, "_cfg_chat", lambda: "oc_tr")
    for mp in CARDS_MOD.values():
        m = _il.import_module(mp)
        monkeypatch.setattr(m, "result_card", lambda j: {"RESULT": j["job_id"]})
    return {"sent": sent, "claim": claim}


def _inflight_job(jid):
    if jid.startswith("tr-"):
        return {"job_id": jid, "stage": "CROSSING", "direction": "tos→oss",
                "bytes_done": 150 * 1024 ** 2, "bytes_total": 300 * 1024 ** 2}
    return {"job_id": jid, "stage": "RUNNING", "operation_label": "预热",
            "files_done": 3, "files_total": 10, "bytes_done": 2048}


def _terminal_job(jid):
    if jid.startswith("tr-"):
        return {"job_id": jid, "stage": "DONE", "direction": "tos→oss", "created_by": "ou_creator",
                "bytes_done": 300 * 1024 ** 2, "bytes_total": 300 * 1024 ** 2}
    return {"job_id": jid, "stage": "DONE", "operation_label": "预热", "created_by": "ou_creator",
            "files_done": 10, "files_total": 10, "bytes_done": 4096}


def _spy(monkeypatch, jid, *, refresh_ret, get_job_ret):
    """给该 jid 对应的 orchestrator 打 refresh/get_job 桩，返回调用记录。"""
    mod = importlib.import_module(MODPATH[jid])
    calls = {"refresh": [], "get_job": []}
    monkeypatch.setattr(mod, "refresh",
                        lambda j: (calls["refresh"].append(j), refresh_ret)[1])
    monkeypatch.setattr(mod, "get_job",
                        lambda j: (calls["get_job"].append(j), get_job_ret)[1])
    return calls


# ── 1) 三前缀都调 refresh：in-flight ─────────────────────────────────────────────

@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_query_by_id_calls_refresh_inflight(jid, cap, push_spy, monkeypatch):
    calls = _spy(monkeypatch, jid, refresh_ret=_inflight_job(jid), get_job_ret=None)
    messages._handle_progress_query("m1", f"查询进度 {jid}")

    assert calls["refresh"] == [jid]         # 核心：本次改动——按 ID 查会 refresh
    assert calls["get_job"] == []            # refresh 命中真值 → 短路，不再读旧 Redis
    txt = cap["texts"][0]
    assert jid in txt
    assert INFLIGHT_STAGE[jid] in txt        # 在途 → 进度文案带在途 stage
    assert push_spy["sent"] == []            # 在途不补推结果卡（也不抢 NX 闸门）
    assert push_spy["claim"]["calls"] == []


# ── 1) 三前缀都调 refresh：终态（文本 + 额外补推结果卡到 created_by） ────────────────

@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_query_by_id_calls_refresh_terminal(jid, cap, push_spy, monkeypatch):
    calls = _spy(monkeypatch, jid, refresh_ret=_terminal_job(jid), get_job_ret=None)
    messages._handle_progress_query("m1", f"查询进度 {jid}")

    assert calls["refresh"] == [jid]
    txt = cap["texts"][0]                     # 文本仍照常发（补推卡是额外一张，不替代文本）
    assert jid in txt and "DONE" in txt       # refresh 自愈后看到真实终态
    # 终态 → 经 NX 闸门补推对应链 result_card，首参=job.created_by、次参=该链配置频道
    chain = CHAIN_OF[jid]
    assert push_spy["claim"]["calls"] == [jid]
    assert push_spy["sent"] == [("ou_creator", CHAT_OF[chain], {"RESULT": jid})]


# ── 2) refresh None → 回退 get_job；两者都 None → 未找到 ─────────────────────────

@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_refresh_none_falls_back_to_get_job_then_not_found(jid, cap, monkeypatch):
    calls = _spy(monkeypatch, jid, refresh_ret=None, get_job_ret=None)
    messages._handle_progress_query("m1", f"查询进度 {jid}")

    assert calls["refresh"] == [jid]         # 先 refresh
    assert calls["get_job"] == [jid]         # refresh 返 None → 回退 get_job
    assert cap["texts"] and "未找到任务" in cap["texts"][0]


@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_refresh_none_uses_get_job_stale_value(jid, cap, monkeypatch):
    """refresh 返 None（如无 task_id 不重查），get_job 有旧记录 → 用旧记录出文案，不误报未找到。"""
    calls = _spy(monkeypatch, jid, refresh_ret=None, get_job_ret=_inflight_job(jid))
    messages._handle_progress_query("m1", f"查询进度 {jid}")

    assert calls["refresh"] == [jid] and calls["get_job"] == [jid]
    txt = cap["texts"][0]
    assert jid in txt and INFLIGHT_STAGE[jid] in txt
    assert "未找到任务" not in txt


# ── 2) refresh 自身 try 吞 poll 异常仍返可用 job（测真实 orchestrator.refresh）─────

@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_real_refresh_swallows_poll_error_returns_job(jid, monkeypatch):
    mod = importlib.import_module(MODPATH[jid])
    if jid.startswith("tr-"):
        job = {"job_id": jid, "stage": mod.STAGE_CROSSING, "cross_job_name": "wuji_il_x",
               "source": "tos://a/x/", "dest": "oss://b/y/", "direction": "tos→oss"}
    else:
        job = {"job_id": jid, "stage": mod.STAGE_RUNNING, "task_id": "task-x",
               "operation_label": "预热"}
    mod._save(job)

    def _boom(j):
        raise RuntimeError("cloud poll 挂了")
    monkeypatch.setattr(mod, "poll_once", _boom)

    out = mod.refresh(jid)                   # refresh 内部 try 吞掉 poll 异常
    assert out is not None                   # 仍返回可用 job（不是 None）
    assert out["job_id"] == jid
    # 阶段维持原样（poll 抛错未能推进/回落）
    assert out["stage"] == (mod.STAGE_CROSSING if jid.startswith("tr-") else mod.STAGE_RUNNING)


# ── 3) tr- 消息补 bytes_done / bytes_total 实时进度 ─────────────────────────────

def test_tr_message_shows_bytes_done_and_total(cap, monkeypatch):
    job = {"job_id": TR_JID, "stage": "CROSSING", "direction": "tos→oss",
           "bytes_done": 150 * 1024 ** 2, "bytes_total": 300 * 1024 ** 2}
    _spy(monkeypatch, TR_JID, refresh_ret=job, get_job_ret=None)
    messages._handle_progress_query("m1", f"查询进度 {TR_JID}")

    txt = cap["texts"][0]
    from core.transfer.orchestrator import fmt_size
    done = fmt_size(150 * 1024 ** 2)
    total = fmt_size(300 * 1024 ** 2)
    assert f"{done} / {total}" in txt        # 有 done → "done / total"
    assert done == "150.0 MB" and total == "300.0 MB"


def test_tr_message_total_only_when_no_bytes_done(cap, monkeypatch):
    job = {"job_id": TR_JID, "stage": "CROSSING", "direction": "tos→oss",
           "bytes_total": 300 * 1024 ** 2}     # 无 bytes_done
    _spy(monkeypatch, TR_JID, refresh_ret=job, get_job_ret=None)
    messages._handle_progress_query("m1", f"查询进度 {TR_JID}")

    txt = cap["texts"][0]
    from core.transfer.orchestrator import fmt_size
    total = fmt_size(300 * 1024 ** 2)
    assert f"进度 {total}" in txt      # "进度 <total>"
    assert " / " not in txt                    # 无 done → 不出现 "x / y"


def test_tr_message_includes_error_when_failed(cap, push_spy, monkeypatch):
    job = {"job_id": TR_JID, "stage": "FAILED", "direction": "tos→oss",
           "bytes_total": 10, "error": "boom-detail", "created_by": "ou_creator"}
    _spy(monkeypatch, TR_JID, refresh_ret=job, get_job_ret=None)
    messages._handle_progress_query("m1", f"查询进度 {TR_JID}")

    txt = cap["texts"][0]
    assert "FAILED" in txt and "boom-detail" in txt
    # FAILED 也是终态 → 补推结果卡（此处 result_card 已被哨兵桩替换）
    assert push_spy["sent"] == [("ou_creator", CHAT_OF["transfer"], {"RESULT": TR_JID})]


# ── 4) progress_card_v2：含 job_id + 「查询进度」提示行，schema 2.0 无按钮/表单 ──────

def test_progress_card_v2_has_job_id_and_query_hint():
    from core.transfer import cards
    job = {"job_id": "tr-hint01", "stage": "CROSSING",
           "source": "tos://a/x/", "dest": "oss://b/y/"}
    d = cards.progress_card_v2(job)

    assert d["schema"] == "2.0"
    md = d["body"]["elements"][0]["content"]
    assert "tr-hint01" in md                       # 卡里本就显示 job_id
    assert "查询进度" in md         # #49 新增提示行「发送 查询进度 <job_id>」
    # 提示行本身带上 job_id，用户可直接照抄
    hint_lines = [ln for ln in md.splitlines() if "查询进度" in ln]
    assert hint_lines and "tr-hint01" in hint_lines[0]
    # 仍是纯展示：无 form / button（同家族 2.0→2.0 原地替换，不触 200830）
    assert "form" not in str(d) and "button" not in str(d)


# ── 5) 无 ID 的进度查询仍弹 query_input_card（回归） ─────────────────────────────

def test_no_id_pops_query_input_card(cap):
    messages._handle_progress_query("m1", "查询进度")
    assert not cap["texts"]                         # 不出纯文本
    assert cap["cards"], "无 ID 应弹输入卡"
    dumped = json.dumps(cap["cards"][0], ensure_ascii=False)
    assert "query_progress_by_id" in dumped and "job_id" in dumped


# ── MEDIUM：_push_terminal_result_card 直测（NX 闸门 + created_by 目标 + 重试按钮）──

@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_push_terminal_sends_to_created_by_via_gate(jid, push_spy):
    """闸门未占 → 推对应链 result_card，首参=created_by、次参=该链配置频道。"""
    chain = CHAIN_OF[jid]
    job = {"job_id": jid, "stage": "DONE", "created_by": "ou_creator"}
    messages._push_terminal_result_card(chain, jid, job)

    assert push_spy["claim"]["calls"] == [jid]      # 抢了一次 NX 闸门
    assert push_spy["sent"] == [("ou_creator", CHAT_OF[chain], {"RESULT": jid})]


@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_push_terminal_created_by_missing_falls_back_empty(jid, push_spy):
    """无 created_by → 首参 ""（_send_card 内 target or chat 回落配置频道）。"""
    chain = CHAIN_OF[jid]
    messages._push_terminal_result_card(chain, jid, {"job_id": jid, "stage": "DONE"})
    assert push_spy["sent"] == [("", CHAT_OF[chain], {"RESULT": jid})]


@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_push_terminal_gate_taken_no_double_push(jid, push_spy):
    """NX 闸门已被在线线程/对账占 → _claim 返 False → 不推（去重只推一次）。"""
    push_spy["claim"]["ret"] = False
    chain = CHAIN_OF[jid]
    job = {"job_id": jid, "stage": "DONE", "created_by": "ou_creator"}
    messages._push_terminal_result_card(chain, jid, job)

    assert push_spy["claim"]["calls"] == [jid]      # 抢闸门了
    assert push_spy["sent"] == []                   # 但被吞，不双推


def _failed_job(chain, jid):
    """构造各链 result_card 失败分支所需的完整 job（走真实 result_card）。"""
    base = {"job_id": jid, "stage": "FAILED", "created_by": "ou_creator", "error": "boom"}
    if chain == "transfer":
        base.update(direction="tos→oss", source="tos://a/x/", dest="oss://b/y/")
    elif chain == "cpfs":
        base.update(operation_label="预热", fs_id="bmcpfs-x", directory="/a/b/")
    else:  # vepfs
        base.update(operation_label="预热", fs_id="vepfs-x", sub_path="/a/b/",
                    tos_bucket="bk", tos_prefix="p/")
    return base


@pytest.mark.parametrize("jid", [VEPFS_JID, CPFS_JID, TR_JID])
def test_push_terminal_failed_card_has_retry_button(jid, monkeypatch):
    """FAILED 终态推的富结果卡含对应链的「重试」按钮（走真实 result_card，不桩）。"""
    import core.dsw_scheduler as sched
    from core.feishu_bot import actions
    chain = CHAIN_OF[jid]
    monkeypatch.setattr(sched, "_claim_dataflow_notify", lambda j: True)
    monkeypatch.setattr(actions, "_cfg_vepfs_chat", lambda: "oc_vepfs")
    monkeypatch.setattr(actions, "_cfg_cpfs_chat", lambda: "oc_cpfs")
    monkeypatch.setattr(actions, "_cfg_chat", lambda: "oc_tr")
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append(card))

    messages._push_terminal_result_card(chain, jid, _failed_job(chain, jid))
    assert sent, "FAILED 应推结果卡"
    dumped = json.dumps(sent[0], ensure_ascii=False)
    assert RETRY_ACTION[chain] in dumped        # 失败卡带重试按钮
    assert "重试" in dumped
