"""#51 续 —— 飞书入口：messages 意图/进度查询 + actions 三 handler。

engine 全程 mock（run_to_completion/estimate_source 桩掉）；Redis 用 fakeredis；不连 SSH。
"""
import pytest

from core.feishu_bot import messages, actions


class _SyncThread:
    """同步跑后台线程体，便于断言 handler 后台行为（不真起线程）。"""
    def __init__(self, target=None, daemon=None, **k):
        self._t = target

    def start(self):
        self._t()


# ══════════════════════════════════════════════════════════════════════════════
# messages —— 意图识别
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "数据迁移（泰国H200）",     # 全角括号
    "数据迁移(泰国H200)",       # 半角括号
    "泰国H200",
    "迁到泰国",
    "帮我迁移到泰国服务器",
    "数据迁移 泰国 H200",
])
def test_is_ssh_transfer_intent_hits(text):
    assert messages._is_ssh_transfer_intent(text) is True


@pytest.mark.parametrize("text", [
    "把 tos://a 迁到 oss://b",   # 跨云话术（无泰国）→ 不该被 SSH 链抢
    "迁移一下这个目录",           # 普通迁移无泰国
    "帮我做数据迁移",             # 无泰国/H200
    "查询进度 tr-abc123",
    "",
])
def test_is_ssh_transfer_intent_misses(text):
    assert messages._is_ssh_transfer_intent(text) is False


def test_job_id_re_recognizes_sgp():
    m = messages._JOB_ID_RE.search("查询进度 sgp-abc123def")
    assert m and m.group(1) == "sgp-abc123def"


def test_job_id_re_still_recognizes_others():
    assert messages._JOB_ID_RE.search("tr-aabbcc").group(1) == "tr-aabbcc"
    assert messages._JOB_ID_RE.search("cpfs-112233").group(1) == "cpfs-112233"


def test_job_id_re_too_short_no_match():
    assert messages._JOB_ID_RE.search("sgp-ab") is None   # <6 hex


# ══════════════════════════════════════════════════════════════════════════════
# messages —— _handle_progress_query sgp- 分支
# ══════════════════════════════════════════════════════════════════════════════

def test_progress_query_sgp_routes_to_ssh_orchestrator(monkeypatch):
    """sgp- → ssh_transfer orchestrator.refresh，文本含 progress_line。"""
    from core.feishu_bot import messaging
    from core.ssh_transfer import orchestrator as o

    replies = []
    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, text: replies.append(text))
    job = {"job_id": "sgp-abc123", "stage": o.STAGE_STAGE1,
           "bytes_done": 100, "bytes_total": 200, "speed_bps": 50}
    monkeypatch.setattr(o, "refresh", lambda jid: job)

    messages._handle_progress_query("m1", "查询进度 sgp-abc123", "ou_x")
    assert replies and "sgp-abc123" in replies[0]
    assert "已传" in replies[0]                 # progress_line 内容入文本


def test_progress_query_sgp_terminal_pushes_result_card(monkeypatch):
    """终态查询：文本回执外，经 _claim_dataflow_notify 闸门补推 result_card 到 created_by。"""
    from core.feishu_bot import messaging
    from core.ssh_transfer import orchestrator as o
    from core.ssh_transfer import cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, text: None)
    job = {"job_id": "sgp-aabbcc", "stage": o.STAGE_DONE, "created_by": "ou_creator",
           "bytes_done": 200, "bytes_total": 200, "speed_bps": 0}
    monkeypatch.setattr(o, "refresh", lambda jid: job)

    pushed = []
    monkeypatch.setattr(sched, "_claim_dataflow_notify", lambda jid: True)
    monkeypatch.setattr(sched, "_send_card",
                        lambda oid, chat, card: pushed.append((oid, card)))
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["job_id"]})

    messages._handle_progress_query("m1", "查询进度 sgp-aabbcc", "ou_x")
    assert pushed == [("ou_creator", {"RESULT": "sgp-aabbcc"})]


def test_progress_query_sgp_terminal_gate_blocked_no_push(monkeypatch):
    """终态但闸门被别路径占（claim False）→ 抢闸门但不补推结果卡。"""
    from core.feishu_bot import messaging
    from core.ssh_transfer import orchestrator as o
    from core.ssh_transfer import cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, text: None)
    job = {"job_id": "sgp-ddeeff", "stage": o.STAGE_FAILED, "created_by": "ou_c",
           "error": "boom", "bytes_total": 0}
    monkeypatch.setattr(o, "refresh", lambda jid: job)
    calls, pushed = [], []
    monkeypatch.setattr(sched, "_claim_dataflow_notify",
                        lambda jid: calls.append(jid) or False)
    monkeypatch.setattr(sched, "_send_card", lambda *a: pushed.append(a))
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["job_id"]})

    messages._handle_progress_query("m1", "查询进度 sgp-ddeeff", "ou_x")
    assert calls == ["sgp-ddeeff"]     # 抢了闸门
    assert pushed == []                # 但没抢到 → 不推


def test_progress_query_sgp_not_found(monkeypatch):
    from core.feishu_bot import messaging
    from core.ssh_transfer import orchestrator as o

    replies = []
    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, text: replies.append(text))
    monkeypatch.setattr(o, "refresh", lambda jid: None)
    monkeypatch.setattr(o, "get_job", lambda jid: None)

    messages._handle_progress_query("m1", "查询进度 sgp-fedcba", "ou_x")
    assert replies and "未找到" in replies[0]


# ══════════════════════════════════════════════════════════════════════════════
# actions —— _h_submit_ssh_transfer
# ══════════════════════════════════════════════════════════════════════════════

def test_submit_empty_source_error_toast():
    r = actions._h_submit_ssh_transfer({}, "ou_x", "chat", {"source": "   "})
    assert r["toast"]["type"] == "error"
    assert "源" in r["toast"]["content"]


def test_submit_returns_toast_then_pushes_confirm_card(monkeypatch):
    """3s 内只回 success toast；后台线程解析+估算→推 confirm_card（2.0）。"""
    from core.ssh_transfer import orchestrator
    import core.dsw_scheduler as sched

    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    monkeypatch.setattr(orchestrator, "estimate_source", lambda plan: (123, 4, True))
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b, ok=True: False)
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append((o, card)))
    monkeypatch.setattr(sched, "_send_text", lambda *a: pytest.fail("正常路径不发文本"))

    r = actions._h_submit_ssh_transfer(
        {}, "ou_x", "chat",
        {"source": "oss://wuji-data-tran/ossutil_output/", "dest": "test"})
    assert r["toast"]["type"] == "success"
    assert len(sent) == 1
    oid, card = sent[0]
    assert oid == "ou_x"
    assert card["schema"] == "2.0"        # confirm_card 是 2.0


def test_submit_bad_path_sends_error_text(monkeypatch):
    """后台解析非法路径 → _send_text 错误，不推卡。"""
    from core.ssh_transfer import orchestrator
    import core.dsw_scheduler as sched

    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    monkeypatch.setattr(orchestrator, "estimate_source",
                        lambda plan: pytest.fail("非法路径不该估算"))
    texts = []
    monkeypatch.setattr(sched, "_send_text", lambda o, c, t: texts.append(t))
    monkeypatch.setattr(sched, "_send_card", lambda *a: pytest.fail("非法路径不推卡"))

    r = actions._h_submit_ssh_transfer({}, "ou_x", "chat", {"source": "tos://x/a/"})
    assert r["toast"]["type"] == "success"    # 即时 toast 仍是 async 成功
    assert texts and "路径" in texts[0]


# ══════════════════════════════════════════════════════════════════════════════
# actions —— _h_confirm_ssh_transfer
# ══════════════════════════════════════════════════════════════════════════════

def _job(stage="NEW", **over):
    j = {"job_id": "sgp-abc123", "stage": stage, "bytes_total": 1, "estimate_ok": True,
         "source_uri": "oss://wuji-data-tran/ossutil_output/",
         "source_prefix": "ossutil_output/", "source_bucket": "wuji-data-tran",
         "dest_uri": "wuji@host:/root/test/", "dest_rel": "test/", "created_by": ""}
    j.update(over)
    return j


@pytest.fixture
def C(monkeypatch):
    from core.ssh_transfer import orchestrator
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b, ok=True: False)
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    return actions, orchestrator


def _store_get_save(monkeypatch, orch, job):
    store = {job["job_id"]: job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    return store


def test_confirm_launches_once_returns_progress_v2(C, monkeypatch):
    """确认默认回 progress_card_v2（2.0）；NX 锁 ex=120 已置。"""
    actions, orch = C
    store = _store_get_save(monkeypatch, orch, _job())

    r1 = actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {})
    assert r1["toast"]["type"] == "success"
    assert r1["card"]["type"] == "raw"
    assert r1["card"]["data"]["schema"] == "2.0"       # 同家族替换避 200830
    assert "button" not in str(r1["card"]["data"])
    # NX launch 锁存在，TTL 约 120
    from utils.redis_client import get_redis
    r = get_redis()
    assert r.get("ssh:transfer:launch:sgp-abc123") == "1"
    assert 0 < r.ttl("ssh:transfer:launch:sgp-abc123") <= 120


def test_confirm_second_click_blocked_by_nx_lock(C, monkeypatch):
    """连点第二次：NX 锁已占 → 回 info「已下发」，不重复下发。"""
    actions, orch = C
    launched = []
    monkeypatch.setattr(orch, "run_to_completion",
                        lambda job, **k: launched.append(job["job_id"]) or job)
    _store_get_save(monkeypatch, orch, _job())

    actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {})
    r2 = actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {})
    assert r2["toast"]["type"] == "info"
    assert "已下发" in r2["toast"]["content"]
    assert launched == ["sgp-abc123"]                  # 只真下发一次


def test_confirm_approval_blocks_non_admin(C, monkeypatch):
    """超阈值 + 非管理员 → error toast；launch 锁不被占（admin 仍可下发）。"""
    actions, orch = C
    monkeypatch.setattr(orch, "needs_approval", lambda b, ok=True: True)
    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    _store_get_save(monkeypatch, orch, _job(bytes_total=10 ** 15))

    r = actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_not_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "审批" in r["toast"]["content"]
    assert "card" not in r
    from utils.redis_client import get_redis
    assert get_redis().get("ssh:transfer:launch:sgp-abc123") is None


def test_confirm_backfills_empty_created_by(C, monkeypatch):
    """created_by 空 → 用确认者 open_id 回填并落库。"""
    actions, orch = C
    store = _store_get_save(monkeypatch, orch, _job(created_by=""))
    actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_me", "chat", {})
    assert store["sgp-abc123"]["created_by"] == "ou_me"


def test_confirm_does_not_overwrite_created_by(C, monkeypatch):
    actions, orch = C
    store = _store_get_save(monkeypatch, orch, _job(created_by="ou_owner"))
    actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_other", "chat", {})
    assert store["sgp-abc123"]["created_by"] == "ou_owner"


def test_confirm_reply_v2_false_toast_only(C, monkeypatch):
    """reply_v2=False（retry 触发）→ 只回 toast，无 card（避 1.0/2.0 混触 200830）。"""
    actions, orch = C
    _store_get_save(monkeypatch, orch, _job())
    r = actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {},
                                        reply_v2=False)
    assert r["toast"]["type"] == "success"
    assert "card" not in r


def test_confirm_missing_job_id_error(C):
    actions, orch = C
    r = actions._h_confirm_ssh_transfer({}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "card" not in r


def test_confirm_job_not_found_error(C, monkeypatch):
    actions, orch = C
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    r = actions._h_confirm_ssh_transfer({"job_id": "sgp-x"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "card" not in r


def test_confirm_stage_not_new_or_failed_info(C, monkeypatch):
    actions, orch = C
    _store_get_save(monkeypatch, orch, _job(stage="STAGE1"))
    r = actions._h_confirm_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "card" not in r


# ══════════════════════════════════════════════════════════════════════════════
# actions —— _h_retry_ssh_transfer
# ══════════════════════════════════════════════════════════════════════════════

def test_retry_clears_keys_resets_stage_keeps_stage1_rc(C, monkeypatch):
    """retry：清 launch+notified 键、stage=NEW、保留 stage1_rc、转 confirm reply_v2=False。"""
    actions, orch = C
    store = _store_get_save(monkeypatch, orch,
                            _job(stage="FAILED", stage1_rc=0, launched=True, error="boom"))
    from utils.redis_client import get_redis
    r = get_redis()
    r.set("dataflow:notified:sgp-abc123", 1)

    resp = actions._h_retry_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {})
    # 转发到 confirm(reply_v2=False) → 只回 toast、无 card
    assert resp["toast"]["type"] == "success"
    assert "card" not in resp
    # stage1_rc 保留（段1已成功，只重段2）
    assert store["sgp-abc123"]["stage1_rc"] == 0
    # notified 键被清（不再被 30 天旧标记吞掉完成通知）
    assert r.get("dataflow:notified:sgp-abc123") is None


def test_retry_missing_job_error(C, monkeypatch):
    actions, orch = C
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    resp = actions._h_retry_ssh_transfer({"job_id": "sgp-x"}, "ou_admin", "chat", {})
    assert resp["toast"]["type"] == "error"


# ══════════════════════════════════════════════════════════════════════════════
# actions —— handler 注册
# ══════════════════════════════════════════════════════════════════════════════

def test_handlers_registered():
    for name in ("submit_ssh_transfer", "confirm_ssh_transfer", "retry_ssh_transfer"):
        assert name in actions._ACTION_HANDLERS
