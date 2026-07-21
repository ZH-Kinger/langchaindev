"""PFS 跨云直传 —— 飞书入口：messages 意图/排序/进度查询 + actions 三 handler。

子链全程 mock（run_to_completion 桩）；Redis 用 fakeredis；不连云。
"""
import json

import pytest

from core.feishu_bot import messages, actions
from config.settings import settings

VEPFS_FS = "vepfs-abc"
CPFS_FS = "cpfs-xyz"
_MAP = {
    f"vepfs://{VEPFS_FS}": {"region": "cn-beijing", "tos_bucket": "wuji-dc-bj", "tos_prefix": "pfs-staging/"},
    f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "wuji-il-hz",
                          "oss_prefix": "pfs-staging/", "dataflow_id": "df-123"},
}


@pytest.fixture(autouse=True)
def _map(monkeypatch):
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(_MAP))


class _SyncThread:
    def __init__(self, target=None, args=(), daemon=None, **k):
        self._t, self._a = target, args

    def start(self):
        if self._t:
            self._t(*self._a)


class _NoopThread:
    """start() 什么都不做（用于隔离 _process_message 里的 _auto_map_user 后台线程）。"""
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# messages —— 意图识别
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "把 vepfs 数据迁到 cpfs",       # 同时提两种 PFS
    "vepfs 迁 cpfs",
    "PFS直传",
    "跨云PFS搬运",
    "帮我做 pfs 互传",
])
def test_is_pfs_transfer_intent_hits(text):
    assert messages._is_pfs_transfer_intent(text) is True


@pytest.mark.parametrize("text", [
    "vepfs沉降",                    # 只提一个 PFS → 单链预热/沉降，不是直传
    "cpfs预热",
    "把 tos://a 迁到 oss://b",      # 普通跨云对象迁移
    "帮我做数据迁移",
    "查询进度 tr-abc123",
    "",
])
def test_is_pfs_transfer_intent_misses(text):
    assert messages._is_pfs_transfer_intent(text) is False


def test_job_id_re_recognizes_xpfs():
    m = messages._JOB_ID_RE.search("查询进度 xpfs-abc123def")
    assert m and m.group(1) == "xpfs-abc123def"


def test_job_id_re_xpfs_too_short():
    assert messages._JOB_ID_RE.search("xpfs-ab") is None


# ══════════════════════════════════════════════════════════════════════════════
# messages —— 排序（pfs 在 ssh/transfer 之前；关时不路由）
# ══════════════════════════════════════════════════════════════════════════════

def test_routing_pfs_before_ssh_when_enabled(monkeypatch):
    """启用 + 命中 pfs 意图 → 推 pfs entry_card，先于 ssh 分支 return（ssh 探针不被触及）。"""
    from core.feishu_bot import messaging
    monkeypatch.setattr(messages.settings, "PFS_TRANSFER_ENABLED", True)
    monkeypatch.setattr(messages.threading, "Thread", _NoopThread)
    cards = []
    monkeypatch.setattr(messaging, "_feishu_reply_card", lambda mid, c: cards.append(c))
    # ssh 意图探针：若流程越过 pfs 分支到 ssh 就炸（证明未越过）
    monkeypatch.setattr(messages, "_is_ssh_transfer_intent",
                        lambda t: (_ for _ in ()).throw(RuntimeError("reached-ssh")))

    messages._process_message("m", "c", "把 vepfs 数据迁到 cpfs", "ou")
    assert len(cards) == 1
    assert cards[0]["schema"] == "2.0"
    assert "PFS" in json.dumps(cards[0], ensure_ascii=False)


def test_routing_pfs_disabled_falls_through(monkeypatch):
    """关闭 → pfs 分支跳过，流程落到 ssh 分支（探针被触及）→ 不推 pfs 卡。"""
    from core.feishu_bot import messaging
    monkeypatch.setattr(messages.settings, "PFS_TRANSFER_ENABLED", False)
    monkeypatch.setattr(messages.threading, "Thread", _NoopThread)
    cards = []
    monkeypatch.setattr(messaging, "_feishu_reply_card", lambda mid, c: cards.append(c))
    monkeypatch.setattr(messages, "_is_ssh_transfer_intent",
                        lambda t: (_ for _ in ()).throw(RuntimeError("reached-ssh")))

    with pytest.raises(RuntimeError, match="reached-ssh"):
        messages._process_message("m", "c", "把 vepfs 数据迁到 cpfs", "ou")
    assert cards == []          # pfs 卡未推


# ══════════════════════════════════════════════════════════════════════════════
# messages —— _handle_progress_query xpfs- 分支
# ══════════════════════════════════════════════════════════════════════════════

def test_progress_query_xpfs_routes_to_pfs(monkeypatch):
    from core.feishu_bot import messaging
    from core.pfs_transfer import orchestrator as o

    replies = []
    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, t: replies.append(t))
    job = {"job_id": "xpfs-abc123", "stage": o.STAGE_CROSSING, "direction": "vepfs2cpfs",
           "sink_done": True, "cross_done": False, "preheat_done": False,
           "bytes_done": 10, "bytes_total": 100}
    monkeypatch.setattr(o, "refresh", lambda jid: job)

    messages._handle_progress_query("m", "查询进度 xpfs-abc123", "ou_x")
    assert replies and "xpfs-abc123" in replies[0]
    assert "沉降=True" in replies[0]      # 段完成状态入文本


def test_progress_query_xpfs_terminal_pushes_result_card(monkeypatch):
    from core.feishu_bot import messaging
    from core.pfs_transfer import orchestrator as o
    from core.pfs_transfer import cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, t: None)
    job = {"job_id": "xpfs-abcdef", "stage": o.STAGE_DONE, "created_by": "ou_creator",
           "direction": "vepfs2cpfs", "sink_done": True, "cross_done": True,
           "preheat_done": True, "bytes_done": 100, "bytes_total": 100}
    monkeypatch.setattr(o, "refresh", lambda jid: job)
    pushed = []
    monkeypatch.setattr(sched, "_claim_dataflow_notify", lambda jid: True)
    monkeypatch.setattr(sched, "_send_card", lambda oid, chat, c: pushed.append((oid, c)))
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["job_id"]})

    messages._handle_progress_query("m", "查询进度 xpfs-abcdef", "ou_x")
    assert pushed == [("ou_creator", {"RESULT": "xpfs-abcdef"})]


def test_progress_query_xpfs_gate_blocked_no_push(monkeypatch):
    from core.feishu_bot import messaging
    from core.pfs_transfer import orchestrator as o
    from core.pfs_transfer import cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, t: None)
    job = {"job_id": "xpfs-a1b2c3", "stage": o.STAGE_FAILED, "created_by": "ou_c",
           "direction": "vepfs2cpfs", "error": "boom", "bytes_total": 0}
    monkeypatch.setattr(o, "refresh", lambda jid: job)
    calls, pushed = [], []
    monkeypatch.setattr(sched, "_claim_dataflow_notify", lambda jid: calls.append(jid) or False)
    monkeypatch.setattr(sched, "_send_card", lambda *a: pushed.append(a))
    monkeypatch.setattr(cards, "result_card", lambda j: {"R": j["job_id"]})

    messages._handle_progress_query("m", "查询进度 xpfs-a1b2c3", "ou_x")
    assert calls == ["xpfs-a1b2c3"]      # 抢闸门
    assert pushed == []                 # 没抢到 → 不推


def test_progress_query_xpfs_not_found(monkeypatch):
    from core.feishu_bot import messaging
    from core.pfs_transfer import orchestrator as o
    replies = []
    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, t: replies.append(t))
    monkeypatch.setattr(o, "refresh", lambda jid: None)
    monkeypatch.setattr(o, "get_job", lambda jid: None)
    messages._handle_progress_query("m", "查询进度 xpfs-ffeedd", "ou_x")
    assert replies and "未找到" in replies[0]


# ══════════════════════════════════════════════════════════════════════════════
# actions —— _h_submit_pfs_transfer
# ══════════════════════════════════════════════════════════════════════════════

def test_submit_empty_addr_error_toast():
    r = actions._h_submit_pfs_transfer({}, "ou_x", "chat", {"source": "  ", "dest": ""})
    assert r["toast"]["type"] == "error"


def test_submit_pushes_confirm_card(monkeypatch):
    import core.dsw_scheduler as sched
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda oid, chat, c: sent.append((oid, c)))
    monkeypatch.setattr(sched, "_send_text", lambda *a: pytest.fail("正常路径不发文本"))

    r = actions._h_submit_pfs_transfer(
        {}, "ou_x", "chat",
        {"source": f"vepfs://{VEPFS_FS}/wzh/", "dest": f"cpfs://{CPFS_FS}/team/"})
    assert r["toast"]["type"] == "success"
    assert len(sent) == 1 and sent[0][0] == "ou_x"
    assert sent[0][1]["schema"] == "2.0"          # confirm_card 2.0


def test_submit_bad_path_sends_error_text(monkeypatch):
    import core.dsw_scheduler as sched
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    texts = []
    monkeypatch.setattr(sched, "_send_text", lambda oid, chat, t: texts.append(t))
    monkeypatch.setattr(sched, "_send_card", lambda *a: pytest.fail("非法路径不推卡"))

    actions._h_submit_pfs_transfer(
        {}, "ou_x", "chat", {"source": "oss://x/a/", "dest": f"cpfs://{CPFS_FS}/b/"})
    assert texts and ("路径" in texts[0] or "对象存储" in texts[0])


# ══════════════════════════════════════════════════════════════════════════════
# actions —— _h_confirm_pfs_transfer
# ══════════════════════════════════════════════════════════════════════════════

def _pfs_job(stage="NEW", **over):
    j = {"job_id": "xpfs-abc123", "chain": "pfs", "direction": "vepfs2cpfs", "stage": stage,
         "created_by": "", "approx_bytes": 0, "size_known": False,
         "bytes_done": 0, "bytes_total": 0,
         "src_pfs": {"scheme": "vepfs", "fs_id": "vepfs-a", "sub_path": "wzh/", "region": "cn-beijing"},
         "dst_pfs": {"scheme": "cpfs", "fs_id": "cpfs-b", "sub_path": "team/", "region": "cn-hangzhou"},
         "src_staging": {"scheme": "tos", "bucket": "b1", "prefix": "p/xpfs-abc123/",
                         "region": "cn-beijing", "dataflow_id": ""},
         "dst_staging": {"scheme": "oss", "bucket": "b2", "prefix": "p/xpfs-abc123/",
                         "region": "cn-hangzhou", "dataflow_id": "df-1"},
         "sink_done": False, "cross_done": False, "preheat_done": False,
         "sink_job_id": "", "cross_job_id": "", "preheat_job_id": "", "launched": False}
    j.update(over)
    return j


@pytest.fixture
def C(monkeypatch):
    from core.pfs_transfer import orchestrator
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    # MED-2 收紧：确认门禁改成"非 admin 一律拦"，与自报量无关。管理员 = ou_admin。
    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    return actions, orchestrator


def _store(monkeypatch, orch, job):
    store = {job["job_id"]: job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    return store


def test_confirm_launches_returns_progress_v2(C, monkeypatch):
    actions, orch = C
    _store(monkeypatch, orch, _pfs_job())
    r = actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success"
    assert r["card"]["data"]["schema"] == "2.0"        # progress_card_v2 同家族避 200830
    assert "button" not in str(r["card"]["data"])
    from utils.redis_client import get_redis
    rc = get_redis()
    assert rc.get("pfs:transfer:launch:xpfs-abc123") == "1"
    assert 0 < rc.ttl("pfs:transfer:launch:xpfs-abc123") <= 120


def test_confirm_second_click_blocked(C, monkeypatch):
    actions, orch = C
    launched = []
    monkeypatch.setattr(orch, "run_to_completion",
                        lambda job, **k: launched.append(job["job_id"]) or job)
    _store(monkeypatch, orch, _pfs_job())
    actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    r2 = actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert r2["toast"]["type"] == "info"
    assert launched == ["xpfs-abc123"]                 # 只真下发一次


def test_confirm_non_admin_always_blocked(C, monkeypatch):
    """MED-2 锁死：非管理员一律拦——即便自报小值（size_known 小）也拦、不下发、不起线程、不占锁。"""
    actions, orch = C
    launched = []
    monkeypatch.setattr(orch, "run_to_completion",
                        lambda job, **k: launched.append(1) or job)
    # 自报一个很小的量（历史绕过手法）——现在也必须被拦
    _store(monkeypatch, orch, _pfs_job(size_known=True, approx_bytes=1024))
    r = actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_not_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "管理员" in r["toast"]["content"]
    assert "card" not in r
    assert launched == []                                   # 未起链
    from utils.redis_client import get_redis
    assert get_redis().get("pfs:transfer:launch:xpfs-abc123") is None   # 锁未占


def test_confirm_admin_passes(C, monkeypatch):
    actions, orch = C
    _store(monkeypatch, orch, _pfs_job())
    r = actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success" and "card" in r


def test_confirm_backfills_created_by(C, monkeypatch):
    actions, orch = C
    store = _store(monkeypatch, orch, _pfs_job(created_by=""))
    actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert store["xpfs-abc123"]["created_by"] == "ou_admin"   # 确认者（管理员）回填


def test_confirm_does_not_overwrite_created_by(C, monkeypatch):
    """管理员确认，但 job 已有 created_by（原发起人）→ 不覆盖。"""
    actions, orch = C
    store = _store(monkeypatch, orch, _pfs_job(created_by="ou_owner"))
    actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert store["xpfs-abc123"]["created_by"] == "ou_owner"


def test_confirm_reply_v2_false_toast_only(C, monkeypatch):
    actions, orch = C
    _store(monkeypatch, orch, _pfs_job())
    r = actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {},
                                        reply_v2=False)
    assert r["toast"]["type"] == "success" and "card" not in r


def test_confirm_missing_job_id(C):
    actions, orch = C
    r = actions._h_confirm_pfs_transfer({}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "card" not in r


def test_confirm_job_not_found(C, monkeypatch):
    actions, orch = C
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    r = actions._h_confirm_pfs_transfer({"job_id": "xpfs-x"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "card" not in r


def test_confirm_stage_not_new_or_failed(C, monkeypatch):
    actions, orch = C
    _store(monkeypatch, orch, _pfs_job(stage="CROSSING"))
    r = actions._h_confirm_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "card" not in r


# ══════════════════════════════════════════════════════════════════════════════
# actions —— _h_retry_pfs_transfer
# ══════════════════════════════════════════════════════════════════════════════

def test_retry_clears_keys_keeps_stage_markers(C, monkeypatch):
    """retry：清 launch+notified 键、stage=NEW、保留 sink_done/cross_done、转 confirm reply_v2=False。"""
    actions, orch = C
    store = _store(monkeypatch, orch,
                   _pfs_job(stage="FAILED", sink_done=True, cross_done=True,
                            launched=True, error="boom"))
    from utils.redis_client import get_redis
    rc = get_redis()
    rc.set("dataflow:notified:xpfs-abc123", 1)
    rc.set("pfs:transfer:launch:xpfs-abc123", 1)

    resp = actions._h_retry_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_admin", "chat", {})
    assert resp["toast"]["type"] == "success" and "card" not in resp   # reply_v2=False
    j = store["xpfs-abc123"]
    assert j["sink_done"] is True and j["cross_done"] is True           # 段完成标记保留
    assert j["error"] == ""
    assert rc.get("dataflow:notified:xpfs-abc123") is None              # notified 键被清


def test_retry_missing_job(C, monkeypatch):
    actions, orch = C
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    r = actions._h_retry_pfs_transfer({"job_id": "xpfs-x"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error"


def test_retry_non_admin_blocked_by_confirm_gate(C, monkeypatch):
    """retry 转发给 confirm → 非管理员同样被审批门拦（MED-2）。"""
    actions, orch = C
    launched = []
    monkeypatch.setattr(orch, "run_to_completion", lambda job, **k: launched.append(1) or job)
    _store(monkeypatch, orch, _pfs_job(stage="FAILED", sink_done=True))
    r = actions._h_retry_pfs_transfer({"job_id": "xpfs-abc123"}, "ou_not_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "管理员" in r["toast"]["content"]
    assert launched == []


# ══════════════════════════════════════════════════════════════════════════════
# handler 注册
# ══════════════════════════════════════════════════════════════════════════════

def test_handlers_registered():
    for name in ("submit_pfs_transfer", "confirm_pfs_transfer", "retry_pfs_transfer"):
        assert name in actions._ACTION_HANDLERS
