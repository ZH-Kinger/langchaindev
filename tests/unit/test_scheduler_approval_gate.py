"""`core.dsw_scheduler._process_new_ticket` 的审批门禁回归。

安全整改批次 1：`ADMIN_FEISHU_OPEN_ID` 未配置时，原实现是**自动批准**（fail-open）——
只要管理员 open_id 漏配（新部署 / 隔离热备机 / `.env` 改了却用 `restart` 而非
force-recreate），任何人提个 Jira 工单就能让 bot 直接建出 GPU 实例，无人审批、真金白银。
现改为 fail-closed：拒绝 + Jira 评论提示 + return。

本文件钉住两件事：
  1. **不自动批准**：`_set_approved` / `manage_pai_dsw` 一次都不能被调到（危险动作没执行）。
  2. **不刷评论**：该分支既不 `_set_approved` 也不写 `dsw:ticket:{key}`，工单会一直停在 Jira
     「待办」被每轮捞回来，而 `TICKET_POLL_INTERVAL` 只有 20 秒 → 不过
     `_mark_approval_notified` NX 闸门就是 180 条评论/小时、4320 条/天。

以及对照组：`ADMIN_FEISHU_OPEN_ID` **有值**时的兄弟分支（发审批卡）行为不得改变；
不需审批 / 已批准的工单照常建实例（fail-closed 不能把正常路堵死）。
"""
import json

import pytest

# **必须在收集期（模块顶层）导入**，不能挪进 fixture/用例里：`core.dsw_scheduler` 用的是
# `from utils.redis_client import get_redis`（导入即绑定）。若首次导入发生在某个用例的
# fixture 之后，绑定到的就是那一个用例的 fakeredis lambda，之后所有用例都在往一个早已废弃的
# 假 Redis 里写 —— 断言会莫名其妙地空。收集期导入才拿到真函数、每个用例各自读到自己的 fake。
import core.dsw_scheduler as _sched  # noqa: F401

TICKET_KEY = "GPU-2001"
ADMIN = "ou_admin_real"


def _ticket(key=TICKET_KEY, *, needs_approval=True, open_id="", extra=""):
    desc = "\n".join(filter(None, [
        f"dsw_instance_name=dsw-{key.lower()}",
        "dsw_gpu_count=8",
        "dsw_duration_hours=24",
        "dsw_purpose=大规模预训练",
        "dsw_requester_name=张三",
        f"dsw_needs_approval={'true' if needs_approval else 'false'}",
        f"feishu_open_id={open_id}" if open_id else "",
        extra,
    ]))
    return {"key": key, "fields": {"description": desc}}


@pytest.fixture
def sched(monkeypatch):
    """把 `_process_new_ticket` 会碰到的所有出口装上哨兵。

    关键：`_set_approved` 的哨兵**仍执行原逻辑**（不是空壳）。否则在未修版上
    `_is_approved` 会因为哨兵不写 Redis 而返回 False → 提前 return → 「没建实例」的断言
    会假绿。包一层记账再转调真身，未修版才会真的一路走到 `manage_pai_dsw`。
    """
    s = _sched

    calls = {
        "comments": [], "set_approved": [], "create": [],
        "cards": [], "texts": [], "transitions": [], "poll": [],
    }

    real_set_approved = s._set_approved

    def _spy_set_approved(key):
        calls["set_approved"].append(key)
        return real_set_approved(key)

    monkeypatch.setattr(s, "_set_approved", _spy_set_approved)
    monkeypatch.setattr(s, "add_comment", lambda key, text: calls["comments"].append((key, text)))
    monkeypatch.setattr(s, "transition_ticket", lambda *a, **k: calls["transitions"].append((a, k)))
    monkeypatch.setattr(s, "_send_card", lambda oid, cid, card: calls["cards"].append((oid, cid, card)))
    monkeypatch.setattr(s, "_send_text", lambda oid, cid, text: calls["texts"].append((oid, cid, text)))
    monkeypatch.setattr(s, "_poll_until_running", lambda *a, **k: calls["poll"].append(a))

    def _fake_create(**kwargs):
        calls["create"].append(kwargs)
        return "✅ 实例创建成功\n实例 ID：dsw-fake-0001"

    monkeypatch.setattr(s, "manage_pai_dsw", _fake_create)
    monkeypatch.setattr(s.settings, "PAI_DSW_DEFAULT_IMAGE", "registry/test:latest", raising=False)

    calls["mod"] = s
    return calls


# ── 1. 主用例：ADMIN 未配置 → 拒绝 + 只评论一次 ────────────────────────────────

def test_no_admin_configured_refuses_and_comments_exactly_once(sched, monkeypatch):
    """连续两轮（模拟 20s 一次的轮询）→ 评论恰 1 条、绝不自动批准、绝不建实例。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    s._process_new_ticket(_ticket())
    s._process_new_ticket(_ticket())

    # 危险动作一次都没执行
    assert sched["set_approved"] == [], "ADMIN 未配置时绝不能自动批准工单"
    assert sched["create"] == [], "没有审批人 → 一个 DSW 实例都不能建"
    assert not s._is_approved(TICKET_KEY)
    assert sched["transitions"] == []
    assert sched["poll"] == []

    # 评论恰一条（NX 闸门生效），且是「未配置管理员」那条
    assert len(sched["comments"]) == 1, f"重复评论：{sched['comments']}"
    key, text = sched["comments"][0]
    assert key == TICKET_KEY
    assert "未配置审批管理员" in text


def test_no_admin_configured_stays_quiet_over_many_polls(sched, monkeypatch):
    """20 轮轮询（= 真实节奏下的约 7 分钟）仍只有 1 条评论、0 次建实例。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    for _ in range(20):
        s._process_new_ticket(_ticket())

    assert len(sched["comments"]) == 1
    assert sched["create"] == []
    assert sched["set_approved"] == []


def test_no_admin_gate_key_is_the_shared_approval_notified_key(sched, monkeypatch, fake_redis):
    """去重用的就是兄弟分支同一把 NX 闸门键（`dsw:approval_notified:{key}`），带 TTL。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    s._process_new_ticket(_ticket())

    gate_key = s._APPROVAL_NOTIFIED_PREFIX + TICKET_KEY
    assert fake_redis.exists(gate_key)
    ttl = fake_redis.ttl(gate_key)
    assert 0 < ttl <= 7 * 86400


def test_no_admin_does_not_write_ticket_state(sched, monkeypatch, fake_redis):
    """本分支刻意不写 `dsw:ticket:{key}` —— 记账用，防止有人误以为它靠状态记录去重。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    s._process_new_ticket(_ticket())

    assert s._redis_get(TICKET_KEY) is None


def test_no_admin_refusal_holds_for_registered_requester(sched, monkeypatch):
    """申请人有 open_id 且已完成 RAM 映射时同样拒绝（不是靠前面的映射门挡住的）。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")
    import core.feishu_bot as fb
    monkeypatch.setattr(fb, "_is_registered", lambda oid: True, raising=False)

    s._process_new_ticket(_ticket(open_id="ou_requester"))
    s._process_new_ticket(_ticket(open_id="ou_requester"))

    assert sched["create"] == []
    assert sched["set_approved"] == []
    assert len(sched["comments"]) == 1


def test_no_admin_refusal_is_per_ticket_not_global(sched, monkeypatch):
    """闸门按工单 key 隔离：两张不同工单各得一条提示，不会互相吞掉。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    s._process_new_ticket(_ticket("GPU-3001"))
    s._process_new_ticket(_ticket("GPU-3002"))
    s._process_new_ticket(_ticket("GPU-3001"))
    s._process_new_ticket(_ticket("GPU-3002"))

    assert sorted(k for k, _ in sched["comments"]) == ["GPU-3001", "GPU-3002"]
    assert sched["create"] == []


# ── 2. Redis 不可用：闸门 fail-open 退化成「每轮一条」，但绝不 fail-open 到自动批准 ──

def test_redis_down_degrades_to_one_comment_per_poll_but_never_approves(sched, monkeypatch):
    """`_mark_approval_notified` 在 Redis 挂掉时返回 True（宁可重复也别漏报）。

    这是刻意的降级：评论会变多，但**审批门本身不得跟着 fail-open**。
    """
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    def _boom():
        raise RuntimeError("redis down")

    monkeypatch.setattr(s, "get_redis", _boom)

    s._process_new_ticket(_ticket())
    s._process_new_ticket(_ticket())

    assert len(sched["comments"]) == 2          # 降级：每轮一条
    assert sched["create"] == []                # 但绝不建实例
    assert sched["set_approved"] == []          # 也绝不自动批准


# ── 3. 对照组：ADMIN 有值 → 兄弟分支（发审批卡）行为不变 ───────────────────────

def test_admin_configured_sends_approval_card_once(sched, monkeypatch):
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", ADMIN)

    s._process_new_ticket(_ticket())
    s._process_new_ticket(_ticket())

    assert len(sched["cards"]) == 1, "审批卡也只发一次"
    target_open_id, target_chat, card = sched["cards"][0]
    assert target_open_id == ADMIN
    assert "待审批" in json.dumps(card, ensure_ascii=False)

    assert len(sched["comments"]) == 1
    assert "已发送审批通知" in sched["comments"][0][1]

    # 等审批期间同样不许建实例 / 不许自动批准
    assert sched["create"] == []
    assert sched["set_approved"] == []


def test_admin_configured_card_carries_approve_reject_actions(sched, monkeypatch):
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", ADMIN)

    s._process_new_ticket(_ticket())

    blob = json.dumps(sched["cards"][0][2], ensure_ascii=False)
    assert "approve_gpu" in blob and "reject_gpu" in blob
    assert TICKET_KEY in blob


# ── 4. 正常路径没被 fail-closed 堵死 ─────────────────────────────────────────

def test_ticket_not_needing_approval_still_creates_without_admin(sched, monkeypatch):
    """小规模工单（`dsw_needs_approval` 非 true）与管理员配置无关，照常建实例。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")

    s._process_new_ticket(_ticket(needs_approval=False))

    assert len(sched["create"]) == 1
    assert sched["comments"], "创建成功应有 Jira 评论"


def test_already_approved_ticket_creates_even_without_admin(sched, monkeypatch):
    """管理员点过批准（`_set_approved` 已落）的工单不受本门影响 —— 门只管「尚未批准」。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")
    s._set_approved(TICKET_KEY)
    sched["set_approved"].clear()

    s._process_new_ticket(_ticket())

    assert len(sched["create"]) == 1


def test_existing_ticket_state_short_circuits_before_gate(sched, monkeypatch):
    """已在跟踪中的工单（`dsw:ticket:{key}` 有值）直接返回，不评论也不建实例。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")
    s._redis_set(TICKET_KEY, {"instance_id": "dsw-old"})

    s._process_new_ticket(_ticket())

    assert sched["comments"] == []
    assert sched["create"] == []


def test_unregistered_requester_blocked_before_approval_gate(sched, monkeypatch):
    """无 RAM 映射的申请人在更早一道门被挡下，不会走到审批门（顺序不变）。"""
    s = sched["mod"]
    monkeypatch.setattr(s.settings, "ADMIN_FEISHU_OPEN_ID", "")
    import core.feishu_bot as fb
    monkeypatch.setattr(fb, "_is_registered", lambda oid: False, raising=False)

    s._process_new_ticket(_ticket(open_id="ou_requester"))

    assert sched["create"] == []
    assert len(sched["comments"]) == 1
    assert "映射" in sched["comments"][0][1]
