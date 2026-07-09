"""#40 屏蔽 DSW 到期自动关机（settings.DSW_IDLE_STOP_ENABLED，默认 False）。

语义：
  flag=False（默认）→ `_check_running_instances` **不**发「即将到期」警告卡、
    即便 warned+超时也**不调** `manage_pai_dsw(action="stop")`；但 GPU 空转提醒照发。
  flag=True → 恢复旧行为：到期前 15min 发警告卡；warned 后超时自动 stop。

GPU 空转段有 `if not warned` 前门 —— flag 关时该函数永不置 warned，故空转段照跑；
两个场景（未警告→将到期→空转 / 已警告→超时→本应 stop）分开构造，避免 warned 前门互斥。
"""
import time

import pytest


@pytest.fixture
def sched(monkeypatch):
    """隔离 _check_running_instances 的所有外部依赖，返回 (模块, 记录器 dict)。"""
    from core import dsw_scheduler as s

    rec = {
        "stop": [],       # manage_pai_dsw 调用（action, instance_id）
        "cards": [],      # _send_card 调用（card dict）
        "texts": [],      # _send_text 调用（text 内容）
        "saved": [],      # _redis_set
        "deleted": [],    # _redis_delete
        "transition": [], # transition_ticket
        "comment": [],    # add_comment
    }

    monkeypatch.setattr(s, "manage_pai_dsw",
                        lambda **k: rec["stop"].append(k) or "stopped-ok")
    monkeypatch.setattr(s, "_send_card",
                        lambda o, c, card: rec["cards"].append(card))
    monkeypatch.setattr(s, "_send_text",
                        lambda o, c, text: rec["texts"].append(text))
    monkeypatch.setattr(s, "_redis_set",
                        lambda k, d: rec["saved"].append((k, dict(d))))
    monkeypatch.setattr(s, "_redis_delete",
                        lambda k: rec["deleted"].append(k))
    monkeypatch.setattr(s, "transition_ticket",
                        lambda k, st: rec["transition"].append((k, st)))
    monkeypatch.setattr(s, "add_comment",
                        lambda k, c: rec["comment"].append((k, c)))
    monkeypatch.setattr(s, "_make_idle_warn_card",
                        lambda *a, **k: {"IDLE_WARN": True})
    # 警告卡以 IDLE_WARN 标记区分于其它卡
    return s, rec


def _install_state(s, monkeypatch, state):
    monkeypatch.setattr(s, "_all_tracked_keys", lambda: ["TK-1"])
    monkeypatch.setattr(s, "_redis_get", lambda k: dict(state))


def _set_flag(s, monkeypatch, value):
    monkeypatch.setattr(s.settings, "DSW_IDLE_STOP_ENABLED", value)


def _base_state(now, **over):
    """未警告、还有约 8min 到期（remaining<=15min）、GPU 空转已够 warn 时长。"""
    st = {
        "instance_id": "dsw-abc",
        "instance_name": "inst-abc",
        "open_id": "ou_user",
        "chat_id": "oc_chat",
        "start_ts": now - 8 * 3600 + 500,   # remaining ≈ 500s <= 900s
        "duration_hours": 8,
        "warned": False,
        "warn_ts": 0,
        "gpu_count": 2,
        # GPU 空转：低于阈值、已持续超过 GPU_IDLE_WARN_MINUTES
        "idle_since": now - (30 * 60 + 100),
        "idle_warned": False,
    }
    st.update(over)
    return st


def _warned_timeout_state(now, **over):
    """已警告、warn_ts 超过 stop_seconds（本应自动 stop）。"""
    st = {
        "instance_id": "dsw-xyz",
        "instance_name": "inst-xyz",
        "open_id": "ou_user",
        "chat_id": "oc_chat",
        "start_ts": now - 9 * 3600,
        "duration_hours": 8,
        "warned": True,
        "warn_ts": now - (30 * 60 + 120),   # 超过 DSW_IDLE_STOP_MINUTES(30) * 60
        "gpu_count": 2,
    }
    st.update(over)
    return st


# ── 默认值 ────────────────────────────────────────────────────────────────────

def test_default_flag_is_false():
    from config.settings import settings
    assert settings.DSW_IDLE_STOP_ENABLED is False


# ── flag=False（默认）：不警告、不 stop，但空转仍提醒 ─────────────────────────────

def test_flag_off_about_to_expire_no_warn_no_stop_but_idle_reminder(sched, monkeypatch):
    s, rec = sched
    now = time.time()
    _set_flag(s, monkeypatch, False)
    # 空转段读的是低利用率
    monkeypatch.setattr(s, "_get_instance_gpu_util", lambda name: 1.0)
    _install_state(s, monkeypatch, _base_state(now))

    s._check_running_instances()

    assert rec["stop"] == []                             # 从不调 stop
    assert not any(c.get("IDLE_WARN") for c in rec["cards"])  # 无到期警告卡
    # 空转提醒照发
    assert any("空转" in t for t in rec["texts"]), rec["texts"]


def test_flag_off_warned_and_timeout_still_no_stop(sched, monkeypatch):
    s, rec = sched
    now = time.time()
    _set_flag(s, monkeypatch, False)
    monkeypatch.setattr(s, "_get_instance_gpu_util", lambda name: None)
    _install_state(s, monkeypatch, _warned_timeout_state(now))

    s._check_running_instances()

    assert rec["stop"] == []           # 即便 warned+超时也绝不自动 stop
    assert rec["cards"] == []          # 无警告卡
    assert rec["transition"] == []     # 不流转工单
    assert rec["deleted"] == []        # 不清 redis


def test_flag_off_gpu_busy_no_reminder(sched, monkeypatch):
    """flag=False 且 GPU 利用率高 → 既不 stop 也不提醒（纯基线）。"""
    s, rec = sched
    now = time.time()
    _set_flag(s, monkeypatch, False)
    monkeypatch.setattr(s, "_get_instance_gpu_util", lambda name: 88.0)   # 高于阈值
    _install_state(s, monkeypatch, _base_state(now))

    s._check_running_instances()

    assert rec["stop"] == []
    assert rec["texts"] == []
    assert not any(c.get("IDLE_WARN") for c in rec["cards"])


# ── flag=True：恢复旧行为 ──────────────────────────────────────────────────────

def test_flag_on_about_to_expire_sends_warn_card_no_stop(sched, monkeypatch):
    s, rec = sched
    now = time.time()
    _set_flag(s, monkeypatch, True)
    monkeypatch.setattr(s, "_get_instance_gpu_util", lambda name: 1.0)
    _install_state(s, monkeypatch, _base_state(now))

    s._check_running_instances()

    assert any(c.get("IDLE_WARN") for c in rec["cards"])   # 发到期警告卡
    assert rec["stop"] == []                               # 首轮只警告不 stop
    # 警告后 warned/warn_ts 落盘
    assert any(v.get("warned") for _, v in rec["saved"])


def test_flag_on_warned_timeout_auto_stops(sched, monkeypatch):
    s, rec = sched
    now = time.time()
    _set_flag(s, monkeypatch, True)
    monkeypatch.setattr(s, "_get_instance_gpu_util", lambda name: None)
    _install_state(s, monkeypatch, _warned_timeout_state(now))

    s._check_running_instances()

    assert rec["stop"] == [{"action": "stop", "instance_id": "dsw-xyz"}]
    assert any("已自动停止" in t for t in rec["texts"])
    assert rec["transition"] == [("TK-1", "完成")]
    assert rec["deleted"] == ["TK-1"]
