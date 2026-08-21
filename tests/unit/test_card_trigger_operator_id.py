"""card.action.trigger 的 operator 形状 —— 管理员门禁的入口。

线上事故：`_handle_card_trigger_sync` 只按 `operator.operator_id.open_id` 取身份，
而 `card.action.trigger` 事件里是 **`operator.open_id`（扁平，少一层）**。
于是卡片回调拿到的 open_id 恒为空 → `_is_admin("")` fail-closed 判 False →
**管理员点自己的按钮也被拒**，且 happy path 无日志、毫无线索。

这里锁住两件事：能取到身份；取不到时**仍然 fail-closed**（不能为了修这个 bug 而放宽门禁）。
"""
import pytest

from core.feishu_bot import actions


@pytest.fixture(autouse=True)
def _no_dedup(monkeypatch):
    monkeypatch.setattr(actions, "card_action_is_duplicate", lambda *a, **k: False)


@pytest.fixture
def seen(monkeypatch):
    box = {}
    monkeypatch.setattr(actions, "_process_action",
                        lambda name, val, oid, chat, form_value=None: box.update(
                            action=name, open_id=oid) or {"ok": True})
    return box


def _payload(operator):
    return {"schema": "2.0",
            "header": {"event_type": "card.action.trigger"},
            "event": {"operator": operator,
                      "action": {"value": {"action": "confirm_ssh_transfer", "job_id": "j1"}},
                      "context": {"open_chat_id": "oc_1", "open_message_id": "om_1"}}}


def test_flat_operator_open_id_is_read(seen):
    """card.action.trigger 的真实形状：operator.open_id。"""
    actions._handle_card_trigger_sync(_payload({"open_id": "ou_flat", "tenant_key": "t"}))
    assert seen["open_id"] == "ou_flat"


def test_nested_operator_id_still_works(seen):
    """别的事件类型是 operator.operator_id.open_id —— 兼容不能丢。"""
    actions._handle_card_trigger_sync(_payload({"operator_id": {"open_id": "ou_nested"}}))
    assert seen["open_id"] == "ou_nested"


def test_flat_wins_when_both_present(seen):
    actions._handle_card_trigger_sync(
        _payload({"open_id": "ou_flat", "operator_id": {"open_id": "ou_nested"}}))
    assert seen["open_id"] == "ou_flat"


@pytest.mark.parametrize("operator", [{}, None, "not-a-dict", {"operator_id": "not-a-dict"},
                                      {"open_id": ""}, {"operator_id": {}}])
def test_unidentifiable_operator_yields_empty_not_crash(seen, operator):
    """取不到身份时返回空串、不抛 —— 后续由 _is_admin 判 False（fail-closed）。

    **绝不能**为了让按钮能用而在这里编一个身份出来。"""
    actions._handle_card_trigger_sync(_payload(operator))
    assert seen["open_id"] == ""


def test_empty_open_id_is_never_admin(monkeypatch):
    """守住 fail-closed：空身份永远不是管理员，哪怕管理员配置也是空的。"""
    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)
    assert actions._is_admin("") is False
    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin", raising=False)
    assert actions._is_admin("") is False
    assert actions._is_admin("ou_admin") is True
