"""core/feishu_bot.py 绑定流程单元测试。"""
import pytest


OPEN_ID = "ou_bind_test"
RAM_USER = {
    "user_id":      "1234567",
    "user_name":    "wzh.wang",
    "display_name": "王梓涵",
}


# ── _is_registered ──────────────────────────────────────────────────────────

def test_is_registered_user_ak_path(monkeypatch):
    from core import feishu_bot
    monkeypatch.setattr("utils.aliyun_user_creds.has_user_ak", lambda oid: True)
    assert feishu_bot._is_registered(OPEN_ID) is True


def test_is_registered_ram_map_path(monkeypatch):
    from core import feishu_bot
    monkeypatch.setattr("utils.aliyun_user_creds.has_user_ak", lambda oid: False)
    monkeypatch.setattr(
        "tools.aliyun.ram.get_ram_user_by_open_id",
        lambda oid: {"user_name": "u"} if oid == OPEN_ID else None,
    )
    assert feishu_bot._is_registered(OPEN_ID) is True


def test_is_registered_returns_false_when_no_binding(monkeypatch):
    from core import feishu_bot
    monkeypatch.setattr("utils.aliyun_user_creds.has_user_ak", lambda oid: False)
    monkeypatch.setattr(
        "tools.aliyun.ram.get_ram_user_by_open_id",
        lambda oid: None,
    )
    assert feishu_bot._is_registered(OPEN_ID) is False


def test_is_registered_empty_open_id():
    from core import feishu_bot
    assert feishu_bot._is_registered("") is False


# ── _handle_ram_bind 四种指令路由 ────────────────────────────────────────────

def test_handle_unbind_ak(monkeypatch, mock_feishu_send):
    from core import feishu_bot
    deleted = {"called": False}
    def fake_del(oid):
        deleted["called"] = True
        return True
    monkeypatch.setattr("utils.aliyun_user_creds.delete_user_ak", fake_del)

    handled = feishu_bot._handle_ram_bind("msg_1", OPEN_ID, "解绑AK")
    assert handled is True
    assert deleted["called"] is True
    assert any("已解绑" in m["text"] for m in mock_feishu_send)


def test_handle_unbind_ak_when_nothing_to_delete(monkeypatch, mock_feishu_send):
    from core import feishu_bot
    monkeypatch.setattr("utils.aliyun_user_creds.delete_user_ak", lambda oid: False)
    feishu_bot._handle_ram_bind("msg_2", OPEN_ID, "解绑AK")
    assert any("尚未绑定" in m["text"] for m in mock_feishu_send)


def test_handle_view_binding(monkeypatch, mock_feishu_send):
    from core import feishu_bot
    monkeypatch.setattr(
        "utils.aliyun_user_creds.get_user_ak_meta",
        lambda oid: {"created_ts": 1700000000, "last_used_ts": 1700000100,
                     "ak_id_masked": "LTAI5tXX****"},
    )
    monkeypatch.setattr(
        "tools.aliyun.ram.get_ram_user_by_open_id",
        lambda oid: {"user_name": "wzh.wang"},
    )
    handled = feishu_bot._handle_ram_bind("msg_3", OPEN_ID, "查看绑定")
    assert handled is True
    text = "\n".join(m["text"] for m in mock_feishu_send)
    assert "LTAI5tXX****" in text
    assert "RAM 映射" in text


def test_handle_register_ak_command(monkeypatch, mock_feishu_send):
    """「绑定AK」/「注册AK」 → 发送 AK 注册卡片引导。"""
    from core import feishu_bot
    handled = feishu_bot._handle_ram_bind("msg_4", OPEN_ID, "绑定AK")
    assert handled is True
    # 应触发 _send_ak_register_card → 走 _feishu_reply（已在 fixture mock）
    assert len(mock_feishu_send) >= 1


def test_handle_bind_ram_user_success(monkeypatch, mock_feishu_send, mock_ram_api):
    from core import feishu_bot
    mock_ram_api["users"].append(RAM_USER)
    # save_user_map 拦截
    saved = {}
    def fake_save_map(oid, name, uid):
        saved["open_id"] = oid; saved["name"] = name; saved["uid"] = uid
    monkeypatch.setattr("tools.aliyun.ram.save_user_map", fake_save_map)

    handled = feishu_bot._handle_ram_bind("msg_5", OPEN_ID, "绑定RAM: wzh.wang")
    assert handled is True
    assert saved["open_id"] == OPEN_ID
    assert saved["uid"] == RAM_USER["user_id"]
    assert any("已建立 RAM 映射" in m["text"] for m in mock_feishu_send)


def test_handle_bind_ram_user_not_found(monkeypatch, mock_feishu_send, mock_ram_api):
    from core import feishu_bot
    mock_ram_api["users"] = []   # 没有任何 RAM 用户
    handled = feishu_bot._handle_ram_bind("msg_6", OPEN_ID, "绑定RAM: ghost")
    assert handled is True
    assert any("未找到" in m["text"] for m in mock_feishu_send)


def test_handle_returns_false_for_unrelated_text():
    from core import feishu_bot
    assert feishu_bot._handle_ram_bind("msg_7", OPEN_ID, "今天天气怎么样") is False


# ── submit_ak_register 卡片表单 ──────────────────────────────────────────────

def test_submit_ak_register_success(monkeypatch, mock_ram_api):
    from core import feishu_bot
    mock_ram_api["users"].append(RAM_USER)
    saved = {}
    def fake_save_ak(oid, ak, sk):
        saved["ak"] = ak; saved["sk"] = sk
        return True
    monkeypatch.setattr("utils.aliyun_user_creds.save_user_ak", fake_save_ak)
    monkeypatch.setattr("tools.aliyun.ram.save_user_map", lambda *a: None)

    result = feishu_bot._process_action(
        "submit_ak_register", {},
        open_id=OPEN_ID, chat_id="oc_x",
        form_value={"ak_id": "LTAI5tValidAK", "ak_secret": "TheSecret"},
    )
    assert saved["ak"] == "LTAI5tValidAK"
    assert saved["sk"] == "TheSecret"
    assert result["toast"]["type"] == "success"


def test_submit_ak_register_missing_fields():
    from core import feishu_bot
    result = feishu_bot._process_action(
        "submit_ak_register", {},
        open_id=OPEN_ID, chat_id="oc_x",
        form_value={"ak_id": "", "ak_secret": "x"},
    )
    assert result["toast"]["type"] == "error"
    assert "AccessKey" in result["toast"]["content"]


def test_submit_ak_register_rejects_invalid_prefix():
    from core import feishu_bot
    result = feishu_bot._process_action(
        "submit_ak_register", {},
        open_id=OPEN_ID, chat_id="oc_x",
        form_value={"ak_id": "AKIA_WRONG_FORMAT", "ak_secret": "x"},
    )
    assert result["toast"]["type"] == "error"
    assert "LTAI" in result["toast"]["content"]


def test_submit_ak_register_save_failure(monkeypatch):
    from core import feishu_bot
    monkeypatch.setattr("utils.aliyun_user_creds.save_user_ak", lambda *a: False)
    result = feishu_bot._process_action(
        "submit_ak_register", {},
        open_id=OPEN_ID, chat_id="oc_x",
        form_value={"ak_id": "LTAI5tValidAK", "ak_secret": "TheSecret"},
    )
    assert result["toast"]["type"] == "error"
    assert "保存失败" in result["toast"]["content"]