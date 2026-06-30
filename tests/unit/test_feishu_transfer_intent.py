"""Feishu transfer intent shortcuts should route to real cards, not the LLM fallback."""

from core.feishu_bot import messages


def test_transfer_entry_intent_direction_shortcuts():
    assert messages._is_transfer_entry_intent("tos到oss")
    assert messages._is_transfer_entry_intent("tos 到 oss")
    assert messages._is_transfer_entry_intent("发一个卡片消息，我来填")
    assert messages._is_transfer_entry_intent("卡片消息呢")


def test_extract_transfer_paths_without_spaces_between_urls():
    source, dest = messages._extract_transfer_paths(
        "tos://wuji-dc-shanghai/trans/到oss://wuji-bucket-hangzhou/wuji-il/wuji-hand-teleop-data/LQX_pick_place/"
    )

    assert source == "tos://wuji-dc-shanghai/trans/"
    assert dest == "oss://wuji-bucket-hangzhou/wuji-il/wuji-hand-teleop-data/LQX_pick_place/"


def test_two_paths_are_transfer_intent_even_without_action_word():
    text = "tos://wuji-dc-shanghai/trans/到oss://wuji-bucket-hangzhou/wuji-il/"

    assert messages._is_transfer_intent(text)

def test_transfer_text_confirm_is_intercepted():
    assert messages._is_transfer_confirm_text("确认")
    assert messages._is_transfer_confirm_text("确认迁移")
    assert not messages._is_transfer_confirm_text("确认一下 GPU 状态")


def test_transfer_menu_text_opens_entry_card():
    assert messages._is_transfer_entry_intent("跨平台对象存储迁移")
    assert messages._is_transfer_entry_intent("对象存储迁移")
    assert not messages._is_transfer_intent("跨平台对象存储迁移")


def test_sink_preheat_menu_text_is_intercepted():
    assert messages._is_sink_preheat_entry_intent("数据沉降/预热")
    assert messages._is_sink_preheat_entry_intent("CEPFS沉降")
    assert not messages._is_transfer_entry_intent("数据沉降/预热")
