"""tools/feishu/cards.py 卡片原语：精确 dict 形状（迁移以 dict 相等为验收线）。"""
from tools.feishu.cards import card, div, fields, btn, buttons, note, hr, img


def test_card_default():
    c = card("标题", [hr()])
    assert c == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "标题"}, "template": "blue"},
        "elements": [{"tag": "hr"}],
    }


def test_card_color_and_update_multi():
    c = card("t", [], color="red", update_multi=True)
    assert c["config"] == {"wide_screen_mode": True, "update_multi": True}
    assert c["header"]["template"] == "red"


def test_card_headerless():
    c = card(None, [div("x")])
    assert "header" not in c
    assert list(c.keys()) == ["config", "elements"]


def test_card_key_order():
    # json.dumps 字节级一致依赖插入顺序
    assert list(card("t", []).keys()) == ["config", "header", "elements"]


def test_div():
    assert div("**a**") == {"tag": "div", "text": {"tag": "lark_md", "content": "**a**"}}


def test_fields():
    f = fields(("申请人", "张三"), ("工单", "GPU-1"))
    assert f == {"tag": "div", "fields": [
        {"is_short": True, "text": {"tag": "lark_md", "content": "**申请人**\n张三"}},
        {"is_short": True, "text": {"tag": "lark_md", "content": "**工单**\nGPU-1"}},
    ]}


def test_btn_value():
    b = btn("✅ 批准", {"action": "approve_gpu"}, type="primary")
    assert b == {"tag": "button", "text": {"tag": "plain_text", "content": "✅ 批准"},
                 "type": "primary", "value": {"action": "approve_gpu"}}
    assert list(b.keys()) == ["tag", "text", "type", "value"]


def test_btn_default_empty_value():
    assert btn("x")["value"] == {}


def test_btn_url():
    b = btn("Grafana", url="https://g.example/explore")
    assert b == {"tag": "button", "text": {"tag": "plain_text", "content": "Grafana"},
                 "type": "default", "url": "https://g.example/explore"}
    assert "value" not in b


def test_buttons():
    assert buttons(btn("a"), btn("b"))["tag"] == "action"
    assert len(buttons(btn("a"), btn("b"))["actions"]) == 2


def test_note_hr_img():
    assert note("由 AIOps 生成") == {"tag": "note", "elements": [
        {"tag": "plain_text", "content": "由 AIOps 生成"}]}
    assert hr() == {"tag": "hr"}
    assert img("img_v3_x", "趋势图") == {"tag": "img", "img_key": "img_v3_x",
                                        "alt": {"tag": "plain_text", "content": "趋势图"},
                                        "mode": "fit_horizontal"}
