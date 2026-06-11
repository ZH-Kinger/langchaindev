"""
飞书交互卡片构建原语（低级 builder）。

全仓库手搓 card dict 的公共约定收口于此：config(update_multi/wide_screen_mode)、
header 配色（blue/green/orange/red 表状态）、lark_md div、fields 栅格、action 按钮、
hr 分隔、note 脚注、img 图块。各业务卡片布局仍留在各自模块，仅改用这些原语拼装。

key 插入顺序与既有字面量一致（config→header→elements；tag→text→type→value|url），
保证迁移后 json.dumps 输出字节级不变。

非 agent 工具，勿加入 tools/feishu/__init__ 导出。
"""


def card(title, elements, *, color="blue", wide=True, update_multi=False):
    """整卡。title=None → 无 header（如图表回复卡）；update_multi=True 供回调原地替换。"""
    cfg = {"wide_screen_mode": wide}
    if update_multi:
        cfg["update_multi"] = True
    out = {"config": cfg}
    if title is not None:
        out["header"] = {"title": {"tag": "plain_text", "content": title}, "template": color}
    out["elements"] = list(elements)
    return out


def div(md):
    """lark_md 文本块。"""
    return {"tag": "div", "text": {"tag": "lark_md", "content": md}}


def fields(*pairs, short=True):
    """两列栅格：(label, value) → '**label**\\nvalue'。"""
    return {"tag": "div", "fields": [
        {"is_short": short, "text": {"tag": "lark_md", "content": f"**{l}**\n{v}"}}
        for l, v in pairs]}


def btn(text, value=None, type="default", url=None):
    """按钮。url= → URL 跳转按钮（无 value）；否则回调按钮（value 默认 {}）。"""
    b = {"tag": "button", "text": {"tag": "plain_text", "content": text}, "type": type}
    if url is not None:
        b["url"] = url
    else:
        b["value"] = value or {}
    return b


def buttons(*btns):
    """按钮行。"""
    return {"tag": "action", "actions": list(btns)}


def note(text):
    """脚注（时间戳/出处）。"""
    return {"tag": "note", "elements": [{"tag": "plain_text", "content": text}]}


def hr():
    return {"tag": "hr"}


def img(image_key, alt):
    return {"tag": "img", "img_key": image_key,
            "alt": {"tag": "plain_text", "content": alt}, "mode": "fit_horizontal"}
