"""数据预热/沉降统一向导卡：选云 → 选地区 → 选文件系统 + 填源/目的地址。

阿里 CPFS↔OSS 与 火山 vePFS↔TOS 共用同一套卡。方向由源/目的地址自动判断：
带 oss:// 或 tos:// 的一侧=对象存储，另一侧=文件系统目录。
源=对象存储 → 预热(加载到文件系统)；源=文件系统 → 沉降(刷回对象存储)。

桶不做下拉，直接写在 oss://.../ 或 tos://.../ 地址里；文件系统按地区级联下拉。
"""

_CLOUD = {
    "aliyun": {
        "name": "阿里 CPFS↔OSS", "fs": "CPFS 文件系统", "obj": "OSS",
        "obj_scheme": "oss://", "fs_scheme": "cpfs://",
        "submit": "submit_cpfs_dataflow", "pick_region": "pick_region_aliyun",
        "tpl": "blue", "icon": "\U0001f504",
    },
    "volcano": {
        "name": "火山 vePFS↔TOS", "fs": "vePFS 文件系统", "obj": "TOS",
        "obj_scheme": "tos://", "fs_scheme": "vepfs://",
        "submit": "submit_vepfs_dataflow", "pick_region": "pick_region_volcano",
        "tpl": "orange", "icon": "\U0001f30b",
    },
}


def _pt(s):
    return {"tag": "plain_text", "content": s}


def _same_name_options():
    return [
        {"text": _pt("跳过同名（默认，永不覆盖）"), "value": "skip"},
        {"text": _pt("保留最新（比最后修改时间）"), "value": "keeplatest"},
        {"text": _pt("覆盖同名"), "value": "overwrite"},
    ]


def entry_card():
    """入口：选云平台 → 进入对应向导（选地区→选文件系统→填地址）。"""
    intro = ("**数据预热 / 沉降**：选云平台 → 选地区 → 选文件系统 → 填源/目的地址。\n"
             "• 阿里 CPFS ↔ OSS　• 火山 vePFS ↔ TOS\n"
             "> 方向自动判断：源=对象存储→**预热**；源=文件系统→**沉降**。")
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f504 数据预热 / 沉降"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": intro},
            {"tag": "button", "text": _pt("阿里 CPFS ↔ OSS"), "type": "primary",
             "behaviors": [{"type": "callback", "value": {"action": "pick_cloud_aliyun"}}]},
            {"tag": "button", "text": _pt("火山 vePFS ↔ TOS"), "type": "primary",
             "behaviors": [{"type": "callback", "value": {"action": "pick_cloud_volcano"}}]},
        ]},
    }


def region_card(cloud, regions):
    """选地区：每个地区一个按钮，点了带 cloud+region 进 form_card。"""
    c = _CLOUD[cloud]
    elems = [{"tag": "markdown", "content": f"**{c['name']}** —— 先选地区："}]
    regions = regions or []
    for r in regions:
        elems.append({"tag": "button", "text": _pt(r), "type": "primary",
                      "behaviors": [{"type": "callback",
                                     "value": {"action": c["pick_region"], "region": r}}]})
    if not regions:
        elems.append({"tag": "markdown",
                      "content": "> 未发现地区（发现服务为空/无权限），点下面按钮手动录入地址。"})
        elems.append({"tag": "button", "text": _pt("手动录入"), "type": "primary",
                      "behaviors": [{"type": "callback",
                                     "value": {"action": c["pick_region"], "region": ""}}]})
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt(f"{c['icon']} {c['name']} · 选地区"), "template": c["tpl"]},
        "body": {"elements": elems},
    }


def form_card(cloud, region, fs_options):
    """选文件系统（该地区，级联下拉）+ 填源/目的地址。方向自动判断，无操作/桶选择器。"""
    c = _CLOUD[cloud]
    fs_options = fs_options or []
    if fs_options:
        fs_el = {"tag": "select_static", "name": "fs", "required": False,
                 "placeholder": _pt(f"选择{c['fs']}"),
                 "options": [{"text": _pt(o["text"]), "value": o["value"]} for o in fs_options]}
    else:
        fs_el = {"tag": "input", "name": "fs", "required": True,
                 "label": _pt(f"{c['fs']}（{c['fs_scheme']}<fs-id> 或 fs-id）"),
                 "placeholder": _pt(f"{c['fs_scheme']}<fs-id>")}
    intro = (f"**{c['name']}** · 地区 **{region or '(手动)'}**\n"
             f"选{c['fs']}，再填**源地址**、**目的地址**：一侧是文件系统目录 `/dir/`，"
             f"另一侧是对象存储 `{c['obj_scheme']}bucket/prefix/`（桶写在地址里）。\n"
             f"> 源=对象存储→**预热**；源=文件系统→**沉降**。")
    form_elems = [
        fs_el,
        {"tag": "input", "name": "source", "label": _pt("源地址"), "required": True,
         "placeholder": _pt(f"/label/ 或 {c['obj_scheme']}bk/pfx/")},
        {"tag": "input", "name": "dest", "label": _pt("目的地址"), "required": True,
         "placeholder": _pt(f"{c['obj_scheme']}bk/pfx/ 或 /label/")},
        {"tag": "select_static", "name": "same_name", "required": False,
         "placeholder": _pt("同名策略（默认跳过）"), "options": _same_name_options()},
        {"tag": "button", "text": _pt("➡️ 解析预览"), "type": "primary",
         "form_action_type": "submit", "name": "submit",
         "behaviors": [{"type": "callback", "value": {"action": c["submit"], "region": region}}]},
    ]
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt(f"{c['icon']} {c['name']} · {region or '手动'}"), "template": c["tpl"]},
        "body": {"elements": [
            {"tag": "markdown", "content": intro},
            {"tag": "form", "name": "dataflow_form", "elements": form_elems},
        ]},
    }
