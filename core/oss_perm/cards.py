"""OSS 权限同步的飞书卡片。

- audit_form_card：对账「表单卡」(卡片 JSON 2.0)，定时推送用。三段式：
  ① 粒度单选(默认桶级) ② 成员多选(默认全选；取消勾选=本轮跳过) ③ 单个「确认下发」按钮。
  提交回调 action=approve_oss_perm_selected，form_value={level, selected:[用户名]}。
- audit_card：旧版 1.0 对账卡(两按钮整批下发)，保留兜底，现已不被定时任务调用。
- result_card：下发结果卡。
"""
from tools.feishu.cards import btn, buttons, card, div, hr


def _pt(content):
    return {"tag": "plain_text", "content": content}


def _fmt_prefixes(s):
    return ", ".join(sorted(x or "<整桶>" for x in s)) if s else ""


def _row_detail(row):
    if row["status"] == "missing":
        return "全部少授（RAM 无该策略）"
    parts = []
    for bucket, d in row["diff"].items():
        over, under, seg = d["over"], d["under"], []
        o = []
        if over["read"]:
            o.append("读 " + _fmt_prefixes(over["read"]))
        if over["write"]:
            o.append("写 " + _fmt_prefixes(over["write"]))
        if o:
            seg.append("多授[" + "; ".join(o) + "]")
        u = []
        if under["read"]:
            u.append("读 " + _fmt_prefixes(under["read"]))
        if under["write"]:
            u.append("写 " + _fmt_prefixes(under["write"]))
        if u:
            seg.append("少授[" + "; ".join(u) + "]")
        parts.append(f"{bucket}: " + " ".join(seg))
    return " / ".join(parts)


def audit_card(diff):
    """对账卡片。有差异 → 橙色 + 批准按钮；一致 → 绿色仅通知。（孤儿策略不展示）"""
    rows = diff["rows"]
    changed = [r for r in rows if r["status"] != "ok"]
    has_change = bool(changed)
    title = ("⏳ OSS 权限待同步" if has_change else "✅ OSS 权限一致") + f"（{len(rows)} 人对账）"

    elements = []
    if changed:
        lines = [f"**{r['name']}**（{r['username']}）：{_row_detail(r)}" for r in changed[:40]]
        if len(changed) > 40:
            lines.append(f"…（共 {len(changed)} 人需同步）")
        elements.append(div("**需同步：**\n" + "\n".join(lines)))
    if not has_change:
        elements.append(div("RAM 实际权限与飞书表格期望一致，无需同步。"))
    else:
        elements.append(hr())
        elements.append(div(
            "管理员选粒度下发到 RAM（幂等，只补授不回收多授）：\n"
            "**桶级** = 整桶读写，先放开观察有无问题；**目录级** = 收紧到子目录前缀。"))
        elements.append(buttons(
            btn("✅ 批准·桶级", {"action": "approve_oss_perm", "level": "bucket"}, "primary"),
            btn("✅ 批准·目录级", {"action": "approve_oss_perm", "level": "dir"}, "primary")))

    return card(title, elements, color="orange" if has_change else "green")


def result_card(summary):
    """下发结果卡片。每行列出该成员实际限制到的桶/目录范围。"""
    s = summary
    lvl = {"bucket": "桶级", "dir": "目录级"}.get(s.get("level"), "")
    tag = f"（{lvl}）" if lvl else ""
    head = f"OSS 权限下发完成{tag}：成功 {s['ok']}，失败 {s['fail']}，无RAM用户 {s['no_user']}"
    lines = s["lines"][:40]
    body = "\n".join(lines)
    if len(s["lines"]) > 40:
        body += f"\n…（共 {len(s['lines'])} 条）"
    return card(head, [div(body or "（无）")], color="red" if s["fail"] else "green")


# 多选最多放这么多成员（飞书 multi_select 选项上限保守取值）；超出则截断并提示
_MEMBER_OPT_LIMIT = 50


def audit_form_card(diff):
    """对账表单卡（卡片 JSON 2.0）。一致 → 绿色仅通知；有差异 → 粒度单选 + 成员多选 + 确认下发。"""
    rows = diff["rows"]
    changed = [r for r in rows if r["status"] != "ok"]
    title = ("⏳ OSS 权限待同步" if changed else "✅ OSS 权限一致") + f"（{len(rows)} 人对账）"
    base = {"schema": "2.0", "config": {"wide_screen_mode": True},
            "header": {"title": _pt(title), "template": "orange" if changed else "green"}}

    if not changed:
        base["body"] = {"elements": [
            {"tag": "markdown", "content": "RAM 实际权限与飞书表格期望一致，无需同步。"}]}
        return base

    shown = changed[:_MEMBER_OPT_LIMIT]
    lines = [f"- **{r['name']}**（{r['username']}）：{_row_detail(r)}" for r in shown]
    member_opts = [{"text": _pt(f"{r['name']} ({r['username']})"), "value": r["username"]}
                   for r in shown]
    if len(changed) > _MEMBER_OPT_LIMIT:
        lines.append(f"- …（共 {len(changed)} 人，仅前 {_MEMBER_OPT_LIMIT} 人可在下方勾选，"
                     f"其余请用命令行 `--only` 处理）")
    all_vals = [o["value"] for o in member_opts]
    lvl_opts = [
        {"text": _pt("桶级 — 整桶读写，先放开观察"), "value": "bucket"},
        {"text": _pt("目录级 — 收紧到子目录前缀"), "value": "dir"},
    ]

    base["body"] = {"elements": [
        {"tag": "markdown", "content": "**需同步：**\n" + "\n".join(lines)},
        {"tag": "hr"},
        {"tag": "form", "name": "oss_perm_form", "elements": [
            {"tag": "markdown", "content": "**① 粒度**"},
            {"tag": "select_static", "name": "level", "initial_option": "bucket",
             "placeholder": _pt("选择下发粒度"), "options": lvl_opts},
            {"tag": "markdown", "content": "**② 下发对象**（默认全选；取消勾选 = 本轮不发此人）"},
            {"tag": "multi_select_static", "name": "selected", "selected_values": all_vals,
             "placeholder": _pt("选择要下发的成员"), "options": member_opts},
            {"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
             "form_action_type": "submit", "name": "submit",
             "behaviors": [{"type": "callback", "value": {"action": "approve_oss_perm_selected"}}]},
        ]},
    ]}
    return base
