"""OSS 权限同步的飞书卡片：对账卡片（带批准按钮）+ 下发结果卡片。"""
from core.oss_perm.permsync import POLICY_PREFIX
from tools.feishu.cards import btn, buttons, card, div, hr


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
    """对账卡片。有差异/孤儿 → 橙色 + 「批准并下发」按钮；一致 → 绿色仅通知。"""
    rows, orphans = diff["rows"], diff["orphans"]
    changed = [r for r in rows if r["status"] != "ok"]
    has_change = bool(changed or orphans)
    title = ("⏳ OSS 权限待同步" if has_change else "✅ OSS 权限一致") + f"（{len(rows)} 人对账）"

    elements = []
    if changed:
        lines = [f"**{r['name']}**（{r['username']}）：{_row_detail(r)}" for r in changed[:40]]
        if len(changed) > 40:
            lines.append(f"…（共 {len(changed)} 人需同步）")
        elements.append(div("**需同步：**\n" + "\n".join(lines)))
    if orphans:
        ol = [f"• {POLICY_PREFIX}{u}" + (f"（附加于 {', '.join(att)}）" if att else "")
              for u, att in sorted(orphans.items())]
        elements.append(div("**孤儿策略（建议回收）：**\n" + "\n".join(ol)))
    if not has_change:
        elements.append(div("RAM 实际权限与飞书表格期望一致，无需同步。"))
    else:
        elements.append(hr())
        elements.append(div("管理员点「批准并下发」将按当前表格下发到 RAM（幂等；不回收多授/孤儿）。"))
        elements.append(buttons(
            btn("✅ 批准并下发", {"action": "approve_oss_perm"}, "primary")))

    return card(title, elements, color="orange" if has_change else "green")


def result_card(summary):
    """下发结果卡片。"""
    s = summary
    head = f"OSS 权限下发完成：成功 {s['ok']}，失败 {s['fail']}，无RAM用户 {s['no_user']}"
    lines = s["lines"][:40]
    body = "\n".join(lines)
    if len(s["lines"]) > 40:
        body += f"\n…（共 {len(s['lines'])} 条）"
    return card(head, [div(body or "（无）")], color="red" if s["fail"] else "green")
