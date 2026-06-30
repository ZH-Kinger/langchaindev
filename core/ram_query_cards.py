"""Feishu cards for read-only Aliyun RAM account queries."""
from __future__ import annotations

from typing import Any


def _pt(content: str) -> dict[str, str]:
    return {"tag": "plain_text", "content": content}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _yes_no(value: Any) -> str:
    return "\u662f" if bool(value) else "\u5426"


def _md_escape(value: Any) -> str:
    text = _clean(value)
    return text.replace("`", "'")


def query_entry_card() -> dict[str, Any]:
    """Return a form card. Users must submit basic info before any RAM query runs."""
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("RAM \u8d26\u53f7\u67e5\u8be2"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": (
                "\u586b\u5199\u8981\u67e5\u8be2\u7684 RAM \u8d26\u53f7\u57fa\u672c\u4fe1\u606f\u3002\n"
                "\u652f\u6301\u586b\u5199\u77ed\u767b\u5f55\u540d\u3001\u5b8c\u6574\u767b\u5f55\u540d\u6216\u963f\u91cc\u4e91\u767b\u5f55\u94fe\u63a5\u3002\n"
                "> \u7ed3\u679c\u4e0d\u4f1a\u8fd4\u56de\u5bc6\u7801\u6216 AccessKey Secret\u3002"
            )},
            {"tag": "form", "name": "ram_query_form", "elements": [
                {"tag": "input", "name": "login_name", "label": _pt("\u767b\u5f55\u540d\u79f0"), "required": True,
                 "placeholder": _pt("\u5982 zhangxiaoxiong \u6216 zhangxiaoxiong@1704065796538912.onaliyun.com")},
                {"tag": "input", "name": "display_name", "label": _pt("\u663e\u793a\u540d\u79f0\uff08\u53ef\u9009\uff09"),
                 "placeholder": _pt("\u7528\u4e8e\u548c RAM \u8d26\u53f7\u4fe1\u606f\u6838\u5bf9")},
                {"tag": "input", "name": "email", "label": _pt("\u5b89\u5168\u90ae\u7bb1\uff08\u53ef\u9009\uff09"),
                 "placeholder": _pt("user@example.com")},
                {"tag": "input", "name": "mobile", "label": _pt("\u5b89\u5168\u624b\u673a\uff08\u53ef\u9009\uff09"),
                 "placeholder": _pt("86-19100000000")},
                {"tag": "input", "name": "reason", "label": _pt("\u67e5\u8be2\u5907\u6ce8\uff08\u53ef\u9009\uff09"),
                 "placeholder": _pt("\u5982\uff1a\u786e\u8ba4\u5ba1\u6279\u521b\u5efa\u7ed3\u679c")},
                {"tag": "button", "text": _pt("\u67e5\u8be2\u8d26\u53f7"), "type": "primary",
                 "form_action_type": "submit", "name": "submit",
                 "behaviors": [{"type": "callback", "value": {"action": "submit_ram_query"}}]},
            ]},
        ]},
    }


def _mobile_digits(value: str) -> str:
    return "".join(ch for ch in _clean(value) if ch.isdigit())


def _compare_lines(user: dict[str, Any], requested: dict[str, str]) -> list[str]:
    checks: list[tuple[str, str, str, str]] = [
        ("\u663e\u793a\u540d", "display_name", "display_name", "text"),
        ("\u5b89\u5168\u90ae\u7bb1", "email", "email", "email"),
        ("\u5b89\u5168\u624b\u673a", "mobile", "mobile_phone", "mobile"),
    ]
    lines: list[str] = []
    for label, req_key, user_key, mode in checks:
        expected = _clean(requested.get(req_key))
        if not expected:
            continue
        actual = _clean(user.get(user_key))
        if mode == "email":
            ok = expected.lower() == actual.lower()
        elif mode == "mobile":
            e_digits = _mobile_digits(expected)
            a_digits = _mobile_digits(actual)
            ok = bool(e_digits and a_digits and e_digits[-11:] == a_digits[-11:])
        else:
            ok = expected == actual
        status = "\u4e00\u81f4" if ok else "\u4e0d\u4e00\u81f4"
        lines.append(f"- **{label}**\uff1a{status}\uff08\u586b\u5199 `{_md_escape(expected)}` / RAM `{_md_escape(actual) or '-'}`\uff09")
    return lines


def query_result_card(user: dict[str, Any] | None, *, requested: dict[str, str] | None = None) -> dict[str, Any]:
    requested = requested or {}
    login_name = requested.get("login_name") or requested.get("user_name") or requested.get("q") or "-"
    if not user:
        return {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {"title": _pt("RAM \u8d26\u53f7\u672a\u627e\u5230"), "template": "orange"},
            "body": {"elements": [
                {"tag": "markdown", "content": f"\u672a\u627e\u5230 RAM \u7528\u6237\uff1a`{_md_escape(login_name)}`\n\n\u8bf7\u68c0\u67e5\u767b\u5f55\u540d\u6216\u5ba1\u6279\u521b\u5efa\u72b6\u6001\u3002"},
                {"tag": "hr"},
                {"tag": "button", "text": _pt("\u91cd\u65b0\u586b\u5199"), "type": "default",
                 "behaviors": [{"type": "callback", "value": {"action": "open_ram_query"}}]},
            ]},
        }

    groups = user.get("groups") or []
    if isinstance(groups, str):
        groups_text = groups
    else:
        groups_text = ", ".join(str(x) for x in groups) or "-"
    keys = user.get("access_keys") or []
    key_lines = []
    for key in keys:
        key_lines.append(
            f"- `{_md_escape(key.get('access_key_id'))}` / {key.get('status') or '-'} / {key.get('create_date') or '-'}"
        )
    if not key_lines:
        key_lines.append("- -")

    verify = _compare_lines(user, requested)
    verify_md = "\n\n**\u586b\u5199\u4fe1\u606f\u6838\u5bf9**\n" + "\n".join(verify) if verify else ""
    md = (
        f"**\u767b\u5f55\u540d\u79f0**\uff1a`{_md_escape(user.get('user_name'))}`\n"
        f"**\u767b\u5f55\u5730\u5740\u7528\u6237\u540d**\uff1a`{_md_escape(user.get('login_principal'))}`\n"
        f"**\u663e\u793a\u540d\u79f0**\uff1a{_md_escape(user.get('display_name')) or '-'}\n"
        f"**\u5b89\u5168\u90ae\u7bb1**\uff1a{_md_escape(user.get('email')) or '-'}\n"
        f"**\u5b89\u5168\u624b\u673a**\uff1a{_md_escape(user.get('mobile_phone')) or '-'}\n"
        f"**\u7528\u6237\u7ec4**\uff1a{_md_escape(groups_text)}\n"
        f"**\u63a7\u5236\u53f0\u8bbf\u95ee**\uff1a{_yes_no(user.get('console_access'))}\n"
        f"**AccessKey \u6570\u91cf**\uff1a{user.get('access_key_count', len(keys))}\n"
        f"**\u521b\u5efa\u65f6\u95f4**\uff1a{_md_escape(user.get('create_date')) or '-'}\n"
        f"**\u6700\u540e\u767b\u5f55**\uff1a{_md_escape(user.get('last_login_date')) or '-'}"
        f"{verify_md}\n\n"
        "**AccessKey ID / \u72b6\u6001 / \u521b\u5efa\u65f6\u95f4**\n" + "\n".join(key_lines) +
        "\n\n> \u67e5\u8be2\u63a5\u53e3\u53ea\u8fd4\u56de AccessKey ID\uff0c\u4e0d\u8fd4\u56de AccessKey Secret \u6216\u767b\u5f55\u5bc6\u7801\u3002"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("RAM \u8d26\u53f7\u67e5\u8be2\u7ed3\u679c"), "template": "green"},
        "body": {"elements": [
            {"tag": "markdown", "content": md},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("\u518d\u67e5\u4e00\u4e2a"), "type": "default",
             "behaviors": [{"type": "callback", "value": {"action": "open_ram_query"}}]},
        ]},
    }


def query_error_card(login_name: str, error: str) -> dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("RAM \u8d26\u53f7\u67e5\u8be2\u5931\u8d25"), "template": "red"},
        "body": {"elements": [
            {"tag": "markdown", "content": f"**\u67e5\u8be2\u5bf9\u8c61**\uff1a`{_md_escape(login_name)}`\n**\u9519\u8bef**\uff1a{_md_escape(error)}"},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("\u91cd\u65b0\u586b\u5199"), "type": "default",
             "behaviors": [{"type": "callback", "value": {"action": "open_ram_query"}}]},
        ]},
    }