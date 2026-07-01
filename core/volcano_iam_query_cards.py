"""Feishu cards：火山引擎 IAM 子用户只读查询（镜像 RAM 查询卡）。"""
from __future__ import annotations

from typing import Any


def _pt(content: str) -> dict[str, str]:
    return {"tag": "plain_text", "content": content}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _yes_no(value: Any) -> str:
    return "是" if bool(value) else "否"


def _md(value: Any) -> str:
    return _clean(value).replace("`", "'")


def query_entry_card() -> dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("火山引擎 IAM 账户查询"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": (
                "填写要查询的火山引擎 IAM 子用户名。\n"
                "> 结果不会返回登录密码或 AccessKey Secret。"
            )},
            {"tag": "form", "name": "volcano_query_form", "elements": [
                {"tag": "input", "name": "user_name", "label": _pt("用户名"), "required": True,
                 "placeholder": _pt("如 zhangxiaoxiong")},
                {"tag": "input", "name": "display_name", "label": _pt("显示名称（可选）"),
                 "placeholder": _pt("用于和 IAM 账号信息核对")},
                {"tag": "input", "name": "email", "label": _pt("邮箱（可选）"),
                 "placeholder": _pt("user@example.com")},
                {"tag": "input", "name": "mobile", "label": _pt("手机（可选）"),
                 "placeholder": _pt("13100000000")},
                {"tag": "button", "text": _pt("查询账户"), "type": "primary",
                 "form_action_type": "submit", "name": "submit",
                 "behaviors": [{"type": "callback", "value": {"action": "submit_volcano_query"}}]},
            ]},
        ]},
    }


def query_result_card(user: dict[str, Any] | None, *, requested: dict[str, str] | None = None) -> dict[str, Any]:
    requested = requested or {}
    user_name = requested.get("user_name") or requested.get("login_name") or requested.get("q") or "-"
    if not user:
        return {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {"title": _pt("火山 IAM 账户未找到"), "template": "orange"},
            "body": {"elements": [
                {"tag": "markdown", "content": f"未找到火山 IAM 子用户：`{_md(user_name)}`\n\n请检查用户名或创建状态。"},
                {"tag": "hr"},
                {"tag": "button", "text": _pt("重新填写"), "type": "default",
                 "behaviors": [{"type": "callback", "value": {"action": "open_volcano_query"}}]},
            ]},
        }

    groups = user.get("groups") or []
    groups_text = groups if isinstance(groups, str) else (", ".join(str(x) for x in groups) or "-")
    keys = user.get("access_keys") or []
    key_lines = [f"- `{_md(k.get('access_key_id'))}` / {k.get('status') or '-'} / {k.get('create_date') or '-'}"
                 for k in keys] or ["- -"]
    md = (
        f"**用户名**：`{_md(user.get('user_name'))}`\n"
        f"**账号 ID**：`{_md(user.get('account_id')) or '-'}`\n"
        f"**显示名称**：{_md(user.get('display_name')) or '-'}\n"
        f"**邮箱**：{_md(user.get('email')) or '-'}\n"
        f"**手机**：{_md(user.get('mobile_phone')) or '-'}\n"
        f"**用户组**：{_md(groups_text)}\n"
        f"**控制台访问**：{_yes_no(user.get('console_access'))}\n"
        f"**AccessKey 数量**：{user.get('access_key_count', len(keys))}\n"
        f"**创建时间**：{_md(user.get('create_date')) or '-'}\n\n"
        "**AccessKey ID / 状态 / 创建时间**\n" + "\n".join(key_lines) +
        "\n\n> 查询接口只返回 AccessKey ID，不返回 Secret 或登录密码。"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("火山 IAM 账户查询结果"), "template": "green"},
        "body": {"elements": [
            {"tag": "markdown", "content": md},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("再查一个"), "type": "default",
             "behaviors": [{"type": "callback", "value": {"action": "open_volcano_query"}}]},
        ]},
    }


def query_error_card(user_name: str, error: str) -> dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("火山 IAM 账户查询失败"), "template": "red"},
        "body": {"elements": [
            {"tag": "markdown", "content": f"**查询对象**：`{_md(user_name)}`\n**错误**：{_md(error)}"},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("重新填写"), "type": "default",
             "behaviors": [{"type": "callback", "value": {"action": "open_volcano_query"}}]},
        ]},
    }
