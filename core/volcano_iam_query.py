"""Read-only 火山引擎 IAM 子用户查询（镜像阿里云 RAM 查询）。

复用 core.ram_approval 里已有的火山 IAM client（volcenginesdkiam）。只读，不返回任何密钥。
"""
from __future__ import annotations

from typing import Any


class VolcanoQueryError(RuntimeError):
    """火山 IAM 查询失败。"""


class VolcanoUserNotFound(VolcanoQueryError):
    """火山 IAM 子用户不存在。"""


def _client_and_models():
    from core.ram_approval import make_volcano_iam_client, _volcano_iam_models
    return make_volcano_iam_client(), _volcano_iam_models()


def _is_not_found(exc: Exception) -> bool:
    try:
        from core.ram_approval import _volcano_is_not_exist
        return _volcano_is_not_exist(exc)
    except Exception:
        text = str(exc).lower()
        return any(t in text for t in ("notfound", "not_found", "notexist", "no such"))


def _list_access_keys(client: Any, models: Any, user_name: str) -> list[dict[str, str]]:
    try:
        resp = client.list_access_keys(models.ListAccessKeysRequest(user_name=user_name))
        keys = getattr(resp, "access_key_metadata", None) or []
        return [{
            "access_key_id": getattr(k, "access_key_id", "") or "",
            "status": getattr(k, "status", "") or "",
            "create_date": getattr(k, "create_date", "") or "",
        } for k in keys]
    except Exception:
        return []


def _login_profile_exists(client: Any, models: Any, user_name: str) -> bool:
    try:
        client.get_login_profile(models.GetLoginProfileRequest(user_name=user_name))
        return True
    except Exception:
        return False


def _list_groups(client: Any, models: Any, user_name: str) -> list[str]:
    try:
        fn = getattr(client, "list_groups_for_user", None)
        req = getattr(models, "ListGroupsForUserRequest", None)
        if not fn or not req:
            return []
        resp = fn(req(user_name=user_name))
        groups = getattr(resp, "user_groups", None) or getattr(resp, "groups", None) or []
        out = []
        for g in groups:
            name = getattr(g, "user_group_name", "") or getattr(g, "group_name", "")
            if name:
                out.append(name)
        return out
    except Exception:
        return []


def query_volcano_iam_account(user_name: str, *, client: Any | None = None,
                              models: Any | None = None) -> dict[str, Any]:
    """返回火山 IAM 子用户信息（不含任何密钥）。不存在抛 VolcanoUserNotFound。"""
    user_name = (user_name or "").strip().strip("`'\"<>")
    if not user_name:
        raise VolcanoQueryError("missing user_name")
    if client is None or models is None:
        client, models = _client_and_models()
    try:
        resp = client.get_user(models.GetUserRequest(user_name=user_name))
    except Exception as exc:
        if _is_not_found(exc):
            raise VolcanoUserNotFound(user_name) from exc
        raise VolcanoQueryError(f"failed to query Volcano IAM user: {exc}") from exc

    user = getattr(resp, "user", None)
    data = {f: getattr(user, f, "") or "" for f in (
        "user_name", "display_name", "email", "mobile_phone",
        "account_id", "description", "create_date", "update_date",
    )}
    data["groups"] = _list_groups(client, models, user_name)
    data["console_access"] = _login_profile_exists(client, models, user_name)
    data["access_keys"] = _list_access_keys(client, models, user_name)
    data["access_key_count"] = len(data["access_keys"])
    return data
