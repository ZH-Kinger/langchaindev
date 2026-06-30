"""Read-only Aliyun RAM account query helpers."""
from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

from config.settings import settings


class RamQueryError(RuntimeError):
    """Raised when RAM query cannot be completed."""


class RamUserNotFound(RamQueryError):
    """Raised when a RAM user does not exist."""


def normalize_login_name(raw: str) -> str:
    """Accept RAM username, RAM login principal, or Aliyun login URL."""
    value = (raw or "").strip().strip("`'\"<>")
    if not value:
        return ""

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        username = parse_qs(parsed.query).get("username", [""])[0]
        if username:
            value = username.strip()

    domain = (
        getattr(settings, "ALIYUN_RAM_LOGIN_DOMAIN", "")
        or getattr(settings, "ALIYUN_RAM_DEFAULT_DOMAIN", "")
        or ""
    ).strip().lstrip("@").lower()
    if "@" in value:
        name, principal_domain = value.rsplit("@", 1)
        principal_domain_lc = principal_domain.lower()
        if (domain and principal_domain_lc == domain) or principal_domain_lc.endswith(".onaliyun.com"):
            value = name
    return value


def build_login_principal(user_name: str) -> str:
    domain = (
        getattr(settings, "ALIYUN_RAM_LOGIN_DOMAIN", "")
        or getattr(settings, "ALIYUN_RAM_DEFAULT_DOMAIN", "")
        or ""
    ).strip().lstrip("@")
    return f"{user_name}@{domain}" if domain else user_name


def _make_ram_client() -> Any:
    try:
        from alibabacloud_ram20150501.client import Client
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError as exc:
        raise RamQueryError("alibabacloud_ram20150501 is not installed") from exc

    ak = getattr(settings, "PAI_DSW_ACCESS_KEY_ID", "") or getattr(settings, "ALIYUN_ACCESS_KEY_ID", "")
    sk = getattr(settings, "PAI_DSW_ACCESS_KEY_SECRET", "") or getattr(settings, "ALIYUN_ACCESS_KEY_SECRET", "")
    if not ak or not sk:
        raise RamQueryError("missing RAM query AccessKey")
    return Client(open_api_models.Config(access_key_id=ak, access_key_secret=sk, endpoint="ram.aliyuncs.com"))


def _is_not_found(exc: Exception) -> bool:
    text = str(exc)
    return "EntityNotExist" in text or "NotFound" in text or "NoSuch" in text


def _obj_to_dict(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: getattr(obj, field, "") or "" for field in fields}


def _list_groups(client: Any, user_name: str) -> list[str]:
    try:
        from alibabacloud_ram20150501 import models as ram_models
        resp = client.list_groups_for_user(ram_models.ListGroupsForUserRequest(user_name=user_name))
        groups = getattr(getattr(resp.body, "groups", None), "group", None) or []
        return [getattr(g, "group_name", "") for g in groups if getattr(g, "group_name", "")]
    except Exception:
        return []


def _login_profile_exists(client: Any, user_name: str) -> bool:
    try:
        from alibabacloud_ram20150501 import models as ram_models
        client.get_login_profile(ram_models.GetLoginProfileRequest(user_name=user_name))
        return True
    except Exception:
        return False


def _list_access_keys(client: Any, user_name: str) -> list[dict[str, str]]:
    try:
        from alibabacloud_ram20150501 import models as ram_models
        resp = client.list_access_keys(ram_models.ListAccessKeysRequest(user_name=user_name))
        keys = getattr(getattr(resp.body, "access_keys", None), "access_key", None) or []
        result = []
        for key in keys:
            result.append({
                "access_key_id": getattr(key, "access_key_id", "") or "",
                "status": getattr(key, "status", "") or "",
                "create_date": getattr(key, "create_date", "") or "",
            })
        return result
    except Exception:
        return []


def query_ram_account(login_name: str, *, client: Any | None = None) -> dict[str, Any]:
    """Return RAM account details without any secret fields."""
    from alibabacloud_ram20150501 import models as ram_models

    user_name = normalize_login_name(login_name)
    if not user_name:
        raise RamQueryError("missing login_name")

    client = client or _make_ram_client()
    try:
        resp = client.get_user(ram_models.GetUserRequest(user_name=user_name))
    except Exception as exc:
        if _is_not_found(exc):
            raise RamUserNotFound(user_name) from exc
        raise RamQueryError(f"failed to query RAM user: {exc}") from exc

    user = getattr(resp.body, "user", None)
    data = _obj_to_dict(user, (
        "user_id", "user_name", "display_name", "email", "mobile_phone",
        "comments", "create_date", "update_date", "last_login_date",
    ))
    data["login_principal"] = build_login_principal(data.get("user_name") or user_name)
    data["groups"] = _list_groups(client, user_name)
    data["console_access"] = _login_profile_exists(client, user_name)
    data["access_keys"] = _list_access_keys(client, user_name)
    data["access_key_count"] = len(data["access_keys"])
    return data
