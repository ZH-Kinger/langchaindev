"""
阿里云 STS AssumeRole 工具。

Master 账号（settings.ALIYUN_BOT_MASTER_AK_*）调 sts:AssumeRole，
为每个飞书用户换取临时 AK/SK/Token，用于调云 API。

凭证按 (open_id, role_arn) 缓存到 Redis，过期前 5 分钟自动刷新。
"""
import json
import time
from typing import Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── 用户组 → 角色 ARN 映射（启动时解析一次）──────────────────────────────────

def _parse_role_mapping() -> list[dict]:
    try:
        raw = settings.ALIYUN_BOT_ROLE_MAPPING_RAW or "[]"
        items = json.loads(raw)
        return [x for x in items if isinstance(x, dict) and x.get("group") and x.get("role")]
    except Exception as e:
        logger.warning("[STS] 角色映射解析失败，回退到 default 角色: %s", e)
        return []


_ROLE_MAPPING = _parse_role_mapping()


def _role_arn_for_user(open_id: str) -> Optional[str]:
    """按用户所属 RAM 用户组选角色 ARN；查不到组时返回默认角色。"""
    if not _ROLE_MAPPING:
        return settings.ALIYUN_BOT_ROLE_DEFAULT or None

    try:
        from tools.aliyun.ram import get_ram_user_by_open_id
        ram_user = get_ram_user_by_open_id(open_id) or {}
        user_name = ram_user.get("user_name", "")
        if not user_name:
            return settings.ALIYUN_BOT_ROLE_DEFAULT or None

        groups = _list_user_groups(user_name)
        for mapping in _ROLE_MAPPING:
            if mapping["group"] in groups:
                return mapping["role"]
    except Exception as e:
        logger.warning("[STS] 查询用户组失败，回退到 default 角色: %s", e)
    return settings.ALIYUN_BOT_ROLE_DEFAULT or None


def _list_user_groups(ram_user_name: str) -> list[str]:
    """用 Master AK 调 RAM ListGroupsForUser。"""
    try:
        from alibabacloud_ram20150501.client import Client
        from alibabacloud_ram20150501 import models as ram_models
        from alibabacloud_tea_openapi import models as open_api_models

        cfg = open_api_models.Config(
            access_key_id=settings.ALIYUN_BOT_MASTER_AK_ID,
            access_key_secret=settings.ALIYUN_BOT_MASTER_AK_SECRET,
            endpoint="ram.aliyuncs.com",
        )
        client = Client(cfg)
        req = ram_models.ListGroupsForUserRequest(user_name=ram_user_name)
        resp = client.list_groups_for_user(req)
        return [g.group_name for g in (resp.body.groups.group or []) if g.group_name]
    except Exception as e:
        logger.debug("[STS] ListGroupsForUser 失败 user=%s: %s", ram_user_name, e)
        return []


# ── STS AssumeRole + Redis 缓存 ─────────────────────────────────────────────

_CACHE_PREFIX = "aliyun:sts:"
_RENEW_MARGIN_SECONDS = 5 * 60  # 过期前 5 分钟刷新


def _cache_key(open_id: str, role_arn: str) -> str:
    return f"{_CACHE_PREFIX}{open_id}:{role_arn}"


def _read_cache(open_id: str, role_arn: str) -> Optional[dict]:
    try:
        from utils.redis_client import get_redis
        raw = get_redis().get(_cache_key(open_id, role_arn))
        if not raw:
            return None
        cred = json.loads(raw)
        if float(cred.get("expire_ts", 0)) - time.time() < _RENEW_MARGIN_SECONDS:
            return None  # 即将过期，强制刷新
        return cred
    except Exception:
        return None


def _write_cache(open_id: str, role_arn: str, cred: dict) -> None:
    try:
        from utils.redis_client import get_redis
        ttl = max(60, int(float(cred["expire_ts"]) - time.time()))
        get_redis().setex(_cache_key(open_id, role_arn), ttl, json.dumps(cred))
    except Exception:
        pass


def _do_assume_role(role_arn: str, session_name: str) -> Optional[dict]:
    """实际调 STS AssumeRole。session_name 用 open_id（截断到 32 字符）。"""
    if not settings.ALIYUN_BOT_MASTER_AK_ID or not settings.ALIYUN_BOT_MASTER_AK_SECRET:
        logger.error("[STS] Master AK 未配置，无法 AssumeRole")
        return None

    try:
        from alibabacloud_sts20150401.client import Client
        from alibabacloud_sts20150401 import models as sts_models
        from alibabacloud_tea_openapi import models as open_api_models

        cfg = open_api_models.Config(
            access_key_id=settings.ALIYUN_BOT_MASTER_AK_ID,
            access_key_secret=settings.ALIYUN_BOT_MASTER_AK_SECRET,
            endpoint=f"sts.{settings.ALIYUN_STS_REGION_ID}.aliyuncs.com",
        )
        client = Client(cfg)
        req = sts_models.AssumeRoleRequest(
            role_arn=role_arn,
            role_session_name=session_name[:32] or "feishu-bot",
            duration_seconds=settings.ALIYUN_STS_DURATION_SECONDS,
        )
        resp = client.assume_role(req)
        c = resp.body.credentials
        return {
            "access_key_id":     c.access_key_id,
            "access_key_secret": c.access_key_secret,
            "security_token":    c.security_token,
            "expire_ts":         _iso_to_ts(c.expiration),
            "role_arn":          role_arn,
        }
    except Exception as e:
        logger.error("[STS] AssumeRole 失败 role=%s err=%s", role_arn, e, exc_info=True)
        return None


def assume_role_with_policy(
    role_arn: str,
    session_policy,
    duration_seconds: int,
    session_name: str = "temp-ak",
) -> Optional[dict]:
    """一次性 AssumeRole：传内联 session Policy 现场收窄会话权限（= role policy ∩ session policy），
    并自定 duration（900–43200s）。用于临时 AK 发放的 STS 分支（外部/卖数据，≤12h 自灭）。

    不走 (open_id, role_arn) 缓存——外部发放是一次性的，不缓存、不复用；与 assume_role_for_user 隔离。
    session_policy 可为 dict（自动 json 序列化）或已序列化的 str。失败返回 None。
    """
    if not settings.ALIYUN_BOT_MASTER_AK_ID or not settings.ALIYUN_BOT_MASTER_AK_SECRET:
        logger.error("[STS] Master AK 未配置，无法 AssumeRole(with policy)")
        return None
    if not role_arn:
        logger.error("[STS] assume_role_with_policy 缺 role_arn")
        return None
    try:
        from alibabacloud_sts20150401.client import Client
        from alibabacloud_sts20150401 import models as sts_models
        from alibabacloud_tea_openapi import models as open_api_models

        cfg = open_api_models.Config(
            access_key_id=settings.ALIYUN_BOT_MASTER_AK_ID,
            access_key_secret=settings.ALIYUN_BOT_MASTER_AK_SECRET,
            endpoint=f"sts.{settings.ALIYUN_STS_REGION_ID}.aliyuncs.com",
        )
        client = Client(cfg)
        duration = max(900, min(int(duration_seconds), 43200))
        policy_str = (json.dumps(session_policy, ensure_ascii=False)
                      if isinstance(session_policy, (dict, list)) else (session_policy or None))
        req = sts_models.AssumeRoleRequest(
            role_arn=role_arn,
            role_session_name=(session_name or "temp-ak")[:32],
            duration_seconds=duration,
            policy=policy_str,
        )
        resp = client.assume_role(req)
        c = resp.body.credentials
        return {
            "access_key_id":     c.access_key_id,
            "access_key_secret": c.access_key_secret,
            "security_token":    c.security_token,
            "expire_ts":         _iso_to_ts(c.expiration),
            "role_arn":          role_arn,
        }
    except Exception as e:
        logger.error("[STS] assume_role_with_policy 失败 role=%s err=%s", role_arn, e, exc_info=True)
        return None


def _iso_to_ts(iso_str: str) -> float:
    """阿里云返回的 expiration 是 '2025-01-01T12:00:00Z' 格式。"""
    from datetime import datetime, timezone
    try:
        return datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return time.time() + settings.ALIYUN_STS_DURATION_SECONDS


# ── 对外主接口 ──────────────────────────────────────────────────────────────

def assume_role_for_user(open_id: str) -> Optional[dict]:
    """
    为指定飞书用户换取临时凭证。
    返回 {access_key_id, access_key_secret, security_token, expire_ts, role_arn}
    失败返回 None。
    """
    role_arn = _role_arn_for_user(open_id)
    if not role_arn:
        logger.warning("[STS] open_id=%s 无可用角色 ARN", open_id)
        return None

    cached = _read_cache(open_id, role_arn)
    if cached:
        return cached

    cred = _do_assume_role(role_arn, session_name=open_id or "feishu-bot")
    if cred:
        _write_cache(open_id, role_arn, cred)
        logger.info("[STS] AssumeRole 成功 open_id=%s role=%s 有效期至=%s",
                    open_id, role_arn.rsplit("/", 1)[-1],
                    time.strftime("%H:%M:%S", time.localtime(cred["expire_ts"])))
    return cred


def invalidate_cache(open_id: str) -> None:
    """强制使指定用户的所有 STS 缓存失效（用于权限变更后）。"""
    try:
        from utils.redis_client import get_redis
        r = get_redis()
        for k in r.scan_iter(f"{_CACHE_PREFIX}{open_id}:*"):
            r.delete(k)
    except Exception:
        pass