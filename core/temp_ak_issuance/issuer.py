"""凭证生成：按有效期窗口分流（STS 单发 / 方案 B 建号建 AK），附时间窗 policy。

分流边界（用户拍板）：`expire − now ≤ TEMP_AK_STS_MAX_SECONDS`（默认 12h）→ STS；否则 → 方案 B。
用 expire−now（而非 expire−not_before）判：一张 STS token 从签发起最多活 12h、只覆盖 [now, now+12h]。
到期超 12h（跨天/跨周）单张 STS 覆盖不到 → 必须方案 B。未来生效由 policy DateGreaterThan 惰性化。

凭证（含 secret/token）只由 issue() 当次返回给下发层，绝不落 Redis/日志。
"""
from __future__ import annotations

import time
from typing import Optional

from config.settings import settings
from utils.logger import get_logger

from . import policy

logger = get_logger(__name__)

STS_MODE = "sts"
RAM_MODE = "ram"

STS_HARD_CAP = 43200   # 阿里 AssumeRole DurationSeconds 硬顶（12h）——误配的 TEMP_AK_STS_MAX_SECONDS 不得放大


def _sts_limit() -> int:
    """STS 分流/时长上限：配置值与硬顶 43200 取小，防误配 >12h 把长窗口误判 STS 后 token 提前死。"""
    return min(int(settings.TEMP_AK_STS_MAX_SECONDS), STS_HARD_CAP)


def classify_mode(expire: float, now: Optional[float] = None) -> str:
    """expire−now ≤ 上限（≤43200）→ sts；否则 → ram。"""
    now = now if now is not None else time.time()
    window = max(0.0, float(expire) - now)
    return STS_MODE if window <= _sts_limit() else RAM_MODE


def _sts_duration(expire: float, now: Optional[float] = None) -> int:
    now = now if now is not None else time.time()
    return max(900, min(int(float(expire) - now), _sts_limit()))


def plan(grant: dict) -> dict:
    """dry-run：只产计划、不调云。返回描述本次将如何发放的 dict（含 policy 预览）。"""
    mode = grant.get("mode") or classify_mode(grant["expire"])
    nb, exp = grant["not_before"], grant["expire"]
    src_ips = grant.get("source_ips") or None
    prefix = grant.get("prefix", "")
    caps = grant.get("caps") or []
    if mode == STS_MODE:
        doc = policy.build_session_policy(
            grant["bucket"], prefix=prefix, caps=caps,
            not_before=nb, expire=exp, source_ips=src_ips)
        return {"mode": STS_MODE, "duration_seconds": _sts_duration(exp),
                "role_arn": settings.TEMP_AK_OSS_ROLE_ARN, "policy": doc,
                "has_token": True}
    doc = policy.build_policy_with_window(
        grant["bucket"], prefix=prefix, caps=caps,
        not_before=nb, expire=exp, source_ips=src_ips)
    return {"mode": RAM_MODE, "user_name": grant["user_name"],
            "policy_name": grant["policy_name"], "policy": doc, "has_token": False}


def issue(grant: dict) -> dict:
    """真发放。返回 {access_key_id, access_key_secret, security_token, expire_ts, mode}。

    STS：assume_role_with_policy（含 token，到点自灭，无需清理）。
    方案 B：建 RAM user（无控制台/无组）+ 建 AK + 建/附时间窗 policy（到期由 cleanup 硬删）。
    """
    mode = grant.get("mode") or classify_mode(grant["expire"])
    if mode == STS_MODE:
        return _issue_sts(grant)
    return _issue_ram(grant)


def _issue_sts(grant: dict) -> dict:
    from utils import aliyun_sts
    if not settings.TEMP_AK_OSS_ROLE_ARN:
        raise IssueError("STS 分支缺 TEMP_AK_OSS_ROLE_ARN（宽 OSS 角色）")
    doc = policy.build_session_policy(
        grant["bucket"], prefix=grant.get("prefix", ""), caps=grant.get("caps") or [],
        not_before=grant["not_before"], expire=grant["expire"],
        source_ips=grant.get("source_ips") or None)
    cred = aliyun_sts.assume_role_with_policy(
        settings.TEMP_AK_OSS_ROLE_ARN, doc,
        _sts_duration(grant["expire"]), session_name=grant["grant_id"])
    if not cred:
        raise IssueError("STS AssumeRole 失败（见日志）")
    cred["mode"] = STS_MODE
    return cred


def _issue_ram(grant: dict) -> dict:
    from alibabacloud_ram20150501 import models as m
    client = permsync_client()
    user = grant["user_name"]
    pol_name = grant["policy_name"]

    # 1) 建 user（幂等：已存在则复用；不开控制台、不入组）
    try:
        client.get_user(m.GetUserRequest(user_name=user))
    except Exception as e:
        if _err(e) == "EntityNotExist.User":
            from . import orchestrator as _o
            client.create_user(m.CreateUserRequest(
                user_name=user, display_name=_o.display_name_for(grant),
                comments=(grant.get("reason") or "temp-ak external issuance")[:128] or None))
        else:
            raise

    # 2) 建/更新时间窗 policy（不存在建、存在则新增默认版本，rotate 清旧）
    doc = policy.build_policy_with_window(
        grant["bucket"], prefix=grant.get("prefix", ""), caps=grant.get("caps") or [],
        not_before=grant["not_before"], expire=grant["expire"],
        source_ips=grant.get("source_ips") or None)
    import json as _json
    doc_str = _json.dumps(doc, ensure_ascii=False)
    try:
        client.get_policy(m.GetPolicyRequest(policy_type="Custom", policy_name=pol_name))
        client.create_policy_version(m.CreatePolicyVersionRequest(
            policy_name=pol_name, policy_document=doc_str, set_as_default=True,
            rotate_strategy="DeleteOldestNonDefaultVersionWhenLimitExceeded"))
    except Exception as e:
        if _err(e) in ("EntityNotExist.Policy", "EntityNotExist.CustomPolicy"):
            client.create_policy(m.CreatePolicyRequest(
                policy_name=pol_name, policy_document=doc_str,
                description="temp-ak external issuance (time-boxed)"))
        else:
            raise

    # 3) 附加 policy（幂等）
    try:
        client.attach_policy_to_user(m.AttachPolicyToUserRequest(
            policy_type="Custom", policy_name=pol_name, user_name=user))
    except Exception as e:
        if _err(e) != "EntityAlreadyExists.User.Policy":
            raise

    # 4) 建 AK（外部拿这组长期 AK；到期由 policy 时间窗拒 + cleanup 硬删）
    resp = client.create_access_key(m.CreateAccessKeyRequest(user_name=user))
    ak = getattr(resp.body, "access_key", None)
    return {
        "access_key_id":     getattr(ak, "access_key_id", "") or "",
        "access_key_secret": getattr(ak, "access_key_secret", "") or "",
        "security_token":    "",
        "expire_ts":         float(grant["expire"]),
        "mode":              RAM_MODE,
    }


def rewrite_ram_window(grant: dict) -> None:
    """方案 B 延期：只改写自定义 policy 的时间窗（新版本设默认，rotate 清旧），AK/user 不变、不重发凭证。"""
    from alibabacloud_ram20150501 import models as m
    import json as _json
    if not grant.get("policy_name"):
        raise IssueError("方案 B 延期缺 policy_name")
    client = permsync_client()
    doc = policy.build_policy_with_window(
        grant["bucket"], prefix=grant.get("prefix", ""), caps=grant.get("caps") or [],
        not_before=grant["not_before"], expire=grant["expire"],
        source_ips=grant.get("source_ips") or None)
    client.create_policy_version(m.CreatePolicyVersionRequest(
        policy_name=grant["policy_name"], policy_document=_json.dumps(doc, ensure_ascii=False),
        set_as_default=True, rotate_strategy="DeleteOldestNonDefaultVersionWhenLimitExceeded"))


def permsync_client():
    """RAM 可写 AK 客户端（复用 oss_perm 那把 ALIYUN_ACCESS_KEY_*；Master AK 只有 STS+RAMReadOnly 建不了号）。"""
    from core.oss_perm.permsync import make_ram_client
    return make_ram_client()


def _err(e) -> str:
    from core.oss_perm.permsync import _err_code
    return _err_code(e) or ""


class IssueError(RuntimeError):
    """凭证发放失败。"""
