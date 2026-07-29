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


def classify_mode(expire: float, now: Optional[float] = None, profile=None) -> str:
    """expire−now ≤ 上限（≤43200）→ sts；否则 → ram。

    **非默认账号一律 RAM，绝不 STS（fail-closed）**：STS 分支用的是全局 Master AK 去
    AssumeRole `settings.TEMP_AK_OSS_ROLE_ARN` —— 那是**现有账号**的宽 OSS 角色。若让第二账号的
    申请走这条，签出来的会是现有账号身份的凭证，申请人填一个现有账号的桶就能越权拿到其数据，
    正是多账号隔离要防的事。第二账号也没有、也不该有这样一个宽角色。
    不能只依赖线上把 TEMP_AK_STS_MAX_SECONDS 设成 0——那个配置的代码默认值是 43200（危险侧）。
    """
    from . import accounts
    slug = getattr(profile, "slug", accounts.DEFAULT_SLUG) if profile is not None else accounts.DEFAULT_SLUG
    if slug != accounts.DEFAULT_SLUG:
        return RAM_MODE
    now = now if now is not None else time.time()
    window = max(0.0, float(expire) - now)
    return STS_MODE if window <= _sts_limit() else RAM_MODE


def _sts_duration(expire: float, now: Optional[float] = None) -> int:
    now = now if now is not None else time.time()
    return max(900, min(int(float(expire) - now), _sts_limit()))


def plan(grant: dict) -> dict:
    """dry-run：只产计划、不调云。返回描述本次将如何发放的 dict（含 policy 预览）。"""
    mode = grant.get("mode") or classify_mode(grant["expire"], profile=_grant_profile(grant))
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
    mode = grant.get("mode") or classify_mode(grant["expire"], profile=_grant_profile(grant))
    if mode == STS_MODE:
        return _issue_sts(grant)
    return _issue_ram(grant)


def _issue_sts(grant: dict) -> dict:
    from utils import aliyun_sts
    from . import accounts
    # 纵深防御：即便上游 mode 判定被绕过/被畸形记录带进来，也绝不用现有账号的 Master AK +
    # 现有账号的角色 ARN 给别的账号签凭证（那会把 A 账号的数据权限发给 B 账号的申请人）。
    # **两套真相源都要看**：只看 grant["account"] 的话，`account 被抹掉 + grant_id 仍是
    # tak1949-` 这种畸形记录会通过账号门（auditor 实测复现过），与 RAM 路径的校验不对称。
    if accounts.assert_account_consistent(grant) != accounts.DEFAULT_SLUG:
        raise IssueError(
            f"账号 {grant.get('account')} 不支持 STS 单发（该分支只对默认主账号有效，"
            f"其角色 ARN 属于默认账号）；本单应走方案 B 长期 AK。这是隔离硬门，不要绕过。")
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
    client = permsync_client(grant)          # 按 grant 所属账号取 AK
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
    client = permsync_client(grant)          # 按 grant 所属账号取 AK
    doc = policy.build_policy_with_window(
        grant["bucket"], prefix=grant.get("prefix", ""), caps=grant.get("caps") or [],
        not_before=grant["not_before"], expire=grant["expire"],
        source_ips=grant.get("source_ips") or None)
    client.create_policy_version(m.CreatePolicyVersionRequest(
        policy_name=grant["policy_name"], policy_document=_json.dumps(doc, ensure_ascii=False),
        set_as_default=True, rotate_strategy="DeleteOldestNonDefaultVersionWhenLimitExceeded"))


def _grant_profile(grant: dict | None):
    """grant → 账号档案（供 mode 判定与凭证选择）。档案缺失时退默认档：mode 判定退默认只会
    **更保守**（默认档才可能判 STS），而真正的硬门在 _issue_sts 里按 grant["account"] 再拦一次。"""
    from . import accounts
    try:
        return accounts.by_slug((grant or {}).get("account", ""))
    except Exception:
        return accounts.default()


def permsync_client(grant: dict | None = None):
    """RAM 可写 AK 客户端。**按 grant 所属账号取该账号自己的 AK**。

    不传 grant（或 grant 无 account）时复用 oss_perm 那把 ALIYUN_ACCESS_KEY_*，与多账号化前一致。
    Master AK 只有 STS+RAMReadOnly，建不了号，所以这里不能用它。
    """
    from . import accounts
    slug = accounts.assert_account_consistent(grant or {})   # 两套账号真相源必须一致
    if slug != accounts.DEFAULT_SLUG:
        return accounts.ram_client(accounts.by_slug(slug))
    from core.oss_perm.permsync import make_ram_client
    return make_ram_client()


def _err(e) -> str:
    from core.oss_perm.permsync import _err_code
    return _err_code(e) or ""


class IssueError(RuntimeError):
    """凭证发放失败。"""
