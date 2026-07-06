"""Feishu approval app integration for Aliyun RAM account creation."""
from __future__ import annotations

import json
import os
import re
import smtplib
import time
from dataclasses import dataclass, field, replace
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote

import requests

from config.settings import settings
from core.oss_perm.permsync import _err_code, make_ram_client
from tools.feishu.notify import _get_access_token
from utils.logger import get_logger
from utils.redis_client import get_redis

logger = get_logger(__name__)

FEISHU_BASE = "https://open.feishu.cn/open-apis"

PLATFORM_ALIYUN_RAM = "aliyun_ram"
PLATFORM_VOLCANO_IAM = "volcano_iam"
PLATFORM_LABELS = {
    PLATFORM_ALIYUN_RAM: "\u963f\u91cc\u4e91 RAM",
    PLATFORM_VOLCANO_IAM: "\u706b\u5c71\u5f15\u64ce IAM",
}

REDIS_INSTANCE_PREFIX = "ram_approval:instance:"
REDIS_LOCK_PREFIX = "ram_approval:lock:"
REDIS_INDEX_KEY = "ram_approval:instances"
REDIS_LOCK_TTL_SECONDS = 600

APPROVED_STATUSES = {
    "APPROVED",
    "APPROVE",
    "PASS",
    "PASSED",
    "AGREE",
    "AGREED",
    "COMPLETED",
}

FIELD_CONFIG = {
    "login_name": ("FEISHU_RAM_APPROVAL_FIELD_LOGIN_NAME", ("登录名称", "登录名", "RAM登录名称", "UserName", "user_name", "login_name")),
    "display_name": ("FEISHU_RAM_APPROVAL_FIELD_DISPLAY_NAME", ("显示名称", "显示名", "姓名", "DisplayName", "display_name")),
    "email": ("FEISHU_RAM_APPROVAL_FIELD_EMAIL", ("安全邮箱", "邮箱", "Email", "email")),
    "mobile": ("FEISHU_RAM_APPROVAL_FIELD_MOBILE", ("安全手机", "手机", "手机号", "MobilePhone", "mobile_phone")),
    "password": ("FEISHU_RAM_APPROVAL_FIELD_PASSWORD", ("登录密码", "密码", "Password", "password")),
    "confirm_password": ("FEISHU_RAM_APPROVAL_FIELD_CONFIRM_PASSWORD", ("确认登录密码", "确认密码", "ConfirmPassword", "confirm_password")),
    "groups": ("FEISHU_RAM_APPROVAL_FIELD_GROUPS", ("用户组", "RAM用户组", "GroupName", "groups")),
    "aliyun_groups": ("FEISHU_RAM_APPROVAL_FIELD_ALIYUN_GROUPS", ("\u963f\u91cc\u4e91RAM\u7528\u6237\u7ec4", "\u963f\u91cc\u4e91\u7528\u6237\u7ec4", "RAM\u7528\u6237\u7ec4", "AliyunGroups", "aliyun_groups")),
    "volcano_groups": ("FEISHU_RAM_APPROVAL_FIELD_VOLCANO_GROUPS", ("\u706b\u5c71\u5f15\u64ceIAM\u7528\u6237\u7ec4", "\u706b\u5c71IAM\u7528\u6237\u7ec4", "\u706b\u5c71\u7528\u6237\u7ec4", "VolcanoGroups", "volcano_groups")),
    "platforms": ("FEISHU_RAM_APPROVAL_FIELD_PLATFORMS", ("\u5e73\u53f0", "\u8d26\u53f7\u5e73\u53f0", "\u4e91\u5e73\u53f0", "\u521b\u5efa\u5e73\u53f0", "Platform", "platform", "platforms")),
    "console_access": ("FEISHU_RAM_APPROVAL_FIELD_CONSOLE_ACCESS", ("控制台访问", "启用控制台访问", "ConsoleAccess", "console_access")),
    "access_key": ("FEISHU_RAM_APPROVAL_FIELD_ACCESS_KEY", ("永久 AccessKey 访问", "永久AccessKey访问", "AccessKey 访问", "创建AccessKey", "AccessKey", "access_key")),
    "password_reset_required": ("FEISHU_RAM_APPROVAL_FIELD_PASSWORD_RESET", ("密码首次登录重置", "首次登录重置密码", "PasswordResetRequired", "password_reset_required")),
    "mfa_bind_required": ("FEISHU_RAM_APPROVAL_FIELD_MFA_REQUIRED", ("首次登录强制绑定 MFA", "首次登录强制绑定MFA", "强制绑定 MFA", "MFABindRequired", "mfa_bind_required")),
    "reason": ("FEISHU_RAM_APPROVAL_FIELD_REASON", ("申请原因", "用途", "原因", "reason")),
    "comments": ("FEISHU_RAM_APPROVAL_FIELD_COMMENTS", ("备注", "说明", "Comments", "comments")),
}


class RamApprovalError(RuntimeError):
    """Raised when an approval instance cannot be converted or applied."""


@dataclass(frozen=True)
class ApprovalFormValues:
    by_id: dict[str, Any] = field(default_factory=dict)
    by_name: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RamAccountRequest:
    login_name: str
    display_name: str = ""
    email: str = ""
    mobile_phone: str = ""
    password: str = ""
    confirm_password: str = ""
    groups: tuple[str, ...] = ()
    aliyun_groups: tuple[str, ...] = ()
    volcano_groups: tuple[str, ...] = ()
    platforms: tuple[str, ...] = (PLATFORM_ALIYUN_RAM,)
    console_access: bool = False
    permanent_access_key: bool = False
    password_reset_required: bool = False
    mfa_bind_required: bool = False
    reason: str = ""
    comments: str = ""
    requester_open_id: str = ""
    requester_user_id: str = ""
    approval_instance_code: str = ""


@dataclass
class RamAccountResult:
    user_name: str
    platform: str = PLATFORM_ALIYUN_RAM
    platform_label: str = PLATFORM_LABELS[PLATFORM_ALIYUN_RAM]
    account_id: str = ""
    created_user: bool = False
    updated_user_profile: bool = False
    set_security_phone: bool = False
    set_security_email: bool = False
    created_login_profile: bool = False
    created_access_key: bool = False
    added_groups: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    access_key_id: str = ""
    access_key_secret: str = ""
    login_principal: str = ""
    login_url: str = ""
    platform_results: list["RamAccountResult"] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return True



def event_log_summary(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "event_type": _event_type(payload),
        "approval_code": _extract_approval_code(payload),
        "instance_code": _extract_instance_code(payload),
        "status": _extract_status(payload),
    }


def should_handle_event(payload: dict[str, Any]) -> bool:
    """Return whether this event looks like the configured RAM approval flow."""
    event_type = _event_type(payload).lower()
    if "approval" not in event_type and not _extract_approval_code(payload):
        return False
    target = getattr(settings, "FEISHU_RAM_APPROVAL_CODE", "")
    code = _extract_approval_code(payload)
    return bool(target and (not code or code == target))


def handle_approval_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Process a Feishu approval status-change event in a background thread."""
    if not should_handle_event(payload):
        return {"ignored": True, "reason": "not_ram_approval"}

    status = (_extract_status(payload) or "").upper()
    if status and status not in APPROVED_STATUSES:
        logger.info("[ram_approval] ignore non-approved status=%s", status)
        return {"ignored": True, "reason": f"status={status}"}

    instance_code = _extract_instance_code(payload)
    if not status and not instance_code:
        # 无终态状态、也无审批实例：多为「抄送(cc)」或流程里新增审批节点产生的中间/通知事件。
        # 这类事件不可用于建号，静默忽略（避免误报"审批实例: - / status is missing"失败卡）。
        logger.info("[ram_approval] ignore non-actionable approval event (no status & no instance)")
        return {"ignored": True, "reason": "no_status_no_instance"}
    if instance_code and _is_instance_done(instance_code):
        _mark_duplicate(instance_code, "already_processed")
        return {"ignored": True, "reason": "already_processed", "instance_code": instance_code}

    lock_key = _claim_instance(instance_code)
    if instance_code and not lock_key:
        _mark_duplicate(instance_code, "already_processing")
        return {"ignored": True, "reason": "already_processing", "instance_code": instance_code}

    req: RamAccountRequest | None = None
    approval_comment_id = ""
    try:
        detail = fetch_approval_instance(instance_code) if instance_code else _event_obj(payload)
        status = status or (_extract_status(detail) or "").upper()
        if not status:
            # 拉取实例后仍无状态：非终态/通知类事件，不可执行建号 → 静默忽略（不再误报失败）。
            logger.info("[ram_approval] no status resolved for instance=%s; ignore", instance_code or "-")
            return {"ignored": True, "reason": "no_status", "instance_code": instance_code}
        if status not in APPROVED_STATUSES:
            logger.info("[ram_approval] ignore non-approved status=%s", status)
            return {"ignored": True, "reason": f"status={status}"}

        target = getattr(settings, "FEISHU_RAM_APPROVAL_CODE", "")
        code = _extract_approval_code(detail) or _extract_approval_code(payload)
        if target and code and code != target:
            return {"ignored": True, "reason": "approval_code_mismatch"}

        req = parse_ram_account_request(detail, event_payload=payload)
        save_approval_record(req, detail=detail, event_payload=payload, result_status="processing")

        _assert_account_delivery_ready(req)
        # 处理中通知为 best-effort：评论/邮件失败绝不阻断建号（建号才是主目标）
        try:
            approval_comment_id = notify_processing(req)
        except Exception:
            logger.warning("[ram_approval] notify_processing failed (non-blocking)", exc_info=True)
            approval_comment_id = ""

        if getattr(settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False):
            result = dry_run_account_result(req)
            save_approval_result(req, result, result_status="dry_run")
            _safe_notify_result(req, result, approval_comment_id=approval_comment_id)
            return {"ignored": False, "dry_run": True, "user_name": result.user_name, "platforms": list(req.platforms)}

        result = create_accounts_for_platforms(req)
        save_approval_result(req, result, result_status="success")
        _safe_notify_result(req, result, approval_comment_id=approval_comment_id)
        return {"ignored": False, "user_name": result.user_name, "platforms": list(req.platforms)}
    except Exception as exc:
        safe_instance = instance_code or (req.approval_instance_code if req else "")
        save_approval_failure(safe_instance, exc, req=req, event_payload=payload)
        logger.error("[ram_approval] apply failed instance=%s", safe_instance, exc_info=True)
        notify_failure(safe_instance, exc, req=req, approval_comment_id=approval_comment_id)
        return {"ignored": False, "error": str(exc)}
    finally:
        if lock_key:
            _release_instance_lock(lock_key)

def fetch_approval_instance(instance_code: str) -> dict[str, Any]:
    if not instance_code:
        raise RamApprovalError("缺少审批 instance_code")
    token = _get_access_token()
    resp = requests.get(
        f"{FEISHU_BASE}/approval/v4/instances/{instance_code}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RamApprovalError(f"获取飞书审批实例失败: {data.get('msg') or data}")
    return data.get("data") or {}


def parse_ram_account_request(instance_detail: dict[str, Any], event_payload: dict[str, Any] | None = None) -> RamAccountRequest:
    values = extract_form_values(instance_detail)

    login_name = _field_text(values, "login_name")
    display_name = _field_text(values, "display_name") or login_name
    password = _field_text(values, "password")
    confirm_password = _field_text(values, "confirm_password")
    email = _normalize_email(_field_text(values, "email"))
    mobile = _normalize_mobile(_field_text(values, "mobile"))
    platforms = tuple(_normalize_platforms(_field_any(values, "platforms")))
    groups = tuple(_normalize_groups(_field_any(values, "groups"), _allowed_groups_for_platforms(platforms)))
    aliyun_groups = tuple(_normalize_groups(_field_any(values, "aliyun_groups"), _allowed_groups(PLATFORM_ALIYUN_RAM)))
    volcano_groups = tuple(_normalize_groups(_field_any(values, "volcano_groups"), _allowed_groups(PLATFORM_VOLCANO_IAM)))
    requester_open_id, requester_user_id = _extract_requester_ids(instance_detail, event_payload or {})

    req = RamAccountRequest(
        login_name=login_name,
        display_name=display_name,
        email=email,
        mobile_phone=mobile,
        password=password,
        confirm_password=confirm_password,
        groups=groups,
        aliyun_groups=aliyun_groups,
        volcano_groups=volcano_groups,
        platforms=platforms,
        console_access=_field_bool(values, "console_access", default=False),
        permanent_access_key=_field_bool(values, "access_key", default=False),
        password_reset_required=_field_bool(values, "password_reset_required", default=False),
        mfa_bind_required=_field_bool(values, "mfa_bind_required", default=False),
        reason=_field_text(values, "reason"),
        comments=_field_text(values, "comments"),
        requester_open_id=requester_open_id,
        requester_user_id=requester_user_id,
        approval_instance_code=_extract_instance_code(instance_detail) or _extract_instance_code(event_payload or {}),
    )
    validate_request(req)
    return req


def extract_form_values(instance_detail: dict[str, Any]) -> ApprovalFormValues:
    by_id: dict[str, Any] = {}
    by_name: dict[str, Any] = {}
    for item in _iter_form_items(instance_detail):
        field_id = _first_text(item, ("id", "widget_id", "field_id", "custom_id"))
        field_name = _first_text(item, ("name", "title", "label"))
        if "value" in item:
            value = _load_json_maybe(item.get("value"))
        elif "option" in item:
            value = _load_json_maybe(item.get("option"))
        elif "text" in item:
            value = item.get("text")
        else:
            continue
        if field_id:
            by_id[field_id] = value
        if field_name:
            by_name[field_name] = value
    return ApprovalFormValues(by_id=by_id, by_name=by_name)


def validate_request(req: RamAccountRequest) -> None:
    if not req.login_name:
        raise RamApprovalError("审批表单缺少登录名称")
    if not re.fullmatch(r"[A-Za-z0-9.@_-]{1,64}", req.login_name):
        raise RamApprovalError("登录名称只能包含字母、数字、'.'、'_'、'-'、'@'，且长度不超过 64")
    if req.console_access:
        if not req.password:
            raise RamApprovalError("已开启控制台访问，但缺少登录密码")
        if req.confirm_password and req.password != req.confirm_password:
            raise RamApprovalError("登录密码和确认登录密码不一致")
    if not req.email:
        raise RamApprovalError("审批表单缺少安全邮箱，未创建 RAM 用户")
    if not req.mobile_phone:
        raise RamApprovalError("审批表单缺少安全手机，未创建 RAM 用户")
    for platform in req.platforms or (PLATFORM_ALIYUN_RAM,):
        allowed = _allowed_groups(platform)
        if not allowed:
            continue
        selected_groups = _groups_for_platform(req, platform)
        invalid = [g for g in selected_groups if g not in allowed]
        if invalid:
            label = PLATFORM_LABELS.get(platform, platform)
            raise RamApprovalError(f"{label} 用户组不在允许范围内: {', '.join(invalid)}")


def create_ram_account(
    req: RamAccountRequest,
    client: Any | None = None,
    verification_client: Any | None = None,
    set_verification_info: bool = True,
) -> RamAccountResult:
    from alibabacloud_ram20150501 import models as m

    client = client or make_ram_client()
    result = RamAccountResult(user_name=req.login_name)
    result.platform = PLATFORM_ALIYUN_RAM
    result.platform_label = PLATFORM_LABELS[PLATFORM_ALIYUN_RAM]
    result.login_principal = build_login_principal(req.login_name)
    result.login_url = build_login_url(req.login_name)

    existing_user = None
    try:
        existing_resp = client.get_user(m.GetUserRequest(user_name=req.login_name))
        existing_user = getattr(getattr(existing_resp, "body", None), "user", None)
        result.skipped.append("RAM 用户已存在")
    except Exception as exc:
        if _is_not_exist(exc, "User"):
            create_req = m.CreateUserRequest(
                user_name=req.login_name,
                display_name=req.display_name or req.login_name,
                email=req.email or None,
                mobile_phone=req.mobile_phone or None,
                comments=_comments(req) or None,
            )
            try:
                client.create_user(create_req)
                result.created_user = True
            except Exception as create_exc:
                if _is_already_exists(create_exc):
                    result.skipped.append("RAM 用户已存在")
                else:
                    raise
        else:
            raise

    if _ensure_user_profile(client, req, existing_user, force=result.created_user):
        result.updated_user_profile = True

    if set_verification_info:
        result.set_security_phone, result.set_security_email = _set_verification_info(
            req, verification_client=verification_client
        )

    if req.console_access:
        try:
            client.get_login_profile(m.GetLoginProfileRequest(user_name=req.login_name))
            result.skipped.append("控制台登录配置已存在")
        except Exception as exc:
            if _is_not_exist(exc, "LoginProfile"):
                try:
                    client.create_login_profile(m.CreateLoginProfileRequest(
                        user_name=req.login_name,
                        password=req.password,
                        password_reset_required=req.password_reset_required,
                        mfabind_required=req.mfa_bind_required,
                    ))
                    result.created_login_profile = True
                except Exception as create_exc:
                    if _is_already_exists(create_exc):
                        result.skipped.append("控制台登录配置已存在")
                    else:
                        raise
            else:
                raise

    for group in req.groups:
        try:
            client.add_user_to_group(m.AddUserToGroupRequest(user_name=req.login_name, group_name=group))
            result.added_groups.append(group)
        except Exception as exc:
            if _is_already_exists(exc) or "Exist" in (_err_code(exc) or ""):
                result.skipped.append(f"已在用户组 {group}")
            else:
                raise

    if req.permanent_access_key:
        existing = _list_access_key_ids(client, req.login_name)
        if existing:
            result.access_key_id = existing[0]
            result.skipped.append("AccessKey 已存在，未重复创建")
        else:
            resp = client.create_access_key(m.CreateAccessKeyRequest(user_name=req.login_name))
            ak = getattr(resp.body, "access_key", None)
            result.access_key_id = getattr(ak, "access_key_id", "") or ""
            result.access_key_secret = getattr(ak, "access_key_secret", "") or ""
            result.created_access_key = True

    return result



def _groups_for_platform(req: RamAccountRequest, platform: str) -> tuple[str, ...]:
    if platform == PLATFORM_ALIYUN_RAM:
        return req.aliyun_groups or req.groups
    if platform == PLATFORM_VOLCANO_IAM:
        return req.volcano_groups or req.groups
    return req.groups


def _platform_group_map(req: RamAccountRequest) -> dict[str, list[str]]:
    return {
        platform: list(_groups_for_platform(req, platform))
        for platform in req.platforms or (PLATFORM_ALIYUN_RAM,)
    }


def _request_for_platform(req: RamAccountRequest, platform: str) -> RamAccountRequest:
    return replace(req, groups=_groups_for_platform(req, platform), platforms=(platform,))


def create_accounts_for_platforms(req: RamAccountRequest) -> RamAccountResult:
    """Create the requested account on every selected platform."""
    results: list[RamAccountResult] = []
    for platform in req.platforms or (PLATFORM_ALIYUN_RAM,):
        if platform == PLATFORM_ALIYUN_RAM:
            results.append(create_ram_account(_request_for_platform(req, platform)))
        elif platform == PLATFORM_VOLCANO_IAM:
            results.append(create_volcano_iam_account(_request_for_platform(req, platform)))
        else:
            raise RamApprovalError(f"unsupported platform: {platform}")
    if len(results) == 1:
        return results[0]

    aggregate = RamAccountResult(
        user_name=req.login_name,
        platform="multi",
        platform_label="\u591a\u5e73\u53f0",
        platform_results=results,
    )
    aggregate.created_user = any(r.created_user for r in results)
    aggregate.updated_user_profile = any(r.updated_user_profile for r in results)
    aggregate.set_security_phone = any(r.set_security_phone for r in results)
    aggregate.set_security_email = any(r.set_security_email for r in results)
    aggregate.created_login_profile = any(r.created_login_profile for r in results)
    aggregate.created_access_key = any(r.created_access_key for r in results)
    aggregate.added_groups = [f"{r.platform_label}:{g}" for r in results for g in r.added_groups]
    aggregate.skipped = [f"{r.platform_label}:{s}" for r in results for s in r.skipped]
    return aggregate


def dry_run_account_result(req: RamAccountRequest) -> RamAccountResult:
    results: list[RamAccountResult] = []
    for platform in req.platforms or (PLATFORM_ALIYUN_RAM,):
        result = RamAccountResult(
            user_name=req.login_name,
            platform=platform,
            platform_label=PLATFORM_LABELS.get(platform, platform),
        )
        if platform == PLATFORM_ALIYUN_RAM:
            result.login_principal = build_login_principal(req.login_name)
            result.login_url = build_login_url(req.login_name)
        elif platform == PLATFORM_VOLCANO_IAM:
            result.login_principal = req.login_name
            result.login_url = build_volcano_login_url(req.login_name)
        result.skipped.append("dry_run_no_create")
        results.append(result)
    if len(results) == 1:
        return results[0]
    return RamAccountResult(
        user_name=req.login_name,
        platform="multi",
        platform_label="\u591a\u5e73\u53f0",
        platform_results=results,
        skipped=[f"{r.platform_label}:dry_run_no_create" for r in results],
    )


def create_volcano_iam_account(
    req: RamAccountRequest,
    client: Any | None = None,
    models: Any | None = None,
) -> RamAccountResult:
    """Create a Volcano Engine IAM user using the official volcengine SDK."""
    client = client or make_volcano_iam_client()
    models = models or _volcano_iam_models()
    result = RamAccountResult(
        user_name=req.login_name,
        platform=PLATFORM_VOLCANO_IAM,
        platform_label=PLATFORM_LABELS[PLATFORM_VOLCANO_IAM],
        login_principal=req.login_name,
        login_url=build_volcano_login_url(req.login_name),
    )

    existing_user = None
    try:
        resp = client.get_user(models.GetUserRequest(user_name=req.login_name))
        existing_user = getattr(resp, "user", None)
        result.account_id = getattr(existing_user, "account_id", "") or ""
        result.skipped.append("volcano_iam_user_exists")
    except Exception as exc:
        if _volcano_is_not_exist(exc):
            resp = client.create_user(models.CreateUserRequest(
                user_name=req.login_name,
                display_name=req.display_name or req.login_name,
                email=req.email or None,
                mobile_phone=_volcano_mobile(req.mobile_phone) or None,
                description=_comments(req) or None,
            ))
            user = getattr(resp, "user", None)
            result.account_id = getattr(user, "account_id", "") or ""
            result.created_user = True
        elif _volcano_is_already_exists(exc):
            result.skipped.append("volcano_iam_user_exists")
        else:
            raise

    if _ensure_volcano_user_profile(client, models, req, existing_user, force=result.created_user):
        result.updated_user_profile = True

    if req.console_access:
        # 火山 get_login_profile 对不存在的 profile 也返回默认桩(login_allowed=False)、不报错，
        # 不能靠它判断存在与否。直接 create 开启登录；已存在则 update 打开 login_allowed。
        try:
            client.create_login_profile(models.CreateLoginProfileRequest(
                user_name=req.login_name,
                password=req.password,
                password_reset_required=req.password_reset_required,
                login_allowed=True,
            ))
            result.created_login_profile = True
        except Exception as exc:
            if _volcano_is_already_exists(exc):
                client.update_login_profile(models.UpdateLoginProfileRequest(
                    user_name=req.login_name,
                    password=req.password or None,
                    password_reset_required=req.password_reset_required,
                    login_allowed=True,
                ))
                result.created_login_profile = True
            else:
                raise

    for group in req.groups:
        try:
            client.add_user_to_group(models.AddUserToGroupRequest(
                user_name=req.login_name,
                user_group_name=group,
            ))
            result.added_groups.append(group)
        except Exception as exc:
            if _volcano_is_already_exists(exc):
                result.skipped.append(f"volcano_group_exists:{group}")
            else:
                raise

    if req.permanent_access_key:
        existing = _list_volcano_access_key_ids(client, models, req.login_name)
        if existing:
            result.access_key_id = existing[0]
            result.skipped.append("volcano_access_key_exists")
        else:
            resp = client.create_access_key(models.CreateAccessKeyRequest(user_name=req.login_name))
            ak = getattr(resp, "access_key", None)
            result.access_key_id = getattr(ak, "access_key_id", "") or ""
            result.access_key_secret = getattr(ak, "secret_access_key", "") or ""
            result.created_access_key = True

    return result


def make_volcano_iam_client() -> Any:
    ak = getattr(settings, "VOLCANO_ACCESS_KEY", "") or getattr(settings, "TOS_ACCESS_KEY", "")
    sk = getattr(settings, "VOLCANO_SECRET_KEY", "") or getattr(settings, "TOS_SECRET_KEY", "")
    if not ak or not sk:
        raise RamApprovalError("missing Volcano IAM AccessKey")
    try:
        import volcenginesdkcore as core
        from volcenginesdkiam.api.iam_api import IAMApi
    except ImportError as exc:
        raise RamApprovalError("volcengine-python-sdk is not installed") from exc
    cfg = core.Configuration()
    cfg.ak = ak
    cfg.sk = sk
    cfg.region = getattr(settings, "VOLCANO_IAM_REGION", "") or getattr(settings, "VOLCANO_REGION", "") or "cn-beijing"
    return IAMApi(core.ApiClient(cfg))


def _volcano_iam_models() -> Any:
    import volcenginesdkiam.models as models
    return models


def _ensure_volcano_user_profile(
    client: Any,
    models: Any,
    req: RamAccountRequest,
    existing_user: Any | None,
    force: bool = False,
) -> bool:
    if existing_user is None and not force:
        return False
    updates: dict[str, Any] = {}
    desired_display = req.display_name or req.login_name
    if desired_display and (force or desired_display != getattr(existing_user, "display_name", "")):
        updates["new_display_name"] = desired_display
    if req.email and (force or req.email != getattr(existing_user, "email", "")):
        updates["new_email"] = req.email
    vmobile = _volcano_mobile(req.mobile_phone)
    if vmobile and (force or vmobile != getattr(existing_user, "mobile_phone", "")):
        updates["new_mobile_phone"] = vmobile
    comments = _comments(req)
    if comments and (force or comments != getattr(existing_user, "description", "")):
        updates["new_description"] = comments
    if not updates:
        return False
    client.update_user(models.UpdateUserRequest(
        user_name=req.login_name,
        new_user_name=req.login_name,
        **updates,
    ))
    return True


def _list_volcano_access_key_ids(client: Any, models: Any, user_name: str) -> list[str]:
    resp = client.list_access_keys(models.ListAccessKeysRequest(user_name=user_name))
    keys = getattr(resp, "access_key_metadata", None) or []
    return [getattr(k, "access_key_id", "") for k in keys if getattr(k, "access_key_id", "")]


def _volcano_err_code(exc: Exception) -> str:
    # 优先读响应 body 里的 Error.Code/Message（如 UserNotExist），比 HTTP status(404) 更有意义。
    body = getattr(exc, "body", "") or ""
    if body:
        try:
            data = json.loads(body)
            found = _deep_first(data, ("Code", "code", "ErrorCode", "error_code", "Message", "message"))
            if found:
                return found
        except Exception:
            pass
    for attr in ("code", "error_code", "reason", "status"):
        value = getattr(exc, attr, "")
        if value:
            return str(value)
    if body:
        return str(body)
    return str(exc)


def _volcano_mobile(raw: str) -> str:
    """火山 IAM 手机号格式：去掉阿里式 '86-' 前缀/横杠，保留纯 11 位号码（否则报 InvalidMobilePhone）。"""
    digits = "".join(ch for ch in (raw or "") if ch.isdigit())
    if not digits:
        return ""
    if digits.startswith("86") and len(digits) > 11:
        digits = digits[-11:]
    return digits


def _volcano_is_not_exist(exc: Exception) -> bool:
    # 同时看 err_code 和完整异常串（body 里常含 "does not exist"）。
    text = (_volcano_err_code(exc) + " " + str(exc)).lower()
    return any(token in text for token in
               ("notfound", "not_found", "notexist", "not exist", "no such", "does not exist"))


def _volcano_is_already_exists(exc: Exception) -> bool:
    text = _volcano_err_code(exc).lower()
    return ("already" in text or "exist" in text or "conflict" in text) and not _volcano_is_not_exist(exc)


def build_volcano_login_url(user_name: str = "") -> str:
    return getattr(settings, "VOLCANO_IAM_LOGIN_URL", "") or "https://console.volcengine.com/auth/login/"


def _ensure_user_profile(
    client: Any,
    req: RamAccountRequest,
    existing_user: Any | None,
    force: bool = False,
) -> bool:
    """Sync RAM user profile fields after create or for an existing user."""
    from alibabacloud_ram20150501 import models as m

    if existing_user is None and not force:
        return False
    updates: dict[str, Any] = {}
    desired_display = req.display_name or req.login_name
    if desired_display and (force or desired_display != getattr(existing_user, "display_name", "")):
        updates["new_display_name"] = desired_display
    if req.email and (force or req.email != getattr(existing_user, "email", "")):
        updates["new_email"] = req.email
    if req.mobile_phone and (force or req.mobile_phone != getattr(existing_user, "mobile_phone", "")):
        updates["new_mobile_phone"] = req.mobile_phone
    comments = _comments(req)
    if comments and (force or comments != getattr(existing_user, "comments", "")):
        updates["new_comments"] = comments
    if not updates:
        return False
    client.update_user(m.UpdateUserRequest(
        user_name=req.login_name,
        new_user_name=req.login_name,
        **updates,
    ))
    return True


def _set_verification_info(
    req: RamAccountRequest,
    verification_client: Any | None = None,
) -> tuple[bool, bool]:
    """Set RAM user's MFA verification phone/email through IMS OpenAPI."""
    user_principal_name = build_login_principal(req.login_name)
    if not user_principal_name:
        return False, False

    phone_set = False
    email_set = False
    if req.mobile_phone:
        _call_ims_api(
            "SetVerificationInfo",
            {
                "UserPrincipalName": user_principal_name,
                "VerifyType": "sms",
                "MobilePhone": req.mobile_phone,
            },
            client=verification_client,
        )
        phone_set = True
    if req.email:
        _call_ims_api(
            "SetVerificationInfo",
            {
                "UserPrincipalName": user_principal_name,
                "VerifyType": "email",
                "Email": req.email,
            },
            client=verification_client,
        )
        email_set = True
    return phone_set, email_set


def _call_ims_api(action: str, query: dict[str, Any], client: Any | None = None) -> dict[str, Any]:
    from alibabacloud_tea_openapi import utils_models as open_api_util_models
    from alibabacloud_tea_openapi.utils import Utils
    from darabonba.runtime import RuntimeOptions

    ims_client = client or _make_ims_client()
    clean_query = {k: v for k, v in query.items() if v is not None and v != ""}
    request = open_api_util_models.OpenApiRequest(query=Utils.query(clean_query))
    params = open_api_util_models.Params(
        action=action,
        version="2019-08-15",
        protocol="HTTPS",
        pathname="/",
        method="POST",
        auth_type="AK",
        style="RPC",
        req_body_type="formData",
        body_type="json",
    )
    return ims_client.call_api(params, request, RuntimeOptions())


def _make_ims_client() -> Any:
    from alibabacloud_tea_openapi.client import Client as OpenApiClient
    from alibabacloud_tea_openapi import models as open_api_models

    ak = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID") or settings.ALIYUN_ACCESS_KEY_ID
    sk = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or settings.ALIYUN_ACCESS_KEY_SECRET
    if not ak or not sk:
        raise RuntimeError("缺少 IMS/RAM AccessKey，无法设置 MFA 安全手机/安全邮箱")
    cfg = open_api_models.Config(access_key_id=ak, access_key_secret=sk)
    cfg.endpoint = "ims.aliyuncs.com"
    return OpenApiClient(cfg)


def build_login_principal(user_name: str) -> str:
    """Return RAM console login principal, e.g. user@domain.onaliyun.com."""
    user_name = (user_name or "").strip()
    if not user_name:
        return ""
    if "@" in user_name:
        return user_name
    domain = _login_domain()
    return f"{user_name}@{domain}" if domain else user_name


def build_login_url(user_name: str) -> str:
    principal = build_login_principal(user_name)
    if not principal:
        return ""
    return "https://signin.aliyun.com/login.htm?username=" + quote(principal, safe="") + "&defaultShowQrCode=false#/main"


def _login_domain() -> str:
    return (
        getattr(settings, "ALIYUN_RAM_LOGIN_DOMAIN", "")
        or getattr(settings, "ALIYUN_RAM_DEFAULT_DOMAIN", "")
        or ""
    ).strip().lstrip("@")

def notify_processing(req: RamAccountRequest) -> str:
    if _delivery_mode() not in {"approval_comment", "comment"}:
        return ""
    text = "\n".join([
        "\u4e91\u8d26\u53f7\u5ba1\u6279\u5df2\u901a\u8fc7\uff0c\u5f00\u59cb\u81ea\u52a8\u521b\u5efa\u8d26\u53f7\u3002",
        f"\u5ba1\u6279\u5b9e\u4f8b: {req.approval_instance_code or '-'}",
        f"\u767b\u5f55\u540d\u79f0: {req.login_name}",
        "\u5e73\u53f0: " + ", ".join(PLATFORM_LABELS.get(p, p) for p in (req.platforms or (PLATFORM_ALIYUN_RAM,))),
        "\u72b6\u6001: processing",
    ])
    return _send_approval_comment(req.approval_instance_code, text, _approval_comment_user_id(req))


def _safe_notify_result(
    req: RamAccountRequest,
    result: RamAccountResult,
    *,
    approval_comment_id: str = "",
) -> None:
    try:
        notify_result(req, result, approval_comment_id=approval_comment_id)
    except Exception:
        logger.error("[ram_approval] result notification failed", exc_info=True)


def notify_result(
    req: RamAccountRequest,
    result: RamAccountResult,
    *,
    approval_comment_id: str = "",
) -> None:
    if _delivery_mode() in {"approval_comment", "comment"}:
        _send_approval_comment(
            req.approval_instance_code,
            _account_delivery_text(req, result),
            _approval_comment_user_id(req),
            comment_id=approval_comment_id,
        )
        return

    if _should_send_account_email(req, result):
        _send_account_email(req, result)


def _assert_account_delivery_ready(req: RamAccountRequest) -> None:
    if getattr(settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False):
        return
    if not (req.console_access or req.permanent_access_key):
        return
    if _delivery_mode() in {"approval_comment", "comment"}:
        if not req.approval_instance_code:
            raise RamApprovalError("审批实例 Code 为空，未创建 RAM 用户")
        if not _approval_comment_user_id(req):
            raise RamApprovalError("审批评论用户 ID 未配置，未创建 RAM 用户")
        return
    if _delivery_mode() != "email":
        return
    if not req.email:
        raise RamApprovalError("账号信息邮件收件人为空，未创建 RAM 用户")
    missing = _missing_smtp_settings()
    if missing:
        raise RamApprovalError("账号信息邮件发送未配置，未创建 RAM 用户: " + ", ".join(missing))


def _delivery_mode() -> str:
    return (getattr(settings, "FEISHU_RAM_APPROVAL_DELIVERY", "approval_comment") or "approval_comment").strip().lower()


def _missing_smtp_settings() -> list[str]:
    required = [
        ("SMTP_HOST", getattr(settings, "SMTP_HOST", "")),
        ("SMTP_FROM", getattr(settings, "SMTP_FROM", "")),
    ]
    if getattr(settings, "SMTP_AUTH_REQUIRED", True):
        required.extend([
            ("SMTP_USERNAME", getattr(settings, "SMTP_USERNAME", "")),
            ("SMTP_PASSWORD", getattr(settings, "SMTP_PASSWORD", "")),
        ])
    return [key for key, value in required if not value]


def _should_send_account_email(req: RamAccountRequest, result: RamAccountResult) -> bool:
    if _delivery_mode() != "email":
        return False
    if "dry_run_no_create" in result.skipped:
        return False
    return bool(req.email and (req.console_access or req.permanent_access_key))


def _send_account_email(req: RamAccountRequest, result: RamAccountResult) -> None:
    missing = _missing_smtp_settings()
    if missing:
        raise RamApprovalError("账号信息邮件发送未配置: " + ", ".join(missing))
    msg = EmailMessage()
    msg["Subject"] = f"阿里云 RAM 子账号信息 - {result.login_principal or build_login_principal(req.login_name)}"
    msg["From"] = getattr(settings, "SMTP_FROM", "")
    msg["To"] = req.email
    reply_to = getattr(settings, "SMTP_REPLY_TO", "")
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(_account_delivery_text(req, result))

    host = getattr(settings, "SMTP_HOST", "")
    port = int(getattr(settings, "SMTP_PORT", 465 if getattr(settings, "SMTP_USE_SSL", False) else 587))
    timeout = int(getattr(settings, "SMTP_TIMEOUT_SECONDS", 20))
    client_cls = smtplib.SMTP_SSL if getattr(settings, "SMTP_USE_SSL", False) else smtplib.SMTP
    with client_cls(host, port, timeout=timeout) as smtp:
        if not getattr(settings, "SMTP_USE_SSL", False) and getattr(settings, "SMTP_USE_TLS", True):
            smtp.starttls()
        username = getattr(settings, "SMTP_USERNAME", "")
        password = getattr(settings, "SMTP_PASSWORD", "")
        if username or password:
            smtp.login(username, password)
        smtp.send_message(msg)


def _account_delivery_text(req: RamAccountRequest, result: RamAccountResult) -> str:
    results = result.platform_results or [result]
    sections: list[str] = []
    for item in results:
        login_principal = item.login_principal or (
            build_login_principal(req.login_name) if item.platform == PLATFORM_ALIYUN_RAM else req.login_name
        )
        login_url = item.login_url or (
            build_login_url(req.login_name) if item.platform == PLATFORM_ALIYUN_RAM else build_volcano_login_url(req.login_name)
        )
        access_key_id = item.access_key_id or "-"
        access_key_secret = item.access_key_secret or "-"
        password = req.password if req.console_access else "-"
        item_groups = _groups_for_platform(req, item.platform)
        sections.append("\n".join([
            f"{item.platform_label} \u8d26\u53f7\u4fe1\u606f",
            "",
            "\u767b\u5f55\u5730\u5740",
            login_url or "-",
            "\u767b\u5f55\u8d26\u53f7",
            login_principal or item.user_name,
            "\u767b\u5f55\u5bc6\u7801",
            password,
            "AccessKey ID",
            access_key_id,
            "AccessKey Secret",
            access_key_secret,
            "SecurityPhoneDevice",
            req.mobile_phone or "-",
            "SecurityEmailDevice",
            req.email or "-",
            "\u7528\u6237\u7ec4",
            ", ".join(item_groups) if item_groups else "-",
            "\u521b\u5efa\u7ed3\u679c",
            _result_summary(item),
        ]))
    sections.append("\n".join([
        "\u6ce8\u610f",
        "AccessKey Secret \u53ea\u5728\u672c\u6b21\u5ba1\u6279\u8bc4\u8bba\u4e2d\u5c55\u793a\u4e00\u6b21\u3002",
        "\u8bf7\u7533\u8bf7\u4eba\u9996\u6b21\u767b\u5f55\u540e\u81ea\u884c\u4fee\u6539\u5bc6\u7801\u5e76\u5b8c\u6210 MFA \u6fc0\u6d3b\u3002",
    ]))
    return "\n\n".join(sections)


def _result_summary(result: RamAccountResult) -> str:
    parts = []
    if result.created_user:
        parts.append("\u5df2\u521b\u5efa\u7528\u6237")
    if result.updated_user_profile:
        parts.append("\u5df2\u66f4\u65b0\u7528\u6237\u4fe1\u606f")
    if result.created_login_profile:
        parts.append("\u5df2\u5f00\u542f\u63a7\u5236\u53f0\u767b\u5f55")
    if result.created_access_key:
        parts.append("\u5df2\u521b\u5efa AccessKey")
    if result.added_groups:
        parts.append("\u5df2\u52a0\u5165\u7528\u6237\u7ec4 " + ", ".join(result.added_groups))
    if result.skipped:
        parts.append("\u8df3\u8fc7: " + ", ".join(result.skipped))
    return "\uff1b".join(parts) or "\u65e0\u53d8\u66f4"


def _approval_comment_user_id(req: RamAccountRequest | None = None) -> str:
    configured = getattr(settings, "FEISHU_RAM_APPROVAL_COMMENT_USER_ID", "")
    if configured:
        return configured
    # 申请人 open_id 来自本 app 的事件/实例，跨 app 安全；优先于全局 ADMIN_FEISHU_OPEN_ID
    # （后者可能是别的 app 的 open_id，写评论会报 99992361 open_id cross app）。
    if req and req.requester_open_id:
        return req.requester_open_id
    return getattr(settings, "ADMIN_FEISHU_OPEN_ID", "") or ""


def _approval_comment_user_id_type() -> str:
    return (getattr(settings, "FEISHU_RAM_APPROVAL_COMMENT_USER_ID_TYPE", "open_id") or "open_id").strip()


def _send_approval_comment(
    instance_code: str,
    text: str,
    user_id: str,
    *,
    comment_id: str = "",
) -> str:
    if not instance_code:
        raise RamApprovalError("审批实例 Code 为空，无法写入审批评论")
    if not user_id:
        raise RamApprovalError("审批评论用户 ID 为空，无法写入审批评论")

    body: dict[str, Any] = {
        "content": json.dumps({"text": text}, ensure_ascii=False),
        "disable_bot": False,
    }
    if comment_id:
        body["comment_id"] = comment_id

    token = _get_access_token()
    resp = requests.post(
        f"{FEISHU_BASE}/approval/v4/instances/{quote(instance_code, safe='')}/comments",
        params={"user_id_type": _approval_comment_user_id_type(), "user_id": user_id},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        json=body,
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RamApprovalError(f"写入飞书审批评论失败: {data.get('msg') or data}")
    return str((data.get("data") or {}).get("comment_id") or comment_id or "")


def save_approval_record(
    req: RamAccountRequest,
    detail: dict[str, Any] | None = None,
    event_payload: dict[str, Any] | None = None,
    result_status: str = "processing",
) -> None:
    """Persist a sanitized approval snapshot in Redis. Secrets are never stored."""
    detail = detail or {}
    event_payload = event_payload or {}
    instance_code = req.approval_instance_code or _extract_instance_code(detail) or _extract_instance_code(event_payload)
    if not instance_code:
        return
    now_ms = _now_ms()
    record = {
        "instance_code": instance_code,
        "approval_code": _extract_approval_code(detail) or _extract_approval_code(event_payload),
        "approval_name": _deep_first(detail, ("approval_name", "approvalName")),
        "serial_number": _deep_first(detail, ("serial_number", "serialNumber")),
        "approval_status": _extract_status(detail) or _extract_status(event_payload),
        "result_status": result_status,
        "login_name": req.login_name,
        "display_name": req.display_name,
        "email": req.email,
        "mobile_phone": req.mobile_phone,
        "groups": list(req.groups),
        "aliyun_groups": list(req.aliyun_groups),
        "volcano_groups": list(req.volcano_groups),
        "platform_group_map": _platform_group_map(req),
        "platforms": list(req.platforms),
        "console_access": req.console_access,
        "permanent_access_key": req.permanent_access_key,
        "password_reset_required": req.password_reset_required,
        "mfa_bind_required": req.mfa_bind_required,
        "reason": req.reason,
        "comments": req.comments,
        "requester_open_id": req.requester_open_id,
        "requester_user_id": req.requester_user_id,
        "password_set": bool(req.password),
        "password_length": len(req.password or ""),
        "start_time": _deep_first(detail, ("start_time", "startTime")),
        "end_time": _deep_first(detail, ("end_time", "endTime")),
        "updated_at_ms": now_ms,
    }
    _save_instance_record(instance_code, record)


def save_approval_result(
    req: RamAccountRequest,
    result: RamAccountResult,
    result_status: str = "success",
) -> None:
    instance_code = req.approval_instance_code
    if not instance_code:
        return
    record = {
        "result_status": result_status,
        "result_user_name": result.user_name,
        "platform": result.platform,
        "platform_label": result.platform_label,
        "account_id": result.account_id,
        "created_user": result.created_user,
        "updated_user_profile": result.updated_user_profile,
        "set_security_phone": result.set_security_phone,
        "set_security_email": result.set_security_email,
        "created_login_profile": result.created_login_profile,
        "created_access_key": result.created_access_key,
        "added_groups": list(result.added_groups),
        "skipped": list(result.skipped),
        "access_key_id": result.access_key_id,
        "login_principal": result.login_principal or build_login_principal(req.login_name),
        "login_url": result.login_url or build_login_url(req.login_name),
        "access_key_secret_saved": False,
        "error_message": "",
        "updated_at_ms": _now_ms(),
    }
    if result.platform_results:
        record["platform_results"] = [_approval_result_record(item, req) for item in result.platform_results]
    _save_instance_record(instance_code, record)


def _approval_result_record(result: RamAccountResult, req: RamAccountRequest) -> dict[str, Any]:
    return {
        "platform": result.platform,
        "platform_label": result.platform_label,
        "user_name": result.user_name,
        "account_id": result.account_id,
        "created_user": result.created_user,
        "updated_user_profile": result.updated_user_profile,
        "created_login_profile": result.created_login_profile,
        "created_access_key": result.created_access_key,
        "added_groups": list(result.added_groups),
        "skipped": list(result.skipped),
        "access_key_id": result.access_key_id,
        "login_principal": result.login_principal,
        "login_url": result.login_url,
        "access_key_secret_saved": False,
    }


def save_approval_failure(
    instance_code: str,
    exc: Exception,
    req: RamAccountRequest | None = None,
    event_payload: dict[str, Any] | None = None,
) -> None:
    if not instance_code:
        instance_code = _extract_instance_code(event_payload or {})
    if not instance_code:
        return
    record: dict[str, Any] = {
        "instance_code": instance_code,
        "result_status": "failed",
        "error_message": str(exc),
        "updated_at_ms": _now_ms(),
    }
    if req:
        record.update({
            "login_name": req.login_name,
            "display_name": req.display_name,
            "email": req.email,
            "mobile_phone": req.mobile_phone,
            "groups": list(req.groups),
            "aliyun_groups": list(req.aliyun_groups),
            "volcano_groups": list(req.volcano_groups),
            "platform_group_map": _platform_group_map(req),
            "platforms": list(req.platforms),
            "console_access": req.console_access,
            "permanent_access_key": req.permanent_access_key,
            "password_reset_required": req.password_reset_required,
            "mfa_bind_required": req.mfa_bind_required,
            "reason": req.reason,
            "comments": req.comments,
            "requester_open_id": req.requester_open_id,
            "requester_user_id": req.requester_user_id,
            "password_set": bool(req.password),
            "password_length": len(req.password or ""),
        })
    _save_instance_record(instance_code, record)


def load_approval_record(instance_code: str) -> dict[str, Any]:
    if not instance_code:
        return {}
    try:
        raw = get_redis().get(_instance_record_key(instance_code))
        return json.loads(raw) if raw else {}
    except Exception:
        logger.warning("[ram_approval] failed to load Redis record", exc_info=True)
        return {}


def _save_instance_record(instance_code: str, patch: dict[str, Any]) -> None:
    if not instance_code:
        return
    try:
        r = get_redis()
        key = _instance_record_key(instance_code)
        current_raw = r.get(key)
        current = json.loads(current_raw) if current_raw else {}
        if "created_at_ms" not in current:
            current["created_at_ms"] = _now_ms()
        current.update({k: v for k, v in patch.items() if v is not None})
        r.set(key, json.dumps(current, ensure_ascii=False, sort_keys=True))
        r.zadd(REDIS_INDEX_KEY, {instance_code: int(current.get("updated_at_ms") or _now_ms())})
    except Exception:
        logger.warning("[ram_approval] failed to save Redis record", exc_info=True)


def _is_instance_done(instance_code: str) -> bool:
    record = load_approval_record(instance_code)
    return record.get("result_status") in {"success", "dry_run"}


def _claim_instance(instance_code: str) -> str:
    if not instance_code:
        return ""
    lock_key = REDIS_LOCK_PREFIX + instance_code
    try:
        ok = get_redis().set(lock_key, "1", nx=True, ex=REDIS_LOCK_TTL_SECONDS)
        return lock_key if ok else ""
    except Exception:
        logger.warning("[ram_approval] Redis lock unavailable; continuing without instance lock", exc_info=True)
        return "redis_unavailable"


def _release_instance_lock(lock_key: str) -> None:
    if not lock_key or lock_key == "redis_unavailable":
        return
    try:
        get_redis().delete(lock_key)
    except Exception:
        logger.warning("[ram_approval] failed to release Redis lock", exc_info=True)


def _mark_duplicate(instance_code: str, reason: str) -> None:
    if not instance_code:
        return
    record = load_approval_record(instance_code)
    duplicate_count = int(record.get("duplicate_count") or 0) + 1
    _save_instance_record(instance_code, {
        "duplicate_count": duplicate_count,
        "last_duplicate_reason": reason,
        "updated_at_ms": _now_ms(),
    })


def _instance_record_key(instance_code: str) -> str:
    return REDIS_INSTANCE_PREFIX + instance_code


def _now_ms() -> int:
    return int(time.time() * 1000)

def notify_failure(
    instance_code: str,
    exc: Exception,
    req: RamAccountRequest | None = None,
    *,
    approval_comment_id: str = "",
) -> None:
    text = f"RAM 子账号审批执行失败\n审批实例: {instance_code or '-'}\n错误: {exc}"
    if _delivery_mode() in {"approval_comment", "comment"} and instance_code:
        try:
            _send_approval_comment(
                instance_code,
                text,
                _approval_comment_user_id(req),
                comment_id=approval_comment_id,
            )
        except Exception:
            logger.error("[ram_approval] failure approval comment failed", exc_info=True)
        return
    _send_text(_result_chat_id(), "chat_id", text)


def _result_chat_id() -> str:
    return getattr(settings, "FEISHU_RAM_APPROVAL_RESULT_CHAT_ID", "") or settings.FEISHU_CHAT_ID


def _send_text(receive_id: str, receive_id_type: str, text: str) -> None:
    if not receive_id:
        return
    try:
        token = _get_access_token()
        requests.post(
            f"{FEISHU_BASE}/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"receive_id": receive_id, "msg_type": "text", "content": json.dumps({"text": text}, ensure_ascii=False)},
            timeout=15,
        )
    except Exception:
        logger.error("[ram_approval] send Feishu text failed", exc_info=True)


def _comments(req: RamAccountRequest) -> str:
    parts = []
    if req.comments:
        parts.append(req.comments)
    if req.reason:
        parts.append(f"申请原因: {req.reason}")
    if req.approval_instance_code:
        parts.append(f"Feishu approval: {req.approval_instance_code}")
    return " | ".join(parts)


def _list_access_key_ids(client: Any, user_name: str) -> list[str]:
    from alibabacloud_ram20150501 import models as m

    body = client.list_access_keys(m.ListAccessKeysRequest(user_name=user_name)).body
    keys = getattr(getattr(body, "access_keys", None), "access_key", None) or []
    return [getattr(k, "access_key_id", "") for k in keys if getattr(k, "access_key_id", "")]


def _is_not_exist(exc: Exception, entity: str) -> bool:
    code = _err_code(exc) or ""
    return "NotExist" in code and entity in code


def _is_already_exists(exc: Exception) -> bool:
    code = _err_code(exc) or ""
    return "Already" in code or "Exist" in code and "NotExist" not in code


def _field_any(values: ApprovalFormValues, key: str) -> Any:
    env_name, aliases = FIELD_CONFIG[key]
    specs = _split_specs(getattr(settings, env_name, ""))
    for spec in specs + list(aliases):
        if spec in values.by_id:
            return values.by_id[spec]
        if spec in values.by_name:
            return values.by_name[spec]
    return None


def _field_text(values: ApprovalFormValues, key: str) -> str:
    return _as_text(_field_any(values, key))


def _field_bool(values: ApprovalFormValues, key: str, default: bool = False) -> bool:
    raw = _field_any(values, key)
    if raw is None or raw == "":
        return default
    return _as_bool(raw, default=default)


def _split_specs(raw: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;]", raw or "") if x.strip()]


def _allowed_groups(platform: str = PLATFORM_ALIYUN_RAM) -> tuple[str, ...]:
    if platform == PLATFORM_VOLCANO_IAM:
        raw = getattr(settings, "FEISHU_VOLCANO_IAM_ALLOWED_GROUPS_RAW", "")
        groups = _split_specs(raw)
        return tuple(groups or ["wuji_group", "wuji_product_team", "wuji-opration"])
    raw = getattr(settings, "FEISHU_RAM_APPROVAL_ALLOWED_GROUPS_RAW", "")
    groups = _split_specs(raw)
    return tuple(groups or ["wuji_Algorithm", "wuji_Examination"])


def _allowed_groups_for_platforms(platforms: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for platform in platforms or (PLATFORM_ALIYUN_RAM,):
        for group in _allowed_groups(platform):
            if group not in out:
                out.append(group)
    return tuple(out)


def _normalize_platforms(raw: Any) -> list[str]:
    values = _as_list(raw)
    if not values:
        return [PLATFORM_ALIYUN_RAM]
    out: list[str] = []
    unknown: list[str] = []
    for item in values:
        text = _as_text(item).strip()
        compact = re.sub(r"\s+", "", text).lower()
        platform = ""
        if any(token in compact for token in ("aliyunram", "\u963f\u91cc\u4e91ram", "\u963f\u91ccram", "\u963f\u91cc\u4e91", "alibaba", "ram")):
            platform = PLATFORM_ALIYUN_RAM
        elif any(token in compact for token in ("volcanoiam", "volcengineiam", "\u706b\u5c71\u5f15\u64ceiam", "\u706b\u5c71iam", "\u706b\u5c71\u5f15\u64ce", "volcano", "volcengine")):
            platform = PLATFORM_VOLCANO_IAM
        if platform:
            if platform not in out:
                out.append(platform)
        elif text:
            unknown.append(text)
    if unknown and not out:
        raise RamApprovalError("unrecognized approval platform: " + ", ".join(unknown))
    return out or [PLATFORM_ALIYUN_RAM]


def _normalize_groups(raw: Any, allowed_groups: tuple[str, ...] | None = None) -> list[str]:
    allowed = tuple(allowed_groups or ())
    values = []
    for item in _as_list(raw):
        text = _as_text(item).strip()
        if not text:
            continue
        if allowed:
            if text in allowed:
                values.append(text)
                continue
            matched = False
            for group in allowed:
                if group in text:
                    values.append(group)
                    matched = True
            if matched:
                continue
        values.append(text)
    out = []
    for group in values:
        if group not in out:
            out.append(group)
    return out


def _normalize_email(raw: str) -> str:
    return (raw or "").strip().replace("mailto:", "")


def _normalize_mobile(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = text.replace(" ", "")
    if re.fullmatch(r"1\d{10}", text):
        return f"86-{text}"
    if re.fullmatch(r"\+86[-]?\d{11}", text):
        return "86-" + text.replace("+86", "").lstrip("-")
    return text


def _as_bool(raw: Any, default: bool = False) -> bool:
    raw = _load_json_maybe(raw)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, list):
        return any(_as_bool(x, default=False) for x in raw)
    if isinstance(raw, dict):
        for key in ("checked", "value", "selected", "enable", "enabled"):
            if key in raw:
                return _as_bool(raw[key], default=default)
        return default
    text = _as_text(raw).strip().lower()
    if text in {"true", "1", "yes", "y", "on", "是", "开启", "启用", "允许", "需要", "勾选", "已开启"}:
        return True
    if text in {"false", "0", "no", "n", "off", "否", "关闭", "禁用", "不需要", "未开启", "未勾选"}:
        return False
    return default


def _as_list(raw: Any) -> list[Any]:
    raw = _load_json_maybe(raw)
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        out: list[Any] = []
        for item in raw:
            value = _load_json_maybe(item)
            if isinstance(value, list):
                out.extend(value)
            else:
                out.append(value)
        return out
    if isinstance(raw, dict):
        for key in ("value", "values", "selected", "options"):
            if key in raw:
                return _as_list(raw[key])
        return [raw]
    text = str(raw)
    if any(sep in text for sep in (",", "，", ";", "；", "\n")):
        return [x.strip() for x in re.split(r"[,，;；\n]+", text) if x.strip()]
    return [raw]


def _as_text(raw: Any) -> str:
    raw = _load_json_maybe(raw)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (int, float, bool)):
        return str(raw)
    if isinstance(raw, list):
        return ", ".join(_as_text(x) for x in raw if _as_text(x))
    if isinstance(raw, dict):
        for key in ("text", "name", "label", "value", "zh_cn", "en_us", "id"):
            if key in raw:
                text = _as_text(raw[key])
                if text:
                    return text
        return json.dumps(raw, ensure_ascii=False)
    return str(raw).strip()


def _load_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _iter_form_items(root: Any) -> list[dict[str, Any]]:
    root = _load_json_maybe(root)
    found: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        obj = _load_json_maybe(obj)
        if isinstance(obj, list):
            for item in obj:
                walk(item)
            return
        if not isinstance(obj, dict):
            return

        has_label = any(k in obj for k in ("id", "widget_id", "field_id", "custom_id", "name", "title", "label"))
        has_value = any(k in obj for k in ("value", "option", "text"))
        if has_label and has_value:
            found.append(obj)

        for key in ("data", "event", "form", "form_data", "fields", "items", "children", "widget_list", "values"):
            if key in obj:
                walk(obj[key])

    walk(root)
    return found


def _first_text(obj: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _as_text(obj.get(key))
        if text:
            return text
    return ""


def _event_obj(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event")
    return event if isinstance(event, dict) else payload


def _event_type(payload: dict[str, Any]) -> str:
    header = payload.get("header") if isinstance(payload, dict) else None
    if isinstance(header, dict):
        return header.get("event_type", "") or header.get("eventType", "")
    return ""


def _extract_approval_code(payload: dict[str, Any]) -> str:
    return _deep_first(payload, ("approval_code", "approvalCode", "definition_code", "definitionCode"))


def _extract_instance_code(payload: dict[str, Any]) -> str:
    return _deep_first(payload, ("instance_code", "instanceCode", "approval_instance_code", "approvalInstanceCode"))


def _extract_status(payload: dict[str, Any]) -> str:
    return _deep_first(payload, ("status", "approval_status", "approvalStatus", "instance_status", "instanceStatus"))


def _extract_requester_ids(detail: dict[str, Any], event_payload: dict[str, Any]) -> tuple[str, str]:
    open_id = _deep_first(event_payload, ("open_id", "openId", "requester_open_id", "applicant_open_id"))
    open_id = open_id or _deep_first(detail, ("open_id", "openId", "requester_open_id", "applicant_open_id"))
    user_id = _deep_first(detail, ("user_id", "userId", "requester_user_id", "applicant_user_id"))
    user_id = user_id or _deep_first(event_payload, ("user_id", "userId", "requester_user_id", "applicant_user_id"))
    if user_id.startswith("ou_") and not open_id:
        open_id, user_id = user_id, ""
    return open_id, user_id


def _deep_first(obj: Any, keys: tuple[str, ...]) -> str:
    obj = _load_json_maybe(obj)
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                text = _as_text(obj[key])
                if text:
                    return text
        for value in obj.values():
            text = _deep_first(value, keys)
            if text:
                return text
    elif isinstance(obj, list):
        for value in obj:
            text = _deep_first(value, keys)
            if text:
                return text
    return ""
