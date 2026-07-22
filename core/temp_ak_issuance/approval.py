"""飞书审批事件入口：临时 AK 发放。

复用 ram_approval 的事件解析与 **实例级 status==APPROVED 门禁**（#55 硬化：只认回拉实例详情顶层
实例级 status、无 instance_code fail-safe 不发），但独立审批模板 code + 独立 Redis 命名空间。
审批通过才发放；发放路径二次校验实例 APPROVED，杜绝任何绕审批发凭证的口子。
"""
from __future__ import annotations

import re
from typing import Any

from config.settings import settings
from utils.logger import get_logger

from . import delivery, orchestrator

logger = get_logger(__name__)

# 表单字段名映射：env 覆盖候选 + 内置别名（含真机 widget id，名字改了也能命中）。
# 真机审批「数据外采访问凭证申请」(5B4A…) 实际 5 控件：平台/使用企业名称/权限设置/DateInterval/申请目录。
_FIELDS = {
    "platform":      ("TEMP_AK_FIELD_PLATFORM",      ("平台", "云平台", "platform", "widget17846401222860001")),
    "enterprise":    ("TEMP_AK_FIELD_ENTERPRISE",    ("使用企业名称", "企业名称", "使用方", "外采企业", "enterprise", "widget17846886904010001")),
    "perm":          ("TEMP_AK_FIELD_PERM",          ("权限设置", "权限", "读写", "perm", "widget17846401501570001")),
    "date_interval": ("TEMP_AK_FIELD_DATE_INTERVAL", ("DateInterval", "有效期", "生效到期", "起止时间", "date_interval", "widget17846402309610001")),
    "directory":     ("TEMP_AK_FIELD_DIRECTORY",     ("申请目录", "目录", "路径", "directory", "widget17846402564230001")),
}

# 延期审批「访问凭证延长申请」3 控件：凭证ID / 使用企业信息 / DateInterval（含真机 widget id）。
_EXTEND_FIELDS = {
    "grant_id":      ("TEMP_AK_FIELD_GRANT_ID",      ("凭证ID", "凭证编号", "任务ID", "grant_id", "widget17846399101070001")),
    "enterprise":    ("TEMP_AK_FIELD_ENTERPRISE",    ("使用企业信息", "使用企业名称", "企业信息", "企业名称", "enterprise", "widget17846914004480001")),
    "date_interval": ("TEMP_AK_FIELD_DATE_INTERVAL", ("DateInterval", "有效期", "生效到期", "date_interval", "widget17846911942480001")),
}


def should_handle_event(payload: dict[str, Any]) -> bool:
    """精确匹配 TEMP_AK_APPROVAL_CODE。要求 code 恰等于本模板（不做 no-code 兜底），
    与 ram_approval 按各自 code 分流、互不误抢。"""
    from core import ram_approval
    target = settings.TEMP_AK_APPROVAL_CODE
    if not target:
        return False
    et = ram_approval._event_type(payload).lower()
    code = ram_approval._extract_approval_code(payload)
    if "approval" not in et and not code:
        return False
    return code == target


def handle_temp_ak_event(payload: dict[str, Any]) -> dict[str, Any]:
    from core import ram_approval
    if not should_handle_event(payload):
        return {"ignored": True, "reason": "not_temp_ak"}

    status = (ram_approval._extract_status(payload) or "").upper()
    if status and status not in ram_approval.APPROVED_STATUSES:
        return {"ignored": True, "reason": f"status={status}"}
    instance_code = ram_approval._extract_instance_code(payload)
    if not status and not instance_code:
        return {"ignored": True, "reason": "no_status_no_instance"}

    gid = orchestrator.grant_id_for(instance_code)
    existing = orchestrator.get_grant(gid)
    if existing and existing.get("stage") in (orchestrator.STAGE_ISSUED, orchestrator.STAGE_REVOKED):
        return {"ignored": True, "reason": "already_issued", "grant_id": gid}

    lock = orchestrator.claim(instance_code)
    if instance_code and not lock:
        return {"ignored": True, "reason": "already_processing", "grant_id": gid}

    try:
        # 门禁（照 #55）：无 instance_code → 无法核实整单状态 → 绝不发放（fail-safe）。
        if not instance_code:
            return {"ignored": True, "reason": "no_instance_code"}
        detail = ram_approval.fetch_approval_instance(instance_code)
        # 只认回拉实例详情顶层实例级 status=="APPROVED"（不用 APPROVED_STATUSES：那含节点级 PASS/AGREE）。
        instance_status = str((detail or {}).get("status") or "").strip().upper()
        if instance_status != "APPROVED":
            logger.info("[temp_ak] instance NOT fully approved (status=%s) instance=%s; skip issue",
                        instance_status or "-", instance_code)
            return {"ignored": True, "reason": f"instance_status={instance_status or 'none'}", "grant_id": gid}
        code = ram_approval._extract_approval_code(detail) or ram_approval._extract_approval_code(payload)
        if code and code != settings.TEMP_AK_APPROVAL_CODE:
            return {"ignored": True, "reason": "approval_code_mismatch"}

        spec = parse_temp_ak_request(detail, payload)
        requester, _ = ram_approval._extract_requester_ids(detail, payload)
        grant = orchestrator.create_grant_record(spec, instance_code=instance_code, requester=requester)

        if getattr(settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False):
            logger.info("[temp_ak] dry-run: 计划 %s（未真发）", issuer_plan_summary(grant))
            return {"ignored": False, "dry_run": True, "grant_id": gid, "mode": grant["mode"]}

        grant, creds = orchestrator.issue_grant(grant)
        delivery.deliver(grant, creds)
        return {"ignored": False, "grant_id": gid, "mode": grant["mode"]}
    except Exception as exc:
        g = orchestrator.get_grant(gid)
        if g:
            orchestrator.fail_grant(g, str(exc))
        logger.error("[temp_ak] issue failed instance=%s", instance_code, exc_info=True)
        _notify_internal_failure(instance_code, exc)
        return {"ignored": False, "error": str(exc)}
    finally:
        orchestrator.release(lock)


# ── 延期审批（独立模板 TEMP_AK_EXTEND_APPROVAL_CODE「访问凭证延长申请」）────────────────

def should_handle_extend_event(payload: dict[str, Any]) -> bool:
    from core import ram_approval
    target = settings.TEMP_AK_EXTEND_APPROVAL_CODE
    if not target:
        return False
    et = ram_approval._event_type(payload).lower()
    code = ram_approval._extract_approval_code(payload)
    if "approval" not in et and not code:
        return False
    return code == target


def handle_temp_ak_extend_event(payload: dict[str, Any]) -> dict[str, Any]:
    from core import ram_approval
    if not should_handle_extend_event(payload):
        return {"ignored": True, "reason": "not_temp_ak_extend"}
    status = (ram_approval._extract_status(payload) or "").upper()
    if status and status not in ram_approval.APPROVED_STATUSES:
        return {"ignored": True, "reason": f"status={status}"}
    instance_code = ram_approval._extract_instance_code(payload)
    if not status and not instance_code:
        return {"ignored": True, "reason": "no_status_no_instance"}
    lock = orchestrator.claim(instance_code)
    if instance_code and not lock:
        return {"ignored": True, "reason": "already_processing"}
    try:
        # 门禁同发放：只认回拉实例详情顶层实例级 status=="APPROVED"、无 instance_code fail-safe 不动。
        if not instance_code:
            return {"ignored": True, "reason": "no_instance_code"}
        detail = ram_approval.fetch_approval_instance(instance_code)
        instance_status = str((detail or {}).get("status") or "").strip().upper()
        if instance_status != "APPROVED":
            logger.info("[temp_ak] extend instance NOT approved (status=%s) instance=%s; skip",
                        instance_status or "-", instance_code)
            return {"ignored": True, "reason": f"instance_status={instance_status or 'none'}"}
        code = ram_approval._extract_approval_code(detail) or ram_approval._extract_approval_code(payload)
        if code and code != settings.TEMP_AK_EXTEND_APPROVAL_CODE:
            return {"ignored": True, "reason": "approval_code_mismatch"}

        grant_id, enterprise, not_before, expire = parse_temp_ak_extend_request(detail, payload)
        grant = orchestrator.get_grant(grant_id)
        if not grant:
            raise orchestrator.TempAkError(f"未找到凭证 {grant_id}（可能已过期/清理，请重新发起发放）")
        if grant.get("stage") == orchestrator.STAGE_REVOKED:
            raise orchestrator.TempAkError(f"凭证 {grant_id} 已到期清理，请重新发起发放审批")
        _verify_enterprise(grant, enterprise)
        if instance_code in (grant.get("extend_instances") or []):
            return {"ignored": True, "reason": "extend_already_applied", "grant_id": grant_id}

        grant, creds = orchestrator.extend_grant(grant, not_before, expire, extend_instance=instance_code)
        delivery.deliver_extend(grant, creds)
        return {"ignored": False, "grant_id": grant_id, "mode": grant["mode"]}
    except Exception as exc:
        logger.error("[temp_ak] extend failed instance=%s", instance_code, exc_info=True)
        _notify_internal_failure(instance_code, exc)
        return {"ignored": False, "error": str(exc)}
    finally:
        orchestrator.release(lock)


def parse_temp_ak_extend_request(detail: dict[str, Any], payload: dict[str, Any]):
    from core import ram_approval
    values = ram_approval.extract_form_values(detail)

    def field(key: str):
        env_name, aliases = _EXTEND_FIELDS[key]
        specs = ram_approval._split_specs(getattr(settings, env_name, ""))
        for spec in specs + list(aliases):
            if spec in values.by_id:
                return values.by_id[spec]
            if spec in values.by_name:
                return values.by_name[spec]
        return None

    grant_id = ram_approval._as_text(field("grant_id")).strip()
    enterprise = ram_approval._as_text(field("enterprise")).strip()
    not_before, expire = _parse_date_interval(field("date_interval"))
    if not grant_id:
        raise orchestrator.TempAkError("延期审批缺少凭证ID")
    if not expire:
        raise orchestrator.TempAkError("延期审批缺少新的到期时间（DateInterval）")
    return grant_id, enterprise, not_before, expire


def _verify_enterprise(grant: dict, enterprise: str) -> None:
    """防串企业：延期表单的使用企业信息须与原凭证一致（宽松包含匹配容错格式差异）。

    fail-safe：原凭证有企业名而延期表单留空 → 无法核实归属 → 拒（否则引用别家 grant_id + 留空即绕过防串）。
    仅当原凭证本身无企业名时才跳过（无可比对象）。"""
    orig = (grant.get("enterprise") or "").strip()
    ent = (enterprise or "").strip()
    if not orig:
        return
    if not ent:
        raise orchestrator.TempAkError("延期申请缺少使用企业信息，无法核实凭证归属，拒绝延期")
    if orig != ent and orig not in ent and ent not in orig:
        raise orchestrator.TempAkError(f"使用企业信息与原凭证不符（原：{orig}），拒绝延期")


# ── 表单解析（真机 5 控件：平台/使用企业名称/权限设置/DateInterval/申请目录）──────────

def parse_temp_ak_request(detail: dict[str, Any], payload: dict[str, Any]) -> dict:
    from core import ram_approval
    values = ram_approval.extract_form_values(detail)

    def field(key: str):
        env_name, aliases = _FIELDS[key]
        specs = ram_approval._split_specs(getattr(settings, env_name, ""))
        for spec in specs + list(aliases):
            if spec in values.by_id:
                return values.by_id[spec]
            if spec in values.by_name:
                return values.by_name[spec]
        return None

    platform = _parse_platform(field("platform"))
    enterprise = ram_approval._as_text(field("enterprise")).strip()
    caps = _parse_perm(field("perm"))                 # ⊆ {read(列), download(下载), write(上传)}
    not_before, expire = _parse_date_interval(field("date_interval"))
    bucket, prefix = _parse_directory(field("directory"))

    spec = {
        "platform": platform,
        "enterprise": enterprise,
        "bucket": bucket,
        "prefix": prefix,
        "caps": caps,
        "not_before": not_before,
        "expire": expire,
        # 模板无邮箱字段：凭证定向发给审批发起人(内部申请人)，由其转交外采企业（见 delivery）。
        "recipient_email": "",
        "source_ips": [],
        "reason": f"外采企业：{enterprise}" if enterprise else "",
    }
    _validate_spec(spec)
    return spec


def _validate_spec(spec: dict) -> None:
    import time as _t
    if spec["platform"] == "volcano":
        raise orchestrator.TempAkError("平台=火山云 暂未支持（当前仅阿里云 OSS）；请改选阿里云或联系运维")
    if spec["platform"] != "aliyun":
        raise orchestrator.TempAkError("无法识别申请平台，请在审批表单选择阿里云")
    if not spec["bucket"]:
        raise orchestrator.TempAkError("申请目录缺少桶名（形如 oss://<桶>/<目录>/ 或 <桶>/<目录>/）")
    if not spec.get("caps"):
        raise orchestrator.TempAkError("权限设置未勾选任何项（read/download/write），未发放")
    if not spec["expire"]:
        raise orchestrator.TempAkError("缺少有效期（DateInterval 未取到到期时间）")
    now = _t.time()
    if spec["expire"] <= now:
        raise orchestrator.TempAkError("到期时间已过，未发放")
    if spec["not_before"] and spec["not_before"] >= spec["expire"]:
        raise orchestrator.TempAkError("生效时间必须早于到期时间")
    if not spec["not_before"]:
        spec["not_before"] = now   # 未取到生效时间 = 立即生效


_KEY_RE = re.compile(r"^[^\x00-\x1f]+$")   # 拒控制字符（前缀进 policy JSON 的 Resource ARN）


def _parse_platform(raw) -> str:
    from core import ram_approval
    t = ram_approval._as_text(raw).lower()
    if "火山" in t or "volcano" in t or "volc" in t or "tos" in t:
        return "volcano"
    if "阿里" in t or "aliyun" in t or "alibaba" in t or "oss" in t:
        return "aliyun"
    return "unknown"   # 未知平台不默认 aliyun（避免给非 OSS 目录误发无用 OSS 凭证）→ validate 拒


def _parse_perm(raw) -> list:
    """权限设置多选 → 能力集 caps ⊆ {read,download,write}（三者正交）。

    选项：read(只列)/download(下载 GetObject)/write(上传)/read&write(=read+write)。
    子串判定安全：'download' 不含 read/write；'read&write' 含 read+write。
    """
    from core import ram_approval
    items = ram_approval._as_list(raw)
    text = " ".join(ram_approval._as_text(x) for x in items).lower() if items \
        else ram_approval._as_text(raw).lower()
    caps = []
    if "download" in text:
        caps.append("download")
    if "write" in text:
        caps.append("write")
    if "read" in text:
        caps.append("read")
    return caps


def _parse_directory(raw) -> tuple[str, str]:
    """「申请目录」→ (bucket, prefix)。接受 oss://桶/前缀/、tos://…、桶/前缀、或裸桶名。"""
    from core import ram_approval
    s = ram_approval._as_text(raw).strip()
    if not s:
        return "", ""
    low = s.lower()
    for scheme in ("oss://", "tos://"):
        if low.startswith(scheme):
            s = s[len(scheme):]
            break
    s = s.lstrip("/")
    bucket, _, prefix = s.partition("/")
    bucket = bucket.strip()
    prefix = prefix.strip()
    if bucket and not _KEY_RE.match(bucket):
        raise orchestrator.TempAkError(f"非法申请目录（桶名含非法字符）：{raw!r}")
    if prefix and not _KEY_RE.match(prefix):
        raise orchestrator.TempAkError(f"非法申请目录：{raw!r}")
    return bucket, prefix


def _parse_date_interval(raw) -> tuple[float, float]:
    """DateInterval 控件 → (not_before, expire)。容错 dict{start/end 各种键}/list[2]/单值(仅到期)。"""
    from core import ram_approval
    v = ram_approval._load_json_maybe(raw)
    start = end = 0.0
    if isinstance(v, dict):
        for sk in ("start", "startTime", "start_time", "from", "begin", "start_date"):
            if v.get(sk):
                start = _parse_dt(v.get(sk)); break
        for ek in ("end", "endTime", "end_time", "to", "finish", "end_date"):
            if v.get(ek):
                end = _parse_dt(v.get(ek)); break
    elif isinstance(v, (list, tuple)) and len(v) >= 2:
        start, end = _parse_dt(v[0]), _parse_dt(v[1])
    else:
        end = _parse_dt(v)
    return start, end


def _parse_dt(raw) -> float:
    from datetime import datetime, timedelta, timezone
    from core import ram_approval
    raw = ram_approval._load_json_maybe(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        return v / 1000 if v > 1e11 else v
    s = ram_approval._as_text(raw).strip()
    if not s:
        return 0.0
    if s.isdigit():
        v = float(s)
        return v / 1000 if v > 1e11 else v
    bj = timezone(timedelta(hours=8))
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=bj)
            return dt.timestamp()
        except ValueError:
            continue
    raise orchestrator.TempAkError(f"无法解析时间：{s!r}（用 YYYY-MM-DD HH:MM:SS 或时间戳）")


def issuer_plan_summary(grant: dict) -> str:
    from . import issuer, orchestrator as o
    p = issuer.plan(grant)
    return f"mode={p['mode']} {o.scope_line(grant)} 有效期 {o.fmt_window(grant)}"


def _notify_internal_failure(instance_code: str, exc: Exception) -> None:
    chat = settings.TEMP_AK_CHAT_ID or settings.FEISHU_CHAT_ID
    if not chat:
        return
    try:
        from core.dsw_scheduler import _send_text
        _send_text("", chat, f"❌ 临时 AK 发放失败（审批实例 {instance_code or '-'}）：{exc}")
    except Exception:
        logger.error("[temp_ak] 内部失败通知发送失败", exc_info=True)
