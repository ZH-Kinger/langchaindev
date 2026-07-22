"""临时 AK 发放 grant 状态机 + Redis 记录 + 幂等 + 桶解析。

grant 记录**绝不含 secret/token**（只存 ak_id 供方案 B 到期硬删定位）。
grant_id = hash(审批实例)：一审批实例一凭证；已 ISSUED/REVOKED 幂等短路，不重发。
状态：NEW → ISSUED → REVOKED（方案 B 到期硬删）/ FAILED。Redis temp_ak:grant:{id}，30 天 TTL。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone

from config.settings import settings
from utils.logger import get_logger

from . import issuer

logger = get_logger(__name__)

_KEY_PREFIX = "temp_ak:grant:"
_LOCK_PREFIX = "temp_ak:lock:"
_TTL_SECONDS = 30 * 86400

STAGE_NEW = "NEW"
STAGE_ISSUED = "ISSUED"
STAGE_REVOKED = "REVOKED"
STAGE_FAILED = "FAILED"

_BJ = timezone(timedelta(hours=8))


class TempAkError(RuntimeError):
    """grant 编排错误。"""


# ── Redis 记录 ────────────────────────────────────────────────────────────────

def _key(grant_id: str) -> str:
    return _KEY_PREFIX + grant_id


def grant_id_for(instance_code: str) -> str:
    return "tak-" + hashlib.md5(("temp_ak|" + (instance_code or "")).encode("utf-8")).hexdigest()[:16]


def get_grant(grant_id: str) -> dict | None:
    from utils.redis_client import get_redis
    try:
        raw = get_redis().get(_key(grant_id))
        return json.loads(raw) if raw else None
    except Exception:
        logger.warning("[temp_ak] get_grant failed id=%s", grant_id, exc_info=True)
        return None


def _save(grant: dict) -> None:
    from utils.redis_client import get_redis
    grant["updated_ts"] = time.time()
    try:
        get_redis().setex(_key(grant["grant_id"]), _TTL_SECONDS,
                          json.dumps(grant, ensure_ascii=False))
    except Exception:
        logger.warning("[temp_ak] save failed id=%s", grant.get("grant_id"), exc_info=True)


# ── 幂等锁（独立命名空间，不与 ram_approval 冲突）────────────────────────────────

def claim(instance_code: str) -> str:
    if not instance_code:
        return ""
    lock_key = _LOCK_PREFIX + instance_code
    try:
        from utils.redis_client import get_redis
        return lock_key if get_redis().set(lock_key, "1", nx=True, ex=600) else ""
    except Exception:
        logger.warning("[temp_ak] redis lock unavailable", exc_info=True)
        return "redis_unavailable"


def release(lock_key: str) -> None:
    if not lock_key or lock_key == "redis_unavailable":
        return
    try:
        from utils.redis_client import get_redis
        get_redis().delete(lock_key)
    except Exception:
        pass


# ── 桶解析 ────────────────────────────────────────────────────────────────────

def resolve_bucket(display: str) -> tuple[str, str]:
    """展示桶名 → (region, real_bucket)。先查 TEMP_AK_BUCKET_MAP(JSON)，再回退 permsync.BUCKET_MAP，
    都没有则把 display 当真实桶名原样用（region 未知留空）。"""
    display = (display or "").strip()
    if not display:
        raise TempAkError("审批表单缺少 OSS 桶")
    try:
        m = json.loads(settings.TEMP_AK_BUCKET_MAP_RAW or "{}")
    except Exception:
        m = {}
    if display in m and isinstance(m[display], dict):
        v = m[display]
        return v.get("region", ""), v.get("bucket") or display
    from core.oss_perm.permsync import BUCKET_MAP
    if display in BUCKET_MAP:
        region, bucket = BUCKET_MAP[display]
        return region, bucket
    return "", display   # 表单直接填了真实桶名


def _derive_user_name(spec: dict, instance_code: str) -> str:
    """外部方 RAM 用户名：tempak-<邮箱前缀>-<实例短hash>，收敛到 RAM 命名规则。"""
    base = (spec.get("recipient_email", "").split("@")[0] or "ext")
    base = re.sub(r"[^A-Za-z0-9._-]", "", base)[:24] or "ext"
    short = hashlib.md5((instance_code or base).encode("utf-8")).hexdigest()[:6]
    return f"tempak-{base}-{short}"


# ── grant 生命周期 ────────────────────────────────────────────────────────────

def create_grant_record(spec: dict, *, instance_code: str, requester: str = "",
                        approver: str = "") -> dict:
    """据审批表单 spec 建 grant 记录（幂等：同实例返回已有记录）。

    spec: {bucket(display), region?, read_prefixes, write_prefixes, not_before, expire,
           recipient_email, source_ips?, reason?}
    """
    gid = grant_id_for(instance_code)
    existing = get_grant(gid)
    if existing:
        return existing

    now = time.time()
    region, real_bucket = resolve_bucket(spec["bucket"])
    mode = issuer.classify_mode(spec["expire"], now)
    user_name = _derive_user_name(spec, instance_code)
    grant = {
        "grant_id": gid,
        "stage": STAGE_NEW,
        "mode": mode,
        "platform": spec.get("platform", "aliyun"),
        "enterprise": spec.get("enterprise", ""),
        "bucket": real_bucket,
        "bucket_display": spec["bucket"],
        "region": spec.get("region") or region,
        "read_prefixes": sorted(set(spec.get("read_prefixes") or [])),
        "write_prefixes": sorted(set(spec.get("write_prefixes") or [])),
        "not_before": float(spec["not_before"]),
        "expire": float(spec["expire"]),
        "recipient_email": spec.get("recipient_email", ""),
        "source_ips": list(spec.get("source_ips") or []),
        "user_name": user_name,
        "policy_name": (issuer.policy.POLICY_PREFIX + user_name) if mode == issuer.RAM_MODE else "",
        "ak_id": "",
        "instance_code": instance_code or "",
        "requester": requester,
        "approver": approver,
        "reason": spec.get("reason", ""),
        "error": "",
        "created_ts": now,
        "updated_ts": now,
    }
    _save(grant)
    return grant


def issue_grant(grant: dict) -> tuple[dict, dict | None]:
    """真发放。返回 (grant, creds)。creds 含 secret/token 仅供下发层当次使用，不落库。

    幂等：已 ISSUED 直接返回 (grant, None)——凭证已在当时下发过，绝不重发。
    """
    if grant["stage"] in (STAGE_ISSUED, STAGE_REVOKED):
        return grant, None
    creds = issuer.issue(grant)
    if grant["mode"] == issuer.RAM_MODE:
        grant["ak_id"] = creds.get("access_key_id", "")
    grant["stage"] = STAGE_ISSUED
    grant["issued_ts"] = time.time()
    _save(grant)
    return grant, creds


def fail_grant(grant: dict, error: str) -> None:
    grant["stage"] = STAGE_FAILED
    grant["error"] = str(error)[:500]
    _save(grant)


# ── 展示助手（供卡片/CLI/工具）──────────────────────────────────────────────────

def fmt_ts(epoch: float) -> str:
    if not epoch:
        return "-"
    return datetime.fromtimestamp(int(epoch), tz=_BJ).strftime("%Y-%m-%d %H:%M:%S")


def fmt_window(grant: dict) -> str:
    return f"{fmt_ts(grant.get('not_before'))} → {fmt_ts(grant.get('expire'))}"


def scope_line(grant: dict) -> str:
    r = ", ".join(x or "<整桶>" for x in grant.get("read_prefixes") or []) or "—"
    w = ", ".join(x or "<整桶>" for x in grant.get("write_prefixes") or []) or "—"
    return f"桶 `{grant.get('bucket')}` 读[{r}] 写[{w}]"
