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

from . import accounts, issuer

logger = get_logger(__name__)

# 默认档（现有主账号）的前缀。多账号下真正生效的是各档 profile 里的前缀，这两个常量保留
# 是因为历史数据与既有测试都按它们写死；accounts._default_profile() 取值与之逐字相同。
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

def _profile_of(grant_id: str):
    """凭证ID → 账号档案。前缀认不出来（历史数据 / 档案已下线）时退默认档，与本文件多账号化前一致。"""
    return accounts.by_grant_id(grant_id) or accounts.default()


def _key(grant_id: str) -> str:
    return _profile_of(grant_id).redis_prefix + "grant:" + grant_id


def grant_id_for(instance_code: str, profile=None) -> str:
    """审批实例 → 凭证ID。前缀带账号维度，**共用的延长/撤销审批据此把请求分派回正确账号**。

    hash 输入刻意仍是 "temp_ak|<instance>"（不掺 slug）：审批实例本身就跨账号唯一，
    掺进去只会让历史 grant_id 全部漂移。
    """
    p = profile or accounts.default()
    return p.grant_prefix + hashlib.md5(
        ("temp_ak|" + (instance_code or "")).encode("utf-8")).hexdigest()[:16]


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

def claim(instance_code: str, profile=None) -> str:
    if not instance_code:
        return ""
    p = profile or accounts.default()
    lock_key = p.redis_prefix + "lock:" + instance_code
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

# 桶地域探测结果缓存：(账号 slug, 桶名) → (裸 region, 过期时间戳)。
# 成功长缓存（桶地域几乎不变），失败短缓存（见 probe_bucket_region 里的说明）。
_REGION_PROBE_CACHE: dict[tuple, tuple] = {}
_PROBE_TTL_OK = 24 * 3600
_PROBE_TTL_FAIL = 300
# OSS 桶名规范：小写字母/数字/连字符，首尾字母数字，3–63 位
_BUCKET_NAME_RE = re.compile(r"\A[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]\Z")
# 合法地域形如 cn-hangzhou / ap-southeast-1
_REGION_RE = re.compile(r"\A[a-z]{2}-[a-z0-9\-]+\Z")


def probe_bucket_region(bucket: str, profile=None) -> str:
    """用**该账号自己的凭证**实时探测桶地域（GetBucketLocation）。返回裸 region；探不到返回 ""。

    为什么不复用 `tools.aliyun.oss.detect_bucket_region`（两条都是硬伤，不是风格问题）：

    ① **它按 open_id 走 client factory，取到的是默认账号的凭证。** OSS 桶名是全局唯一的，
       所以不会探到「另一个账号的同名桶」；但**读桶地域的权限是按账号授的** —— 拿默认账号的 AK
       去问别的账号的桶，正常结果就是 403、探不到。用该档自己的凭证才是能探到的那条路。
       （审计更正：早先这里写的理由是"可能撞上同名桶"，那不成立，桶名全局唯一。）

    ② **它探测失败会静默回退到默认地域。** 对凭证发放来说，一个自信的错地域比「未知」更坏：
       使用方会照着连，拿到 403 `must be addressed using the specified endpoint` —— 这个报错
       和「没权限」长得一模一样，排查方向整个跑偏（本次线上问题就是这么来的）。
       所以这里失败一律返回 ""，让 delivery._access_lines 如实显示「未知」并提示去控制台自查。

    探测本身**绝不能让发放失败**：地域只影响凭证正文里的连接信息三行，探不到照发。
    """
    bucket = (bucket or "").strip()
    # 申请人常把中文展示名填进来（"杭州-xxx"）。oss2.Bucket() 的构造器会对非法桶名抛
    # ClientError，而那是在下面的 try 之外抛的 —— 先在这儿挡掉，省一条无谓的堆栈。
    if not bucket or not _BUCKET_NAME_RE.match(bucket):
        return ""
    p = profile or accounts.default()
    key = (p.slug, bucket)
    hit = _REGION_PROBE_CACHE.get(key)
    if hit is not None and hit[1] > time.time():
        return hit[0]

    region = ""
    if p.ak_id and p.ak_secret:
        try:
            region = _probe_region_once(bucket, p.ak_id, p.ak_secret)
        except Exception:
            logger.warning("[temp_ak] %s 探测桶 %s 地域失败，按未知处理", p.label, bucket,
                           exc_info=True)
    else:
        logger.warning("[temp_ak] %s 缺 AK，无法探测桶 %s 的地域", p.label, bucket)

    # region_from_endpoint 对意外域名会吐出**非空但没意义**的串（`data.example.com` → `data`），
    # 那会让正文写出「地域：data」—— 比「未知」更误导，正好违背本函数"绝不猜"的前提。
    if region and not _REGION_RE.match(region):
        logger.warning("[temp_ak] 桶 %s 探到的地域 %r 不像合法地域，按未知处理", bucket, region)
        region = ""

    # 负缓存只给短 TTL：一次瞬时失败（限流/抖动）不该让这个桶**永远**是「未知」——
    # 而 grant 一旦落库就把 region 写死了（create_grant_record），重投也走幂等短路。
    _REGION_PROBE_CACHE[key] = (region, time.time() + (_PROBE_TTL_OK if region else _PROBE_TTL_FAIL))
    if region:
        logger.info("[temp_ak] 桶 %s 地域实时探测为 %s（映射表里没有它）", bucket, region)
    return region


def _probe_region_once(bucket: str, ak: str, sk: str, *, timeout: int = 10) -> str:
    """一次 GetBucketLocation。跨地域时 OSS 会拒绝并在响应里带上正确 endpoint，照样能捞出来。"""
    import oss2
    from tools.aliyun.oss import region_from_endpoint

    auth = oss2.Auth(ak, sk)
    # 先用任意一个 endpoint 问：桶在本地域就直接拿到 location；不在则走下面的 except 分支。
    probe = oss2.Bucket(auth, "https://oss-cn-hangzhou.aliyuncs.com", bucket,
                        connect_timeout=timeout)
    try:
        return region_from_endpoint(probe.get_bucket_location().location)
    except oss2.exceptions.OssError as e:
        # 异地桶：正确 endpoint 在响应头或 body 的 <Endpoint> 里。
        headers = getattr(e, "headers", None) or {}
        for k in ("x-oss-region", "X-Oss-Region"):
            if headers.get(k):
                return region_from_endpoint(headers[k])
        m = re.search(r"<Endpoint>\s*([^<]+?)\s*</Endpoint>", getattr(e, "body", "") or "")
        if m:
            return region_from_endpoint(m.group(1).strip())
        # 探不到的原因值得留痕，否则线上只看到「未知」却不知道是限流、无权限还是桶不存在。
        # **只记 status/code/request_id，绝不记 e.body** —— SignatureDoesNotMatch 的 body 里
        # 带 AccessKeyId 和 StringToSign，进日志就是凭证泄漏。
        logger.warning("[temp_ak] 探测桶 %s 地域未果：status=%s code=%s req_id=%s",
                       bucket, getattr(e, "status", "?"), getattr(e, "code", "?"),
                       getattr(e, "request_id", "?"))
        return ""     # 桶不存在 / 无权限 / 其它 —— 一律当探不到，**不猜**


def resolve_bucket(display: str, profile=None) -> tuple[str, str]:
    """展示桶名 → (region, real_bucket)。先查该账号的桶映射(JSON)，再回退 permsync.BUCKET_MAP，
    都没有则把 display 当真实桶名原样用（region 未知留空）。

    **桶映射按账号取**：两个主账号的桶名可能重名却是不同的桶，共用一张表会把凭证发到错的桶上。
    permsync.BUCKET_MAP 是现有账号的算法组对照表，只对默认档回退。"""
    display = (display or "").strip()
    if not display:
        raise TempAkError("审批表单缺少 OSS 桶")
    p = profile or accounts.default()
    m = accounts.bucket_map(p)
    if display in m and isinstance(m[display], dict):
        v = m[display]
        return v.get("region", ""), v.get("bucket") or display
    # 申请人常常直接填**真实桶名**（表单占位符就是 oss://桶/目录/ 的形状），此时按展示名查不到。
    # 反查一遍映射的 bucket 值，把地域捞回来 —— 否则 region 为空，凭证正文里的
    # 地域/Endpoint/桶域名三行会退化成「未知」，使用方又要去猜该连哪个 endpoint（首单已踩）。
    for v in m.values():
        if isinstance(v, dict) and v.get("bucket") == display:
            return v.get("region", ""), display
    if p.slug == accounts.DEFAULT_SLUG:
        from core.oss_perm.permsync import BUCKET_MAP
        if display in BUCKET_MAP:
            region, bucket = BUCKET_MAP[display]
            return region, bucket
        for region, bucket in BUCKET_MAP.values():
            if bucket == display:
                return region, bucket
    # 映射表里查不到：当真实桶名用，并**实时探一次地域**。新桶不必先维护映射表，
    # 这正是线上那单的成因 —— wuji-rl-dataset 不在表里，region 留空，凭证正文
    # 三行连接信息退化成「未知」，使用方随手用了默认 endpoint 拿到 403。
    return probe_bucket_region(display, p), display


def _derive_user_name(spec: dict, instance_code: str, profile=None) -> str:
    """RAM 登录名：<账号前缀><主体名ASCII化/否则ext>-<实例短hash>。

    RAM user_name 只能 [A-Za-z0-9.@_-]，中文主体名 ASCII 化后可能为空 → 退回 ext（唯一性靠 hash）；
    可读的主体名放 display_name（见 issuer._issue_ram，可中文）。
    主体 = 默认档的「使用企业名称」/ 1949 档的「使用人名称」，逻辑键统一为 enterprise。
    """
    p = profile or accounts.default()
    ent = spec.get("enterprise", "") or spec.get("recipient_email", "").split("@")[0]
    slug = _ascii_slug(ent)
    short = hashlib.md5((instance_code or slug).encode("utf-8")).hexdigest()[:6]
    return f"{p.user_prefix}{slug}-{short}"


def _ascii_slug(text: str) -> str:
    """企业名 → RAM 登录名可用的 ASCII slug。中文转拼音（pypinyin，未装则降级）；都取不到 → 'ext'。"""
    text = (text or "").strip()
    if not text:
        return "ext"
    try:
        from pypinyin import lazy_pinyin
        text = "".join(lazy_pinyin(text))   # 中文→拼音；ASCII 原样保留
    except Exception:
        pass                                 # 未装 pypinyin：仅保留原串里的 ASCII
    slug = re.sub(r"[^A-Za-z0-9]", "", text).lower()[:20]
    return slug or "ext"


def display_name_for(grant: dict) -> str:
    """RAM 控制台显示名（可中文，一眼看出是哪个主体、哪个账号的临时号）。后缀按账号档案取。"""
    ent = (grant.get("enterprise") or "").strip()
    suffix = _profile_of(grant.get("grant_id", "")).display_suffix
    return (f"{ent}{suffix}" if ent else suffix.lstrip("-"))[:128]


# ── grant 生命周期 ────────────────────────────────────────────────────────────

def create_grant_record(spec: dict, *, instance_code: str, requester: str = "",
                        approver: str = "", profile=None) -> dict:
    """据审批表单 spec 建 grant 记录（幂等：同实例返回已有记录）。

    spec: {bucket(display), region?, prefix, caps⊆{read,download,write}, not_before, expire,
           recipient_email, source_ips?, reason?, note?}
    profile: 账号档案，决定凭证ID/Redis 前缀/桶表/命名；不传 = 现有主账号。
    """
    p = profile or accounts.default()
    gid = grant_id_for(instance_code, p)
    existing = get_grant(gid)
    if existing:
        return existing

    now = time.time()
    region, real_bucket = resolve_bucket(spec["bucket"], p)
    mode = issuer.classify_mode(spec["expire"], now, p)   # 非默认账号强制 RAM（STS 角色属默认账号）
    user_name = _derive_user_name(spec, instance_code, p)
    grant = {
        "grant_id": gid,
        "account": p.slug,          # 账号维度：延期/撤销/到期清理据此取对应账号的凭证
        "stage": STAGE_NEW,
        "mode": mode,
        "platform": spec.get("platform", "aliyun"),
        "enterprise": spec.get("enterprise", ""),
        "bucket": real_bucket,
        "bucket_display": spec["bucket"],
        "region": spec.get("region") or region,
        "prefix": spec.get("prefix", ""),
        "caps": list(spec.get("caps") or []),
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
        "note": spec.get("note", ""),        # 「备注」栏（1949 档模板新增），仅记录+回执，不参与授权
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
    grant["error"] = ""     # 清掉上一轮失败原因：否则 FAILED 重试成功后状态卡仍显示「失败原因」
    _save(grant)
    return grant, creds


def fail_grant(grant: dict, error: str) -> None:
    grant["stage"] = STAGE_FAILED
    grant["error"] = str(error)[:500]
    _save(grant)


def extend_grant(grant: dict, not_before, expire, *, extend_instance: str = "") -> tuple[dict, dict | None]:
    """延期一个已发放的 grant（仅 ISSUED 可延）。返回 (grant, creds)。

    方案 B（长期 AK）：只改写 policy 时间窗，AK/SK 不变 → creds=None（不重发）。
    STS：原 token 已自灭，按新窗重签发（新窗 >12h 自动转方案 B）→ creds=新凭证（需重发）。
    """
    if grant["stage"] != STAGE_ISSUED:
        raise TempAkError(f"凭证当前状态 {grant['stage']}，仅 ISSUED 可延期")
    now = time.time()
    if float(expire) <= now:
        raise TempAkError("新到期时间已过")
    if not_before and float(not_before) >= float(expire):
        raise TempAkError("新生效时间必须早于到期时间")
    if not not_before:
        not_before = grant.get("not_before") or now
    old_expire = grant.get("expire")
    grant["not_before"] = float(not_before)
    grant["expire"] = float(expire)

    if grant["mode"] == issuer.RAM_MODE:
        issuer.rewrite_ram_window(grant)          # 同 AK 改写时间窗，不重发凭证
        creds = None
    else:  # STS：原 token 已自灭 → 按新窗重签发（新窗 >12h 自动转方案 B 发长期 AK）
        grant["mode"] = issuer.classify_mode(grant["expire"], now, _profile_of(grant.get("grant_id", "")))
        if grant["mode"] == issuer.RAM_MODE and not grant.get("policy_name"):
            grant["policy_name"] = issuer.policy.POLICY_PREFIX + grant["user_name"]
        creds = issuer.issue(grant)
        if grant["mode"] == issuer.RAM_MODE:
            grant["ak_id"] = creds.get("access_key_id", "")

    grant.setdefault("extends", []).append(
        {"at": now, "old_expire": old_expire, "new_expire": grant["expire"]})
    if extend_instance:
        grant.setdefault("extend_instances", []).append(extend_instance)
    _save(grant)
    return grant, creds


# ── 展示助手（供卡片/CLI/工具）──────────────────────────────────────────────────

def fmt_ts(epoch: float) -> str:
    if not epoch:
        return "-"
    return datetime.fromtimestamp(int(epoch), tz=_BJ).strftime("%Y-%m-%d %H:%M:%S")


def fmt_window(grant: dict) -> str:
    return f"{fmt_ts(grant.get('not_before'))} → {fmt_ts(grant.get('expire'))}"


_CAP_CN = {"read": "列", "download": "下载", "write": "上传"}


def scope_line(grant: dict) -> str:
    caps = "/".join(_CAP_CN.get(c, c) for c in (grant.get("caps") or [])) or "—"
    p = grant.get("prefix") or "<整桶>"
    return f"桶 `{grant.get('bucket')}` 目录 `{p}` 权限[{caps}]"
