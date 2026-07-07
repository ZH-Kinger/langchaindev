"""维护飞书「数据集大盘」多维表格：遍历现有行，按 uri 扫对象存储，回填脚本负责列。

**只填数据、不改表结构、绝不碰人工分析列。** 与 capacity_bitable（另一张 3 表）完全独立。

脚本负责列（写入策略）：
  状态         —— 脚本独占、每次刷新：路径存在→在库 / 列不出对象→已消失（消失只标记不删行）
  云 / 厂商|来源 —— 仅当为空时从 uri 拆填
  时长          —— 仅当为空时，从路径里的 `\\d+h` token 解析才填
  数据集类型     —— 仅当为空且能明确判定（lerobot 目录布局）才填
其余列一律不出现在 update fields 里 → 人工分析列（剖析结论/mono偏移/跳变率/坏帧率/负责人/…）绝不被动。

cpfs:// 或无法识别 scheme 的行：跳过存在性判断（不改状态，避免误标已消失）。
凭证：OSS 走全局 AK（open_id=""，同容量巡检降级路径）；TOS 走静态 AK。
"""
import re
import time

import requests

from config.settings import settings
from utils.logger import get_logger
from core.capacity_bitable import _tenant_token  # 复用租户 token（通用，用 FEISHU_APP_ID/SECRET）

logger = get_logger(__name__)

_BASE = "https://open.feishu.cn/open-apis"
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*h", re.IGNORECASE)
_URI_RE = re.compile(r"^(oss|tos|cpfs|vepfs)://([^/]+)/?(.*)$", re.IGNORECASE)

_F_STATUS = "状态"
_F_CLOUD = "云"
_F_VENDOR = "厂商|来源"
_F_HOURS = "时长"
_F_DTYPE = "数据集类型"
_ST_IN, _ST_GONE = "在库", "已消失"
_CLOUD_BY_SCHEME = {"oss": "OSS", "tos": "TOS", "cpfs": "CPFS", "vepfs": "vePFS"}


# ── 配置 / Bitable 原语（指向数据集大盘表，不动 capacity_bitable）─────────────────
def _app_token() -> str:
    return getattr(settings, "DATASET_DASHBOARD_APP_TOKEN", "")


def _table_id() -> str:
    return getattr(settings, "DATASET_DASHBOARD_TABLE_ID", "")


def _is_configured() -> bool:
    return bool(_app_token() and _table_id())


def _records_url() -> str:
    return f"{_BASE}/bitable/v1/apps/{_app_token()}/tables/{_table_id()}/records"


def _as_text(v) -> str:
    """把字段值（文本 / 列表 / dict）统一取纯文本。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            return v[0].get("text", "")
        return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        return v.get("text", "")
    return str(v)


def _list_records(headers: dict) -> list:
    out, pt = [], ""
    while True:
        params = {"page_size": 500}
        if pt:
            params["page_token"] = pt
        r = requests.get(_records_url(), headers=headers, params=params, timeout=20)
        d = r.json().get("data", {})
        out.extend(d.get("items", []))
        pt = d.get("page_token", "")
        if not d.get("has_more"):
            break
    return out


def _update_one(headers: dict, record_id: str, fields: dict) -> bool:
    r = requests.put(f"{_records_url()}/{record_id}", headers=headers,
                     json={"fields": fields}, timeout=15)
    j = r.json()
    if j.get("code") != 0:
        logger.error("[DatasetDash] 更新失败 rid=%s code=%s msg=%s", record_id, j.get("code"), j.get("msg"))
        return False
    return True


# ── uri 解析 ─────────────────────────────────────────────────────────────────────
def _parse_uri(uri: str):
    """oss://bucket/prefix → (scheme, bucket, prefix)。无法识别返回 None。"""
    m = _URI_RE.match((uri or "").strip())
    if not m:
        return None
    return m.group(1).lower(), m.group(2), m.group(3).strip("/")


def _dir_prefix(prefix: str) -> str:
    p = (prefix or "").strip("/")
    return f"{p}/" if p else ""


def _hours_from(prefix: str):
    m = _HOURS_RE.search(prefix or "")
    if not m:
        return None
    h = float(m.group(1))
    return f"{int(h)}h" if h.is_integer() else f"{h}h"


# ── 存在性 + 轻量属性（max_keys 小，避免枚举大目录）──────────────────────────────
def _oss_probe(bucket: str, prefix: str):
    """返回 (exists: bool|None, subdir_names: list)。creds 不可用返回 (None, [])。"""
    try:
        from tools.aliyun import oss
        b = oss._resolve_bucket("", bucket)
        if b is None:
            return None, []
        dp = _dir_prefix(prefix)
        r = b.list_objects_v2(prefix=dp, delimiter="/", max_keys=100)
        subdirs = [p.rstrip("/").split("/")[-1] for p in getattr(r, "prefix_list", [])]
        has_obj = any(not (o.key.endswith("/") and o.size == 0) for o in getattr(r, "object_list", []))
        return (has_obj or bool(subdirs)), subdirs
    except Exception:
        logger.warning("[DatasetDash] OSS 探测失败 bucket=%s prefix=%s", bucket, prefix, exc_info=True)
        return None, []


def _tos_probe(bucket: str, prefix: str):
    try:
        from utils.volcano_client_factory import get_tos_client
        client = get_tos_client()
        if client is None:
            return None, []
        dp = _dir_prefix(prefix)
        r = client.list_objects_type2(bucket, prefix=dp, delimiter="/", max_keys=100)
        subdirs = [cp.prefix.rstrip("/").split("/")[-1] for cp in getattr(r, "common_prefixes", [])]
        has_obj = any(not (o.key.endswith("/") and o.size == 0) for o in getattr(r, "contents", []))
        return (has_obj or bool(subdirs)), subdirs
    except Exception:
        logger.warning("[DatasetDash] TOS 探测失败 bucket=%s prefix=%s", bucket, prefix, exc_info=True)
        return None, []


def _detect_dataset_type(subdirs: list):
    """轻量判定：含 lerobot 布局标志(meta/data) → 'lerobot'；否则不猜（返回 ''）。"""
    s = {x.lower() for x in subdirs}
    if "meta" in s and "data" in s:
        return "lerobot"
    return ""


# ── 单行：计算要写的脚本列（只含脚本负责列）────────────────────────────────────────
def compute_updates(fields: dict) -> dict:
    uri = _as_text(fields.get("uri"))
    parsed = _parse_uri(uri)
    if not parsed:
        return {}
    scheme, bucket, prefix = parsed
    updates: dict = {}

    # 云 / 厂商|来源：仅填空白
    cloud = _CLOUD_BY_SCHEME.get(scheme)
    if cloud and not _as_text(fields.get(_F_CLOUD)):
        updates[_F_CLOUD] = cloud
    vendor = prefix.split("/")[0] if prefix else ""
    if vendor and not _as_text(fields.get(_F_VENDOR)):
        updates[_F_VENDOR] = vendor

    # 时长：仅填空白
    if not _as_text(fields.get(_F_HOURS)):
        h = _hours_from(prefix)
        if h:
            updates[_F_HOURS] = h

    # 存在性（仅 oss/tos）→ 状态每次刷新；顺带拿子目录判数据集类型
    exists, subdirs = (None, [])
    if scheme == "oss":
        exists, subdirs = _oss_probe(bucket, prefix)
    elif scheme == "tos":
        exists, subdirs = _tos_probe(bucket, prefix)
    if exists is True:
        updates[_F_STATUS] = _ST_IN
    elif exists is False:
        updates[_F_STATUS] = _ST_GONE
    # exists None（creds 不可用 / cpfs / 探测异常）→ 不动状态

    # 数据集类型：仅填空白且能判定
    if exists and not _as_text(fields.get(_F_DTYPE)):
        dt = _detect_dataset_type(subdirs)
        if dt:
            updates[_F_DTYPE] = dt

    return updates


# ── 主入口 ─────────────────────────────────────────────────────────────────────
def run_once(dry_run: bool = False, limit: int = None) -> dict:
    """遍历数据集大盘现有行，回填脚本负责列。dry_run=True 只算不写，返回预览。"""
    if not _is_configured():
        logger.warning("[DatasetDash] 未配置 app_token/table_id，跳过")
        return {"ok": False, "reason": "not_configured"}
    token = _tenant_token()
    if not token:
        logger.error("[DatasetDash] 获取 tenant_access_token 失败")
        return {"ok": False, "reason": "no_token"}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    rows = _list_records(headers)
    if limit:
        rows = rows[:limit]
    stats = {"total": len(rows), "in_stock": 0, "gone": 0, "filled": 0, "written": 0, "skipped": 0}
    preview = []
    for rec in rows:
        f = rec.get("fields", {})
        rid = rec.get("record_id", "")
        upd = compute_updates(f)
        if not upd:
            stats["skipped"] += 1
            continue
        if upd.get(_F_STATUS) == _ST_IN:
            stats["in_stock"] += 1
        elif upd.get(_F_STATUS) == _ST_GONE:
            stats["gone"] += 1
        if any(k != _F_STATUS for k in upd):
            stats["filled"] += 1
        if dry_run:
            preview.append({"uri": _as_text(f.get("uri")), "updates": upd})
        elif _update_one(headers, rid, upd):
            stats["written"] += 1
    stats["ok"] = True
    if dry_run:
        stats["preview"] = preview
    logger.info("[DatasetDash] %s：共%d 在库%d 已消失%d 补列%d 写入%d 跳过%d",
                "预览" if dry_run else "维护", stats["total"], stats["in_stock"],
                stats["gone"], stats["filled"], stats["written"], stats["skipped"])
    return stats
