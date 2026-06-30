"""枚举所有 CPFS↔OSS 的 DataFlow 绑定，做成映射表（供卡片下拉选择）。

来源：CPFS_FILE_SYSTEM_IDS 配置的 fs 清单（'fs_id@region,...'），对每个 fs 调
DescribeDataFlows 读出其 OSS 绑定。结果缓存到 Redis cpfs:dataflow:map（默认 6h）。

每个选项（option）：{region, fs_id, edition, data_flow_id, oss_bucket, oss_prefix,
fs_path, label, value}；value 是 JSON 串，卡片 select 回传后 decode 即得全部定位信息。
"""
import json
import logging

from config.settings import settings
from utils.redis_client import get_redis
from core.cpfs_dataflow import engine_nas

logger = logging.getLogger(__name__)

_MAP_KEY = "cpfs:dataflow:map"

# 屏蔽镜像仓库桶：cri 开头的 OSS bucket 是镜像仓库，不是数据桶，不进映射表。
_EXCLUDE_BUCKET_PREFIXES = ("cri",)


def _excluded_bucket(bucket: str) -> bool:
    b = (bucket or "").lower()
    return any(b.startswith(p) for p in _EXCLUDE_BUCKET_PREFIXES)


def parse_fs_list() -> list[tuple[str, str]]:
    """解析 CPFS_FILE_SYSTEM_IDS → [(fs_id, region)]。兼容单个 CPFS_FILE_SYSTEM_ID。"""
    raw = (getattr(settings, "CPFS_FILE_SYSTEM_IDS", "") or "").strip()
    default_region = settings.CPFS_REGION or "cn-hangzhou"
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "@" in item:
            fs, region = item.split("@", 1)
            out.append((fs.strip(), region.strip() or default_region))
        else:
            out.append((item, default_region))
    if not out and settings.CPFS_FILE_SYSTEM_ID:
        out.append((settings.CPFS_FILE_SYSTEM_ID, default_region))
    return out


def _option(region: str, fs_id: str, flow: dict) -> dict:
    bucket, prefix = _split_source_storage(flow.get("source_storage", ""))
    fs_path = flow.get("fs_path") or "/"
    label = f"{region}·{fs_id}·oss://{bucket}/{prefix} ↔ {fs_path}"
    value = {
        "region": region, "fs_id": fs_id, "data_flow_id": flow.get("data_flow_id", ""),
        "oss_bucket": bucket, "oss_prefix": prefix, "fs_path": fs_path,
    }
    return {
        "region": region, "fs_id": fs_id, "edition": engine_nas.edition(fs_id),
        "data_flow_id": flow.get("data_flow_id", ""),
        "oss_bucket": bucket, "oss_prefix": prefix, "fs_path": fs_path,
        "label": label[:120], "value": json.dumps(value, ensure_ascii=False),
    }


def _split_source_storage(src: str) -> tuple[str, str]:
    """'oss://[acct:]bucket/prefix' → (bucket, prefix)。"""
    s = (src or "").strip()
    if s.startswith("oss://"):
        s = s[len("oss://"):]
    if ":" in s.split("/", 1)[0]:          # 跨账号 acct:bucket
        s = s.split(":", 1)[1]
    bucket, _, prefix = s.partition("/")
    return bucket, prefix


def parse_regions() -> list[str]:
    raw = (getattr(settings, "CPFS_REGIONS", "") or "").strip()
    return [r.strip() for r in raw.split(",") if r.strip()]


def _fs_pairs(open_id: str = "") -> list[tuple[str, str]]:
    """优先用显式 CPFS_FILE_SYSTEM_IDS；否则按 CPFS_REGIONS 扫描 DescribeFileSystems 枚举。"""
    explicit = parse_fs_list()
    if explicit:
        return explicit
    pairs: list[tuple[str, str]] = []
    for region in parse_regions():
        try:
            for fs_id in engine_nas.list_filesystems(region, open_id=open_id):
                pairs.append((fs_id, region))
        except Exception as e:
            logger.warning("[CPFS] 枚举文件系统失败 region=%s: %s", region, e)
    return pairs


def discover(open_id: str = "") -> list[dict]:
    """枚举所有 fs 的 DataFlow，返回 options 列表。单 fs/地域失败不影响其它。"""
    options: list[dict] = []
    for fs_id, region in _fs_pairs(open_id=open_id):
        try:
            for flow in engine_nas.list_dataflows(fs_id, region, open_id=open_id):
                bucket, _ = _split_source_storage(flow.get("source_storage", ""))
                if _excluded_bucket(bucket):
                    continue   # cri 开头=镜像仓库，跳过
                options.append(_option(region, fs_id, flow))
        except Exception as e:
            logger.warning("[CPFS] 枚举 DataFlow 失败 fs=%s region=%s: %s", fs_id, region, e)
    return options


def refresh(open_id: str = "") -> list[dict]:
    """重新发现并写缓存。"""
    options = discover(open_id=open_id)
    try:
        get_redis().setex(_MAP_KEY, settings.CPFS_MAP_TTL_SECONDS,
                          json.dumps(options, ensure_ascii=False))
    except Exception:
        logger.warning("[CPFS] 写 DataFlow 映射缓存失败")
    return options


def get_options(refresh_if_empty: bool = True, open_id: str = "") -> list[dict]:
    """读缓存映射表；空且允许则实时发现并回填。"""
    try:
        raw = get_redis().get(_MAP_KEY)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    if refresh_if_empty:
        return refresh(open_id=open_id)
    return []


def regions(open_id: str = "") -> list[str]:
    """有可用 DataFlow 绑定的地区列表（供级联第一步）。"""
    return sorted({o["region"] for o in get_options(open_id=open_id) if o.get("region")})


def filesystems_in(region: str, open_id: str = "") -> list[dict]:
    """某地区下的 CPFS 文件系统（去重），含 edition。"""
    seen: dict[str, dict] = {}
    for o in get_options(open_id=open_id):
        if o.get("region") == region and o.get("fs_id") and o["fs_id"] not in seen:
            seen[o["fs_id"]] = {"fs_id": o["fs_id"], "edition": o.get("edition", "")}
    return list(seen.values())


def buckets_in(region: str, open_id: str = "") -> list[str]:
    """某地区下被 DataFlow 绑定的 OSS 桶（去重）。"""
    return sorted({o["oss_bucket"] for o in get_options(open_id=open_id)
                   if o.get("region") == region and o.get("oss_bucket")})


def decode_selection(value: str) -> dict:
    """卡片 select 回传的 value(JSON) → dict。非法返回 {}。"""
    try:
        d = json.loads(value)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}
