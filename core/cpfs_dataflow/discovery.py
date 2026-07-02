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


def _discover_from_pairs(pairs: list[tuple[str, str]], open_id: str = "") -> list[dict]:
    options: list[dict] = []
    for fs_id, region in pairs:
        try:
            for flow in engine_nas.list_dataflows(fs_id, region, open_id=open_id):
                bucket, _ = _split_source_storage(flow.get("source_storage", ""))
                if _excluded_bucket(bucket):
                    continue   # cri 开头=镜像仓库，跳过
                options.append(_option(region, fs_id, flow))
        except Exception as e:
            logger.warning("[CPFS] 枚举 DataFlow 失败 fs=%s region=%s: %s", fs_id, region, e)
    return options


def discover(open_id: str = "") -> list[dict]:
    """枚举所有 fs 的 DataFlow 绑定，返回 options 列表（兼容 CLI/旧调用）。"""
    return _discover_from_pairs(_fs_pairs(open_id=open_id), open_id=open_id)


def refresh(open_id: str = "") -> list[dict]:
    """重新发现并写缓存。缓存含 options（绑定）+ filesystems（所有 CPFS，含无绑定的，带名称）。"""
    pairs = _fs_pairs(open_id=open_id)
    name_by_region: dict[str, dict] = {}
    for _f, r in pairs:
        if r not in name_by_region:
            try:
                name_by_region[r] = engine_nas.filesystem_names(r, open_id=open_id)
            except Exception:
                name_by_region[r] = {}
    filesystems = [{"region": r, "fs_id": f, "edition": engine_nas.edition(f),
                    "name": name_by_region.get(r, {}).get(f, "")} for f, r in pairs]
    options = _discover_from_pairs(pairs, open_id=open_id)
    try:
        get_redis().setex(_MAP_KEY, settings.CPFS_MAP_TTL_SECONDS,
                          json.dumps({"options": options, "filesystems": filesystems}, ensure_ascii=False))
    except Exception:
        logger.warning("[CPFS] 写 DataFlow 映射缓存失败")
    return options


def _cached(open_id: str = "", refresh_if_empty: bool = True) -> dict:
    try:
        raw = get_redis().get(_MAP_KEY)
        if raw:
            d = json.loads(raw)
            if isinstance(d, dict):
                return d
            if isinstance(d, list):     # 兼容旧版纯 options 缓存
                return {"options": d, "filesystems": []}
    except Exception:
        pass
    if refresh_if_empty:
        refresh(open_id=open_id)
        try:
            raw = get_redis().get(_MAP_KEY)
            if raw:
                d = json.loads(raw)
                return d if isinstance(d, dict) else {"options": d, "filesystems": []}
        except Exception:
            pass
    return {"options": [], "filesystems": []}


def get_options(refresh_if_empty: bool = True, open_id: str = "") -> list[dict]:
    """DataFlow 绑定选项（有绑定的 fs 才有）。"""
    return _cached(open_id, refresh_if_empty).get("options", [])


def get_filesystems(open_id: str = "") -> list[dict]:
    """所有扫描到的 CPFS 文件系统（含无 DataFlow 的），用于地区/文件系统下拉。"""
    return _cached(open_id).get("filesystems", [])


def regions(open_id: str = "") -> list[str]:
    """有 CPFS 文件系统的地区列表（含暂无 DataFlow 的，供级联第一步）。"""
    return sorted({f["region"] for f in get_filesystems(open_id=open_id) if f.get("region")})


def filesystems_in(region: str, open_id: str = "") -> list[dict]:
    """某地区下的 CPFS 文件系统（去重），含 edition。"""
    seen: dict[str, dict] = {}
    for f in get_filesystems(open_id=open_id):
        if f.get("region") == region and f.get("fs_id") and f["fs_id"] not in seen:
            seen[f["fs_id"]] = {"fs_id": f["fs_id"], "edition": f.get("edition", "")}
    return list(seen.values())


def buckets_in(region: str, open_id: str = "") -> list[str]:
    """某地区下被 DataFlow 绑定的 OSS 桶（去重）。"""
    return sorted({o["oss_bucket"] for o in get_options(open_id=open_id)
                   if o.get("region") == region and o.get("oss_bucket")})


def cpfs_select_options(open_id: str = "") -> list[dict]:
    """录入卡「选 CPFS」下拉：每项含地区+名称+fs-id，value 编码 region+fs_id。
    地区与 CPFS 合成一个下拉（飞书卡片做不到联动过滤），选中即定地区。"""
    opts: list[dict] = []
    seen: set[str] = set()
    for f in get_filesystems(open_id=open_id):
        fid, region = f.get("fs_id"), f.get("region")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        name = f.get("name") or ""
        label = f"{region} · {name} ({fid})" if name else f"{region} · {fid}"
        opts.append({"text": {"tag": "plain_text", "content": label[:100]},
                     "value": json.dumps({"region": region, "fs_id": fid}, ensure_ascii=False)})
    return opts


def decode_selection(value: str) -> dict:
    """卡片 select 回传的 value(JSON) → dict。非法返回 {}。"""
    try:
        d = json.loads(value)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}
