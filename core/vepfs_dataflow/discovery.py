"""枚举 vePFS 文件系统 + TOS 桶，做成下拉选项（供向导卡）。

- fs_options：对配置涉及的每个 region 调 vePFS DescribeFileSystems。
- bucket_options：TOS list_buckets 按 region 过滤。
结果缓存 Redis vepfs:dataflow:optmap（默认 30 分钟）。option.value 形如 `<id>@<region>`，
卡片回传后 split('@') 即得定位信息。发现失败/为空时调用方回退成文本输入框。
"""
import json
import logging

from config.settings import settings
from utils.redis_client import get_redis
from core.vepfs_dataflow import engine_vepfs

logger = logging.getLogger(__name__)

_MAP_KEY = "vepfs:dataflow:optmap"
_TTL = 30 * 60


def _regions() -> list[str]:
    """需要扫描的 region 集合：VEPFS_FILE_SYSTEM_IDS 里出现的 region ∪ VEPFS_REGION。"""
    regions = []
    raw = (getattr(settings, "VEPFS_FILE_SYSTEM_IDS", "") or "").strip()
    for item in raw.split(","):
        item = item.strip()
        if "@" in item:
            r = item.split("@", 1)[1].strip()
            if r:
                regions.append(r)
    default = settings.VEPFS_REGION or "cn-shanghai"
    if default not in regions:
        regions.append(default)
    # 去重保序
    seen, out = set(), []
    for r in regions:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _discover() -> dict:
    """实时发现 {fs: [...], buckets: [...]}。每项 {value, text, region}。"""
    regions = _regions()
    fs_opts, bkt_opts = [], []
    for region in regions:
        try:
            for fs in engine_vepfs.list_filesystems(region):
                fid, name = fs["fs_id"], fs.get("name") or fs["fs_id"]
                fs_opts.append({"value": f"{fid}@{region}", "text": f"{name}（{region}）", "region": region})
        except Exception as e:
            logger.warning("[VEPFS] 发现 fs 失败 region=%s: %s", region, e)
    # TOS 桶：一次 list_buckets 全拿，按 region 过滤到 fs 涉及的 region
    try:
        from utils.volcano_client_factory import get_tos_client
        c = get_tos_client()
        if c is not None:
            r = c.list_buckets()
            for b in (getattr(r, "buckets", None) or []):
                name = getattr(b, "name", "")
                loc = getattr(b, "location", "") or getattr(b, "region", "")
                if name and loc in regions:
                    bkt_opts.append({"value": f"{name}@{loc}", "text": f"{name}（{loc}）", "region": loc})
    except Exception as e:
        logger.warning("[VEPFS] 发现 TOS 桶失败: %s", e)
    return {"fs": fs_opts, "buckets": bkt_opts}


def refresh() -> dict:
    data = _discover()
    try:
        get_redis().setex(_MAP_KEY, _TTL, json.dumps(data, ensure_ascii=False))
    except Exception:
        logger.warning("[VEPFS] 写发现缓存失败")
    return data


def _cached(refresh_if_empty: bool = True) -> dict:
    try:
        raw = get_redis().get(_MAP_KEY)
        if raw:
            d = json.loads(raw)
            if isinstance(d, dict) and (d.get("fs") or d.get("buckets")):
                return d
    except Exception:
        pass
    if refresh_if_empty:
        return refresh()
    return {"fs": [], "buckets": []}


def fs_options() -> list[dict]:
    return _cached().get("fs", [])


def bucket_options() -> list[dict]:
    return _cached().get("buckets", [])
