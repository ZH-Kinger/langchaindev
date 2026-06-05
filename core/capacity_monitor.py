"""
容量巡检：定时盘点 OSS + 火山 TOS 指定父目录下各子目录大小，
带「较上次的增量」推送飞书卡片，超阈值标红。

补上桌面采集脚本缺失的「主动推送」：脚本只写本地 CSV，这里接上飞书推送链路。

数据来源复用工具层的 compute_dir_sizes：
  - OSS → tools.aliyun.oss.compute_dir_sizes(open_id="")（无用户上下文，工厂降级全局 AK）
  - TOS → tools.volcano.tos.compute_dir_sizes（静态 AK）

快照存 Redis：capacity:snapshot:{vendor}:{bucket}:{prefix}（TTL 30 天），
下次巡检据此算每个子目录的增量。

由 core.dsw_scheduler 的后台线程按 CAPACITY_MONITOR_INTERVAL_HOURS 调用 run_capacity_scan()。
"""
import json
import time

from config.settings import settings
from utils.logger import get_logger
from utils import redis_client

logger = get_logger(__name__)

_SNAPSHOT_PREFIX = "capacity:snapshot:"
_SNAPSHOT_TTL = 30 * 86400  # 30 天
_BATCH_CACHE_PREFIX = "capacity:batch:"
_BATCH_CACHE_TTL = 180 * 86400  # 180 天：已交付批次大小缓存

_TB = 1024 ** 4
_GB = 1024 ** 3


def _fmt_size(num_bytes: float) -> str:
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_delta(delta_bytes: float) -> str:
    """子目录较上次的增量，None 上次无记录则空。"""
    if delta_bytes == 0:
        return "—"
    sign = "+" if delta_bytes > 0 else "-"
    return f"{sign}{_fmt_size(abs(delta_bytes))}"


# ── 快照 ──────────────────────────────────────────────────────────────────────

def _snapshot_key(vendor: str, bucket: str, prefix: str) -> str:
    return f"{_SNAPSHOT_PREFIX}{vendor}:{bucket}:{prefix}"


def _load_snapshot(vendor: str, bucket: str, prefix: str) -> dict:
    """返回 {subdir_name: size_bytes}；无记录或 Redis 不可用返回 {}。"""
    try:
        raw = redis_client.get_redis().get(_snapshot_key(vendor, bucket, prefix))
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _save_snapshot(vendor: str, bucket: str, prefix: str, sizes: dict) -> None:
    try:
        redis_client.get_redis().set(_snapshot_key(vendor, bucket, prefix),
                                     json.dumps(sizes), ex=_SNAPSHOT_TTL)
    except Exception:
        logger.warning("[Capacity] 快照写入失败 %s/%s", bucket, prefix, exc_info=True)


# ── 批次大小缓存（已交付批次不再重扫）────────────────────────────────────────

def _batch_cache_key(vendor: str, bucket: str, prefix: str) -> str:
    return f"{_BATCH_CACHE_PREFIX}{vendor}:{bucket}:{prefix}"


def _load_batch_cache(vendor: str, bucket: str, prefix: str) -> dict:
    """返回 {"厂家/批次": (bytes, count)}；关闭/无记录/Redis 不可用返回 {}。"""
    if not settings.CAPACITY_BATCH_CACHE_ENABLED:
        return {}
    try:
        raw = redis_client.get_redis().hgetall(_batch_cache_key(vendor, bucket, prefix))
        out = {}
        for k, v in (raw or {}).items():
            b, _, c = v.partition(":")
            out[k] = (int(b), int(c))
        return out
    except Exception:
        return {}


def _save_family_cache(vendor: str, bucket: str, prefix: str, vendor_name: str, batches: list) -> None:
    """单个厂家扫完即增量写入缓存（断点续传：中断后已完成厂家下次跳过）。排除 '/'。"""
    if not settings.CAPACITY_BATCH_CACHE_ENABLED:
        return
    mapping = {f"{vendor_name}/{name}": f"{size}:{count}"
               for name, size, count in batches if name != "/"}
    if not mapping:
        return
    try:
        r = redis_client.get_redis()
        key = _batch_cache_key(vendor, bucket, prefix)
        r.hset(key, mapping=mapping)
        r.expire(key, _BATCH_CACHE_TTL)
    except Exception:
        logger.warning("[Capacity] 厂家缓存写入失败 %s/%s/%s", bucket, prefix, vendor_name, exc_info=True)


def _save_batch_cache(vendor: str, bucket: str, prefix: str, entries: list) -> None:
    """把本次各批次大小写回缓存（排除每次都实扫的 '/'）。整表重写以清掉已消失的批次。"""
    if not settings.CAPACITY_BATCH_CACHE_ENABLED:
        return
    mapping = {}
    for e in entries:
        for name, size, count in e["batches"]:
            if name == "/":
                continue
            mapping[f"{e['厂家']}/{name}"] = f"{size}:{count}"
    if not mapping:
        return
    try:
        r = redis_client.get_redis()
        key = _batch_cache_key(vendor, bucket, prefix)
        r.delete(key)
        r.hset(key, mapping=mapping)
        r.expire(key, _BATCH_CACHE_TTL)
    except Exception:
        logger.warning("[Capacity] 批次缓存写入失败 %s/%s", bucket, prefix, exc_info=True)


# ── 取数 ──────────────────────────────────────────────────────────────────────

def _fetch_nested(target: dict, cached: dict = None, on_family=None):
    """按 vendor 调两级遍历，返回 entries=[{厂家,total_bytes,total_count,batches}] 或 None。

    cached: 已交付批次大小缓存，命中则跳过实扫。
    on_family(vendor_name, batches): 每扫完一个厂家回调（断点续传逐厂家落库）。
    """
    vendor = target.get("vendor", "").lower()
    bucket = target.get("bucket", "")
    prefix = target.get("prefix", "")
    if vendor == "oss":
        from tools.aliyun.oss import compute_nested_sizes
        entries, _endpoint = compute_nested_sizes(
            open_id="", bucket=bucket, prefix=prefix,
            region=target.get("region", ""), cached=cached, on_family=on_family)
        return entries
    if vendor == "tos":
        from tools.volcano.tos import compute_nested_sizes
        return compute_nested_sizes(bucket, prefix, cached=cached, on_family=on_family)
    logger.warning("[Capacity] 未知 vendor: %s", vendor)
    return None


# ── 卡片 ──────────────────────────────────────────────────────────────────────

def _target_section(target: dict, rows: list, prev: dict):
    """构造单个 target 在合并卡片里的一节，返回 (elements, over, grand_bytes)。"""
    vendor = target.get("vendor", "").upper()
    bucket = target.get("bucket", "")
    prefix = (target.get("prefix", "") or "").rstrip("/") or "/"

    grand_bytes = sum(r[1] for r in rows)
    grand_objs = sum(r[2] for r in rows)
    grand_prev = sum(prev.values()) if prev else 0
    grand_delta = grand_bytes - grand_prev if prev else 0

    threshold_tb = settings.CAPACITY_ALERT_THRESHOLD_TB
    over = bool(threshold_tb) and grand_bytes > threshold_tb * _TB

    icon = {"OSS": "☁️ 阿里云 OSS", "TOS": "🌋 火山 TOS"}.get(vendor, vendor)
    head = f"**{icon} · `{bucket}`**　父目录 `{prefix}`"
    if over:
        head += f"　⚠️ 超阈值 {threshold_tb:.1f} TB"

    table = ["| 子目录 | 大小 | 对象数 | 较上次 |", "|--------|------|--------|--------|"]
    for name, size_bytes, count in sorted(rows, key=lambda r: -r[1]):
        delta = (size_bytes - prev.get(name, 0)) if prev else 0
        table.append(f"| {name} | {_fmt_size(size_bytes)} | {count} | {_fmt_delta(delta)} |")
    table.append(
        f"| **合计** | **{_fmt_size(grand_bytes)}** | **{grand_objs}** | "
        f"**{_fmt_delta(grand_delta) if prev else '首次'}** |"
    )

    elements = [
        {"tag": "div", "text": {"tag": "lark_md", "content": head}},
        {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(table)}},
    ]
    return elements, over, grand_bytes


def _build_combined_card(sections: list, any_over: bool, grand_total: int) -> dict:
    """把多个 target 的节合并成一张卡片。sections = [elements, ...]。"""
    title = ("🔴 容量告警" if any_over else "📦 容量巡检") + \
            f"（OSS + TOS，合计 {_fmt_size(grand_total)}）"
    elements = []
    for i, sec in enumerate(sections):
        if i:
            elements.append({"tag": "hr"})
        elements.extend(sec)
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title},
                   "template": "red" if any_over else "blue"},
        "elements": elements,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _targets() -> list:
    try:
        targets = json.loads(settings.CAPACITY_MONITOR_TARGETS_RAW)
        return targets if isinstance(targets, list) else []
    except Exception as e:
        logger.error("[Capacity] CAPACITY_MONITOR_TARGETS 解析失败: %s", e)
        return []


def run_capacity_scan() -> None:
    """遍历所有 target 盘点，汇总成一张飞书卡片一次推送。单个 target 失败不影响其余。"""
    targets = _targets()
    if not targets:
        logger.info("[Capacity] 无巡检目标，跳过")
        return

    sections: list = []
    any_over = False
    grand_total = 0
    bitable_rows: list = []   # 跨 target 拉平的「厂家」行，供写多维表格

    for target in targets:
        vendor = target.get("vendor", "")
        bucket = target.get("bucket", "")
        prefix = target.get("prefix", "")
        try:
            batch_cache = _load_batch_cache(vendor, bucket, prefix)
            on_family = (lambda vn, batches:
                         _save_family_cache(vendor, bucket, prefix, vn, batches))
            entries = _fetch_nested(target, cached=batch_cache, on_family=on_family)
            if entries is None:
                logger.warning("[Capacity] %s/%s 取数失败（凭证/SDK 不可用），跳过", bucket, prefix)
                continue
            if not entries:
                logger.info("[Capacity] %s/%s 无子目录，跳过", bucket, prefix)
                continue
            _save_batch_cache(vendor, bucket, prefix, entries)   # 回写批次缓存

            prev = _load_snapshot(vendor, bucket, prefix)
            # 卡片用「厂家」汇总级
            rows = [(e["厂家"], e["total_bytes"], e["total_count"]) for e in entries]
            elements, over, total = _target_section(target, rows, prev)
            sections.append(elements)
            any_over = any_over or over
            grand_total += total

            # 多维表格用「厂家+批次」两级
            vu = vendor.upper()
            parent = (prefix or "").rstrip("/")
            for e in entries:
                prev_bytes = prev.get(e["厂家"]) if prev else None
                delta = (e["total_bytes"] - prev_bytes) if prev_bytes is not None else None
                bitable_rows.append({
                    "云厂商": vu, "Bucket": bucket, "父目录": parent, "厂家": e["厂家"],
                    "total_bytes": e["total_bytes"], "total_count": e["total_count"],
                    "delta_bytes": delta, "batches": e["batches"],
                })

            _save_snapshot(vendor, bucket, prefix,
                           {e["厂家"]: e["total_bytes"] for e in entries})
            logger.info("[Capacity] %s %s/%s 盘点完成（%d 个厂家，合计 %s）",
                        vendor, bucket, prefix, len(entries), _fmt_size(total))
        except Exception as e:
            logger.error("[Capacity] 巡检 %s/%s 失败: %s", bucket, prefix, e, exc_info=True)

    if not sections:
        logger.info("[Capacity] 无可推送内容（全部 target 取数失败或为空）")
        return

    # ── 飞书卡片（厂家汇总级）──────────────────────────────────────────────
    chat_id = settings.CAPACITY_MONITOR_CHAT_ID or settings.FEISHU_CHAT_ID
    from core.dsw_scheduler import _send_card  # 延迟导入避免循环依赖
    card = _build_combined_card(sections, any_over, grand_total)
    _send_card("", chat_id, card)
    logger.info("[Capacity] 已推送合并卡片（%d 个 target，合计 %s）",
                len(sections), _fmt_size(grand_total))

    # ── 多维表格（厂家+批次两级）──────────────────────────────────────────
    if settings.CAPACITY_BITABLE_ENABLED:
        try:
            from core.capacity_bitable import write_scan
            readable = time.strftime("%Y-%m-%d %H:%M") + " 巡检"
            remark = f"合计 {_fmt_size(grand_total)}（{len(bitable_rows)} 个厂家）"
            write_scan(bitable_rows, readable, remark)
        except Exception as e:
            logger.error("[Capacity] 写多维表格失败: %s", e, exc_info=True)
