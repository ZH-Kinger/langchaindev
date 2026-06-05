"""
火山引擎 TOS 对象存储工具（只读）。

凭证走静态 AK（utils.volcano_client_factory），与阿里云 STS 隔离。
能力：
  'dir_sizes'    - 统计某父目录下各一级子目录的大小（容量盘点）
  'list_objects' - 列出某 prefix 下的对象

compute_dir_sizes() 抽成可复用函数，供本工具与 core.capacity_monitor 共用。
"""
import logging

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from utils.volcano_client_factory import get_tos_client

logger = logging.getLogger(__name__)


# ── 可复用核心：各子目录大小 ─────────────────────────────────────────────────

def _list_subdirs(client, bucket: str, prefix: str) -> list:
    """返回 prefix 下的一级子目录（每个以 '/' 结尾）。"""
    subdirs, token = [], ""
    while True:
        r = client.list_objects_type2(bucket, prefix=prefix, delimiter="/",
                                      continuation_token=token, max_keys=1000)
        subdirs.extend(cp.prefix for cp in r.common_prefixes)
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return subdirs


def _prefix_size(client, bucket: str, prefix: str):
    """累加 prefix 下所有对象大小，返回 (总字节数, 对象数)。跳过 0 字节目录占位对象。"""
    total_bytes = count = 0
    token = ""
    while True:
        r = client.list_objects_type2(bucket, prefix=prefix,
                                      continuation_token=token, max_keys=1000)
        for obj in r.contents:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            total_bytes += obj.size
            count += 1
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return total_bytes, count


def compute_dir_sizes(bucket: str, prefix: str = ""):
    """统计 prefix 下各一级子目录的大小。

    返回 rows = [(name, size_bytes, count), ...]；凭证不可用时返回 None。
    供 manage_tos 与 capacity_monitor 共用。调用方负责不存在 client 的情况。
    """
    client = get_tos_client()
    if client is None:
        return None
    prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
    try:
        rows = []
        for sub in _list_subdirs(client, bucket, prefix):
            name = sub[len(prefix):].rstrip("/")
            size_bytes, count = _prefix_size(client, bucket, sub)
            rows.append((name, size_bytes, count))
        return rows
    finally:
        client.close()


def _direct_files_size(client, bucket: str, prefix: str):
    """只统计 prefix 直属文件（不含子目录里的），返回 (字节数, 对象数)。"""
    total_bytes = count = 0
    token = ""
    while True:
        r = client.list_objects_type2(bucket, prefix=prefix, delimiter="/",
                                      continuation_token=token, max_keys=1000)
        for obj in r.contents:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            total_bytes += obj.size
            count += 1
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return total_bytes, count


def _grouped_sizes(client, bucket: str, family_prefix: str) -> dict:
    """一趟连续扫描 family_prefix 下所有对象，按第二层目录(批次)归并求和（火山 TOS 版）。

    返回 {批次名: (bytes, count)}；厂家直属文件归入 '/'。调用次数只与对象数有关。
    """
    sizes: dict = {}
    token = ""
    while True:
        r = client.list_objects_type2(bucket, prefix=family_prefix,
                                      continuation_token=token, max_keys=1000)
        for obj in r.contents:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            rel = obj.key[len(family_prefix):]
            batch = rel.split("/", 1)[0] if "/" in rel else "/"
            slot = sizes.setdefault(batch, [0, 0])
            slot[0] += obj.size
            slot[1] += 1
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return {k: (v[0], v[1]) for k, v in sizes.items()}


_GROUPED_THRESHOLD = 30
_MAX_BATCHES = 200      # 批次数超此值判为「平铺」，只记整体不细分


def _scan_family_tos(client, bucket: str, sub: str, vendor_name: str, cached: dict):
    """扫一个厂家，返回 (batches, 新扫批次数, 是否平铺)。逻辑同 OSS 版。"""
    if f"{vendor_name}/ALL" in cached:
        use_grouped, batch_dirs = True, None
    else:
        fam_has_cache = any(k.startswith(vendor_name + "/") for k in cached)
        if fam_has_cache:
            batch_dirs = _list_subdirs(client, bucket, sub)
            new_dirs = [bd for bd in batch_dirs
                        if f"{vendor_name}/{bd[len(sub):].rstrip('/')}" not in cached]
            use_grouped = len(new_dirs) > _GROUPED_THRESHOLD
        else:
            use_grouped, batch_dirs = True, None

    if use_grouped:
        grouped = _grouped_sizes(client, bucket, sub)
        real = [n for n in grouped if n != "/"]
        if len(real) > _MAX_BATCHES:
            tb = sum(s for s, _ in grouped.values())
            tc = sum(c for _, c in grouped.values())
            return [("ALL", tb, tc)], len(real), True
        return [(n, s, c) for n, (s, c) in grouped.items()], len(real), False

    batches, fresh = [], 0
    for bd in batch_dirs:
        batch_name = bd[len(sub):].rstrip("/")
        ck = f"{vendor_name}/{batch_name}"
        if ck in cached:
            size_bytes, count = cached[ck]
        else:
            size_bytes, count = _prefix_size(client, bucket, bd)
            fresh += 1
        batches.append((batch_name, size_bytes, count))
    root_bytes, root_count = _direct_files_size(client, bucket, sub)
    if root_count:
        batches.append(("/", root_bytes, root_count))
    return batches, fresh, False


def compute_nested_sizes(bucket: str, prefix: str = "", cached: dict = None, on_family=None):
    """两级遍历：prefix 下每个「厂家」及其下「批次」的大小（火山 TOS 版）。

    返回 entries = [{"厂家":, "total_bytes":, "total_count":, "batches":[(批次,bytes,count)]}]；
    厂家直属文件归入 '/'。凭证不可用返回 None。

    cached: {"厂家/批次": (bytes, count)} 已扫过的不可变批次；命中则跳过逐对象求和。
    """
    client = get_tos_client()
    if client is None:
        return None
    prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
    cached = cached or {}
    import time
    try:
        families = _list_subdirs(client, bucket, prefix)
        entries = []
        for idx, sub in enumerate(families, 1):                     # 厂家
            vendor_name = sub[len(prefix):].rstrip("/")
            t0 = time.time()
            batches, fresh, flat = _scan_family_tos(client, bucket, sub, vendor_name, cached)
            total_bytes = sum(s for _, s, _ in batches)
            total_count = sum(c for _, _, c in batches)
            entries.append({
                "厂家": vendor_name,
                "total_bytes": total_bytes,
                "total_count": total_count,
                "batches": batches,
            })
            logger.info("[进度] TOS %s [%d/%d] %s%s：%d 批次(%d 新扫) %.2fGB %d 对象 用时%.0fs",
                        bucket, idx, len(families), vendor_name, "·平铺" if flat else "",
                        len(batches), fresh, total_bytes / 1024 ** 3, total_count, time.time() - t0)
            if on_family:                           # 逐厂家落库（断点续传）
                on_family(vendor_name, batches)
        return entries
    finally:
        client.close()


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


# ── Schema ────────────────────────────────────────────────────────────────────

class TOSSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'dir_sizes'    - 统计 prefix 下各一级子目录大小（需 bucket，prefix 为父目录）\n"
            "  'list_objects' - 列出 prefix 下的对象（需 bucket）"
        )
    )
    bucket: str = Field(default="", description="TOS bucket 名称")
    prefix: str = Field(default="", description="对象前缀；dir_sizes 时为父目录")
    max_keys: int = Field(default=50, description="list_objects 返回上限，默认 50")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def manage_tos(action: str, bucket: str = "", prefix: str = "", max_keys: int = 50) -> str:
    action = action.strip().lower()

    # 不在此处 hard-import tos：凭证/SDK 缺失由 get_tos_client 统一返回 None，
    # 下方各分支会给出友好提示；TOS SDK 异常按类名识别，无需顶层导入。
    try:
        if action == "dir_sizes":
            if not bucket:
                return "❌ dir_sizes 需要提供 bucket。"
            rows = compute_dir_sizes(bucket, prefix)
            if rows is None:
                return "❌ TOS 凭证不可用：请配置 TOS_ACCESS_KEY / TOS_SECRET_KEY。"
            parent = (prefix or "").rstrip("/")
            if not rows:
                return f"TOS bucket `{bucket}` 的 `{parent or '/'}` 下没有子目录。"
            lines = [
                f"## TOS bucket `{bucket}` 目录大小（父目录 `{parent or '/'}`）",
                "| 子目录 | 大小 | 对象数 |",
                "|--------|------|--------|",
            ]
            grand_bytes = grand_objs = 0
            for name, size_bytes, count in rows:
                grand_bytes += size_bytes
                grand_objs += count
                lines.append(f"| `{name}` | {_fmt_size(size_bytes)} | {count} |")
            lines.append(f"| **合计** | **{_fmt_size(grand_bytes)}** | **{grand_objs}** |")
            return "\n".join(lines)

        if action == "list_objects":
            if not bucket:
                return "❌ list_objects 需要提供 bucket。"
            client = get_tos_client()
            if client is None:
                return "❌ TOS 凭证不可用：请配置 TOS_ACCESS_KEY / TOS_SECRET_KEY。"
            try:
                pfx = prefix if (not prefix or prefix.endswith("/")) else prefix
                r = client.list_objects_type2(bucket, prefix=pfx or "", max_keys=max_keys)
                objs = [o for o in r.contents if not (o.key.endswith("/") and o.size == 0)]
                if not objs:
                    return f"TOS bucket `{bucket}` 下无对象（prefix={prefix!r}）。"
                lines = [
                    f"## TOS bucket `{bucket}` 对象列表（前 {len(objs)} 条，prefix={prefix or '-'})",
                    "| 对象 Key | 大小 | 修改时间 |",
                    "|---------|------|---------|",
                ]
                for o in objs[:max_keys]:
                    lines.append(f"| `{o.key}` | {_fmt_size(o.size)} | {o.last_modified} |")
                return "\n".join(lines)
            finally:
                client.close()

        return f"❓ 未知 action：{action}，可选 dir_sizes / list_objects。"

    except Exception as e:
        # 按类名识别 TOS SDK 异常，避免顶层 import tos（未装时也能给友好提示）
        name = type(e).__name__
        if name == "TosServerError":
            logger.error("[TOS] 服务端错误 action=%s code=%s", action, getattr(e, "code", ""), exc_info=True)
            return (f"❌ TOS 服务端错误：code={getattr(e, 'code', '')} "
                    f"message={getattr(e, 'message', e)} request_id={getattr(e, 'request_id', '')}")
        if name == "TosClientError":
            logger.error("[TOS] 客户端错误 action=%s err=%s", action, e, exc_info=True)
            return f"❌ TOS 客户端错误：{getattr(e, 'message', e)}（cause: {getattr(e, 'cause', '')}）"
        logger.error("[TOS] 调用失败 action=%s err=%s", action, e, exc_info=True)
        return f"❌ TOS API 调用失败：{e}"


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

tos_tool = StructuredTool.from_function(
    func=manage_tos,
    name="manage_tos",
    description=(
        "查询火山引擎 TOS 对象存储（只读）。"
        "dir_sizes 统计某父目录下各一级子目录的大小（容量盘点）、"
        "list_objects 列出某 prefix 下的对象。"
    ),
    args_schema=TOSSchema,
)
