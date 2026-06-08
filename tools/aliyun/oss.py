"""
阿里云 OSS 管理工具。

凭证通过 STS AssumeRole 获取（utils.aliyun_client_factory），
按飞书 open_id 隔离权限。

写操作（create_bucket / put_object / delete_object / delete_bucket）
必须 confirm=true。
"""
import logging

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings
from utils.aliyun_client_factory import get_oss_service, get_oss_bucket

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class OSSSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'list_buckets'   - 列出所有 bucket\n"
            "  'list_objects'   - 列出 bucket 下的对象（需 bucket），可用 prefix 过滤\n"
            "  'bucket_size'    - 估算 bucket 已使用空间（需 bucket）\n"
            "  'dir_sizes'      - 统计 prefix 下各一级子目录的大小（需 bucket，prefix 为父目录）\n"
            "  'tree'           - 打印 bucket 目录结构（需 bucket，可用 max_depth 控制深度）\n"
            "  'create_bucket'  - 创建 bucket，需 bucket+region；confirm=true\n"
            "  'put_object'     - 上传文本对象，需 bucket+object_key+content；confirm=true\n"
            "  'delete_object'  - 删除对象，需 bucket+object_key；confirm=true\n"
            "  'delete_bucket'  - 删除空 bucket，需 bucket；confirm=true"
        )
    )
    bucket: str = Field(default="", description="bucket 名称")
    prefix: str = Field(default="", description="对象前缀过滤（如 logs/2026/）；dir_sizes 时为父目录")
    region: str = Field(default="", description="bucket 所在 region，留空则自动探测")
    max_keys: int = Field(default=50, description="list_objects 返回上限，默认 50")
    max_depth: int = Field(default=2, description="tree 最大下钻深度，默认 2")

    # 写操作参数
    object_key: str = Field(default="", description="对象 Key，put_object / delete_object 用")
    content: str = Field(default="", description="put_object 内容（文本），UTF-8 编码")
    storage_class: str = Field(
        default="Standard",
        description="创建 bucket 时的存储类型：Standard / IA / Archive / ColdArchive",
    )
    acl: str = Field(
        default="private",
        description="bucket ACL：private / public-read / public-read-write",
    )

    confirm: bool = Field(default=False, description="所有写操作必须显式 true")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def manage_oss(
    action: str,
    bucket: str = "",
    prefix: str = "",
    region: str = "",
    max_keys: int = 50,
    max_depth: int = 2,
    object_key: str = "",
    content: str = "",
    storage_class: str = "Standard",
    acl: str = "private",
    confirm: bool = False,
    open_id: str = "",
) -> str:
    action = action.strip().lower()

    try:
        import oss2
    except ImportError:
        return "❌ oss2 未安装，请执行：pip install oss2"

    try:
        # ── 只读：list_buckets ───────────────────────────────────────────────
        if action == "list_buckets":
            service = get_oss_service(open_id=open_id)
            if service is None:
                return "❌ OSS 凭证不可用：请确认 STS / RAM 映射已配置。"
            buckets = list(oss2.BucketIterator(service))
            if not buckets:
                return "当前账号下无 OSS bucket（或角色无 ListBuckets 权限）。"
            lines = [f"## OSS Bucket 列表（共 {len(buckets)} 个）"]
            lines.append("| 名称 | 区域 | 存储类型 | 创建时间 |")
            lines.append("|------|------|---------|---------|")
            for b in buckets:
                lines.append(
                    f"| `{b.name}` | {b.location} | {b.storage_class} | {b.creation_date} |"
                )
            return "\n".join(lines)

        # ── 只读：list_objects ───────────────────────────────────────────────
        if action == "list_objects":
            if not bucket:
                return "❌ list_objects 需要提供 bucket。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            objects = list(oss2.ObjectIterator(b, prefix=prefix or "", max_keys=max_keys))[:max_keys]
            if not objects:
                return f"bucket `{bucket}` 下无对象（prefix={prefix!r}）。"
            lines = [f"## bucket `{bucket}` 对象列表（前 {len(objects)} 条，prefix={prefix or '-'})"]
            lines.append("| 对象 Key | 大小 | 修改时间 |")
            lines.append("|---------|------|---------|")
            for obj in objects:
                size_str = _fmt_size(obj.size)
                lines.append(f"| `{obj.key}` | {size_str} | {obj.last_modified} |")
            return "\n".join(lines)

        # ── 只读：bucket_size ────────────────────────────────────────────────
        if action == "bucket_size":
            if not bucket:
                return "❌ bucket_size 需要提供 bucket。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            total_size = 0
            total_count = 0
            for obj in oss2.ObjectIterator(b, prefix=prefix or ""):
                total_size += obj.size
                total_count += 1
                if total_count >= 100000:
                    return (
                        f"⚠️ bucket `{bucket}` 对象数超过 10 万，未完整统计。\n"
                        f"已统计前 100000 个对象，累计 {_fmt_size(total_size)}。"
                    )
            return (
                f"## bucket `{bucket}` 用量\n"
                f"- 对象数：**{total_count}**\n"
                f"- 总大小：**{_fmt_size(total_size)}**"
                + (f"\n- 过滤前缀：`{prefix}`" if prefix else "")
            )

        # ── 只读：dir_sizes（各一级子目录大小）──────────────────────────────
        if action == "dir_sizes":
            if not bucket:
                return "❌ dir_sizes 需要提供 bucket。"
            rows, endpoint = compute_dir_sizes(open_id, bucket, prefix, region=region)
            if rows is None:
                return "❌ OSS 凭证不可用：请确认 STS / RAM 映射已配置。"
            parent = (prefix or "").rstrip("/")
            if not rows:
                return f"bucket `{bucket}` 的 `{parent or '/'}` 下没有子目录。"
            lines = [
                f"## bucket `{bucket}` 目录大小（父目录 `{parent or '/'}`，{endpoint}）",
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

        # ── 只读：tree（目录结构）───────────────────────────────────────────
        if action == "tree":
            if not bucket:
                return "❌ tree 需要提供 bucket。"
            return build_tree(open_id, bucket, prefix, max_depth, region=region)

        # ── 写操作：统一拦截 confirm ────────────────────────────────────────
        if action in ("create_bucket", "put_object", "delete_object", "delete_bucket") and not confirm:
            return (
                f"⚠️ OSS `{action}` 是写操作，"
                "请明确传入 `confirm=true` 后再调用。"
            )

        if action == "create_bucket":
            if not bucket:
                return "❌ create_bucket 需要 bucket 名称。"
            from utils.aliyun_client_factory import get_oss_auth
            auth, _ = get_oss_auth(open_id)
            if auth is None:
                return "❌ OSS 凭证不可用。"
            region = region or settings.PAI_DSW_REGION_ID or "cn-hangzhou"
            endpoint = f"https://oss-{region}.aliyuncs.com"
            b = oss2.Bucket(auth, endpoint, bucket)
            # ACL 映射
            acl_map = {
                "private":          oss2.BUCKET_ACL_PRIVATE,
                "public-read":      oss2.BUCKET_ACL_PUBLIC_READ,
                "public-read-write": oss2.BUCKET_ACL_PUBLIC_READ_WRITE,
            }
            b.create_bucket(
                acl_map.get(acl, oss2.BUCKET_ACL_PRIVATE),
                oss2.models.BucketCreateConfig(storage_class=storage_class),
            )
            logger.info("[OSS] 创建 bucket %s region=%s 操作者=%s", bucket, region, open_id)
            return f"✅ bucket `{bucket}` 已创建（region={region}, storage={storage_class}, acl={acl}）。"

        if action == "put_object":
            if not bucket or not object_key:
                return "❌ put_object 需要 bucket 和 object_key。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            b.put_object(object_key, content)
            logger.info("[OSS] 上传对象 %s/%s 大小=%d 操作者=%s",
                        bucket, object_key, len(content), open_id)
            return (
                f"✅ 对象 `{object_key}` 已上传到 bucket `{bucket}`，"
                f"大小 {_fmt_size(len(content.encode('utf-8')))}。"
            )

        if action == "delete_object":
            if not bucket or not object_key:
                return "❌ delete_object 需要 bucket 和 object_key。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            b.delete_object(object_key)
            logger.warning("[OSS] 删除对象 %s/%s 操作者=%s", bucket, object_key, open_id)
            return f"⚠️ 对象 `{object_key}` 已从 bucket `{bucket}` 删除（不可恢复）。"

        if action == "delete_bucket":
            if not bucket:
                return "❌ delete_bucket 需要 bucket。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            b.delete_bucket()
            logger.warning("[OSS] 删除 bucket %s 操作者=%s", bucket, open_id)
            return f"⚠️ bucket `{bucket}` 已删除（必须为空才能删除）。"

        return (
            f"❓ 未知 action：{action}，可选 list_buckets / list_objects / bucket_size / "
            "create_bucket / put_object / delete_object / delete_bucket。"
        )

    except Exception as e:
        logger.error("[OSS] 调用失败 action=%s err=%s", action, e, exc_info=True)
        return f"❌ OSS API 调用失败：{e}"


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


# ── 目录大小 / 目录树（供工具 + 容量监控复用）─────────────────────────────────

# 不下钻的噪音目录（内部分片极多，对了解结构无意义）
_TREE_SKIP_DIRS = {".git", "node_modules", ".svn", ".hg", ".idea", ".vscode",
                   "__pycache__", ".cache"}
# 同级目录超过该数量则降级：只列前 N 个、不再下钻
_TREE_DEGRADE_THRESHOLD = 50


def _endpoint_from_region(region: str) -> str:
    region = region.strip()
    if not region.startswith("oss-"):
        region = "oss-" + region
    return f"https://{region}.aliyuncs.com"


def _detect_region_endpoint(auth, bucket_name: str):
    """自动探测桶所在地域，返回 Endpoint；探测失败回退默认 region。"""
    import oss2
    default = _endpoint_from_region(settings.PAI_DSW_REGION_ID or "cn-hangzhou")
    probe = oss2.Bucket(auth, default, bucket_name)
    try:
        return _endpoint_from_region(probe.get_bucket_location().location)
    except oss2.exceptions.OssError as e:
        headers = getattr(e, "headers", None) or {}
        for key in ("x-oss-region", "X-Oss-Region"):
            if headers.get(key):
                return _endpoint_from_region(headers[key])
        import re
        body = getattr(e, "body", "") or ""
        m = re.search(r"<Endpoint>\s*([^<]+?)\s*</Endpoint>", body)
        if m:
            ep = m.group(1).strip()
            return ep if ep.startswith("http") else "https://" + ep
        return default


def _resolve_bucket(open_id: str, bucket_name: str, region: str = ""):
    """构造 oss2.Bucket：region 留空则自动探测地域（跨地域桶可用）。"""
    from utils.aliyun_client_factory import get_oss_auth
    import oss2
    auth, _ = get_oss_auth(open_id)
    if auth is None:
        return None
    endpoint = (_endpoint_from_region(region) if region
                else _detect_region_endpoint(auth, bucket_name))
    return oss2.Bucket(auth, endpoint, bucket_name)


def _list_subdirs(bucket, prefix: str) -> list:
    """返回 prefix 下的一级子目录（每个以 '/' 结尾）。"""
    subdirs, token = [], ""
    while True:
        r = bucket.list_objects_v2(prefix=prefix, delimiter="/",
                                   continuation_token=token, max_keys=1000)
        subdirs.extend(r.prefix_list)
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return subdirs


def _prefix_size(bucket, prefix: str):
    """累加 prefix 下所有对象大小，返回 (总字节数, 对象数)。跳过 0 字节目录占位对象。"""
    total_bytes = count = 0
    token = ""
    while True:
        r = bucket.list_objects_v2(prefix=prefix, continuation_token=token, max_keys=1000)
        for obj in r.object_list:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            total_bytes += obj.size
            count += 1
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return total_bytes, count


def compute_dir_sizes(open_id: str, bucket: str, prefix: str = "", region: str = ""):
    """统计 prefix 下各一级子目录的大小。

    返回 (rows, endpoint)，rows = [(name, size_bytes, count), ...]。
    凭证不可用时返回 (None, None)。供 manage_oss 与 capacity_monitor 共用。
    """
    b = _resolve_bucket(open_id, bucket, region=region)
    if b is None:
        return None, None
    prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
    rows = []
    for sub in _list_subdirs(b, prefix):
        name = sub[len(prefix):].rstrip("/")
        size_bytes, count = _prefix_size(b, sub)
        rows.append((name, size_bytes, count))
    return rows, b.endpoint


# ── 数据结构判定 + 点目录过滤（供两级遍历共用）──────────────────────────────

def _struct_of_key(key: str) -> str:
    """按文件扩展名判定数据结构：.hdf5→ego；.parquet/.mcap→itw；其余→空。"""
    k = key.lower()
    if k.endswith(".hdf5"):
        return "ego"
    if k.endswith(".parquet") or k.endswith(".mcap"):
        return "itw"
    return ""


_IGNORE_DIRNAMES = {"test"}   # 非交付数据目录：测试目录


def _is_ignored_dirname(name: str) -> bool:
    """该目录名是否忽略：. 开头（.cache/.deliver_server_preflight 等元数据）或正好叫 test。"""
    return name.startswith(".") or name.lower() in _IGNORE_DIRNAMES


def _has_ignored_dir(rel: str) -> bool:
    """rel 为相对路径；任一“目录段”被忽略（. 开头 / test）→ True。"""
    return any(_is_ignored_dirname(seg) for seg in rel.split("/")[:-1])


def agg_struct(structs) -> str:
    """聚合一组结构：单一→该值；混合→'ego/itw'；全空→空。"""
    s = sorted({x for x in structs if x})
    return "/".join(s)


def _direct_files_size(bucket, prefix: str):
    """只统计 prefix 直属文件（不含子目录里的，跳过点文件），返回 (字节数, 对象数, struct)。"""
    total_bytes = count = 0
    struct = ""
    token = ""
    while True:
        r = bucket.list_objects_v2(prefix=prefix, delimiter="/",
                                   continuation_token=token, max_keys=1000)
        for obj in r.object_list:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            name = obj.key[len(prefix):]
            if name.startswith("."):               # 点文件忽略
                continue
            total_bytes += obj.size
            count += 1
            if not struct:
                st = _struct_of_key(obj.key)
                if st:
                    struct = st
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return total_bytes, count, struct


def _grouped_sizes(bucket, family_prefix: str) -> dict:
    """一趟连续扫描 family_prefix 下所有对象，按第二层目录(批次)归并求和。

    返回 {批次名: (bytes, count, struct)}；厂家直属文件归入 '/'。
    跳过任意层级的点目录（.cache/.deliver_server_preflight 等元数据噪音）。
    调用次数只与对象数有关、与批次数无关——批次极多时远快于逐批次扫描。
    """
    sizes: dict = {}
    token = ""
    while True:
        r = bucket.list_objects_v2(prefix=family_prefix, continuation_token=token, max_keys=1000)
        for obj in r.object_list:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            rel = obj.key[len(family_prefix):]
            if _has_ignored_dir(rel):                  # 忽略点目录
                continue
            batch = rel.split("/", 1)[0] if "/" in rel else "/"
            slot = sizes.setdefault(batch, [0, 0, ""])
            slot[0] += obj.size
            slot[1] += 1
            if not slot[2]:
                st = _struct_of_key(obj.key)
                if st:
                    slot[2] = st
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return {k: (v[0], v[1], v[2]) for k, v in sizes.items()}


def _batch_sum(bucket, batch_prefix: str):
    """逐批次扫：累加 batch_prefix 下对象（跳过点目录），返回 (bytes, count, struct)。"""
    total = count = 0
    struct = ""
    token = ""
    while True:
        r = bucket.list_objects_v2(prefix=batch_prefix, continuation_token=token, max_keys=1000)
        for obj in r.object_list:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            if _has_ignored_dir(obj.key[len(batch_prefix):]):
                continue
            total += obj.size
            count += 1
            if not struct:
                st = _struct_of_key(obj.key)
                if st:
                    struct = st
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    return total, count, struct


# 一个厂家下「新批次」数超过此值就改用一趟分组扫描（避免逐批次列举爆炸）
_GROUPED_THRESHOLD = 30
# 批次数超过此值判定为「平铺」：不再细分批次，只记厂家整体大小（一行 'ALL'）
_MAX_BATCHES = 200


def compute_nested_sizes(open_id: str, bucket: str, prefix: str = "", region: str = "",
                         cached: dict = None, on_family=None):
    """两级遍历：prefix 下每个「厂家」(一级子目录) 及其下「批次」(二级子目录) 的大小。

    返回 (entries, endpoint)，entries = [
        {"厂家": name, "total_bytes": int, "total_count": int,
         "batches": [(批次名, bytes, count), ...]},
        ...
    ]。厂家直属文件归入名为 '/' 的批次。凭证不可用返回 (None, None)。

    cached: {"厂家/批次": (bytes, count)} 已扫过的不可变批次；命中则跳过逐对象求和。
    """
    b = _resolve_bucket(open_id, bucket, region=region)
    if b is None:
        return None, None
    prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
    cached = cached or {}

    import time
    families = _list_subdirs(b, prefix)
    entries = []
    for idx, sub in enumerate(families, 1):         # 厂家
        vendor_name = sub[len(prefix):].rstrip("/")
        t0 = time.time()
        batches, fresh, flat = _scan_family_oss(b, sub, vendor_name, cached)
        total_bytes = sum(x[1] for x in batches)
        total_count = sum(x[2] for x in batches)
        entries.append({
            "厂家": vendor_name,
            "total_bytes": total_bytes,
            "total_count": total_count,
            "struct": agg_struct(x[3] for x in batches),
            "batches": batches,
        })
        logger.info("[进度] OSS %s [%d/%d] %s%s：%d 批次(%d 新扫) %.2fGB %d 对象 用时%.0fs",
                    bucket, idx, len(families), vendor_name, "·平铺" if flat else "",
                    len(batches), fresh, total_bytes / 1024 ** 3, total_count, time.time() - t0)
        if on_family:                               # 逐厂家落库（断点续传）
            on_family(vendor_name, batches)
    return entries, b.endpoint


def _scan_family_oss(b, sub: str, vendor_name: str, cached: dict):
    """扫一个厂家，返回 (batches[(批次,bytes,count,struct)], 新扫批次数, 是否平铺)。

    决策：上次判平铺 → 直接整体扫；有缓存 → 只实扫新批次；首扫 → 整体扫一趟。
    整体扫后若批次数超 _MAX_BATCHES 判为平铺，折叠成单行 'ALL'。
    """
    if f"{vendor_name}/ALL" in cached:           # 上次已判平铺，直接整体扫
        use_grouped, batch_dirs = True, None
    else:
        fam_has_cache = any(k.startswith(vendor_name + "/") for k in cached)
        if fam_has_cache:
            batch_dirs = [bd for bd in _list_subdirs(b, sub)
                          if not _is_ignored_dirname(bd[len(sub):].rstrip("/"))]   # 跳过点目录
            new_dirs = [bd for bd in batch_dirs
                        if f"{vendor_name}/{bd[len(sub):].rstrip('/')}" not in cached]
            use_grouped = len(new_dirs) > _GROUPED_THRESHOLD
        else:
            use_grouped, batch_dirs = True, None

    if use_grouped:
        grouped = _grouped_sizes(b, sub)
        real = [n for n in grouped if n != "/"]
        if len(real) > _MAX_BATCHES:                # 平铺：不细分，只记整体
            tb = sum(v[0] for v in grouped.values())
            tc = sum(v[1] for v in grouped.values())
            st = agg_struct(v[2] for v in grouped.values())
            return [("ALL", tb, tc, st)], len(real), True
        return [(n, s, c, st) for n, (s, c, st) in grouped.items()], len(real), False

    batches, fresh = [], 0
    for bd in batch_dirs:                           # 增量：逐新批次扫，老批次用缓存
        batch_name = bd[len(sub):].rstrip("/")
        ck = f"{vendor_name}/{batch_name}"
        if ck in cached:
            size_bytes, count, struct = cached[ck]
        else:
            size_bytes, count, struct = _batch_sum(b, bd)
            fresh += 1
        batches.append((batch_name, size_bytes, count, struct))
    root_bytes, root_count, root_struct = _direct_files_size(b, sub)
    if root_count:
        batches.append(("/", root_bytes, root_count, root_struct))
    return batches, fresh, False


def build_tree(open_id: str, bucket: str, prefix: str = "",
               max_depth: int = 2, region: str = "") -> str:
    """打印 bucket 目录结构（自动探测地域、噪音目录折叠、同级过多降级）。"""
    b = _resolve_bucket(open_id, bucket, region=region)
    if b is None:
        return "❌ OSS 凭证不可用：请确认 STS / RAM 映射已配置。"

    lines = [f"## bucket `{bucket}` 目录结构（{b.endpoint}，max_depth={max_depth}）"]
    stats = {"dirs": 0, "skipped": 0, "omitted": 0}

    def _walk(pfx: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        subs = _list_subdirs(b, pfx)
        indent = "    " * depth
        total = len(subs)
        degraded = bool(_TREE_DEGRADE_THRESHOLD) and total > _TREE_DEGRADE_THRESHOLD
        sample = subs[:_TREE_DEGRADE_THRESHOLD] if degraded else subs
        for sub in sample:
            name = sub[len(pfx):].rstrip("/")
            stats["dirs"] += 1
            if name in _TREE_SKIP_DIRS:
                lines.append(f"{indent}|-- {name}/  (已折叠，未展开)")
                stats["skipped"] += 1
                continue
            lines.append(f"{indent}|-- {name}/")
            if not degraded:
                _walk(sub, depth + 1)
        if degraded:
            omitted = total - len(sample)
            stats["dirs"] += omitted
            stats["omitted"] += omitted
            lines.append(f"{indent}|-- ... （此层共 {total} 个同级目录，超过 "
                         f"{_TREE_DEGRADE_THRESHOLD} 已降级：省略其余 {omitted} 个且不再展开）")

    try:
        import oss2
        _walk(prefix if (not prefix or prefix.endswith("/")) else prefix + "/", 0)
    except oss2.exceptions.OssError as e:
        return f"❌ 访问 bucket `{bucket}` 失败：{e.code} - {e.message}"

    if stats["dirs"] == 0:
        lines.append("(该前缀下没有子目录)")
    else:
        extra = []
        if stats["skipped"]:
            extra.append(f"{stats['skipped']} 个噪音目录已折叠")
        if stats["omitted"]:
            extra.append(f"{stats['omitted']} 个同级目录因过多被省略")
        tail = f"\n共 {stats['dirs']} 个目录"
        if extra:
            tail += "（其中 " + "，".join(extra) + "）"
        lines.append(tail)
    return "\n".join(lines)


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

oss_tool = StructuredTool.from_function(
    func=manage_oss,
    name="manage_oss",
    description=(
        "管理阿里云 OSS。"
        "只读：list_buckets 列 bucket、list_objects 列对象、bucket_size 估算用量、"
        "dir_sizes 统计某父目录下各一级子目录大小、tree 打印目录结构。"
        "写操作（create_bucket / put_object / delete_object / delete_bucket）"
        "必须 confirm=true。"
    ),
    args_schema=OSSSchema,
)
