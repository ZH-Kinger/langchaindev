"""
阿里云 OSS 管理工具。

凭证通过 STS AssumeRole 获取（utils.aliyun_client_factory），
按飞书 open_id 隔离权限。

写操作（create_bucket / put_object / delete_object / delete_bucket）
必须 confirm=true。
"""
import itertools
import logging
import re

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
            # 用 islice 在第一页就截断：ObjectIterator 的 max_keys 只是每页大小，直接
            # list(...) 会翻页枚举整个 prefix 下全部对象（百万级桶→挂线程/OOM）后才切片。
            objects = list(itertools.islice(
                oss2.ObjectIterator(b, prefix=prefix or "", max_keys=max_keys), max_keys))
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


def region_from_endpoint(endpoint: str) -> str:
    """从 endpoint 提取裸 region：https://oss-cn-beijing[-internal].aliyuncs.com → cn-beijing。"""
    host = (endpoint or "").split("://", 1)[-1].split(".", 1)[0]   # oss-cn-beijing[-internal]
    region = host[4:] if host.startswith("oss-") else host
    return region.replace("-internal", "")


def detect_bucket_region(open_id: str, bucket_name: str) -> str:
    """探测 OSS 桶所在裸 region（如 cn-beijing），供桶间迁移拼 region_id/domain。探测失败回退默认。"""
    from utils.aliyun_client_factory import get_oss_auth
    auth, _ = get_oss_auth(open_id)
    if auth is None:
        return settings.PAI_DSW_REGION_ID or "cn-hangzhou"
    return region_from_endpoint(_detect_region_endpoint(auth, bucket_name)) or \
        (settings.PAI_DSW_REGION_ID or "cn-hangzhou")


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


# ── 数据模态判定（读对象路径/文件名 + LeRobot info.json features）+ 点目录过滤 ──
# 「数据结构」列改为如实列出数据集含哪些模态（取代过去靠扩展名猜 itw/ego 的做法）。
# 标签按此固定顺序输出，便于稳定展示。
_MODALITY_ORDER = [
    "rgb", "depth", "imu", "audio", "gaze", "head_pose", "body_pose",
    "ee_pose", "wrist", "hand_keypoints", "mano", "action", "state", "joint", "gripper",
]
# 标签 → 命中关键词（子串，匹配时对象已转小写）
_MODALITY_KW = {
    "rgb":            ("front_camera", "camera", "images", "image", "rgb", ".mp4", ".mkv", "video"),
    "depth":          ("depth",),
    "imu":            ("imu",),
    "audio":          ("mic", ".wav", "audio"),
    "gaze":           ("gaze",),
    "head_pose":      ("head_pose", "obs_head", "head_hands"),
    "body_pose":      ("arm", "shoulder", "hip", "neck", "spine", "torso", "elbow"),  # 全身骨骼(EgoDex)
    "ee_pose":        ("ee_pose", "eef", "end_effector"),
    "wrist":          ("wrist",),
    "hand_keypoints": ("keypoint", "joints2d", "finger", "thumb", "knuckle", "metacarpal", "hand"),
    "mano":           ("mano",),
    "action":         ("action",),
    "state":          (".state", "/state", "observation.state", "qpos"),
    "joint":          ("joint_pos", "joint_state", "joint_angle"),
    "gripper":        ("gripper",),
}
_MODALITY_SAMPLE_CAP = 800   # 每批次只对前 N 个对象采模态（模态在数据集内同质，足够稳定）
_HDF5_MAX_READ = 25 * 1024 ** 2   # 只读小于此大小的 .hdf5（避免下载几十 MB 进内存 OOM/变慢）
_INFO_KEYS_CAP = 3000   # 单批次最多收集 N 个 meta/info.json（一个交付批次可含数十~上百个数据集）


def _modality_bits(s: str) -> int:
    """从一个路径段/文件名/字段名（小写）命中的模态 → bit 集合。"""
    bits = 0
    for i, label in enumerate(_MODALITY_ORDER):
        for kw in _MODALITY_KW[label]:
            if kw in s:
                bits |= (1 << i)
                break
    return bits


def _resolve_modalities(bits: int) -> str:
    """bit 集合 → 固定顺序的模态串，如 'RGB/手部MANO/动作/状态'。"""
    return "/".join(_MODALITY_ORDER[i] for i in range(len(_MODALITY_ORDER)) if bits & (1 << i))


def _features_modality_bits(feature_keys) -> int:
    """LeRobot info.json 的 features 键集合 → 模态 bit。"""
    bits = 0
    for k in feature_keys:
        bits |= _modality_bits(str(k).lower())
    return bits


def agg_modalities(mods) -> str:
    """聚合一组模态串（各为 '/'-join）→ 并集，按固定顺序。"""
    labels = set()
    for m in mods:
        if m:
            labels.update(m.split("/"))
    return "/".join(l for l in _MODALITY_ORDER if l in labels)


_IGNORE_DIRNAMES = {"test"}   # 非交付数据目录：测试目录


def _is_ignored_dirname(name: str) -> bool:
    """该目录名是否忽略：. 开头（.cache/.deliver_server_preflight 等元数据）或正好叫 test。"""
    return name.startswith(".") or name.lower() in _IGNORE_DIRNAMES


def _has_ignored_dir(rel: str) -> bool:
    """rel 为相对路径；任一“目录段”被忽略（. 开头 / test）→ True。"""
    return any(_is_ignored_dirname(seg) for seg in rel.split("/")[:-1])


# 目录名里的时长 token：504h / 100h / 19.75h / 85.11h …
_DUR_RE = re.compile(r"\d+(?:\.\d+)?\s*h(?:\b|_|$)", re.IGNORECASE)


def _has_duration(name: str) -> bool:
    return bool(_DUR_RE.search(name or ""))


def _batch_key(rel: str) -> str:
    """对象相对「厂家前缀」的路径 → 批次键（= 真实数据集根）。

    取到「首个名字含时长 token（如 19.75h）的目录段」为止作为数据集根；
    没有时长目录则退回第一段（与旧的「批次=第二层」一致）；厂家直属文件 �� '/'。
    这样深层嵌套的数据集（如  shutu/wuji/26_0430/wuji_home_scene_19.75h/…）
    会被识别成 `wuji/26_0430/wuji_home_scene_19.75h` 这一真实批次，而非容器 `wuji`。
    """
    parts = rel.split("/")
    if len(parts) == 1:                       # 厂家直属文件
        return "/"
    for i in range(len(parts) - 1):           # 只看目录段（不含末尾文件名）
        if _has_duration(parts[i]):
            return "/".join(parts[:i + 1])
    return parts[0]


# ── 数据集类型识别（按目录结构，list-only，不读文件内容）────────────────────────
# LeRobot 结构标记（看 meta/ 关键文件）+ 其余常见格式（按扩展名）。
_DT_LEROBOT, _DT_V3, _DT_V2, _DT_V21 = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_DT_RLDS, _DT_ROSBAG, _DT_MCAP, _DT_HDF5 = 1 << 4, 1 << 5, 1 << 6, 1 << 7
_DT_ZARR, _DT_RAWCAP = 1 << 8, 1 << 9
_DT_PARQUET, _DT_NUMPY, _DT_PCD = 1 << 10, 1 << 11, 1 << 12


def _dataset_type_bits(within: str) -> int:
    """从「相对批次根的对象路径」识别数据集结构标记位。

    LeRobot 依据官方目录结构：v3.0 有 meta/tasks.parquet + meta/episodes/*.parquet；
    v2.x 有 meta/episodes.jsonl / meta/tasks.jsonl；v2.1 多 meta/episodes_stats.jsonl；
    都含 meta/info.json。其余格式按扩展名判。
    """
    r = within.lower()
    bits = 0
    if "meta/" in r:                                  # LeRobot：只看 meta/ 关键文件
        if r.endswith("meta/info.json"):
            bits |= _DT_LEROBOT
        if r.endswith("meta/tasks.parquet") or "meta/episodes/" in r:
            bits |= _DT_V3
        if r.endswith("meta/episodes.jsonl") or r.endswith("meta/tasks.jsonl"):
            bits |= _DT_V2
        if r.endswith("meta/episodes_stats.jsonl"):
            bits |= _DT_V21
    # Zarr：v3 用 zarr.json；v2 用 .zarray/.zgroup/.zattrs；也可能整体打成 .zarr.zip 或 xxx.zarr/ 目录
    if (r.endswith("zarr.json") or r.endswith(".zarray") or r.endswith(".zgroup")
            or r.endswith(".zattrs") or r.endswith(".zarr.zip") or ".zarr/" in r):
        bits |= _DT_ZARR
    # 原始多传感采集（camera_params/ 标定目录 或 kalibr 标定文件）
    if "camera_params/" in r or r.endswith("kalibr_parameters.yaml"):
        bits |= _DT_RAWCAP
    # RLDS/TFDS：分片 *.tfrecord-00000-of-00010 / dataset_info.json / features.json
    if ".tfrecord" in r or r.endswith("dataset_info.json") or r.endswith("features.json"):
        bits |= _DT_RLDS
    elif r.endswith(".bag") or r.endswith(".db3"):
        bits |= _DT_ROSBAG
    elif r.endswith(".mcap"):
        bits |= _DT_MCAP
    elif r.endswith(".hdf5") or r.endswith(".h5"):
        bits |= _DT_HDF5
    elif r.endswith(".parquet"):                              # 裸 parquet（lerobot 的已被 meta/ 命中）
        bits |= _DT_PARQUET
    elif r.endswith(".npy") or r.endswith(".npz"):
        bits |= _DT_NUMPY
    elif r.endswith(".pcd") or r.endswith(".ply") or r.endswith(".las"):
        bits |= _DT_PCD
    return bits


def _resolve_dataset_type(bits: int) -> str:
    """标记位 → 数据集类型串；无可识别返回空。LeRobot 优先（其数据即 parquet）。"""
    if bits & (_DT_LEROBOT | _DT_V3 | _DT_V2 | _DT_V21):
        if bits & _DT_V3:
            return "lerobot v3.0"
        if bits & _DT_V21:
            return "lerobot v2.1"
        if bits & _DT_V2:
            return "lerobot v2.0"
        return "lerobot"                              # 仅见 info.json，版本未知
    if bits & _DT_ZARR:
        return "zarr"
    if bits & _DT_RLDS:
        return "rlds"
    if bits & _DT_ROSBAG:
        return "rosbag"
    if bits & _DT_RAWCAP:                              # 自定义原始采集（结构特征，优先于裸扩展名）
        return "raw-capture"
    if bits & _DT_MCAP:
        return "mcap"
    if bits & _DT_HDF5:
        return "hdf5"
    if bits & _DT_PARQUET:
        return "parquet"
    if bits & _DT_NUMPY:
        return "numpy"
    if bits & _DT_PCD:
        return "pointcloud"
    return ""                                          # 无识别：批次级由 resolve_dtype 兜底成 other


def resolve_dtype(bits: int) -> str:
    """批次级解析：识别到→具体类型；有对象但啥都没识别到→'other'（散数据）。"""
    return _resolve_dataset_type(bits) or "other"


def agg_dtype(types) -> str:
    """聚合一组数据集类型：有具体类型→去重 join（丢掉 other）；全是 other/空→other 或空。"""
    s = {x for x in types if x}
    real = sorted(x for x in s if x != "other")
    if real:
        return "/".join(real)
    return "other" if "other" in s else ""


def _grouped_sizes(bucket, family_prefix: str) -> dict:
    """一趟连续扫描 family_prefix 下所有对象，按第二层目录(批次)归并求和。

    返回 {批次名: (bytes, count, struct, dtype, info_keys)}；厂家直属文件归入 '/'。
    info_keys = 该批次下「所有」数据集的 meta/info.json key 列表（一个交付批次常含多个
    LeRobot 数据集，须全部读出时长再求和；只读首个会严重少算），无则空列表。
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
            batch = _batch_key(rel)                    # 自适应：下钻到真实数据集根
            within = rel[len(batch) + 1:] if batch != "/" else rel
            slot = sizes.setdefault(batch, [0, 0, 0, 0, [], ""])   # [bytes,count,modbits,dtbits,info_keys,hdf5_key]
            slot[0] += obj.size
            slot[1] += 1
            if slot[1] <= _MODALITY_SAMPLE_CAP:        # 模态：只采前 N 个对象（同质，够稳）
                slot[2] |= _modality_bits(within.lower())
            slot[3] |= _dataset_type_bits(within)
            if within.endswith("meta/info.json") and len(slot[4]) < _INFO_KEYS_CAP:
                slot[4].append(obj.key)                # 收集本批次下「所有」数据集的 info.json（一批次可含多个 LeRobot 数据集）
            if not slot[5] and within.lower().endswith(".hdf5") and obj.size < _HDF5_MAX_READ:
                slot[5] = obj.key
        if r.is_truncated:
            token = r.next_continuation_token
        else:
            break
    # {批次: (bytes,count,modbits,dtype串,info_keys[列表],hdf5_key)}；modbits 留待 _scan_family 合并 features/hdf5 后解析
    return {k: (v[0], v[1], v[2], resolve_dtype(v[3]), v[4], v[5]) for k, v in sizes.items()}


import json as _json


def parse_lerobot_info(raw):
    """LeRobot `meta/info.json` 原始字节/字符串 → (hours, version, modality_bits)。

    时长 = total_frames / fps / 3600；版本来自 codebase_version；模态来自 features 键。
    解析失败 → (None, None, 0)。供 OSS / TOS 各自取到字节后共用。
    """
    try:
        info = _json.loads(raw)
        fps, frames = info.get("fps"), info.get("total_frames")
        hours = round(frames / float(fps) / 3600, 2) if (fps and frames) else None
        ver = info.get("codebase_version")
        feats = info.get("features") or {}
        modbits = _features_modality_bits(feats.keys() if isinstance(feats, dict) else feats)
        return hours, (f"lerobot {ver}" if ver else None), modbits
    except Exception:
        return None, None, 0


def _read_lerobot_info(bucket, info_key: str):
    """OSS：GetObject `meta/info.json` 并解析 → (hours, version, modbits)。失败 (None, None, 0)。

    仅在已确认是 LeRobot 数据集（扫到 meta/info.json）时调用；结果随批次缓存，不每次重读。
    """
    try:
        return parse_lerobot_info(bucket.get_object(info_key).read())
    except Exception:
        return None, None, 0


def hdf5_modality_bits(raw: bytes) -> int:
    """读 HDF5 字节内的所有 group/dataset 路径名 → 模态 bit。需 h5py；缺失或解析错→0。

    供 OSS/TOS 各自取到一个小 .hdf5 文件字节后共用（如 EgoDex 的 camera/transforms/手指关键点）。
    """
    try:
        import io
        import h5py
        bits = 0
        with h5py.File(io.BytesIO(raw), "r") as f:
            names = []
            f.visit(names.append)
            for n in names:
                bits |= _modality_bits(n.lower())
        return bits
    except Exception:
        return 0


def _read_hdf5_modalities(bucket, key: str) -> int:
    """OSS：GetObject 一个小 .hdf5 → 内部字段模态 bit。失败 0。"""
    try:
        return hdf5_modality_bits(bucket.get_object(key).read())
    except Exception:
        return 0


# 批次数超过此值判定为「平铺」：不再细分批次，只记厂家整体大小（一行 'ALL'）
_MAX_BATCHES = 200


def compute_nested_sizes(open_id: str, bucket: str, prefix: str = "", region: str = "",
                         cached: dict = None, on_family=None, skip_flat=None):
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
        batches, fresh, flat = _scan_family_oss(b, sub, vendor_name, cached, skip_flat)
        total_bytes = sum(x[1] for x in batches)
        total_count = sum(x[2] for x in batches)
        entries.append({
            "厂家": vendor_name,
            "total_bytes": total_bytes,
            "total_count": total_count,
            "struct": agg_modalities(x[3] for x in batches),   # 数据结构列=数据模态
            "dtype": agg_dtype(x[4] for x in batches),
            "flat": flat,
            "batches": batches,
        })
        logger.info("[进度] OSS %s [%d/%d] %s%s：%d 批次(%d 新扫) %.2fGB %d 对象 用时%.0fs",
                    bucket, idx, len(families), vendor_name, "·平铺" if flat else "",
                    len(batches), fresh, total_bytes / 1024 ** 3, total_count, time.time() - t0)
        if on_family:                               # 逐厂家落库（断点续传）
            on_family(vendor_name, batches)
    return entries, b.endpoint


def _scan_family_oss(b, sub: str, vendor_name: str, cached: dict = None, skip_flat=None):
    """扫一个厂家，返回 (batches[(批次,bytes,count,struct,dtype,hours)], 新扫批次数, 是否平铺)。

    一趟分组扫描：`_grouped_sizes` 用 `_batch_key` 把对象归并到「真实数据集根」
    （时长目录或第二层），任意深度都对齐到真实批次（如 shutu 的 wuji/26_0430/…_19.75h）。
    批次数超 _MAX_BATCHES 判为平铺，折叠成单行 'ALL'。
    `skip_flat`：本次可跳过重扫的平铺家集合（近期已全扫过）→ 直接复用缓存的 ALL 总量。
    LeRobot 批次额外读 meta/info.json 拿精确时长 + 版本。
    """
    cached = cached or {}
    # 近期扫过的平铺家：复用缓存 ALL，跳过 1.6M 对象全扫
    if skip_flat and vendor_name in skip_flat:
        allk = f"{vendor_name}/ALL"
        if allk in cached:
            s, c, st, dt, hrs = cached[allk]
            return [("ALL", s, c, st, dt, hrs)], 0, True

    grouped = _grouped_sizes(b, sub)                # {批次: (bytes,count,modbits,dtype,info_key,hdf5_key)}
    real = [n for n in grouped if n != "/"]
    if len(real) > _MAX_BATCHES:                    # 平铺：不细分，只记整体
        tb = sum(v[0] for v in grouped.values())
        tc = sum(v[1] for v in grouped.values())
        mbits = 0
        for v in grouped.values():
            mbits |= v[2]
        dt = agg_dtype(v[3] for v in grouped.values())
        return [("ALL", tb, tc, _resolve_modalities(mbits), dt, None)], len(real), True

    batches = []
    for name, (s, c, mbits, dt, info_keys, hdf5_key) in grouped.items():
        hrs = None
        if info_keys:                               # LeRobot：累加本批次下「所有」数据集的 info.json 时长
            total_h, ver = 0.0, None
            for ik in info_keys:
                h, v, feat_mbits = _read_lerobot_info(b, ik)
                if h is not None:
                    total_h += h
                if v and not ver:                   # 同批次版本一致，取首个非空
                    ver = v
                mbits |= feat_mbits
            if ver:
                dt = ver
            hrs = round(total_h, 2) if total_h > 0 else None
        elif hdf5_key:                              # HDF5：读单个小文件内部字段补模态
            mbits |= _read_hdf5_modalities(b, hdf5_key)
        batches.append((name, s, c, _resolve_modalities(mbits), dt, hrs))
    return batches, len(real), False


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
