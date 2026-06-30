"""
跨云迁移路径解析与方向判断（纯逻辑，无外部依赖，便于单测）。

统一路径语法（源必填、目的可选）：
    cpfs://<fs-id>/<dir>/      阿里全闪
    vepfs://<fs-id>/<dir>/     火山全闪
    oss://<bucket>/<prefix>/   阿里对象存储
    tos://<bucket>/<prefix>/   火山对象存储

约定：
  - 只接受目录（以 '/' 结尾），以前缀为单位迁移，与 compute_dir_sizes 语义一致。
  - 目的前缀镜像源前缀（结构不变）；目的桶按 TRANSFER_BUCKET_MAP 推导。
  - 方向由源 scheme 决定链路与引擎（目的端拉取，服务设计决定）：
      tos://  → 阿里在线迁移 → oss        (一期)
      oss://  → 火山迁移      → tos        (二期)
      cpfs:// → 沉降→oss → 火山迁移 → tos  (三期，目的火山)
      vepfs://→ 沉降→tos → 阿里迁移 → oss  (三期，目的阿里)
"""
from dataclasses import dataclass

# 合法 scheme
_OBJ_SCHEMES = ("oss", "tos")
_PFS_SCHEMES = ("cpfs", "vepfs")
_ALL_SCHEMES = _OBJ_SCHEMES + _PFS_SCHEMES

# 全闪 → 本厂对象存储（沉降目的）；对象存储 → 跨云对方对象存储
_SINK_TARGET = {"cpfs": "oss", "vepfs": "tos"}   # 沉降落到同厂对象存储
_CROSS_TARGET = {"oss": "tos", "tos": "oss"}     # 跨云搬到对方对象存储

# 迁移引擎：目的端 scheme → 引擎名
_ENGINE_BY_DEST = {"oss": "mgw", "tos": "tos_mig"}   # 进 oss 用阿里在线迁移(mgw)；进 tos 用火山迁移


@dataclass
class Location:
    """一个对象存储/全闪位置。"""
    scheme: str          # oss/tos/cpfs/vepfs
    bucket: str          # 对象存储桶名 或 全闪 fs-id
    prefix: str          # 目录前缀（保证以 '/' 结尾，根为 ''）

    @property
    def is_object(self) -> bool:
        return self.scheme in _OBJ_SCHEMES

    def uri(self) -> str:
        return f"{self.scheme}://{self.bucket}/{self.prefix}"


class PathError(ValueError):
    """路径语法错误，消息面向用户。"""


def parse_location(raw: str) -> Location:
    """解析单个路径字符串为 Location。非法抛 PathError（消息可直接回显）。"""
    if not raw or "://" not in raw:
        raise PathError(f"路径格式错误：`{raw}`，应形如 `tos://bucket/prefix/`")
    scheme, rest = raw.split("://", 1)
    scheme = scheme.strip().lower()
    if scheme not in _ALL_SCHEMES:
        raise PathError(f"不支持的存储类型 `{scheme}`，仅支持 {'/'.join(_ALL_SCHEMES)}")
    rest = rest.lstrip("/")
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    if not bucket:
        raise PathError(f"路径缺少 bucket / 文件系统 ID：`{raw}`")
    # 规整前缀：非空且不以 / 结尾则补，便于目录语义
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return Location(scheme=scheme, bucket=bucket, prefix=prefix)


def _dest_bucket(src: Location, dest_scheme: str, bucket_map: dict) -> str:
    """按 TRANSFER_BUCKET_MAP 推导目的桶。映射键为 '<src_scheme>://<src_bucket>'。"""
    key = f"{src.scheme}://{src.bucket}"
    mapped = bucket_map.get(key)
    if not mapped:
        raise PathError(
            f"源 `{key}` 未在 TRANSFER_BUCKET_MAP 中配置目的 {dest_scheme} 桶，"
            f"请显式给出目的路径或补全映射。")
    return mapped


@dataclass
class Plan:
    """一次迁移的完整链路计划。"""
    source: Location
    dest: Location               # 最终落地（对方对象存储）
    sink_target: Location | None # 沉降中转（全闪源才有），对象存储源为 None
    engine: str                  # 跨云段引擎：mgw / tos_mig
    direction: str               # 人类可读，如 'tos→oss'

    @property
    def needs_sink(self) -> bool:
        return self.sink_target is not None


def build_plan(source_raw: str, dest_raw: str = "", *, bucket_map: dict) -> Plan:
    """从源（必填）、目的（可选）构造迁移计划。

    目的缺省时按 bucket_map 推导桶、镜像源前缀。
    返回 Plan；非法路径/缺映射抛 PathError。
    """
    src = parse_location(source_raw)

    # 全闪源：先沉降到同厂对象存储，再跨云
    sink_target = None
    cross_src = src
    if src.scheme in _PFS_SCHEMES:
        sink_scheme = _SINK_TARGET[src.scheme]
        # 沉降目的桶也需映射（全闪 fs-id 无法直接当桶名）
        sink_bucket = _dest_bucket(src, sink_scheme, bucket_map)
        sink_target = Location(scheme=sink_scheme, bucket=sink_bucket, prefix=src.prefix)
        cross_src = sink_target   # 跨云段从沉降落点起跳

    dest_scheme = _CROSS_TARGET[cross_src.scheme]

    if dest_raw:
        dest = parse_location(dest_raw)
        if dest.scheme != dest_scheme:
            raise PathError(
                f"方向不一致：源 `{src.scheme}` 应迁往 `{dest_scheme}`，"
                f"但目的是 `{dest.scheme}`。")
    else:
        dest_bucket = _dest_bucket(cross_src, dest_scheme, bucket_map)
        # 镜像源前缀，保持目录结构
        dest = Location(scheme=dest_scheme, bucket=dest_bucket, prefix=cross_src.prefix)

    engine = _ENGINE_BY_DEST[dest_scheme]
    direction = f"{src.scheme}→{dest.scheme}"
    return Plan(source=src, dest=dest, sink_target=sink_target,
                engine=engine, direction=direction)
