"""桶间迁移路径解析（纯逻辑，便于单测）。

只处理**同云同 scheme** 的一次性搬运：
    oss://bucket/prefix/  →  oss://bucket/prefix/    阿里 OSS→OSS（engine=mgw）
    tos://bucket/prefix/  →  tos://bucket/prefix/    火山 TOS→TOS（engine=dms）

源、目的都必填、都以 '/' 结尾（目录语义）。混合 scheme（oss↔tos）不在本模块处理，
提示改用跨云迁移。
"""
from dataclasses import dataclass

_OBJ_SCHEMES = ("oss", "tos")
_ENGINE_BY_SCHEME = {"oss": "mgw", "tos": "dms"}      # 同云迁移引擎
_CLOUD_BY_SCHEME = {"oss": "aliyun", "tos": "volcano"}


class PathError(ValueError):
    """路径语法错误，消息面向用户。"""


@dataclass
class Location:
    scheme: str
    bucket: str
    prefix: str          # 以 '/' 结尾，根为 ''

    def uri(self) -> str:
        return f"{self.scheme}://{self.bucket}/{self.prefix}"


@dataclass
class BucketPlan:
    cloud: str           # aliyun / volcano
    engine: str          # mgw / dms
    src: Location
    dest: Location
    direction: str       # 如 oss→oss / tos→tos


def parse_location(raw: str) -> Location:
    if not raw or "://" not in raw:
        raise PathError(f"路径格式错误：`{raw}`，应形如 `oss://bucket/prefix/` 或 `tos://bucket/prefix/`")
    scheme, rest = raw.split("://", 1)
    scheme = scheme.strip().lower()
    if scheme not in _OBJ_SCHEMES:
        raise PathError(f"桶间迁移只支持对象存储 oss:// / tos://，不支持 `{scheme}`")
    rest = rest.lstrip("/")
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise PathError(f"路径缺少 bucket：`{raw}`")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return Location(scheme=scheme, bucket=bucket, prefix=prefix)


def build_plan(source_raw: str, dest_raw: str) -> BucketPlan:
    """解析源/目的为同云迁移计划。源目的都必填、同 scheme。"""
    if not source_raw or not dest_raw:
        raise PathError("桶间迁移的源、目的地址都必须填写（同云无法自动推导目的桶）。")
    src = parse_location(source_raw)
    dest = parse_location(dest_raw)
    if src.scheme != dest.scheme:
        raise PathError(
            f"源 `{src.scheme}` 与目的 `{dest.scheme}` 不同云——桶间迁移只做同云"
            f"（oss→oss / tos→tos）。跨云 oss↔tos 请用「跨云迁移」。")
    if src.bucket == dest.bucket and src.prefix == dest.prefix:
        raise PathError("源和目的完全相同，无需迁移。")
    return BucketPlan(
        cloud=_CLOUD_BY_SCHEME[src.scheme],
        engine=_ENGINE_BY_SCHEME[src.scheme],
        src=src, dest=dest,
        direction=f"{src.scheme}→{dest.scheme}",
    )
