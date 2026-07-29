"""PFS↔PFS 跨云直传 路径解析 / Plan（纯逻辑，便于单测）。

输入：源 PFS 地址 + 目的 PFS 地址。两种写法：
    显式：`vepfs://<fs-id>/<子目录>/`  `cpfs://<fs-id>/<子目录>/`
    裸挂载路径：`/vepfs/<子目录>/`（用默认 fs VEPFS_FILE_SYSTEM_ID）、`/cpfs/<子目录>/`（CPFS_FILE_SYSTEM_ID）

方向由**源**的 scheme 自动判断：
    源 vepfs → 目的 cpfs  = P1（vePFS→CPFS）
    源 cpfs  → 目的 vepfs = P2（CPFS→vePFS）
同云 / 含对象存储 scheme / 方向不匹配 → 拒绝。

两个中转 staging（沉降落点=跨云源、跨云落点=预热源）从 `PFS_STAGING_MAP` 推导：
    key = `<pfs-scheme>://<fs-id>`；value = {region, tos_bucket/oss_bucket, tos_prefix/oss_prefix[, dataflow_id]}
区域硬约束（配置契约）：map 里每个 fs 的 region 即该 PFS 与其 staging 桶的共同区域
（vePFS↔TOS 同区、CPFS↔OSS 同区）；跨云段②可跨区。

本模块只解析+校验+推导结构；chain 级 staging 前缀（含 chain_id 隔离）在 orchestrator 拼。
"""
import json
import re
from dataclasses import dataclass, field

from config.settings import settings

VEPFS = "vepfs"
CPFS = "cpfs"
_PFS_SCHEMES = (VEPFS, CPFS)
# 对象存储 scheme 不允许作为 PFS 直传的端点（那是跨云/桶间迁移的活）
_OBJ_SCHEMES = ("oss", "tos")

# fs-id：字母数字/连字符/下划线（vePFS `vepfs-xxxx` / CPFS `cpfs-xxxx`/`bmcpfs-xxxx`）
_FS_RE = re.compile(r"\A[A-Za-z0-9._\-]+\Z")
# 子目录每一级：字母/数字/点/下划线/连字符/中文；禁 `..`/`.`/空格/shell 元字符
_SEG_RE = re.compile(r"\A[\w.\-一-鿿]+\Z")


class PfsPathError(ValueError):
    """PFS 直传路径语法/配置错误，消息面向用户。"""


@dataclass
class PfsEndpoint:
    scheme: str        # vepfs / cpfs
    fs_id: str         # 文件系统 ID
    sub_path: str      # 文件系统内子目录，'/' 结尾（根为 ''）
    region: str = ""   # 由 staging map 补


@dataclass
class Staging:
    scheme: str            # tos（火山，配 vePFS）/ oss（阿里，配 CPFS）
    bucket: str
    base_prefix: str       # 配置里的基前缀，'/' 结尾（根为 ''）
    region: str
    dataflow_id: str = ""  # CPFS 侧可选（省 resolve_dataflow）


@dataclass
class Plan:
    direction: str                 # "vepfs2cpfs"(P1) / "cpfs2vepfs"(P2)
    src_pfs: PfsEndpoint
    dst_pfs: PfsEndpoint
    src_staging: Staging           # 沉降落点（跨云源）——源云对象存储
    dst_staging: Staging           # 跨云落点（预热源）——目的云对象存储
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.src_pfs.scheme}://{self.src_pfs.fs_id}/{self.src_pfs.sub_path} "
                f"→ {self.dst_pfs.scheme}://{self.dst_pfs.fs_id}/{self.dst_pfs.sub_path}")


DIRECTION_P1 = "vepfs2cpfs"
DIRECTION_P2 = "cpfs2vepfs"


def _staging_map() -> dict:
    """解析 PFS_STAGING_MAP（JSON）。坏 JSON → 空 dict（上层按缺配置报错）。"""
    try:
        m = json.loads(settings.PFS_STAGING_MAP_RAW or "{}")
        return m if isinstance(m, dict) else {}
    except (ValueError, TypeError):
        return {}


def _mount_prefix(scheme: str) -> str:
    p = (settings.VEPFS_MOUNT_PREFIX if scheme == VEPFS else settings.CPFS_MOUNT_PREFIX) or ""
    return "/" + p.strip("/") if p.strip("/") else ""


def _default_fs(scheme: str) -> str:
    return (settings.VEPFS_FILE_SYSTEM_ID if scheme == VEPFS else settings.CPFS_FILE_SYSTEM_ID) or ""


def _norm_subdir(raw: str) -> str:
    """规整子目录：去首尾斜杠/空白、补尾斜杠、逐级白名单（防注入/穿越）。空→''（fs 根）。"""
    raw = (raw or "").strip().strip("/")
    if not raw:
        return ""
    sub = raw + "/"
    for seg in sub.rstrip("/").split("/"):
        if not seg or seg in (".", "..") or not _SEG_RE.match(seg):
            raise PfsPathError(
                f"子目录含非法字符或路径穿越：`{raw}`。每级只允许 字母/数字/中文/`.`/`_`/`-`，禁 `..`/空格/特殊符号。")
    return sub


def _parse_endpoint(raw: str) -> PfsEndpoint:
    """解析单个 PFS 端点。支持 `vepfs://fs/dir/`、`cpfs://fs/dir/`、裸挂载 `/vepfs/dir/`。"""
    raw = (raw or "").strip()
    if not raw:
        raise PfsPathError("PFS 地址为空。")

    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        scheme = scheme.strip().lower()
        if scheme in _OBJ_SCHEMES:
            raise PfsPathError(
                f"PFS 直传两端必须是 PFS（vepfs:// 或 cpfs://），收到对象存储 `{scheme}://`——"
                f"对象存储搬运请用跨云迁移/桶间迁移。")
        if scheme not in _PFS_SCHEMES:
            raise PfsPathError(f"不支持的 PFS scheme `{scheme}://`，只支持 vepfs:// / cpfs://。")
        rest = rest.lstrip("/")
        if "/" in rest:
            fs_id, sub = rest.split("/", 1)
        else:
            fs_id, sub = rest, ""
        if not fs_id:
            raise PfsPathError(f"PFS 地址缺少文件系统 ID：`{raw}`（应形如 `{scheme}://<fs-id>/子目录/`）。")
        if not _FS_RE.match(fs_id):
            raise PfsPathError(f"文件系统 ID 非法：`{fs_id}`。")
        return PfsEndpoint(scheme=scheme, fs_id=fs_id, sub_path=_norm_subdir(sub))

    # 裸挂载路径：/vepfs/... 或 /cpfs/...
    if raw.startswith("/"):
        for scheme in _PFS_SCHEMES:
            mp = _mount_prefix(scheme)
            if mp and (raw == mp or raw.startswith(mp + "/")):
                sub = raw[len(mp):]
                fs_id = _default_fs(scheme)
                if not fs_id:
                    raise PfsPathError(
                        f"裸挂载路径 `{raw}` 用默认 {scheme.upper()} 文件系统，但未配置 "
                        f"{'VEPFS_FILE_SYSTEM_ID' if scheme == VEPFS else 'CPFS_FILE_SYSTEM_ID'}；"
                        f"请改用显式 `{scheme}://<fs-id>/子目录/`。")
                return PfsEndpoint(scheme=scheme, fs_id=fs_id, sub_path=_norm_subdir(sub))
    raise PfsPathError(
        f"无法识别的 PFS 地址：`{raw}`。应形如 `vepfs://<fs>/dir/`、`cpfs://<fs>/dir/`，"
        f"或裸挂载路径 `{_mount_prefix(VEPFS)}/dir/`、`{_mount_prefix(CPFS)}/dir/`。")


def _staging_for(ep: PfsEndpoint) -> Staging:
    """按 PFS 端点查 PFS_STAGING_MAP，构造其本云 staging（vePFS→TOS / CPFS→OSS）。"""
    m = _staging_map()
    key = f"{ep.scheme}://{ep.fs_id}"
    entry = m.get(key)
    if not isinstance(entry, dict):
        raise PfsPathError(
            f"未在 PFS_STAGING_MAP 找到 `{key}` 的中转桶配置。请配置该文件系统的 "
            f"{'TOS' if ep.scheme == VEPFS else 'OSS'} staging 桶（region + 桶名 + 前缀"
            f"{'[+dataflow_id]' if ep.scheme == CPFS else ''}）。")
    region = str(entry.get("region") or "").strip()
    if not region:
        raise PfsPathError(f"PFS_STAGING_MAP `{key}` 缺 region。")
    if ep.scheme == VEPFS:
        bucket = str(entry.get("tos_bucket") or "").strip()
        prefix = str(entry.get("tos_prefix") or "").strip().strip("/")
        obj_scheme = "tos"
    else:
        bucket = str(entry.get("oss_bucket") or "").strip()
        prefix = str(entry.get("oss_prefix") or "").strip().strip("/")
        obj_scheme = "oss"
    if not bucket:
        raise PfsPathError(f"PFS_STAGING_MAP `{key}` 缺 {obj_scheme}_bucket。")
    ep.region = region
    return Staging(
        scheme=obj_scheme,
        bucket=bucket,
        base_prefix=(prefix + "/") if prefix else "",
        region=region,
        dataflow_id=str(entry.get("dataflow_id") or "").strip(),
    )


def build_plan(source_raw: str, dest_raw: str) -> Plan:
    """对外入口：源 PFS 地址 + 目的 PFS 地址 → Plan。非法抛 PfsPathError（可直接回显）。"""
    src = _parse_endpoint(source_raw)
    dst = _parse_endpoint(dest_raw)

    # 方向 + 组合校验：必须一 vePFS 一 CPFS，方向由源决定
    if src.scheme == dst.scheme:
        raise PfsPathError(
            f"源和目的是同一种 PFS（都是 {src.scheme}）——PFS 直传是 vePFS↔CPFS 跨云。"
            f"同类文件系统之间不走本链。")
    if src.scheme == VEPFS and dst.scheme == CPFS:
        direction = DIRECTION_P1
    elif src.scheme == CPFS and dst.scheme == VEPFS:
        direction = DIRECTION_P2
    else:
        raise PfsPathError(f"不支持的方向：{src.scheme} → {dst.scheme}。")

    src_staging = _staging_for(src)   # 源云对象存储（沉降落点=跨云源）
    dst_staging = _staging_for(dst)   # 目的云对象存储（跨云落点=预热源）

    # P2(CPFS→vePFS) 的跨云段是 OSS→TOS，走火山 DMS —— 而 DMS 1.0 **只能指定目标桶、指不了目标
    # 前缀**，对象保持**源 key 结构**落进目的桶。所以第③段预热能不能找到数据，完全取决于
    # 「源 staging 前缀 == 目的 staging 前缀」这个巧合：两侧都是 pfs-staging/<链ID>/ 时，
    # 源 key 前缀恰好等于预热期望的前缀，才对得上。
    # 一旦两侧配成不同前缀，前两段会正常成功、第③段静默地去错误位置找数据 —— 极难排查。
    # P1(vePFS→CPFS) 的跨云是 TOS→OSS 走阿里在线迁移，能指定目标前缀，不受此限制。
    if direction == DIRECTION_P2 and src_staging.base_prefix != dst_staging.base_prefix:
        raise PfsPathError(
            f"PFS_STAGING_MAP 里两侧 staging 前缀不一致（源 `{src_staging.base_prefix or '<根>'}` "
            f"vs 目的 `{dst_staging.base_prefix or '<根>'}`）。CPFS→vePFS 的跨云段走火山 DMS，"
            f"只能指定目标桶、对象按源 key 结构落地，两侧前缀必须相同，否则预热段会找不到数据。"
            f"请把 `oss_prefix` 与 `tos_prefix` 配成同一个值。")

    return Plan(
        direction=direction,
        src_pfs=src,
        dst_pfs=dst,
        src_staging=src_staging,
        dst_staging=dst_staging,
    )
