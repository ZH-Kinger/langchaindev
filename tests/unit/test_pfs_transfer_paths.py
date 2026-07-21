"""PFS 跨云直传 —— paths.build_plan / 端点解析 / staging 推导 / 校验（纯逻辑单测）。

方向自动判断（源 vepfs→P1 / 源 cpfs→P2）、两 staging 从 PFS_STAGING_MAP 推、
剥挂载前缀、拒同云/对象存储/非法组合、子目录白名单防注入穿越。
PFS_STAGING_MAP 用 monkeypatch settings.PFS_STAGING_MAP_RAW。
"""
import json

import pytest

from core.pfs_transfer import paths
from config.settings import settings

VEPFS_FS = "vepfs-abc"
CPFS_FS = "cpfs-xyz"

_MAP = {
    f"vepfs://{VEPFS_FS}": {"region": "cn-beijing", "tos_bucket": "wuji-dc-bj",
                            "tos_prefix": "pfs-staging/"},
    f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "wuji-il-hz",
                          "oss_prefix": "pfs-staging/", "dataflow_id": "df-123"},
}


@pytest.fixture
def stg_map(monkeypatch):
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(_MAP))
    # 默认 fs（裸挂载路径用）
    monkeypatch.setattr(settings, "VEPFS_FILE_SYSTEM_ID", VEPFS_FS)
    monkeypatch.setattr(settings, "CPFS_FILE_SYSTEM_ID", CPFS_FS)
    monkeypatch.setattr(settings, "VEPFS_MOUNT_PREFIX", "/vepfs")
    monkeypatch.setattr(settings, "CPFS_MOUNT_PREFIX", "/cpfs")
    return _MAP


# ══════════════════════════════════════════════════════════════════════════════
# 方向判定
# ══════════════════════════════════════════════════════════════════════════════

def test_direction_p1_vepfs_to_cpfs(stg_map):
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/wzh/data/", f"cpfs://{CPFS_FS}/team/data/")
    assert p.direction == paths.DIRECTION_P1 == "vepfs2cpfs"
    assert p.src_pfs.scheme == "vepfs" and p.dst_pfs.scheme == "cpfs"
    assert p.src_pfs.fs_id == VEPFS_FS and p.dst_pfs.fs_id == CPFS_FS
    assert p.src_pfs.sub_path == "wzh/data/" and p.dst_pfs.sub_path == "team/data/"


def test_direction_p2_cpfs_to_vepfs(stg_map):
    p = paths.build_plan(f"cpfs://{CPFS_FS}/team/data/", f"vepfs://{VEPFS_FS}/wzh/data/")
    assert p.direction == paths.DIRECTION_P2 == "cpfs2vepfs"
    assert p.src_pfs.scheme == "cpfs" and p.dst_pfs.scheme == "vepfs"


# ══════════════════════════════════════════════════════════════════════════════
# 两 staging 从 map 正确推导
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_staging_derivation(stg_map):
    """P1：源云 staging=vePFS 的 TOS 桶，目的云 staging=CPFS 的 OSS 桶（带 dataflow_id）。"""
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")
    # 源 staging = 沉降落点 = 跨云源 = TOS（火山，配 vePFS）
    assert p.src_staging.scheme == "tos"
    assert p.src_staging.bucket == "wuji-dc-bj"
    assert p.src_staging.base_prefix == "pfs-staging/"      # 补尾斜杠
    assert p.src_staging.region == "cn-beijing"
    assert p.src_staging.dataflow_id == ""                  # vePFS 侧无 dataflow
    # 目的 staging = 跨云落点 = 预热源 = OSS（阿里，配 CPFS）
    assert p.dst_staging.scheme == "oss"
    assert p.dst_staging.bucket == "wuji-il-hz"
    assert p.dst_staging.base_prefix == "pfs-staging/"
    assert p.dst_staging.region == "cn-hangzhou"
    assert p.dst_staging.dataflow_id == "df-123"            # CPFS 侧带 dataflow


def test_p2_staging_derivation_reversed(stg_map):
    """P2：源云 staging=CPFS 的 OSS 桶，目的云 staging=vePFS 的 TOS 桶。"""
    p = paths.build_plan(f"cpfs://{CPFS_FS}/b/", f"vepfs://{VEPFS_FS}/a/")
    assert p.src_staging.scheme == "oss" and p.src_staging.bucket == "wuji-il-hz"
    assert p.src_staging.dataflow_id == "df-123"
    assert p.dst_staging.scheme == "tos" and p.dst_staging.bucket == "wuji-dc-bj"
    assert p.dst_staging.dataflow_id == ""


def test_region_carried_onto_endpoints(stg_map):
    """区域从 map 带出到 PFS 端点（供子 orchestrator 用）。"""
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")
    assert p.src_pfs.region == "cn-beijing"
    assert p.dst_pfs.region == "cn-hangzhou"


def test_base_prefix_no_trailing_when_empty(monkeypatch):
    """staging prefix 留空 → base_prefix 为 ''（根），不硬塞斜杠。"""
    m = {
        f"vepfs://{VEPFS_FS}": {"region": "cn-beijing", "tos_bucket": "b1", "tos_prefix": ""},
        f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "b2", "oss_prefix": ""},
    }
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(m))
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")
    assert p.src_staging.base_prefix == ""
    assert p.dst_staging.base_prefix == ""


# ══════════════════════════════════════════════════════════════════════════════
# 缺 map 条目 / 缺 region / 缺 bucket
# ══════════════════════════════════════════════════════════════════════════════

def test_missing_map_entry_rejected(monkeypatch):
    """map 里没有该 fs → 明确报错。"""
    m = {f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "b2"}}
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(m))
    with pytest.raises(paths.PfsPathError, match="PFS_STAGING_MAP"):
        paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")


def test_missing_region_rejected(monkeypatch):
    m = {
        f"vepfs://{VEPFS_FS}": {"tos_bucket": "b1", "tos_prefix": "p/"},   # 缺 region
        f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "b2"},
    }
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(m))
    with pytest.raises(paths.PfsPathError, match="region"):
        paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")


def test_missing_bucket_rejected(monkeypatch):
    m = {
        f"vepfs://{VEPFS_FS}": {"region": "cn-beijing"},                   # 缺 tos_bucket
        f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "b2"},
    }
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(m))
    with pytest.raises(paths.PfsPathError, match="bucket"):
        paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")


def test_bad_json_map_treated_as_empty(monkeypatch):
    """坏 JSON → 空 map → 缺配置报错（不崩）。"""
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", "{not json")
    with pytest.raises(paths.PfsPathError, match="PFS_STAGING_MAP"):
        paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")


# ══════════════════════════════════════════════════════════════════════════════
# 拒同云 / 拒对象存储 / 拒非法地址
# ══════════════════════════════════════════════════════════════════════════════

def test_reject_same_cloud_vepfs_to_vepfs(stg_map):
    with pytest.raises(paths.PfsPathError, match="同一种 PFS|同类"):
        paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"vepfs://{VEPFS_FS}/b/")


def test_reject_same_cloud_cpfs_to_cpfs(stg_map):
    with pytest.raises(paths.PfsPathError, match="同一种 PFS|同类"):
        paths.build_plan(f"cpfs://{CPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")


@pytest.mark.parametrize("addr", [
    "oss://bucket/prefix/",
    "tos://bucket/prefix/",
])
def test_reject_object_storage_source(stg_map, addr):
    with pytest.raises(paths.PfsPathError, match="对象存储"):
        paths.build_plan(addr, f"cpfs://{CPFS_FS}/b/")


def test_reject_object_storage_dest(stg_map):
    with pytest.raises(paths.PfsPathError, match="对象存储"):
        paths.build_plan(f"vepfs://{VEPFS_FS}/a/", "oss://bucket/x/")


@pytest.mark.parametrize("addr", ["", "   ", "garbage-no-scheme", "ftp://x/y/"])
def test_reject_illegal_source_address(stg_map, addr):
    with pytest.raises(paths.PfsPathError):
        paths.build_plan(addr, f"cpfs://{CPFS_FS}/b/")


@pytest.mark.parametrize("bad", ["vepfs://", "vepfs:///", "cpfs://"])
def test_reject_missing_fs_id(stg_map, bad):
    """空 fs_id（scheme 后无文件系统段）→ 报缺文件系统 ID。
    注意：`vepfs://dir` 会把 dir 当 fs_id（合法），故只有 `vepfs://` / `vepfs:///` 才是缺 fs。"""
    with pytest.raises(paths.PfsPathError, match="文件系统 ID"):
        paths.build_plan(bad, f"cpfs://{CPFS_FS}/b/")


# ══════════════════════════════════════════════════════════════════════════════
# 子目录白名单：拒注入 / 穿越
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad", [
    "a; rm -rf x/",     # 分号 + 空格
    "../etc/passwd/",   # 穿越
    "a/../b/",          # 中段穿越
    "a b/",             # 空格
    "a$(id)/",          # 命令替换
    "a`whoami`/",       # 反引号
    "a|b/",             # 管道
    "a&b/",             # 后台符
    "a>b/",             # 重定向
    "a'b/",             # 引号
    "a\\b/",            # 反斜杠
    "a\nrm/",           # 换行
    "./x/",             # 单点整段
])
def test_reject_subdir_injection_traversal(stg_map, bad):
    with pytest.raises(paths.PfsPathError):
        paths.build_plan(f"vepfs://{VEPFS_FS}/{bad}", f"cpfs://{CPFS_FS}/b/")


@pytest.mark.parametrize("good", [
    "wzh/data/",
    "team-1/sub_dir/",
    "a.b/c.d/",
    "训练数据/2024/",     # 中文
])
def test_accept_valid_subdir(stg_map, good):
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/{good}", f"cpfs://{CPFS_FS}/b/")
    assert p.src_pfs.sub_path == good


def test_empty_subdir_is_fs_root(stg_map):
    """无子目录 → sub_path=''（fs 根）。"""
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/", f"cpfs://{CPFS_FS}/")
    assert p.src_pfs.sub_path == "" and p.dst_pfs.sub_path == ""


# ══════════════════════════════════════════════════════════════════════════════
# 裸挂载路径：剥前缀 + 默认 fs
# ══════════════════════════════════════════════════════════════════════════════

def test_bare_mount_vepfs_strips_prefix_uses_default_fs(stg_map):
    ep = paths._parse_endpoint("/vepfs/wzh/data/")
    assert ep.scheme == "vepfs"
    assert ep.fs_id == VEPFS_FS               # 默认 fs
    assert ep.sub_path == "wzh/data/"         # /vepfs 前缀剥净


def test_bare_mount_cpfs_strips_prefix_uses_default_fs(stg_map):
    ep = paths._parse_endpoint("/cpfs/team/data/")
    assert ep.scheme == "cpfs"
    assert ep.fs_id == CPFS_FS
    assert ep.sub_path == "team/data/"


def test_bare_mount_root_only(stg_map):
    """裸挂载根 /vepfs → sub_path ''。"""
    ep = paths._parse_endpoint("/vepfs")
    assert ep.scheme == "vepfs" and ep.sub_path == ""


def test_bare_mount_no_default_fs_rejected(monkeypatch):
    """裸挂载路径但未配默认 fs → 报错让用户显式给 fs。"""
    monkeypatch.setattr(settings, "VEPFS_FILE_SYSTEM_ID", "")
    monkeypatch.setattr(settings, "VEPFS_MOUNT_PREFIX", "/vepfs")
    with pytest.raises(paths.PfsPathError, match="VEPFS_FILE_SYSTEM_ID|默认"):
        paths._parse_endpoint("/vepfs/wzh/")


def test_bare_mount_not_over_stripped(stg_map):
    """不误剥以 vepfs 开头的真实目录（/vepfsdata 不是挂载前缀）。"""
    with pytest.raises(paths.PfsPathError):
        paths._parse_endpoint("/vepfsdata/x/")


def test_bare_mount_full_plan(stg_map):
    """裸挂载源+目的 → 完整 Plan，方向正确、staging 带出。"""
    p = paths.build_plan("/vepfs/wzh/", "/cpfs/team/")
    assert p.direction == paths.DIRECTION_P1
    assert p.src_pfs.fs_id == VEPFS_FS and p.src_pfs.sub_path == "wzh/"
    assert p.dst_pfs.fs_id == CPFS_FS and p.dst_pfs.sub_path == "team/"
    assert p.src_staging.scheme == "tos" and p.dst_staging.scheme == "oss"


# ══════════════════════════════════════════════════════════════════════════════
# 杂项
# ══════════════════════════════════════════════════════════════════════════════

def test_error_is_valueerror_subclass():
    assert issubclass(paths.PfsPathError, ValueError)


def test_plan_summary_str(stg_map):
    p = paths.build_plan(f"vepfs://{VEPFS_FS}/a/", f"cpfs://{CPFS_FS}/b/")
    s = p.summary()
    assert "vepfs" in s and "cpfs" in s and "→" in s
