"""#42 回归：CPFS/vePFS 数据流动「多套一层 cpfs/vepfs 目录」的挂载前缀剥离。

真机 bug：卡片向导直达走 `make_plan(..., fs_id=...)` 分支，该分支原来用
`normalize_dir(cpfs_path)`（不剥挂载前缀），把 `/cpfs` 当成真实目录名拼进
DstDirectory → 数据落 `<mount>/cpfs/<dir>`（多一层）。修法：抽 `_strip_mount`
（cpfs 用 `settings.CPFS_MOUNT_PREFIX`、vepfs 用新增 `settings.VEPFS_MOUNT_PREFIX`），
在 `_parse_*` 与 `make_plan` 的 fs_id 分支都剥。

本文件锁死：
- `_strip_mount` 语义（含前缀 / 无前缀 / 前缀���身 / 相似但非前缀不误剥）;
- make_plan(fs_id=...) 关键回归——preheat/sink 两方向都不含挂载前缀、不双套;
- make_plan 无 fs_id（cpfs:// / vepfs:// scheme 与裸目录）也剥净;
- vePFS 的 plan_from_addresses（源/目的地址两方向）经 make_plan 兜住;
- settings 默认前缀值。
"""
import pytest

from core.cpfs_dataflow import orchestrator as corch
from core.vepfs_dataflow import engine_vepfs, orchestrator as vorch


# --------------------------------------------------------------------------- #
# CPFS _strip_mount 直测
# --------------------------------------------------------------------------- #
@pytest.fixture
def cpfs_mount(monkeypatch):
    monkeypatch.setattr(corch.settings, "CPFS_MOUNT_PREFIX", "/cpfs")


def test_cpfs_strip_mount_with_prefix(cpfs_mount):
    assert corch._strip_mount("/cpfs/xiaoxiong/sft_data_v1_le/") == "/xiaoxiong/sft_data_v1_le/"


def test_cpfs_strip_mount_no_prefix_unchanged(cpfs_mount):
    assert corch._strip_mount("/xiaoxiong/") == "/xiaoxiong/"


def test_cpfs_strip_mount_prefix_itself_becomes_root(cpfs_mount):
    assert corch._strip_mount("/cpfs") == "/"


def test_cpfs_strip_mount_does_not_misstrip_similar_dir(cpfs_mount):
    # 真实目录名以 cpfs 开头但不是挂载前缀（既不等于 /cpfs 也不以 /cpfs/ 开头）→ 不剥。
    assert corch._strip_mount("/cpfsdata/x/") == "/cpfsdata/x/"


def test_cpfs_strip_mount_trailing_slash_prefix(monkeypatch):
    # 配置带尾斜杠也应被 rstrip 归一，行为一致。
    monkeypatch.setattr(corch.settings, "CPFS_MOUNT_PREFIX", "/cpfs/")
    assert corch._strip_mount("/cpfs/a/b/") == "/a/b/"


# --------------------------------------------------------------------------- #
# CPFS make_plan(fs_id=...) —— 关键回归：卡片向导直达分支
# --------------------------------------------------------------------------- #
def test_cpfs_make_plan_fsid_preheat_strips_mount(cpfs_mount):
    plan = corch.make_plan("preheat", "/cpfs/a/b/", "oss://bk/a/b/", fs_id="cpfs-x")
    # cpfs_dir 与 dst_directory 都不含 /cpfs、不双套
    assert plan.cpfs_dir == "/a/b/"
    assert plan.dst_directory == "/a/b/"
    assert "/cpfs" not in plan.cpfs_dir
    assert "/cpfs" not in plan.dst_directory


def test_cpfs_make_plan_fsid_sink_strips_mount(cpfs_mount):
    plan = corch.make_plan("sink", "/cpfs/a/b/", "oss://bk/x/", fs_id="cpfs-x")
    # sink 方向 CPFS 侧走 Directory
    assert plan.cpfs_dir == "/a/b/"
    assert plan.directory == "/a/b/"
    assert "/cpfs" not in plan.directory


def test_cpfs_make_plan_fsid_prefix_itself(cpfs_mount):
    plan = corch.make_plan("sink", "/cpfs", "oss://bk/x/", fs_id="cpfs-x")
    assert plan.cpfs_dir == "/"


def test_cpfs_make_plan_fsid_no_misstrip(cpfs_mount):
    plan = corch.make_plan("sink", "/cpfsdata/x/", "oss://bk/x/", fs_id="cpfs-x")
    assert plan.cpfs_dir == "/cpfsdata/x/"


# --------------------------------------------------------------------------- #
# CPFS make_plan 无 fs_id —— cpfs:// scheme 与裸 /cpfs/ 路径
# --------------------------------------------------------------------------- #
def test_cpfs_make_plan_scheme_path(cpfs_mount):
    plan = corch.make_plan("preheat", "cpfs://cpfs-x/a/b/", "oss://bk/a/b/")
    assert plan.fs_id == "cpfs-x"
    assert plan.cpfs_dir == "/a/b/"
    assert "/cpfs" not in plan.cpfs_dir


def test_cpfs_make_plan_bare_full_path(monkeypatch):
    monkeypatch.setattr(corch.settings, "CPFS_MOUNT_PREFIX", "/cpfs")
    monkeypatch.setattr(corch.settings, "CPFS_FILE_SYSTEM_ID", "cpfs-default")
    plan = corch.make_plan("preheat", "/cpfs/a/b/", "oss://bk/a/b/")
    assert plan.fs_id == "cpfs-default"
    assert plan.cpfs_dir == "/a/b/"
    assert "/cpfs" not in plan.cpfs_dir


# --------------------------------------------------------------------------- #
# vePFS _strip_mount 直测
# --------------------------------------------------------------------------- #
@pytest.fixture
def vepfs_mount(monkeypatch):
    monkeypatch.setattr(vorch.settings, "VEPFS_MOUNT_PREFIX", "/vepfs")


def test_vepfs_strip_mount_with_prefix(vepfs_mount):
    assert vorch._strip_mount("/vepfs/wzh/") == "/wzh/"


def test_vepfs_strip_mount_no_prefix_unchanged(vepfs_mount):
    assert vorch._strip_mount("/wzh/") == "/wzh/"


def test_vepfs_strip_mount_prefix_itself_becomes_root(vepfs_mount):
    assert vorch._strip_mount("/vepfs") == "/"


def test_vepfs_strip_mount_does_not_misstrip_similar_dir(vepfs_mount):
    assert vorch._strip_mount("/vepfsdata/x/") == "/vepfsdata/x/"


# --------------------------------------------------------------------------- #
# vePFS make_plan(fs_id=...) 与 plan_from_addresses 两入口
# --------------------------------------------------------------------------- #
def test_vepfs_make_plan_fsid_strips_mount(vepfs_mount):
    plan = vorch.make_plan("sink", "/vepfs/wzh/", "tos://bk/wzh/", fs_id="vepfs-x")
    assert plan.sub_path == "/wzh/"
    assert "/vepfs" not in plan.sub_path


def test_vepfs_make_plan_fsid_preheat_strips_mount(vepfs_mount):
    plan = vorch.make_plan("preheat", "/vepfs/wzh/", "tos://bk/wzh/", fs_id="vepfs-x")
    assert plan.sub_path == "/wzh/"
    assert "/vepfs" not in plan.sub_path


def test_vepfs_make_plan_fsid_no_misstrip(vepfs_mount):
    plan = vorch.make_plan("sink", "/vepfsdata/x/", "tos://bk/x/", fs_id="vepfs-x")
    assert plan.sub_path == "/vepfsdata/x/"


def test_vepfs_plan_from_addresses_sink_strips_mount(monkeypatch):
    # 源=vePFS(/vepfs/wzh/)、目的=TOS → 沉降；经 make_plan 兜住剥前缀。
    monkeypatch.setattr(vorch.settings, "VEPFS_MOUNT_PREFIX", "/vepfs")
    monkeypatch.setattr(vorch.settings, "VEPFS_FILE_SYSTEM_ID", "vepfs-def")
    plan = vorch.plan_from_addresses("cn-beijing", "/vepfs/wzh/", "tos://bk/wzh/")
    assert plan.operation == "sink"
    assert plan.sub_path == "/wzh/"
    assert "/vepfs" not in plan.sub_path


def test_vepfs_plan_from_addresses_preheat_strips_mount(monkeypatch):
    # 源=TOS、目的=vePFS(/vepfs/wzh/) → 预热；同样剥净。
    monkeypatch.setattr(vorch.settings, "VEPFS_MOUNT_PREFIX", "/vepfs")
    monkeypatch.setattr(vorch.settings, "VEPFS_FILE_SYSTEM_ID", "vepfs-def")
    plan = vorch.plan_from_addresses("cn-beijing", "tos://bk/wzh/", "/vepfs/wzh/")
    assert plan.operation == "preheat"
    assert plan.sub_path == "/wzh/"
    assert "/vepfs" not in plan.sub_path


# --------------------------------------------------------------------------- #
# settings 默认前缀
# --------------------------------------------------------------------------- #
def test_settings_default_mount_prefixes():
    from config.settings import settings
    assert settings.CPFS_MOUNT_PREFIX == "/cpfs"
    assert settings.VEPFS_MOUNT_PREFIX == "/vepfs"
