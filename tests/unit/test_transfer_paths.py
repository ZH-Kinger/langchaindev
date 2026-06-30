"""core/transfer/paths.py 测试：路径解析、方向判断、目的推导。纯逻辑，全覆盖。"""
import pytest

from core.transfer import paths
from core.transfer.paths import PathError, parse_location, build_plan

BMAP = {
    "tos://wuji-egocentric-data": "wuji-bucket-hangzhou",
    "oss://wuji-bucket-hangzhou": "wuji-egocentric-data",
    "cpfs://cpfs-abc": "wuji-bucket-hangzhou",
    "vepfs://vepfs-xyz": "wuji-egocentric-data",
}


# ── parse_location ──────────────────────────────────────────────────────────

def test_parse_basic():
    loc = parse_location("tos://bk/team/ds/")
    assert (loc.scheme, loc.bucket, loc.prefix) == ("tos", "bk", "team/ds/")
    assert loc.is_object


def test_parse_appends_trailing_slash():
    assert parse_location("oss://bk/a/b").prefix == "a/b/"


def test_parse_root_prefix_empty():
    loc = parse_location("tos://bk")
    assert loc.prefix == ""
    assert loc.bucket == "bk"


def test_parse_scheme_case_insensitive():
    assert parse_location("TOS://bk/p/").scheme == "tos"


@pytest.mark.parametrize("bad", ["", "bk/p", "ftp://bk/p", "tos://", "://bk"])
def test_parse_invalid_raises(bad):
    with pytest.raises(PathError):
        parse_location(bad)


# ── build_plan: 对象存储源 ───────────────────────────────────────────────────

def test_plan_tos_to_oss_explicit_dest():
    p = build_plan("tos://src/team/ds/", "oss://dst/team/ds/", bucket_map=BMAP)
    assert p.direction == "tos→oss"
    assert p.engine == "mgw"
    assert not p.needs_sink
    assert p.dest.bucket == "dst"
    assert p.dest.prefix == "team/ds/"


def test_plan_tos_to_oss_inferred_dest():
    p = build_plan("tos://wuji-egocentric-data/x/y/", bucket_map=BMAP)
    assert p.dest.scheme == "oss"
    assert p.dest.bucket == "wuji-bucket-hangzhou"
    assert p.dest.prefix == "x/y/"        # 镜像源前缀


def test_plan_oss_to_tos_engine():
    p = build_plan("oss://wuji-bucket-hangzhou/a/", bucket_map=BMAP)
    assert p.direction == "oss→tos"
    assert p.engine == "tos_mig"
    assert p.dest.bucket == "wuji-egocentric-data"


def test_plan_inferred_missing_map_raises():
    with pytest.raises(PathError):
        build_plan("tos://unmapped-bucket/x/", bucket_map=BMAP)


def test_plan_direction_mismatch_raises():
    with pytest.raises(PathError):
        build_plan("tos://src/a/", "tos://dst/a/", bucket_map=BMAP)  # tos 应迁往 oss


# ── build_plan: 全闪源（沉降链路）───────────────────────────────────────────

def test_plan_cpfs_full_chain():
    p = build_plan("cpfs://cpfs-abc/data/", bucket_map=BMAP)
    assert p.needs_sink
    assert p.sink_target.scheme == "oss"               # CPFS 沉降到 OSS
    assert p.sink_target.bucket == "wuji-bucket-hangzhou"
    assert p.dest.scheme == "tos"                      # 再跨云到火山
    assert p.engine == "tos_mig"
    assert p.direction == "cpfs→tos"


def test_plan_vepfs_full_chain():
    p = build_plan("vepfs://vepfs-xyz/data/", bucket_map=BMAP)
    assert p.sink_target.scheme == "tos"               # VePFS 沉降到 TOS
    assert p.dest.scheme == "oss"                      # 再跨云到阿里
    assert p.engine == "mgw"
    assert p.direction == "vepfs→oss"
