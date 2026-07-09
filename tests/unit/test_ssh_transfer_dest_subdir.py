"""#51 续 —— 目标子目录（B 语义）+ dest 白名单 + job_id/落库纳入 dest_rel。

纯逻辑（paths）+ orchestrator（fakeredis）。engine 全程不触碰。
B 语义：内容铺进目标子目录（`.../test/`），不套源目录名。空 dest → 镜像源前缀。
dest 也拼进泰国 ssh 双跳命令 → 必须同源前缀走白名单（防注入/穿越）。
"""
import pytest

from core.ssh_transfer import paths, orchestrator as orch
from core.ssh_transfer.paths import SshPathError, build_plan


# ── dest_subdir B 语义 ─────────────────────────────────────────────────────────

def test_dest_subdir_b_semantics():
    """build_plan(source, 'test') → 源前缀不变、dest_subdir='test/'、dest_rel()='test/'。"""
    p = build_plan("oss://wuji-data-tran/ossutil_output/", "test")
    assert p.source_prefix == "ossutil_output/"
    assert p.dest_subdir == "test/"
    assert p.dest_rel() == "test/"          # B 语义：铺进 test/，不套源目录名


def test_dest_empty_mirrors_source_prefix():
    """空 dest → dest_rel() 回落镜像源前缀。"""
    p = build_plan("oss://wuji-data-tran/ossutil_output/")
    assert p.dest_subdir == ""
    assert p.dest_rel() == "ossutil_output/"


def test_dest_empty_equivalent_to_explicit_empty():
    p1 = build_plan("oss://wuji-data-tran/ossutil_output/")
    p2 = build_plan("oss://wuji-data-tran/ossutil_output/", "")
    assert p1.dest_rel() == p2.dest_rel() == "ossutil_output/"


def test_dest_multilevel_ok():
    """多级 dest `a/b` 合法 → 'a/b/'。"""
    p = build_plan("oss://wuji-data-tran/ossutil_output/", "a/b")
    assert p.dest_subdir == "a/b/"
    assert p.dest_rel() == "a/b/"


def test_dest_strips_surrounding_slashes_and_whitespace():
    p = build_plan("oss://wuji-data-tran/ossutil_output/", "  /test/  ")
    assert p.dest_subdir == "test/"


# ── dest 白名单：命令注入 / 路径穿越（新用户输入面，必须挡） ─────────────────────

@pytest.mark.parametrize("dest", [
    "test; rm",          # 分号命令拼接
    "../etc",            # 路径穿越 ..
    "a b",               # 空格
    "a$(id)",            # 命令替换
    "a`id`",             # 反引号
    "a|b",               # 管道
    "a&b",               # 后台/逻辑与
    "a>b",               # 重定向
    "a'b",               # 单引号
    'a"b',               # 双引号
    "a\\b",              # 反斜杠
    "a\nb",              # 换行（远端 shell 可拆第二条命令）
    "a/../b",            # 中段穿越
    "a/./b",             # 中段单点
])
def test_dest_injection_rejected(dest):
    with pytest.raises(SshPathError):
        build_plan("oss://wuji-data-tran/ossutil_output/", dest)


def test_dest_segments_with_dot_underscore_dash_ok():
    p = build_plan("oss://wuji-data-tran/ossutil_output/", "v1.2_final-set")
    assert p.dest_subdir == "v1.2_final-set/"


# ── job_id 纳入 dest_rel（不同 dest 不同 id） ─────────────────────────────────────

def test_job_id_differs_by_dest_subdir():
    p1 = build_plan("oss://wuji-data-tran/ossutil_output/", "test")
    p2 = build_plan("oss://wuji-data-tran/ossutil_output/", "other")
    assert orch._job_id(p1) != orch._job_id(p2)


def test_job_id_empty_dest_matches_explicit_mirror():
    """空 dest 与镜像源前缀应产生同一 job_id（dest_rel 相同）。"""
    p_empty = build_plan("oss://wuji-data-tran/ossutil_output/")
    p_same = build_plan("oss://wuji-data-tran/ossutil_output/", "")
    assert orch._job_id(p_empty) == orch._job_id(p_same)


def test_job_id_custom_dest_differs_from_mirror():
    p_mirror = build_plan("oss://wuji-data-tran/ossutil_output/")
    p_custom = build_plan("oss://wuji-data-tran/ossutil_output/", "test")
    assert orch._job_id(p_mirror) != orch._job_id(p_custom)


# ── create_job_record 存 dest_subdir/dest_rel/dest_uri ──────────────────────────

def test_create_job_record_stores_dest_fields():
    p = build_plan("oss://wuji-data-tran/ossutil_output/", "test")
    job = orch.create_job_record(p, open_id="u1")
    assert job["dest_subdir"] == "test/"
    assert job["dest_rel"] == "test/"
    assert job["dest_uri"].endswith("/test/")          # 落地路径尾接目标子目录
    # 真落库
    assert orch.get_job(job["job_id"])["dest_rel"] == "test/"


def test_create_job_record_empty_dest_uses_source_prefix():
    p = build_plan("oss://wuji-data-tran/ossutil_output/")
    job = orch.create_job_record(p, open_id="u1")
    assert job["dest_subdir"] == ""
    assert job["dest_rel"] == "ossutil_output/"
    assert job["dest_uri"].endswith("/ossutil_output/")
