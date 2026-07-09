"""#51 SSH 迁移链 —— paths.parse_source / build_plan 解析与校验。

纯逻辑，无外部依赖（不 import paramiko / 不连 SSH / 不碰 Redis）。
"""
import pytest

from core.ssh_transfer import paths
from core.ssh_transfer.paths import Plan, SshPathError, parse_source, build_plan


# ── 合法解析 ──────────────────────────────────────────────────────────────────

def test_parse_valid_dir():
    p = parse_source("oss://wuji-data-tran/a/b/")
    assert isinstance(p, Plan)
    assert p.source_bucket == "wuji-data-tran"
    assert p.source_prefix == "a/b/"
    assert p.source_uri() == "oss://wuji-data-tran/a/b/"


def test_parse_appends_trailing_slash():
    """前缀无尾斜杠自动补（目录语义）。"""
    p = parse_source("oss://wuji-data-tran/a/b")
    assert p.source_prefix == "a/b/"
    assert p.source_uri() == "oss://wuji-data-tran/a/b/"


def test_parse_single_level_prefix_gets_slash():
    p = parse_source("oss://bkt/data")
    assert p.source_bucket == "bkt"
    assert p.source_prefix == "data/"


def test_parse_root_prefix_empty():
    """根前缀 oss://bucket/ → prefix=''。"""
    p = parse_source("oss://wuji-data-tran/")
    assert p.source_bucket == "wuji-data-tran"
    assert p.source_prefix == ""
    assert p.source_uri() == "oss://wuji-data-tran/"


def test_parse_bucket_only_no_slash_empty_prefix():
    """oss://bucket（无任何斜杠）→ prefix=''。"""
    p = parse_source("oss://wuji-data-tran")
    assert p.source_bucket == "wuji-data-tran"
    assert p.source_prefix == ""


def test_parse_strips_whitespace():
    p = parse_source("  oss://wuji-data-tran/a/  ")
    assert p.source_bucket == "wuji-data-tran"
    assert p.source_prefix == "a/"


def test_parse_scheme_case_insensitive():
    p = parse_source("OSS://wuji-data-tran/a/")
    assert p.source_bucket == "wuji-data-tran"


# ── 非法输入 → SshPathError（消息可回显） ────────────────────────────────────

def test_reject_tos_scheme():
    with pytest.raises(SshPathError):
        parse_source("tos://some-bucket/a/")


def test_reject_cpfs_scheme():
    with pytest.raises(SshPathError):
        parse_source("cpfs://fs/a/")


def test_reject_missing_bucket_scheme_only():
    with pytest.raises(SshPathError):
        parse_source("oss://")


def test_reject_missing_bucket_only_slash():
    with pytest.raises(SshPathError):
        parse_source("oss:///")


def test_reject_empty_string():
    with pytest.raises(SshPathError):
        parse_source("")


def test_reject_whitespace_only():
    with pytest.raises(SshPathError):
        parse_source("   ")


def test_reject_none():
    with pytest.raises(SshPathError):
        parse_source(None)  # type: ignore[arg-type]


def test_reject_no_scheme():
    with pytest.raises(SshPathError):
        parse_source("wuji-data-tran/a/b/")


def test_ssh_path_error_is_valueerror():
    """SshPathError 继承 ValueError（上层可宽松捕获）。"""
    assert issubclass(SshPathError, ValueError)


# ── 安全白名单：命令注入 / 路径穿越（HIGH，prefix 拼进泰国 ssh 双跳命令） ────────

@pytest.mark.parametrize("raw", [
    "oss://wuji-data-tran/foo; rm -rf x/",   # 分号命令拼接
    "oss://wuji-data-tran/a$(id)/",          # 命令替换 $()
    "oss://wuji-data-tran/a`id`/",           # 反引号命令替换
    "oss://wuji-data-tran/a|b/",             # 管道
    "oss://wuji-data-tran/a b/",             # 空格
    "oss://wuji-data-tran/a&b/",             # 后台/逻辑与
    "oss://wuji-data-tran/a>b/",             # 重定向
    "oss://wuji-data-tran/a'b/",             # 单引号
    "oss://wuji-data-tran/a\"b/",            # 双引号
    "oss://wuji-data-tran/a\\b/",            # 反斜杠
    "oss://wuji-data-tran/a\nb/",            # 换行
])
def test_reject_prefix_injection(raw):
    with pytest.raises(SshPathError):
        parse_source(raw)


@pytest.mark.parametrize("raw", [
    "oss://wuji-data-tran/../../etc/",       # 路径穿越 ..
    "oss://wuji-data-tran/a/../b/",          # 中段 ..
    "oss://wuji-data-tran/./x/",             # 单点 .
    "oss://wuji-data-tran/a/./b/",           # 中段 .
])
def test_reject_prefix_traversal(raw):
    with pytest.raises(SshPathError):
        parse_source(raw)


@pytest.mark.parametrize("raw", [
    "oss://WujiData/a/",                      # 大写
    "oss://ab/a/",                            # 太短（<3）
    "oss://b/a/",                             # 单字符桶
    "oss://-bucket/a/",                       # 首字符非字母数字
    "oss://bucket-/a/",                       # 尾字符非字母数字
    "oss://buck_et/a/",                       # 含下划线（OSS 桶名不允许）
    "oss://bu.ck/a/",                         # 含点（本白名单不允许）
])
def test_reject_illegal_bucket(raw):
    with pytest.raises(SshPathError):
        parse_source(raw)


def test_accept_realistic_path():
    """真机风格路径仍通过、prefix 规整对。"""
    p = parse_source("oss://wuji-data-tran/wangyuran/AgiBotWorld-Beta/")
    assert p.source_bucket == "wuji-data-tran"
    assert p.source_prefix == "wangyuran/AgiBotWorld-Beta/"
    assert p.source_uri() == "oss://wuji-data-tran/wangyuran/AgiBotWorld-Beta/"


def test_accept_segments_with_dot_underscore_dash():
    """段内点/下划线/连字符合法（非 `.`/`..` 整段）。"""
    p = parse_source("oss://wuji-data-tran/v1.2_final-set/")
    assert p.source_prefix == "v1.2_final-set/"


def test_reject_trailing_newline_in_segment():
    """尾锚必须是 \\Z 而非 $：`aaa\\n` 段（换行紧跟斜杠前）不得被 `$` 结尾前匹配放行。

    正则 `$` 会在结尾换行符前匹配 → `oss://bucket/aaa\\n/` 用 `$` 会误放行；`\\Z` 收严后应拒。
    这是注入面：换行可在远端 shell 拆出第二条命令。
    """
    with pytest.raises(SshPathError):
        parse_source("oss://bucket/aaa\n/")


def test_reject_trailing_newline_in_bucket():
    """桶名尾锚同理：`bucket\\n` 不得被放行。"""
    with pytest.raises(SshPathError):
        parse_source("oss://bucket\n/aaa/")


# ── build_plan 对外入口 ───────────────────────────────────────────────────────

def test_build_plan_delegates_to_parse():
    p = build_plan("oss://wuji-data-tran/x/y/")
    assert p.source_bucket == "wuji-data-tran"
    assert p.source_prefix == "x/y/"


def test_build_plan_rejects_bad():
    with pytest.raises(SshPathError):
        build_plan("tos://b/x/")
