"""#63 第二阿里云主账号接入 —— 账号档案注册表（core/temp_ak_issuance/accounts.py）。

本文件只管**档案装配与查找**（registry 层）；跨账号数据隔离的对抗用例见
`test_temp_ak_account_isolation.py`。

覆盖：
  · 装配：默认档恒在；1949 档「缺 code / 缺 AK-ID / 缺 AK-SECRET → 不注册，三者齐 → 注册」；
    profiles() 不缓存（settings 被 monkeypatch 后立刻生效）。
  · by_issue_code / by_grant_id（tak- 与 tak1949- 不互相误认）/ by_slug（未注册 slug **抛错**）。
  · ram_client：显式传该档 ak/sk（不是默认档那对）。
  · issue_codes()：供 routes 白名单。
  · **默认档零回归**：前缀/命名/桶回退/permsync_client() 零参路径与多账号化前逐字一致。
"""
import pytest

from config.settings import settings
from core.temp_ak_issuance import accounts, issuer
from core.temp_ak_issuance import orchestrator as o

DEF_CODE = "5B4A3105-1EF9-4645-99D2-CCF69FE75D06"          # 现有账号「数据外采访问凭证申请」
CODE_1949 = "0133C4FC-8793-4FF3-A759-C4ECE8AC1FF9"         # 1949「数据访问凭证申请（产线）」
UID_1949 = "1339279783371949"


# ── 环境基线：默认档配齐、1949 档默认关闭 ──────────────────────────────────────
#   .env 里这些值可能有/可能没有，测试必须自己钉死，否则断言随机器飘。

@pytest.fixture(autouse=True)
def _base_env(monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_APPROVAL_CODE", DEF_CODE)
    monkeypatch.setattr(settings, "ALIYUN_ACCESS_KEY_ID", "LTAI_DEFAULT")
    monkeypatch.setattr(settings, "ALIYUN_ACCESS_KEY_SECRET", "SK_DEFAULT")
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default")
    # 1949 默认未接入
    monkeypatch.setattr(settings, "TEMP_AK_1949_APPROVAL_CODE", "")
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "")
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW", "{}")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "")
    monkeypatch.setattr(settings, "TEMP_AK_1949_COMMENT_USER_ID", "")


def _on_1949(monkeypatch, *, code=CODE_1949, ak="LTAI_1949", sk="SK_1949"):
    monkeypatch.setattr(settings, "TEMP_AK_1949_APPROVAL_CODE", code)
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_ID", ak)
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_SECRET", sk)


@pytest.fixture
def reg1949(monkeypatch):
    _on_1949(monkeypatch)
    return accounts.by_slug("1949")


# ══════════════════════════════════════════════════════════════════════════════
# settings 新字段登记
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", [
    "ALIYUN_1949_ACCESS_KEY_ID", "ALIYUN_1949_ACCESS_KEY_SECRET",
    "TEMP_AK_1949_APPROVAL_CODE", "TEMP_AK_1949_BUCKET_MAP_RAW",
    "TEMP_AK_1949_CHAT_ID", "TEMP_AK_1949_COMMENT_USER_ID",
    "TEMP_AK_1949_FIELD_SUBJECT", "TEMP_AK_1949_FIELD_PERM",
    "TEMP_AK_1949_FIELD_DATE_INTERVAL", "TEMP_AK_1949_FIELD_DIRECTORY",
    "TEMP_AK_1949_FIELD_NOTE",
])
def test_settings_declares_1949_fields(name):
    assert hasattr(settings, name), f"settings 缺字段 {name}（档案装配会静默取不到）"


# ══════════════════════════════════════════════════════════════════════════════
# 档案装配
# ══════════════════════════════════════════════════════════════════════════════

def test_default_profile_always_registered():
    ps = accounts.profiles()
    assert [p.slug for p in ps] == [accounts.DEFAULT_SLUG]
    assert accounts.DEFAULT_SLUG == ""


def test_1949_not_registered_only_when_all_config_empty():
    """三项（code/ak/sk）全空 = 该账号从未接入 → 不注册。"""
    assert [p.slug for p in accounts.profiles()] == [""]


@pytest.mark.parametrize("code,ak,sk", [
    (CODE_1949, "", ""),          # 只配了审批 code
    ("", "LTAI_1949", "SK_1949"),  # 只配了 AK
    (CODE_1949, "LTAI_1949", ""),  # AK 配了一半
])
def test_1949_registered_when_any_config_present(monkeypatch, code, ak, sk):
    """**任一相关配置存在即注册**（dev 的刻意选择）。

    理由：不注册会让该账号的 Redis 命名空间变成孤儿——`sweep_expired` 不再扫
    `temp_ak_1949:grant:*`（返回空、看着像"没有到期的"），而 `_key()` 又把 `tak1949-…`
    按默认档算成 `temp_ak:grant:…` 读不到 → 已发出去的长期 AK/子号永久残留在云上无人察觉。
    注册（前缀恒定、扫得到）+ 真要建号时在 ram_client() 显式抛错 = 可观测的失败。"""
    _on_1949(monkeypatch, code=code, ak=ak, sk=sk)
    assert [p.slug for p in accounts.profiles()] == ["", "1949"]


def test_1949_registered_without_ak_still_cannot_build_client(monkeypatch):
    """注册了但缺 AK：命名空间/前缀照旧可用（扫得到），但一旦要建号/删号就**显式抛错**。"""
    _on_1949(monkeypatch, ak="")
    p = accounts.by_slug("1949")
    assert p.redis_prefix == "temp_ak_1949:" and p.grant_prefix == "tak1949-"
    with pytest.raises(accounts.UnknownAccountError):
        accounts.ram_client(p)


def test_1949_without_code_is_not_routed(monkeypatch):
    """没配审批 code 的档案不参与发放路由（注册 ≠ 放行审批）。"""
    _on_1949(monkeypatch, code="")
    assert accounts.issue_codes() == {DEF_CODE}
    assert accounts.by_issue_code("") is None
    assert accounts.by_issue_code(CODE_1949) is None


def test_1949_registered_when_code_and_ak_present(reg1949):
    slugs = [p.slug for p in accounts.profiles()]
    assert slugs == ["", "1949"]
    assert reg1949.uid == UID_1949
    assert reg1949.issue_code == CODE_1949
    assert reg1949.ak_id == "LTAI_1949"
    assert reg1949.ak_secret == "SK_1949"


def test_profiles_is_not_cached(monkeypatch):
    """刻意不缓存：同一进程内改 settings 立刻生效（否则用例互相污染 / 运维改 .env 重启才生效）。"""
    assert len(accounts.profiles()) == 1
    _on_1949(monkeypatch)
    assert len(accounts.profiles()) == 2
    _on_1949(monkeypatch, code="", ak="", sk="")
    assert len(accounts.profiles()) == 1


def test_1949_isolation_prefixes(reg1949):
    """隔离维度的确切取值——改这些常量等于换命名空间，必须钉死。"""
    assert reg1949.grant_prefix == "tak1949-"
    assert reg1949.redis_prefix == "temp_ak_1949:"
    assert reg1949.user_prefix == "tempak-1949-"
    assert reg1949.display_suffix == "-1949产线临时用户"
    assert reg1949.subject_label == "使用人名称"


def test_1949_has_no_platform_field_default_has(reg1949):
    assert reg1949.has_platform_field is False      # 该模板无「平台」单选 → 恒阿里云
    assert accounts.default().has_platform_field is True


@pytest.mark.parametrize("key,widget", [
    ("enterprise",    "widget17846410216400001"),   # 「使用人名称」（逻辑键复用 enterprise）
    ("perm",          "widget17852975709640001"),
    ("date_interval", "widget17852976459760001"),
    ("directory",     "widget17852975954890001"),
    ("note",          "widget17852976732260001"),
])
def test_1949_field_widget_ids(reg1949, key, widget):
    """飞书 API 实拉的 widget id —— 写错就解析不到任何字段（表单全空 → 一律拒发）。"""
    env_name, aliases = reg1949.fields[key]
    assert widget in aliases
    assert env_name.startswith("TEMP_AK_1949_FIELD_")


def test_1949_fields_have_no_platform_key(reg1949):
    assert "platform" not in reg1949.fields
    assert "platform" in accounts.default().fields


def test_profile_is_frozen(reg1949):
    """档案不可变：任何一处 accounts.by_slug() 拿到的都不能被就地改成别的账号。"""
    with pytest.raises(Exception):
        reg1949.ak_id = "LTAI_HIJACK"


def test_label_readable(reg1949):
    assert accounts.default().label == "主账号"
    assert reg1949.label == "账号1949"


# ══════════════════════════════════════════════════════════════════════════════
# by_issue_code
# ══════════════════════════════════════════════════════════════════════════════

def test_by_issue_code_default():
    assert accounts.by_issue_code(DEF_CODE).slug == ""


def test_by_issue_code_1949(reg1949):
    assert accounts.by_issue_code(CODE_1949).slug == "1949"


def test_by_issue_code_each_only_matches_own(reg1949):
    """两账号 code 严格互斥——错配就会用 A 的模板解析 B 的表单、发到 A 的桶。"""
    assert accounts.by_issue_code(DEF_CODE).slug == ""
    assert accounts.by_issue_code(CODE_1949).slug == "1949"


def test_by_issue_code_unknown_none(reg1949):
    assert accounts.by_issue_code("SOME-OTHER-CODE") is None


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_by_issue_code_blank_none(bad):
    assert accounts.by_issue_code(bad) is None


def test_by_issue_code_ignores_profile_with_empty_code(monkeypatch):
    """默认档 code 留空时，空串不得匹配到默认档（否则无 code 的审批会被当成发放申请）。"""
    monkeypatch.setattr(settings, "TEMP_AK_APPROVAL_CODE", "")
    assert accounts.by_issue_code("") is None
    assert accounts.by_issue_code(DEF_CODE) is None


def test_by_issue_code_strips_whitespace(reg1949):
    assert accounts.by_issue_code(f"  {CODE_1949}  ").slug == "1949"


# ══════════════════════════════════════════════════════════════════════════════
# by_grant_id —— 共用的延长/撤销审批靠它分派
# ══════════════════════════════════════════════════════════════════════════════

def test_by_grant_id_default_prefix(reg1949):
    assert accounts.by_grant_id("tak-0123456789abcdef").slug == ""


def test_by_grant_id_1949_prefix(reg1949):
    assert accounts.by_grant_id("tak1949-0123456789abcdef").slug == "1949"


def test_by_grant_id_prefixes_do_not_cross_claim(reg1949):
    """对抗：两个前缀都以 `tak` 开头，绝不能互相误认（认错=拿错账号的 AK 改/删对方的号）。"""
    assert accounts.by_grant_id("tak-abc").slug != "1949"
    assert accounts.by_grant_id("tak1949-abc").slug != ""
    # 最长前缀优先：即便将来加 `tak19-` 之类的档，tak1949- 也不该被短前缀抢走
    picked = accounts.by_grant_id("tak1949-abc")
    assert picked.grant_prefix == "tak1949-"


def test_by_grant_id_unregistered_profile_returns_none(monkeypatch):
    """1949 档已下线（.env 删了 AK）但 Redis 还有它的 grant → 认不出来返 None，
    **不硬认成默认档**（否则会去默认账号里找/删同名对象）。"""
    assert accounts.by_grant_id("tak1949-abc") is None


@pytest.mark.parametrize("bad", ["", "   ", None, "xpfs-abc", "sgp-abc", "tak", "TAK-abc"])
def test_by_grant_id_rejects_foreign_ids(reg1949, bad):
    assert accounts.by_grant_id(bad) is None


# ══════════════════════════════════════════════════════════════════════════════
# by_slug —— 未注册必须抛错，绝不静默退默认档
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("blank", ["", None])
def test_by_slug_blank_is_default(blank):
    assert accounts.by_slug(blank).slug == ""


def test_by_slug_registered(reg1949):
    assert accounts.by_slug("1949").ak_id == "LTAI_1949"


def test_by_slug_unregistered_raises():
    """**本次隔离的核心守卫**：档案下线后拿 slug 取档必须炸，不能回默认档 ——
    回默认档就等于拿现有账号的 AK 去删/改 1949 账号的用户。"""
    with pytest.raises(accounts.UnknownAccountError):
        accounts.by_slug("1949")


def test_by_slug_garbage_raises(reg1949):
    with pytest.raises(accounts.UnknownAccountError):
        accounts.by_slug("9999")


def test_unknown_account_error_is_runtime_error():
    assert issubclass(accounts.UnknownAccountError, RuntimeError)


# ══════════════════════════════════════════════════════════════════════════════
# ram_client —— 显式传参
# ══════════════════════════════════════════════════════════════════════════════

def _spy_make_ram_client(monkeypatch):
    seen = {}
    import core.oss_perm.permsync as permsync

    def fake(ak="", sk=""):
        seen["ak"], seen["sk"] = ak, sk
        return object()

    monkeypatch.setattr(permsync, "make_ram_client", fake)
    return seen


def test_ram_client_passes_that_accounts_ak(monkeypatch, reg1949):
    seen = _spy_make_ram_client(monkeypatch)
    accounts.ram_client(reg1949)
    assert seen == {"ak": "LTAI_1949", "sk": "SK_1949"}
    # 绝不能是默认档那对
    assert seen["ak"] != "LTAI_DEFAULT" and seen["sk"] != "SK_DEFAULT"


def test_ram_client_default_profile_also_explicit(monkeypatch):
    seen = _spy_make_ram_client(monkeypatch)
    accounts.ram_client(accounts.default())
    assert seen == {"ak": "LTAI_DEFAULT", "sk": "SK_DEFAULT"}


def test_ram_client_missing_ak_raises(monkeypatch):
    """档案没 AK 直接炸，不落到 make_ram_client 的 env/settings 回退路径。"""
    called = _spy_make_ram_client(monkeypatch)
    monkeypatch.setattr(settings, "ALIYUN_ACCESS_KEY_ID", "")
    with pytest.raises(accounts.UnknownAccountError):
        accounts.ram_client(accounts.default())
    assert called == {}


# ══════════════════════════════════════════════════════════════════════════════
# issue_codes / bucket_map
# ══════════════════════════════════════════════════════════════════════════════

def test_issue_codes_both(reg1949):
    assert accounts.issue_codes() == {DEF_CODE, CODE_1949}


def test_issue_codes_only_default():
    assert accounts.issue_codes() == {DEF_CODE}


def test_issue_codes_skips_empty(monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_APPROVAL_CODE", "")
    assert accounts.issue_codes() == set()


def test_bucket_map_per_profile(monkeypatch, reg1949):
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW",
                        '{"共享名": {"region": "oss-cn-hangzhou", "bucket": "default-real"}}')
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW",
                        '{"共享名": {"region": "oss-cn-shanghai", "bucket": "prod1949-real"}}')
    assert accounts.bucket_map(accounts.default())["共享名"]["bucket"] == "default-real"
    assert accounts.bucket_map(accounts.by_slug("1949"))["共享名"]["bucket"] == "prod1949-real"


@pytest.mark.parametrize("raw", ["{not json", "", "[1,2]", "null"])
def test_bucket_map_bad_json_is_empty(monkeypatch, raw):
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW", raw)
    assert accounts.bucket_map(accounts.default()) == {}


# ══════════════════════════════════════════════════════════════════════════════
# 默认档零回归 —— 现有账号的一切取值不得因引入多账号而漂移
# ══════════════════════════════════════════════════════════════════════════════

def test_default_profile_literal_values(reg1949):
    """1949 档注册后，默认档取值也必须逐字不变。"""
    d = accounts.default()
    assert d.slug == ""
    assert d.grant_prefix == "tak-"
    assert d.redis_prefix == "temp_ak:"
    assert d.user_prefix == "tempak-"
    assert d.display_suffix == "-临时外采用户"
    assert d.subject_label == "使用企业名称"
    assert d.comment_user_id == ""          # 默认档从不覆盖评论身份
    assert d.ak_id == "LTAI_DEFAULT"


def test_default_redis_prefix_matches_legacy_constants(reg1949):
    """accounts 的默认前缀必须与 orchestrator 里的历史常量一字不差（历史数据还在那儿）。"""
    d = accounts.default()
    assert o._KEY_PREFIX == d.redis_prefix + "grant:"
    assert o._LOCK_PREFIX == d.redis_prefix + "lock:"


def test_default_grant_id_and_key_unchanged(reg1949):
    gid = o.grant_id_for("inst_legacy")
    assert gid.startswith("tak-")
    assert o._key(gid) == "temp_ak:grant:" + gid == o._KEY_PREFIX + gid


def test_default_lock_key_unchanged(reg1949, fake_redis):
    lock = o.claim("inst_legacy_lock")
    assert lock == "temp_ak:lock:inst_legacy_lock" == o._LOCK_PREFIX + "inst_legacy_lock"


def test_default_user_name_and_display_unchanged(reg1949):
    assert o._derive_user_name({"enterprise": "Acme"}, "inst_1").startswith("tempak-Acme".lower())
    assert o.display_name_for({"enterprise": "某某科技", "grant_id": "tak-abc"}) == "某某科技-临时外采用户"
    # 没有 grant_id 的历史记录也回默认后缀
    assert o.display_name_for({"enterprise": "某某科技"}) == "某某科技-临时外采用户"


def test_default_resolve_bucket_still_falls_back_to_permsync_map(reg1949, monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    region, bucket = o.resolve_bucket("新加坡-wuji-sing")
    assert (region, bucket) == ("oss-ap-southeast-1", "wuji-sing")


def test_permsync_client_zero_arg_path_unchanged(monkeypatch, reg1949):
    """`permsync_client()` / 无 account 字段的历史 grant → 走 make_ram_client() **零参**旧路径。"""
    calls = []
    import core.oss_perm.permsync as permsync
    monkeypatch.setattr(permsync, "make_ram_client",
                        lambda *a, **k: (calls.append((a, k)), object())[1])

    issuer.permsync_client()
    issuer.permsync_client({})                       # 无 account 键（历史 grant）
    issuer.permsync_client({"account": ""})          # 显式默认档
    assert calls == [((), {}), ((), {}), ((), {})]
