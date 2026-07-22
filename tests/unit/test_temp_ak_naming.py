"""#59 A —— 命名规范化（orchestrator._ascii_slug / _derive_user_name / display_name_for）
+ issuer._issue_ram 建号 display_name。

pypinyin 本地未装 → 测降级路径（纯中文→'ext'、混合→只留 ASCII）；
中文→拼音分支用注入假 pypinyin 模块 monkeypatch 测。
"""
import sys
import types

import pytest

from core.temp_ak_issuance import orchestrator as o, issuer


# ── _ascii_slug ───────────────────────────────────────────────────────────────

def test_slug_ascii_lowercased():
    assert o._ascii_slug("TEST") == "test"


def test_slug_pure_chinese_no_pypinyin_falls_back_ext():
    """纯中文 + 无 pypinyin（本地未装）→ ASCII 化后为空 → 'ext'。"""
    # 确保没有 pypinyin（本地默认未装；若装了则跳过该降级断言）
    if "pypinyin" in sys.modules or _pypinyin_installed():
        pytest.skip("环境装了 pypinyin，降级分支不适用")
    assert o._ascii_slug("某某科技") == "ext"


def test_slug_mixed_keeps_ascii_only_without_pypinyin():
    if _pypinyin_installed():
        pytest.skip("环境装了 pypinyin，混合会转拼音")
    assert o._ascii_slug("Acme科技") == "acme"


def test_slug_empty_ext():
    assert o._ascii_slug("") == "ext"
    assert o._ascii_slug("   ") == "ext"
    assert o._ascii_slug(None) == "ext"


def test_slug_truncated_to_20():
    assert len(o._ascii_slug("a" * 50)) == 20


def test_slug_strips_non_alnum():
    assert o._ascii_slug("A.c-m_e 1!@#") == "acme1"


def test_slug_pinyin_branch_via_fake_module(monkeypatch):
    """注入假 pypinyin.lazy_pinyin → 中文走拼音分支。"""
    fake = types.ModuleType("pypinyin")
    fake.lazy_pinyin = lambda text: ["mou", "mou"]   # 忽略输入，返固定拼音
    monkeypatch.setitem(sys.modules, "pypinyin", fake)
    assert o._ascii_slug("某某") == "moumou"


def test_slug_pinyin_branch_mixed(monkeypatch):
    fake = types.ModuleType("pypinyin")
    fake.lazy_pinyin = lambda text: ["Acme", "ke", "ji"]
    monkeypatch.setitem(sys.modules, "pypinyin", fake)
    assert o._ascii_slug("Acme科技") == "acmekeji"


def _pypinyin_installed() -> bool:
    try:
        import pypinyin  # noqa
        return True
    except Exception:
        return False


# ── _derive_user_name ─────────────────────────────────────────────────────────

def test_derive_user_name_format():
    name = o._derive_user_name({"enterprise": "TEST"}, "inst_1")
    assert name.startswith("tempak-test-")
    # tempak-test-<6hex>
    tail = name.rsplit("-", 1)[-1]
    assert len(tail) == 6
    assert all(c in "0123456789abcdef" for c in tail)


def test_derive_user_name_deterministic_per_instance():
    a = o._derive_user_name({"enterprise": "Acme"}, "inst_x")
    b = o._derive_user_name({"enterprise": "Acme"}, "inst_x")
    assert a == b
    assert a != o._derive_user_name({"enterprise": "Acme"}, "inst_y")


def test_derive_user_name_chinese_enterprise_ext_slug():
    if _pypinyin_installed():
        pytest.skip("环境装了 pypinyin")
    name = o._derive_user_name({"enterprise": "某某科技"}, "inst_z")
    assert name.startswith("tempak-ext-")


def test_derive_user_name_only_ram_safe_chars():
    name = o._derive_user_name({"enterprise": "Acme 科技!"}, "inst_q")
    body = name[len("tempak-"):].rsplit("-", 1)[0]
    assert all(c.islower() or c.isdigit() for c in body)


# ── display_name_for ──────────────────────────────────────────────────────────

def test_display_name_with_enterprise_keeps_chinese():
    assert o.display_name_for({"enterprise": "某某科技"}) == "某某科技-临时外采用户"


def test_display_name_empty_enterprise():
    assert o.display_name_for({"enterprise": ""}) == "临时外采用户"
    assert o.display_name_for({}) == "临时外采用户"


def test_display_name_truncated_128():
    dn = o.display_name_for({"enterprise": "企" * 200})
    assert len(dn) <= 128


# ── issuer._issue_ram 建号 display_name ───────────────────────────────────────

class _FakeRamClient:
    """记录 create_user / policy / ak 调用；user 默认不存在 → 触发 create_user。"""
    def __init__(self):
        self.created = {}

    def get_user(self, req):
        from core.oss_perm.permsync import _err_code  # noqa
        e = RuntimeError("not exist")
        e.code = "EntityNotExist.User"
        raise e

    def create_user(self, req):
        self.created["user_name"] = req.user_name
        self.created["display_name"] = req.display_name

    def get_policy(self, req):
        e = RuntimeError("no policy")
        e.code = "EntityNotExist.Policy"
        raise e

    def create_policy(self, req):
        self.created["policy"] = req.policy_name

    def attach_policy_to_user(self, req):
        self.created["attached"] = req.policy_name

    def create_access_key(self, req):
        class _AK:
            access_key_id = "LTAI_NEW"
            access_key_secret = "SK"

        class _Body:
            access_key = _AK()

        class _Resp:
            body = _Body()
        return _Resp()


def test_issue_ram_passes_display_name(monkeypatch):
    import time
    fake = _FakeRamClient()
    monkeypatch.setattr(issuer, "permsync_client", lambda: fake)
    now = time.time()
    grant = {
        "grant_id": "tak-abc", "mode": "ram", "bucket": "b", "prefix": "p/",
        "caps": ["read"], "enterprise": "某某科技",
        "not_before": now, "expire": now + 5 * 86400,
        "user_name": "tempak-ext-abc123", "policy_name": "temp-ak-auto-tempak-ext-abc123",
        "source_ips": [], "reason": "",
    }
    creds = issuer._issue_ram(grant)
    # create_user 收到的 display_name = 企业可读名（中文 OK）
    assert fake.created["display_name"] == "某某科技-临时外采用户"
    assert fake.created["user_name"] == "tempak-ext-abc123"
    assert creds["access_key_id"] == "LTAI_NEW"
