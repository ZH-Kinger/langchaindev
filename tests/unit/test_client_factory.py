"""utils/aliyun_client_factory.py 单元测试：凭证优先级 + 工厂函数兜底。"""
import pytest


# ── _resolve_cred 三级优先级 ────────────────────────────────────────────────

def test_resolve_user_ak_takes_priority(monkeypatch, mock_aliyun_sts):
    """① 用户绑定 AK 存在 → 不调 STS，直接用用户 AK。"""
    from utils import aliyun_client_factory as factory
    monkeypatch.setattr(
        "utils.aliyun_user_creds.get_user_ak",
        lambda oid: ("USER_AK", "USER_SK") if oid == "ou_user_ak" else None,
    )
    sts_called = {"n": 0}
    def counted_sts(oid):
        sts_called["n"] += 1
        return mock_aliyun_sts
    monkeypatch.setattr("utils.aliyun_sts.assume_role_for_user", counted_sts)
    monkeypatch.setattr(
        "utils.aliyun_client_factory.assume_role_for_user", counted_sts
    )

    cred = factory._resolve_cred("ou_user_ak")
    assert cred == {"ak": "USER_AK", "sk": "USER_SK", "token": ""}
    assert sts_called["n"] == 0, "用户 AK 存在时不应调 STS"


def test_resolve_falls_back_to_sts(monkeypatch, mock_aliyun_sts):
    """② 无用户 AK + STS 成功 → 用 STS 临时凭证。"""
    from utils import aliyun_client_factory as factory
    monkeypatch.setattr("utils.aliyun_user_creds.get_user_ak", lambda oid: None)
    monkeypatch.setattr(
        "utils.aliyun_client_factory.assume_role_for_user",
        lambda oid: mock_aliyun_sts,
    )

    cred = factory._resolve_cred("ou_no_ak")
    assert cred["ak"] == mock_aliyun_sts["access_key_id"]
    assert cred["sk"] == mock_aliyun_sts["access_key_secret"]
    assert cred["token"] == mock_aliyun_sts["security_token"]


def test_resolve_falls_back_to_global(monkeypatch):
    """③ 无用户 AK + STS 失败 + 全局有 → 用全局。"""
    from utils import aliyun_client_factory as factory
    from config.settings import settings

    monkeypatch.setattr("utils.aliyun_user_creds.get_user_ak", lambda oid: None)
    monkeypatch.setattr("utils.aliyun_client_factory.assume_role_for_user", lambda oid: None)
    monkeypatch.setattr(settings, "PAI_DSW_ACCESS_KEY_ID", "GLOBAL_AK")
    monkeypatch.setattr(settings, "PAI_DSW_ACCESS_KEY_SECRET", "GLOBAL_SK")

    cred = factory._resolve_cred("ou_no_ak")
    assert cred == {"ak": "GLOBAL_AK", "sk": "GLOBAL_SK", "token": ""}


def test_resolve_returns_none_when_all_missing(monkeypatch):
    """④ 用户 AK + STS + 全局都不可用 → None。"""
    from utils import aliyun_client_factory as factory
    from config.settings import settings

    monkeypatch.setattr("utils.aliyun_user_creds.get_user_ak", lambda oid: None)
    monkeypatch.setattr("utils.aliyun_client_factory.assume_role_for_user", lambda oid: None)
    monkeypatch.setattr(settings, "PAI_DSW_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "PAI_DSW_ACCESS_KEY_SECRET", "")

    assert factory._resolve_cred("ou_x") is None


def test_resolve_without_open_id_uses_global(monkeypatch):
    """open_id 为空（如调度器后台任务）→ 跳过用户 AK 和 STS，直接全局。"""
    from utils import aliyun_client_factory as factory
    from config.settings import settings
    monkeypatch.setattr(settings, "PAI_DSW_ACCESS_KEY_ID", "GLOBAL_AK")
    monkeypatch.setattr(settings, "PAI_DSW_ACCESS_KEY_SECRET", "GLOBAL_SK")

    cred = factory._resolve_cred("")
    assert cred == {"ak": "GLOBAL_AK", "sk": "GLOBAL_SK", "token": ""}


# ── get_pai_dsw_client：cred 缺失时返回 None ─────────────────────────────────

def test_get_pai_dsw_client_returns_none_when_no_cred(monkeypatch):
    from utils import aliyun_client_factory as factory
    monkeypatch.setattr(factory, "_resolve_cred", lambda oid: None)
    assert factory.get_pai_dsw_client("ou_x") is None


# ── _resolve_cred 用户 AK 读取异常时仍走 STS ─────────────────────────────────

def test_user_ak_exception_falls_back_to_sts(monkeypatch, mock_aliyun_sts):
    from utils import aliyun_client_factory as factory
    def raise_err(oid):
        raise RuntimeError("Redis down")
    monkeypatch.setattr("utils.aliyun_user_creds.get_user_ak", raise_err)
    monkeypatch.setattr(
        "utils.aliyun_client_factory.assume_role_for_user",
        lambda oid: mock_aliyun_sts,
    )

    cred = factory._resolve_cred("ou_buggy")
    assert cred["ak"] == mock_aliyun_sts["access_key_id"]