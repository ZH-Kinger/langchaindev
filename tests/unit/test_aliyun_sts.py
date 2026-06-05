"""utils/aliyun_sts.py 单元测试。"""
import json
import time
import pytest


def _reload_sts_with_mapping(mapping_json: str, default_arn: str = "acs:ram::123:role/Default"):
    """重新加载 aliyun_sts 模块以让 _ROLE_MAPPING 重新解析。"""
    from config.settings import settings
    settings.ALIYUN_BOT_ROLE_MAPPING_RAW = mapping_json
    settings.ALIYUN_BOT_ROLE_DEFAULT = default_arn

    import importlib
    from utils import aliyun_sts
    importlib.reload(aliyun_sts)
    return aliyun_sts


# ── 角色 ARN 选择 ────────────────────────────────────────────────────────────

def test_role_arn_no_mapping_returns_default(monkeypatch, mock_ram_api):
    sts = _reload_sts_with_mapping("[]", "acs:ram::123:role/Default")
    assert sts._role_arn_for_user("ou_any") == "acs:ram::123:role/Default"


def test_role_arn_group_match(monkeypatch, mock_ram_api):
    """用户在 algo-team 组 → 选 algo 角色。"""
    mapping = json.dumps([
        {"group": "algo-team",     "role": "acs:ram::123:role/Algo"},
        {"group": "platform-team", "role": "acs:ram::123:role/Platform"},
    ])
    sts = _reload_sts_with_mapping(mapping)

    # mock 飞书↔RAM 映射 + 用户组
    from tools.aliyun import ram as ram_module
    monkeypatch.setattr(ram_module, "get_ram_user_by_open_id",
                        lambda oid: {"user_name": "wzh.wang"} if oid == "ou_wzh" else None)
    mock_ram_api["groups"]["wzh.wang"] = ["algo-team"]

    # 同时需要让 sts 模块里的 _list_user_groups 用上 mock
    monkeypatch.setattr(sts, "_list_user_groups",
                        lambda name: mock_ram_api["groups"].get(name, []))

    assert sts._role_arn_for_user("ou_wzh") == "acs:ram::123:role/Algo"


def test_role_arn_no_match_returns_default(monkeypatch, mock_ram_api):
    mapping = json.dumps([{"group": "algo-team", "role": "acs:ram::123:role/Algo"}])
    sts = _reload_sts_with_mapping(mapping, "acs:ram::123:role/Default")

    from tools.aliyun import ram as ram_module
    monkeypatch.setattr(ram_module, "get_ram_user_by_open_id",
                        lambda oid: {"user_name": "stranger"})
    monkeypatch.setattr(sts, "_list_user_groups", lambda name: [])  # 不在任何组

    assert sts._role_arn_for_user("ou_stranger") == "acs:ram::123:role/Default"


def test_role_arn_list_groups_exception_returns_default(monkeypatch, mock_ram_api):
    mapping = json.dumps([{"group": "g", "role": "acs:ram::123:role/G"}])
    sts = _reload_sts_with_mapping(mapping, "acs:ram::123:role/Default")

    from tools.aliyun import ram as ram_module
    monkeypatch.setattr(ram_module, "get_ram_user_by_open_id",
                        lambda oid: {"user_name": "u"})
    monkeypatch.setattr(sts, "_list_user_groups",
                        lambda name: (_ for _ in ()).throw(Exception("RAM API down")))

    assert sts._role_arn_for_user("ou_u") == "acs:ram::123:role/Default"


# ── AssumeRole 缓存 ──────────────────────────────────────────────────────────

def test_assume_role_uses_cache(monkeypatch, mock_aliyun_sts):
    """同一 open_id 第二次 assume 应直接命中 Redis 缓存，不再调 _do_assume_role。"""
    sts = _reload_sts_with_mapping("[]", "acs:ram::123:role/Default")
    # 重新 patch（reload 后 fixture 失效）
    monkeypatch.setattr(sts, "_do_assume_role", lambda *a, **k: dict(mock_aliyun_sts))

    call_count = {"n": 0}
    real_do = sts._do_assume_role
    def counted(*a, **k):
        call_count["n"] += 1
        return real_do(*a, **k)
    monkeypatch.setattr(sts, "_do_assume_role", counted)

    cred1 = sts.assume_role_for_user("ou_cache_test")
    cred2 = sts.assume_role_for_user("ou_cache_test")
    assert cred1 == cred2
    assert call_count["n"] == 1, "第二次应命中缓存，不再调 AssumeRole"


def test_assume_role_refresh_when_near_expiry(monkeypatch):
    """缓存的凭证若 < 5min 到期，应强制刷新。"""
    sts = _reload_sts_with_mapping("[]", "acs:ram::123:role/Default")
    near_expire = {
        "access_key_id":     "STS.OLD",
        "access_key_secret": "OLD",
        "security_token":    "OLD",
        "expire_ts":         time.time() + 60,   # 1 分钟后到期 → 强制刷新
        "role_arn":          "acs:ram::123:role/Default",
    }
    # 写一个即将过期的缓存
    from utils.redis_client import get_redis
    get_redis().setex(
        sts._cache_key("ou_refresh", near_expire["role_arn"]),
        300, json.dumps(near_expire),
    )

    fresh = {
        "access_key_id":     "STS.NEW",
        "access_key_secret": "NEW",
        "security_token":    "NEW",
        "expire_ts":         time.time() + 3600,
        "role_arn":          "acs:ram::123:role/Default",
    }
    monkeypatch.setattr(sts, "_do_assume_role", lambda *a, **k: fresh)

    result = sts.assume_role_for_user("ou_refresh")
    assert result["access_key_id"] == "STS.NEW", "应触发刷新"


def test_invalidate_cache_removes_all_user_entries(monkeypatch, mock_aliyun_sts):
    sts = _reload_sts_with_mapping("[]", "acs:ram::123:role/Default")
    monkeypatch.setattr(sts, "_do_assume_role", lambda *a, **k: dict(mock_aliyun_sts))

    sts.assume_role_for_user("ou_invalidate")
    from utils.redis_client import get_redis
    keys_before = list(get_redis().scan_iter("aliyun:sts:ou_invalidate:*"))
    assert len(keys_before) == 1

    sts.invalidate_cache("ou_invalidate")
    keys_after = list(get_redis().scan_iter("aliyun:sts:ou_invalidate:*"))
    assert keys_after == []
