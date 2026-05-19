"""utils/aliyun_user_creds.py 单元测试。"""
import time
from utils.aliyun_user_creds import (
    save_user_ak, get_user_ak, has_user_ak,
    delete_user_ak, get_user_ak_meta,
)


OPEN_ID = "ou_test_user_001"
AK_ID   = "LTAI5tTEST_AK_ID_XXXXX"
AK_SK   = "TEST_AK_SECRET_YYYYYYYYYYY"


def test_save_and_get_roundtrip():
    assert save_user_ak(OPEN_ID, AK_ID, AK_SK) is True
    result = get_user_ak(OPEN_ID)
    assert result == (AK_ID, AK_SK)


def test_save_empty_fields_fails():
    assert save_user_ak("", AK_ID, AK_SK) is False
    assert save_user_ak(OPEN_ID, "", AK_SK) is False
    assert save_user_ak(OPEN_ID, AK_ID, "") is False


def test_has_user_ak():
    assert has_user_ak(OPEN_ID) is False
    save_user_ak(OPEN_ID, AK_ID, AK_SK)
    assert has_user_ak(OPEN_ID) is True


def test_get_unknown_returns_none():
    assert get_user_ak("ou_does_not_exist") is None


def test_delete_user_ak():
    save_user_ak(OPEN_ID, AK_ID, AK_SK)
    assert delete_user_ak(OPEN_ID) is True
    assert get_user_ak(OPEN_ID) is None
    # 二次删除返回 False
    assert delete_user_ak(OPEN_ID) is False


def test_meta_contains_timestamps_and_masked_ak():
    save_user_ak(OPEN_ID, AK_ID, AK_SK)
    meta = get_user_ak_meta(OPEN_ID)
    assert meta is not None
    assert meta["created_ts"] > 0
    assert meta["last_used_ts"] > 0
    # 8 位前缀 + ****
    assert meta["ak_id_masked"] == AK_ID[:8] + "****"
    # 密钥本身不应出现在 meta 中
    assert AK_SK not in str(meta)


def test_get_refreshes_last_used_ts(fake_redis):
    save_user_ak(OPEN_ID, AK_ID, AK_SK)
    meta1 = get_user_ak_meta(OPEN_ID)
    time.sleep(1.1)
    get_user_ak(OPEN_ID)   # 触发 last_used 更新
    meta2 = get_user_ak_meta(OPEN_ID)
    assert meta2["last_used_ts"] > meta1["last_used_ts"]


def test_data_actually_encrypted_in_redis(fake_redis):
    """直接看 Redis 里存的内容，应该是密文不是明文。"""
    save_user_ak(OPEN_ID, AK_ID, AK_SK)
    raw = fake_redis.hgetall(f"feishu:user_creds:{OPEN_ID}")
    # 明文 AK/SK 不应出现在任何字段值里
    for v in raw.values():
        assert AK_ID not in v, f"明文 AK 泄露在 Redis：{v}"
        assert AK_SK not in v, f"明文 SK 泄露在 Redis：{v}"
    # 密文应是 Fernet 格式
    assert raw["ak_id_enc"].startswith("gAAAAA")
    assert raw["ak_secret_enc"].startswith("gAAAAA")


def test_save_refuses_when_no_encryption_key(monkeypatch, fake_redis):
    """BOT_CREDS_ENCRYPTION_KEY 未配置时，save_user_ak 必须拒绝写入。"""
    from config.settings import settings
    from utils.crypto import _fernet
    monkeypatch.setattr(settings, "BOT_CREDS_ENCRYPTION_KEY", "")
    _fernet.cache_clear()

    ok = save_user_ak(OPEN_ID, AK_ID, AK_SK)
    assert ok is False, "缺加密 key 时应拒绝写入"
    # Redis 里不应有任何记录
    raw = fake_redis.hgetall(f"feishu:user_creds:{OPEN_ID}")
    assert raw == {}, "拒绝写入后 Redis 不应有数据"
