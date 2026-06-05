"""utils/crypto.py 单元测试。"""
import pytest
from utils.crypto import encrypt, decrypt, _fernet


def test_encrypt_decrypt_roundtrip():
    plaintext = "hello-aliyun-AK-LTAI5tXXXXXX"
    ct = encrypt(plaintext)
    assert ct != plaintext, "密文不应等于明文"
    assert ct.startswith("gAAAAA"), "Fernet 密文应以 gAAAAA 开头"
    assert decrypt(ct) == plaintext


def test_encrypt_empty_string():
    assert encrypt("") == ""
    assert decrypt("") == ""


def test_decrypt_legacy_plaintext():
    """旧数据没加密、不是 fernet 格式时，应原样返回（兼容老数据）。"""
    legacy = "LTAI5tLegacyPlainAK"
    assert decrypt(legacy) == legacy


def test_decrypt_tampered_ciphertext():
    """篡改中段会破坏 HMAC，应返回空串。"""
    ct = encrypt("foo")
    # 改中段字符（非头部，避开 Fernet 固定 magic byte）
    mid = len(ct) // 2
    tampered = ct[:mid] + ("X" if ct[mid] != "X" else "Y") + ct[mid + 1:]
    result = decrypt(tampered)
    assert result == "", f"被篡改的 Fernet 密文应解密失败返回空串，实际：{result!r}"


def test_no_key_falls_back_to_passthrough(monkeypatch):
    """无 BOT_CREDS_ENCRYPTION_KEY 时降级为明文（带告警，但不抛异常）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "BOT_CREDS_ENCRYPTION_KEY", "")
    _fernet.cache_clear()

    plain = "no-key-mode"
    ct = encrypt(plain)
    assert ct == plain, "无 key 时应原样返回"
    assert decrypt(ct) == plain


def test_is_key_configured_true():
    """已配置 key 时返回 True（依赖 conftest 的 fernet_key fixture）。"""
    from utils.crypto import is_key_configured
    assert is_key_configured() is True


def test_is_key_configured_false(monkeypatch):
    """未配置 key 时返回 False。"""
    from config.settings import settings
    from utils.crypto import is_key_configured, _fernet
    monkeypatch.setattr(settings, "BOT_CREDS_ENCRYPTION_KEY", "")
    _fernet.cache_clear()
    assert is_key_configured() is False


def test_encrypt_strict_raises_when_no_key(monkeypatch):
    """encrypt_strict 在缺 key 时必须抛 CryptoNotConfigured（不降级）。"""
    from config.settings import settings
    from utils.crypto import encrypt_strict, CryptoNotConfigured, _fernet
    monkeypatch.setattr(settings, "BOT_CREDS_ENCRYPTION_KEY", "")
    _fernet.cache_clear()

    with pytest.raises(CryptoNotConfigured):
        encrypt_strict("sensitive-ak")


def test_encrypt_strict_works_when_key_configured():
    """配置 key 时 encrypt_strict 行为与 encrypt 一致。"""
    from utils.crypto import encrypt_strict, decrypt
    ct = encrypt_strict("LTAI5tXXX")
    assert ct.startswith("gAAAAA")
    assert decrypt(ct) == "LTAI5tXXX"
