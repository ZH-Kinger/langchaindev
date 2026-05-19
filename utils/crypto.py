"""
Fernet 对称加密封装。

用于敏感字段（用户 AK/SK 等）写入 Redis 前的加密。
加密 key 从 settings.BOT_CREDS_ENCRYPTION_KEY 读取。

安全策略：
  - encrypt_strict(): 缺 key 抛 CryptoNotConfigured，用于写入敏感字段的路径
  - encrypt():        缺 key 降级为明文 + 告警（保留，兼容非敏感场景）
  - decrypt():        缺 key 透传，兼容旧明文数据，确保历史数据不丢失
"""
from functools import lru_cache
from typing import Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CryptoNotConfigured(RuntimeError):
    """BOT_CREDS_ENCRYPTION_KEY 未配置或格式错误时抛出。"""
    pass


_WARN_ONCE = {"shown": False}


def _warn_no_key_once() -> None:
    if not _WARN_ONCE["shown"]:
        logger.warning(
            "[Crypto] BOT_CREDS_ENCRYPTION_KEY 未配置！敏感字段将以明文存入 Redis。"
            "生产环境请按 .env.example 注释生成 Fernet key 并填入。"
        )
        _WARN_ONCE["shown"] = True


@lru_cache(maxsize=1)
def _fernet() -> Optional[object]:
    key = (settings.BOT_CREDS_ENCRYPTION_KEY or "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as e:
        logger.error("[Crypto] Fernet 初始化失败（key 格式错误？）: %s", e)
        return None


def is_key_configured() -> bool:
    """检查加密 key 是否已正确配置（格式合法）。供启动校验和写入路径使用。"""
    return _fernet() is not None


def encrypt_strict(plaintext: str) -> str:
    """加密字符串。**未配置 key 时直接抛 CryptoNotConfigured**（不降级）。

    用于必须加密的敏感字段写入路径（如用户 AK/SK）。
    """
    if not plaintext:
        return ""
    f = _fernet()
    if f is None:
        raise CryptoNotConfigured(
            "BOT_CREDS_ENCRYPTION_KEY 未配置或格式错误。\n"
            "生成命令：python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            "把输出填到 .env 的 BOT_CREDS_ENCRYPTION_KEY= 行后即可。"
        )
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def encrypt(plaintext: str) -> str:
    """加密字符串。无 key 时降级为明文 + 告警（宽松模式，保留向后兼容）。

    敏感字段（AK/SK 等）请用 encrypt_strict()。
    """
    if not plaintext:
        return ""
    f = _fernet()
    if f is None:
        _warn_no_key_once()
        return plaintext
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str) -> str:
    """
    解密。无 key 时返回原文（兼容旧明文数据）。
    解密失败返回空字符串（旧 key 切换 / 篡改）。
    """
    if not ciphertext:
        return ""
    f = _fernet()
    if f is None:
        _warn_no_key_once()
        return ciphertext
    try:
        return f.decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except Exception as e:
        logger.warning("[Crypto] 解密失败（可能是旧明文或 key 已变更）: %s", e)
        # 兼容旧明文：若密文看起来不是 fernet 格式（不以 gAAAA 开头），尝试当明文返回
        if not ciphertext.startswith("gAAAA"):
            return ciphertext
        return ""
