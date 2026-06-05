"""
用户阿里云 AK/SK 加密存取（Redis）。

存储结构（Redis Hash）：
  Key:   feishu:user_creds:{open_id}
  Field: ak_id_enc      → Fernet 密文
         ak_secret_enc  → Fernet 密文
         created_ts     → 首次绑定时间戳
         last_used_ts   → 最近一次使用（决定闲置清理）

TTL：USER_AK_IDLE_TTL_SECONDS（默认 30 天），每次使用刷新。
"""
import json
import time
from typing import Optional

from config.settings import settings
from utils.crypto import encrypt_strict, decrypt, CryptoNotConfigured
from utils.logger import get_logger

logger = get_logger(__name__)


_PREFIX = "feishu:user_creds:"


def _key(open_id: str) -> str:
    return f"{_PREFIX}{open_id}"


# ── 写入 ──────────────────────────────────────────────────────────────────────

def save_user_ak(open_id: str, ak_id: str, ak_secret: str) -> bool:
    """加密存入用户 AK/SK。返回 True 表示成功。

    BOT_CREDS_ENCRYPTION_KEY 未配置时**拒绝写入**（返回 False），
    防止明文 AK/SK 流入 Redis。
    """
    if not open_id or not ak_id or not ak_secret:
        return False
    try:
        from utils.redis_client import get_redis
        # encrypt_strict 在缺 key 时抛 CryptoNotConfigured，本函数捕获后拒绝写入
        ak_id_enc     = encrypt_strict(ak_id)
        ak_secret_enc = encrypt_strict(ak_secret)
        r = get_redis()
        data = {
            "ak_id_enc":     ak_id_enc,
            "ak_secret_enc": ak_secret_enc,
            "created_ts":    str(int(time.time())),
            "last_used_ts":  str(int(time.time())),
        }
        r.hset(_key(open_id), mapping=data)
        r.expire(_key(open_id), settings.USER_AK_IDLE_TTL_SECONDS)
        logger.info("[UserCreds] 绑定 open_id=%s ak=%s****", open_id, ak_id[:8])
        return True
    except CryptoNotConfigured as e:
        logger.error("[UserCreds] 拒绝写入：%s", e)
        return False
    except Exception as e:
        logger.error("[UserCreds] 保存失败 open_id=%s err=%s", open_id, e)
        return False


# ── 读取 ──────────────────────────────────────────────────────────────────────

def get_user_ak(open_id: str) -> Optional[tuple[str, str]]:
    """
    返回 (ak_id, ak_secret)，未绑定或解密失败返回 None。
    成功时刷新 last_used_ts 和 TTL。
    """
    if not open_id:
        return None
    try:
        from utils.redis_client import get_redis
        r = get_redis()
        h = r.hgetall(_key(open_id))
        if not h:
            return None
        ak_id     = decrypt(h.get("ak_id_enc", "") or "")
        ak_secret = decrypt(h.get("ak_secret_enc", "") or "")
        if not ak_id or not ak_secret:
            return None
        # 刷新 last_used 和 TTL
        r.hset(_key(open_id), "last_used_ts", str(int(time.time())))
        r.expire(_key(open_id), settings.USER_AK_IDLE_TTL_SECONDS)
        return ak_id, ak_secret
    except Exception as e:
        logger.warning("[UserCreds] 读取失败 open_id=%s err=%s", open_id, e)
        return None


def has_user_ak(open_id: str) -> bool:
    """快速检查用户是否已绑定 AK（不解密，不刷新 TTL）。"""
    if not open_id:
        return False
    try:
        from utils.redis_client import get_redis
        return bool(get_redis().exists(_key(open_id)))
    except Exception:
        return False


# ── 删除 ──────────────────────────────────────────────────────────────────────

def delete_user_ak(open_id: str) -> bool:
    """解绑用户 AK。返回 True 表示有数据被删除。"""
    if not open_id:
        return False
    try:
        from utils.redis_client import get_redis
        deleted = get_redis().delete(_key(open_id))
        if deleted:
            logger.info("[UserCreds] 解绑 open_id=%s", open_id)
        return bool(deleted)
    except Exception as e:
        logger.error("[UserCreds] 删除失败 open_id=%s err=%s", open_id, e)
        return False


# ── 元信息查询（用于管理面板）─────────────────────────────────────────────────

def get_user_ak_meta(open_id: str) -> Optional[dict]:
    """返回绑定时间和最近使用时间（不解密 AK 本身）。"""
    if not open_id:
        return None
    try:
        from utils.redis_client import get_redis
        h = get_redis().hgetall(_key(open_id))
        if not h:
            return None
        return {
            "created_ts":   int(h.get("created_ts",   "0") or 0),
            "last_used_ts": int(h.get("last_used_ts", "0") or 0),
            "ak_id_masked": (decrypt(h.get("ak_id_enc", "") or "")[:8] + "****"),
        }
    except Exception:
        return None
