"""
Redis 客户端单例
统一管理连接，供对话记忆、结果缓存、告警去重三个功能复用。
"""
import json
import hashlib
import redis
from functools import wraps
from config.settings import settings

# ── 连接单例 ──────────────────────────────────────────────────────────────────
_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            socket_connect_timeout=5,
            decode_responses=True,
        )
    return _client


def is_redis_available() -> bool:
    try:
        return get_redis().ping()
    except Exception:
        return False


# ── B. 分析结果缓存 ───────────────────────────────────────────────────────────
CACHE_TTL = 300   # 5 分钟


def cache_analysis(func):
    """
    装饰器：对 smart_data_analysis(file_name) 的结果做 Redis 缓存。
    缓存 key = analysis:{file_name}:{mtime}  — 文件修改后自动失效。
    """
    import os

    @wraps(func)
    def wrapper(file_name: str) -> str:
        try:
            r = get_redis()
            file_path = os.path.join(settings.DATA_PATH, file_name)
            mtime = int(os.path.getmtime(file_path)) if os.path.exists(file_path) else 0
            cache_key = f"analysis:{file_name}:{mtime}"

            cached = r.get(cache_key)
            if cached:
                return f"[缓存] {cached}"

            result = func(file_name)

            r.setex(cache_key, CACHE_TTL, result)
            return result
        except Exception:
            # Redis 不可用时降级为直接执行
            return func(file_name)

    return wrapper


# ── C. 告警去重状态 ───────────────────────────────────────────────────────────
DEDUP_TTL = 86400   # 去重集合保留 24 小时


def get_dedup_set_key(file_name: str) -> str:
    return f"alarms:dedup:{file_name}"


def load_seen_alarms(file_name: str) -> set:
    """从 Redis 读取已处理的 node:message 集合"""
    try:
        r = get_redis()
        return r.smembers(get_dedup_set_key(file_name))
    except Exception:
        return set()


def save_seen_alarms(file_name: str, alarm_keys: set) -> None:
    """将新处理的告警写入 Redis set"""
    try:
        r = get_redis()
        key = get_dedup_set_key(file_name)
        if alarm_keys:
            r.sadd(key, *alarm_keys)
            r.expire(key, DEDUP_TTL)
    except Exception:
        pass


def count_seen_alarms(file_name: str) -> int:
    """返回已去重的告警数量"""
    try:
        return get_redis().scard(get_dedup_set_key(file_name))
    except Exception:
        return 0