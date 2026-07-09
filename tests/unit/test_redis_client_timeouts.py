"""#46 C4：get_redis() 构造 redis.Redis 时带 socket_timeout / health_check_interval /
retry_on_timeout（读超时兜底，Redis 卡住时请求/轮询线程不再无限阻塞）。

conftest 的 autouse fake_redis 会把 redis_client.get_redis 换成返回 fakeredis 的 lambda，
故在模块导入期（fixture 之前）抓住真正的 get_redis 原函数来测构造参数。
"""
from utils import redis_client as _rc

# 收集期抓原函数（此时 autouse fixture 尚未 patch）
_ORIG_GET_REDIS = _rc.get_redis


def test_get_redis_passes_read_timeout_kwargs(monkeypatch):
    captured = {}

    class _FakeRedis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # 让原 get_redis 走到构造分支：清单例 + 换掉 redis.Redis
    monkeypatch.setattr(_rc, "_client", None)
    monkeypatch.setattr(_rc.redis, "Redis", _FakeRedis)

    client = _ORIG_GET_REDIS()
    assert isinstance(client, _FakeRedis)
    assert captured["socket_timeout"] == 5
    assert captured["health_check_interval"] == 30
    assert captured["retry_on_timeout"] is True
    # 既有参数不回归
    assert captured["socket_connect_timeout"] == 5
    assert captured["decode_responses"] is True


def test_fakeredis_still_compatible_with_new_kwargs():
    """新增 kwargs 与实际 redis-py/ fakeredis 兼容——conftest 的 fakeredis 正常可用。"""
    import fakeredis
    # 注：retry_on_timeout 在 redis-py 6.0+ 已 deprecated（7.4 仍功能可用、只 warn），
    # 故此处只验非 deprecated 的两个 kwarg 兼容；传参本身由 test_get_redis_passes_read_timeout_kwargs 断言。
    fake = fakeredis.FakeRedis(
        decode_responses=True, socket_timeout=5, health_check_interval=30,
    )
    fake.set("k", "v")
    assert fake.get("k") == "v"
