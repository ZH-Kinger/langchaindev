"""#38-4 会话历史 TTL：`core.agent._save_turn` 每轮 `expire(key, 7天)` 续期。

目标：Redis 里的会话历史空闲 7 天自动过期，不再无限堆积；且截断行为
（MAX_HISTORY）不受影响。用 conftest 的 fakeredis（autouse）跑。
"""
import json

import core.agent as agent

SID = "ou_ttl_test"


def _key():
    return agent._session_key(SID)


def test_save_turn_sets_ttl(fake_redis):
    agent._save_turn(SID, "问题一", "答复一")
    ttl = fake_redis.ttl(_key())
    # ≈ 7 天（604800s），留 5s 余量给执行耗时
    assert agent.HISTORY_TTL_SECONDS - 5 <= ttl <= agent.HISTORY_TTL_SECONDS
    assert ttl > 0


def test_ttl_is_renewed_each_turn(fake_redis):
    """再存一轮会把 TTL 续回 7 天：先人为把 TTL 压到 100s，存一轮后应弹回。"""
    agent._save_turn(SID, "q1", "a1")
    fake_redis.expire(_key(), 100)
    assert fake_redis.ttl(_key()) <= 100

    agent._save_turn(SID, "q2", "a2")
    ttl = fake_redis.ttl(_key())
    assert ttl > 100
    assert agent.HISTORY_TTL_SECONDS - 5 <= ttl <= agent.HISTORY_TTL_SECONDS


def test_truncation_unchanged(fake_redis, monkeypatch):
    """截断行为不变：存 15 轮（30 条）后，列表被裁到 MAX_HISTORY 条，且保留最新的。

    #43 二轮起，累计到 FOLD_TRIGGER(30) 才折叠一次，压缩走 `threading.Thread` 后台跑。
    此处桩掉 `get_cloud_llm`（避免真调 LLM）+ 把 `threading.Thread` 换成同步假线程
    （避免真起后台线程在 monkeypatch teardown 后仍运行导致 flaky/走网络）。
    """
    import types as _types
    monkeypatch.setattr(agent, "get_cloud_llm",
                        lambda *a, **k: type("L", (), {"invoke": lambda self, p: type("R", (), {"content": "s"})()})())

    class _SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._t, self._a, self._k = target, args, kwargs or {}

        def start(self):
            if self._t is not None:
                self._t(*self._a, **self._k)
    monkeypatch.setattr(agent, "threading", _types.SimpleNamespace(Thread=_SyncThread))
    for i in range(15):
        agent._save_turn(SID, f"q{i}", f"a{i}")

    items = fake_redis.lrange(_key(), 0, -1)
    assert len(items) == agent.MAX_HISTORY   # 20

    # 最新一轮的 human/ai 应仍在（在列表尾部）
    last_two = [json.loads(x) for x in items[-2:]]
    assert last_two[0] == {"role": "human", "content": "q14"}
    assert last_two[1] == {"role": "ai", "content": "a14"}

    # 截断后仍带 TTL（截断不该抹掉过期时间）
    assert fake_redis.ttl(_key()) > 0


def test_save_turn_noop_when_redis_down(monkeypatch, fake_redis):
    """Redis 不可用时静默跳过，不写入、不抛异常。"""
    monkeypatch.setattr(agent, "is_redis_available", lambda: False)
    agent._save_turn(SID, "q", "a")
    assert fake_redis.exists(_key()) == 0
