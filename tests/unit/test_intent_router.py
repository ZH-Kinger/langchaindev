"""core/intent_router.py 单元测试。

LLM 调用全部 mock，不打真实网络。
"""
import pytest


# ── _parse_intents：JSON 解析容错 ────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    # 标准 JSON
    ('["advisor"]',                    ["advisor"]),
    ('["cluster", "monitor"]',         ["cluster", "monitor"]),
    # 大小写归一
    ('["ADVISOR"]',                    ["advisor"]),
    # markdown 代码块包裹
    ('```json\n["training"]\n```',     ["training"]),
    ('```\n["ecs"]\n```',              ["ecs"]),
    # 带前后说明文字
    ('结果：["oss"]',                   ["oss"]),
    ('好的，我选 ["sls", "monitor"] 这两个。',  ["sls", "monitor"]),
    # 空数组合法
    ('[]',                             []),
    # 完全垃圾
    ('我不知道',                        []),
    ('',                               []),
    # 非数组的 JSON
    ('{"intent": "advisor"}',          []),
    # 含非字符串元素
    ('["advisor", 123, null]',         ["advisor"]),
])
def test_parse_intents(raw, expected):
    from core.intent_router import _parse_intents
    assert _parse_intents(raw) == expected


# ── _filter_known_intents：过滤未知意图 ─────────────────────────────────────

def test_filter_drops_unknown():
    from core.intent_router import _filter_known_intents
    result = _filter_known_intents(["advisor", "made_up_xyz", "cluster"])
    assert result == ["advisor", "cluster"]


def test_filter_dedupes_preserving_order():
    from core.intent_router import _filter_known_intents
    result = _filter_known_intents(["monitor", "monitor", "ops"])
    assert result == ["monitor", "ops"]


def test_filter_caps_at_two():
    from core.intent_router import _filter_known_intents
    result = _filter_known_intents(["advisor", "cluster", "monitor", "training"])
    assert len(result) == 2
    assert result == ["advisor", "cluster"]


def test_filter_all_unknown_returns_empty():
    from core.intent_router import _filter_known_intents
    assert _filter_known_intents(["foo", "bar"]) == []


# ── _build_prompt：必须包含所有意图描述 ──────────────────────────────────────

def test_prompt_contains_every_intent():
    from core.intent_router import _build_prompt, INTENT_DESCRIPTIONS
    prompt = _build_prompt("帮我看看集群")
    for intent in INTENT_DESCRIPTIONS:
        assert intent in prompt, f"prompt 缺少意图 {intent}"
    assert "帮我看看集群" in prompt
    assert "JSON" in prompt


# ── route()：端到端，mock LLM ────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, output):
        self.output = output
        self.invoke_count = 0

    def invoke(self, prompt):
        self.invoke_count += 1
        return _FakeResp(self.output)


def _patch_llm(monkeypatch, output):
    """让 get_cloud_llm 返回固定输出的假 LLM。"""
    fake = _FakeLLM(output)
    monkeypatch.setattr("core.llm_factory.get_cloud_llm", lambda **kw: fake)
    # 同时清缓存
    from core.intent_router import clear_cache
    clear_cache()
    return fake


def test_route_returns_llm_intents(monkeypatch):
    _patch_llm(monkeypatch, '["advisor", "cluster"]')
    from core.intent_router import route
    assert route("帮我优化一下整个集群") == ["advisor", "cluster"]


def test_route_filters_unknown(monkeypatch):
    _patch_llm(monkeypatch, '["advisor", "made_up_intent"]')
    from core.intent_router import route
    assert route("xxx") == ["advisor"]


def test_route_handles_markdown_wrapped(monkeypatch):
    _patch_llm(monkeypatch, '```json\n["training"]\n```')
    from core.intent_router import route
    assert route("帮我分析训练") == ["training"]


def test_route_empty_string():
    from core.intent_router import route
    assert route("") == []
    assert route("   ") == []


def test_route_uses_cache(monkeypatch):
    fake = _patch_llm(monkeypatch, '["monitor"]')
    from core.intent_router import route
    route("查 cpu 趋势")
    route("查 cpu 趋势")
    route("查 cpu 趋势")
    assert fake.invoke_count == 1, "同输入应命中缓存"


def test_route_llm_exception_returns_empty(monkeypatch):
    """LLM 异常时静默返回空列表，让上层降级。"""
    def raise_err(**kw):
        raise RuntimeError("Network down")
    monkeypatch.setattr("core.llm_factory.get_cloud_llm", raise_err)
    from core.intent_router import clear_cache, route
    clear_cache()
    assert route("查 oss bucket") == []


def test_route_llm_returns_non_array(monkeypatch):
    _patch_llm(monkeypatch, '{"intent": "advisor"}')
    from core.intent_router import route
    assert route("xxx") == []


def test_route_llm_returns_empty_array(monkeypatch):
    _patch_llm(monkeypatch, '[]')
    from core.intent_router import route
    assert route("没意图") == []
