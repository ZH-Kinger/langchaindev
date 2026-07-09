"""#38-2/3/4 AgentExecutor 兜底上限 + 常量存在性。

- `_build_executor` / `_get_streaming_executor` 构造出的 AgentExecutor 必须带上
  `verbose=False` + `max_iterations=8` + `max_execution_time=60`，防 LLM 反复调
  工具把用户挂住/烧钱、且不再把推理过程打进飞书日志。
- 常量 `AGENT_MAX_ITERATIONS / AGENT_MAX_EXECUTION_TIME / HISTORY_TTL_SECONDS`
  存在且取值符合预期，并被 executor 真实引用（改常量即改行为）。

LLM/agent 构造全部打桩，不触网。
"""
import pytest

import core.agent as agent


@pytest.fixture
def stub_agent_build(monkeypatch):
    """打桩 LLM、prompt、create_tool_calling_agent，避免真实 LLM 初始化/联网。
    create_tool_calling_agent 返回一个真正的 Runnable，让 AgentExecutor 的
    pydantic 校验通过。"""
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(agent, "get_cloud_llm", lambda *a, **k: object())
    monkeypatch.setattr(agent, "get_agent_prompt", lambda: object())
    monkeypatch.setattr(agent, "create_tool_calling_agent",
                        lambda llm, tools, prompt: RunnableLambda(lambda x: x))
    # 流式 executor 带 lru_cache，测前后都清一遍避免串味
    agent._get_streaming_executor.cache_clear()
    yield
    agent._get_streaming_executor.cache_clear()


# ── 常量 ────────────────────────────────────────────────────────────────────

def test_constants_present_and_valued():
    assert agent.AGENT_MAX_ITERATIONS == 8
    assert agent.AGENT_MAX_EXECUTION_TIME == 60
    assert agent.HISTORY_TTL_SECONDS == 7 * 86400  # 604800，7 天


# ── 非流式 executor ─────────────────────────────────────────────────────────

def test_build_executor_limits(stub_agent_build):
    ex = agent._build_executor(tools=[])   # [] → 回退 ALL_TOOLS（真实工具，不触网）
    assert ex.verbose is False
    assert ex.max_iterations == agent.AGENT_MAX_ITERATIONS == 8
    assert ex.max_execution_time == agent.AGENT_MAX_EXECUTION_TIME == 60


# ── 流式 executor ───────────────────────────────────────────────────────────

def test_streaming_executor_limits(stub_agent_build):
    ex = agent._get_streaming_executor(frozenset())
    assert ex.verbose is False
    assert ex.max_iterations == 8
    assert ex.max_execution_time == 60


def test_limits_track_constants(stub_agent_build, monkeypatch):
    """executor 的上限真取自模块常量：改常量 → 新建的 executor 跟着变。"""
    monkeypatch.setattr(agent, "AGENT_MAX_ITERATIONS", 3)
    monkeypatch.setattr(agent, "AGENT_MAX_EXECUTION_TIME", 12)
    ex = agent._build_executor(tools=[])
    assert ex.max_iterations == 3
    assert ex.max_execution_time == 12
