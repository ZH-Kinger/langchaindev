"""Bot 对话完善：工具按消息缩小 / 未知兜底 / 帮助菜单。"""
import pytest


def _names(tools):
    return {t.name for t in tools}


def test_keyword_route_none_on_unknown():
    from core import agent
    assert agent._keyword_route_names("dataset_catalog") is None
    assert agent._keyword_route_names("随便一句没意义的话") is None
    # 已知的仍返回集合
    assert agent._keyword_route_names("查一下 oss 用量")


def test_scoped_oss_no_transfer(monkeypatch):
    from core import agent
    tools, intents = agent.select_tools_scoped("看看杭州 oss 用量")   # 快通道命中 oss
    assert tools is not None
    names = _names(tools)
    assert any("oss" in n for n in names)
    assert "manage_transfer" not in names          # 不相关消息拿不到迁移工具


def test_scoped_transfer_has_transfer_tool():
    from core import agent
    tools, intents = agent.select_tools_scoped("把 tos://a/ 迁到 oss://b/")
    assert tools is not None
    assert "manage_transfer" in _names(tools)


def test_scoped_unknown_returns_none(monkeypatch):
    from core import agent, intent_router
    monkeypatch.setattr(intent_router, "route", lambda x: [])   # LLM 也判不出
    tools, intents = agent.select_tools_scoped("dataset_catalog")
    assert tools is None and intents == []


def test_scoped_llm_intent(monkeypatch):
    from core import agent, intent_router
    monkeypatch.setattr(intent_router, "route", lambda x: ["knowledge"])
    tools, intents = agent.select_tools_scoped("这套系统的部署规范是什么")   # 无快通道词
    assert tools is not None and intents == ["knowledge"]


def test_build_executor_uses_subset(monkeypatch):
    from core import agent
    captured = {}

    class _FakeExec:
        def __init__(self, agent=None, tools=None, **kw):
            captured["tools"] = tools

    monkeypatch.setattr(agent, "get_cloud_llm", lambda *a, **k: object())
    monkeypatch.setattr(agent, "get_agent_prompt", lambda *a, **k: object())
    monkeypatch.setattr(agent, "create_tool_calling_agent", lambda llm, tools, prompt: object())
    monkeypatch.setattr(agent, "AgentExecutor", _FakeExec)

    subset = agent.ALL_TOOLS[:2]
    agent._build_executor(subset)
    assert captured["tools"] == subset
    agent._build_executor()                 # 无参 → 全量
    assert captured["tools"] == agent.ALL_TOOLS


def test_help_intent_and_menu():
    from core.feishu_bot import messages
    assert messages._is_help_intent("帮助")
    assert messages._is_help_intent("你能做什么")
    assert messages._is_help_intent("有什么功能")
    assert not messages._is_help_intent("帮我查一下卡分布")
    menu = messages._capability_menu()
    for kw in ("卡分布", "算力日报", "OSS", "迁移", "知识库"):
        assert kw in menu
