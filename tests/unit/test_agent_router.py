"""core/agent._select_tools 三层路由测试。

测试层次：
  ① 快通道：含专有名词时直接走关键词路由，不调 LLM
  ② LLM 路由：mock intent_router.route，断言工具集正确
  ③ 兜底：LLM 返回空时降级到老关键词路由

LLM 调用全部 mock。
"""
import pytest


# ── 快通道 ① ──────────────────────────────────────────────────────────────
# 含专有名词时直接走关键词路由，不调 LLM（_FAST_PATH_TOKENS 命中）

@pytest.mark.parametrize("text, must_contain", [
    # DSW
    ("查看我的 dsw 实例",                "manage_pai_dsw"),
    ("启动 pai dsw",                     "manage_pai_dsw"),
    # Jira
    ("我的 jira 工单",                   "manage_jira"),
    ("最近的 gpu申请 记录",               "manage_jira"),
    # ECS / OSS / SLS（专有名词）
    ("列一下 ecs",                       "manage_ecs"),
    ("看看 oss bucket 用量",             "manage_oss"),
    ("查 sls 日志",                      "manage_sls"),
    # K8s / Pod
    ("帮我重启 prod 的 pod",              "restart_k8s_service"),
    ("k8s 集群有问题",                   "restart_k8s_service"),
    # 监控
    ("看 prometheus 指标",               "query_infrastructure_metrics"),
    ("打开 grafana",                     "query_infrastructure_metrics"),
    # 算力效率日报（MFU）— 线上踩坑：曾被 算力→monitor / 效率→advisor 泛词分支抢走
    ("GPU日报",                          "cluster_mfu_report"),
    ("看下今天的mfu",                    "cluster_mfu_report"),
    ("算力效率日报推一下",                "cluster_mfu_report"),
    ("发个算力日报",                     "cluster_mfu_report"),
])
def test_fast_path_skips_llm(monkeypatch, text, must_contain):
    """快通道命中：不应调 LLM，应直接由关键词路由产出工具集。"""
    # 监视 LLM 路由是否被调用
    llm_called = {"n": 0}
    def fake_route(t):
        llm_called["n"] += 1
        return []
    monkeypatch.setattr("core.intent_router.route", fake_route)

    from core.agent import _select_tools
    tools = _select_tools(text)
    names = {t.name for t in tools}

    assert must_contain in names, (
        f"快通道路由失败：输入 {text!r}，期望 {must_contain!r}，实得 {sorted(names)}"
    )
    assert llm_called["n"] == 0, "快通道命中时不应调用 LLM 路由"


# ── LLM 路由 ② ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("intents, must_contain", [
    (["advisor"],                        "advise_gpu_cluster"),
    (["cluster"],                        "cluster_health_report"),
    (["cluster", "advisor"],             "cluster_health_report"),
    (["training"],                       "analyze_gpu_training"),
    (["monitor"],                        "query_infrastructure_metrics"),
    (["notify"],                         "push_report_to_feishu"),
    (["inspect"],                        "inspect_dsw_instance"),
    (["knowledge"],                      "query_knowledge"),
])
def test_llm_route_decides_tools(monkeypatch, intents, must_contain):
    """LLM 路由返回意图名 → 应转为对应工具集。"""
    monkeypatch.setattr("core.intent_router.route", lambda t: intents)

    from core.agent import _select_tools
    # 不含快通道专有名词的自然语言输入
    tools = _select_tools("帮我看下整体效率情况")
    names = {t.name for t in tools}
    assert must_contain in names


def test_llm_route_multi_intent_merges_tool_groups(monkeypatch):
    """LLM 返回多个意图 → 工具集应合并。"""
    monkeypatch.setattr("core.intent_router.route", lambda t: ["cluster", "monitor"])

    from core.agent import _select_tools
    tools = _select_tools("帮我整体看下情况")
    names = {t.name for t in tools}
    assert "cluster_health_report" in names
    assert "query_infrastructure_metrics" in names


# ── 兜底 ③ ────────────────────────────────────────────────────────────────

def test_llm_empty_result_falls_back_to_legacy(monkeypatch):
    """LLM 返回空时降级到老关键词路由，仍能识别『集群优化』。"""
    monkeypatch.setattr("core.intent_router.route", lambda t: [])

    from core.agent import _select_tools
    tools = _select_tools("集群优化建议")
    names = {t.name for t in tools}
    # 老路由的「集群优化」分支命中 cluster + advisor
    assert "advise_gpu_cluster" in names


def test_llm_unknown_intent_falls_back(monkeypatch):
    """LLM 返回的意图都不在 TOOL_GROUPS 中时降级到关键词路由。

    intent_router.route 会过滤未知项，所以这里直接模拟它返回空（已被过滤）。
    """
    monkeypatch.setattr("core.intent_router.route", lambda t: [])

    from core.agent import _select_tools
    tools = _select_tools("效率怎么提升")
    names = {t.name for t in tools}
    # 老关键词路由能命中「效率」→ advisor
    assert "advise_gpu_cluster" in names


def test_llm_exception_falls_back(monkeypatch):
    """LLM 路由模块异常时降级（极少触发，但要测）。"""
    def raise_err(t):
        raise RuntimeError("router crashed")
    monkeypatch.setattr("core.intent_router.route", raise_err)

    from core.agent import _select_tools
    tools = _select_tools("集群优化建议")
    names = {t.name for t in tools}
    # 兜底关键词路由仍能识别
    assert "advise_gpu_cluster" in names


def test_completely_unknown_text_returns_all_tools(monkeypatch):
    """输入与所有意图都无关 → LLM 返空 → 老关键词路由也无命中 → 全量。"""
    monkeypatch.setattr("core.intent_router.route", lambda t: [])

    from core.agent import _select_tools
    from tools import ALL_TOOLS
    tools = _select_tools("今天天气真好")
    assert len(tools) == len(ALL_TOOLS)


# ── 边界 ──────────────────────────────────────────────────────────────────

def test_fast_path_overrides_llm_decision(monkeypatch):
    """快通道触发时，即使 LLM 想返回别的也不会被调到。"""
    monkeypatch.setattr(
        "core.intent_router.route",
        lambda t: ["knowledge"],  # 假如 LLM 想路由到 knowledge
    )
    from core.agent import _select_tools
    tools = _select_tools("查 oss bucket")
    names = {t.name for t in tools}
    assert "manage_oss" in names  # 但快通道走 oss


def test_dsw_keyword_does_not_leak_to_advisor():
    """『dsw』走快通道 → pai_dsw 组，不应混入 advisor。"""
    from core.agent import _select_tools
    tools = _select_tools("查看 dsw 实例")
    names = {t.name for t in tools}
    assert "manage_pai_dsw" in names
    assert "cluster_health_report" not in names


# ── 知识库覆盖：快通道不应抢『知识库/文档/手册』场景 ─────────────────────────

def test_knowledge_intent_overrides_k8s_fast_path(monkeypatch):
    """含『知识库』+ k8s 时不走快通道，LLM 决定。"""
    llm_called = {"n": 0}
    def fake_route(t):
        llm_called["n"] += 1
        return ["knowledge"]
    monkeypatch.setattr("core.intent_router.route", fake_route)

    from core.agent import _select_tools
    tools = _select_tools("看下知识库怎么部署 K8s")
    names = {t.name for t in tools}
    assert "query_knowledge" in names, "知识库意图应被识别"
    assert llm_called["n"] == 1, "应调用 LLM 判定而不是快通道直接抢"


def test_knowledge_intent_overrides_dsw_fast_path(monkeypatch):
    """含『文档』+ dsw 时也让 LLM 决定。"""
    llm_called = {"n": 0}
    monkeypatch.setattr(
        "core.intent_router.route",
        lambda t: (llm_called.__setitem__("n", llm_called["n"] + 1) or ["knowledge"]),
    )

    from core.agent import _select_tools
    _select_tools("dsw 的使用文档在哪")
    assert llm_called["n"] == 1


def test_pure_k8s_still_takes_fast_path(monkeypatch):
    """无知识库词时，k8s 仍然走快通道（不调 LLM）。"""
    llm_called = {"n": 0}
    def fake_route(t):
        llm_called["n"] += 1
        return []
    monkeypatch.setattr("core.intent_router.route", fake_route)

    from core.agent import _select_tools
    tools = _select_tools("重启 k8s 的 pod")
    names = {t.name for t in tools}
    assert "restart_k8s_service" in names
    assert llm_called["n"] == 0, "纯 k8s 输入仍应走快通道"
