# tools/__init__.py
from .system_tool import system_stats_tool
from .rag_tool import query_knowledge
from .analysis_skills import alarm_reduction_tool
from .ops_skills import k8s_restart_tool
from .monitor_skills import cpu_analyzer_tool
from .prometheus_tool import prometheus_tool
from .feishu_tool import feishu_tool
from .gpu_advisor_tool import gpu_advisor_tool
from .pai_dsw_tool import pai_dsw_tool
from .jira_tool import jira_tool
from .ram_tool import ram_tool
from .gpu_training_advisor import gpu_training_advisor_tool
from .dsw_instance_inspector import dsw_instance_inspector_tool
from .cluster_health_tool import cluster_health_report_tool

ALL_TOOLS = [
    system_stats_tool,
    query_knowledge,
    alarm_reduction_tool,
    k8s_restart_tool,
    cpu_analyzer_tool,
    prometheus_tool,
    feishu_tool,
    gpu_advisor_tool,
    pai_dsw_tool,
    jira_tool,
    ram_tool,
    gpu_training_advisor_tool,
    dsw_instance_inspector_tool,
    cluster_health_report_tool,
]

# 工具分组：供 agent.py 做关键词路由，单一来源
TOOL_GROUPS = {
    "knowledge": {"query_knowledge"},
    "monitor":   {"system_data_manager", "analyze_node_cpu_trend", "query_infrastructure_metrics"},
    "ops":       {"compress_system_alarms", "restart_k8s_service"},
    "notify":    {"push_report_to_feishu"},
    "advisor":   {"advise_gpu_cluster", "query_infrastructure_metrics"},
    "pai_dsw":   {"manage_pai_dsw"},
    "jira":      {"manage_jira"},
    "training":  {"analyze_gpu_training", "query_infrastructure_metrics"},
    "inspect":   {"inspect_dsw_instance", "manage_pai_dsw"},
    "cluster":   {"cluster_health_report", "inspect_dsw_instance"},
}

# 启动时校验 TOOL_GROUPS 中的名字与 ALL_TOOLS 一致，防止静默路由失效
_ALL_TOOL_NAMES = {t.name for t in ALL_TOOLS}
for _g, _ns in TOOL_GROUPS.items():
    _unknown = _ns - _ALL_TOOL_NAMES
    if _unknown:
        raise ValueError(f"TOOL_GROUPS[{_g!r}] 含未知工具名: {_unknown}")
