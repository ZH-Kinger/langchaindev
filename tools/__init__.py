# tools/__init__.py — 工具按云服务/业务功能分类，子包统一暴露
from .aliyun import (
    pai_dsw_tool,
    ecs_tool,
    oss_tool,
    sls_tool,
    ram_tool,
    prometheus_tool,
    dsw_instance_inspector_tool,
    gpu_advisor_tool,
    gpu_training_advisor_tool,
    cluster_health_report_tool,
    cluster_mfu_report_tool,
)
from .volcano import tos_tool, vepfs_dataflow_tool
from .transfer import transfer_tool
from .cpfs import cpfs_dataflow_tool
from .pfs_transfer import pfs_transfer_tool
from .feishu import feishu_tool
from .jira import jira_tool, jira_workflow_tool
from .github import github_workflow_tool
from .knowledge import query_knowledge
from .ops import (
    system_stats_tool,
    alarm_reduction_tool,
    cpu_analyzer_tool,
    k8s_restart_tool,
)

ALL_TOOLS = [
    # ── 阿里云 ────────────────────────────────────────────────
    pai_dsw_tool,
    ecs_tool,
    oss_tool,
    sls_tool,
    ram_tool,
    prometheus_tool,
    dsw_instance_inspector_tool,
    gpu_advisor_tool,
    gpu_training_advisor_tool,
    cluster_health_report_tool,
    cluster_mfu_report_tool,
    # ── 火山引擎 ─────────────────────────────────────────────
    tos_tool,
    vepfs_dataflow_tool,
    # ── 跨云迁移 ─────────────────────────────────────────────
    transfer_tool,
    # ── CPFS 预热/沉降 ───────────────────────────────────────
    cpfs_dataflow_tool,
    # ── PFS 跨云直传（vePFS↔CPFS 3 段链）─────────────────────
    pfs_transfer_tool,
    # ── 飞书 / Jira / GitHub ─────────────────────────────────
    feishu_tool,
    jira_tool,
    jira_workflow_tool,
    github_workflow_tool,
    # ── 知识库 ───────────────────────────────────────────────
    query_knowledge,
    # ── 运维通用 ─────────────────────────────────────────────
    system_stats_tool,
    alarm_reduction_tool,
    cpu_analyzer_tool,
    k8s_restart_tool,
]

# 工具分组：供 agent.py 做关键词路由，单一来源
TOOL_GROUPS = {
    "knowledge": {"query_knowledge"},
    "monitor":   {"system_data_manager", "analyze_node_cpu_trend", "query_infrastructure_metrics", "manage_ram"},
    "ops":       {"compress_system_alarms", "restart_k8s_service"},
    "notify":    {"push_report_to_feishu"},
    "advisor":   {"advise_gpu_cluster", "query_infrastructure_metrics"},
    "pai_dsw":   {"manage_pai_dsw"},
    "jira":      {"manage_jira"},
    "training":  {"analyze_gpu_training", "query_infrastructure_metrics"},
    "inspect":   {"inspect_dsw_instance", "manage_pai_dsw"},
    "cluster":   {"cluster_health_report", "cluster_mfu_report", "inspect_dsw_instance"},
    "workflow":  {"query_jira_workflow", "query_github_workflow"},
    "ecs":       {"manage_ecs"},
    "oss":       {"manage_oss"},
    "sls":       {"manage_sls"},
    "tos":       {"manage_tos"},
    "transfer":  {"manage_transfer"},
    "cpfs":      {"manage_cpfs_dataflow"},
    "vepfs":     {"manage_vepfs_dataflow"},
    "pfs_transfer": {"manage_pfs_transfer"},
}

# 启动时校验 TOOL_GROUPS 中的名字与 ALL_TOOLS 一致，防止静默路由失效
_ALL_TOOL_NAMES = {t.name for t in ALL_TOOLS}
for _g, _ns in TOOL_GROUPS.items():
    _unknown = _ns - _ALL_TOOL_NAMES
    if _unknown:
        raise ValueError(f"TOOL_GROUPS[{_g!r}] 含未知工具名: {_unknown}")
