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
from .dsw_instance_inspector import dsw_instance_inspector_tool, cluster_health_report_tool

# Agent 可直接导入的完整工具列表
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
