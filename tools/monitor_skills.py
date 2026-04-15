"""
CPU 趋势分析工具（已升级）。
原有死代码（硬编码假 Prometheus URL）已废弃，
功能代理到 prometheus_tool，保持工具名称和导出符号不变。
"""
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from tools.prometheus_tool import query_prometheus_metrics


class CPUAnalysisSchema(BaseModel):
    cluster_name: str = Field(
        description="K8s 集群名称（向后兼容参数，当前版本不使用）"
    )
    node_name: str = Field(
        description="节点名称或 IP，将作为 node_filter 过滤 Prometheus 数据"
    )


def analyze_cpu_trend(cluster_name: str, node_name: str) -> str:
    return query_prometheus_metrics(
        query_type="cpu",
        node_filter=node_name,
        duration_minutes=60,
    )


cpu_analyzer_tool = StructuredTool.from_function(
    func=analyze_cpu_trend,
    name="analyze_node_cpu_trend",
    description="分析指定节点过去 1 小时的 CPU 使用率趋势，判断是持续高压、瞬时抖动还是运行平稳。",
    args_schema=CPUAnalysisSchema,
)
