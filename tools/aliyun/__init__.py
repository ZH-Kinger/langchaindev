"""阿里云相关工具：PAI DSW / ECS / OSS / SLS / Prometheus / RAM / GPU Advisor 等。"""
from .pai_dsw import pai_dsw_tool
from .ecs import ecs_tool
from .oss import oss_tool
from .sls import sls_tool
from .ram import ram_tool
from .prometheus import prometheus_tool
from .dsw_inspector import dsw_instance_inspector_tool
from .gpu_advisor import gpu_advisor_tool
from .gpu_training_advisor import gpu_training_advisor_tool
from .cluster_health import cluster_health_report_tool
from .cluster_mfu import cluster_mfu_report_tool

__all__ = [
    "pai_dsw_tool",
    "ecs_tool",
    "oss_tool",
    "sls_tool",
    "ram_tool",
    "prometheus_tool",
    "dsw_instance_inspector_tool",
    "gpu_advisor_tool",
    "gpu_training_advisor_tool",
    "cluster_health_report_tool",
    "cluster_mfu_report_tool",
]
