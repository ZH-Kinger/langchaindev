"""
核心工具路径验证（pytest 风格）。
覆盖告警降噪、K8s 重启安全边界、K8s 重启允许场景。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.ops.analysis import alarm_reduction_tool
from tools.ops.k8s import k8s_restart_tool


def test_alarm_reduction_returns_non_empty_string():
    """告警降噪返回非空 markdown 报告，且不能抛未捕获异常。"""
    res = alarm_reduction_tool.run({"file_name": "raw_alarms_200.csv"})
    assert isinstance(res, str)
    assert len(res) > 0, "告警降噪工具应返回非空字符串"
    assert "Traceback" not in res, "告警降噪工具不应抛出未捕获异常"


def test_k8s_restart_protected_namespace_rejected():
    """prod 命名空间应被安全策略拦截。"""
    res = k8s_restart_tool.run({
        "pod_name":  "order-service-v1",
        "namespace": "prod",
        "reason":    "测试安全拦截",
    })
    assert any(kw in res for kw in ("拒绝", "禁止", "受保护")), \
        f"prod 命名空间应被拦截，实际：{res!r}"


def test_k8s_restart_allowed_namespace_succeeds():
    """staging 等非保护命名空间应允许重启。"""
    res = k8s_restart_tool.run({
        "pod_name":  "order-service-v1",
        "namespace": "staging",
        "reason":    "OOM 内存溢出，需重启恢复",
    })
    assert any(kw in res for kw in ("成功", "重启")), \
        f"staging 命名空间应允许重启，实际：{res!r}"
