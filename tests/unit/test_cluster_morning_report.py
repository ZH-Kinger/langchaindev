"""core/dsw_scheduler._send_cluster_morning_report 测试。

mock prometheus 报告生成 + 飞书推送，断言：
  - 正常报告 → 推送
  - PROMETHEUS_URL 缺失 / 报告异常 / 开关关闭 → 不推送
"""
import pytest


@pytest.fixture
def patch_report(monkeypatch):
    """注入可控的报告与推送捕获。"""
    sent = []
    from tools.aliyun import prometheus as prom
    from tools.feishu import notify
    monkeypatch.setattr(prom, "query_prometheus_metrics",
                        lambda **k: "[REPORT_START]\nOVERALL_COLOR: green\n...[REPORT_END]")
    monkeypatch.setattr(notify, "send_feishu_report",
                        lambda content, title="": sent.append((title, content)) or "✅ ok")
    from config.settings import settings
    monkeypatch.setattr(settings, "PROMETHEUS_URL", "https://prom.example")
    monkeypatch.setattr(settings, "CLUSTER_MORNING_REPORT_ENABLED", True)
    return sent


def test_sends_when_report_ok(patch_report):
    from core.dsw_scheduler import _send_cluster_morning_report
    _send_cluster_morning_report()
    assert len(patch_report) == 1
    assert "集群监控早报" in patch_report[0][0]


def test_skip_when_disabled(patch_report, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CLUSTER_MORNING_REPORT_ENABLED", False)
    from core.dsw_scheduler import _send_cluster_morning_report
    _send_cluster_morning_report()
    assert patch_report == []


def test_skip_when_no_prometheus(patch_report, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "PROMETHEUS_URL", "")
    from core.dsw_scheduler import _send_cluster_morning_report
    _send_cluster_morning_report()
    assert patch_report == []


def test_skip_when_report_error(patch_report, monkeypatch):
    from tools.aliyun import prometheus as prom
    monkeypatch.setattr(prom, "query_prometheus_metrics", lambda **k: "❌ PROMETHEUS_URL 未配置")
    from core.dsw_scheduler import _send_cluster_morning_report
    _send_cluster_morning_report()
    assert patch_report == []          # 报告异常不推送（避免推错误卡片）
