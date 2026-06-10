"""core/dsw_scheduler._send_cluster_morning_report 测试（集群 MFU 早报）。

mock MFU 卡片构建 + 飞书推送，断言：
  - 正常 → 推送卡片
  - 开关关闭 / PROMETHEUS_URL 缺失 / FEISHU_CHAT_ID 缺失 → 不推送
"""
import pytest


@pytest.fixture
def patch_report(monkeypatch):
    """注入可控的 MFU 卡片与推送捕获。"""
    sent = []
    import core.dsw_scheduler as sched
    from tools.aliyun import cluster_mfu
    monkeypatch.setattr(cluster_mfu, "build_mfu_card",
                        lambda *a, **k: {"header": {"title": {"content": "🧮 集群算力效率（MFU）早报"}}})
    monkeypatch.setattr(sched, "_send_card",
                        lambda open_id, chat_id, card: sent.append((chat_id, card)))
    from config.settings import settings
    monkeypatch.setattr(settings, "PROMETHEUS_URL", "https://prom.example")
    monkeypatch.setattr(settings, "FEISHU_CHAT_ID", "oc_test")
    monkeypatch.setattr(settings, "CLUSTER_MORNING_REPORT_ENABLED", True)
    return sent


def test_sends_when_ok(patch_report):
    from core.dsw_scheduler import _send_cluster_morning_report
    _send_cluster_morning_report()
    assert len(patch_report) == 1
    assert patch_report[0][0] == "oc_test"
    assert "MFU" in patch_report[0][1]["header"]["title"]["content"]


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


def test_skip_when_no_chat(patch_report, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "FEISHU_CHAT_ID", "")
    from core.dsw_scheduler import _send_cluster_morning_report
    _send_cluster_morning_report()
    assert patch_report == []
