"""
Grafana 连通性验证（集成测试 — 默认 skip）。
跑：pytest -m integration tests/integration/test_grafana.py
需要 GRAFANA_URL 已在 .env 中配置。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import requests

pytestmark = pytest.mark.integration

from config.settings import settings


def test_grafana_health():
    """Grafana /api/health 应返回 200。"""
    settings.setup_env()
    url = settings.GRAFANA_URL
    if not url:
        pytest.skip("GRAFANA_URL 未配置")
    try:
        r = requests.get(f"{url}/api/health", timeout=5)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"❌ Grafana 连接失败: {e}")
    assert r.status_code == 200, f"健康检查应返回 200，实际 {r.status_code}"


def test_grafana_datasource_query():
    """通过 UID 查询 Prometheus 数据源应返回 200。"""
    settings.setup_env()
    url = settings.GRAFANA_URL
    key = settings.GRAFANA_API_KEY
    uid = settings.GRAFANA_DATASOURCE_UID
    if not (url and key and uid):
        pytest.skip("GRAFANA_URL / API_KEY / DATASOURCE_UID 未完整配置")
    r = requests.get(
        f"{url}/api/datasources/uid/{uid}",
        headers={"Authorization": f"Bearer {key}"},
        timeout=5,
    )
    assert r.status_code == 200, f"数据源 {uid} 查询应返回 200，实际 {r.status_code}"
    name = r.json().get("name", "")
    assert name, "数据源名称不应为空"
