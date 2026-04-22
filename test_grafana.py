"""
Grafana 连通性验证。
运行：python test_grafana.py
需要 GRAFANA_URL 已在 .env 中配置，否则跳过网络断言。
"""
from config.settings import settings
import requests

settings.setup_env()

url = settings.GRAFANA_URL
key = settings.GRAFANA_API_KEY
uid = settings.GRAFANA_DATASOURCE_UID

print(f"GRAFANA_URL:      {url or '(未配置)'}")
print(f"API_KEY:          {'已配置' if key else '未配置'}")
print(f"DATASOURCE_UID:   {uid or '(未配置)'}")

if not url:
    print("⚠️  GRAFANA_URL 未填写，跳过网络测试。")
else:
    # ── 健康检查 ────────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{url}/api/health", timeout=5)
        print(f"\nGrafana 健康检查: HTTP {r.status_code}  {r.json()}")
        assert r.status_code == 200, \
            f"Grafana /api/health 应返回 200，实际 {r.status_code}"
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"❌ Grafana 连接失败: {e}") from e

    # ── 数据源查询（需要 API Key）──────────────────────────────────────────
    if key and uid:
        r2 = requests.get(
            f"{url}/api/datasources/uid/{uid}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        name = r2.json().get("name", r2.text[:80])
        print(f"数据源查询: HTTP {r2.status_code}  {name}")
        assert r2.status_code == 200, \
            f"数据源 {uid} 查询应返回 200，实际 {r2.status_code}"
        assert name, "数据源名称不应为空"
    else:
        print("⚠️  API_KEY 或 DATASOURCE_UID 未配置，跳过数据源验证。")

print("\n✅ 全部断言通过")