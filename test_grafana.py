from config.settings import settings
import requests

settings.setup_env()

url = settings.GRAFANA_URL
key = settings.GRAFANA_API_KEY
uid = settings.GRAFANA_DATASOURCE_UID

print(f'GRAFANA_URL: {url}')
print(f'API_KEY: {"已配置" if key else "未配置"}')
print(f'DATASOURCE_UID: {uid}')

if not url:
    print('ERROR: GRAFANA_URL 未填写')
else:
    try:
        r = requests.get(f'{url}/api/health', timeout=5)
        print(f'Grafana 连通性: {r.status_code}', r.json())
    except Exception as e:
        print(f'连接失败: {e}')

    if key:
        try:
            r = requests.get(f'{url}/api/datasources/uid/{uid}', 
                             headers={'Authorization': f'Bearer {key}'}, timeout=5)
            print(f'数据源: {r.status_code}', r.json().get('name', r.text[:80]))
        except Exception as e:
            print(f'数据源查询失败: {e}')
