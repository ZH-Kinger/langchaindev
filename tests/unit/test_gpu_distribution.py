"""GPU 卡分布单测：聚合(地区×卡型 + 用户在算卡数) / HTML 渲染 / 摘要卡 / 意图。"""


def test_gather_aggregates(monkeypatch):
    from tools.aliyun import gpu_distribution as G
    from tools.aliyun import cluster_mfu

    def fake_grouped(promql, keys):
        if "TOTAL" in promql:
            return {("cn-beijing", "GU8T"): 64.0, ("cn-hangzhou", "GU8T"): 256.0,
                    ("ap-southeast-1", "L20X"): 48.0}
        if "REQUEST" in promql:
            return {("cn-beijing", "GU8T"): 64.0, ("cn-hangzhou", "GU8T"): 111.0,
                    ("ap-southeast-1", "L20X"): 48.0}
        # 用户在算：jobUserId × regionId
        return {("111", "cn-beijing"): 64.0, ("222", "ap-southeast-1"): 48.0,
                ("333", "cn-hangzhou"): 32.0}

    monkeypatch.setattr(cluster_mfu, "_grouped", fake_grouped)
    monkeypatch.setattr(G, "_name_map", lambda: {"111": "张宸豪", "222": "钱秉笙", "333": "梁嘉琦"})
    monkeypatch.setattr(G, "_merge_dsw", lambda users: None)

    g = G._gather()
    byname = {(r["region_name"], r["gpu_name"]): r for r in g["regions"]}
    assert byname[("北京", "H20")]["used"] == 64 and byname[("北京", "H20")]["total"] == 64
    assert byname[("杭州", "H20")]["used"] == 111 and byname[("杭州", "H20")]["free"] == 145
    assert byname[("新加坡", "H200")]["rate"] == 100.0
    # 用户降序 + 姓名映射 + 卡型按地区精确带出
    assert g["users"][0]["name"] == "张宸豪" and g["users"][0]["total"] == 64
    assert g["users"][0]["by_type"] == {"H20": 64}
    u2 = next(u for u in g["users"] if u["name"] == "钱秉笙")
    assert u2["by_type"] == {"H200": 48}
    assert g["total_cards"] == 64 + 256 + 48
    assert g["used_cards"] == 64 + 111 + 48
    assert g["active_cards"] == 64 + 48 + 32
    assert g["user_count"] == 3


def test_gather_unknown_user_falls_back_to_id(monkeypatch):
    from tools.aliyun import gpu_distribution as G
    from tools.aliyun import cluster_mfu
    monkeypatch.setattr(cluster_mfu, "_grouped", lambda q, k: (
        {("cn-beijing", "GU8T"): 8.0} if "TOTAL" in q else
        {("cn-beijing", "GU8T"): 8.0} if "REQUEST" in q else
        {("999", "cn-beijing"): 8.0}))
    monkeypatch.setattr(G, "_name_map", lambda: {})   # 映射为空
    monkeypatch.setattr(G, "_merge_dsw", lambda users: None)
    g = G._gather()
    assert g["users"][0]["name"] == "999"   # 找不到姓名 → 回退用 id


def test_build_html_top10_and_collapse():
    from tools.aliyun import gpu_distribution as G
    users = [{"name": f"u{i}", "total": 100 - i, "by_type": {"H20": 100 - i}, "by_region": {}}
             for i in range(15)]
    g = {"gathered_at": 1700000000,
         "regions": [{"region_name": "北京", "gpu_name": "H20", "total": 64, "used": 20,
                      "free": 44, "rate": 31.2}],
         "users": users, "total_cards": 64, "used_cards": 20,
         "active_cards": sum(u["total"] for u in users), "user_count": 15}
    html = G.build_html(g, token="TK")
    assert "location.reload()" in html          # 定时刷新（非 meta，防外链超时卡死）
    assert "展开其余 5 人" in html          # 15 - 10
    assert "北京" in html and "u0" in html
    assert "TK" not in html                 # token 绝不写进页面正文
    assert "立即刷新" in html               # 手动刷新按钮
    assert "共 15 人" in html               # 标题按实际人数
    assert "justify-content:center" in html  # 居中
    # 交互一律 addEventListener，绝无内联 onclick（飞书 webview 会静默屏蔽内联处理器）
    assert "onclick=" not in html
    assert "addEventListener" in html
    assert "UI v7" in html
    assert 'id="btn-refresh"' in html       # 刷新按钮用 id 绑定


def test_gather_timeseries(monkeypatch):
    from tools.aliyun import gpu_distribution as G
    from tools.aliyun import cluster_mfu
    monkeypatch.setattr(cluster_mfu, "_grouped",
                        lambda q, k: {("cn-beijing", "GU8T"): 64.0, ("cn-hangzhou", "GU8T"): 256.0})
    monkeypatch.setattr(cluster_mfu, "_peak_for", lambda t: 148.0)

    def fake_rs(expr, *a, **k):
        if expr.startswith("count("):                # 在算(合并 SM>thr)——须在 SM_UTIL 之前判
            return {1000.0: 16.0, 1600.0: 16.0}
        if "DUTTY_UTIL" in expr:                     # per-region GPU 算力利用率(资源组 DutyCycle)
            return {1000.0: 70.0, 1600.0: 72.0}
        if "PIP_TENSOR_ACTIVE" in expr:              # per-region Tensor 管线活跃%
            return {1000.0: 1.8, 1600.0: 1.9}
        if "ACCELERATOR_REQUEST" in expr:            # 已分配
            return {1000.0: 20.0, 1600.0: 22.0}
        if "ACCELERATOR_TOTAL" in expr:              # 总
            return {1000.0: 64.0, 1600.0: 64.0}
        return {}

    monkeypatch.setattr(cluster_mfu, "_range_series", fake_rs)
    ts = G._gather_timeseries(hours=24)
    assert ts["labels"]
    # 头版 GPU 算力利用率 = 资源组 DutyCycle，每地区一条
    assert len(ts["duty_region"]) == 2 and all("H20" in k for k in ts["duty_region"])
    assert abs(list(ts["duty_region"].values())[0][0] - 70.0) < 0.1
    # 卡数三线：已分配/在算/空闲(=总-已分配)
    assert ts["cards"]["allocated"][0] == 20 and ts["cards"]["active"][0] == 16
    assert ts["cards"]["idle"][0] == 44          # 64 - 20
    # 只剩 Tensor 管线活跃；SM / TFLOPS / 显存 均已移除
    assert "sm_region" not in ts and "tflops_region" not in ts and "mem_util" not in ts
    assert len(ts["tensor_region"]) == 2
    assert abs(list(ts["tensor_region"].values())[0][0] - 1.8) < 0.1


def test_build_html_with_charts():
    from tools.aliyun import gpu_distribution as G
    g = {"gathered_at": 1700000000,
         "regions": [{"region_name": "北京", "gpu_name": "H20", "total": 64, "used": 20, "free": 44, "rate": 31.2}],
         "users": [{"name": "张三", "total": 20, "by_type": {"H20": 20}, "by_region": {}}],
         "total_cards": 64, "used_cards": 20, "active_cards": 20, "user_count": 1}
    series = {"labels": ["10:00", "10:10"],
              "duty_region": {"北京·H20": [70, 72]},
              "tensor_region": {"北京·H20": [1.8, 1.9]},
              "cards": {"allocated": [20, 22], "active": [16, 16], "idle": [44, 42]},
              "hours": 24}
    html = G.build_html(g, series, token="SECRET")
    assert 'id="c_util"' in html and 'id="c_cards"' in html and 'id="c_tensor"' in html
    # SM / TFLOPS / 显存 图均已删
    assert 'id="c_sm"' not in html and 'id="c_tflops"' not in html and 'id="c_mem"' not in html
    assert "chart.js" in html.lower()          # Chart.js CDN
    assert '已分配 / 在算 / 空闲' in html        # 卡数三线
    assert 'class="rg' in html and "3天" in html and "24h" in html   # 时间范围选择器
    assert "GPU 算力利用率 DutyCycle" in html   # 头版=资源组 DutyCycle(阿里云同口径)
    assert "SM 利用率" not in html               # SM 已删
    assert "MFU 计算器" not in html and 'id="m_p"' not in html  # 计算器已移除
    assert "SECRET" not in html                # token 不入正文
    # 时间范围按钮改 data-h + addEventListener（无内联 onclick）
    assert "onclick=" not in html
    assert "addEventListener" in html
    for h in ("1", "6", "12", "24", "72"):
        assert f'data-h="{h}"' in html         # 5 个范围按钮各带 data-h
    # 无 series 时不渲染图表（渐进：仍出表格）
    html2 = G.build_html(g, None, token="SECRET")
    assert 'id="c_sm"' not in html2 and "北京" in html2


def test_dist_url_and_summary_card(monkeypatch):
    from tools.aliyun import gpu_distribution as G
    from config.settings import settings
    monkeypatch.setattr(settings, "GPU_DIST_BASE_URL", "http://x:8088")
    monkeypatch.setattr(settings, "RAM_QUERY_API_TOKEN", "TK")
    assert G.dist_url() == "http://x:8088/gpu/distribution?token=TK"
    monkeypatch.setattr(settings, "GPU_DIST_BASE_URL", "")
    assert G.dist_url() == ""   # 无基址 → 空

    import json
    g = {"gathered_at": 1700000000, "regions": [
        {"region_name": "北京", "gpu_name": "H20", "total": 64, "used": 20, "free": 44, "rate": 31.2}],
        "users": [{"name": "张三", "total": 20, "by_type": {"H20": 20}, "by_region": {}}],
        "total_cards": 64, "used_cards": 20, "active_cards": 20, "user_count": 1}
    sc = json.dumps(G.summary_card(g, "http://x/gpu/distribution?token=T"), ensure_ascii=False)
    assert "打开实时页面" in sc and "北京" in sc and "张三" in sc


def test_gpu_dist_intent():
    from core.feishu_bot import messages
    assert messages._is_gpu_dist_intent("看看卡分布")
    assert messages._is_gpu_dist_intent("谁在用卡")
    assert messages._is_gpu_dist_intent("每个人用了多少卡")
    assert not messages._is_gpu_dist_intent("帮我申请一张 A100")
    assert not messages._is_gpu_dist_intent("查一下 jira")
