"""`/api/ram/user` 与 `/gpu/distribution` 的 token 门禁 fail-closed 回归。

安全整改批次 1（B2）：两处门禁原来都会**回退到 `FEISHU_VERIFICATION_TOKEN`**：

    _ram_api_authorized:  RAM_QUERY_API_TOKEN or FEISHU_VERIFICATION_TOKEN
    _gpu_dist_authorized: GPU_DIST_TOKEN or RAM_QUERY_API_TOKEN or FEISHU_VERIFICATION_TOKEN

线上 `RAM_QUERY_API_TOKEN` 实测为空 → `/api/ram/user` 一直在拿 webhook 的验证 token 当钥匙。
两个面共用一把钥匙 = 任一侧泄漏另一侧全开：拿到它就能伪造 `/feishu/card_action`、把 open_id
填成管理员，过掉 OSS 权限下发 / 迁移超阈值确认等全部管理员门禁。

现改为只认各自专用 token，没配就一律 403。本文件对**每条 token 取值路径**
（`X-API-Token` 头 / `Authorization: Bearer` / `?token=`）分别验证，并且每条拒绝用例都装
哨兵证明后端动作（RAM 查询 / GPU 快照采集）**一次都没被执行**，而不只是看状态码。

另外钉住 `_api_token_ok` 的非 ASCII 兜底：`hmac.compare_digest` 对含非 ASCII 的 str 抛
TypeError —— 上一轮 `_token_verified` 已经踩过一次（匿名请求即可把接口打成 500，而 500 经
`utils/logger` 的 ERROR 回调会给管理员刷飞书私信 = 免鉴权的告警放大器）。`_api_token_ok`
是同类新代码，这里留常绿用例防回归。
"""
import pytest

# 收集期导入：与 fakeredis fixture 的执行顺序解耦（这些模块里若有 `from ... import get_redis`
# 之类的导入即绑定写法，首次导入落在 fixture 之后会绑到某一个用例的假 Redis 上）。
from core.feishu_bot import routes as _routes  # noqa: F401
from core import ram_query as _ram_query  # noqa: F401
from tools.aliyun import gpu_distribution as _gd  # noqa: F401

WEBHOOK_TOKEN = "V_webhook_verification_token"
RAM_TOKEN = "R_ram_query_token"
GPU_TOKEN = "G_gpu_dist_token"


@pytest.fixture
def api(monkeypatch):
    """test_client + 后端哨兵（被调到即视为门禁失守）。"""
    from core.feishu_bot import routes
    from core import ram_query
    from tools.aliyun import gpu_distribution

    hits = {"ram_query": [], "gpu_dist": []}

    def _ram_sentinel(login_name, *a, **k):
        hits["ram_query"].append(login_name)
        return {"user_name": login_name, "access_keys": []}

    monkeypatch.setattr(ram_query, "query_ram_account", _ram_sentinel)

    def _dist_sentinel(*a, **k):
        hits["gpu_dist"].append(("get_distribution", a, k))
        return {"regions": [], "users": []}

    monkeypatch.setattr(gpu_distribution, "get_distribution", _dist_sentinel)
    monkeypatch.setattr(gpu_distribution, "get_timeseries", lambda *a, **k: {})
    monkeypatch.setattr(gpu_distribution, "build_html", lambda *a, **k: "<html>ok</html>")
    monkeypatch.setattr(routes.settings, "GPU_DIST_ENABLED", True, raising=False)

    hits["client"] = routes.app.test_client()
    hits["routes"] = routes
    return hits


def _no_dedicated_token(monkeypatch, routes, *, ram="", gpu=""):
    """线上现状：专用 token 为空，webhook token 有值。"""
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", ram, raising=False)
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", gpu, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", WEBHOOK_TOKEN, raising=False)


# ── 1. /api/ram/user：RAM_QUERY_API_TOKEN 为空 → webhook token 一律不认 ────────

@pytest.mark.parametrize("carrier", ["header", "bearer", "query"])
def test_ram_api_rejects_webhook_token_when_no_dedicated_token(api, monkeypatch, carrier):
    routes = api["routes"]
    _no_dedicated_token(monkeypatch, routes)

    url, headers = "/api/ram/user?login_name=alice", {}
    if carrier == "header":
        headers = {"X-API-Token": WEBHOOK_TOKEN}
    elif carrier == "bearer":
        headers = {"Authorization": f"Bearer {WEBHOOK_TOKEN}"}
    else:
        url += f"&token={WEBHOOK_TOKEN}"

    resp = api["client"].get(url, headers=headers)

    assert resp.status_code == 403, f"{carrier} 路径拿 webhook token 竟然过门了"
    assert resp.get_json() == {"ok": False, "error": "unauthorized"}
    assert api["ram_query"] == [], "403 之外还必须证明 RAM 查询一次没执行"


def test_ram_api_rejects_webhook_token_on_post_body_path(api, monkeypatch):
    """POST 走 body 取 login_name，同样过不去。"""
    routes = api["routes"]
    _no_dedicated_token(monkeypatch, routes)

    resp = api["client"].post(
        "/api/ram/user", json={"login_name": "alice"},
        headers={"X-API-Token": WEBHOOK_TOKEN},
    )

    assert resp.status_code == 403
    assert api["ram_query"] == []


def test_ram_api_rejects_everything_when_no_token_configured_at_all(api, monkeypatch):
    """三个 token 全空 → 无论带不带什么都 403（不存在「空 token == 空 supplied」的洞）。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", "", raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", "", raising=False)

    for headers in ({}, {"X-API-Token": ""}, {"X-API-Token": "anything"}):
        resp = api["client"].get("/api/ram/user?login_name=alice", headers=headers)
        assert resp.status_code == 403
    assert api["ram_query"] == []


def test_ram_api_still_works_with_its_own_token(api, monkeypatch):
    """阳性对照：配了专用 token 就照常放行 —— 证明上面的 403 不是路由本身坏了。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", WEBHOOK_TOKEN, raising=False)

    resp = api["client"].get("/api/ram/user?login_name=alice",
                             headers={"X-API-Token": RAM_TOKEN})

    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    assert api["ram_query"] == ["alice"]


def test_ram_api_rejects_gpu_token_cross_use(api, monkeypatch):
    """两个接口的 token 不互通：拿 GPU 页面 token 打 RAM 接口 → 403。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", GPU_TOKEN, raising=False)

    resp = api["client"].get("/api/ram/user?login_name=alice",
                             headers={"X-API-Token": GPU_TOKEN})

    assert resp.status_code == 403
    assert api["ram_query"] == []


# ── 2. /gpu/distribution：同款三态 ────────────────────────────────────────────

@pytest.mark.parametrize("carrier", ["header", "bearer", "query"])
@pytest.mark.parametrize("stolen,ram_cfg", [
    # 旧回退链是 GPU → RAM → webhook。逐级构造「该级恰好是链上第一个非空值」的配置，
    # 否则短路在前一级、后面那级根本没被验到（假绿）。
    (RAM_TOKEN, RAM_TOKEN),   # 旧链第 2 级
    (WEBHOOK_TOKEN, ""),      # 旧链第 3 级：RAM 也留空，webhook 才轮得到
])
def test_gpu_page_rejects_fallback_tokens_when_no_dedicated_token(
        api, monkeypatch, carrier, stolen, ram_cfg):
    """`GPU_DIST_TOKEN` 为空时，webhook token 与 RAM 接口 token 都不再是钥匙。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", "", raising=False)
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", ram_cfg, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", WEBHOOK_TOKEN, raising=False)

    url, headers = "/gpu/distribution", {}
    if carrier == "header":
        headers = {"X-API-Token": stolen}
    elif carrier == "bearer":
        headers = {"Authorization": f"Bearer {stolen}"}
    else:
        url += f"?token={stolen}"

    resp = api["client"].get(url, headers=headers)

    assert resp.status_code == 403
    assert b"unauthorized" in resp.data
    assert api["gpu_dist"] == [], "403 之外还必须证明没去采 GPU 快照"


def test_gpu_page_still_works_with_its_own_token(api, monkeypatch):
    """阳性对照：`GPU_DIST_TOKEN` 有值且带对 → 200，页面照常渲染。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", GPU_TOKEN, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", WEBHOOK_TOKEN, raising=False)

    resp = api["client"].get(f"/gpu/distribution?token={GPU_TOKEN}")

    assert resp.status_code == 200
    assert len(api["gpu_dist"]) == 1


def test_gpu_page_disabled_flag_wins_over_token(api, monkeypatch):
    """开关关掉时先 404，且同样不采数据（不因为 token 对就绕过开关）。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "GPU_DIST_ENABLED", False, raising=False)
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", GPU_TOKEN, raising=False)

    resp = api["client"].get(f"/gpu/distribution?token={GPU_TOKEN}")

    assert resp.status_code == 404
    assert api["gpu_dist"] == []


def test_dist_url_not_generated_without_dedicated_token(monkeypatch):
    """链接生成侧同款 fail-closed：没有 `GPU_DIST_TOKEN` 就不推带 token 的链接进群。"""
    from tools.aliyun import gpu_distribution as gd

    monkeypatch.setattr(gd.settings, "GPU_DIST_BASE_URL", "https://bot.example.com", raising=False)
    monkeypatch.setattr(gd.settings, "GPU_DIST_TOKEN", "", raising=False)
    monkeypatch.setattr(gd.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)
    monkeypatch.setattr(gd.settings, "FEISHU_VERIFICATION_TOKEN", WEBHOOK_TOKEN, raising=False)

    url = gd.dist_url()

    assert url == ""
    assert WEBHOOK_TOKEN not in url and RAM_TOKEN not in url

    monkeypatch.setattr(gd.settings, "GPU_DIST_TOKEN", GPU_TOKEN, raising=False)
    assert gd.dist_url() == f"https://bot.example.com/gpu/distribution?token={GPU_TOKEN}"


# ── 3. 非 ASCII token → 403 而非 500（compare_digest TypeError 兜底）────────────

_NON_ASCII_QUERY = "令牌不对"        # URL query，任意 unicode 都能传
_NON_ASCII_HEADER = "ténoken"       # HTTP 头只保证 latin-1，用 latin-1 内的非 ASCII 字符


@pytest.mark.parametrize("bad", [_NON_ASCII_QUERY, "🔑", "токен"])
def test_ram_api_non_ascii_query_token_is_403_not_500(api, monkeypatch, bad):
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)

    resp = api["client"].get(f"/api/ram/user?login_name=alice&token={bad}")

    assert resp.status_code == 403, "非 ASCII token 不能打成 500（500 会经 ERROR 回调刷管理员飞书）"
    assert api["ram_query"] == []


def test_ram_api_non_ascii_header_token_is_403_not_500(api, monkeypatch):
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)

    resp = api["client"].get("/api/ram/user?login_name=alice",
                             headers={"X-API-Token": _NON_ASCII_HEADER})

    assert resp.status_code == 403
    assert api["ram_query"] == []


def test_ram_api_non_ascii_bearer_token_is_403_not_500(api, monkeypatch):
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)

    resp = api["client"].get("/api/ram/user?login_name=alice",
                             headers={"Authorization": f"Bearer {_NON_ASCII_HEADER}"})

    assert resp.status_code == 403
    assert api["ram_query"] == []


@pytest.mark.parametrize("bad", [_NON_ASCII_QUERY, "🔑"])
def test_gpu_page_non_ascii_query_token_is_403_not_500(api, monkeypatch, bad):
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", GPU_TOKEN, raising=False)

    resp = api["client"].get(f"/gpu/distribution?token={bad}")

    assert resp.status_code == 403
    assert api["gpu_dist"] == []


def test_gpu_page_non_ascii_header_token_is_403_not_500(api, monkeypatch):
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", GPU_TOKEN, raising=False)

    resp = api["client"].get("/gpu/distribution",
                             headers={"X-API-Token": _NON_ASCII_HEADER})

    assert resp.status_code == 403
    assert api["gpu_dist"] == []


def test_non_ascii_expected_token_also_survives(api, monkeypatch):
    """反向：配置侧写了非 ASCII token（运维手滑）也不能炸，且只有原样带对才放行。"""
    routes = api["routes"]
    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", "令牌", raising=False)

    assert api["client"].get("/api/ram/user?login_name=alice&token=xyz").status_code == 403
    assert api["ram_query"] == []
    assert api["client"].get("/api/ram/user?login_name=alice&token=令牌").status_code == 200


# ── 4. `_api_token_ok` 单元级 ────────────────────────────────────────────────

@pytest.mark.parametrize("supplied,expected,ok", [
    ("abc", "abc", True),
    ("abc", "abd", False),
    ("", "abc", False),
    ("abc", "", False),
    ("", "", False),
    ("令牌", "abc", False),
    ("abc", "令牌", False),
    ("令牌", "令牌", True),
])
def test_api_token_ok_matrix(supplied, expected, ok):
    from core.feishu_bot import routes

    with routes.app.test_request_context(f"/?token={supplied}"):
        assert routes._api_token_ok(expected) is ok


def test_api_token_ok_precedence_header_over_bearer_over_query():
    """取值优先级：`X-API-Token` > `Authorization: Bearer` > `?token=`。"""
    from core.feishu_bot import routes

    with routes.app.test_request_context(
        "/?token=q",
        headers={"X-API-Token": "h", "Authorization": "Bearer b"},
    ):
        assert routes._api_token_ok("h") is True
        assert routes._api_token_ok("b") is False
        assert routes._api_token_ok("q") is False

    with routes.app.test_request_context("/?token=q", headers={"Authorization": "Bearer b"}):
        assert routes._api_token_ok("b") is True
        assert routes._api_token_ok("q") is False


def test_authorized_helpers_read_only_their_own_setting(monkeypatch):
    """直接钉住两个 helper 的取值来源，防止有人把回退链加回去。"""
    from core.feishu_bot import routes

    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", "", raising=False)
    monkeypatch.setattr(routes.settings, "GPU_DIST_TOKEN", "", raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", WEBHOOK_TOKEN, raising=False)

    with routes.app.test_request_context(f"/?token={WEBHOOK_TOKEN}"):
        assert routes._ram_api_authorized() is False
        assert routes._gpu_dist_authorized() is False

    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", RAM_TOKEN, raising=False)
    with routes.app.test_request_context(f"/?token={RAM_TOKEN}"):
        assert routes._ram_api_authorized() is True
        assert routes._gpu_dist_authorized() is False, "GPU 页面不得回退到 RAM 接口 token"
