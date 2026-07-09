"""#46 B3/C7：notify._get_access_token 模块级缓存（到期前 5min 刷新 + 双检锁）。

修复「每次都取 token、无缓存」——早报 N 用户=N 次取 token，端点有限流、历史踩过坑。
"""
import pytest


class _Resp:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self): pass
    def json(self): return self._data


@pytest.fixture
def token_env(monkeypatch):
    """清缓存 + mock requests.post 计次，返回计数器与响应控制。"""
    from tools.feishu import notify
    # 重置模块级缓存（跨测试污染）
    notify._token_cache["token"] = ""
    notify._token_cache["expire_ts"] = 0.0

    ctrl = {"count": 0, "token": "TOK1", "expire": 7200, "code": 0}

    def _post(url, json=None, timeout=None):
        ctrl["count"] += 1
        return _Resp({"code": ctrl["code"], "app_access_token": ctrl["token"],
                      "expire": ctrl["expire"], "msg": "err"})

    monkeypatch.setattr(notify.requests, "post", _post)
    yield notify, ctrl
    notify._token_cache["token"] = ""
    notify._token_cache["expire_ts"] = 0.0


def test_first_fetch_then_cache_hit(token_env):
    notify, ctrl = token_env
    assert notify._get_access_token() == "TOK1"
    assert ctrl["count"] == 1
    # 第二次命中缓存，不再 HTTP
    assert notify._get_access_token() == "TOK1"
    assert ctrl["count"] == 1, "缓存命中不应再取 token"


def test_refetch_after_expiry(token_env, monkeypatch):
    notify, ctrl = token_env
    assert notify._get_access_token() == "TOK1"
    assert ctrl["count"] == 1

    # 把缓存过期时间挪到过去 → 视为过期，重取
    notify._token_cache["expire_ts"] = 0.0
    ctrl["token"] = "TOK2"
    assert notify._get_access_token() == "TOK2"
    assert ctrl["count"] == 2


def test_expiry_uses_early_refresh_margin(token_env):
    """expire=7200 → expire_ts 提前至少 5min（<= now+6900），防边界拿到即将失效 token。"""
    import time
    notify, ctrl = token_env
    ctrl["expire"] = 7200
    notify._get_access_token()
    assert notify._token_cache["expire_ts"] <= time.time() + (7200 - 300) + 1


def test_error_code_raises(token_env):
    notify, ctrl = token_env
    ctrl["code"] = 99
    with pytest.raises(RuntimeError):
        notify._get_access_token()
    # 失败不污染缓存
    assert notify._token_cache["token"] == ""
