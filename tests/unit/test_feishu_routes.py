"""routes.py 拆分后路由冒烟：challenge 回显 + 包再导出兼容。

/health 不在此测（探测依赖会发真网络请求，属集成测试范畴）。
"""


def _client():
    from core.feishu_bot import app
    return app.test_client()


def test_event_url_verification_echo():
    resp = _client().post("/feishu/event", json={"type": "url_verification", "challenge": "x-123"})
    assert resp.status_code == 200
    assert resp.get_json() == {"challenge": "x-123"}


def test_card_action_get_challenge_echo():
    resp = _client().get("/feishu/card_action?challenge=y-456")
    assert resp.status_code == 200
    assert resp.get_json() == {"challenge": "y-456"}


def test_card_action_post_url_verification_echo():
    resp = _client().post("/feishu/card_action", json={"type": "url_verification", "challenge": "z-789"})
    assert resp.status_code == 200
    assert resp.get_json() == {"challenge": "z-789"}


def test_package_reexports():
    import core.feishu_bot as fb
    # main.py / dsw_scheduler / 测试依赖的硬入口
    assert callable(fb.run)
    assert callable(fb._is_registered)
    assert callable(fb._process_action)
    assert fb.app is fb.routes.app
