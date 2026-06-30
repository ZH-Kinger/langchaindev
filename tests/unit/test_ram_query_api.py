from types import SimpleNamespace


def test_normalize_login_name_from_principal_and_url(monkeypatch):
    from core import ram_query

    monkeypatch.setattr(ram_query.settings, "ALIYUN_RAM_LOGIN_DOMAIN", "1704065796538912.onaliyun.com")

    assert ram_query.normalize_login_name("alice@1704065796538912.onaliyun.com") == "alice"
    assert ram_query.normalize_login_name(
        "https://signin.aliyun.com/login.htm?username=bob%401704065796538912.onaliyun.com#/main"
    ) == "bob"


def test_normalize_login_name_strips_onaliyun_domain_without_config(monkeypatch):
    from core import ram_query

    monkeypatch.setattr(ram_query.settings, "ALIYUN_RAM_LOGIN_DOMAIN", "")

    assert ram_query.normalize_login_name("alice@1704065796538912.onaliyun.com") == "alice"

class _FakeResp:
    def __init__(self, body):
        self.body = body


class _FakeRamClient:
    def get_user(self, request):
        return _FakeResp(SimpleNamespace(user=SimpleNamespace(
            user_id="123",
            user_name=request.user_name,
            display_name="Alice",
            email="alice@example.com",
            mobile_phone="86-13800138000",
            comments="",
            create_date="2026-01-01T00:00:00Z",
            update_date="2026-01-02T00:00:00Z",
            last_login_date="",
        )))

    def list_groups_for_user(self, request):
        return _FakeResp(SimpleNamespace(groups=SimpleNamespace(group=[SimpleNamespace(group_name="wuji_Algorithm")])))

    def get_login_profile(self, request):
        return _FakeResp(SimpleNamespace(login_profile=SimpleNamespace()))

    def list_access_keys(self, request):
        return _FakeResp(SimpleNamespace(access_keys=SimpleNamespace(access_key=[
            SimpleNamespace(access_key_id="ak1", status="Active", create_date="2026-01-03T00:00:00Z")
        ])))


def test_query_ram_account_returns_non_secret_fields(monkeypatch):
    from core import ram_query

    monkeypatch.setattr(ram_query.settings, "ALIYUN_RAM_LOGIN_DOMAIN", "example.onaliyun.com")
    user = ram_query.query_ram_account("alice@example.onaliyun.com", client=_FakeRamClient())

    assert user["user_name"] == "alice"
    assert user["login_principal"] == "alice@example.onaliyun.com"
    assert user["groups"] == ["wuji_Algorithm"]
    assert user["console_access"] is True
    assert user["access_keys"] == [{"access_key_id": "ak1", "status": "Active", "create_date": "2026-01-03T00:00:00Z"}]
    assert "access_key_secret" not in str(user).lower()


def test_ram_user_api_requires_token(monkeypatch):
    from core.feishu_bot.routes import app, settings

    monkeypatch.setattr(settings, "RAM_QUERY_API_TOKEN", "secret-token")
    resp = app.test_client().get("/api/ram/user?login_name=alice")

    assert resp.status_code == 403


def test_ram_user_api_returns_query_result(monkeypatch):
    from core.feishu_bot import routes

    monkeypatch.setattr(routes.settings, "RAM_QUERY_API_TOKEN", "secret-token")
    monkeypatch.setattr(
        "core.ram_query.query_ram_account",
        lambda login_name: {"user_name": login_name, "access_keys": []},
    )

    resp = routes.app.test_client().get(
        "/api/ram/user?login_name=alice",
        headers={"X-API-Token": "secret-token"},
    )

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "exists": True, "user": {"user_name": "alice", "access_keys": []}}