"""ECS / OSS / SLS 工具行为测试：confirm 拦截 + 凭证缺失降级。"""
import pytest


# ── ECS ──────────────────────────────────────────────────────────────────────

class _FakeClient:
    """空 stub，仅用于让 client is None 分支不触发；不应被实际调用。"""
    def __getattr__(self, name):
        def _should_not_be_called(*a, **k):
            raise AssertionError(f"client.{name} 被调用 — 但本测试 confirm 应在前拦截")
        return _should_not_be_called


@pytest.fixture
def ecs_with_fake_client(monkeypatch):
    monkeypatch.setattr("tools.aliyun.ecs.get_ecs_client", lambda **kw: _FakeClient())


@pytest.mark.parametrize("action", ["create", "start", "stop", "reboot", "delete"])
def test_ecs_write_requires_confirm(ecs_with_fake_client, action):
    from tools.aliyun.ecs import manage_ecs
    result = manage_ecs(action=action, instance_id="i-test", confirm=False, open_id="ou_x")
    assert "confirm=true" in result
    assert "写操作" in result or "请明确" in result


def test_ecs_no_client_returns_friendly_error(monkeypatch):
    from tools.aliyun.ecs import manage_ecs
    monkeypatch.setattr("tools.aliyun.ecs.get_ecs_client", lambda **kw: None)
    result = manage_ecs(action="list", open_id="ou_x")
    assert "凭证不可用" in result or "❌" in result


def test_ecs_unknown_action(monkeypatch):
    from tools.aliyun.ecs import manage_ecs
    monkeypatch.setattr("tools.aliyun.ecs.get_ecs_client", lambda **kw: _FakeClient())
    result = manage_ecs(action="unknown_xyz", open_id="ou_x")
    assert "未知 action" in result


def test_ecs_create_without_required_fields(ecs_with_fake_client):
    from tools.aliyun.ecs import manage_ecs
    # confirm=true 但缺 instance_type/image_id 等
    result = manage_ecs(action="create", confirm=True, open_id="ou_x")
    assert "缺少必填参数" in result


# ── OSS ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def oss_with_fake_clients(monkeypatch):
    monkeypatch.setattr("tools.aliyun.oss.get_oss_service", lambda **kw: _FakeClient())
    monkeypatch.setattr("tools.aliyun.oss.get_oss_bucket", lambda *a, **kw: _FakeClient())


@pytest.mark.parametrize("action", ["create_bucket", "put_object",
                                      "delete_object", "delete_bucket"])
def test_oss_write_requires_confirm(oss_with_fake_clients, action):
    from tools.aliyun.oss import manage_oss
    result = manage_oss(action=action, bucket="b", object_key="k",
                         content="c", confirm=False, open_id="ou_x")
    assert "confirm=true" in result


def test_oss_no_service_returns_friendly_error(monkeypatch):
    from tools.aliyun.oss import manage_oss
    monkeypatch.setattr("tools.aliyun.oss.get_oss_service", lambda **kw: None)
    result = manage_oss(action="list_buckets", open_id="ou_x")
    assert "❌" in result


def test_oss_unknown_action(oss_with_fake_clients):
    from tools.aliyun.oss import manage_oss
    result = manage_oss(action="strange_action", open_id="ou_x")
    assert "未知 action" in result


def test_oss_list_objects_missing_bucket(oss_with_fake_clients):
    from tools.aliyun.oss import manage_oss
    result = manage_oss(action="list_objects", open_id="ou_x")
    assert "需要提供 bucket" in result


# ── SLS ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sls_with_fake_client(monkeypatch):
    monkeypatch.setattr("tools.aliyun.sls.get_sls_client", lambda **kw: _FakeClient())


@pytest.mark.parametrize("action", ["create_project", "create_logstore"])
def test_sls_write_requires_confirm(sls_with_fake_client, action):
    from tools.aliyun.sls import manage_sls
    result = manage_sls(action=action, project="p", logstore="l",
                          confirm=False, open_id="ou_x")
    assert "confirm=true" in result


def test_sls_no_client_returns_friendly_error(monkeypatch):
    from tools.aliyun.sls import manage_sls
    monkeypatch.setattr("tools.aliyun.sls.get_sls_client", lambda **kw: None)
    result = manage_sls(action="list_projects", open_id="ou_x")
    assert "凭证不可用" in result or "❌" in result


def test_sls_unknown_action(sls_with_fake_client):
    from tools.aliyun.sls import manage_sls
    result = manage_sls(action="???", open_id="ou_x")
    assert "未知 action" in result


def test_sls_query_missing_required_fields(sls_with_fake_client):
    from tools.aliyun.sls import manage_sls
    result = manage_sls(action="query", project="", logstore="", open_id="ou_x")
    assert "需要提供" in result