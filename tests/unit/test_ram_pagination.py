"""#46 C6：list_ram_users_api marker 分页——子用户 >100 不再截断当全量。

RAM ListUsers 单页上限 100，超过被 is_truncated 截断。用 marker 循环取全量。
"""
from types import SimpleNamespace

import pytest


def _resp(users, truncated=False, marker=None):
    return SimpleNamespace(body=SimpleNamespace(
        users=SimpleNamespace(user=[
            SimpleNamespace(user_id=f"id{u}", user_name=u, display_name=u.upper()) for u in users
        ]),
        is_truncated=truncated,
        marker=marker,
    ))


class _FakeClient:
    def __init__(self, cfg): pass


@pytest.fixture
def patch_client(monkeypatch):
    """把 alibabacloud_ram20150501.client.Client 换掉；返回可编排 list_users 行为的挂载点。"""
    import alibabacloud_ram20150501.client as ram_client_mod
    state = {"pages": [], "calls": []}

    class _C(_FakeClient):
        def list_users(self, req):
            state["calls"].append(getattr(req, "marker", None))
            return state["pages"].pop(0)

    monkeypatch.setattr(ram_client_mod, "Client", _C)
    return state


def test_two_pages_merged(patch_client):
    from tools.aliyun import ram
    patch_client["pages"] = [
        _resp(["alice", "bob"], truncated=True, marker="M1"),
        _resp(["carol"], truncated=False, marker=None),
    ]
    out = ram.list_ram_users_api()
    names = [u["user_name"] for u in out]
    assert names == ["alice", "bob", "carol"]        # 两页合并全量
    # 第一页不带 marker，第二页带上一页返回的 marker
    assert patch_client["calls"] == [None, "M1"]


def test_single_page_one_call(patch_client):
    from tools.aliyun import ram
    patch_client["pages"] = [_resp(["alice"], truncated=False)]
    out = ram.list_ram_users_api()
    assert [u["user_name"] for u in out] == ["alice"]
    assert len(patch_client["calls"]) == 1           # 未截断 → 只一次调用


def test_truncated_but_no_marker_stops(patch_client):
    """is_truncated=True 但没给 marker → 不死循环，停在当前页。"""
    from tools.aliyun import ram
    patch_client["pages"] = [_resp(["alice"], truncated=True, marker=None)]
    out = ram.list_ram_users_api()
    assert [u["user_name"] for u in out] == ["alice"]
    assert len(patch_client["calls"]) == 1


def test_exception_returns_empty(monkeypatch):
    import alibabacloud_ram20150501.client as ram_client_mod

    class _Boom(_FakeClient):
        def list_users(self, req):
            raise RuntimeError("ram down")

    monkeypatch.setattr(ram_client_mod, "Client", _Boom)
    from tools.aliyun import ram
    assert ram.list_ram_users_api() == []
