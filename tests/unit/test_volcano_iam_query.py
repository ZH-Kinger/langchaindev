"""火山引擎 IAM 账户查询：后端 / 卡片 / 意图 / handler。"""
import pytest

from core import volcano_iam_query as vq
from core import volcano_iam_query_cards as vcards
from core.feishu_bot import messages, actions


# ── 假 SDK ────────────────────────────────────────────────────────────────
class _User:
    user_name = "zhangxiaoxiong"; display_name = "张小熊"; email = "z@x.com"
    mobile_phone = "13100000000"; account_id = "2100000001"; description = ""
    create_date = "2026-06-01"; update_date = ""


class _Ak:
    access_key_id = "AKLTxxxx"; status = "active"; create_date = "2026-06-02"


class FakeModels:
    class GetUserRequest:
        def __init__(self, user_name): self.user_name = user_name
    class ListAccessKeysRequest:
        def __init__(self, user_name): pass
    class GetLoginProfileRequest:
        def __init__(self, user_name): pass
    class ListGroupsForUserRequest:
        def __init__(self, user_name): pass


class FakeClient:
    def __init__(self, exists=True): self.exists = exists
    def get_user(self, req):
        if not self.exists:
            raise RuntimeError("UserNotFound: no such user")
        class R: user = _User()
        return R()
    def list_access_keys(self, req):
        class R: access_key_metadata = [_Ak()]
        return R()
    def get_login_profile(self, req): return object()
    def list_groups_for_user(self, req):
        class G: user_group_name = "wuji_group"
        class R: user_groups = [G()]
        return R()


def test_query_returns_info_no_secret():
    data = vq.query_volcano_iam_account("zhangxiaoxiong", client=FakeClient(), models=FakeModels())
    assert data["user_name"] == "zhangxiaoxiong" and data["account_id"] == "2100000001"
    assert data["console_access"] is True
    assert data["access_key_count"] == 1 and data["access_keys"][0]["access_key_id"] == "AKLTxxxx"
    assert data["groups"] == ["wuji_group"]
    # 不含任何密钥字段
    assert "secret" not in str(data).lower()


def test_query_not_found():
    with pytest.raises(vq.VolcanoUserNotFound):
        vq.query_volcano_iam_account("ghost", client=FakeClient(exists=False), models=FakeModels())


def test_query_missing_name():
    with pytest.raises(vq.VolcanoQueryError):
        vq.query_volcano_iam_account("  ")


def test_intent_and_cards():
    assert messages._is_volcano_account_query_entry_intent("火山引擎IAM")
    assert messages._is_volcano_account_query_entry_intent("火山引擎 iam")
    ec = vcards.query_entry_card()
    form = ec["body"]["elements"][1]
    assert any(e.get("name") == "user_name" for e in form["elements"])
    # 结果卡 / 未找到 / 错误卡
    assert vcards.query_result_card({"user_name": "z", "account_id": "a"})["header"]["template"] == "green"
    assert vcards.query_result_card(None, requested={"user_name": "z"})["header"]["template"] == "orange"
    assert vcards.query_error_card("z", "boom")["header"]["template"] == "red"


def test_submit_handler(monkeypatch):
    monkeypatch.setattr(vq, "query_volcano_iam_account",
                        lambda name, **k: {"user_name": name, "account_id": "acc", "access_keys": [],
                                           "groups": [], "console_access": False, "access_key_count": 0})
    out = actions._h_submit_volcano_query({}, "ou_1", "chat", {"user_name": "zhangxiaoxiong"})
    assert out["toast"]["type"] == "success"
    assert out["card"]["data"]["header"]["template"] == "green"


def test_submit_handler_requires_name():
    out = actions._h_submit_volcano_query({}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "error"


def test_registered():
    assert "submit_volcano_query" in actions._ACTION_HANDLERS
    assert "open_volcano_query" in actions._ACTION_HANDLERS
