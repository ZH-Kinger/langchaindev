import json
from types import SimpleNamespace

import pytest


class FakeRamError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _detail(form, **extra):
    data = {
        "approval_code": "09F25B71-E1FF-434F-842F-7F2A09F35FAB",
        "instance_code": "inst_1",
        "user_id": "ou_requester",
        "form": json.dumps(form, ensure_ascii=False),
    }
    data.update(extra)
    return data


def test_parse_request_defaults_password_reset_and_mfa_off(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_ALLOWED_GROUPS_RAW",
                        "wuji_Algorithm,wuji_Examination")
    form = [
        {"id": "f1", "name": "登录名称", "value": "zhangsan"},
        {"id": "f2", "name": "显示名称", "value": "张三"},
        {"id": "f3", "name": "安全邮箱", "value": "mailto:zhangsan@example.com"},
        {"id": "f4", "name": "安全手机", "value": "13800138000"},
        {"id": "f5", "name": "登录密码", "value": "P@ssw0rd123"},
        {"id": "f6", "name": "确认登录密码", "value": "P@ssw0rd123"},
        {"id": "f7", "name": "用户组", "value": ["wuji_Algorithm", "wuji_Examination"]},
        {"id": "f8", "name": "控制台访问", "value": "是"},
        {"id": "f9", "name": "永久 AccessKey 访问", "value": "否"},
    ]

    req = ram_approval.parse_ram_account_request(_detail(form))

    assert req.login_name == "zhangsan"
    assert req.display_name == "张三"
    assert req.email == "zhangsan@example.com"
    assert req.mobile_phone == "86-13800138000"
    assert req.groups == ("wuji_Algorithm", "wuji_Examination")
    assert req.console_access is True
    assert req.permanent_access_key is False
    assert req.password_reset_required is False
    assert req.mfa_bind_required is False
    assert req.requester_open_id == "ou_requester"


def test_parse_request_password_mismatch_rejected():
    from core import ram_approval

    form = [
        {"name": "登录名称", "value": "lisi"},
        {"name": "登录密码", "value": "one"},
        {"name": "确认登录密码", "value": "two"},
        {"name": "控制台访问", "value": True},
    ]

    with pytest.raises(ram_approval.RamApprovalError, match="不一致"):
        ram_approval.parse_ram_account_request(_detail(form))


def test_parse_request_field_id_overrides_label(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_LOGIN_NAME", "login_id")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_EMAIL", "email_id")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_MOBILE", "mobile_id")
    form = [
        {"id": "login_id", "name": "不是登录名称", "value": "wangwu"},
        {"name": "登录名称", "value": "wrong"},
        {"id": "email_id", "name": "安全邮箱", "value": "wangwu@example.com"},
        {"id": "mobile_id", "name": "安全手机", "value": "13800138000"},
    ]

    req = ram_approval.parse_ram_account_request(_detail(form))

    assert req.login_name == "wangwu"


def test_parse_request_requires_email_and_mobile(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_LOGIN_NAME", "login")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_EMAIL", "email")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_MOBILE", "mobile")

    with pytest.raises(ram_approval.RamApprovalError, match="安全邮箱"):
        ram_approval.parse_ram_account_request(_detail([
            {"id": "login", "name": "登录名称", "value": "needemail"},
            {"id": "mobile", "name": "安全手机", "value": "13800138000"},
        ]))

    with pytest.raises(ram_approval.RamApprovalError, match="安全手机"):
        ram_approval.parse_ram_account_request(_detail([
            {"id": "login", "name": "登录名称", "value": "needmobile"},
            {"id": "email", "name": "安全邮箱", "value": "needmobile@example.com"},
        ]))


class FakeVerificationClient:
    def __init__(self):
        self.calls = []

    def call_api(self, params, request, runtime):
        self.calls.append((params.action, request.query))
        return {"body": {"RequestId": "req_id"}, "statusCode": 200}


class FakeRamClient:
    def __init__(self, access_keys=None):
        self.calls = []
        self.create_user_req = None
        self.login_req = None
        self.update_user_req = None
        self.group_reqs = []
        self.access_keys = access_keys or []

    def get_user(self, req):
        self.calls.append(("get_user", req.user_name))
        raise FakeRamError("EntityNotExist.User")

    def create_user(self, req):
        self.calls.append(("create_user", req.user_name))
        self.create_user_req = req

    def update_user(self, req):
        self.calls.append(("update_user", req.user_name))
        self.update_user_req = req

    def get_login_profile(self, req):
        self.calls.append(("get_login_profile", req.user_name))
        raise FakeRamError("EntityNotExist.LoginProfile")

    def create_login_profile(self, req):
        self.calls.append(("create_login_profile", req.user_name))
        self.login_req = req

    def add_user_to_group(self, req):
        self.calls.append(("add_user_to_group", req.group_name))
        self.group_reqs.append(req)

    def list_access_keys(self, req):
        self.calls.append(("list_access_keys", req.user_name))
        keys = [SimpleNamespace(access_key_id=k) for k in self.access_keys]
        return SimpleNamespace(body=SimpleNamespace(access_keys=SimpleNamespace(access_key=keys)))

    def create_access_key(self, req):
        self.calls.append(("create_access_key", req.user_name))
        return SimpleNamespace(body=SimpleNamespace(access_key=SimpleNamespace(
            access_key_id="ak_id",
            access_key_secret="ak_secret",
        )))


def test_create_ram_account_calls_expected_sdk_requests():
    from core.ram_approval import RamAccountRequest, create_ram_account

    client = FakeRamClient()
    req = RamAccountRequest(
        login_name="zhaoliu",
        display_name="赵六",
        email="zhaoliu@example.com",
        mobile_phone="86-13800138000",
        password="P@ssw0rd123",
        groups=("wuji_Algorithm",),
        console_access=True,
        permanent_access_key=True,
        password_reset_required=False,
        mfa_bind_required=False,
    )

    verification = FakeVerificationClient()
    result = create_ram_account(req, client=client, verification_client=verification)

    assert result.created_user is True
    assert result.updated_user_profile is True
    assert result.set_security_phone is True
    assert result.set_security_email is True
    assert result.created_login_profile is True
    assert result.created_access_key is True
    assert result.access_key_id == "ak_id"
    assert result.access_key_secret == "ak_secret"
    assert client.create_user_req.display_name == "赵六"
    assert client.create_user_req.email == "zhaoliu@example.com"
    assert client.create_user_req.mobile_phone == "86-13800138000"
    assert client.update_user_req.new_user_name == "zhaoliu"
    assert client.update_user_req.new_email == "zhaoliu@example.com"
    assert client.update_user_req.new_mobile_phone == "86-13800138000"
    assert [c[0] for c in verification.calls] == ["SetVerificationInfo", "SetVerificationInfo"]
    assert verification.calls[0][1]["VerifyType"] == "sms"
    assert verification.calls[0][1]["MobilePhone"] == "86-13800138000"
    assert verification.calls[1][1]["VerifyType"] == "email"
    assert verification.calls[1][1]["Email"] == "zhaoliu@example.com"
    assert client.login_req.password_reset_required is False
    assert client.login_req.mfabind_required is False
    assert [r.group_name for r in client.group_reqs] == ["wuji_Algorithm"]


def test_create_ram_account_does_not_duplicate_access_key():
    from core.ram_approval import RamAccountRequest, create_ram_account

    client = FakeRamClient(access_keys=["existing_ak"])
    req = RamAccountRequest(login_name="sunqi", permanent_access_key=True)

    result = create_ram_account(req, client=client, set_verification_info=False)

    assert result.created_access_key is False
    assert result.access_key_id == "existing_ak"
    assert ("create_access_key", "sunqi") not in client.calls

class FakeRedis:
    def __init__(self):
        self.store = {}
        self.zsets = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def delete(self, key):
        self.store.pop(key, None)
        return 1

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)


def test_redis_record_sanitizes_secrets(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    fake = FakeRedis()
    monkeypatch.setattr(ram_approval, "get_redis", lambda: fake)
    req = RamAccountRequest(
        login_name="audituser",
        display_name="Audit User",
        email="audit@example.com",
        password="secret-password",
        permanent_access_key=True,
        approval_instance_code="inst_audit",
    )

    ram_approval.save_approval_record(req, detail={"status": "APPROVED"})
    ram_approval.save_approval_result(
        req,
        RamAccountResult(
            user_name="audituser",
            created_user=True,
            created_access_key=True,
            access_key_id="ak_id",
            access_key_secret="ak_secret",
        ),
    )

    record = ram_approval.load_approval_record("inst_audit")
    blob = fake.store["ram_approval:instance:inst_audit"]
    assert record["password_set"] is True
    assert record["password_length"] == len("secret-password")
    assert record["access_key_id"] == "ak_id"
    assert record["access_key_secret_saved"] is False
    assert "secret-password" not in blob
    assert "ak_secret" not in blob


def test_handle_event_dedupes_processed_instance(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountResult

    fake = FakeRedis()
    monkeypatch.setattr(ram_approval, "get_redis", lambda: fake)
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_CODE", "approval_code_x")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_LOGIN_NAME", "login")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_EMAIL", "email")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_FIELD_MOBILE", "mobile")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False)
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DELIVERY", "email")
    monkeypatch.setattr(ram_approval, "notify_result", lambda *a, **k: None)
    monkeypatch.setattr(ram_approval, "notify_failure", lambda *a, **k: None)

    detail = _detail(
        [
            {"id": "login", "name": "login", "value": "dedupuser"},
            {"id": "email", "name": "安全邮箱", "value": "dedup@example.com"},
            {"id": "mobile", "name": "安全手机", "value": "13800138000"},
        ],
        approval_code="approval_code_x",
        instance_code="inst_dedup",
        status="APPROVED",
    )
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: detail)
    calls = []

    def fake_create(req):
        calls.append(req.login_name)
        return RamAccountResult(user_name=req.login_name, created_user=True)

    monkeypatch.setattr(ram_approval, "create_ram_account", fake_create)
    payload = {
        "header": {"event_type": "approval_instance.status_changed_v4"},
        "event": {"approval_code": "approval_code_x", "instance_code": "inst_dedup", "status": "APPROVED"},
    }

    first = ram_approval.handle_approval_event(payload)
    second = ram_approval.handle_approval_event(payload)

    assert first["ignored"] is False
    assert second == {"ignored": True, "reason": "already_processed", "instance_code": "inst_dedup"}
    assert calls == ["dedupuser"]
    assert ram_approval.load_approval_record("inst_dedup")["duplicate_count"] == 1


class ExistingUserFakeRamClient(FakeRamClient):
    def get_user(self, req):
        self.calls.append(("get_user", req.user_name))
        return SimpleNamespace(body=SimpleNamespace(user=SimpleNamespace(
            user_name=req.user_name,
            display_name="old",
            email="",
            mobile_phone="",
            comments="",
        )))

    def get_login_profile(self, req):
        self.calls.append(("get_login_profile", req.user_name))
        return SimpleNamespace(body=SimpleNamespace(login_profile=SimpleNamespace(user_name=req.user_name)))


def test_create_ram_account_checks_aliyun_before_creating_existing_user():
    from core.ram_approval import RamAccountRequest, create_ram_account

    client = ExistingUserFakeRamClient(access_keys=["existing_ak"])
    req = RamAccountRequest(
        login_name="existinguser",
        display_name="Existing User",
        email="existing@example.com",
        mobile_phone="86-13800138000",
        password="P@ssw0rd123",
        console_access=True,
        permanent_access_key=True,
    )

    verification = FakeVerificationClient()
    result = create_ram_account(req, client=client, verification_client=verification)

    assert result.created_user is False
    assert result.updated_user_profile is True
    assert result.created_login_profile is False
    assert result.created_access_key is False
    assert result.access_key_id == "existing_ak"
    assert ("create_user", "existinguser") not in client.calls
    assert ("create_login_profile", "existinguser") not in client.calls
    assert client.update_user_req.new_user_name == "existinguser"
    assert client.update_user_req.new_email == "existing@example.com"
    assert client.update_user_req.new_mobile_phone == "86-13800138000"
    assert result.set_security_phone is True
    assert result.set_security_email is True
    assert len(verification.calls) == 2
    assert ("create_access_key", "existinguser") not in client.calls

def test_build_login_url_matches_aliyun_console_format(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "ALIYUN_RAM_LOGIN_DOMAIN", "1704065796538912.onaliyun.com")

    assert ram_approval.build_login_principal("zhangxiaoxiong") == "zhangxiaoxiong@1704065796538912.onaliyun.com"
    assert ram_approval.build_login_url("zhangxiaoxiong") == (
        "https://signin.aliyun.com/login.htm?username="
        "zhangxiaoxiong%401704065796538912.onaliyun.com"
        "&defaultShowQrCode=false#/main"
    )


def test_redis_result_stores_login_url(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    fake = FakeRedis()
    monkeypatch.setattr(ram_approval, "get_redis", lambda: fake)
    monkeypatch.setattr(ram_approval.settings, "ALIYUN_RAM_LOGIN_DOMAIN", "1704065796538912.onaliyun.com")
    req = RamAccountRequest(login_name="copyuser", approval_instance_code="inst_login_url")
    result = RamAccountResult(user_name="copyuser")
    result.login_principal = ram_approval.build_login_principal("copyuser")
    result.login_url = ram_approval.build_login_url("copyuser")

    ram_approval.save_approval_result(req, result, result_status="success")

    record = ram_approval.load_approval_record("inst_login_url")
    assert record["login_principal"] == "copyuser@1704065796538912.onaliyun.com"
    assert record["login_url"].endswith("copyuser%401704065796538912.onaliyun.com&defaultShowQrCode=false#/main")


def test_account_delivery_text_contains_expected_fields(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    monkeypatch.setattr(ram_approval.settings, "ALIYUN_RAM_LOGIN_DOMAIN", "1704065796538912.onaliyun.com")
    req = RamAccountRequest(
        login_name="zhangxiaoxiong",
        email="zhang.xiaoxiong@wuji.tech",
        mobile_phone="86-13283257362",
        password="login-password",
        console_access=True,
        permanent_access_key=True,
    )
    result = RamAccountResult(
        user_name="zhangxiaoxiong",
        access_key_id="LTAI5t",
        access_key_secret="nthsl676n2G05",
    )
    result.login_principal = ram_approval.build_login_principal(req.login_name)
    result.login_url = ram_approval.build_login_url(req.login_name)

    text = ram_approval._account_delivery_text(req, result)

    assert "zhangxiaoxiong@1704065796538912.onaliyun.com" in text
    assert "login-password" in text
    assert "LTAI5t" in text
    assert "nthsl676n2G05" in text
    assert "SecurityPhoneDevice" in text
    assert "86-13283257362" in text
    assert "SecurityEmailDevice" in text
    assert "zhang.xiaoxiong@wuji.tech" in text


def test_notify_result_sends_account_email_not_private_feishu(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DELIVERY", "email")
    req = RamAccountRequest(
        login_name="mailuser",
        email="mailuser@example.com",
        mobile_phone="86-13800138000",
        password="login-password",
        console_access=True,
        permanent_access_key=True,
        requester_open_id="ou_requester",
    )
    result = RamAccountResult(
        user_name="mailuser",
        created_user=True,
        created_access_key=True,
        access_key_id="ak_id",
        access_key_secret="ak_secret",
    )
    sent_email = []
    sent_feishu = []
    monkeypatch.setattr(ram_approval, "_send_account_email", lambda req, result: sent_email.append((req.email, result.access_key_secret)))
    monkeypatch.setattr(ram_approval, "_send_text", lambda receive_id, receive_id_type, text: sent_feishu.append((receive_id, receive_id_type, text)))

    ram_approval.notify_result(req, result)

    assert sent_email == [("mailuser@example.com", "ak_secret")]
    assert sent_feishu == []


def test_notify_result_writes_approval_comment_not_group(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DELIVERY", "approval_comment")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_COMMENT_USER_ID", "ou_admin")
    req = RamAccountRequest(
        login_name="commentuser",
        email="commentuser@example.com",
        mobile_phone="86-13800138000",
        password="login-password",
        console_access=True,
        permanent_access_key=True,
        approval_instance_code="inst_comment",
    )
    result = RamAccountResult(
        user_name="commentuser",
        created_user=True,
        created_access_key=True,
        access_key_id="ak_id",
        access_key_secret="ak_secret",
    )
    sent_comments = []
    sent_feishu = []
    monkeypatch.setattr(
        ram_approval,
        "_send_approval_comment",
        lambda instance_code, text, user_id, comment_id="": sent_comments.append(
            (instance_code, text, user_id, comment_id)
        ) or "comment_1",
    )
    monkeypatch.setattr(ram_approval, "_send_text", lambda receive_id, receive_id_type, text: sent_feishu.append(text))

    ram_approval.notify_result(req, result, approval_comment_id="comment_0")

    assert sent_comments
    assert sent_comments[0][0] == "inst_comment"
    assert sent_comments[0][2] == "ou_admin"
    assert sent_comments[0][3] == "comment_0"
    assert "ak_secret" in sent_comments[0][1]
    assert "login-password" in sent_comments[0][1]
    assert sent_feishu == []

def test_delivery_ready_requires_smtp_before_secret_creation(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False)
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DELIVERY", "email")
    monkeypatch.setattr(ram_approval.settings, "SMTP_HOST", "")
    monkeypatch.setattr(ram_approval.settings, "SMTP_FROM", "")
    monkeypatch.setattr(ram_approval.settings, "SMTP_USERNAME", "")
    monkeypatch.setattr(ram_approval.settings, "SMTP_PASSWORD", "")
    monkeypatch.setattr(ram_approval.settings, "SMTP_AUTH_REQUIRED", True)
    req = RamAccountRequest(
        login_name="needsmtp",
        email="needsmtp@example.com",
        mobile_phone="86-13800138000",
        password="P@ssw0rd123",
        console_access=True,
        permanent_access_key=True,
    )

    with pytest.raises(ram_approval.RamApprovalError, match="SMTP_HOST"):
        ram_approval._assert_account_delivery_ready(req)




def test_parse_request_platforms_multi_select():
    from core import ram_approval

    form = [
        {"name": "\u767b\u5f55\u540d\u79f0", "value": "dualuser"},
        {"name": "\u5b89\u5168\u90ae\u7bb1", "value": "dual@example.com"},
        {"name": "\u5b89\u5168\u624b\u673a", "value": "13800138000"},
        {"name": "\u5e73\u53f0", "value": ["\u963f\u91cc\u4e91RAM", "\u706b\u5c71\u5f15\u64ceIAM"]},
    ]

    req = ram_approval.parse_ram_account_request(_detail(form))

    assert req.platforms == (ram_approval.PLATFORM_ALIYUN_RAM, ram_approval.PLATFORM_VOLCANO_IAM)


def test_parse_request_single_platform_group_labels_trim_to_real_names(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "FEISHU_VOLCANO_IAM_ALLOWED_GROUPS_RAW", "", raising=False)
    form = [
        {"name": "\u767b\u5f55\u540d\u79f0", "value": "volcsingle"},
        {"name": "\u5b89\u5168\u90ae\u7bb1", "value": "volcsingle@example.com"},
        {"name": "\u5b89\u5168\u624b\u673a", "value": "13800138000"},
        {"name": "\u5e73\u53f0", "value": "\u706b\u5c71\u5f15\u64ceIAM"},
        {"name": "\u7528\u6237\u7ec4", "value": ["wuji_group(\u7b97\u6cd5\u7ec4)", "wuji-opration(\u7ba1\u7406\u5458)"]},
    ]

    req = ram_approval.parse_ram_account_request(_detail(form))

    assert req.platforms == (ram_approval.PLATFORM_VOLCANO_IAM,)
    assert req.groups == ("wuji_group", "wuji-opration")
    assert ram_approval._groups_for_platform(req, ram_approval.PLATFORM_VOLCANO_IAM) == (
        "wuji_group",
        "wuji-opration",
    )


def test_parse_request_aliyun_group_labels_trim_to_real_names(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_ALLOWED_GROUPS_RAW", "wuji_Algorithm,wuji_Examination")
    form = [
        {"name": "\u767b\u5f55\u540d\u79f0", "value": "alisingle"},
        {"name": "\u5b89\u5168\u90ae\u7bb1", "value": "alisingle@example.com"},
        {"name": "\u5b89\u5168\u624b\u673a", "value": "13800138000"},
        {"name": "\u5e73\u53f0", "value": "\u963f\u91cc\u4e91RAM"},
        {"name": "\u7528\u6237\u7ec4", "value": ["wuji_Algorithm(\u7b97\u6cd5)", "wuji_Examination(\u6d4b\u8bd5)"]},
    ]

    req = ram_approval.parse_ram_account_request(_detail(form))

    assert req.platforms == (ram_approval.PLATFORM_ALIYUN_RAM,)
    assert req.groups == ("wuji_Algorithm", "wuji_Examination")


def test_parse_request_platform_specific_groups(monkeypatch):
    from core import ram_approval

    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_ALLOWED_GROUPS_RAW", "ali_group")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_VOLCANO_IAM_ALLOWED_GROUPS_RAW", "volc_group", raising=False)
    form = [
        {"name": "\u767b\u5f55\u540d\u79f0", "value": "dualgroup"},
        {"name": "\u5b89\u5168\u90ae\u7bb1", "value": "dualgroup@example.com"},
        {"name": "\u5b89\u5168\u624b\u673a", "value": "13800138000"},
        {"name": "\u5e73\u53f0", "value": ["\u963f\u91cc\u4e91RAM", "\u706b\u5c71\u5f15\u64ceIAM"]},
        {"name": "\u963f\u91cc\u4e91RAM\u7528\u6237\u7ec4", "value": ["ali_group"]},
        {"name": "\u706b\u5c71\u5f15\u64ceIAM\u7528\u6237\u7ec4", "value": ["volc_group"]},
    ]

    req = ram_approval.parse_ram_account_request(_detail(form))

    assert req.aliyun_groups == ("ali_group",)
    assert req.volcano_groups == ("volc_group",)
    assert ram_approval._groups_for_platform(req, ram_approval.PLATFORM_ALIYUN_RAM) == ("ali_group",)
    assert ram_approval._groups_for_platform(req, ram_approval.PLATFORM_VOLCANO_IAM) == ("volc_group",)


def test_create_accounts_for_platforms_uses_platform_specific_groups(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    calls = []

    def fake_ram(platform_req):
        calls.append((ram_approval.PLATFORM_ALIYUN_RAM, platform_req.groups, platform_req.platforms))
        return RamAccountResult(
            user_name=platform_req.login_name,
            platform=ram_approval.PLATFORM_ALIYUN_RAM,
            platform_label=ram_approval.PLATFORM_LABELS[ram_approval.PLATFORM_ALIYUN_RAM],
            added_groups=list(platform_req.groups),
        )

    def fake_volcano(platform_req):
        calls.append((ram_approval.PLATFORM_VOLCANO_IAM, platform_req.groups, platform_req.platforms))
        return RamAccountResult(
            user_name=platform_req.login_name,
            platform=ram_approval.PLATFORM_VOLCANO_IAM,
            platform_label=ram_approval.PLATFORM_LABELS[ram_approval.PLATFORM_VOLCANO_IAM],
            added_groups=list(platform_req.groups),
        )

    monkeypatch.setattr(ram_approval, "create_ram_account", fake_ram)
    monkeypatch.setattr(ram_approval, "create_volcano_iam_account", fake_volcano)
    req = RamAccountRequest(
        login_name="multiuser",
        email="multi@example.com",
        mobile_phone="86-13800138000",
        groups=("fallback_group",),
        aliyun_groups=("ali_group",),
        volcano_groups=("volc_group",),
        platforms=(ram_approval.PLATFORM_ALIYUN_RAM, ram_approval.PLATFORM_VOLCANO_IAM),
    )

    result = ram_approval.create_accounts_for_platforms(req)

    assert calls == [
        (ram_approval.PLATFORM_ALIYUN_RAM, ("ali_group",), (ram_approval.PLATFORM_ALIYUN_RAM,)),
        (ram_approval.PLATFORM_VOLCANO_IAM, ("volc_group",), (ram_approval.PLATFORM_VOLCANO_IAM,)),
    ]
    assert result.added_groups == [
        f"{ram_approval.PLATFORM_LABELS[ram_approval.PLATFORM_ALIYUN_RAM]}:ali_group",
        f"{ram_approval.PLATFORM_LABELS[ram_approval.PLATFORM_VOLCANO_IAM]}:volc_group",
    ]


class FakeVolcanoError(Exception):
    pass


class FakeVolcanoIamClient:
    def __init__(self, access_keys=None):
        self.calls = []
        self.create_user_req = None
        self.update_user_req = None
        self.login_req = None
        self.group_reqs = []
        self.access_keys = access_keys or []

    def get_user(self, req):
        self.calls.append(("get_user", req.user_name))
        raise FakeVolcanoError("NotFound.User")

    def create_user(self, req):
        self.calls.append(("create_user", req.user_name))
        self.create_user_req = req
        return SimpleNamespace(user=SimpleNamespace(account_id="volc-account", user_name=req.user_name))

    def update_user(self, req):
        self.calls.append(("update_user", req.user_name))
        self.update_user_req = req

    def get_login_profile(self, req):
        self.calls.append(("get_login_profile", req.user_name))
        raise FakeVolcanoError("NotFound.LoginProfile")

    def create_login_profile(self, req):
        self.calls.append(("create_login_profile", req.user_name))
        self.login_req = req

    def add_user_to_group(self, req):
        self.calls.append(("add_user_to_group", req.user_group_name))
        self.group_reqs.append(req)

    def list_access_keys(self, req):
        self.calls.append(("list_access_keys", req.user_name))
        keys = [SimpleNamespace(access_key_id=k) for k in self.access_keys]
        return SimpleNamespace(access_key_metadata=keys)

    def create_access_key(self, req):
        self.calls.append(("create_access_key", req.user_name))
        return SimpleNamespace(access_key=SimpleNamespace(
            access_key_id="volc_ak",
            secret_access_key="volc_sk",
        ))


def test_create_volcano_iam_account_calls_expected_sdk_requests(monkeypatch):
    from core.ram_approval import RamAccountRequest, create_volcano_iam_account

    client = FakeVolcanoIamClient()
    req = RamAccountRequest(
        login_name="volcuser",
        display_name="Volc User",
        email="volc@example.com",
        mobile_phone="86-13800138000",
        password="P@ssw0rd123",
        groups=("wuji_Algorithm",),
        platforms=("volcano_iam",),
        console_access=True,
        permanent_access_key=True,
        password_reset_required=False,
    )

    result = create_volcano_iam_account(req, client=client)

    assert result.platform == "volcano_iam"
    assert result.created_user is True
    assert result.updated_user_profile is True
    assert result.created_login_profile is True
    assert result.created_access_key is True
    assert result.access_key_id == "volc_ak"
    assert result.access_key_secret == "volc_sk"
    assert result.account_id == "volc-account"
    assert client.create_user_req.display_name == "Volc User"
    assert client.create_user_req.email == "volc@example.com"
    assert client.create_user_req.mobile_phone == "13800138000"      # 火山去掉 86- 前缀
    assert client.update_user_req.new_user_name == "volcuser"
    assert client.login_req.login_allowed is True
    assert client.login_req.password_reset_required is False
    assert [r.user_group_name for r in client.group_reqs] == ["wuji_Algorithm"]


def test_create_accounts_for_platforms_aggregates_results(monkeypatch):
    from core import ram_approval
    from core.ram_approval import RamAccountRequest, RamAccountResult

    req = RamAccountRequest(
        login_name="multiuser",
        email="multi@example.com",
        mobile_phone="86-13800138000",
        platforms=(ram_approval.PLATFORM_ALIYUN_RAM, ram_approval.PLATFORM_VOLCANO_IAM),
    )
    monkeypatch.setattr(
        ram_approval,
        "create_ram_account",
        lambda req: RamAccountResult(
            user_name=req.login_name,
            platform=ram_approval.PLATFORM_ALIYUN_RAM,
            platform_label=ram_approval.PLATFORM_LABELS[ram_approval.PLATFORM_ALIYUN_RAM],
            access_key_id="ali_ak",
        ),
    )
    monkeypatch.setattr(
        ram_approval,
        "create_volcano_iam_account",
        lambda req: RamAccountResult(
            user_name=req.login_name,
            platform=ram_approval.PLATFORM_VOLCANO_IAM,
            platform_label=ram_approval.PLATFORM_LABELS[ram_approval.PLATFORM_VOLCANO_IAM],
            access_key_id="volc_ak",
        ),
    )

    result = ram_approval.create_accounts_for_platforms(req)
    text = ram_approval._account_delivery_text(req, result)

    assert result.platform == "multi"
    assert [r.platform for r in result.platform_results] == ["aliyun_ram", "volcano_iam"]
    assert "ali_ak" in text
    assert "volc_ak" in text


def test_volcano_not_exist_recognized_from_404_body():
    """火山 get_user 404 UserNotExist 应被识别为“用户不存在”，走创建而非报错。"""
    from core import ram_approval as ra

    class ApiExc(Exception):
        status = 404
        reason = "404 Page not found"
        body = ('{"ResponseMetadata":{"Error":{"Code":"UserNotExist",'
                '"Message":"User \'test\' does not exist."}}}')
        def __str__(self):
            return f"(404) Reason: 404 Page not found HTTP response body: {self.body}"

    e = ApiExc()
    assert ra._volcano_is_not_exist(e) is True
    assert ra._volcano_is_already_exists(e) is False
    assert "UserNotExist" == ra._volcano_err_code(e)


def test_volcano_mobile_strips_aliyun_prefix():
    from core.ram_approval import _volcano_mobile
    assert _volcano_mobile("86-13100000000") == "13100000000"
    assert _volcano_mobile("13100000000") == "13100000000"
    assert _volcano_mobile("+86 131 0000 0000") == "13100000000"
    assert _volcano_mobile("") == ""
