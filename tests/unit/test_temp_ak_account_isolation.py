"""#63 第二阿里云主账号 —— **数据隔离对抗用例**（最重要的一组）。

不变量（任一条破了都是「拿 A 账号的凭证去动 B 账号」级别的事故）：
  I1  两账号的 grant 落在各自 Redis 命名空间，互相读不到。
  I2  到期清理（sweep_expired）逐档扫、每个 grant 用**自己账号**的 AK 删号，绝不串。
  I3  撤销（revoke_grant）按 grant["account"] 取 AK；档案下线时**炸**，不退默认档。
  I4  共用的延长/撤销审批按凭证ID 前缀分派回正确账号的 grant。
  I5  桶映射按账号取：同一个展示桶名在两账号映射到不同真实桶时不串。
  I6  `permsync.make_ram_client(ak, sk)` 显式传参时**不读** env `ALIBABA_CLOUD_*`（命门：
      运维一旦设了那对 env，零参路径会把所有账号的建号请求劫持到同一个账号）。
  I7  默认档（现有账号）行为与多账号化前逐字一致。

另含 1949 档表单解析（无「平台」控件恒 aliyun、多一个「备注」）。
"""
import time

import pytest

from config.settings import settings
from core import ram_approval
from core.temp_ak_issuance import accounts, approval, cleanup, issuer
from core.temp_ak_issuance import orchestrator as o

DEF_CODE = "5B4A3105-1EF9-4645-99D2-CCF69FE75D06"
CODE_1949 = "0133C4FC-8793-4FF3-A759-C4ECE8AC1FF9"
EXT_CODE = "E9333E62-37D3-4644-9117-C994D6035EFD"      # 两账号**共用**的延长/撤销审批


@pytest.fixture(autouse=True)
def _both_accounts(monkeypatch):
    """两账号都注册（隔离用例的前提）。默认档与 1949 档各自一套 AK/桶表。"""
    monkeypatch.setattr(settings, "TEMP_AK_APPROVAL_CODE", DEF_CODE)
    monkeypatch.setattr(settings, "TEMP_AK_EXTEND_APPROVAL_CODE", EXT_CODE)
    monkeypatch.setattr(settings, "ALIYUN_ACCESS_KEY_ID", "LTAI_DEFAULT")
    monkeypatch.setattr(settings, "ALIYUN_ACCESS_KEY_SECRET", "SK_DEFAULT")
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    monkeypatch.setattr(settings, "TEMP_AK_1949_APPROVAL_CODE", CODE_1949)
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_ID", "LTAI_1949")
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "SK_1949")
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW", "{}")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "")
    monkeypatch.setattr(settings, "TEMP_AK_1949_COMMENT_USER_ID", "")


def _p1949():
    return accounts.by_slug("1949")


def _deregister_1949(monkeypatch):
    """把 1949 档彻底下线 = **三项配置全空**（dev 已改成「任一配置存在即注册」，
    只抹 AK 只会让档案变成「注册了但建不了 client」，不是下线）。"""
    monkeypatch.setattr(settings, "TEMP_AK_1949_APPROVAL_CODE", "")
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "")


def _strip_1949_ak(monkeypatch):
    """只抹 AK：档案仍注册（命名空间保住、扫得到），但建号/删号必须显式失败。"""
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "")


def _spec(**over):
    now = time.time()
    base = {
        "platform": "aliyun",
        "enterprise": "外采公司A",
        "bucket": "wuji-sing",
        "prefix": "team/data/",
        "caps": ["read", "download"],
        "not_before": now,
        "expire": now + 5 * 86400,       # >12h → 方案 B（ram）
        "recipient_email": "",
        "source_ips": [],
        "reason": "",
    }
    base.update(over)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# I1 Redis 命名空间隔离
# ══════════════════════════════════════════════════════════════════════════════

def test_two_accounts_grants_land_in_own_namespaces(fake_redis):
    gd = o.create_grant_record(_spec(), instance_code="inst_D")
    g9 = o.create_grant_record(_spec(enterprise="张三"), instance_code="inst_9", profile=_p1949())

    assert gd["grant_id"].startswith("tak-") and gd["account"] == ""
    assert g9["grant_id"].startswith("tak1949-") and g9["account"] == "1949"

    keys = set(fake_redis.keys("*grant:*"))
    assert "temp_ak:grant:" + gd["grant_id"] in keys
    assert "temp_ak_1949:grant:" + g9["grant_id"] in keys
    # 反向：1949 的记录绝不能出现在默认命名空间（否则默认档的 sweep 会拿错 AK 去删它）
    assert "temp_ak:grant:" + g9["grant_id"] not in keys
    assert "temp_ak_1949:grant:" + gd["grant_id"] not in keys


def test_get_grant_reads_back_per_account(fake_redis):
    gd = o.create_grant_record(_spec(), instance_code="inst_D")
    g9 = o.create_grant_record(_spec(enterprise="张三"), instance_code="inst_9", profile=_p1949())
    assert o.get_grant(gd["grant_id"])["enterprise"] == "外采公司A"
    assert o.get_grant(g9["grant_id"])["enterprise"] == "张三"
    assert o.get_grant(g9["grant_id"])["account"] == "1949"


def test_same_instance_code_yields_two_independent_grants(fake_redis):
    """极端：同一个 instance_code（不同审批模板下可能巧合）在两档下是**两条独立记录**，
    不会互相幂等短路成一条（短路会让第二个账号拿到第一个账号的桶/用户名）。"""
    gd = o.create_grant_record(_spec(), instance_code="same_inst")
    g9 = o.create_grant_record(_spec(bucket="prod-bucket"), instance_code="same_inst",
                               profile=_p1949())
    assert gd["grant_id"] != g9["grant_id"]
    assert gd["bucket"] == "wuji-sing" and g9["bucket"] == "prod-bucket"


def test_user_and_policy_names_carry_account_prefix(fake_redis):
    g9 = o.create_grant_record(_spec(enterprise="Acme"), instance_code="inst_9", profile=_p1949())
    assert g9["user_name"].startswith("tempak-1949-")
    assert g9["policy_name"] == issuer.policy.POLICY_PREFIX + g9["user_name"]
    # 默认档不带 1949 段（云上对象名不重叠）
    gd = o.create_grant_record(_spec(enterprise="Acme"), instance_code="inst_D")
    assert gd["user_name"].startswith("tempak-") and "1949" not in gd["user_name"]


def test_display_name_suffix_per_account(fake_redis):
    g9 = o.create_grant_record(_spec(enterprise="张三"), instance_code="inst_9", profile=_p1949())
    assert o.display_name_for(g9) == "张三-1949产线临时用户"
    gd = o.create_grant_record(_spec(enterprise="外采公司A"), instance_code="inst_D")
    assert o.display_name_for(gd) == "外采公司A-临时外采用户"


def test_claim_lock_namespaces_are_separate(fake_redis):
    """两档各自的幂等锁互不阻塞（共用一把会让一个账号的审批把另一个账号的挡掉）。"""
    a = o.claim("shared_inst")
    b = o.claim("shared_inst", _p1949())
    assert a == "temp_ak:lock:shared_inst"
    assert b == "temp_ak_1949:lock:shared_inst"
    assert a and b
    assert o.claim("shared_inst") == ""                 # 各自仍 NX 幂等
    assert o.claim("shared_inst", _p1949()) == ""


# ══════════════════════════════════════════════════════════════════════════════
# 假 RAM client 工厂：记录用哪对 AK 建的、对谁下了手
# ══════════════════════════════════════════════════════════════════════════════

class _FakeRam:
    def __init__(self, ak, sk):
        self.ak, self.sk = ak, sk
        self.deleted_users = []
        self.deleted_policies = []
        self.deleted_aks = []

    # 删号序列用到的调用
    def list_access_keys(self, req):
        class _AK:
            access_key_id = "LTAI_TARGET"

        class _AKs:
            access_key = [_AK()]

        class _Body:
            access_keys = _AKs()

        class _Resp:
            body = _Body()
        return _Resp()

    def update_access_key(self, req):
        return None

    def delete_access_key(self, req):
        self.deleted_aks.append((req.user_name, req.user_access_key_id))

    def detach_policy_from_user(self, req):
        return None

    def list_policy_versions(self, req):
        class _Body:
            policy_versions = None

        class _Resp:
            body = _Body()
        return _Resp()

    def delete_policy(self, req):
        self.deleted_policies.append(req.policy_name)

    def delete_user(self, req):
        self.deleted_users.append(req.user_name)


@pytest.fixture
def ram_clients(monkeypatch):
    """桩 permsync.make_ram_client：按传入的 ak/sk 造 client 并登记。

    零参调用（默认档路径）登记成 ak=""——这正是默认档的旧行为，与 1949 档的显式传参区分开。
    """
    made = []
    import core.oss_perm.permsync as permsync

    def fake(ak="", sk=""):
        c = _FakeRam(ak, sk)
        made.append(c)
        return c

    monkeypatch.setattr(permsync, "make_ram_client", fake)
    return made


def _issued(profile=None, **over):
    """已 ISSUED 的方案 B grant（可直接喂 revoke_grant / sweep_expired）。"""
    p = profile or accounts.default()
    now = time.time()
    gid = o.grant_id_for(over.pop("instance_code", "inst_" + p.slug), p)
    g = {
        "grant_id": gid, "account": p.slug, "stage": o.STAGE_ISSUED, "mode": issuer.RAM_MODE,
        "platform": "aliyun", "enterprise": "主体" + (p.slug or "def"),
        "bucket": "b", "prefix": "p/", "caps": ["read"],
        "not_before": now - 7200, "expire": now - 60,          # 已过期
        "user_name": p.user_prefix + "sub-abc",
        "policy_name": issuer.policy.POLICY_PREFIX + p.user_prefix + "sub-abc",
        "ak_id": "LTAI_TARGET", "requester": "ou_a", "source_ips": [],
    }
    g.update(over)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# I3 revoke_grant 按账号取 AK
# ══════════════════════════════════════════════════════════════════════════════

def test_revoke_1949_grant_uses_1949_ak(fake_redis, ram_clients):
    g = _issued(_p1949())
    o._save(g)
    assert cleanup.revoke_grant(g) is True
    assert len(ram_clients) == 1
    c = ram_clients[0]
    assert (c.ak, c.sk) == ("LTAI_1949", "SK_1949")       # **不是** LTAI_DEFAULT
    assert c.deleted_users == ["tempak-1949-sub-abc"]
    assert o.get_grant(g["grant_id"])["stage"] == o.STAGE_REVOKED


def test_revoke_default_grant_keeps_zero_arg_path(fake_redis, ram_clients):
    """默认档零回归：仍走 make_ram_client() 零参（读 ALIYUN_ACCESS_KEY_*），不传 ak/sk。"""
    g = _issued()
    o._save(g)
    assert cleanup.revoke_grant(g) is True
    assert (ram_clients[0].ak, ram_clients[0].sk) == ("", "")
    assert ram_clients[0].deleted_users == ["tempak-sub-abc"]


def test_revoke_legacy_grant_without_account_key(fake_redis, ram_clients):
    """历史 grant 没有 account 键 → 默认档路径，行为与多账号化前一致。"""
    g = _issued()
    g.pop("account")
    o._save(g)
    assert cleanup.revoke_grant(g) is True
    assert (ram_clients[0].ak, ram_clients[0].sk) == ("", "")


def test_revoke_1949_grant_raises_when_ak_removed(monkeypatch, fake_redis, ram_clients):
    """**对抗核心 A**：.env 抹了 1949 的 AK（档案仍注册）后撤销它的 grant，
    必须炸（fail-closed），绝不能悄悄用默认档的 AK 去删——那会在错误账号里删同名用户。"""
    g = _issued(_p1949())
    o._save(g)
    _strip_1949_ak(monkeypatch)
    with pytest.raises(accounts.UnknownAccountError):
        cleanup.revoke_grant(g)
    assert ram_clients == []                # 一个 client 都没造，更没删任何东西


def test_revoke_1949_grant_raises_when_profile_deregistered(monkeypatch, fake_redis, ram_clients):
    """**对抗核心 B**：整档下线（三项全空）后同样 fail-closed，不退默认档。"""
    g = _issued(_p1949())
    o._save(g)
    _deregister_1949(monkeypatch)
    with pytest.raises(accounts.UnknownAccountError):
        cleanup.revoke_grant(g)
    assert ram_clients == []


def test_revoke_sts_grant_never_touches_ram(fake_redis, ram_clients):
    """STS 自灭：只翻状态，不建任何 RAM client（也就没有取错账号 AK 的机会）。"""
    g = _issued(_p1949(), mode=issuer.STS_MODE)
    o._save(g)
    assert cleanup.revoke_grant(g) is True
    assert ram_clients == []
    assert o.get_grant(g["grant_id"])["stage"] == o.STAGE_REVOKED


# ══════════════════════════════════════════════════════════════════════════════
# I2 sweep_expired 逐档不串
# ══════════════════════════════════════════════════════════════════════════════

def test_sweep_expired_handles_each_account_with_own_client(fake_redis, ram_clients):
    gd = _issued()
    g9 = _issued(_p1949())
    o._save(gd)
    o._save(g9)

    revoked = cleanup.sweep_expired()
    assert set(revoked) == {gd["grant_id"], g9["grant_id"]}

    by_ak = {(c.ak, c.sk): c for c in ram_clients}
    assert set(by_ak) == {("", ""), ("LTAI_1949", "SK_1949")}
    # 每个用户只被**自己账号**那个 client 删过
    assert by_ak[("", "")].deleted_users == ["tempak-sub-abc"]
    assert by_ak[("LTAI_1949", "SK_1949")].deleted_users == ["tempak-1949-sub-abc"]
    # 默认 client 绝不能碰 1949 的用户（这条就是「不会用默认 client 去删 1949 的用户」）
    assert "tempak-1949-sub-abc" not in by_ak[("", "")].deleted_users
    assert "tempak-1949-sub-abc" not in by_ak[("", "")].deleted_policies


def test_sweep_scans_both_prefixes(monkeypatch, fake_redis, ram_clients):
    """按档遍历各自前缀（不是全前缀单扫）——扫描键必须两个前缀都出现。"""
    seen = []
    real_scan = fake_redis.scan_iter

    def spy(match=None, **kw):
        seen.append(match)
        return real_scan(match=match, **kw)

    monkeypatch.setattr(fake_redis, "scan_iter", spy)
    cleanup.sweep_expired()
    assert seen == ["temp_ak:grant:*", "temp_ak_1949:grant:*"]


def test_sweep_still_scans_1949_when_only_ak_removed(monkeypatch, fake_redis, ram_clients):
    """只抹 AK（档案仍注册）→ 命名空间**照旧被扫到**（不会静默漏清理），
    但删号时 fail-closed、不会拿默认 client 去删；记录留在原位等运维补 AK。"""
    g9 = _issued(_p1949())
    o._save(g9)
    _strip_1949_ak(monkeypatch)
    seen = []
    real_scan = fake_redis.scan_iter
    monkeypatch.setattr(fake_redis, "scan_iter",
                        lambda match=None, **kw: (seen.append(match), real_scan(match=match, **kw))[1])
    revoked = cleanup.sweep_expired()
    assert "temp_ak_1949:grant:*" in seen        # 扫得到 = 可观测
    assert revoked == []                          # 但没删成
    assert ram_clients == []                      # 更没用默认 client 去删
    assert fake_redis.get("temp_ak_1949:grant:" + g9["grant_id"])


def test_sweep_skips_1949_namespace_when_fully_deregistered(monkeypatch, fake_redis, ram_clients):
    """整档下线（三项全空）后不再扫该命名空间——记录不动、更不会被默认 client 顺手删掉。

    注：这也是 dev 把注册条件放宽成「任一配置存在即注册」的原因：真下线会让这批
    云上残留失去自动清理（时间窗仍拒调用，纵深防御在；但 artifact 需人工清）。"""
    g9 = _issued(_p1949())
    o._save(g9)
    _deregister_1949(monkeypatch)
    revoked = cleanup.sweep_expired()
    assert revoked == []
    assert ram_clients == []
    assert fake_redis.get("temp_ak_1949:grant:" + g9["grant_id"])


def test_sweep_one_account_failure_does_not_block_the_other(monkeypatch, fake_redis, ram_clients):
    """某档扫挂（如该账号 AK 被抹）不影响另一档继续清理。"""
    gd = _issued()
    g9 = _issued(_p1949())
    o._save(gd)
    o._save(g9)

    real = cleanup.revoke_grant

    def boom(grant, **kw):
        if grant.get("account") == "1949":
            raise RuntimeError("1949 账号 AK 失效")
        return real(grant, **kw)

    monkeypatch.setattr(cleanup, "revoke_grant", boom)
    revoked = cleanup.sweep_expired()
    assert revoked == [gd["grant_id"]]


def test_sweep_ignores_not_yet_expired_and_non_issued(fake_redis, ram_clients):
    future = _issued(_p1949(), expire=time.time() + 86400)
    failed = _issued(_p1949(), instance_code="other", stage=o.STAGE_FAILED)
    o._save(future)
    o._save(failed)
    assert cleanup.sweep_expired() == []
    assert ram_clients == []


# ══════════════════════════════════════════════════════════════════════════════
# I4 共用的延长/撤销审批按凭证ID 前缀分派
# ══════════════════════════════════════════════════════════════════════════════

def _ext_event(instance="ext_inst_1", status="APPROVED"):
    return {"header": {"event_type": "approval_instance"},
            "event": {"approval_code": EXT_CODE, "status": status, "instance_code": instance}}


def _ext_detail(grant_id, enterprise, *, action="延长", instance="ext_inst_1"):
    return {
        "status": "APPROVED",
        "approval_code": EXT_CODE,
        "instance_code": instance,
        "form": [
            {"name": "凭证ID", "value": grant_id},
            {"name": "撤销/延长", "value": action},
            {"name": "使用企业信息", "value": enterprise},
            {"name": "DateInterval", "value": {"start": "", "end": "2099-06-01 00:00:00"}},
        ],
    }


@pytest.fixture
def ext_spy(monkeypatch):
    seen = {"extend": [], "revoke": [], "notify": []}
    monkeypatch.setattr(o, "extend_grant",
                        lambda g, nb, exp, **kw: (seen["extend"].append(g) or (g, None)))
    monkeypatch.setattr(approval.delivery, "deliver_extend", lambda g, c: None)
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda g, **kw: (seen["revoke"].append(g) or True))
    monkeypatch.setattr(approval, "_notify_internal_action",
                        lambda g, text: seen["notify"].append(text))
    return seen


def test_shared_extend_approval_routes_tak1949_to_1949_grant(monkeypatch, fake_redis, ext_spy):
    """喂 tak1949- 凭证ID → 取到 **1949 档** 的 grant（共用审批的分派命门）。"""
    g9 = _issued(_p1949(), stage=o.STAGE_ISSUED, expire=time.time() + 3600)
    o._save(g9)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail(g9["grant_id"], g9["enterprise"]))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert res.get("error") is None, res
    assert len(ext_spy["extend"]) == 1
    assert ext_spy["extend"][0]["account"] == "1949"
    assert ext_spy["extend"][0]["grant_id"] == g9["grant_id"]


def test_shared_extend_approval_routes_tak_to_default_grant(monkeypatch, fake_redis, ext_spy):
    gd = _issued(stage=o.STAGE_ISSUED, expire=time.time() + 3600)
    o._save(gd)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail(gd["grant_id"], gd["enterprise"]))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert res.get("error") is None, res
    assert ext_spy["extend"][0]["account"] == ""


def test_shared_extend_does_not_leak_across_accounts(monkeypatch, fake_redis, ext_spy):
    """两档各有一条 grant 且 instance hash 相同：喂 1949 的 ID 绝不能动到默认档那条。"""
    inst = "same_inst"
    gd = _issued(instance_code=inst, stage=o.STAGE_ISSUED,
                 expire=time.time() + 3600, enterprise="甲公司")
    g9 = _issued(_p1949(), instance_code=inst, stage=o.STAGE_ISSUED,
                 expire=time.time() + 3600, enterprise="张三")
    o._save(gd)
    o._save(g9)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail(g9["grant_id"], "张三"))
    approval.handle_temp_ak_extend_event(_ext_event())
    touched = ext_spy["extend"][0]
    assert touched["grant_id"] == g9["grant_id"]
    assert touched["enterprise"] == "张三"
    assert touched["grant_id"] != gd["grant_id"]


def test_shared_revoke_approval_routes_to_1949_grant(monkeypatch, fake_redis, ext_spy):
    g9 = _issued(_p1949(), stage=o.STAGE_ISSUED, expire=time.time() + 3600)
    o._save(g9)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail(g9["grant_id"], g9["enterprise"], action="撤销"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert res.get("action") == "revoke"
    assert len(ext_spy["revoke"]) == 1
    assert ext_spy["revoke"][0]["account"] == "1949"
    assert ext_spy["extend"] == []


def test_shared_extend_cross_subject_still_rejected(monkeypatch, fake_redis, ext_spy):
    """防串（同账号内张冠李戴）在 1949 档同样生效：主体名不符 → ��。"""
    g9 = _issued(_p1949(), stage=o.STAGE_ISSUED, expire=time.time() + 3600, enterprise="张三")
    o._save(g9)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail(g9["grant_id"], "李四"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert "error" in res
    assert ext_spy["extend"] == [] and ext_spy["revoke"] == []


def test_verify_enterprise_label_follows_account(fake_redis):
    """措辞按该账号表单叫法（1949=「使用人名称」/ 默认=「使用企业名称」）。"""
    with pytest.raises(o.TempAkError) as e9:
        approval._verify_enterprise({"account": "1949", "enterprise": "张三"}, "")
    assert "使用人名称" in str(e9.value)

    with pytest.raises(o.TempAkError) as ed:
        approval._verify_enterprise({"account": "", "enterprise": "甲公司"}, "")
    assert "使用企业名称" in str(ed.value)


def test_verify_enterprise_unknown_account_still_fails_safe(monkeypatch, fake_redis):
    """档案下线时 label 取不到也不能放行（fail-safe 判据不依赖档案）。"""
    _deregister_1949(monkeypatch)
    with pytest.raises(o.TempAkError):
        approval._verify_enterprise({"account": "1949", "enterprise": "张三"}, "")


# ── LOW-2 人名档要求精确相等（包含匹配对人名太松）─────────────────────────────

@pytest.mark.parametrize("orig,given", [
    ("张三", "张三丰"),        # 原名是给定名的子串 —— 旧的宽松包含会放行
    ("张三丰", "张三"),        # 反向
    ("张三", "张三 "),         # 仅两端空白 → strip 后相等，见下面的 ok 用例
])
def test_verify_person_subject_requires_exact_match(fake_redis, orig, given):
    g = {"account": "1949", "enterprise": orig}
    if orig.strip() == given.strip():
        approval._verify_enterprise(g, given)       # strip 后相等 → 放行
        return
    with pytest.raises(o.TempAkError) as e:
        approval._verify_enterprise(g, given)
    assert "使用人名称" in str(e.value)


def test_verify_person_subject_exact_match_ok(fake_redis):
    approval._verify_enterprise({"account": "1949", "enterprise": "张三"}, "张三")


def test_verify_enterprise_subject_keeps_loose_match(fake_redis):
    """默认档（企业名）保持宽松包含匹配——容错「有限公司」等后缀差异，零回归。"""
    g = {"account": "", "enterprise": "甲公司"}
    approval._verify_enterprise(g, "甲公司有限公司")
    approval._verify_enterprise({"account": "", "enterprise": "甲公司有限公司"}, "甲公司")
    with pytest.raises(o.TempAkError):
        approval._verify_enterprise(g, "乙公司")


def test_person_strict_check_applies_to_revoke_too(monkeypatch, fake_redis, ext_spy):
    """撤销分支同样受人名精确校验约束（「张三丰」不能撤「张三」的凭证）。"""
    g9 = _issued(_p1949(), stage=o.STAGE_ISSUED, expire=time.time() + 3600, enterprise="张三")
    o._save(g9)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail(g9["grant_id"], "张三丰", action="撤销"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert "error" in res
    assert ext_spy["revoke"] == []


# ══════════════════════════════════════════════════════════════════════════════
# I5 桶映射按账号取
# ══════════════════════════════════════════════════════════════════════════════

def test_resolve_bucket_same_display_different_real_bucket(monkeypatch):
    """**同名展示桶在两档映射到不同真实桶** —— 串了就等于把凭证发到别人的桶上。"""
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW",
                        '{"数据桶": {"region": "oss-cn-hangzhou", "bucket": "default-real"}}')
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW",
                        '{"数据桶": {"region": "oss-cn-shanghai", "bucket": "prod1949-real"}}')
    assert o.resolve_bucket("数据桶") == ("oss-cn-hangzhou", "default-real")
    assert o.resolve_bucket("数据桶", _p1949()) == ("oss-cn-shanghai", "prod1949-real")


def test_resolve_bucket_1949_does_not_fall_back_to_permsync_map(monkeypatch):
    """permsync.BUCKET_MAP 是**现有账号**算法组的对照表；1949 档不得回退到它，
    否则「新加坡-wuji-sing」会在 1949 账号里解析成另一个账号的真实桶名。"""
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW", "{}")
    region, bucket = o.resolve_bucket("新加坡-wuji-sing", _p1949())
    assert (region, bucket) == ("", "新加坡-wuji-sing")     # 原样当真实桶名，不查别家表
    # 默认档仍回退（零回归）
    assert o.resolve_bucket("新加坡-wuji-sing") == ("oss-ap-southeast-1", "wuji-sing")


def test_resolve_bucket_1949_map_does_not_leak_into_default(monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW",
                        '{"仅1949": {"region": "oss-cn-shanghai", "bucket": "prod1949-real"}}')
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    assert o.resolve_bucket("仅1949") == ("", "仅1949")       # 默认档看不到 1949 的表
    assert o.resolve_bucket("仅1949", _p1949())[1] == "prod1949-real"


# ── 按真实桶名反查地域（真机首单暴露：申请人填的是真实桶名，不是展示名）─────────

@pytest.fixture
def maps(monkeypatch):
    """两档各一张桶表；1949 的取值照服务器实配（展示名「产线数据」→ 真桶 wuji-product/裸地域）。"""
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW",
                        '{"共享名": {"region": "cn-hangzhou", "bucket": "default-real"}}')
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW",
                        '{"产线数据": {"region": "cn-shenzhen", "bucket": "wuji-product"}}')


def test_resolve_bucket_by_display_name_unchanged(maps):
    """① 填展示名 → 原有行为不变（反查这层不得干扰正查）。"""
    assert o.resolve_bucket("产线数据", _p1949()) == ("cn-shenzhen", "wuji-product")
    assert o.resolve_bucket("共享名") == ("cn-hangzhou", "default-real")


def test_resolve_bucket_by_real_bucket_name(maps):
    """② 填**真实桶名**也能把地域捞回来（首单踩的就是这里：region 空 → 正文三行退化成「未知」）。"""
    assert o.resolve_bucket("wuji-product", _p1949()) == ("cn-shenzhen", "wuji-product")
    assert o.resolve_bucket("default-real") == ("cn-hangzhou", "default-real")


def test_resolve_bucket_real_name_end_to_end_gives_endpoint(maps):
    """②' 端到端：填真实桶名 → 凭证正文不再是「未知」，而是可用的 endpoint。"""
    from core.temp_ak_issuance import delivery
    region, bucket = o.resolve_bucket("wuji-product", _p1949())
    lines = "\n".join(delivery._access_lines({"account": "1949", "region": region,
                                             "bucket": bucket}))
    assert "未知" not in lines
    assert "外网 Endpoint：oss-cn-shenzhen.aliyuncs.com" in lines
    assert "桶域名：wuji-product.oss-cn-shenzhen.aliyuncs.com" in lines


def test_resolve_bucket_miss_returns_passthrough_without_raising(maps):
    """③ 两种查法都不中 → `("", display)` 原样当真实桶名用，不抛（表单直填桶名仍可发）。"""
    assert o.resolve_bucket("完全没见过的桶", _p1949()) == ("", "完全没见过的桶")
    assert o.resolve_bucket("some-raw-bucket") == ("", "some-raw-bucket")


def test_resolve_bucket_reverse_lookup_tolerates_bad_entries(monkeypatch):
    """映射值不是 dict / 缺 bucket 键时反查不炸（运维手写 JSON 容错）。"""
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW",
                        '{"坏1": "not-a-dict", "坏2": {"region": "cn-x"}, '
                        '"好": {"region": "cn-shenzhen", "bucket": "wuji-product"}}')
    assert o.resolve_bucket("wuji-product", _p1949()) == ("cn-shenzhen", "wuji-product")
    assert o.resolve_bucket("not-a-dict", _p1949()) == ("", "not-a-dict")


def test_default_profile_reverse_lookup_in_permsync_map(monkeypatch):
    """④ 默认档对 permsync.BUCKET_MAP 也反查，且拿到的 region 是**带前缀**的。"""
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    assert o.resolve_bucket("新加坡-wuji-sing") == ("oss-ap-southeast-1", "wuji-sing")   # 正查
    assert o.resolve_bucket("wuji-sing") == ("oss-ap-southeast-1", "wuji-sing")          # 反查
    assert o.resolve_bucket("wuji-test-data") == ("oss-cn-beijing", "wuji-test-data")


def test_1949_reverse_lookup_never_touches_permsync_map(monkeypatch):
    """⑤ 1949 档**不**回退 permsync.BUCKET_MAP（正查反查都不许）——那是默认账号的桶表，
    串了就等于把第二账号申请人的凭证指向现有账号的桶。"""
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW", "{}")
    assert o.resolve_bucket("wuji-sing", _p1949()) == ("", "wuji-sing")            # 反查不中
    assert o.resolve_bucket("新加坡-wuji-sing", _p1949()) == ("", "新加坡-wuji-sing")  # 正查也不中
    # 默认档同一输入仍能解析（零回归、证明上面不是因为表本身失效）
    assert o.resolve_bucket("wuji-sing")[0] == "oss-ap-southeast-1"


def test_reverse_lookup_does_not_cross_accounts(maps):
    """两档各自的真实桶名互不可见（反查这层新增了一条匹配路径，别让它成为跨账号通道）。"""
    assert o.resolve_bucket("wuji-product") == ("", "wuji-product")          # 默认档看不到 1949 的桶
    assert o.resolve_bucket("default-real", _p1949()) == ("", "default-real")  # 反之亦然


def test_create_grant_by_real_bucket_name_records_region(maps, fake_redis):
    """落库层面：填真实桶名建 grant → region 有值、bucket 原样、display 记原文。"""
    g9 = o.create_grant_record(_spec(bucket="wuji-product"), instance_code="i9rev",
                               profile=_p1949())
    assert g9["bucket"] == "wuji-product"
    assert g9["region"] == "cn-shenzhen"
    assert g9["bucket_display"] == "wuji-product"


def test_create_grant_uses_account_bucket_map(monkeypatch, fake_redis):
    monkeypatch.setattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW",
                        '{"数据桶": {"region": "oss-cn-shanghai", "bucket": "prod1949-real"}}')
    monkeypatch.setattr(settings, "TEMP_AK_BUCKET_MAP_RAW",
                        '{"数据桶": {"region": "oss-cn-hangzhou", "bucket": "default-real"}}')
    g9 = o.create_grant_record(_spec(bucket="数据桶"), instance_code="i9", profile=_p1949())
    assert g9["bucket"] == "prod1949-real"
    assert g9["bucket_display"] == "数据桶"
    assert g9["region"] == "oss-cn-shanghai"


# ══════════════════════════════════════════════════════════════════════════════
# I6 make_ram_client 显式传参不读 env —— 本次隔离的命门
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ram_sdk_spy(monkeypatch):
    """桩 SDK 的 RamClient，抓到底用了哪对 ak/sk（函数内 import → patch 模块属性即生效）。"""
    seen = {}
    import alibabacloud_ram20150501.client as ram_client_mod

    class _Spy:
        def __init__(self, cfg):
            seen["ak"] = cfg.access_key_id
            seen["sk"] = cfg.access_key_secret
            seen["endpoint"] = cfg.endpoint

    monkeypatch.setattr(ram_client_mod, "Client", _Spy)
    return seen


def test_make_ram_client_explicit_ignores_env(monkeypatch, ram_sdk_spy):
    """**命门**：env `ALIBABA_CLOUD_*` 已设（模拟运维在服务器上设过）时，
    显式传参必须仍用传入的那对——否则所有账号的建号请求会被劫持到同一个账号。"""
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "LTAI_ENV_HIJACK")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "SK_ENV_HIJACK")
    from core.oss_perm.permsync import make_ram_client
    make_ram_client(ak="LTAI_1949", sk="SK_1949")
    assert ram_sdk_spy["ak"] == "LTAI_1949"
    assert ram_sdk_spy["sk"] == "SK_1949"
    assert "LTAI_ENV_HIJACK" not in (ram_sdk_spy["ak"], ram_sdk_spy["sk"])
    assert ram_sdk_spy["endpoint"] == "ram.aliyuncs.com"


def test_make_ram_client_explicit_ignores_settings(monkeypatch, ram_sdk_spy):
    monkeypatch.delenv("ALIBABA_CLOUD_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", raising=False)
    from core.oss_perm.permsync import make_ram_client
    make_ram_client(ak="LTAI_1949", sk="SK_1949")
    assert (ram_sdk_spy["ak"], ram_sdk_spy["sk"]) == ("LTAI_1949", "SK_1949")
    assert ram_sdk_spy["ak"] != settings.ALIYUN_ACCESS_KEY_ID


def test_make_ram_client_zero_arg_still_reads_env(monkeypatch, ram_sdk_spy):
    """零参旧行为逐字不变（env 优先）——正是它促成了「必须显式传参」这条规矩，钉住不许漂。"""
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "LTAI_ENV")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "SK_ENV")
    from core.oss_perm.permsync import make_ram_client
    make_ram_client()
    assert (ram_sdk_spy["ak"], ram_sdk_spy["sk"]) == ("LTAI_ENV", "SK_ENV")


def test_make_ram_client_zero_arg_falls_back_to_settings(monkeypatch, ram_sdk_spy):
    monkeypatch.delenv("ALIBABA_CLOUD_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", raising=False)
    from core.oss_perm.permsync import make_ram_client
    make_ram_client()
    assert ram_sdk_spy["ak"] == "LTAI_DEFAULT"          # settings.ALIYUN_ACCESS_KEY_ID


@pytest.mark.parametrize("ak,sk", [("LTAI_1949", ""), ("", "SK_1949")])
def test_make_ram_client_partial_args_raise_instead_of_falling_back(monkeypatch, ram_sdk_spy,
                                                                    ak, sk):
    """只传一半 → **抛错**，绝不静默回落到 env/settings（那会拿另一个账号的凭证去建号）。"""
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "LTAI_ENV")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "SK_ENV")
    from core.oss_perm.permsync import make_ram_client
    with pytest.raises(RuntimeError):
        make_ram_client(ak=ak, sk=sk)
    assert ram_sdk_spy == {}          # client 压根没造出来


def test_accounts_ram_client_end_to_end_uses_profile_ak(monkeypatch, ram_sdk_spy):
    """整条链：accounts.ram_client(1949 档) → make_ram_client(显式) → SDK 收到 1949 的 AK。"""
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "LTAI_ENV_HIJACK")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "SK_ENV_HIJACK")
    accounts.ram_client(_p1949())
    assert (ram_sdk_spy["ak"], ram_sdk_spy["sk"]) == ("LTAI_1949", "SK_1949")


def test_issue_ram_end_to_end_client_is_1949(monkeypatch, ram_sdk_spy, fake_redis):
    """issuer._issue_ram 对 1949 grant 建号时，SDK 拿到的必须是 1949 的 AK。"""
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "LTAI_ENV_HIJACK")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "SK_ENV_HIJACK")
    g = _issued(_p1949(), expire=time.time() + 5 * 86400)
    try:
        issuer._issue_ram(g)          # _Spy 没有 get_user 等方法 → 会抛，但 client 已被造出来
    except Exception:
        pass
    assert (ram_sdk_spy["ak"], ram_sdk_spy["sk"]) == ("LTAI_1949", "SK_1949")


# ══════════════════════════════════════════════════════════════════════════════
# STS 分支的账号硬门（两道：classify 强制 RAM + _issue_sts 纵深拦截）
#   为什么这组最要紧：STS 用**全局 Master AK** 去 AssumeRole `TEMP_AK_OSS_ROLE_ARN`——
#   那是**现有账号**的宽 OSS 角色。若 1949 的申请走了这条，签出来的是现有账号身份的凭证，
#   申请人填一个现有账号的桶名就能拿到别人的数据。
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sts_spy(monkeypatch):
    """桩 STS：一旦被调用就留痕（配合断言「非默认账号绝不该走到这里」）。"""
    seen = {}
    monkeypatch.setattr(settings, "TEMP_AK_OSS_ROLE_ARN",
                        "acs:ram::DEFAULT_UID_9999:role/TempAkOssRole")
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts, "assume_role_with_policy",
                        lambda arn, doc, dur, session_name="": (
                            seen.update(arn=arn) or {"access_key_id": "STS.AK",
                                                     "access_key_secret": "SK",
                                                     "security_token": "TOK",
                                                     "expire_ts": 0.0}))
    return seen


def test_classify_mode_forces_ram_for_non_default_account(monkeypatch):
    """**门 1**：即便阈值配成 12h（代码默认值就是危险侧的 43200），非默认账号也一律 RAM。"""
    monkeypatch.setattr(settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    short = time.time() + 3600                       # 1h 窗口，默认档下必判 sts
    assert issuer.classify_mode(short, profile=_p1949()) == issuer.RAM_MODE
    assert issuer.classify_mode(short, profile=accounts.default()) == issuer.STS_MODE
    assert issuer.classify_mode(short) == issuer.STS_MODE          # 不传 profile = 默认档，零回归


def test_create_grant_1949_short_window_is_ram(monkeypatch, fake_redis):
    """端到端：1949 的 1h 申请（阈值开到 12h）落库仍是方案 B，不是 STS。"""
    monkeypatch.setattr(settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    g9 = o.create_grant_record(_spec(expire=time.time() + 3600), instance_code="i9s",
                               profile=_p1949())
    assert g9["mode"] == issuer.RAM_MODE
    assert g9["policy_name"]                          # 方案 B 需要 policy 名
    gd = o.create_grant_record(_spec(expire=time.time() + 3600), instance_code="iDs")
    assert gd["mode"] == issuer.STS_MODE              # 默认档零回归


def test_issue_sts_hard_gate_rejects_non_default_account(sts_spy, fake_redis):
    """**门 2（纵深）**：哪怕 mode 被历史记录/篡改带成 sts，_issue_sts 也按 grant["account"] 拦死。"""
    g9 = _issued(_p1949(), mode=issuer.STS_MODE, expire=time.time() + 3600)
    with pytest.raises(issuer.IssueError) as e:
        issuer._issue_sts(g9)
    assert "1949" in str(e.value)
    assert sts_spy == {}, "非默认账号绝不能调到 AssumeRole（那会用默认账号的角色签凭证）"


def test_issue_dispatch_honours_sts_hard_gate(sts_spy, fake_redis):
    """走公开入口 issue()：mode 已是 sts 的 1949 grant → 抛错，不签发。"""
    g9 = _issued(_p1949(), mode=issuer.STS_MODE, expire=time.time() + 3600)
    with pytest.raises(issuer.IssueError):
        issuer.issue(g9)
    assert sts_spy == {}


def test_issue_sts_still_works_for_default_account(sts_spy, fake_redis):
    """零回归：默认档的 STS 单发照旧（门只拦非默认档）。"""
    gd = _issued(mode=issuer.STS_MODE, expire=time.time() + 3600)
    creds = issuer._issue_sts(gd)
    assert creds["mode"] == issuer.STS_MODE
    assert sts_spy["arn"] == "acs:ram::DEFAULT_UID_9999:role/TempAkOssRole"


def test_plan_for_1949_never_previews_sts(monkeypatch, fake_redis):
    """dry-run 也不得给 1949 出 STS 计划（否则运维照着计划以为能发 STS）。"""
    monkeypatch.setattr(settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    g9 = _issued(_p1949(), expire=time.time() + 3600)
    g9.pop("mode")
    p = issuer.plan(g9)
    assert p["mode"] == issuer.RAM_MODE
    assert "role_arn" not in p


def test_extend_sts_grant_of_1949_converts_to_ram(monkeypatch, fake_redis, ram_clients):
    """延期一条（历史遗留的）1949 STS grant：重签发时必须转方案 B，用 1949 的 AK 建号。"""
    monkeypatch.setattr(settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts, "assume_role_with_policy",
                        lambda *a, **k: pytest.fail("1949 不得走 STS 重签发"))
    monkeypatch.setattr(issuer, "_issue_ram", lambda g: {
        "access_key_id": "LTAI_NEW1949", "access_key_secret": "SK",
        "security_token": "", "expire_ts": g["expire"], "mode": issuer.RAM_MODE})
    g9 = _issued(_p1949(), mode=issuer.STS_MODE, policy_name="",
                 expire=time.time() + 3600, stage=o.STAGE_ISSUED)
    o._save(g9)
    g2, creds = o.extend_grant(g9, 0, time.time() + 7200)
    assert g2["mode"] == issuer.RAM_MODE
    assert g2["policy_name"].endswith(g9["user_name"])
    assert creds["access_key_id"] == "LTAI_NEW1949"


def test_grant_profile_falls_back_to_default_but_hard_gate_holds(monkeypatch, sts_spy, fake_redis):
    """整档下线时 _grant_profile 退默认档（mode 判定退默认只会**更保守**）——
    但 _issue_sts 的硬门仍按 grant["account"] 拦住，不会因为退默认档就用默认账号的角色签凭证。"""
    monkeypatch.setattr(settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    g9 = _issued(_p1949(), expire=time.time() + 3600)
    _deregister_1949(monkeypatch)
    assert issuer._grant_profile(g9).slug == ""                        # 退默认档
    g9.pop("mode")
    with pytest.raises(issuer.IssueError):
        issuer.issue(g9)                                               # 仍被硬门拦死
    assert sts_spy == {}


def test_grant_profile_keeps_1949_when_only_ak_removed(monkeypatch, fake_redis):
    """只抹 AK 时档案仍在 → _grant_profile 仍取 1949（mode 判定继续强制 RAM，不会误判 STS）。"""
    monkeypatch.setattr(settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    g9 = _issued(_p1949(), expire=time.time() + 3600)
    _strip_1949_ak(monkeypatch)
    assert issuer._grant_profile(g9).slug == "1949"
    assert issuer.classify_mode(g9["expire"], profile=issuer._grant_profile(g9)) == issuer.RAM_MODE


# ══════════════════════════════════════════════════════════════════════════════
# MED-2 交叉校验：grant["account"] 与 grant_id 前缀必须同源
# ══════════════════════════════════════════════════════════════════════════════

def test_permsync_client_rejects_account_prefix_contradiction(fake_redis, ram_clients):
    """`{"grant_id":"tak1949-x","account":""}` → 抛错。

    放行的后果（假成功）：用默认账号的 AK 去删 `tempak-1949-…` → RAM 回 EntityNotExist.User →
    cleanup 只记日志继续走 → 状态被置成 REVOKED，而云上真号还活着、AK 还能用。"""
    with pytest.raises(accounts.UnknownAccountError) as e:
        issuer.permsync_client({"grant_id": "tak1949-x", "account": ""})
    assert "自相矛盾" in str(e.value) or "1949" in str(e.value)
    assert ram_clients == []


def test_permsync_client_rejects_reverse_contradiction(fake_redis, ram_clients):
    with pytest.raises(accounts.UnknownAccountError):
        issuer.permsync_client({"grant_id": "tak-x", "account": "1949"})
    assert ram_clients == []


def test_permsync_client_accepts_consistent_pairs(fake_redis, ram_clients):
    issuer.permsync_client({"grant_id": "tak1949-x", "account": "1949"})
    issuer.permsync_client({"grant_id": "tak-x", "account": ""})
    issuer.permsync_client({"grant_id": "tak-x"})            # 无 account 键（历史）
    assert [(c.ak, c.sk) for c in ram_clients] == [
        ("LTAI_1949", "SK_1949"), ("", ""), ("", "")]


def test_permsync_client_unknown_prefix_is_not_treated_as_contradiction(fake_redis, ram_clients):
    """认不出前缀（by_grant_id→None）时不算矛盾，按 account 走——不误伤将来的新前缀。"""
    issuer.permsync_client({"grant_id": "weird-x", "account": ""})
    assert (ram_clients[0].ak, ram_clients[0].sk) == ("", "")


def test_revoke_contradictory_grant_fails_instead_of_fake_success(fake_redis, ram_clients):
    """端到端：矛盾 grant 走 revoke_grant → 抛错，**绝不能**被置成 REVOKED（假成功）。"""
    g = _issued(_p1949())
    g["account"] = ""                     # 篡改：grant_id 还是 tak1949-
    o._save(g)
    with pytest.raises(accounts.UnknownAccountError):
        cleanup.revoke_grant(g)
    assert g["stage"] == o.STAGE_ISSUED
    assert ram_clients == []


# ══════════════════════════════════════════════════════════════════════════════
# MED-3 内部群按账号分（回执/告警不混进别的账号运维群）
# ══════════════════════════════════════════════════════════════════════════════

def test_chat_id_for_prefers_profile_then_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default_ops")
    monkeypatch.setattr(settings, "FEISHU_CHAT_ID", "oc_global")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "oc_1949_ops")
    assert accounts.chat_id_for(profile=_p1949()) == "oc_1949_ops"
    assert accounts.chat_id_for({"account": "1949"}) == "oc_1949_ops"
    assert accounts.chat_id_for({"account": ""}) == "oc_default_ops"
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "")
    assert accounts.chat_id_for({"account": "1949"}) == "oc_default_ops"   # 档案未配 → 全局
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "")
    assert accounts.chat_id_for({"account": "1949"}) == "oc_global"


def test_chat_id_for_unknown_account_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default_ops")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "oc_1949_ops")
    _deregister_1949(monkeypatch)
    assert accounts.chat_id_for({"account": "1949"}) == "oc_default_ops"   # 取不到档案不炸


@pytest.fixture
def text_spy(monkeypatch):
    sent = []
    import core.dsw_scheduler as sched
    monkeypatch.setattr(sched, "_send_text", lambda target, chat, text: sent.append((chat, text)))
    return sent


def test_internal_action_notice_goes_to_own_account_chat(monkeypatch, text_spy):
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default_ops")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "oc_1949_ops")
    approval._notify_internal_action(_issued(_p1949()), "已撤销")
    assert text_spy[0][0] == "oc_1949_ops"
    approval._notify_internal_action(_issued(), "已撤销")
    assert text_spy[1][0] == "oc_default_ops"


def test_internal_failure_notice_goes_to_own_account_chat(monkeypatch, text_spy):
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default_ops")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "oc_1949_ops")
    approval._notify_internal_failure("i9", RuntimeError("boom"), _p1949())
    assert text_spy[0][0] == "oc_1949_ops"
    approval._notify_internal_failure("iD", RuntimeError("boom"))
    assert text_spy[1][0] == "oc_default_ops"          # 不传 profile → 全局（零回归）


def test_creds_undelivered_alert_goes_to_own_account_chat(monkeypatch, text_spy):
    from core.temp_ak_issuance import delivery
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default_ops")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "oc_1949_ops")
    delivery._alert_creds_undelivered(_issued(_p1949()))
    assert text_spy[0][0] == "oc_1949_ops"
    assert "SK_" not in text_spy[0][1]                 # 告警不带任何 secret


def test_issue_failure_notice_uses_1949_chat_end_to_end(monkeypatch, fake_redis, text_spy):
    """1949 发放失败的内部告警落 1949 的群（而不是现有账号的运维群）。"""
    monkeypatch.setattr(settings, "TEMP_AK_CHAT_ID", "oc_default_ops")
    monkeypatch.setattr(settings, "TEMP_AK_1949_CHAT_ID", "oc_1949_ops")
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _detail_1949())
    monkeypatch.setattr(issuer, "issue",
                        lambda g: (_ for _ in ()).throw(issuer.IssueError("建号失败")))
    res = approval.handle_temp_ak_event(_issue_event(CODE_1949, instance="i9fail"))
    assert "error" in res
    assert text_spy and text_spy[0][0] == "oc_1949_ops"


# ══════════════════════════════════════════════════════════════════════════════
# 1949 档表单解析（无「平台」控件 / 多一个「备注」）
# ══════════════════════════════════════════════════════════════════════════════

def _detail_1949(*, by_id=False, subject="张三", perm=None, directory="oss://prod-bucket/line/a/",
                 note="产线扩容用", date_interval=None):
    perm = perm if perm is not None else ["read", "download"]
    date_interval = date_interval if date_interval is not None else \
        {"start": "", "end": "2099-01-01 00:00:00"}
    names = {
        "enterprise":    "widget17846410216400001" if by_id else "使用人名称",
        "perm":          "widget17852975709640001" if by_id else "权限设置",
        "date_interval": "widget17852976459760001" if by_id else "DateInterval",
        "directory":     "widget17852975954890001" if by_id else "访问目录",
        "note":          "widget17852976732260001" if by_id else "备注",
    }
    key = "id" if by_id else "name"
    return {
        "status": "APPROVED", "approval_code": CODE_1949, "instance_code": "i9",
        "form": [
            {key: names["enterprise"], "value": subject},
            {key: names["perm"], "value": perm},
            {key: names["date_interval"], "value": date_interval},
            {key: names["directory"], "value": directory},
            {key: names["note"], "value": note},
        ],
    }


@pytest.mark.parametrize("by_id", [False, True])
def test_parse_1949_form(by_id):
    """无「平台」控件 → 恒 aliyun（不能被解析成 unknown 而全量拒发）；「备注」进 spec.note。"""
    spec = approval.parse_temp_ak_request(_detail_1949(by_id=by_id), {}, _p1949())
    assert spec["platform"] == "aliyun"
    assert spec["enterprise"] == "张三"
    assert spec["bucket"] == "prod-bucket"
    assert spec["prefix"] == "line/a/"
    assert set(spec["caps"]) == {"read", "download"}
    assert spec["expire"] > time.time()
    assert spec["note"] == "产线扩容用"
    assert spec["reason"] == "使用人名称：张三"          # 措辞按该档 subject_label


def test_parse_1949_note_optional():
    spec = approval.parse_temp_ak_request(_detail_1949(note=""), {}, _p1949())
    assert spec["note"] == ""


def test_parse_1949_ignores_a_platform_widget_if_present():
    """1949 模板即便被人加了「平台=火山云」控件，也不该被读成火山（该档无 platform 字段映射）。"""
    d = _detail_1949()
    d["form"].append({"name": "平台", "value": "火山云"})
    spec = approval.parse_temp_ak_request(d, {}, _p1949())
    assert spec["platform"] == "aliyun"


def test_parse_1949_empty_caps_still_rejected():
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_request(_detail_1949(perm=[]), {}, _p1949())


def test_parse_1949_missing_bucket_rejected():
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_request(_detail_1949(directory=""), {}, _p1949())


def test_parse_1949_widget_ids_needed_when_names_renamed():
    """字段名被改（只剩 widget id 可认）时仍能解析——widget id 就是这条保险。"""
    d = _detail_1949(by_id=True)
    spec = approval.parse_temp_ak_request(d, {}, _p1949())
    assert spec["bucket"] == "prod-bucket" and spec["caps"]


# ── 默认档解析零回归 ──────────────────────────────────────────────────────────

def _detail_default(*, platform="阿里云", by_id=False):
    names = {
        "platform":      "widget17846401222860001" if by_id else "平台",
        "enterprise":    "widget17846886904010001" if by_id else "使用企业名称",
        "perm":          "widget17846401501570001" if by_id else "权限设置",
        "date_interval": "widget17846402309610001" if by_id else "DateInterval",
        "directory":     "widget17846402564230001" if by_id else "申请目录",
    }
    key = "id" if by_id else "name"
    return {
        "status": "APPROVED", "approval_code": DEF_CODE, "instance_code": "iD",
        "form": [
            {key: names["platform"], "value": platform},
            {key: names["enterprise"], "value": "外采公司A"},
            {key: names["perm"], "value": ["read", "write"]},
            {key: names["date_interval"], "value": {"start": "", "end": "2099-01-01 00:00:00"}},
            {key: names["directory"], "value": "oss://wuji-sing/team/data/"},
        ],
    }


@pytest.mark.parametrize("by_id", [False, True])
def test_parse_default_form_unchanged(by_id):
    """默认档解析结果与多账号化前一致（含不传 profile 的调用方式）。"""
    d = _detail_default(by_id=by_id)
    a = approval.parse_temp_ak_request(d, {})                       # 不传 profile
    b = approval.parse_temp_ak_request(d, {}, accounts.default())   # 显式默认档
    for spec in (a, b):
        assert spec["platform"] == "aliyun"
        assert spec["enterprise"] == "外采公司A"
        assert spec["bucket"] == "wuji-sing"
        assert spec["prefix"] == "team/data/"
        assert set(spec["caps"]) == {"read", "write"}
        assert spec["reason"] == "使用企业名称：外采公司A"
        assert spec["note"] == ""            # 默认档模板无「备注」控件
    # 两种调用方式结果一致。`not_before` 排除比较：表单没填生效时间 → _validate_spec 填 now()，
    # 两次解析天然差几微秒（别把它写成 a == b，那是 flaky）。
    assert a.pop("not_before") > 0 and b.pop("not_before") > 0
    assert a == b


def test_parse_default_still_rejects_volcano():
    """默认档「平台=火山云」仍然拒（1949 的恒 aliyun 不能把这条门禁冲掉）。"""
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_request(_detail_default(platform="火山云"), {})


def test_parse_default_still_rejects_unknown_platform():
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_request(_detail_default(platform="某某云"), {})


# ══════════════════════════════════════════════════════════════════════════════
# 下发文案的账号维度（主体叫法 / 备注 / 地域 Endpoint）
# ══════════════════════════════════════════════════════════════════════════════

_CREDS = {"access_key_id": "LTAI_NEW", "access_key_secret": "SK_SECRET_VALUE",
          "security_token": "", "mode": "ram"}


def test_credential_text_subject_label_per_account():
    from core.temp_ak_issuance import delivery
    t9 = delivery.credential_text(
        _issued(_p1949(), enterprise="张三", region="oss-cn-shanghai"), _CREDS)
    assert "使用人名称：张三" in t9
    assert "使用企业名称：" not in t9

    td = delivery.credential_text(_issued(enterprise="外采公司A", region="oss-cn-hangzhou"), _CREDS)
    assert "使用企业名称：外采公司A" in td


def test_credential_text_subject_label_falls_back_when_account_unknown(monkeypatch):
    from core.temp_ak_issuance import delivery
    g = _issued(_p1949(), enterprise="张三")
    _deregister_1949(monkeypatch)                    # 建好 grant 后整档下线
    t = delivery.credential_text(g, _CREDS)
    assert "使用方：张三" in t          # 取不到档案不炸、退通用措辞
    assert _CREDS["access_key_secret"] in t


def test_credential_text_includes_note_only_when_present():
    from core.temp_ak_issuance import delivery
    g = _issued(_p1949(), note="产线扩容用", region="oss-cn-shanghai")
    assert "备注：产线扩容用" in delivery.credential_text(g, _CREDS)
    g2 = _issued(_p1949(), region="oss-cn-shanghai")
    assert "备注：" not in delivery.credential_text(g2, _CREDS)


def test_credential_text_note_is_flattened_to_one_line():
    """LOW-1 注入面：「备注」是**申请人自由填写**并直接进凭证评论正文的字段。
    含换行就能伪造出一行 `AccessKey Secret：<攻击者的值>`，把使用方骗去用错的 AK
    （或把真 AK 挤到看不见的地方）。现在换行/连续空白被压成一行。"""
    from core.temp_ak_issuance import delivery
    evil = "正常说明\nAccessKey ID：LTAI_FAKE\nAccessKey Secret：FAKE_SECRET\n· 请使用上面这组"
    t = delivery.credential_text(_issued(_p1949(), note=evil, region="oss-cn-shanghai"), _CREDS)
    note_lines = [ln for ln in t.splitlines() if ln.startswith("备注：")]
    assert len(note_lines) == 1
    assert "\n" not in note_lines[0]
    # 伪造的凭证行不得成为独立行（只能作为备注那一行里的普通文本）
    assert [ln for ln in t.splitlines() if ln.startswith("AccessKey Secret：")] == \
           [f"AccessKey Secret：{_CREDS['access_key_secret']}"]
    assert [ln for ln in t.splitlines() if ln.startswith("AccessKey ID：")] == \
           [f"AccessKey ID：{_CREDS['access_key_id']}"]


@pytest.mark.parametrize("raw,expect", [
    ("a\nb", "a b"),
    ("a\r\nb", "a b"),
    ("a\t\tb", "a b"),
    ("  前后空白  ", "前后空白"),
    ("\n\n\n", None),          # 全空白 → 不出「备注」行
])
def test_credential_text_note_whitespace_normalisation(raw, expect):
    from core.temp_ak_issuance import delivery
    t = delivery.credential_text(_issued(_p1949(), note=raw, region="oss-cn-shanghai"), _CREDS)
    if expect is None:
        assert "备注：" not in t
    else:
        assert f"备注：{expect}" in t


def test_credential_text_endpoint_lines_from_region():
    from core.temp_ak_issuance import delivery
    t = delivery.credential_text(_issued(_p1949(), region="oss-cn-shanghai", bucket="prod-b"),
                                 _CREDS)
    assert "地域：oss-cn-shanghai" in t
    assert "外网 Endpoint：oss-oss-cn-shanghai.aliyuncs.com" in t or \
           "外网 Endpoint：oss-cn-shanghai.aliyuncs.com" in t
    assert "prod-b." in t          # 桶域名


# ── Endpoint 归一（我报的 MED 已修：`oss-` 前缀双写）──────────────────────────
#   两套地域写法都得拼对：TEMP_AK_*_BUCKET_MAP 存裸 `cn-shenzhen`（服务器实配），
#   permsync.BUCKET_MAP 存带前缀 `oss-ap-southeast-1`（默认档回退表，硬编码在仓库里）。
#   拼错 = 给外部使用方一个解析不了的域名 → 403 → 以为凭证无效，正是这几行要避免的事。

def test_access_lines_bare_region():
    """裸地域（服务器 1949 桶表的写法）→ 补上 `oss-` 前缀。"""
    from core.temp_ak_issuance import delivery
    lines = delivery._access_lines({"account": "1949", "region": "cn-shenzhen",
                                    "bucket": "wuji-product"})
    assert lines[0] == "地域：cn-shenzhen"
    assert lines[1] == "外网 Endpoint：oss-cn-shenzhen.aliyuncs.com"
    assert lines[2] == "桶域名：wuji-product.oss-cn-shenzhen.aliyuncs.com"
    assert "oss-oss-" not in "\n".join(lines)


def test_access_lines_already_prefixed_region_not_doubled():
    """已带前缀的地域（permsync.BUCKET_MAP 的写法）→ **不得**再补一层。

    这就是我报的那个 bug 的回归钉子：修前这里是 `oss-oss-ap-southeast-1.aliyuncs.com`。"""
    from core.temp_ak_issuance import delivery
    lines = delivery._access_lines({"account": "", "region": "oss-ap-southeast-1",
                                    "bucket": "wuji-sing"})
    assert lines[0] == "地域：oss-ap-southeast-1"
    assert lines[1] == "外网 Endpoint：oss-ap-southeast-1.aliyuncs.com"
    assert lines[2] == "桶域名：wuji-sing.oss-ap-southeast-1.aliyuncs.com"
    assert "oss-oss-" not in "\n".join(lines)


@pytest.mark.parametrize("region", ["", "   ", None])
def test_access_lines_empty_region_hints_unknown(region):
    """空/空白 region → 仍走「地域/Endpoint：未知」单行分支，不瞎猜、不拼出 `oss-.aliyuncs.com`。"""
    from core.temp_ak_issuance import delivery
    lines = delivery._access_lines({"account": "", "region": region, "bucket": "b"})
    assert len(lines) == 1
    assert lines[0].startswith("地域/Endpoint：未知")
    assert "aliyuncs.com" not in lines[0].split("外网 Endpoint")[0].replace("endpoint", "")
    assert "oss-.aliyuncs.com" not in lines[0]
    assert "桶域名" not in "\n".join(lines)


def test_access_lines_no_bucket_omits_bucket_host():
    from core.temp_ak_issuance import delivery
    lines = delivery._access_lines({"account": "", "region": "cn-shenzhen", "bucket": ""})
    assert len(lines) == 2 and "桶域名" not in "\n".join(lines)


@pytest.mark.parametrize("display,expect_host", [
    ("新加坡-wuji-sing", "oss-ap-southeast-1.aliyuncs.com"),      # BUCKET_MAP 展示名（带前缀 region）
    ("wuji-sing", "oss-ap-southeast-1.aliyuncs.com"),             # BUCKET_MAP 真实桶名反查
    ("北京-wuji-test-data", "oss-cn-beijing.aliyuncs.com"),
])
def test_resolve_bucket_to_endpoint_end_to_end(display, expect_host):
    """**端到端串起来**：resolve_bucket 拿到的 region 一路进凭证正文，绝不双前缀。"""
    from core.temp_ak_issuance import delivery
    region, bucket = o.resolve_bucket(display)
    text = "\n".join(delivery._access_lines({"account": "", "region": region, "bucket": bucket}))
    assert f"外网 Endpoint：{expect_host}" in text
    assert f"桶域名：{bucket}.{expect_host}" in text
    assert "oss-oss-" not in text


def test_credential_text_endpoint_not_doubled_for_prefixed_region():
    """整篇凭证正文层面再钉一次（使用方真正拿到的就是这段文本）。"""
    from core.temp_ak_issuance import delivery
    region, bucket = o.resolve_bucket("新加坡-wuji-sing")
    t = delivery.credential_text(_issued(region=region, bucket=bucket), _CREDS)
    assert "oss-oss-" not in t
    assert "oss-ap-southeast-1.aliyuncs.com" in t


def test_credential_text_missing_region_hints_instead_of_guessing():
    """桶没配地域映射时不瞎猜 endpoint（猜错 → OSS 403，使用方会以为凭证无效）。"""
    from core.temp_ak_issuance import delivery
    t = delivery.credential_text(_issued(_p1949(), region=""), _CREDS)
    assert "未知" in t
    assert "aliyuncs.com" not in t.split("有效期")[0].replace("未知", "")


def test_extended_text_uses_account_label_and_no_secret():
    from core.temp_ak_issuance import delivery
    t = delivery._extended_text(_issued(_p1949(), enterprise="张三"))
    assert "使用人名称：张三" in t
    assert "SK_SECRET_VALUE" not in t and "AccessKey Secret" not in t


def test_comment_user_id_override_per_account(monkeypatch):
    from core.temp_ak_issuance import delivery
    monkeypatch.setattr(ram_approval, "_approval_comment_user_id", lambda: "ou_admin_global")
    assert delivery._comment_user_id(_issued(_p1949())) == "ou_admin_global"
    monkeypatch.setattr(settings, "TEMP_AK_1949_COMMENT_USER_ID", "ou_1949_owner")
    assert delivery._comment_user_id(_issued(_p1949())) == "ou_1949_owner"
    # 默认档永不被 1949 的配置影响
    assert delivery._comment_user_id(_issued()) == "ou_admin_global"


def test_grant_record_still_has_no_secret_with_account_dimension(fake_redis):
    g9 = o.create_grant_record(_spec(), instance_code="i9sec", profile=_p1949())
    banned = {"access_key_secret", "security_token", "secret", "token", "sk"}
    assert banned.isdisjoint(g9.keys())
    raw = fake_redis.get("temp_ak_1949:grant:" + g9["grant_id"])
    assert "SK_1949" not in raw and "LTAI_1949" not in raw      # 账号 AK 不得落库


# ══════════════════════════════════════════════════════════════════════════════
# 发放事件按 code 定档（两账号互不误抢）
# ══════════════════════════════════════════════════════════════════════════════

def _issue_event(code, instance="inst_x", status="PASS"):
    return {"header": {"event_type": "approval_instance"},
            "event": {"approval_code": code, "status": status, "instance_code": instance}}


@pytest.mark.parametrize("code,slug", [(DEF_CODE, ""), (CODE_1949, "1949")])
def test_issue_profile_by_code(code, slug):
    assert approval._issue_profile(_issue_event(code)).slug == slug
    assert approval.should_handle_event(_issue_event(code)) is True


def test_issue_profile_unknown_code_none():
    assert approval._issue_profile(_issue_event("SOME-OTHER")) is None
    assert approval.should_handle_event(_issue_event("SOME-OTHER")) is False


def test_issue_profile_no_code_no_fallback():
    ev = {"header": {"event_type": "approval_instance"}, "event": {"instance_code": "i"}}
    assert approval._issue_profile(ev) is None


def test_issue_profile_1949_not_handled_when_fully_deregistered(monkeypatch):
    _deregister_1949(monkeypatch)
    assert approval.should_handle_event(_issue_event(CODE_1949)) is False
    assert approval.should_handle_event(_issue_event(DEF_CODE)) is True


def test_issue_profile_not_routed_when_only_code_removed(monkeypatch):
    """只删 code（AK 还在）：档案仍注册（前缀可用、清理照跑），但审批不被路由。"""
    monkeypatch.setattr(settings, "TEMP_AK_1949_APPROVAL_CODE", "")
    assert accounts.by_slug("1949").redis_prefix == "temp_ak_1949:"     # 档案还在
    assert approval.should_handle_event(_issue_event(CODE_1949)) is False
    assert accounts.issue_codes() == {DEF_CODE}


def test_issue_event_accepted_but_fails_loudly_when_ak_removed(monkeypatch, fake_redis, text_spy):
    """只删 AK（code 还在）：审批**仍被受理**（档案注册、白名单放行），
    然后在建号时**显式失败**——可观测的失败胜过静默失效。"""
    _strip_1949_ak(monkeypatch)
    assert approval.should_handle_event(_issue_event(CODE_1949)) is True
    assert CODE_1949 in accounts.issue_codes()
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _detail_1949())
    res = approval.handle_temp_ak_event(_issue_event(CODE_1949, instance="i9noak"))
    assert "error" in res
    assert "缺少 RAM 可写 AK" in res["error"]
    # grant 落在 1949 命名空间、状态 FAILED（后续补 AK 可重试）
    g = o.get_grant(o.grant_id_for("i9noak", _p1949()))
    assert g and g["stage"] == o.STAGE_FAILED and g["account"] == "1949"


def test_ram_client_error_message_names_the_missing_ak(monkeypatch):
    _strip_1949_ak(monkeypatch)
    with pytest.raises(accounts.UnknownAccountError) as e:
        accounts.ram_client(_p1949())
    assert "缺少 RAM 可写 AK" in str(e.value)
    assert "账号1949" in str(e.value)


def test_handle_issue_event_1949_end_to_end(monkeypatch, fake_redis):
    """1949 发放审批走通：grant 落 1949 命名空间 + account=1949 + 用 1949 的 AK 建号。"""
    monkeypatch.setattr(settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False, raising=False)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _detail_1949())
    seen = {}
    monkeypatch.setattr(issuer, "issue", lambda g: seen.setdefault("grant", g) and None or {
        "access_key_id": "LTAI_NEW", "access_key_secret": "SK", "security_token": "",
        "expire_ts": g["expire"], "mode": g["mode"]})
    monkeypatch.setattr(approval.delivery, "deliver", lambda g, c: seen.setdefault("delivered", g))

    res = approval.handle_temp_ak_event(_issue_event(CODE_1949, instance="i9"))
    assert res["ignored"] is False, res
    gid = res["grant_id"]
    assert gid.startswith("tak1949-")
    assert fake_redis.get("temp_ak_1949:grant:" + gid)
    assert fake_redis.get("temp_ak:grant:" + gid) is None
    assert seen["grant"]["account"] == "1949"
    assert seen["grant"]["user_name"].startswith("tempak-1949-")
    assert seen["delivered"]["grant_id"] == gid


def test_handle_issue_event_code_mismatch_between_event_and_detail(monkeypatch, fake_redis):
    """事件说 1949、回拉详情却是默认档的 code → 拒（防错账号发放）。"""
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: dict(_detail_1949(), approval_code=DEF_CODE))
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("code 不一致不得发放"))
    res = approval.handle_temp_ak_event(_issue_event(CODE_1949, instance="i9x"))
    assert res["reason"] == "approval_code_mismatch"


def test_handle_issue_event_1949_gated_on_instance_approved(monkeypatch, fake_redis):
    """#55 硬化门禁在 1949 档同样生效：实例非 APPROVED → 不发。"""
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: dict(_detail_1949(), status="PENDING"))
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("实例未 APPROVED 不得发放"))
    res = approval.handle_temp_ak_event(_issue_event(CODE_1949, instance="i9p"))
    assert res["ignored"] is True
    assert "PENDING" in res["reason"]


# ══════════════════════════════════════════════════════════════════════════════
# routes 白名单并入各档发放 code
# ══════════════════════════════════════════════════════════════════════════════

def test_allowlist_includes_both_issue_codes(monkeypatch):
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", "RAM-CODE")
    allow = routes._approval_allowlist()
    assert DEF_CODE in allow and CODE_1949 in allow and EXT_CODE in allow


def test_allowlist_excludes_account_without_code(monkeypatch):
    """没配审批 code 的账号不放行（注册 ≠ 放行）。"""
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", "RAM-CODE")
    monkeypatch.setattr(settings, "TEMP_AK_1949_APPROVAL_CODE", "")
    allow = routes._approval_allowlist()
    assert CODE_1949 not in allow
    assert DEF_CODE in allow


def test_allowlist_still_includes_account_missing_ak(monkeypatch):
    """配了 code 但缺 AK 时**仍放行**（MED-1 的取舍）：受理后在建号处显式报错，
    而不是把审批悄悄丢掉让申请人干等。"""
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", "RAM-CODE")
    _strip_1949_ak(monkeypatch)
    assert CODE_1949 in routes._approval_allowlist()


def test_allowlist_survives_accounts_failure(monkeypatch):
    """档案装配抛错时白名单仍含基础 code（不能因新账号把现有链路带崩）。"""
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", "RAM-CODE")
    monkeypatch.setattr(accounts, "issue_codes",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    allow = routes._approval_allowlist()
    assert DEF_CODE in allow and EXT_CODE in allow and "RAM-CODE" in allow
