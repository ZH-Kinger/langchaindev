"""#58 B —— P0.5 临时 AK 延期入口（approval / orchestrator / issuer / delivery / cards）。

覆盖：
  · should_handle_extend_event：精确 == TEMP_AK_EXTEND_APPROVAL_CODE；发放 code 不命中。
  · parse_temp_ak_extend_request：3 控件（凭证ID/使用企业信息/DateInterval），by_id + by_name；缺凭证ID/到期各抛。
  · _verify_enterprise（MED-1 fail-safe）：orig 有值+ent 空→抛；orig 空→跳过；匹配→ok；不符→抛。
  · handle_temp_ak_extend_event 门禁：实例非 APPROVED/无 instance_code 不动；grant 不存在→抛；
    grant REVOKED→抛；企业不符→抛；同一 extend 实例二次投递→extend_already_applied 不重复应用。
  · orchestrator.extend_grant：方案B creds=None+改窗+extends；STS≤12h 重签发；STS>12h 转 ram+ak 更新；
    stage!=ISSUED 抛；新到期已过/生效≥到期 抛。
  · issuer.rewrite_ram_window：mock RAM 断言 create_policy_version(set_as_default,policy_name)、缺 policy_name 抛。
  · delivery.deliver_extend：creds→凭证卡；None→extended_card（无 secret）；都推内部回执脱敏。
"""
import json
import time

import pytest

from core.temp_ak_issuance import approval, orchestrator as o, issuer, delivery, cards
from core import ram_approval

EXT = "EXTCODE-EXT-9999"
TAK = "TAKCODE-ISSUE-1111"


@pytest.fixture(autouse=True)
def _codes(monkeypatch):
    monkeypatch.setattr(approval.settings, "TEMP_AK_EXTEND_APPROVAL_CODE", EXT)
    monkeypatch.setattr(approval.settings, "TEMP_AK_APPROVAL_CODE", TAK)
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")


def _ext_event(code=EXT, status="APPROVED", instance="ext_inst_1"):
    ev = {"header": {"event_type": "approval_instance"}, "event": {}}
    if code:
        ev["event"]["approval_code"] = code
    if status:
        ev["event"]["status"] = status
    if instance:
        ev["event"]["instance_code"] = instance
    return ev


def _ext_detail(instance_status, *, grant_id="tak-abc", enterprise="外采公司A",
                date_interval=None, code=EXT, by_id=False):
    date_interval = date_interval if date_interval is not None else \
        {"start": "", "end": "2099-06-01 00:00:00"}
    names = {
        "grant_id": "widget17846399101070001" if by_id else "凭证ID",
        "enterprise": "widget17846914004480001" if by_id else "使用企业信息",
        "date_interval": "widget17846911942480001" if by_id else "DateInterval",
    }
    key = "id" if by_id else "name"
    return {
        "status": instance_status,
        "approval_code": code,
        "instance_code": "ext_inst_1",
        "form": [
            {key: names["grant_id"], "value": grant_id},
            {key: names["enterprise"], "value": enterprise},
            {key: names["date_interval"], "value": date_interval},
        ],
    }


def _issued_grant(**over):
    now = time.time()
    g = {
        "grant_id": "tak-abc", "stage": o.STAGE_ISSUED, "mode": "ram",
        "platform": "aliyun", "enterprise": "外采公司A", "bucket": "wuji-sing",
        "prefix": "team/data/", "caps": ["read", "download", "write"],
        "not_before": now - 100, "expire": now + 3600,
        "user_name": "tempak-ext-abc", "policy_name": "temp-ak-auto-tempak-ext-abc",
        "ak_id": "LTAI_ORIG", "requester": "ou_alice", "source_ips": [],
    }
    g.update(over)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# should_handle_extend_event
# ══════════════════════════════════════════════════════════════════════════════

def test_should_handle_extend_exact_match():
    assert approval.should_handle_extend_event(_ext_event(code=EXT)) is True


def test_should_handle_extend_rejects_issue_code():
    assert approval.should_handle_extend_event(_ext_event(code=TAK)) is False


def test_should_handle_extend_no_code_false():
    ev = {"header": {"event_type": "approval_instance"}, "event": {"instance_code": "i"}}
    assert approval.should_handle_extend_event(ev) is False


def test_should_handle_extend_target_unset(monkeypatch):
    monkeypatch.setattr(approval.settings, "TEMP_AK_EXTEND_APPROVAL_CODE", "")
    assert approval.should_handle_extend_event(_ext_event(code=EXT)) is False


# ══════════════════════════════════════════════════════════════════════════════
# parse_temp_ak_extend_request
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("by_id", [False, True])
def test_parse_extend_full(by_id):
    detail = _ext_detail("APPROVED", grant_id="tak-xyz", enterprise="外采公司B", by_id=by_id)
    gid, ent, nb, exp = approval.parse_temp_ak_extend_request(detail, {})
    assert gid == "tak-xyz"
    assert ent == "外采公司B"
    assert exp > 0
    assert nb == 0.0     # 空 start → 0（extend_grant 补原 not_before）


def test_parse_extend_with_start():
    detail = _ext_detail("APPROVED",
                         date_interval={"start": "2027-01-01 00:00:00", "end": "2027-06-01 00:00:00"})
    gid, ent, nb, exp = approval.parse_temp_ak_extend_request(detail, {})
    assert nb > 0 and exp > nb


def test_parse_extend_missing_grant_id_raises():
    detail = _ext_detail("APPROVED", grant_id="")
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_extend_request(detail, {})


def test_parse_extend_missing_expire_raises():
    detail = _ext_detail("APPROVED", date_interval={"start": "", "end": ""})
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_extend_request(detail, {})


# ══════════════════════════════════════════════════════════════════════════════
# _verify_enterprise（MED-1 fail-safe）
# ══════════════════════════════════════════════════════════════════════════════

def test_verify_enterprise_orig_set_ent_empty_rejected():
    """MED-1：原凭证有企业名 + 延期表单留空 → 抛（防「引用别家 grant_id + 留空绕过」）。"""
    with pytest.raises(o.TempAkError):
        approval._verify_enterprise({"enterprise": "甲公司"}, "")


def test_verify_enterprise_orig_empty_skips():
    """原凭证本身无企业名 → 无可比对象 → 跳过（不抛）。"""
    approval._verify_enterprise({"enterprise": ""}, "")
    approval._verify_enterprise({"enterprise": ""}, "任意公司")   # orig 空也跳过


def test_verify_enterprise_exact_match_ok():
    approval._verify_enterprise({"enterprise": "甲公司"}, "甲公司")


def test_verify_enterprise_loose_contains_ok():
    approval._verify_enterprise({"enterprise": "甲公司"}, "甲公司（北京）有限")
    approval._verify_enterprise({"enterprise": "甲公司（北京）"}, "甲公司")


def test_verify_enterprise_mismatch_rejected():
    with pytest.raises(o.TempAkError):
        approval._verify_enterprise({"enterprise": "甲公司"}, "乙公司")


# ══════════════════════════════════════════════════════════════════════════════
# handle_temp_ak_extend_event 门禁 + 幂等
# ══════════════════════════════════════════════════════════════════════════════

def _spy_extend(monkeypatch):
    calls = {"extend": 0, "deliver": 0}

    def fake_extend(grant, nb, exp, *, extend_instance=""):
        calls["extend"] += 1
        grant["expire"] = float(exp)
        return grant, None

    monkeypatch.setattr(o, "extend_grant", fake_extend)
    monkeypatch.setattr(delivery, "deliver_extend",
                        lambda g, c: calls.__setitem__("deliver", calls["deliver"] + 1))
    return calls


def test_extend_instance_pending_no_action(monkeypatch):
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("PENDING"))
    res = approval.handle_temp_ak_extend_event(_ext_event(status="PASS"))
    assert calls["extend"] == 0
    assert res["ignored"] is True
    assert "PENDING" in res["reason"]


def test_extend_no_instance_code_fail_safe(monkeypatch):
    calls = _spy_extend(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: pytest.fail("无 instance_code 不该回拉"))
    res = approval.handle_temp_ak_extend_event(_ext_event(status="PASS", instance=""))
    assert calls["extend"] == 0
    assert res["reason"] == "no_instance_code"


def test_extend_grant_not_found_errors(monkeypatch):
    calls = _spy_extend(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("APPROVED", grant_id="tak-missing"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["extend"] == 0
    assert "error" in res


def test_extend_grant_revoked_errors(monkeypatch):
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant(stage=o.STAGE_REVOKED))
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("APPROVED"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["extend"] == 0
    assert "error" in res
    assert "清理" in res["error"] or "revok" in res["error"].lower()


def test_extend_enterprise_mismatch_errors(monkeypatch):
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant(enterprise="甲公司"))
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("APPROVED", enterprise="乙公司"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["extend"] == 0
    assert "error" in res


def test_extend_happy_path(monkeypatch):
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("APPROVED"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["extend"] == 1
    assert calls["deliver"] == 1
    assert res["ignored"] is False
    assert res["grant_id"] == "tak-abc"


def test_extend_idempotent_second_delivery(monkeypatch):
    """同一 extend 实例二次投递 → extend_already_applied，不重复 extend。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant(extend_instances=["ext_inst_1"]))
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("APPROVED"))
    res = approval.handle_temp_ak_extend_event(_ext_event(instance="ext_inst_1"))
    assert calls["extend"] == 0
    assert res["reason"] == "extend_already_applied"


# ══════════════════════════════════════════════════════════════════════════════
# orchestrator.extend_grant
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_ram_rewrites_window_no_creds(monkeypatch):
    """方案B：改写时间窗、AK 不变、creds=None、extends 追加。"""
    called = []
    monkeypatch.setattr(issuer, "rewrite_ram_window", lambda g: called.append(g["grant_id"]))
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("方案B 延期不该重签发"))
    g = _issued_grant(mode="ram")
    o._save(g)
    now = time.time()
    new_exp = now + 10 * 86400
    g2, creds = o.extend_grant(g, 0, new_exp, extend_instance="ext_inst_1")
    assert creds is None
    assert g2["expire"] == float(new_exp)
    assert g2["ak_id"] == "LTAI_ORIG"           # AK 不变
    assert called == ["tak-abc"]                # 走了 rewrite_ram_window
    assert g2["extends"] and g2["extends"][-1]["new_expire"] == float(new_exp)
    assert "ext_inst_1" in g2["extend_instances"]


def test_extend_sts_short_window_reissues_sts(monkeypatch):
    """STS 新窗 ≤12h → 仍 sts 重签发。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    monkeypatch.setattr(issuer, "rewrite_ram_window", lambda g: pytest.fail("STS 不该改写 RAM policy"))
    monkeypatch.setattr(issuer, "issue", lambda g: {
        "access_key_id": "STS.NEW", "access_key_secret": "SK", "security_token": "TOK",
        "expire_ts": g["expire"], "mode": "sts"})
    g = _issued_grant(mode="sts", policy_name="", ak_id="")
    o._save(g)
    now = time.time()
    g2, creds = o.extend_grant(g, 0, now + 3600)
    assert g2["mode"] == "sts"
    assert creds["security_token"] == "TOK"
    assert g2["ak_id"] == ""                    # STS 不落 ak_id


def test_extend_sts_long_window_switches_to_ram(monkeypatch):
    """STS 新窗 >12h → 自动转方案B、发长期 AK、ak_id 更新。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    monkeypatch.setattr(issuer, "issue", lambda g: {
        "access_key_id": "LTAI_NEW", "access_key_secret": "SK", "security_token": "",
        "expire_ts": g["expire"], "mode": "ram"})
    g = _issued_grant(mode="sts", policy_name="", ak_id="")
    o._save(g)
    now = time.time()
    g2, creds = o.extend_grant(g, 0, now + 5 * 86400)   # 5 天 > 12h
    assert g2["mode"] == "ram"
    assert g2["ak_id"] == "LTAI_NEW"
    assert g2["policy_name"]                     # 转 ram 补 policy_name
    assert creds["access_key_id"] == "LTAI_NEW"


@pytest.mark.parametrize("stage", ["NEW", "REVOKED", "FAILED"])
def test_extend_non_issued_rejected(monkeypatch, stage):
    monkeypatch.setattr(issuer, "rewrite_ram_window", lambda g: None)
    g = _issued_grant(stage=stage)
    with pytest.raises(o.TempAkError):
        o.extend_grant(g, 0, time.time() + 3600)


def test_extend_new_expire_past_rejected():
    g = _issued_grant(mode="ram")
    with pytest.raises(o.TempAkError):
        o.extend_grant(g, 0, time.time() - 100)


def test_extend_not_before_ge_expire_rejected():
    g = _issued_grant(mode="ram")
    now = time.time()
    with pytest.raises(o.TempAkError):
        o.extend_grant(g, now + 7200, now + 3600)


# ══════════════════════════════════════════════════════════════════════════════
# issuer.rewrite_ram_window
# ══════════════════════════════════════════════════════════════════════════════

class _FakeRamClient:
    def __init__(self):
        self.calls = []

    def create_policy_version(self, req):
        self.calls.append(("create_policy_version", req))


def test_rewrite_ram_window_sets_default_version(monkeypatch):
    from core.oss_perm import permsync
    fake = _FakeRamClient()
    monkeypatch.setattr(permsync, "make_ram_client", lambda: fake)
    now = time.time()
    grant = {"grant_id": "tak-abc", "bucket": "b", "prefix": "r/", "caps": ["read"], "not_before": now, "expire": now + 10 * 86400,
             "policy_name": "temp-ak-auto-tempak-x", "source_ips": []}
    issuer.rewrite_ram_window(grant)
    assert len(fake.calls) == 1
    name, req = fake.calls[0]
    assert name == "create_policy_version"
    assert req.set_as_default is True
    assert req.policy_name == "temp-ak-auto-tempak-x"
    # policy_document 含时间窗 +08:00
    assert "+08:00" in req.policy_document


def test_rewrite_ram_window_missing_policy_name_raises(monkeypatch):
    from core.oss_perm import permsync
    monkeypatch.setattr(permsync, "make_ram_client",
                        lambda: pytest.fail("缺 policy_name 不该建客户端"))
    now = time.time()
    grant = {"grant_id": "g", "bucket": "b", "prefix": "r/", "caps": ["read"],
             "not_before": now, "expire": now + 100, "policy_name": "", "source_ips": []}
    with pytest.raises(issuer.IssueError):
        issuer.rewrite_ram_window(grant)


# ══════════════════════════════════════════════════════════════════════════════
# delivery.deliver_extend
# ══════════════════════════════════════════════════════════════════════════════

CREDS = {"access_key_id": "STS.AK", "access_key_secret": "EXTEND_SECRET_SK",
         "security_token": "EXTEND_TOK", "expire_ts": 0.0, "mode": "sts"}


@pytest.fixture
def send_spy(monkeypatch):
    import core.dsw_scheduler as sched
    sent_cards, sent_text, posts = [], [], []
    monkeypatch.setattr(sched, "_send_card",
                        lambda oid, chat, card: sent_cards.append((oid, chat, card)))
    monkeypatch.setattr(sched, "_send_text",
                        lambda oid, chat, text: sent_text.append((oid, chat, text)))
    monkeypatch.setattr(delivery.settings, "TEMP_AK_CHAT_ID", "oc_group", raising=False)
    from tools.feishu import notify
    monkeypatch.setattr(notify, "_get_access_token", lambda: "tok")

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"code": 0}

    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: posts.append(kw.get("json") or {}) or FakeResp())
    return {"cards": sent_cards, "text": sent_text, "posts": posts}


def _flat(c):
    return json.dumps(c, ensure_ascii=False)


def test_deliver_extend_reissue_sends_credential(send_spy):
    """STS 重签发（creds 非空）→ 私信新凭证卡（含 secret）+ 内部回执脱敏。"""
    delivery.deliver_extend(_issued_grant(mode="sts", requester="ou_alice"), CREDS)
    assert len(send_spy["posts"]) == 1
    assert "EXTEND_SECRET_SK" in send_spy["posts"][0]["content"]
    # 内部群回执脱敏
    group = next(c[2] for c in send_spy["cards"] if c[1] == "oc_group")
    assert "EXTEND_SECRET_SK" not in _flat(group)


def test_deliver_extend_same_ak_sends_extended_card_no_secret(send_spy):
    """方案B 同 AK（creds=None）→ extended_card（无 secret）+ 内部回执。"""
    delivery.deliver_extend(_issued_grant(mode="ram", requester="ou_alice"), None)
    # 私信一张 extended_card（HTTP post），不含 secret/token
    assert len(send_spy["posts"]) == 1
    content = send_spy["posts"][0]["content"]
    assert "EXTEND_SECRET_SK" not in content
    assert "EXTEND_TOK" not in content
    # content 是 json.dumps(card)（ensure_ascii 转义），解回来再核文案
    decoded = _flat(json.loads(content))
    assert "延长" in decoded
    # 内部回执也推
    assert any(c[1] == "oc_group" for c in send_spy["cards"])


def test_extended_card_no_secret():
    s = _flat(cards.extended_card(_issued_grant(mode="ram")))
    for secret in ("EXTEND_SECRET_SK", "EXTEND_TOK"):
        assert secret not in s
    assert "延长" in s
