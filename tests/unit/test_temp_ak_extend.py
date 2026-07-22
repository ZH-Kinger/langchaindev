"""#58/#60 —— 临时 AK 延长/撤销二合一审批入口（approval / orchestrator / issuer）。

#60：延期审批改「延长/撤销二合一」，加 撤销/延长 单选（widget17847115181700001），DateInterval 非必填。
覆盖：
  · should_handle_extend_event：精确 == TEMP_AK_EXTEND_APPROVAL_CODE；发放 code 不命中。
  · _parse_extend_action：撤销/吊销/撤回/revoke→revoke；延长/空/其它→extend（默认，向后兼容）。
  · parse_temp_ak_extend_request：现返 **5 元组** (action, grant_id, enterprise, not_before, expire)；
    缺凭证ID→抛；缺到期不再在 parse 抛（挪到 handler 按 action 校验）。
  · _verify_enterprise（MED-1 fail-safe）：orig 有值+ent 空→抛；orig 空→跳过；匹配→ok；不符→抛。
  · handle_temp_ak_extend_event：
    - 延长分支：实例非 APPROVED/无 instance_code 不动、grant 不存在/REVOKED/企业不符/缺到期各抛、幂等。
    - 撤销分支：APPROVED→防串→cleanup.revoke_grant 被调、落 revoke_instances 幂等、已 REVOKED 短路、
      部分失败 ok=False 通知人工、防串对撤销也生效。
  · orchestrator.extend_grant：方案B creds=None+改窗+extends；STS≤12h 重签发；STS>12h 转 ram+ak 更新；
    stage!=ISSUED 抛；新到期已过/生效≥到期 抛。
  · issuer.rewrite_ram_window：mock RAM 断言 create_policy_version(set_as_default,policy_name)、缺 policy_name 抛。
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
                action="延长", date_interval=None, code=EXT, by_id=False):
    date_interval = date_interval if date_interval is not None else \
        {"start": "", "end": "2099-06-01 00:00:00"}
    names = {
        "grant_id": "widget17846399101070001" if by_id else "凭证ID",
        "action": "widget17847115181700001" if by_id else "撤销/延长",
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
            {key: names["action"], "value": action},
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
    """现返回 5 元组 (action, grant_id, enterprise, not_before, expire)。默认 action=extend。"""
    detail = _ext_detail("APPROVED", grant_id="tak-xyz", enterprise="外采公司B", by_id=by_id)
    action, gid, ent, nb, exp = approval.parse_temp_ak_extend_request(detail, {})
    assert action == "extend"
    assert gid == "tak-xyz"
    assert ent == "外采公司B"
    assert exp > 0
    assert nb == 0.0     # 空 start → 0（extend_grant 补原 not_before）


def test_parse_extend_with_start():
    detail = _ext_detail("APPROVED",
                         date_interval={"start": "2027-01-01 00:00:00", "end": "2027-06-01 00:00:00"})
    action, gid, ent, nb, exp = approval.parse_temp_ak_extend_request(detail, {})
    assert nb > 0 and exp > nb


def test_parse_extend_revoke_action():
    detail = _ext_detail("APPROVED", action="撤销")
    action, gid, ent, nb, exp = approval.parse_temp_ak_extend_request(detail, {})
    assert action == "revoke"


def test_parse_extend_missing_grant_id_raises():
    detail = _ext_detail("APPROVED", grant_id="")
    with pytest.raises(o.TempAkError):
        approval.parse_temp_ak_extend_request(detail, {})


def test_parse_extend_missing_expire_no_longer_raises_in_parse():
    """expire 校验挪到 handler（撤销不需要到期）→ parse 不再因缺到期抛。"""
    detail = _ext_detail("APPROVED", date_interval={"start": "", "end": ""})
    action, gid, ent, nb, exp = approval.parse_temp_ak_extend_request(detail, {})
    assert exp == 0.0     # 缺到期 → 0，由 handler 按 action 校验


# ── _parse_extend_action ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["撤销", "吊销", "撤回", "revoke", "Revoke", "REVOKE"])
def test_parse_extend_action_revoke(raw):
    assert approval._parse_extend_action(raw) == "revoke"


@pytest.mark.parametrize("raw", ["延长", "延期", "", "其它随便", None])
def test_parse_extend_action_extend_default(raw):
    assert approval._parse_extend_action(raw) == "extend"


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
    calls = {"extend": 0, "deliver": 0, "revoke": 0, "notify": []}

    def fake_extend(grant, nb, exp, *, extend_instance=""):
        calls["extend"] += 1
        grant["expire"] = float(exp)
        return grant, None

    monkeypatch.setattr(o, "extend_grant", fake_extend)
    monkeypatch.setattr(delivery, "deliver_extend",
                        lambda g, c: calls.__setitem__("deliver", calls["deliver"] + 1))
    # 撤销分支：桩 cleanup.revoke_grant（默认成功）+ 内部通知
    from core.temp_ak_issuance import cleanup
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda g, *a, **k: (calls.__setitem__("revoke", calls["revoke"] + 1) or True))
    monkeypatch.setattr(approval, "_notify_internal_action",
                        lambda g, text: calls["notify"].append(text))
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


def test_extend_missing_expire_rejected_in_handler(monkeypatch):
    """延长分支缺到期（DateInterval 空）→ handler 抛（校验从 parse 挪到 handler）。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _ext_detail("APPROVED", action="延长",
                                                 date_interval={"start": "", "end": ""}))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["extend"] == 0
    assert "error" in res


# ══════════════════════════════════════════════════════════════════════════════
# handle_temp_ak_extend_event —— 撤销分支（action=revoke）
# ══════════════════════════════════════════════════════════════════════════════

def _revoke_detail(instance_status="APPROVED", *, grant_id="tak-abc", enterprise="外采公司A"):
    return _ext_detail(instance_status, grant_id=grant_id, enterprise=enterprise,
                       action="撤销", date_interval={"start": "", "end": ""})


def test_revoke_branch_calls_cleanup(monkeypatch):
    """撤销分支：门禁 APPROVED → 防串 → cleanup.revoke_grant 被调 → 不走 extend/deliver。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _revoke_detail())
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["revoke"] == 1
    assert calls["extend"] == 0
    assert calls["deliver"] == 0
    assert res["ignored"] is False
    assert res["action"] == "revoke"
    assert res["ok"] is True
    assert calls["notify"]                       # 内部通知发出


def test_revoke_branch_records_revoke_instance(monkeypatch):
    """撤销后落 revoke_instances（幂等基础）。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _revoke_detail())
    approval.handle_temp_ak_extend_event(_ext_event(instance="ext_inst_1"))
    stored = o.get_grant("tak-abc")
    assert "ext_inst_1" in (stored.get("revoke_instances") or [])


def test_revoke_idempotent_same_instance(monkeypatch):
    """同一撤销实例二次投递 → revoke_already_applied，不重复删。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant(revoke_instances=["ext_inst_1"]))
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _revoke_detail())
    res = approval.handle_temp_ak_extend_event(_ext_event(instance="ext_inst_1"))
    assert calls["revoke"] == 0
    assert res["reason"] == "revoke_already_applied"


def test_revoke_already_revoked_short_circuits(monkeypatch):
    """grant 已 REVOKED → already_revoked 短路、不再调 cleanup。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant(stage=o.STAGE_REVOKED))
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _revoke_detail())
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["revoke"] == 0
    assert res["reason"] == "already_revoked"
    assert calls["notify"]                       # 通知「已是撤销状态」


def test_revoke_enterprise_mismatch_rejected(monkeypatch):
    """防串对撤销分支同样生效：不能拿别家 grant_id 来撤。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant(enterprise="甲公司"))
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _revoke_detail(enterprise="乙公司"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["revoke"] == 0
    assert "error" in res


def test_revoke_grant_not_found_rejected(monkeypatch):
    calls = _spy_extend(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _revoke_detail(grant_id="tak-missing"))
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert calls["revoke"] == 0
    assert "error" in res


def test_revoke_partial_failure_notifies(monkeypatch):
    """cleanup.revoke_grant 返 False（部分失败）→ ok=False + 通知人工核查。"""
    calls = _spy_extend(monkeypatch)
    from core.temp_ak_issuance import cleanup
    monkeypatch.setattr(cleanup, "revoke_grant", lambda g, *a, **k: False)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _revoke_detail())
    res = approval.handle_temp_ak_extend_event(_ext_event())
    assert res["action"] == "revoke"
    assert res["ok"] is False
    assert any("核查" in t or "失败" in t for t in calls["notify"])


def test_revoke_pending_instance_no_action(monkeypatch):
    """撤销实例非 APPROVED → 门禁拦下，不删。"""
    calls = _spy_extend(monkeypatch)
    o._save(_issued_grant())
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _revoke_detail(instance_status="PENDING"))
    res = approval.handle_temp_ak_extend_event(_ext_event(status="PASS"))
    assert calls["revoke"] == 0
    assert res["ignored"] is True


# ── _notify_internal_action（真函数 → dsw_scheduler._send_text）───────────────

def test_notify_internal_action_sends_text(monkeypatch):
    sent = []
    import core.dsw_scheduler as sched
    monkeypatch.setattr(sched, "_send_text",
                        lambda oid, chat, text: sent.append((oid, chat, text)))
    monkeypatch.setattr(approval.settings, "TEMP_AK_CHAT_ID", "oc_group", raising=False)
    approval._notify_internal_action(_issued_grant(grant_id="tak-n", enterprise="外采公司X"),
                                     "✅ 凭证已按审批撤销")
    assert sent
    _oid, chat, text = sent[-1]
    assert chat == "oc_group"
    assert "tak-n" in text and "外采公司X" in text
    assert "撤销" in text


def test_notify_internal_action_no_chat_noop(monkeypatch):
    called = []
    import core.dsw_scheduler as sched
    monkeypatch.setattr(sched, "_send_text", lambda *a, **k: called.append(1))
    monkeypatch.setattr(approval.settings, "TEMP_AK_CHAT_ID", "", raising=False)
    monkeypatch.setattr(approval.settings, "FEISHU_CHAT_ID", "", raising=False)
    approval._notify_internal_action(_issued_grant(), "x")
    assert called == []            # 无群配置 → 不发


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


# 注：deliver_extend 的审批评论下发 + _extended_text 脱敏在 test_temp_ak_delivery.py 覆盖；
#     #59 已移除 cards.extended_card（延期改走审批评论 _extended_text，非卡片）。
