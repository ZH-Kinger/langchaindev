"""#59 B —— 下发改审批评论（delivery.py 重写；不再私信/credential_card/直连 HTTP）。

覆盖：
  · deliver(creds)：凭证走 ram_approval._send_approval_comment（贴发放审批实例，text 含 secret）
    + 内部群脱敏回执卡；评论失败 + ram → _alert_creds_undelivered。
  · deliver_extend：贴到延期实例（extend_instances[-1] 回退 instance_code）；creds→凭证评论、None→_extended_text(无 secret)。
  · credential_text 含 secret；_extended_text 无 secret；receipt/status 卡无 secret；grant 记录无 secret。
"""
import json
import time

import pytest

from core.temp_ak_issuance import delivery, cards, orchestrator as o
from core import ram_approval


CREDS_STS = {"access_key_id": "STS.AK", "access_key_secret": "SECRET_SK_XYZ",
             "security_token": "TOKEN_ABC", "expire_ts": 0.0, "mode": "sts"}
CREDS_RAM = {"access_key_id": "LTAI_RAM", "access_key_secret": "RAM_SECRET_SK",
             "security_token": "", "expire_ts": 0.0, "mode": "ram"}
_SECRETS = ["SECRET_SK_XYZ", "TOKEN_ABC", "RAM_SECRET_SK"]


def _grant(**over):
    now = time.time()
    g = {
        "grant_id": "tak-deliv", "stage": o.STAGE_ISSUED, "mode": "sts",
        "platform": "aliyun", "enterprise": "外采公司A", "bucket": "wuji-sing",
        "prefix": "team/data/", "caps": ["read", "download", "write"],
        "not_before": now, "expire": now + 3600, "region": "oss-ap-southeast-1",
        "user_name": "tempak-ext-abc", "ak_id": "", "requester": "ou_alice",
        "instance_code": "issue_inst_1", "source_ips": [],
    }
    g.update(over)
    return g


def _flat(c):
    return json.dumps(c, ensure_ascii=False)


@pytest.fixture
def spy(monkeypatch):
    """桩审批评论 + 内部群卡/文本，捕获调用。"""
    comments = []   # (instance_code, text, user_id)
    cards_sent = []
    text_sent = []
    monkeypatch.setattr(ram_approval, "_send_approval_comment",
                        lambda ic, text, uid, **k: comments.append((ic, text, uid)) or "cmt1")
    monkeypatch.setattr(ram_approval, "_approval_comment_user_id", lambda *a, **k: "ou_admin")
    import core.dsw_scheduler as sched
    monkeypatch.setattr(sched, "_send_card",
                        lambda oid, chat, card: cards_sent.append((oid, chat, card)))
    monkeypatch.setattr(sched, "_send_text",
                        lambda oid, chat, text: text_sent.append((oid, chat, text)))
    monkeypatch.setattr(delivery.settings, "TEMP_AK_CHAT_ID", "oc_group", raising=False)
    return {"comments": comments, "cards": cards_sent, "text": text_sent}


# ── deliver（发放）────────────────────────────────────────────────────────────

def test_deliver_posts_credential_comment_to_issue_instance(spy):
    delivery.deliver(_grant(mode="sts"), CREDS_STS)
    assert len(spy["comments"]) == 1
    ic, text, uid = spy["comments"][0]
    assert ic == "issue_inst_1"                    # 贴到发放审批实例
    assert "SECRET_SK_XYZ" in text                 # 评论正文含 secret
    assert "TOKEN_ABC" in text
    # #60：不再推内部群回执卡（信息只在审批评论里）
    assert spy["cards"] == []


def test_deliver_comment_user_id_uses_requester(spy):
    delivery.deliver(_grant(requester="ou_bob"), CREDS_STS)
    _ic, _text, uid = spy["comments"][0]
    assert uid == "ou_bob"


def test_deliver_comment_user_id_fallback_admin(spy):
    delivery.deliver(_grant(requester=""), CREDS_STS)
    _ic, _text, uid = spy["comments"][0]
    assert uid == "ou_admin"                       # 回退 _approval_comment_user_id


def test_deliver_comment_failure_ram_alerts(spy, monkeypatch):
    monkeypatch.setattr(ram_approval, "_send_approval_comment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    delivery.deliver(_grant(mode="ram", ak_id="LTAI_RAM"), CREDS_RAM)
    assert spy["text"], "ram 评论失败应发内部告警"
    alert = spy["text"][-1][2]
    assert "revoke" in alert.lower() or "吊销" in alert
    assert "RAM_SECRET_SK" not in alert            # 告警不含 secret


def test_deliver_sts_comment_failure_no_alert(spy, monkeypatch):
    """STS 评论失败不告警（token 自灭无残留）——只 ram 才告警。"""
    monkeypatch.setattr(ram_approval, "_send_approval_comment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    delivery.deliver(_grant(mode="sts"), CREDS_STS)
    assert spy["text"] == []


def test_deliver_no_creds_noop(spy):
    """creds=None → 既不评论也不推卡（#60：deliver 无内部回执）。"""
    delivery.deliver(_grant(), None)
    assert spy["comments"] == []
    assert spy["cards"] == []


# ── deliver_extend（延期）─────────────────────────────────────────────────────

def test_deliver_extend_posts_to_extend_instance(spy):
    """延期评论贴到延期实例（extend_instances[-1]），而非发放实例。"""
    g = _grant(mode="sts", instance_code="issue_inst_1",
               extend_instances=["ext_inst_1", "ext_inst_2"])
    delivery.deliver_extend(g, CREDS_STS)
    ic, text, _uid = spy["comments"][0]
    assert ic == "ext_inst_2"                      # 最近一次延期实例
    assert "SECRET_SK_XYZ" in text                 # 重签发 → 含新凭证


def test_deliver_extend_fallback_to_issue_instance(spy):
    """无 extend_instances → 回退发放 instance_code。"""
    g = _grant(mode="ram", instance_code="issue_inst_1", ak_id="LTAI_RAM")
    delivery.deliver_extend(g, None)               # 同 AK
    ic, _text, _uid = spy["comments"][0]
    assert ic == "issue_inst_1"


def test_deliver_extend_same_ak_no_secret(spy):
    """方案B 同 AK（creds=None）→ _extended_text 评论（无 secret）。"""
    g = _grant(mode="ram", extend_instances=["ext_inst_9"], ak_id="LTAI_RAM")
    delivery.deliver_extend(g, None)
    ic, text, _uid = spy["comments"][0]
    assert ic == "ext_inst_9"
    for secret in _SECRETS:
        assert secret not in text
    assert "延长" in text


def test_deliver_extend_reissue_has_secret(spy):
    g = _grant(mode="ram", extend_instances=["ext_inst_9"], ak_id="LTAI_RAM")
    delivery.deliver_extend(g, CREDS_RAM)          # 重签发
    _ic, text, _uid = spy["comments"][0]
    assert "RAM_SECRET_SK" in text


# ── 文本 / 卡片脱敏 ───────────────────────────────────────────────────────────

def test_credential_text_sts_has_secret_and_token():
    txt = delivery.credential_text(_grant(), CREDS_STS)
    assert "SECRET_SK_XYZ" in txt and "TOKEN_ABC" in txt


def test_credential_text_ram_no_token_line():
    txt = delivery.credential_text(_grant(mode="ram"), CREDS_RAM)
    assert "RAM_SECRET_SK" in txt
    assert "SecurityToken" not in txt


def test_extended_text_no_secret():
    txt = delivery._extended_text(_grant(mode="ram"))
    for secret in _SECRETS:
        assert secret not in txt
    assert "延长" in txt


def test_receipt_card_redacted():
    s = _flat(cards.receipt_card(_grant(mode="ram", ak_id="LTAI_RAM")))
    for secret in _SECRETS:
        assert secret not in s
    assert "外采公司A" in s


def test_status_card_redacted():
    s = _flat(cards.status_card(_grant()))
    for secret in _SECRETS:
        assert secret not in s


def test_grant_record_has_no_secret_keys(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    now = time.time()
    g = o.create_grant_record({
        "platform": "aliyun", "enterprise": "E", "bucket": "b",
        "prefix": "p/", "caps": ["read"], "not_before": now,
        "expire": now + 3600, "recipient_email": "", "source_ips": [], "reason": "",
    }, instance_code="inst_deliv")
    banned = {"access_key_secret", "security_token", "secret", "token"}
    assert banned.isdisjoint(g.keys())
