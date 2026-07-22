"""#57 临时 AK 发放 —— 下发脱敏 + 私信发起人（重构后无邮箱）。

覆盖：
  · credential_card / credential_text：含 secret/token（仅私信发起人的凭证卡）。
  · receipt_card / status_card：**脱敏**——不含 access_key_secret / security_token。
  · grant 记录：不含 secret/token 键。
  · deliver：有 requester → 凭证卡私信发起人(target=requester) + 内部群脱敏回执；
    无 requester → 凭证不发(不进群防泄漏) + 内部群仍收回执 + ram 模式发醒目告警。
"""
import time

import pytest

from core.temp_ak_issuance import delivery, cards, orchestrator as o


CREDS_STS = {"access_key_id": "STS.AKID", "access_key_secret": "SUPER_SECRET_SK",
             "security_token": "SEC_TOKEN_XYZ", "expire_ts": 0.0, "mode": "sts"}
CREDS_RAM = {"access_key_id": "LTAI_RAM", "access_key_secret": "RAM_SECRET_SK",
             "security_token": "", "expire_ts": 0.0, "mode": "ram"}

_SECRETS = ["SUPER_SECRET_SK", "SEC_TOKEN_XYZ", "RAM_SECRET_SK"]


def _grant(**over):
    now = time.time()
    g = {
        "grant_id": "tak-deliv", "stage": o.STAGE_ISSUED, "mode": "sts",
        "platform": "aliyun", "enterprise": "外采公司A", "bucket": "wuji-sing",
        "read_prefixes": ["team/data/"], "write_prefixes": ["team/data/"],
        "not_before": now, "expire": now + 3600, "region": "oss-ap-southeast-1",
        "user_name": "tempak-ext-abc", "ak_id": "", "requester": "ou_requester",
        "source_ips": [],
    }
    g.update(over)
    return g


def _flatten(card_dict) -> str:
    import json
    return json.dumps(card_dict, ensure_ascii=False)


# ── credential_card 含 secret ────────────────────────────────────────────────

def test_credential_card_contains_secret_and_token():
    s = _flatten(cards.credential_card(_grant(), CREDS_STS))
    assert "SUPER_SECRET_SK" in s
    assert "SEC_TOKEN_XYZ" in s
    assert "STS.AKID" in s


def test_credential_text_ram_no_token_line():
    txt = delivery.credential_text(_grant(mode="ram"), CREDS_RAM)
    assert "RAM_SECRET_SK" in txt
    assert "SecurityToken" not in txt   # ram 无 token


def test_credential_text_sts_has_token_line():
    txt = delivery.credential_text(_grant(), CREDS_STS)
    assert "SecurityToken" in txt
    assert "SEC_TOKEN_XYZ" in txt


# ── receipt / status 卡脱敏 ──────────────────────────────────────────────────

def test_receipt_card_redacted():
    s = _flatten(cards.receipt_card(_grant(mode="ram", ak_id="LTAI_RAM")))
    for secret in _SECRETS:
        assert secret not in s
    # 企业/平台仍展示
    assert "外采公司A" in s


def test_status_card_redacted():
    s = _flatten(cards.status_card(_grant(stage=o.STAGE_ISSUED)))
    for secret in _SECRETS:
        assert secret not in s


def test_grant_record_has_no_secret_keys(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    now = time.time()
    g = o.create_grant_record({
        "platform": "aliyun", "enterprise": "E", "bucket": "b",
        "read_prefixes": ["r/"], "write_prefixes": [], "not_before": now,
        "expire": now + 3600, "recipient_email": "", "source_ips": [], "reason": "",
    }, instance_code="inst_deliv")
    banned = {"access_key_secret", "security_token", "secret", "token"}
    assert banned.isdisjoint(g.keys())


# ── deliver：私信发起人 + 内部群脱敏回执 ───────────────────────────────────────

@pytest.fixture
def send_spy(monkeypatch):
    """桩内部群 _send_card/_send_text（共享发送）+ 凭证私信的直连 HTTP（requests.post + token）。"""
    import core.dsw_scheduler as sched
    sent_cards = []   # (target, chat, card)  —— 内部群回执
    sent_text = []    # (target, chat, text)  —— 内部告警
    posts = []        # {"receive_id":.., "content":..} —— 私信凭证
    monkeypatch.setattr(sched, "_send_card",
                        lambda open_id, chat, card: sent_cards.append((open_id, chat, card)))
    monkeypatch.setattr(sched, "_send_text",
                        lambda open_id, chat, text: sent_text.append((open_id, chat, text)))
    monkeypatch.setattr(delivery.settings, "TEMP_AK_CHAT_ID", "oc_internal_group", raising=False)

    from tools.feishu import notify
    monkeypatch.setattr(notify, "_get_access_token", lambda: "tok")

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"code": 0, "msg": "ok"}

    import requests

    def fake_post(url, **kw):
        posts.append(kw.get("json") or {})
        return FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)
    return {"cards": sent_cards, "text": sent_text, "posts": posts}


def test_deliver_sends_creds_to_requester_and_receipt_to_group(send_spy):
    delivery.deliver(_grant(requester="ou_alice"), CREDS_STS)
    # 凭证私信：直连 HTTP，receive_id=发起人，content 含 secret/token
    assert len(send_spy["posts"]) == 1
    post = send_spy["posts"][0]
    assert post["receive_id"] == "ou_alice"
    assert "SUPER_SECRET_SK" in post["content"]
    assert "SEC_TOKEN_XYZ" in post["content"]
    # 内部群回执脱敏
    group_card = next(c[2] for c in send_spy["cards"] if c[1] == "oc_internal_group")
    assert "SUPER_SECRET_SK" not in _flatten(group_card)


def test_deliver_no_requester_no_creds_leak(send_spy):
    """无发起人 open_id → 凭证不发（不进群防泄漏、不发 HTTP）；内部群仍收脱敏回执。"""
    delivery.deliver(_grant(mode="ram", requester="", ak_id="LTAI_RAM"), CREDS_RAM)
    # 无凭证 HTTP 私信
    assert send_spy["posts"] == []
    # 群回执脱敏
    for _t, _c, card in send_spy["cards"]:
        assert "RAM_SECRET_SK" not in _flatten(card)
    assert any(c[1] == "oc_internal_group" for c in send_spy["cards"])


def _resp(status, code):
    class R:
        status_code = status

        @staticmethod
        def json():
            return {"code": code, "msg": "boom"}
    return R()


def test_deliver_feishu_nonzero_code_triggers_alert(send_spy, monkeypatch):
    """LOW-A：飞书返回 code!=0（凭证实际没送达）→ ram 模式触发醒目告警（不静默"以为送达"）。"""
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: _resp(200, 99991663))
    delivery.deliver(_grant(mode="ram", requester="ou_alice", ak_id="LTAI_RAM"), CREDS_RAM)
    assert send_spy["text"], "code!=0 应触发 undelivered 告警"
    assert send_spy["posts"] == []   # checked 发送内部无 posts 记录（用的是真 requests.post 桩，不入 posts）


def test_deliver_feishu_bad_status_triggers_alert(send_spy, monkeypatch):
    """LOW-A：status!=200 → 抛 → ram 告警。"""
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: _resp(500, 0))
    delivery.deliver(_grant(mode="ram", requester="ou_alice", ak_id="LTAI_RAM"), CREDS_RAM)
    assert send_spy["text"], "status!=200 应触发 undelivered 告警"


def test_deliver_feishu_ok_no_alert(send_spy):
    """LOW-A：code==0 且 status==200 → 正常、不告警（send_spy 默认返回 code0/200）。"""
    delivery.deliver(_grant(mode="ram", requester="ou_alice", ak_id="LTAI_RAM"), CREDS_RAM)
    assert send_spy["text"] == []
    assert len(send_spy["posts"]) == 1


def test_checked_send_error_message_excludes_secret(monkeypatch):
    """LOW-A：飞书失败抛出的异常信息只含 code/msg，绝不含卡片 body/secret。"""
    import requests
    from tools.feishu import notify
    monkeypatch.setattr(notify, "_get_access_token", lambda: "tok")
    monkeypatch.setattr(requests, "post", lambda url, **kw: _resp(200, 99991663))
    with pytest.raises(RuntimeError) as ei:
        delivery._send_creds_to_requester(_grant(requester="ou_alice"), CREDS_STS)
    msg = str(ei.value)
    for secret in _SECRETS:
        assert secret not in msg
    assert "99991663" in msg   # 只暴露 code


def test_deliver_ram_undelivered_alerts(send_spy):
    """方案 B 私信失败 → 醒目告警管理员（secret 不可恢复）。"""
    delivery.deliver(_grant(mode="ram", requester="", ak_id="LTAI_RAM"), CREDS_RAM)
    assert send_spy["text"], "应发内部告警"
    alert = send_spy["text"][-1][2]
    assert "revoke" in alert.lower() or "吊销" in alert
    # 告警本身不含 secret
    assert "RAM_SECRET_SK" not in alert


def test_deliver_sts_undelivered_no_alert(send_spy):
    """STS 私信失败不告警（token 到点自灭、无残留可清）——只 ram 才告警。"""
    delivery.deliver(_grant(mode="sts", requester=""), CREDS_STS)
    assert send_spy["text"] == []


def test_deliver_no_creds_only_receipt(send_spy):
    """creds=None（已发放短路场景）→ 不发凭证，只推内部回执。"""
    delivery.deliver(_grant(requester="ou_alice"), None)
    for _t, _c, card in send_spy["cards"]:
        for secret in _SECRETS:
            assert secret not in _flatten(card)
    # 仍推了群回执
    assert any(c[1] == "oc_internal_group" for c in send_spy["cards"])
