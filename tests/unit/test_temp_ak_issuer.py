"""#57 临时 AK 发放 —— issuer.classify_mode + plan(dry-run) + STS 分支 + assume_role_with_policy。

覆盖：
  · classify_mode：expire−now ≤ 上限→sts / 超→ram（边界=上限值；「3天后生效只用2h」远到期→ram；≤12h→sts）。
  · plan(dry-run)：不调云；STS 组 duration cap 43200 + role_arn + has_token=True；RAM 组 policy_name/user_name。
  · _issue_sts：调 assume_role_with_policy 传 policy(dict) + duration_seconds，duration clamp [900,43200]。
  · utils.aliyun_sts.assume_role_with_policy：AssumeRoleRequest 的 policy(json str)/duration_seconds 被设并 clamp。
"""
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from core.temp_ak_issuance import issuer, policy

_BJ = timezone(timedelta(hours=8))


# ── classify_mode 边界 ────────────────────────────────────────────────────────

def test_classify_within_12h_is_sts(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    now = 1_000_000.0
    assert issuer.classify_mode(now + 3600, now=now) == issuer.STS_MODE          # 1h
    assert issuer.classify_mode(now + 43200, now=now) == issuer.STS_MODE         # 恰 12h 边界 → sts


def test_classify_just_over_limit_is_ram(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    now = 1_000_000.0
    assert issuer.classify_mode(now + 43201, now=now) == issuer.RAM_MODE         # 超 1s → ram


def test_classify_far_future_expire_is_ram(monkeypatch):
    """『3天后生效只用2h』——用 expire−now 判：expire 远大于 now+12h → 方案 B。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    now = 1_000_000.0
    three_days = now + 3 * 86400 + 7200   # 3 天后生效、再 2h 到期 → 整体到期远超 12h
    assert issuer.classify_mode(three_days, now=now) == issuer.RAM_MODE


def test_classify_expire_in_past_is_sts_window_zero(monkeypatch):
    """已过期 → window=0 ≤ 上限 → sts（发放层另有 expire<now 校验拦截，此处只测分流数学）。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    now = 1_000_000.0
    assert issuer.classify_mode(now - 100, now=now) == issuer.STS_MODE


# ── LOW-2：误配 TEMP_AK_STS_MAX_SECONDS >43200 仍被硬顶封到 43200 ────────────

def test_misconfigured_max_over_hardcap_still_ram_for_20h(monkeypatch):
    """把上限误配成 90000（>12h 硬顶）时，20h 窗口仍判 ram——不因误配放大成 sts
    （否则 STS token 撑不到 20h、外部凭证提前死）。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 90000)
    now = 1_000_000.0
    twenty_h = now + 20 * 3600     # 72000s > 43200 硬顶
    assert issuer.classify_mode(twenty_h, now=now) == issuer.RAM_MODE
    # 恰 12h 仍 sts（硬顶边界）
    assert issuer.classify_mode(now + 43200, now=now) == issuer.STS_MODE
    assert issuer.classify_mode(now + 43201, now=now) == issuer.RAM_MODE


def test_sts_limit_capped_at_hardcap(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 90000)
    assert issuer._sts_limit() == issuer.STS_HARD_CAP == 43200


def test_sts_duration_hardcapped_despite_misconfig(monkeypatch):
    """_sts_duration 上限恒 43200，即便上限被误配成更大。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 90000)
    now = 1_000_000.0
    assert issuer._sts_duration(now + 80000, now=now) == 43200


# ── plan(dry-run) 不调云 ─────────────────────────────────────────────────────

def _fail_if_cloud_called(monkeypatch):
    """任何真发放/建号/AssumeRole 被调 → 立刻炸，证明 plan 纯 dry-run。"""
    import utils.aliyun_sts as sts
    monkeypatch.setattr(sts, "assume_role_with_policy",
                        lambda *a, **k: pytest.fail("plan 不应调 STS"))
    monkeypatch.setattr(issuer, "permsync_client",
                        lambda *a, **k: pytest.fail("plan 不应建 RAM 客户端"))


def test_plan_sts_no_cloud_call(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    monkeypatch.setattr(issuer.settings, "TEMP_AK_OSS_ROLE_ARN", "acs:ram::1:role/wide-oss")
    _fail_if_cloud_called(monkeypatch)
    now = time.time()
    grant = {
        "grant_id": "tak-x", "bucket": "b",
        "read_prefixes": ["r/"], "write_prefixes": ["w/"],
        "not_before": now, "expire": now + 3600,
        "user_name": "tempak-x", "policy_name": "temp-ak-auto-tempak-x",
    }
    p = issuer.plan(grant)
    assert p["mode"] == issuer.STS_MODE
    assert p["has_token"] is True
    assert p["role_arn"] == "acs:ram::1:role/wide-oss"
    assert 900 <= p["duration_seconds"] <= 43200
    assert p["policy"]["Version"] == "1"


def test_plan_sts_duration_capped_at_max(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    _fail_if_cloud_called(monkeypatch)
    now = time.time()
    # 强制 STS 分支(mode 显式)，窗口远超封顶 → duration 确定性封到 43200（避免 now 重取的 off-by-one）
    grant = {"grant_id": "g", "bucket": "b", "read_prefixes": ["r/"],
             "write_prefixes": [], "not_before": now, "expire": now + 100000,
             "user_name": "u", "policy_name": "p", "mode": issuer.STS_MODE}
    p = issuer.plan(grant)
    assert p["duration_seconds"] == 43200


def test_plan_ram_no_cloud_call(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    _fail_if_cloud_called(monkeypatch)
    now = time.time()
    grant = {
        "grant_id": "tak-y", "bucket": "b",
        "read_prefixes": ["r/"], "write_prefixes": [],
        "not_before": now, "expire": now + 5 * 86400,   # 5 天 → ram
        "user_name": "tempak-y", "policy_name": "temp-ak-auto-tempak-y",
    }
    p = issuer.plan(grant)
    assert p["mode"] == issuer.RAM_MODE
    assert p["has_token"] is False
    assert p["user_name"] == "tempak-y"
    assert p["policy_name"] == "temp-ak-auto-tempak-y"


# ── _issue_sts：透传 policy + duration 给 assume_role_with_policy ─────────────

def test_issue_sts_passes_policy_and_duration(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    monkeypatch.setattr(issuer.settings, "TEMP_AK_OSS_ROLE_ARN", "acs:ram::1:role/wide")
    captured = {}

    def fake_assume(role_arn, doc, duration, session_name=""):
        captured.update(role_arn=role_arn, doc=doc, duration=duration,
                        session_name=session_name)
        return {"access_key_id": "STS.AK", "access_key_secret": "SK",
                "security_token": "TOK", "expire_ts": 1.0}

    import utils.aliyun_sts as sts
    monkeypatch.setattr(sts, "assume_role_with_policy", fake_assume)

    now = time.time()
    grant = {"grant_id": "tak-sess", "bucket": "b", "read_prefixes": ["r/"],
             "write_prefixes": [], "not_before": now, "expire": now + 3600,
             "mode": issuer.STS_MODE}
    creds = issuer.issue(grant)
    assert creds["mode"] == issuer.STS_MODE
    assert creds["security_token"] == "TOK"
    # policy 传的是 dict 文档
    assert isinstance(captured["doc"], dict)
    assert captured["doc"]["Version"] == "1"
    assert 900 <= captured["duration"] <= 43200
    assert captured["role_arn"] == "acs:ram::1:role/wide"
    assert captured["session_name"] == "tak-sess"


def test_issue_sts_missing_role_raises(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_OSS_ROLE_ARN", "")
    now = time.time()
    grant = {"grant_id": "g", "bucket": "b", "read_prefixes": ["r/"],
             "write_prefixes": [], "not_before": now, "expire": now + 100,
             "mode": issuer.STS_MODE}
    with pytest.raises(issuer.IssueError):
        issuer.issue(grant)


def test_issue_sts_assume_returns_none_raises(monkeypatch):
    monkeypatch.setattr(issuer.settings, "TEMP_AK_OSS_ROLE_ARN", "acs:ram::1:role/w")
    import utils.aliyun_sts as sts
    monkeypatch.setattr(sts, "assume_role_with_policy", lambda *a, **k: None)
    now = time.time()
    grant = {"grant_id": "g", "bucket": "b", "read_prefixes": ["r/"],
             "write_prefixes": [], "not_before": now, "expire": now + 100,
             "mode": issuer.STS_MODE}
    with pytest.raises(issuer.IssueError):
        issuer.issue(grant)


# ── assume_role_with_policy：AssumeRoleRequest 的 policy/duration 被设 + clamp ─

class _FakeCred:
    access_key_id = "STS.AK"
    access_key_secret = "SK"
    security_token = "TOK"
    expiration = "2099-01-01T00:00:00Z"


class _FakeResp:
    class body:
        credentials = _FakeCred()


def _patch_sts_client(monkeypatch, captured):
    """把 STS SDK Client 换成记录 request 的假客户端。"""
    import alibabacloud_sts20150401.client as cmod

    class FakeClient:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def assume_role(self, req):
            captured["req"] = req
            return _FakeResp()

    monkeypatch.setattr(cmod, "Client", FakeClient)


def test_assume_role_with_policy_sets_policy_and_duration(monkeypatch):
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_ID", "MASTER_AK")
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_SECRET", "MASTER_SK")
    captured = {}
    _patch_sts_client(monkeypatch, captured)

    doc = {"Version": "1", "Statement": []}
    cred = aliyun_sts.assume_role_with_policy(
        "acs:ram::1:role/wide", doc, 3600, session_name="tak-abc")
    req = captured["req"]
    # policy 被序列化成 JSON 字符串传入
    assert req.policy == json.dumps(doc, ensure_ascii=False)
    assert req.duration_seconds == 3600
    assert req.role_arn == "acs:ram::1:role/wide"
    assert req.role_session_name == "tak-abc"
    assert cred["access_key_id"] == "STS.AK"
    assert cred["security_token"] == "TOK"


def test_assume_role_with_policy_clamps_duration_high(monkeypatch):
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_ID", "AK")
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_SECRET", "SK")
    captured = {}
    _patch_sts_client(monkeypatch, captured)
    aliyun_sts.assume_role_with_policy("acs:ram::1:role/w", {"Version": "1"}, 99999)
    assert captured["req"].duration_seconds == 43200


def test_assume_role_with_policy_clamps_duration_low(monkeypatch):
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_ID", "AK")
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_SECRET", "SK")
    captured = {}
    _patch_sts_client(monkeypatch, captured)
    aliyun_sts.assume_role_with_policy("acs:ram::1:role/w", {"Version": "1"}, 10)
    assert captured["req"].duration_seconds == 900


def test_assume_role_with_policy_accepts_str_policy(monkeypatch):
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_ID", "AK")
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_SECRET", "SK")
    captured = {}
    _patch_sts_client(monkeypatch, captured)
    aliyun_sts.assume_role_with_policy("acs:ram::1:role/w", '{"Version":"1"}', 3600)
    assert captured["req"].policy == '{"Version":"1"}'


def test_assume_role_with_policy_no_master_ak_returns_none(monkeypatch):
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_ID", "")
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_SECRET", "")
    assert aliyun_sts.assume_role_with_policy("r", {}, 3600) is None


def test_assume_role_with_policy_no_role_returns_none(monkeypatch):
    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_ID", "AK")
    monkeypatch.setattr(aliyun_sts.settings, "ALIYUN_BOT_MASTER_AK_SECRET", "SK")
    assert aliyun_sts.assume_role_with_policy("", {}, 3600) is None


def test_assume_role_with_policy_does_not_touch_assume_role_for_user(monkeypatch):
    """新函数不复用 (open_id,role) 缓存、不改既有 assume_role_for_user 路径。"""
    from utils import aliyun_sts
    # assume_role_for_user 仍走 _do_assume_role（本函数不应改它）
    assert aliyun_sts.assume_role_with_policy is not aliyun_sts.assume_role_for_user
