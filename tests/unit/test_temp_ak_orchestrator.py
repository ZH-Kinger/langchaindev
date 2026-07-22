"""#57 临时 AK 发放 —— orchestrator 状态机 / Redis / 幂等 / 桶解析。

覆盖：
  · grant_id_for：同实例同值 / 异实例异值 / tak- 前缀。
  · create_grant_record：首建 NEW+mode+user_name、记录不含 secret/token 键；幂等返旧不复位。
  · issue_grant：已 ISSUED 短路返 (grant,None) 不重发；成功置 ISSUED、方案 B 落 ak_id。
  · resolve_bucket：TEMP_AK_BUCKET_MAP → permsync.BUCKET_MAP → 原样 三路。
  · claim：NX 幂等（同实例第二次拿不到锁）。
fakeredis（conftest 自动注入）。
"""
import time

import pytest

from core.temp_ak_issuance import orchestrator as o, issuer


# ── grant_id_for ──────────────────────────────────────────────────────────────

def test_grant_id_prefix_and_determinism():
    a = o.grant_id_for("inst_1")
    b = o.grant_id_for("inst_1")
    assert a == b
    assert a.startswith("tak-")


def test_grant_id_differs_per_instance():
    assert o.grant_id_for("inst_1") != o.grant_id_for("inst_2")


# ── create_grant_record ───────────────────────────────────────────────────────

def _spec(**over):
    now = time.time()
    base = {
        "bucket": "wuji-sing",
        "read_prefixes": ["r/", "r/"],   # 含重复，验去重
        "write_prefixes": ["w/"],
        "not_before": now,
        "expire": now + 3600,            # 1h → sts
        "recipient_email": "ext@example.com",
        "source_ips": [],
        "reason": "sell data",
    }
    base.update(over)
    return base


def test_create_record_new_fields(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g = o.create_grant_record(_spec(), instance_code="inst_A", requester="ou_req")
    assert g["stage"] == o.STAGE_NEW
    assert g["mode"] == issuer.STS_MODE            # 1h → sts
    assert g["user_name"].startswith("tempak-")
    assert g["instance_code"] == "inst_A"
    assert g["requester"] == "ou_req"
    assert g["read_prefixes"] == ["r/"]            # 去重排序
    assert g["bucket"] == "wuji-sing"


def test_create_record_has_no_secret_or_token_keys(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g = o.create_grant_record(_spec(), instance_code="inst_secret")
    banned = {"access_key_secret", "security_token", "secret", "token", "sk"}
    assert banned.isdisjoint(g.keys())
    assert g["ak_id"] == ""                        # 建号前未知


def test_create_record_idempotent_returns_existing(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g1 = o.create_grant_record(_spec(), instance_code="inst_idem")
    # 篡改到 ISSUED 落库，再次 create 应原样返回、不复位到 NEW
    g1["stage"] = o.STAGE_ISSUED
    o._save(g1)
    g2 = o.create_grant_record(_spec(reason="different"), instance_code="inst_idem")
    assert g2["stage"] == o.STAGE_ISSUED
    assert g2["grant_id"] == g1["grant_id"]


def test_create_record_ram_mode_sets_policy_name(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    now = time.time()
    g = o.create_grant_record(_spec(expire=now + 5 * 86400), instance_code="inst_ram")
    assert g["mode"] == issuer.RAM_MODE
    assert g["policy_name"] == issuer.policy.POLICY_PREFIX + g["user_name"]


def test_create_record_sts_mode_no_policy_name(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g = o.create_grant_record(_spec(), instance_code="inst_sts")
    assert g["mode"] == issuer.STS_MODE
    assert g["policy_name"] == ""


# ── issue_grant ───────────────────────────────────────────────────────────────

def test_issue_grant_success_ram_sets_ak_id(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    now = time.time()
    g = o.create_grant_record(_spec(expire=now + 5 * 86400), instance_code="inst_iss_ram")
    monkeypatch.setattr(issuer, "issue", lambda grant: {
        "access_key_id": "LTAI_RAM_AK", "access_key_secret": "SK",
        "security_token": "", "expire_ts": g["expire"], "mode": issuer.RAM_MODE})
    g2, creds = o.issue_grant(g)
    assert g2["stage"] == o.STAGE_ISSUED
    assert g2["ak_id"] == "LTAI_RAM_AK"
    assert "issued_ts" in g2
    # 落库的 grant 仍不含 secret
    stored = o.get_grant(g["grant_id"])
    assert "access_key_secret" not in stored
    assert creds["access_key_secret"] == "SK"      # creds 仅当次返回


def test_issue_grant_already_issued_short_circuits(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g = o.create_grant_record(_spec(), instance_code="inst_short")
    g["stage"] = o.STAGE_ISSUED
    o._save(g)
    monkeypatch.setattr(issuer, "issue",
                        lambda grant: pytest.fail("已 ISSUED 不得重发凭证"))
    g2, creds = o.issue_grant(g)
    assert creds is None
    assert g2["stage"] == o.STAGE_ISSUED


def test_issue_grant_revoked_short_circuits(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g = o.create_grant_record(_spec(), instance_code="inst_rev")
    g["stage"] = o.STAGE_REVOKED
    o._save(g)
    monkeypatch.setattr(issuer, "issue",
                        lambda grant: pytest.fail("已 REVOKED 不得重发"))
    g2, creds = o.issue_grant(g)
    assert creds is None


def test_issue_grant_sts_does_not_store_ak_id(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    g = o.create_grant_record(_spec(), instance_code="inst_sts_iss")   # sts
    monkeypatch.setattr(issuer, "issue", lambda grant: {
        "access_key_id": "STS.AK", "access_key_secret": "SK",
        "security_token": "TOK", "expire_ts": g["expire"], "mode": issuer.STS_MODE})
    g2, creds = o.issue_grant(g)
    assert g2["stage"] == o.STAGE_ISSUED
    assert g2["ak_id"] == ""     # STS 不落 ak_id（无需清理）
    assert creds["security_token"] == "TOK"


# ── resolve_bucket 三路 ───────────────────────────────────────────────────────

def test_resolve_bucket_from_temp_ak_map(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW",
                        '{"外部桶": {"region": "oss-cn-shanghai", "bucket": "real-ext"}}')
    region, bucket = o.resolve_bucket("外部桶")
    assert region == "oss-cn-shanghai"
    assert bucket == "real-ext"


def test_resolve_bucket_falls_back_to_permsync_map(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    # permsync.BUCKET_MAP 内含「新加坡-wuji-sing」
    region, bucket = o.resolve_bucket("新加坡-wuji-sing")
    assert bucket == "wuji-sing"
    assert region == "oss-ap-southeast-1"


def test_resolve_bucket_passthrough_real_name(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    region, bucket = o.resolve_bucket("some-raw-bucket")
    assert bucket == "some-raw-bucket"
    assert region == ""


def test_resolve_bucket_empty_raises(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    with pytest.raises(o.TempAkError):
        o.resolve_bucket("")


def test_resolve_bucket_bad_json_falls_through(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{not json")
    region, bucket = o.resolve_bucket("some-raw-bucket")
    assert bucket == "some-raw-bucket"


# ── claim NX 幂等 ─────────────────────────────────────────────────────────────

def test_claim_nx_idempotent():
    first = o.claim("inst_lock")
    assert first
    second = o.claim("inst_lock")
    assert second == ""            # 已被占，拿不到
    o.release(first)
    third = o.claim("inst_lock")   # 释放后可重拿
    assert third


def test_claim_empty_instance_returns_empty():
    assert o.claim("") == ""
