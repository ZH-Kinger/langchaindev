"""#57 临时 AK 发放 —— policy.py 时间窗 policy 生成。

覆盖：
  · iso8601_bj 时区正确（+08:00）。
  · build_policy_with_window：每条 statement 含 DateGreaterThan/DateLessThan（值为 ISO8601 +08:00）；
    同 statement 多运算符共存(AND)；List 语句非整桶时带 oss:Prefix、桶级动作含 GetBucketAcl；
    对象写动作集不含 DeleteObject；source_ips 注入 IpAddress(acs:SourceIp)；
    read/write 空集不出对应 statement；'' 前缀=整桶(无 oss:Prefix)。
  · build_session_policy：≤2048 通过、超限抛 PolicyTooLargeError。
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from core.temp_ak_issuance import policy

_BJ = timezone(timedelta(hours=8))
# 固定生效/到期 epoch（北京墙钟）
NB = datetime(2026, 8, 1, 0, 0, 0, tzinfo=_BJ).timestamp()
EXP = datetime(2026, 8, 2, 0, 0, 0, tzinfo=_BJ).timestamp()


def _stmt_by_action(doc, action):
    """返回第一条 Action 含指定动作的 statement。"""
    for s in doc["Statement"]:
        if action in s["Action"]:
            return s
    return None


# ── iso8601_bj ──────────────────────────────────────────────────────────────

def test_iso8601_bj_offset_and_format():
    assert policy.iso8601_bj(NB) == "2026-08-01T00:00:00+08:00"
    assert policy.iso8601_bj(EXP) == "2026-08-02T00:00:00+08:00"
    # 带时区解析回来 epoch 一致
    parsed = datetime.strptime(policy.iso8601_bj(NB), "%Y-%m-%dT%H:%M:%S%z")
    assert int(parsed.timestamp()) == int(NB)


# ── 时间窗条件（每条 statement）──────────────────────────────────────────────

def test_every_statement_has_date_window_conditions():
    doc = policy.build_policy_with_window(
        "wuji-sing", read_prefixes=["r/"], write_prefixes=["w/"],
        not_before=NB, expire=EXP)
    assert doc["Version"] == "1"
    assert doc["Statement"], "至少一条 statement"
    for s in doc["Statement"]:
        cond = s["Condition"]
        assert cond["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
        assert cond["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)


def test_conditions_coexist_in_same_statement_as_AND():
    """同一 Condition dict 内 DateGreaterThan+DateLessThan(+StringLike/IpAddress) 即 AND。"""
    doc = policy.build_policy_with_window(
        "b", read_prefixes=["team/"], not_before=NB, expire=EXP,
        source_ips=["1.2.3.4"])
    list_stmt = _stmt_by_action(doc, "oss:GetBucketAcl")
    cond = list_stmt["Condition"]
    # 三类运算符共存于同一 Condition = AND
    assert "DateGreaterThan" in cond
    assert "DateLessThan" in cond
    assert "StringLike" in cond            # 非整桶前缀
    assert "IpAddress" in cond             # source_ips


# ── 桶级动作 / GetBucketAcl / DeleteObject ───────────────────────────────────

def test_bucket_statement_includes_getbucketacl():
    doc = policy.build_policy_with_window(
        "b", read_prefixes=["r/"], not_before=NB, expire=EXP)
    bucket_stmt = _stmt_by_action(doc, "oss:GetBucketAcl")
    assert bucket_stmt is not None
    # 桶级 Resource 只到桶（无对象通配）
    assert bucket_stmt["Resource"] == ["acs:oss:*:*:b"]
    # 仍含既有列举动作
    assert "oss:ListObjects" in bucket_stmt["Action"]


def test_write_actions_exclude_delete_object():
    doc = policy.build_policy_with_window(
        "b", write_prefixes=["w/"], not_before=NB, expire=EXP)
    write_stmt = _stmt_by_action(doc, "oss:PutObject")
    assert write_stmt is not None
    assert "oss:DeleteObject" not in write_stmt["Action"]
    # 外部写只给上传/分片
    assert set(write_stmt["Action"]) == {
        "oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"}


def test_module_write_action_set_never_has_delete():
    assert "oss:DeleteObject" not in policy.WRITE_OBJECT_ACTIONS
    assert "oss:GetBucketAcl" in policy.BUCKET_ACTIONS


# ── source_ips → IpAddress ───────────────────────────────────────────────────

def test_source_ips_inject_ipaddress_condition():
    doc = policy.build_policy_with_window(
        "b", read_prefixes=["r/"], not_before=NB, expire=EXP,
        source_ips=["203.0.113.7", "198.51.100.0/24"])
    read_stmt = _stmt_by_action(doc, "oss:GetObject")
    assert read_stmt["Condition"]["IpAddress"]["acs:SourceIp"] == \
        ["203.0.113.7", "198.51.100.0/24"]


def test_no_source_ips_no_ipaddress_condition():
    doc = policy.build_policy_with_window(
        "b", read_prefixes=["r/"], not_before=NB, expire=EXP)
    for s in doc["Statement"]:
        assert "IpAddress" not in s["Condition"]


# ── read/write 空集 ───────────────────────────────────────────────────────────

def test_empty_read_no_read_statement():
    doc = policy.build_policy_with_window(
        "b", write_prefixes=["w/"], not_before=NB, expire=EXP)
    assert _stmt_by_action(doc, "oss:GetObject") is None
    assert _stmt_by_action(doc, "oss:PutObject") is not None


def test_empty_write_no_write_statement():
    doc = policy.build_policy_with_window(
        "b", read_prefixes=["r/"], not_before=NB, expire=EXP)
    assert _stmt_by_action(doc, "oss:PutObject") is None
    assert _stmt_by_action(doc, "oss:GetObject") is not None


def test_both_empty_only_bucket_statement():
    doc = policy.build_policy_with_window("b", not_before=NB, expire=EXP)
    # 只剩桶级 List 语句
    assert len(doc["Statement"]) == 1
    assert _stmt_by_action(doc, "oss:GetBucketAcl") is not None
    # 无前缀 → 无 oss:Prefix
    assert "StringLike" not in doc["Statement"][0]["Condition"]


# ── '' 前缀 = 整桶（无 oss:Prefix）───────────────────────────────────────────

def test_empty_prefix_means_whole_bucket_no_prefix_condition():
    doc = policy.build_policy_with_window(
        "b", read_prefixes=[""], not_before=NB, expire=EXP)
    read_stmt = _stmt_by_action(doc, "oss:GetObject")
    # 整桶对象 ARN
    assert read_stmt["Resource"] == ["acs:oss:*:*:b/*"]
    bucket_stmt = _stmt_by_action(doc, "oss:GetBucketAcl")
    assert "StringLike" not in bucket_stmt["Condition"]


def test_nonempty_prefix_adds_oss_prefix_condition():
    doc = policy.build_policy_with_window(
        "b", read_prefixes=["team/"], write_prefixes=["drop/"],
        not_before=NB, expire=EXP)
    bucket_stmt = _stmt_by_action(doc, "oss:GetBucketAcl")
    conds = bucket_stmt["Condition"]["StringLike"]["oss:Prefix"]
    assert "team/" in conds and "team/*" in conds
    assert "drop/" in conds and "drop/*" in conds


# ── session policy 上限 ───────────────────────────────────────────────────────

def test_session_policy_within_limit_ok():
    doc = policy.build_session_policy(
        "b", read_prefixes=["r/"], write_prefixes=["w/"],
        not_before=NB, expire=EXP)
    assert len(json.dumps(doc, ensure_ascii=False)) <= policy.SESSION_POLICY_MAX
    # 与 build_policy_with_window 同结构
    assert doc["Version"] == "1"


def test_session_policy_too_large_raises():
    # 灌大量长前缀撑爆 2048
    big = [f"very/long/prefix/segment/number-{i:04d}/data/" for i in range(200)]
    with pytest.raises(policy.PolicyTooLargeError):
        policy.build_session_policy(
            "bucket-name", read_prefixes=big, write_prefixes=big,
            not_before=NB, expire=EXP)


def test_policy_too_large_error_is_valueerror():
    assert issubclass(policy.PolicyTooLargeError, ValueError)
