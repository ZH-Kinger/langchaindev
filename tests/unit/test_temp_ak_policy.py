"""#58 续 —— policy.py 权限模型重构（read/download/write 三者正交）+ 时间窗。

权限模型（用户拍板）：
  · read     → 只给 List（ListObjects+桶信息/ACL），**无 GetObject（不能下载）**。
  · download → 只给 GetObject/GetObjectAcl，**无 ListObjects**。
  · write    → 只给 PutObject/Abort/ListParts，**绝无 DeleteObject、无 GetObject**。
每条 statement 叠时间窗（DateGreaterThan/DateLessThan +08:00 AND）+ 可选 IpAddress。
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from core.temp_ak_issuance import policy

_BJ = timezone(timedelta(hours=8))
NB = datetime(2026, 8, 1, 0, 0, 0, tzinfo=_BJ).timestamp()
EXP = datetime(2026, 8, 2, 0, 0, 0, tzinfo=_BJ).timestamp()


def _doc(caps, prefix="team/data/", **kw):
    return policy.build_policy_with_window(
        "b", prefix=prefix, caps=caps, not_before=NB, expire=EXP, **kw)


def _all_actions(doc):
    out = set()
    for s in doc["Statement"]:
        out.update(s["Action"])
    return out


def _stmt_by_action(doc, action):
    for s in doc["Statement"]:
        if action in s["Action"]:
            return s
    return None


# ── iso8601_bj ──────────────────────────────────────────────────────────────

def test_iso8601_bj_offset_and_format():
    assert policy.iso8601_bj(NB) == "2026-08-01T00:00:00+08:00"
    parsed = datetime.strptime(policy.iso8601_bj(NB), "%Y-%m-%dT%H:%M:%S%z")
    assert int(parsed.timestamp()) == int(NB)


# ── 三类能力正交 ─────────────────────────────────────────────────────────────

def test_read_gives_list_not_getobject():
    acts = _all_actions(_doc(["read"]))
    assert "oss:ListObjects" in acts
    assert "oss:GetBucketAcl" in acts          # 桶级读含 ACL
    assert "oss:GetObject" not in acts         # read 不能下载
    assert "oss:PutObject" not in acts


def test_download_gives_getobject_not_list():
    acts = _all_actions(_doc(["download"]))
    assert "oss:GetObject" in acts
    assert "oss:GetObjectAcl" in acts
    assert "oss:ListObjects" not in acts       # download 不含列举
    assert "oss:PutObject" not in acts


def test_write_gives_upload_no_delete_no_get():
    acts = _all_actions(_doc(["write"]))
    assert acts == {"oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"}
    assert "oss:DeleteObject" not in acts       # 外部绝不给删
    assert "oss:GetObject" not in acts
    assert "oss:ListObjects" not in acts


def test_combo_all_three_no_delete():
    acts = _all_actions(_doc(["read", "download", "write"]))
    assert "oss:ListObjects" in acts
    assert "oss:GetObject" in acts
    assert "oss:PutObject" in acts
    assert "oss:DeleteObject" not in acts       # 组合也永不含删


def test_read_download_combo():
    """真实多选 read+download → 有 List + GetObject，无上传/删除。"""
    acts = _all_actions(_doc(["read", "download"]))
    assert "oss:ListObjects" in acts
    assert "oss:GetObject" in acts
    assert "oss:PutObject" not in acts
    assert "oss:DeleteObject" not in acts


def test_read_plus_write_without_download_has_no_getobject():
    """read + write（无 download）→ 有 List + Put，但仍无 GetObject（下载须单独勾 download）。"""
    acts = _all_actions(_doc(["read", "write"]))
    assert "oss:ListObjects" in acts
    assert "oss:PutObject" in acts
    assert "oss:GetObject" not in acts


def test_empty_caps_empty_policy():
    doc = _doc([])
    assert doc["Statement"] == []


def test_module_action_sets_never_delete_or_cross_contaminate():
    assert "oss:DeleteObject" not in policy.WRITE_ACTIONS
    assert "oss:GetObject" not in policy.LIST_ACTIONS      # read 集不含下载
    assert "oss:ListObjects" not in policy.DOWNLOAD_ACTIONS
    assert "oss:GetBucketAcl" in policy.LIST_ACTIONS


# ── Resource / prefix ────────────────────────────────────────────────────────

def test_read_statement_bucket_resource_with_prefix_condition():
    doc = _doc(["read"], prefix="team/data/")
    st = _stmt_by_action(doc, "oss:ListObjects")
    assert st["Resource"] == ["acs:oss:*:*:b"]
    conds = st["Condition"]["StringLike"]["oss:Prefix"]
    assert "team/data/" in conds and "team/data/*" in conds


def test_read_whole_bucket_no_prefix_condition():
    doc = _doc(["read"], prefix="")
    st = _stmt_by_action(doc, "oss:ListObjects")
    assert "StringLike" not in st["Condition"]


def test_download_object_arn_with_prefix():
    doc = _doc(["download"], prefix="team/data/")
    st = _stmt_by_action(doc, "oss:GetObject")
    assert st["Resource"] == ["acs:oss:*:*:b/team/data/*"]


def test_download_object_arn_whole_bucket():
    doc = _doc(["download"], prefix="")
    st = _stmt_by_action(doc, "oss:GetObject")
    assert st["Resource"] == ["acs:oss:*:*:b/*"]


def test_write_object_arn_with_prefix():
    doc = _doc(["write"], prefix="drop/")
    st = _stmt_by_action(doc, "oss:PutObject")
    assert st["Resource"] == ["acs:oss:*:*:b/drop/*"]


# ── 时间窗（每条 statement）──────────────────────────────────────────────────

def test_every_statement_has_time_window():
    doc = _doc(["read", "download", "write"])
    assert doc["Statement"]
    for s in doc["Statement"]:
        cond = s["Condition"]
        assert cond["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
        assert cond["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)


def test_conditions_and_in_read_statement():
    """read 非整桶 statement 同 Condition 内 Date* + StringLike + IpAddress 共存 = AND。"""
    doc = _doc(["read"], prefix="p/", source_ips=["1.2.3.4"])
    st = _stmt_by_action(doc, "oss:ListObjects")
    cond = st["Condition"]
    assert "DateGreaterThan" in cond and "DateLessThan" in cond
    assert "StringLike" in cond and "IpAddress" in cond


def test_source_ips_inject_ipaddress_all_statements():
    doc = _doc(["read", "download", "write"], source_ips=["203.0.113.7"])
    for s in doc["Statement"]:
        assert s["Condition"]["IpAddress"]["acs:SourceIp"] == ["203.0.113.7"]


def test_no_source_ips_no_ipaddress():
    doc = _doc(["download"])
    for s in doc["Statement"]:
        assert "IpAddress" not in s["Condition"]


# ── session policy 上限 ───────────────────────────────────────────────────────

def test_session_policy_within_limit_ok():
    doc = policy.build_session_policy(
        "b", prefix="team/data/", caps=["read", "download", "write"],
        not_before=NB, expire=EXP)
    assert len(json.dumps(doc, ensure_ascii=False)) <= policy.SESSION_POLICY_MAX
    assert doc["Version"] == "1"


def test_session_policy_too_large_raises():
    huge_prefix = "seg/" * 700   # 撑爆 2048
    with pytest.raises(policy.PolicyTooLargeError):
        policy.build_session_policy(
            "bucket-name", prefix=huge_prefix, caps=["read", "download", "write"],
            not_before=NB, expire=EXP)


def test_policy_too_large_error_is_valueerror():
    assert issubclass(policy.PolicyTooLargeError, ValueError)
