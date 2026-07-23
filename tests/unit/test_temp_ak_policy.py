"""临时 AK 发放 policy.py —— 严格对齐用户手动固化的权威模板（temp-ak-auto-tempak-nuoyiteng-7df6a7）。

线上 bug 根因（本文件锁死不重现）：原实现把桶信息动作（GetBucketInfo/Stat/Acl）与 ListObjects
塞进同一条 read 语句、且整条带了 oss:Prefix 的 StringLike 条件。GetBucket* 是桶级操作、请求不带
prefix 参数，被 oss:Prefix 条件卡死 → 拒绝 → 用户拿到凭证却访问不了桶。

修复后模型（build_policy_with_window，caps ⊆ {read,download,write}，三者正交）：
  · 桶信息条  → Action=[GetBucketInfo,GetBucketStat,GetBucketAcl]，Resource=[桶]，**无 Condition**
               （无 oss:Prefix、无时间窗）。read 或 download 任一勾选即给。
  · List 条   → Action=[ListObjects,GetBucketMultipartUploads]，Resource=[桶]，
               Condition=时间窗 +（prefix 非空时）oss:Prefix StringLike[prefix, prefix+"*"]。read 勾选给。
  · 下载条    → Action=[GetObject]，Resource=[桶/前缀*]，Condition=时间窗��download 勾选给。
  · 写条      → Action=[PutObject,AbortMultipartUpload,ListParts]（无任何 delete），
               Resource=[桶/前缀*]，Condition=时间窗。write 勾选给。
时间窗 = DateGreaterThan/DateLessThan on acs:CurrentTime（ISO8601 +08:00，AND）。
四条同时出现时顺序 = 桶信息 / List / 下载 / 写。
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from core.temp_ak_issuance import policy

_BJ = timezone(timedelta(hours=8))
NB = datetime(2026, 8, 1, 0, 0, 0, tzinfo=_BJ).timestamp()
EXP = datetime(2026, 8, 2, 0, 0, 0, tzinfo=_BJ).timestamp()

# 桶信息动作三件套（就是当初被误塞进带 Prefix 的 List 语句、导致被拒的那三个）。
_BUCKET_INFO = {"oss:GetBucketInfo", "oss:GetBucketStat", "oss:GetBucketAcl"}


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


def _bucket_info_stmt(doc):
    """含 GetBucketInfo 的那条（桶信息条）。无则 None。"""
    return _stmt_by_action(doc, "oss:GetBucketInfo")


def _list_stmt(doc):
    """含 ListObjects 的那条（List 条）。无则 None。"""
    return _stmt_by_action(doc, "oss:ListObjects")


# ── iso8601_bj ──────────────────────────────────────────────────────────────

def test_iso8601_bj_offset_and_format():
    assert policy.iso8601_bj(NB) == "2026-08-01T00:00:00+08:00"
    parsed = datetime.strptime(policy.iso8601_bj(NB), "%Y-%m-%dT%H:%M:%S%z")
    assert int(parsed.timestamp()) == int(NB)


# ── 单勾 read：桶信息条(无Condition) + List条(带Prefix+时间窗) ──────────────────

def test_read_only_yields_bucketinfo_plus_list():
    doc = _doc(["read"], prefix="team/data/")
    assert len(doc["Statement"]) == 2

    info = _bucket_info_stmt(doc)
    lst = _list_stmt(doc)
    assert info is not None and lst is not None

    # 桶信息条：Action==三件套、Resource=桶、**完全无 Condition**（无 oss:Prefix、无时间窗）
    assert set(info["Action"]) == _BUCKET_INFO
    assert info["Resource"] == ["acs:oss:*:*:b"]
    assert "Condition" not in info

    # List 条：ListObjects + GetBucketMultipartUploads、Resource=桶、带时间窗 + oss:Prefix
    assert set(lst["Action"]) == {"oss:ListObjects", "oss:GetBucketMultipartUploads"}
    assert lst["Resource"] == ["acs:oss:*:*:b"]
    prefs = lst["Condition"]["StringLike"]["oss:Prefix"]
    assert prefs == ["team/data/", "team/data/*"]
    assert lst["Condition"]["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
    assert lst["Condition"]["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)


def test_read_only_list_stmt_has_no_getobject_and_no_bucketinfo_actions():
    """List 条不含 GetObject（read 不能下载），也不含桶信息三件套（避免重现被拒 bug）。"""
    lst = _list_stmt(_doc(["read"]))
    assert "oss:GetObject" not in lst["Action"]
    assert not (_BUCKET_INFO & set(lst["Action"]))


def test_read_only_no_getobject_anywhere():
    assert "oss:GetObject" not in _all_actions(_doc(["read"]))
    assert "oss:PutObject" not in _all_actions(_doc(["read"]))


# ── 单勾 download：桶信息条 + GetObject条(Resource=桶/前缀*) ─────────────────────

def test_download_only_yields_bucketinfo_plus_getobject():
    doc = _doc(["download"], prefix="team/data/")
    assert len(doc["Statement"]) == 2

    info = _bucket_info_stmt(doc)
    dl = _stmt_by_action(doc, "oss:GetObject")
    assert info is not None and dl is not None

    assert set(info["Action"]) == _BUCKET_INFO
    assert "Condition" not in info

    # 下载条：Resource 是 桶/前缀*（对象级），不是桶
    assert dl["Action"] == ["oss:GetObject"]
    assert dl["Resource"] == ["acs:oss:*:*:b/team/data/*"]
    assert dl["Resource"] != ["acs:oss:*:*:b"]
    assert dl["Condition"]["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
    assert dl["Condition"]["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)


def test_download_only_no_list_no_write():
    acts = _all_actions(_doc(["download"]))
    assert "oss:ListObjects" not in acts       # download 不含列举
    assert "oss:PutObject" not in acts


def test_download_object_arn_whole_bucket():
    dl = _stmt_by_action(_doc(["download"], prefix=""), "oss:GetObject")
    assert dl["Resource"] == ["acs:oss:*:*:b/*"]


# ── 单勾 write：只出写条、无 delete、且不出桶信息条 ─────────────────────────────

def test_write_only_single_statement_no_bucketinfo():
    doc = _doc(["write"], prefix="drop/")
    assert len(doc["Statement"]) == 1           # 只有写条
    st = doc["Statement"][0]
    assert set(st["Action"]) == {"oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"}
    assert st["Resource"] == ["acs:oss:*:*:b/drop/*"]
    # write 不触发桶信息条
    assert _bucket_info_stmt(doc) is None


def test_write_only_no_delete_no_get_no_list():
    acts = _all_actions(_doc(["write"]))
    assert "oss:DeleteObject" not in acts
    assert "oss:DeleteMultipleObjects" not in acts
    assert "oss:GetObject" not in acts
    assert "oss:ListObjects" not in acts
    assert not (_BUCKET_INFO & acts)            # write 不给桶信息


# ── 全勾 read+download+write：四条，顺序=桶信息/List/下载/写 ─────────────────────

def test_all_three_four_statements_in_order():
    doc = _doc(["read", "download", "write"], prefix="team/data/")
    stmts = doc["Statement"]
    assert len(stmts) == 4

    # 顺序：桶信息 → List → 下载 → 写
    assert set(stmts[0]["Action"]) == _BUCKET_INFO
    assert "Condition" not in stmts[0]
    assert set(stmts[1]["Action"]) == {"oss:ListObjects", "oss:GetBucketMultipartUploads"}
    assert stmts[2]["Action"] == ["oss:GetObject"]
    assert set(stmts[3]["Action"]) == {"oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"}


def test_all_three_no_delete():
    acts = _all_actions(_doc(["read", "download", "write"]))
    assert "oss:DeleteObject" not in acts


# ── 关键回归（防 bug 重现）───────────────────────────────────────────────────

@pytest.mark.parametrize("caps", [["read"], ["download"], ["read", "download"],
                                  ["read", "download", "write"]])
def test_bucketinfo_statement_never_carries_prefix_condition(caps):
    """桶信息条【绝不】带 oss:Prefix / 任何 Condition —— 直接锁死线上 bug 根因。"""
    info = _bucket_info_stmt(_doc(caps, prefix="team/data/"))
    assert info is not None
    assert "Condition" not in info              # 无时间窗、更无 oss:Prefix
    # 保险：即便将来加了 Condition，也绝不能出现 oss:Prefix
    assert "oss:Prefix" not in json.dumps(info)


@pytest.mark.parametrize("caps", [["read"], ["read", "download"],
                                  ["read", "download", "write"]])
def test_getbucket_info_actions_absent_from_prefixed_list_statement(caps):
    """带 oss:Prefix 条件的 List 语句里【不】出现 GetBucketInfo/Stat/Acl（当初被拒的病灶）。"""
    doc = _doc(caps, prefix="team/data/")
    # 找出所有带 oss:Prefix 条件的语句
    prefixed = [s for s in doc["Statement"]
                if "StringLike" in s.get("Condition", {})
                and "oss:Prefix" in s["Condition"]["StringLike"]]
    assert prefixed, "read 非整桶时应有一条带 oss:Prefix 的 List 语句"
    for s in prefixed:
        assert not (_BUCKET_INFO & set(s["Action"]))     # 桶信息三件套绝不在此
        assert "oss:GetObject" not in s["Action"]


def test_module_action_sets_no_cross_contamination():
    assert "oss:DeleteObject" not in policy.WRITE_ACTIONS
    assert "oss:GetObject" not in policy.LIST_ACTIONS       # read 集不含下载
    assert "oss:ListObjects" not in policy.DOWNLOAD_ACTIONS
    # 桶信息三件套独立成集，不混入 List（否则会被带上 oss:Prefix 而被拒）
    for a in _BUCKET_INFO:
        assert a in policy.BUCKET_INFO_ACTIONS
        assert a not in policy.LIST_ACTIONS


# ── caps 为空 → 空 Statement ─────────────────────────────────────────────────

def test_empty_caps_empty_policy():
    doc = _doc([])
    assert doc["Statement"] == []


def test_none_caps_empty_policy():
    doc = policy.build_policy_with_window(
        "b", prefix="p/", caps=None, not_before=NB, expire=EXP)
    assert doc["Statement"] == []


# ── prefix 为空（整桶）──────────────────────────────────────────────────────

def test_read_whole_bucket_no_prefix_condition():
    lst = _list_stmt(_doc(["read"], prefix=""))
    assert "StringLike" not in lst["Condition"]         # 整桶：不带 oss:Prefix
    # 但时间窗仍在
    assert "DateGreaterThan" in lst["Condition"]
    # 整桶时桶信息条仍存在且无 Condition
    info = _bucket_info_stmt(_doc(["read"], prefix=""))
    assert "Condition" not in info


# ── 时间窗 / IP：作用于 List/下载/写，不作用于桶信息条 ────────────────────────

def test_time_window_on_conditioned_statements_only():
    """有 Condition 的语句（List/下载/写）都带 Date*；桶信息条无 Condition。"""
    doc = _doc(["read", "download", "write"])
    conditioned = [s for s in doc["Statement"] if "Condition" in s]
    assert len(conditioned) == 3                        # List + 下载 + 写
    for s in conditioned:
        assert s["Condition"]["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
        assert s["Condition"]["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)
    assert "Condition" not in _bucket_info_stmt(doc)


def test_read_list_condition_is_and_of_date_prefix_ip():
    """read 非整桶 List 语句：Date* + StringLike + IpAddress 同 Condition = AND。"""
    lst = _list_stmt(_doc(["read"], prefix="p/", source_ips=["1.2.3.4"]))
    cond = lst["Condition"]
    assert "DateGreaterThan" in cond and "DateLessThan" in cond
    assert "StringLike" in cond and "IpAddress" in cond


def test_source_ips_inject_ipaddress_on_conditioned_statements():
    doc = _doc(["read", "download", "write"], source_ips=["203.0.113.7"])
    for s in doc["Statement"]:
        if "Condition" in s:
            assert s["Condition"]["IpAddress"]["acs:SourceIp"] == ["203.0.113.7"]
    # 桶信息条无 Condition，不该被塞 IP
    assert "Condition" not in _bucket_info_stmt(doc)


def test_no_source_ips_no_ipaddress():
    doc = _doc(["download"])
    for s in doc["Statement"]:
        if "Condition" in s:
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


def test_session_policy_same_shape_as_window():
    """build_session_policy 结构与 build_policy_with_window 一致（只多 2048 校验）。"""
    kw = dict(prefix="p/", caps=["read", "download"], not_before=NB, expire=EXP)
    assert policy.build_session_policy("b", **kw) == policy.build_policy_with_window("b", **kw)


def test_policy_too_large_error_is_valueerror():
    assert issubclass(policy.PolicyTooLargeError, ValueError)
