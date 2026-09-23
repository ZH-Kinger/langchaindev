"""临时 AK 发放 policy.py —— 严格对齐用户手动固化的权威模板（temp-ak-auto-tempak-nuoyiteng-7df6a7）。

线上 bug 根因（本文件锁死不重现）：原实现把桶信息动作（GetBucketInfo/Stat/Acl）与 ListObjects
塞进同一条 read 语句、且整条带了 oss:Prefix 的 StringLike 条件。GetBucket* 是桶级操作、请求不带
prefix 参数，被 oss:Prefix 条件卡死 → 拒绝 → 用户拿到凭证却访问不了桶。

修复后模型（build_policy_with_window，caps ⊆ {read,download,write}，三者正交）：
  · 桶信息条  → Action=[GetBucketInfo,GetBucketStat,GetBucketAcl,GetBucketLocation]，Resource=[桶]，**无 Condition**
               （无 oss:Prefix、无时间窗）。**caps 非空即给**（2026-08-19 起；原为
               「read/download 任一勾选」，漏了 write-only，见下方 test_write_only_* 的说明）。
  · List 条   → Action=[ListObjects,GetBucketMultipartUploads]，Resource=[桶]，
               Condition=时间窗 +（prefix 非空时）oss:Prefix StringLike[prefix, prefix+"*"]。read 勾选给。
  · 下载条    → Action=[GetObject,GetObjectVersion]，Resource=[桶/前缀*]，Condition=时间窗��download 勾选给。
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
# GetBucketLocation：S3 兼容客户端建连先探地域（2026-09-23 加）
_BUCKET_INFO = {
    "oss:GetBucketInfo", "oss:GetBucketStat", "oss:GetBucketAcl", "oss:GetBucketLocation",
}


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


def _download_family(actions):
    """所有 `oss:GetObject` 前缀族动作（GetObject / GetObjectVersion / GetObjectAcl / ...）。

    正交性断言必须按**前缀族**查，不能列精确串：DOWNLOAD_ACTIONS 每加一个动作，
    精确串断言就多一个漏网之鱼（2026-09-23 加 GetObjectVersion 时就正好漏了）。
    """
    return {a for a in actions if a.startswith("oss:GetObject")}


def _write_family(actions):
    """写入族（Put* / Append* / Abort* / 任何 Delete* / ListParts）。

    注意 `oss:GetBucketMultipartUploads` **不在**此列：它是桶级「列出进行中的分片上传」，
    属 read 的 LIST_ACTIONS，按 `MultipartUpload` 子串一刀切会误伤。
    """
    return {a for a in actions
            if a.startswith(("oss:Put", "oss:Delete", "oss:Append", "oss:Abort"))
            or a == "oss:ListParts"}


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
    assert "StringLike" not in info["Condition"]        # 有时间窗，但绝无 oss:Prefix

    # List 条：ListObjects + GetBucketMultipartUploads、Resource=桶、带时间窗 + oss:Prefix
    assert set(lst["Action"]) == {"oss:ListObjects", "oss:GetBucketMultipartUploads"}
    assert lst["Resource"] == ["acs:oss:*:*:b"]
    prefs = lst["Condition"]["StringLike"]["oss:Prefix"]
    assert prefs == ["team/data/", "team/data/*"]
    assert lst["Condition"]["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
    assert lst["Condition"]["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)


def test_read_only_list_stmt_has_no_getobject_and_no_bucketinfo_actions():
    """List 条不含**任何** oss:GetObject* 动作（read 不能下载），也不含桶信息四件套。"""
    lst = _list_stmt(_doc(["read"]))
    assert not _download_family(lst["Action"])
    assert not (_BUCKET_INFO & set(lst["Action"]))


def test_read_only_no_getobject_anywhere():
    """read 集里**没有任何** `oss:GetObject` 前缀族动作。

    【2026-09-23 改成前缀族】原来写的是精确串 `"oss:GetObject" not in acts`，
    `DOWNLOAD_ACTIONS` 新增 `oss:GetObjectVersion` 后这条**拦不住**它被误塞进 read 分支 ——
    正交性（read/download/write 三者不串味）是这次改动唯一可能被破坏的不变量，
    而当时没有任何测试能发现它。
    """
    acts = _all_actions(_doc(["read"]))
    assert not _download_family(acts), f"read 里混进了下载动作：{sorted(_download_family(acts))}"
    assert not _write_family(acts), f"read 里混进了写动作：{sorted(_write_family(acts))}"


# ── 单勾 download：桶信息条 + GetObject条(Resource=桶/前缀*) ─────────────────────

def test_download_only_yields_bucketinfo_plus_getobject():
    doc = _doc(["download"], prefix="team/data/")
    assert len(doc["Statement"]) == 2

    info = _bucket_info_stmt(doc)
    dl = _stmt_by_action(doc, "oss:GetObject")
    assert info is not None and dl is not None

    assert set(info["Action"]) == _BUCKET_INFO
    assert "StringLike" not in info["Condition"]

    # 下载条：Resource 是 桶/前缀*（对象级），不是桶
    assert dl["Action"] == ["oss:GetObject", "oss:GetObjectVersion"]
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


# ── 单勾 write：写条 + 桶信息条，无 delete/get/list ────────────────────────────
#
# 【行为变更 2026-08-19】原先断言的是「write 不触发桶信息条」。那个边界**从未被验证过**：
# 它是 aa879f4「对齐权威模板」时推出来的，而那份模板（tempak-nuoyiteng-7df6a7）是
# read+write+download 全勾的，**根本不含 write-only 这个场景**。
#
# 线上「元客」是第一单 write-only，于是撞上了：策略里只有一条 PutObject，连 GetBucketInfo
# 都没有 → ossutil / SDK / 控制台在上传前普遍先探一次桶 → 403 → 现场表现成
# 「凭证发了但什么都干不了、策略里看不到任何路径」，被误判成「权限策略没建好」。
#
# 现在改成「勾了任何一项就给桶信息」。下面两条断言随之更新，但**正交性仍然守住**：
# 桶信息只含三个只读元数据动作，不含 ListObjects —— 只勾上传的外部方看不到桶里有什么。

def test_write_only_gets_bucket_info_and_write_statements():
    doc = _doc(["write"], prefix="drop/")
    assert len(doc["Statement"]) == 2           # 桶信息条 + 写条
    info = _bucket_info_stmt(doc)
    assert info is not None, "只勾上传的凭证拿不到桶信息 = 客户端探桶即 403（线上元客那单）"
    assert "StringLike" not in info["Condition"]   # 有时间窗，但桶级操作绝不叠 oss:Prefix
    assert info["Resource"] == ["acs:oss:*:*:b"]
    st = [x for x in doc["Statement"] if x is not info][0]
    assert set(st["Action"]) == {"oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"}
    assert st["Resource"] == ["acs:oss:*:*:b/drop/*"]


def test_write_only_no_delete_no_get_no_list():
    """给桶元数据 ≠ 给列举/下载/删除。正交性不能被上面那个修复带偏。

    【2026-09-23 改成前缀族/整篇兜底】下载侧原来只查精确串 `oss:GetObject`，
    新增的 `oss:GetObjectVersion` 从这条底下能直接溜进 write 集。
    """
    doc = _doc(["write"])
    acts = _all_actions(doc)
    assert "delete" not in json.dumps(doc).lower(), "write 集出现删除动作"
    assert not _download_family(acts), f"write 里混进了下载动作：{sorted(_download_family(acts))}"
    assert "oss:ListObjects" not in acts        # ← 关键：看不到桶里有什么


# ── 全勾 read+download+write：四条，顺序=桶信息/List/下载/写 ─────────────────────

def test_all_three_four_statements_in_order():
    doc = _doc(["read", "download", "write"], prefix="team/data/")
    stmts = doc["Statement"]
    assert len(stmts) == 4

    # 顺序：桶信息 → List → 下载 → 写
    assert set(stmts[0]["Action"]) == _BUCKET_INFO
    assert "StringLike" not in stmts[0]["Condition"]
    assert set(stmts[1]["Action"]) == {"oss:ListObjects", "oss:GetBucketMultipartUploads"}
    assert stmts[2]["Action"] == ["oss:GetObject", "oss:GetObjectVersion"]
    assert set(stmts[3]["Action"]) == {"oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"}


@pytest.mark.parametrize("caps", [["read"], ["download"], ["write"],
                                  ["read", "download"], ["read", "write"],
                                  ["download", "write"], ["read", "download", "write"]])
def test_no_delete_action_anywhere_in_document(caps):
    """整篇兜底：策略文档里**任何位置**都不许出现 "delete"（大小写不敏感）。

    【2026-09-23 从 `"oss:DeleteObject" not in acts` 换成整篇扫】
    原写法只挡住一个名字。版本控制桶的删除叫 `oss:DeleteObjectVersion`，批删叫
    `oss:DeleteMultipleObjects`，还有 `oss:DeleteBucket*` —— 全都查不到。
    发出去的是给外部方的凭证，误加任何一个删除动作的后果是别人的数据被删掉。
    现有四个动作集里没有任何合法动作含 "delete"，桶名/前缀也由本用例自己控制，不会误伤。
    """
    doc = _doc(caps, prefix="team/data/")
    flat = json.dumps(doc).lower()
    assert "delete" not in flat, f"策略里出现删除动作：{json.dumps(doc, ensure_ascii=False)}"


# ── 关键回归（防 bug 重现）───────────────────────────────────────────────────

@pytest.mark.parametrize("caps", [["read"], ["download"], ["read", "download"],
                                  ["read", "download", "write"]])
def test_bucketinfo_statement_never_carries_prefix_condition(caps):
    """桶信息条【绝不】带 oss:Prefix —— 直接锁死线上 bug 根因。

    【2026-08-21】这条原本还断言「没有任何 Condition」，那是把两件事混了：
    真正的不变量是 **不能有 oss:Prefix**（桶级请求不带 prefix 参数，叠上去必被拒），
    而时间窗是 Date 条件、与 prefix 无关。现已按设计叠上时间窗
    （原先没有 → 凭证到期后仍可调，见 test_bucketinfo_carries_time_window）。
    """
    info = _bucket_info_stmt(_doc(caps, prefix="team/data/"))
    assert info is not None
    assert "oss:Prefix" not in json.dumps(info)
    assert "StringLike" not in (info.get("Condition") or {})


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
        assert not (_BUCKET_INFO & set(s["Action"]))     # 桶信息四件套绝不在此
        # 前缀族：GetBucketLocation / GetObjectVersion 这类新动作也不许溜进带 Prefix 的语句
        assert not _download_family(s["Action"])
        assert not [a for a in s["Action"] if a.startswith("oss:GetBucket")
                    and a != "oss:GetBucketMultipartUploads"], (
            f"带 oss:Prefix 条件的语句里出现桶级动作 {s['Action']} —— 桶级请求不带 prefix，必被拒")


def test_module_action_sets_no_cross_contamination():
    assert "oss:DeleteObject" not in policy.WRITE_ACTIONS
    assert "oss:GetObject" not in policy.LIST_ACTIONS       # read 集不含下载
    assert "oss:ListObjects" not in policy.DOWNLOAD_ACTIONS
    # 桶信息四件套独立成集，不混入 List（否则会被带上 oss:Prefix 而被拒）
    for a in _BUCKET_INFO:
        assert a in policy.BUCKET_INFO_ACTIONS
        assert a not in policy.LIST_ACTIONS

    # ── 2026-09-23 两个新动作各自的串味防线 ──────────────────────────────────
    # GetObjectVersion 必须**只**待在 download 集：进了 read 就等于「只勾列举」的外部方
    # 能带 version id 直接取对象内容；进了 write 就等于上传方能读回全桶历史版本。
    assert "oss:GetObjectVersion" in policy.DOWNLOAD_ACTIONS
    assert "oss:GetObjectVersion" not in policy.LIST_ACTIONS
    assert "oss:GetObjectVersion" not in policy.WRITE_ACTIONS
    assert "oss:GetObjectVersion" not in policy.BUCKET_INFO_ACTIONS
    # GetBucketLocation 必须留在桶信息条里。混进 LIST_ACTIONS 就会被带上 oss:Prefix ——
    # 桶级请求不带 prefix 参数 → 服务端判假拒绝 → 「拿了凭证访问不了桶」那个老坑原地重演
    # （docs/collab/research/oss-least-privilege-proposal.md:7,86，线上真踩过）。
    assert "oss:GetBucketLocation" in policy.BUCKET_INFO_ACTIONS
    assert "oss:GetBucketLocation" not in policy.LIST_ACTIONS
    assert "oss:GetBucketLocation" not in policy.DOWNLOAD_ACTIONS
    assert "oss:GetBucketLocation" not in policy.WRITE_ACTIONS


def test_no_action_set_contains_delete():
    """四个模块级动作集**逐个**确认无删除动作（比文档扫描更早一层的防线）。"""
    for name in ("BUCKET_INFO_ACTIONS", "LIST_ACTIONS", "DOWNLOAD_ACTIONS", "WRITE_ACTIONS"):
        acts = getattr(policy, name)
        assert not [a for a in acts if "delete" in a.lower()], f"{name} 含删除动作：{acts}"


def test_list_objects_versions_never_granted():
    """`oss:ListObjectVersions` 是真正的扩面（能枚举历史版本 + 被 delete marker 删掉的对象），
    policy.py 的注释明令「别再补」—— 这里把它钉住，防止下次「顺手加一个」。"""
    doc = _doc(["read", "download", "write"], prefix="team/data/")
    assert "oss:ListObjectVersions" not in json.dumps(doc)
    for name in ("BUCKET_INFO_ACTIONS", "LIST_ACTIONS", "DOWNLOAD_ACTIONS", "WRITE_ACTIONS"):
        assert "oss:ListObjectVersions" not in getattr(policy, name)


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
    # 整桶时桶信息条仍存在；它有时间窗但**绝无 oss:Prefix**
    info = _bucket_info_stmt(_doc(["read"], prefix=""))
    assert "StringLike" not in (info.get("Condition") or {})
    assert "DateGreaterThan" in info["Condition"]


# ── 时间窗 / IP：作用于 List/下载/写，不作用于桶信息条 ────────────────────────

def test_time_window_on_every_statement():
    """【2026-08-21 起】**每条语句**都带时间窗，桶信息条也不例外。

    原先桶信息条无 Condition，于是凭证到期后外部方仍能调 GetBucketInfo/Stat/Acl，
    直到清理任务当天 HOUR:35 硬删用户（最坏 ~24h，清理失败更久）——
    与「泄漏也随到期自动失效」的设计宣称矛盾。
    """
    doc = _doc(["read", "download", "write"])
    conditioned = [s for s in doc["Statement"] if "Condition" in s]
    assert len(conditioned) == 4                        # 桶信息 + List + 下载 + 写
    for s in conditioned:
        assert s["Condition"]["DateGreaterThan"]["acs:CurrentTime"] == policy.iso8601_bj(NB)
        assert s["Condition"]["DateLessThan"]["acs:CurrentTime"] == policy.iso8601_bj(EXP)


def test_read_list_condition_is_and_of_date_prefix_ip():
    """read 非整桶 List 语句：Date* + StringLike + IpAddress 同 Condition = AND。"""
    lst = _list_stmt(_doc(["read"], prefix="p/", source_ips=["1.2.3.4"]))
    cond = lst["Condition"]
    assert "DateGreaterThan" in cond and "DateLessThan" in cond
    assert "StringLike" in cond and "IpAddress" in cond


def test_source_ips_inject_ipaddress_on_every_statement():
    """【2026-08-21 起】IP 限制覆盖**所有**语句，含桶信息条。

    这是变**严**不是变松：锁了出口 IP 却让 GetBucketInfo/Stat 能从任意 IP 调，
    等于给了个绕过口子。IpAddress 限的是请求来源、与「桶级请求不带 prefix」无关，
    不会重蹈 aa879f4 那个坑。
    """
    doc = _doc(["read", "download", "write"], source_ips=["203.0.113.7"])
    for s in doc["Statement"]:
        assert s["Condition"]["IpAddress"]["acs:SourceIp"] == ["203.0.113.7"]
    assert _bucket_info_stmt(doc)["Condition"]["IpAddress"]["acs:SourceIp"] == ["203.0.113.7"]


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


# OSS 桶名上限 63 字符；120 字符前缀 ≈ 线上「部门/项目/数据集/批次/」这类四段目录的现实长度。
_MAX_BUCKET = "b" * 63
_LONG_PREFIX = "seg-dir/" * 15          # 120 字符


def test_session_policy_realistic_worst_case_within_limit():
    """2048 余量回归：**现实最坏情况**（63 字符桶名 + 120 字符前缀 + 三 caps 全勾）仍要放得下。

    原来只有 `test_session_policy_within_limit_ok`（桶名 "b" + 前缀 "team/data/"），离上限还有
    七百多字符 —— 再加多少动作都不会红，等于没在看预算。2026-09-23 新增两个动作精确 +49 字符
    （典型场景 1241→1290）。这条盯的是「下次再加动作、真吃光预算」时立刻红，
    而不是等某个长前缀申请单在发放那一刻炸。

    实测（2026-09-23）：此形状 1902 字符，余量 146。
    审批路径恒 `source_ips=[]`（approval.py:308），所以这就是审批发放的最坏形状。
    """
    doc = policy.build_session_policy(
        _MAX_BUCKET, prefix=_LONG_PREFIX, caps=["read", "download", "write"],
        not_before=NB, expire=EXP)
    size = len(json.dumps(doc, ensure_ascii=False))
    assert size <= policy.SESSION_POLICY_MAX, (
        f"现实最坏情况的 session policy {size} 字符已超 {policy.SESSION_POLICY_MAX} —— "
        "长前缀申请单会在发放时抛 PolicyTooLargeError")


def test_session_policy_budget_headroom_is_not_yet_exhausted():
    """把余量本身写成数字断言：低于 100 字符就该在加动作时停下来想想，而不是等它爆。"""
    doc = policy.build_policy_with_window(
        _MAX_BUCKET, prefix=_LONG_PREFIX, caps=["read", "download", "write"],
        not_before=NB, expire=EXP)
    headroom = policy.SESSION_POLICY_MAX - len(json.dumps(doc, ensure_ascii=False))
    assert headroom >= 100, (
        f"2048 预算只剩 {headroom} 字符（每个新动作约 +25）。再加动作前先确认 STS 路径"
        "（TEMP_AK_STS_MAX_SECONDS>0 时才走）能不能接受长前缀申请单直接发放失败")


def test_session_policy_worst_case_with_source_ips_overflows_today():
    """事实锁（非回归）：最坏形状**再加出口 IP 限制**就超 2048，走 STS 会直接抛。

    实测 63 字符桶名 + 120 字符前缀 + 三 caps：无 IP 1902 / 一个 IP 2094 / 两个 IP 2158。
    影响面很窄且 fail-loud：① 审批路径 `source_ips` 恒 `[]`（approval.py:308），只有 CLI
    `--source-ip` 能设；② 线上 `TEMP_AK_STS_MAX_SECONDS=0`，根本不走 session policy；
    ③ 真撞上时抛 `PolicyTooLargeError` 并提示「缩短目录或改方案 B」，不会静默发出弱凭证。
    若将来压缩了策略体积（如合并语句），本用例会红 —— 那时把它改成 within-limit 断言即可。
    """
    with pytest.raises(policy.PolicyTooLargeError):
        policy.build_session_policy(
            _MAX_BUCKET, prefix=_LONG_PREFIX, caps=["read", "download", "write"],
            not_before=NB, expire=EXP, source_ips=["203.0.113.7", "198.51.100.9"])


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
