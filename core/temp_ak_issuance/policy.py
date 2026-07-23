"""临时 AK 发放的 OSS 权限 policy 生成（时间区间条件变体）。

权限模型（严格对齐用户手动固化的权威模板，与审批「权限设置」勾选项一一对应，三者正交）：
  · 桶信息   → GetBucketInfo/GetBucketStat/GetBucketAcl，**独立成条、Resource=桶、无 Prefix、无时间窗**
              （read/download 任一勾选即给；桶级操作不带 prefix，绝不叠 oss:Prefix 条件否则被拒→用户访问不了）。
  · read     → List（ListObjects + GetBucketMultipartUploads），Resource=桶 + oss:Prefix 条件，能看清单、不能下载。
  · download → 只给 **GetObject**，Resource=桶/前缀*，能下载对象内容。
  · write    → 只给 **PutObject/AbortMultipartUpload/ListParts**（上传/分片），**绝不含任何删除动作**。
List/下载/写语句叠加生效/到期时间窗（DateGreaterThan/DateLessThan，acs:CurrentTime ISO8601 +08:00，AND）——
服务端每次调用按当前时间判定，泄漏也随到期自动失效。可选 acs:SourceIp 锁外部出口 IP。
纯逻辑、不调云、不改 permsync 源。build_policy_with_window 供方案 B；build_session_policy 供 STS（≤2048）。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from core.oss_perm import permsync

POLICY_PREFIX = "temp-ak-auto-"      # 方案 B 自定义策略命名前缀（区别于 permsync 的 wuji-oss-auto-）
SESSION_POLICY_MAX = 2048            # AssumeRole Policy 参数字符上限

# 动作集严格对齐用户手动固化的权威模板（temp-ak-auto-tempak-nuoyiteng-7df6a7），三者正交、无删除动作。
# 桶信息语句独立成条、Resource=桶、无 Prefix 条件（桶级操作不带 prefix 参数，混进 List+Prefix 条件会被拒→用户访问不了）。
BUCKET_INFO_ACTIONS = ["oss:GetBucketInfo", "oss:GetBucketStat", "oss:GetBucketAcl"]
LIST_ACTIONS = ["oss:ListObjects", "oss:GetBucketMultipartUploads"]              # read：列清单，带 oss:Prefix
DOWNLOAD_ACTIONS = ["oss:GetObject"]                                            # download：仅选了才下发
WRITE_ACTIONS = ["oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"]  # 无 DeleteObject

_BJ = timezone(timedelta(hours=8))


def iso8601_bj(epoch: float) -> str:
    """epoch 秒 → '2026-08-01T00:00:00+08:00'（北京时区，RAM Condition Date* 要求带时区）。"""
    return datetime.fromtimestamp(int(epoch), tz=_BJ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def _time_conditions(not_before: float, expire: float, source_ips=None) -> dict:
    """时间窗（+ 可选 IP）Condition 块。多运算符同一 statement 即 AND。"""
    cond = {
        "DateGreaterThan": {"acs:CurrentTime": iso8601_bj(not_before)},
        "DateLessThan":    {"acs:CurrentTime": iso8601_bj(expire)},
    }
    if source_ips:
        cond["IpAddress"] = {"acs:SourceIp": list(source_ips)}
    return cond


def build_policy_with_window(
    bucket: str,
    *,
    prefix: str = "",
    caps,
    not_before: float,
    expire: float,
    source_ips=None,
) -> dict:
    """单桶单目录(prefix) + 能力集 caps ⊆ {read,download,write} + 时间窗 → RAM policy 文档(dict)。

    read → List 语句（桶级，非整桶时带 oss:Prefix）；download → GetObject 对象语句；write → 上传对象语句。
    caps 为空则产出空策略（无 statement，等于什么都不授）。'' prefix 表示整桶。
    """
    caps = set(caps or [])
    base = f"acs:oss:*:*:{bucket}"
    obj_arn = permsync._obj_arn(base, prefix)
    stmts: list[dict] = []

    # read 或 download 任一勾选，都给桶信息（能定位桶、看基本信息）。独立成条：Resource=桶、无 Prefix、无时间窗，
    # 与用户权威模板一致（桶级操作不带 prefix 参数，绝不能叠 oss:Prefix 条件，否则被拒）。
    if "read" in caps or "download" in caps:
        stmts.append({
            "Effect": "Allow",
            "Action": BUCKET_INFO_ACTIONS,
            "Resource": [base],
        })
    if "read" in caps:
        cond = _time_conditions(not_before, expire, source_ips)
        if prefix:
            cond["StringLike"] = {"oss:Prefix": [prefix, prefix + "*"]}
        stmts.append({
            "Effect": "Allow",
            "Action": LIST_ACTIONS,
            "Resource": [base],
            "Condition": cond,
        })
    if "download" in caps:
        stmts.append({
            "Effect": "Allow",
            "Action": DOWNLOAD_ACTIONS,
            "Resource": [obj_arn],
            "Condition": _time_conditions(not_before, expire, source_ips),
        })
    if "write" in caps:
        stmts.append({
            "Effect": "Allow",
            "Action": WRITE_ACTIONS,
            "Resource": [obj_arn],
            "Condition": _time_conditions(not_before, expire, source_ips),
        })

    return {"Version": "1", "Statement": stmts}


def build_session_policy(
    bucket: str,
    *,
    prefix: str = "",
    caps,
    not_before: float,
    expire: float,
    source_ips=None,
) -> dict:
    """STS AssumeRole 的 session policy：结构同 build_policy_with_window，但校验 ≤2048 字符。

    会话最终权限 = 宽角色 policy ∩ 本 session policy。超上限抛 PolicyTooLargeError。
    """
    doc = build_policy_with_window(
        bucket, prefix=prefix, caps=caps,
        not_before=not_before, expire=expire, source_ips=source_ips,
    )
    size = len(json.dumps(doc, ensure_ascii=False))
    if size > SESSION_POLICY_MAX:
        raise PolicyTooLargeError(
            f"session policy {size} 字符超过 STS 上限 {SESSION_POLICY_MAX}，"
            f"请缩短目录/前缀或改用方案 B（长期 AK）")
    return doc


class PolicyTooLargeError(ValueError):
    """session policy 超过 STS Policy 参数 2048 字符上限。"""
