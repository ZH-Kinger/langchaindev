"""临时 AK 发放的 OSS 权限 policy 生成（时间区间条件变体）。

复用 core.oss_perm.permsync 的对象级 ARN 与动作集，叠加：
  · 时间窗条件（DateGreaterThan=生效 / DateLessThan=到期，acs:CurrentTime ISO8601 +08:00，同一 statement AND）——
    服务端每次调用按当前时间判定，即使凭证泄漏，过期后一切调用自动被拒（外部/卖数据场景的关键控制）。
  · 桶级读补 GetBucketAcl（用户参考 policy 要求）。
  · 可选 acs:SourceIp 锁外部方出口 IP。
本模块纯逻辑、不调云、不改 permsync 源。build_policy_with_window 供方案 B（长期 AK）；
build_session_policy 供 STS（AssumeRole Policy 参数，≤2048 字符）。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from core.oss_perm import permsync

POLICY_PREFIX = "temp-ak-auto-"      # 方案 B 自定义策略命名前缀（区别于 permsync 的 wuji-oss-auto-）
SESSION_POLICY_MAX = 2048            # AssumeRole Policy 参数字符上限

# 桶级读+列举（含用户要的 GetBucketAcl）。对象级读沿用 permsync；对象级写为外部场景收紧——
# 只给上传/分片相关，绝不含 DeleteObject（防外部删数据）。
BUCKET_ACTIONS = list(permsync.LIST_BUCKET_ACTIONS) + ["oss:GetBucketAcl"]
READ_OBJECT_ACTIONS = list(permsync.READ_OBJECT_ACTIONS)                       # GetObject, GetObjectAcl
WRITE_OBJECT_ACTIONS = ["oss:PutObject", "oss:AbortMultipartUpload", "oss:ListParts"]

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
    read_prefixes=None,
    write_prefixes=None,
    not_before: float,
    expire: float,
    source_ips=None,
) -> dict:
    """单桶、读/写前缀 + 时间窗（+ 可选 IP）→ RAM policy 文档(dict)。

    read/write 传空集合即该向不授权。'' 前缀表示整桶。List 语句限定到读写涉及的前缀（oss:Prefix）。
    """
    reads = {p for p in (read_prefixes or []) }
    writes = {p for p in (write_prefixes or []) }
    base = f"acs:oss:*:*:{bucket}"
    stmts: list[dict] = []

    if reads:
        stmts.append({
            "Effect": "Allow",
            "Action": READ_OBJECT_ACTIONS,
            "Resource": [permsync._obj_arn(base, p) for p in sorted(reads)],
            "Condition": _time_conditions(not_before, expire, source_ips),
        })
    if writes:
        stmts.append({
            "Effect": "Allow",
            "Action": WRITE_OBJECT_ACTIONS,
            "Resource": [permsync._obj_arn(base, p) for p in sorted(writes)],
            "Condition": _time_conditions(not_before, expire, source_ips),
        })

    # 桶级 List/Info：限定到读写涉及的前缀（非整桶时加 oss:Prefix），叠加时间窗（+IP）。
    list_cond = _time_conditions(not_before, expire, source_ips)
    all_pref = reads | writes
    if all_pref and "" not in all_pref:
        conds = []
        for p in sorted(all_pref):
            conds.extend([p, p + "*"])
        list_cond["StringLike"] = {"oss:Prefix": conds}
    stmts.append({
        "Effect": "Allow",
        "Action": BUCKET_ACTIONS,
        "Resource": [base],
        "Condition": list_cond,
    })

    return {"Version": "1", "Statement": stmts}


def build_session_policy(
    bucket: str,
    *,
    read_prefixes=None,
    write_prefixes=None,
    not_before: float,
    expire: float,
    source_ips=None,
) -> dict:
    """STS AssumeRole 的 session policy：结构同 build_policy_with_window，但校验 ≤2048 字符。

    会话最终权限 = 宽角色 policy ∩ 本 session policy。超上限时抛 PolicyTooLargeError（提示改窄前缀/走方案 B）。
    """
    doc = build_policy_with_window(
        bucket, read_prefixes=read_prefixes, write_prefixes=write_prefixes,
        not_before=not_before, expire=expire, source_ips=source_ips,
    )
    size = len(json.dumps(doc, ensure_ascii=False))
    if size > SESSION_POLICY_MAX:
        raise PolicyTooLargeError(
            f"session policy {size} 字符超过 STS 上限 {SESSION_POLICY_MAX}，"
            f"请减少前缀数量或改用方案 B（长期 AK）")
    return doc


class PolicyTooLargeError(ValueError):
    """session policy 超过 STS Policy 参数 2048 字符上限。"""
