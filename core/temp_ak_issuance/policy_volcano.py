"""火山 TOS 临时凭证 policy 生成（对标 policy.py，火山 4 点差异）。

与阿里 OSS 版差异（research §3/§7 真机坐实）：
1. 时间键 `volc:CurrentTime`（阿里 `acs:CurrentTime`），**UTC `Z` 格式**（非 +08:00）。
2. Resource TRN `trn:tos:::bucket[/prefix*]`（阿里 `acs:oss:*:*:bucket/prefix*`），**无 region/account 段**。
3. 动作 `tos:*`；List 前缀条件键 `tos:prefix`（阿里 `oss:Prefix`）。
4. policy 文档**不加 `Version` 字段**（阿里加 `"Version":"1"`）。
权限模型 read/download/write 三者正交、无任何 delete（同阿里）。纯逻辑、不调云。
"""
from __future__ import annotations

from datetime import datetime, timezone

POLICY_PREFIX = "tempak-tos-auto-"      # 火山独立前缀，区别阿里 temp-ak-auto-

# 三类能力各自动作集（互不重叠；download≠read；write 无 DeleteObject）。以 research §3 为准，按需增补。
LIST_ACTIONS = ["tos:ListBucket", "tos:HeadBucket", "tos:GetBucketLocation"]       # read：列清单不下载
DOWNLOAD_ACTIONS = ["tos:GetObject"]                                              # download：下载
WRITE_ACTIONS = ["tos:PutObject", "tos:AbortMultipartUpload",
                 "tos:ListMultipartUploadParts"]                                  # write：上传/分片，无 delete


def iso8601_utc(epoch: float) -> str:
    """epoch → '2026-07-21T11:35:53Z'（UTC Z，火山官方例格式）。"""
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _time_conditions(not_before: float, expire: float, source_ips=None) -> dict:
    cond = {
        "DateGreaterThan": {"volc:CurrentTime": iso8601_utc(not_before)},
        "DateLessThan":    {"volc:CurrentTime": iso8601_utc(expire)},
    }
    if source_ips:
        cond["IpAddress"] = {"volc:SourceIp": list(source_ips)}   # P1 取证后启用（research §9 #4）
    return cond


def _obj_trn(bucket: str, prefix: str) -> str:
    return f"trn:tos:::{bucket}/{prefix}*" if prefix else f"trn:tos:::{bucket}/*"


def build_policy_with_window(bucket: str, *, prefix: str = "", caps,
                            not_before: float, expire: float, source_ips=None) -> dict:
    """单桶单目录 + caps⊆{read,download,write} + 时间窗 → TOS policy dict（无 Version）。

    read → ListBucket 桶级语句(trn:tos:::bucket)+非整桶带 tos:prefix；download → GetObject 对象语句；
    write → 上传对象语句。caps 空 → 空 Statement。
    """
    caps = set(caps or [])
    obj = _obj_trn(bucket, prefix)
    stmts: list[dict] = []

    if "download" in caps:
        stmts.append({"Effect": "Allow", "Action": DOWNLOAD_ACTIONS, "Resource": [obj],
                      "Condition": _time_conditions(not_before, expire, source_ips)})
    if "write" in caps:
        stmts.append({"Effect": "Allow", "Action": WRITE_ACTIONS, "Resource": [obj],
                      "Condition": _time_conditions(not_before, expire, source_ips)})
    if "read" in caps:
        cond = _time_conditions(not_before, expire, source_ips)
        if prefix:
            cond["StringLike"] = {"tos:prefix": [prefix, prefix + "*"]}
        stmts.append({"Effect": "Allow", "Action": LIST_ACTIONS,
                      "Resource": [f"trn:tos:::{bucket}"], "Condition": cond})

    return {"Statement": stmts}
