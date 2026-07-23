"""火山 TOS 临时凭证发放引擎（方案 B：长期 IAM AK + 时间条件 policy）。

对标 issuer._issue_ram / rewrite_ram_window，火山 SDK 差异（planner 核对 volcenginesdkiam 签名 +
research §4/§7 真机坐实）：
- 复用 ram_approval.make_volcano_iam_client()（凭证 VOLCANO_ACCESS_KEY→回退 TOS_ACCESS_KEY，IAM region cn-beijing）。
- 建 AK 响应取 `resp.access_key.secret_access_key`（阿里是 resp.body.access_key.access_key_secret）。
- policy 无 version 概念：延期用 update_policy 改整篇；不存在则 create_policy。
- 建 user 不开 console/不入组/附自定义 policy（对标 create_volcano_iam_account 但反过来）。
凭证（含 secret）只当次返回给下发层，绝不落 Redis/日志。
"""
from __future__ import annotations

import json

from utils.logger import get_logger

from . import policy_volcano

logger = get_logger(__name__)


def _client_models():
    from core import ram_approval
    return ram_approval.make_volcano_iam_client(), ram_approval._volcano_iam_models()


def _doc(grant: dict) -> dict:
    return policy_volcano.build_policy_with_window(
        grant["bucket"], prefix=grant.get("prefix", ""), caps=grant.get("caps") or [],
        not_before=grant["not_before"], expire=grant["expire"], source_ips=grant.get("source_ips") or None)


def plan(grant: dict) -> dict:
    """dry-run：出火山 TOS policy 预览，不调云。"""
    return {"mode": "ram", "user_name": grant["user_name"], "policy_name": grant["policy_name"],
            "policy": _doc(grant), "has_token": False}


def issue(grant: dict) -> dict:
    """真发放（火山方案 B）。返回 {access_key_id, access_key_secret, security_token:"", expire_ts, mode:"ram"}。"""
    from core import ram_approval
    from . import orchestrator as o
    client, m = _client_models()
    user = grant["user_name"]
    pol = grant["policy_name"]

    # 1) 建 user（幂等：get_user 404→create；不开 console、不入组）
    try:
        client.get_user(m.GetUserRequest(user_name=user))
    except Exception as e:
        if ram_approval._volcano_is_not_exist(e):
            client.create_user(m.CreateUserRequest(
                user_name=user, display_name=o.display_name_for(grant),
                description=(grant.get("reason") or "temp-ak tos external issuance")[:128] or None))
        else:
            raise

    # 2) 建/更新时间窗 policy（无 version：不存在建、存在改整篇）
    doc_str = json.dumps(_doc(grant), ensure_ascii=False)
    try:
        client.create_policy(m.CreatePolicyRequest(
            policy_name=pol, policy_document=doc_str,
            description="temp-ak tos external issuance (time-boxed)"))
    except Exception as e:
        if ram_approval._volcano_is_already_exists(e):
            client.update_policy(m.UpdatePolicyRequest(policy_name=pol, new_policy_document=doc_str))
        else:
            raise

    # 3) 附加 policy（幂等）
    try:
        client.attach_user_policy(m.AttachUserPolicyRequest(
            policy_name=pol, policy_type="Custom", user_name=user))
    except Exception as e:
        if not ram_approval._volcano_is_already_exists(e):
            raise

    # 4) 建 AK（外部拿这组长期 AK；到期由 policy 时间窗拒 + cleanup 硬删）
    resp = client.create_access_key(m.CreateAccessKeyRequest(user_name=user))
    ak = getattr(resp, "access_key", None)   # 火山：resp.access_key（非 resp.body）
    return {
        "access_key_id":     getattr(ak, "access_key_id", "") or "",
        "access_key_secret": getattr(ak, "secret_access_key", "") or "",   # 火山：secret_access_key
        "security_token":    "",
        "expire_ts":         float(grant["expire"]),
        "mode":              "ram",
    }


def rewrite_window(grant: dict) -> None:
    """延期：update_policy 改整篇时间窗。AK/user 不变、不重发凭证（~30-40s 传播，非即时）。"""
    client, m = _client_models()
    if not grant.get("policy_name"):
        raise RuntimeError("火山方案 B 延期缺 policy_name")
    client.update_policy(m.UpdatePolicyRequest(
        policy_name=grant["policy_name"], new_policy_document=json.dumps(_doc(grant), ensure_ascii=False)))
