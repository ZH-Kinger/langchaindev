"""火山 TOS 临时凭证到期/手动硬删（对标 cleanup.revoke_grant RAM 分支，火山 SDK 差异）。

删序（research §7 坐实）：list AK → 每个 update_access_key(Inactive) → delete_access_key(单删真机返 400、吞掉)
→ detach_user_policy → delete_policy → delete_user（级联删净残留 AK，复扫 0 残留）。全步幂等。
纵深防御：policy 时间窗到期已服务端拒一切调用，硬删只清 artifact，某步失败不影响安全。
火山 SDK 差异：DeleteAccessKeyRequest(access_key_id=)（阿里 user_access_key_id=）；DeletePolicyRequest 只收
policy_name（无 policy_type/无 version）；list_access_keys → resp.access_key_metadata[].access_key_id。
"""
from __future__ import annotations

import time

from utils.logger import get_logger

from . import orchestrator as o

logger = get_logger(__name__)


def _list_ak_ids(client, m, user_name: str) -> list[str]:
    try:
        resp = client.list_access_keys(m.ListAccessKeysRequest(user_name=user_name))
        keys = getattr(resp, "access_key_metadata", None) or []
        return [getattr(k, "access_key_id", "") for k in keys if getattr(k, "access_key_id", "")]
    except Exception:
        return []


def revoke_grant(grant: dict, *, log=None) -> bool:
    """火山方案 B 到期/手动硬删：软失效 AK → 删 AK(吞400) → 解绑 policy → 删 policy → 删 user(级联兜底)。幂等。"""
    from core import ram_approval
    log = log or logger.info
    if grant.get("stage") == o.STAGE_REVOKED:
        return True
    client = ram_approval.make_volcano_iam_client()
    m = ram_approval._volcano_iam_models()
    user = grant.get("user_name", "")
    pol = grant.get("policy_name", "")
    try:
        if user:
            for akid in _list_ak_ids(client, m, user):
                try:
                    client.update_access_key(m.UpdateAccessKeyRequest(
                        access_key_id=akid, user_name=user, status="Inactive"))
                except Exception:
                    pass
                try:
                    client.delete_access_key(m.DeleteAccessKeyRequest(access_key_id=akid, user_name=user))
                except Exception as e:
                    log(f"    ✗ 删 AK {akid} 失败(靠 delete_user 级联兜底): {e}")
            if pol:
                try:
                    client.detach_user_policy(m.DetachUserPolicyRequest(
                        policy_name=pol, policy_type="Custom", user_name=user))
                except Exception:
                    pass
        if pol:
            try:
                client.delete_policy(m.DeletePolicyRequest(policy_name=pol))
            except Exception as e:
                if not ram_approval._volcano_is_not_exist(e):
                    log(f"    ✗ 删 policy {pol} 失败: {e}")
        if user:
            try:
                client.delete_user(m.DeleteUserRequest(user_name=user))
            except Exception as e:
                if not ram_approval._volcano_is_not_exist(e):
                    log(f"    ✗ 删 user {user} 失败: {e}")
        grant["stage"] = o.STAGE_REVOKED
        grant["revoked_ts"] = time.time()
        o._save(grant)
        return True
    except Exception:
        logger.error("[temp_ak] 火山 revoke 失败 grant=%s", grant.get("grant_id"), exc_info=True)
        return False
