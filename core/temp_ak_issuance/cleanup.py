"""方案 B 到期硬删（STS 无需清理，自灭）。

删除顺序照 RAM 依赖前置（照 jichuan 删号经验）：停用 AK → 删 AK → 解绑 policy → 删 policy（含清版本）→ 删 user。
纵深防御：policy 的 DateLessThan 到期后已服务端拒一切调用，硬删只清残留 artifact——某次失败不影响安全。
所有步骤幂等（删已删的 try/except 吞）。
"""
from __future__ import annotations

import json
import time

from utils.logger import get_logger

from . import orchestrator as o

logger = get_logger(__name__)


def _list_ak_ids(client, m, user_name: str) -> list[str]:
    try:
        resp = client.list_access_keys(m.ListAccessKeysRequest(user_name=user_name))
        aks = getattr(getattr(resp.body, "access_keys", None), "access_key", None) or []
        return [a.access_key_id for a in aks]
    except Exception:
        return []


def _delete_policy_all_versions(client, m, name: str) -> None:
    from core.oss_perm.permsync import _err_code
    try:
        vs = client.list_policy_versions(m.ListPolicyVersionsRequest(
            policy_type="Custom", policy_name=name))
        for v in (getattr(getattr(vs.body, "policy_versions", None), "policy_version", None) or []):
            if not getattr(v, "is_default_version", False):
                try:
                    # DeletePolicyVersionRequest 签名只有 (policy_name, version_id)——不接受 policy_type
                    # （那是 ListPolicyVersionsRequest 才有的参数）。误传会构造即抛 TypeError→版本删不掉。
                    client.delete_policy_version(m.DeletePolicyVersionRequest(
                        policy_name=name, version_id=v.version_id))
                except Exception:
                    pass
        client.delete_policy(m.DeletePolicyRequest(policy_name=name))
    except Exception as e:
        if _err_code(e) not in ("EntityNotExist.Policy", "EntityNotExist.CustomPolicy"):
            logger.warning("[temp_ak] delete policy %s: %s", name, e)


def revoke_grant(grant: dict, *, log=None) -> bool:
    """吊销一个 grant。STS 只翻状态（自灭）；方案 B 走硬删序列。返回是否成功翻到 REVOKED。"""
    log = log or logger.info
    if grant.get("stage") == o.STAGE_REVOKED:
        return True
    if grant.get("mode") != o.issuer.RAM_MODE:
        grant["stage"] = o.STAGE_REVOKED
        grant["revoked_ts"] = time.time()
        o._save(grant)
        return True

    from alibabacloud_ram20150501 import models as m
    from core.oss_perm.permsync import _err_code, make_ram_client
    client = make_ram_client()
    user = grant.get("user_name", "")
    pol = grant.get("policy_name", "")
    try:
        if user:
            for akid in _list_ak_ids(client, m, user):
                try:
                    client.update_access_key(m.UpdateAccessKeyRequest(
                        user_access_key_id=akid, user_name=user, status="Inactive"))
                except Exception:
                    pass
                try:
                    client.delete_access_key(m.DeleteAccessKeyRequest(
                        user_access_key_id=akid, user_name=user))
                except Exception as e:
                    log(f"    ✗ 删 AK {akid} 失败: {e}")
            if pol:
                try:
                    client.detach_policy_from_user(m.DetachPolicyFromUserRequest(
                        policy_type="Custom", policy_name=pol, user_name=user))
                except Exception:
                    pass
        if pol:
            _delete_policy_all_versions(client, m, pol)
        if user:
            try:
                client.delete_user(m.DeleteUserRequest(user_name=user))
            except Exception as e:
                if _err_code(e) != "EntityNotExist.User":
                    log(f"    ✗ 删 user {user} 失败: {e}")
        grant["stage"] = o.STAGE_REVOKED
        grant["revoked_ts"] = time.time()
        o._save(grant)
        return True
    except Exception:
        logger.error("[temp_ak] revoke failed grant=%s", grant.get("grant_id"), exc_info=True)
        return False


def sweep_expired(now: float | None = None) -> list[str]:
    """扫 Redis temp_ak:grant:*，对已 ISSUED 且 expire<now 的方案 B grant 硬删。返回被吊销的 grant_id 列表。"""
    from utils.redis_client import get_redis
    now = now if now is not None else time.time()
    revoked: list[str] = []
    try:
        r = get_redis()
        for key in r.scan_iter(o._KEY_PREFIX + "*"):
            raw = r.get(key)
            if not raw:
                continue
            try:
                grant = json.loads(raw)
            except Exception:
                continue
            if grant.get("stage") != o.STAGE_ISSUED:
                continue
            if float(grant.get("expire", 0)) >= now:
                continue
            if revoke_grant(grant):
                revoked.append(grant["grant_id"])
                logger.info("[temp_ak] 到期硬删 grant=%s user=%s", grant["grant_id"], grant.get("user_name"))
    except Exception:
        logger.error("[temp_ak] sweep failed", exc_info=True)
    return revoked
