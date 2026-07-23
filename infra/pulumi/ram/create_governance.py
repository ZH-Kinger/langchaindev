"""创建权限收归（INFRA-1B）：ECS 实例 + RAM 用户/AK 创建权限，从多数人收回、只留特权组。

两条自定义策略 + 双面挂载：
  wuji-deny-create  (Effect: Deny)  → 非特权用户组 + 非特权 BotRole-*
  wuji-allow-create (Effect: Allow) → CREATOR_GROUP + 特权 BotRole-*
RAM 里 Deny 永远压过任何 Allow，故显式 Deny 可杜绝别处遗漏的 Allow。

铁律（见 plan）：
- 特权成员不得同时在被 Deny 的组里（Deny 会赢）。
- Bot 管理 AK 归属的 RAM 用户必须豁免（不在被 Deny 的组），否则审批开户/OSS 下发全废。
"""
import json
from typing import Dict

import pulumi
import pulumi_alicloud as alicloud

import config

DENY_POLICY_NAME = "wuji-deny-create"
ALLOW_POLICY_NAME = "wuji-allow-create"


def _doc(effect: str) -> str:
    return json.dumps({
        "Version": "1",
        "Statement": [{
            "Effect": effect,
            "Action": config.create_governance_actions(),
            "Resource": "*",
        }],
    })


def build(groups: Dict[str, alicloud.ram.Group],
          roles: Dict[str, alicloud.ram.Role]) -> None:
    deny = alicloud.ram.Policy(
        "policy-deny-create",
        policy_name=DENY_POLICY_NAME,
        policy_document=_doc("Deny"),
        description="收归创建权限：显式拒绝 ECS 实例 + RAM 用户/AK 创建（非特权组/角色）",
        force=True,   # 更新时允许删除旧版本
    )
    allow = alicloud.ram.Policy(
        "policy-allow-create",
        policy_name=ALLOW_POLICY_NAME,
        policy_document=_doc("Allow"),
        description="特权组/角色：允许 ECS 实例 + RAM 用户/AK 创建",
        force=True,
    )

    # ── 用户面：Deny → 非特权组；Allow → 特权组 ──────────────────────────────────
    denied = config.denied_groups()
    for g in denied:
        grp = groups.get(g)
        gname = grp.name if grp is not None else g
        alicloud.ram.GroupPolicyAttachment(
            f"deny-create-group-{g.replace('_', '-').lower()}",
            group_name=gname,
            policy_name=deny.policy_name,
            policy_type="Custom",
        )

    if config.CREATOR_GROUP:
        cg = groups.get(config.CREATOR_GROUP)
        cgname = cg.name if cg is not None else config.CREATOR_GROUP
        alicloud.ram.GroupPolicyAttachment(
            f"allow-create-group-{config.CREATOR_GROUP.replace('_', '-').lower()}",
            group_name=cgname,
            policy_name=allow.policy_name,
            policy_type="Custom",
        )
    else:
        pulumi.log.warn(
            "creatorGroup 未配置：只挂了 Deny，没有任何组被授予创建权限。"
            "请在 Pulumi.<stack>.yaml 设 aiops-infra:creatorGroup=<现有管理组名>。"
        )

    # ── 角色面：非特权 BotRole-* 挂 Deny；特权角色挂 Allow ────────────────────────
    # Bot manage_ecs 走 AssumeRole 后的角色权限，用户面 Deny 不覆盖角色，故角色面单独收。
    for spec in config.roles():
        role = roles.get(spec.name)
        if role is None:
            continue
        slug = spec.name.replace("_", "-").lower()
        if spec.is_creator:
            alicloud.ram.RolePolicyAttachment(
                f"allow-create-role-{slug}",
                role_name=role.name,
                policy_name=allow.policy_name,
                policy_type="Custom",
            )
        else:
            alicloud.ram.RolePolicyAttachment(
                f"deny-create-role-{slug}",
                role_name=role.name,
                policy_name=deny.policy_name,
                policy_type="Custom",
            )

    pulumi.export("createGovernance", {
        "denyPolicy": DENY_POLICY_NAME,
        "allowPolicy": ALLOW_POLICY_NAME,
        "deniedGroups": denied,
        "creatorGroup": config.CREATOR_GROUP or "(未配置)",
        "actions": config.create_governance_actions(),
    })
