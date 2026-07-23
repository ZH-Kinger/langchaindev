"""BotRole-* 角色（STS AssumeRole 目标），来自 ALIYUN_BOT_ROLE_MAPPING（单一事实源）。

本模块只声明**角色本体 + 信任策略**（INFRA-1）。角色的**权限策略**来自活体导出，
在 import 后作为单独的 RolePolicyAttachment 管理（见 README 的 live-export 步骤）——
此处留 TODO(live)。创建权限收归（INFRA-1B）对角色面的 Deny/Allow 挂载统一在
`ram/create_governance.py` 里做，避免分散。

信任策略必须与活体角色的 AssumeRolePolicyDocument 一致，否则 import 后 preview 出 diff。
"""
import json
from typing import Dict

import pulumi
import pulumi_alicloud as alicloud

import config


def _assume_role_doc() -> str:
    principals = config.trust_principal_arns()
    return json.dumps({
        "Statement": [{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {"RAM": principals},
        }],
        "Version": "1",
    })


def build() -> Dict[str, alicloud.ram.Role]:
    config.require_account_uid()
    doc = _assume_role_doc()
    out: Dict[str, alicloud.ram.Role] = {}
    for spec in config.roles():
        res_name = "role-" + spec.name.replace("_", "-").lower()
        role = alicloud.ram.Role(
            res_name,
            name=spec.name,
            document=doc,
            description=("Bot STS 角色"
                         + ("（特权：保留创建权限）" if spec.is_creator else "（默认：无创建权限）")),
            force=False,
        )
        out[spec.name] = role
        # TODO(live): 该角色在活体上已挂的权限策略，import 后在此补 RolePolicyAttachment。
        #   例：alicloud.ram.RolePolicyAttachment(f"{res_name}-attach-<Policy>",
        #        role_name=role.name, policy_name="<Policy>", policy_type="System"|"Custom")
    if not out:
        pulumi.log.warn("未解析出任何 BotRole-*（检查 ALIYUN_BOT_ROLE_MAPPING / ROLE_DEFAULT）")
    return out
