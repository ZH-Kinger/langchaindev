"""RAM 用户组：审批白名单组 + 特权组（CREATOR_GROUP）。

全部 import-first：`name` 必须与活体组名一致，import 后 preview 应 no-op。
注意：pulumi-alicloud 资源属性名可能随 provider 版本微调（如 name/group_name），
若 preview 报未知属性，按已装版本 `pulumi about` / provider schema 调整。
"""
from typing import Dict

import pulumi_alicloud as alicloud

import config


def build() -> Dict[str, alicloud.ram.Group]:
    groups: Dict[str, alicloud.ram.Group] = {}
    for g in config.all_groups():
        # resource name 用安全 slug；RAM 侧真实名用 name=（= import 标识）
        res_name = "group-" + g.replace("_", "-").lower()
        groups[g] = alicloud.ram.Group(
            res_name,
            name=g,
            force=False,  # 删除保护：组内有成员时不强删
        )
    return groups
