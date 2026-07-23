"""Master 用户 feishu-bot-master + 两个托管策略挂载。

Master 只应有 AliyunSTSAssumeRoleAccess + AliyunRAMReadOnlyAccess（真实资源权限在
BotRole-* 上）。**上线前在活体账号核实实际挂了哪些托管策略**，与此处一致才能 import no-op。
"""
import pulumi_alicloud as alicloud

import config

# Master 用户应挂的阿里云托管策略（System 类型）。以活体为准。
MASTER_MANAGED_POLICIES = ["AliyunSTSAssumeRoleAccess", "AliyunRAMReadOnlyAccess"]


def build() -> alicloud.ram.User:
    user = alicloud.ram.User(
        "master-user",
        name=config.MASTER_USER,
        force=False,
    )
    for pol in MASTER_MANAGED_POLICIES:
        alicloud.ram.UserPolicyAttachment(
            f"master-attach-{pol.lower()}",
            user_name=user.name,
            policy_name=pol,
            policy_type="System",
        )
    return user
