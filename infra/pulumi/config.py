"""单一事实源：从 Bot 读的同一份 .env + Pulumi stack config 解析 RAM/STS 结构。

设计要点：`BotRole-*` 角色、用户组、账号 UID 全部来自 Bot 运行时读的同一份
`ALIYUN_BOT_ROLE_MAPPING` / `ALIYUN_BOT_ROLE_DEFAULT` / `ALIYUN_BOT_ACCOUNT_UID`，
避免"云上角色 vs Bot 期望"漂移。envFile 指向项目 .env（默认 ../../.env）。
"""
import json
import os
from typing import List, NamedTuple

import pulumi

_cfg = pulumi.Config()            # namespace: aiops-infra


# ── 加载 Bot 的 .env（不覆盖已存在的进程环境变量，CI 里用真实环境优先）──────────────
def _load_env() -> None:
    env_file = _cfg.get("envFile")
    if not env_file:
        return
    path = env_file if os.path.isabs(env_file) else os.path.join(os.getcwd(), env_file)
    if not os.path.exists(path):
        pulumi.log.warn(f"envFile 不存在，跳过：{path}（将只用进程环境变量）")
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        pulumi.log.warn("未安装 python-dotenv，跳过 envFile 加载")
        return
    for k, v in dotenv_values(path).items():
        if v is not None:
            os.environ.setdefault(k, v)


_load_env()


# ── 基础标量 ──────────────────────────────────────────────────────────────────────
ACCOUNT_UID: str = os.environ.get("ALIYUN_BOT_ACCOUNT_UID", "").strip()
MASTER_USER: str = _cfg.get("masterUserName") or "feishu-bot-master"
TRUST_PRINCIPAL: str = _cfg.get("trustPrincipal") or "user"   # user | root
CREATOR_GROUP: str = (_cfg.get("creatorGroup") or "").strip()


def _split(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


# 审批白名单组 = 普通(非特权)组基线（与 settings.FEISHU_RAM_APPROVAL_ALLOWED_GROUPS 对齐）
APPROVAL_GROUPS: List[str] = _split(_cfg.get("approvalGroups") or "wuji_Algorithm,wuji_Examination")


def all_groups() -> List[str]:
    """本项目纳管的全部用户组 = 普通组 + 特权组（去重、保序）。"""
    seen, out = set(), []
    for g in APPROVAL_GROUPS + ([CREATOR_GROUP] if CREATOR_GROUP else []):
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def denied_groups() -> List[str]:
    """被"创建权限收归"显式 Deny 命中的组 = 普通组，排除特权组。"""
    return [g for g in APPROVAL_GROUPS if g and g != CREATOR_GROUP]


# ── BotRole-* 角色（来自 ALIYUN_BOT_ROLE_MAPPING，单一事实源）──────────────────────
class RoleSpec(NamedTuple):
    name: str          # BotRole-Xxx（= import 时的 RAM 角色名，必须与活体一致）
    arn: str
    groups: List[str]  # 映射到该角色的用户组
    is_creator: bool   # 是否为特权(CREATOR_GROUP)映射的角色 → 保留 ecs:RunInstances


def _role_name_from_arn(arn: str) -> str:
    # acs:ram::UID:role/BotRole-Algo -> BotRole-Algo
    return arn.rsplit("role/", 1)[-1] if "role/" in arn else arn


def default_role_arn() -> str:
    a = os.environ.get("ALIYUN_BOT_ROLE_DEFAULT", "").strip()
    if a:
        return a
    return f"acs:ram::{ACCOUNT_UID}:role/BotRole-Default" if ACCOUNT_UID else ""


def _parse_role_mapping() -> dict:
    """{role_arn: [groups...]}。容错解析，非法项跳过。"""
    raw = os.environ.get("ALIYUN_BOT_ROLE_MAPPING", "") or "[]"
    try:
        arr = json.loads(raw)
    except Exception:
        pulumi.log.warn("ALIYUN_BOT_ROLE_MAPPING 解析失败，按空处理")
        arr = []
    by_arn: dict = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        role, group = item.get("role"), item.get("group")
        if not role:
            continue
        by_arn.setdefault(role, [])
        if group:
            by_arn[role].append(group)
    return by_arn


def roles() -> List[RoleSpec]:
    by_arn = _parse_role_mapping()
    d = default_role_arn()
    if d:
        by_arn.setdefault(d, [])
    out = []
    for arn, groups in by_arn.items():
        out.append(RoleSpec(
            name=_role_name_from_arn(arn),
            arn=arn,
            groups=groups,
            is_creator=bool(CREATOR_GROUP and CREATOR_GROUP in groups),
        ))
    return out


def trust_principal_arns() -> List[str]:
    """角色信任策略的被信任主体。必须与活体角色的 AssumeRolePolicyDocument 一致，
    否则 import 后 preview 会出 diff。trustPrincipal=user → 只信任 master 用户；root → 整账号。"""
    if not ACCOUNT_UID:
        return []
    if TRUST_PRINCIPAL == "root":
        return [f"acs:ram::{ACCOUNT_UID}:root"]
    return [f"acs:ram::{ACCOUNT_UID}:user/{MASTER_USER}"]


# ── 创建权限收归（INFRA-1B）动作集（最终以活体核对后为准）────────────────────────────
ECS_CREATE_ACTIONS = ["ecs:RunInstances", "ecs:CreateInstance"]
RAM_CREATE_ACTIONS = [
    "ram:CreateUser", "ram:CreateAccessKey", "ram:CreateLoginProfile",
    "ram:AddUserToGroup", "ram:CreatePolicy", "ram:CreatePolicyVersion",
    "ram:AttachPolicyToUser", "ram:CreateRole",
]


def create_governance_actions() -> List[str]:
    return ECS_CREATE_ACTIONS + RAM_CREATE_ACTIONS


def require_account_uid() -> None:
    if not ACCOUNT_UID:
        raise pulumi.RunError(
            "ALIYUN_BOT_ACCOUNT_UID 为空：无法构造角色 ARN / 信任主体。"
            "请在 envFile(.env) 或环境变量里填真实主账号 UID。"
        )
