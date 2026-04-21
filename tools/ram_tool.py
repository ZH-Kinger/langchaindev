"""
阿里云 RAM 用户管理工具。
从现有 DSW 实例中提取团队成员，或通过 RAM API 查询（需 ram:ListUsers 权限）。
"""
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── 从 DSW 实例提取团队成员（无需 RAM 权限）──────────────────────────────────────

def list_team_members() -> list[dict]:
    """从现有 DSW 实例中提取所有团队成员（user_name + user_id），去重后返回。"""
    try:
        from tools.pai_dsw_tool import _client
        from alibabacloud_pai_dsw20220101 import models as dsw_models

        client = _client()
        if not client:
            return []
        req = dsw_models.ListInstancesRequest(
            page_size=100, page_number=1,
            workspace_id=settings.PAI_DSW_WORKSPACE_ID,
            resource_id=settings.PAI_DSW_RESOURCE_ID,
        )
        resp = client.list_instances(req)
        seen: dict[str, str] = {}
        for inst in resp.body.instances or []:
            uid = inst.user_id or ""
            name = inst.user_name or ""
            if uid and name and name != "power-application-user" and uid not in seen:
                seen[uid] = name
        return [{"user_id": uid, "user_name": name} for uid, name in seen.items()]
    except Exception as e:
        print(f"[RAM] 获取团队成员失败: {e}")
        return []


# ── RAM API 查询（需主账号或有 ram:ListUsers 权限的子账号）──────────────────────

def list_ram_users_api() -> list[dict]:
    """通过 RAM API 列出所有子用户，需 ram:ListUsers 权限。失败时返回空列表。"""
    try:
        from alibabacloud_ram20150501.client import Client
        from alibabacloud_ram20150501 import models as ram_models
        from alibabacloud_tea_openapi import models as open_api_models

        # 优先用 PAI DSW 主账号凭证（权限更高），降级用 Prometheus 凭证
        ak = settings.PAI_DSW_ACCESS_KEY_ID or settings.ALIYUN_ACCESS_KEY_ID
        sk = settings.PAI_DSW_ACCESS_KEY_SECRET or settings.ALIYUN_ACCESS_KEY_SECRET
        cfg = open_api_models.Config(
            access_key_id=ak,
            access_key_secret=sk,
            endpoint="ram.aliyuncs.com",
        )
        client = Client(cfg)
        resp = client.list_users(ram_models.ListUsersRequest())
        return [
            {
                "user_id": u.user_id or "",
                "user_name": u.user_name or "",
                "display_name": u.display_name or "",
            }
            for u in (resp.body.users.user or [])
        ]
    except Exception:
        return []


# ── 飞书 open_id ↔ RAM user 映射（存 Redis）────────────────────────────────────

_MAP_KEY = "feishu:user_ram_map"


def save_user_map(open_id: str, user_name: str, user_id: str = "") -> None:
    """保存飞书 open_id → RAM 用户名/ID 映射。"""
    try:
        from utils.redis_client import get_redis
        import json
        r = get_redis()
        raw = r.get(_MAP_KEY)
        mapping = json.loads(raw) if raw else {}
        mapping[open_id] = {"user_name": user_name, "user_id": user_id}
        r.set(_MAP_KEY, json.dumps(mapping, ensure_ascii=False))
    except Exception as e:
        print(f"[RAM] 保存映射失败: {e}")


def get_user_map() -> dict:
    """返回完整映射表 {open_id: {user_name, user_id}}。"""
    try:
        from utils.redis_client import get_redis
        import json
        raw = get_redis().get(_MAP_KEY)
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def get_ram_user_by_open_id(open_id: str) -> Optional[dict]:
    """按飞书 open_id 查找对应的 RAM 用户信息。"""
    return get_user_map().get(open_id)


# ── LangChain 工具函数 ────────────────────────────────────────────────────────

class RAMSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'list_members'  - 从 DSW 实例提取团队成员列表（无需 RAM 权限）\n"
            "  'list_ram'      - 通过 RAM API 列出所有子用户（需 ram:ListUsers 权限）\n"
            "  'show_map'      - 查看当前飞书 open_id ↔ RAM 用户映射表"
        )
    )


def _manage_ram(action: str) -> str:
    action = action.strip().lower()

    if action == "list_members":
        members = list_team_members()
        if not members:
            return "未找到团队成员（DSW 实例为空或 API 不可用）。"
        lines = [f"## 团队成员（从 DSW 实例提取，共 {len(members)} 人）\n"]
        for m in members:
            lines.append(f"- **{m['user_name']}**  `{m['user_id']}`")
        return "\n".join(lines)

    if action == "list_ram":
        users = list_ram_users_api()
        if not users:
            return "❌ RAM API 调用失败，当前 AK/SK 可能缺少 ram:ListUsers 权限。"
        lines = [f"## RAM 子用户（共 {len(users)} 个）\n"]
        for u in users:
            lines.append(
                f"- **{u['display_name'] or u['user_name']}**  "
                f"登录名: `{u['user_name']}`  ID: `{u['user_id']}`"
            )
        return "\n".join(lines)

    if action == "show_map":
        mapping = get_user_map()
        if not mapping:
            return "当前没有飞书 open_id ↔ RAM 用户映射记录。"
        lines = ["## 飞书用户 ↔ RAM 用户映射\n"]
        for oid, info in mapping.items():
            lines.append(f"- `{oid}` → **{info['user_name']}**  `{info['user_id']}`")
        return "\n".join(lines)

    return f"❓ 未知操作：{action}"


ram_tool = StructuredTool.from_function(
    func=_manage_ram,
    name="manage_ram",
    description=(
        "管理阿里云 RAM 用户信息。"
        "支持列出团队成员（从 DSW 实例提取）、查询 RAM 子用户列表、查看飞书用户映射表。"
        "当用户询问「团队成员」、「有哪些用户」、「谁有账号」等问题时调用。"
    ),
    args_schema=RAMSchema,
)
