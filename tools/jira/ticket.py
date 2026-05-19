"""
Jira REST API v2 工具。
用于 GPU 工单的查询、创建、更新、评论操作。
"""
import json
from functools import lru_cache
from typing import Optional

import requests
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── HTTP 会话（复用连接）────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _session(jira_url: str, pat: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {pat}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return s


def _s() -> Optional[requests.Session]:
    url = settings.JIRA_URL.rstrip("/")
    pat = settings.JIRA_PAT
    if not url or not pat:
        return None
    return _session(url, pat)


def _base() -> str:
    return settings.JIRA_URL.rstrip("/")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def get_gpu_tickets(status: str = "待办", max_results: int = 20) -> list[dict]:
    """拉取 Jira GPU 申请工单列表。"""
    s = _s()
    if not s:
        return []
    project = settings.JIRA_PROJECT_KEY
    jql = f'project="{project}" AND labels="gpu-request" ORDER BY created DESC'
    if status:
        jql = f'project="{project}" AND labels="gpu-request" AND status="{status}" ORDER BY created ASC'
    try:
        resp = s.get(
            f"{_base()}/rest/api/2/search",
            params={"jql": jql, "maxResults": max_results,
                    "fields": "summary,status,description,labels,assignee,reporter,customfield_10000"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("issues", [])
    except Exception as e:
        print(f"[Jira] 查询工单失败: {e}")
        return []


def create_gpu_ticket(
    instance_name: str,
    gpu_count: str,
    duration_hours: str,
    purpose: str,
    cpu_cores: str = "",
    memory_gb: str = "",
    image_name: str = "",
    priority: str = "中",
    ssh_public_key: str = "",
    reporter_open_id: str = "",
    reporter_name: str = "",
    chat_id: str = "",
    needs_approval: bool = False,
) -> Optional[str]:
    """创建 GPU 申请工单，返回 Jira issue key（如 GPU-42）。"""
    s = _s()
    if not s:
        return None

    ssh_display = "是（公钥已提供）" if ssh_public_key else "否"
    lines = [
        "*GPU 资源申请*\n",
        f"- 实例名称：{instance_name}",
        f"- GPU 数量：{gpu_count} 卡",
        f"- CPU 核数：{cpu_cores or '默认'}",
        f"- 内存：{memory_gb or '默认'} GB",
        f"- 使用时长：{duration_hours} 小时",
        f"- 优先级：{priority}",
        f"- SSH 连接：{ssh_display}",
        f"- 镜像：{image_name or '默认'}",
        f"- 申请用途：{purpose}",
        f"- 申请人 OpenID：{reporter_open_id}",
        f"- 飞书 ChatID：{chat_id}",
        "",
        f"dsw_instance_name={instance_name}",
        f"dsw_gpu_count={gpu_count}",
        f"dsw_cpu_cores={cpu_cores}",
        f"dsw_memory_gb={memory_gb}",
        f"dsw_duration_hours={duration_hours}",
        f"dsw_image={image_name}",
        f"dsw_ssh_public_key={ssh_public_key}",
        f"dsw_priority={priority}",
        f"feishu_open_id={reporter_open_id}",
        f"feishu_chat_id={chat_id}",
        f"dsw_purpose={purpose}",
        f"dsw_requester_name={reporter_name or reporter_open_id}",
        f"dsw_needs_approval={'true' if needs_approval else 'false'}",
    ]
    description = "\n".join(lines)

    body = {
        "fields": {
            "project": {"key": settings.JIRA_PROJECT_KEY},
            "summary": (
                f"[GPU申请] {instance_name} "
                f"x{gpu_count}GPU {duration_hours}h [{priority}] "
                f"— {reporter_name or reporter_open_id}"
            ),
            "issuetype": {"name": settings.JIRA_ISSUE_TYPE},
            "description": description,
            "labels": ["gpu-request"],
            "priority": {"name": {"高": "High", "中": "Medium", "低": "Low"}.get(priority, "Medium")},
        }
    }
    try:
        resp = s.post(f"{_base()}/rest/api/2/issue", json=body, timeout=15)
        resp.raise_for_status()
        return resp.json().get("key")
    except Exception as e:
        print(f"[Jira] 创建工单失败: {e}")
        return None


def update_ticket_field(issue_key: str, fields: dict) -> bool:
    """更新工单字段（如 description 追加实例ID）。"""
    s = _s()
    if not s:
        return False
    try:
        resp = s.put(
            f"{_base()}/rest/api/2/issue/{issue_key}",
            json={"fields": fields},
            timeout=15,
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        print(f"[Jira] 更新工单失败: {e}")
        return False


def add_comment(issue_key: str, comment: str) -> bool:
    """向工单添加评论。"""
    s = _s()
    if not s:
        return False
    try:
        resp = s.post(
            f"{_base()}/rest/api/2/issue/{issue_key}/comment",
            json={"body": comment},
            timeout=15,
        )
        return resp.status_code == 201
    except Exception as e:
        print(f"[Jira] 添加评论失败: {e}")
        return False


def transition_ticket(issue_key: str, target_status: str) -> bool:
    """
    将工单流转到目标状态。
    target_status: 'In Progress' | 'Done' | 'Closed'
    """
    s = _s()
    if not s:
        return False
    try:
        resp = s.get(f"{_base()}/rest/api/2/issue/{issue_key}/transitions", timeout=15)
        resp.raise_for_status()
        transitions = resp.json().get("transitions", [])
        tid = next((t["id"] for t in transitions
                    if t["name"].lower() == target_status.lower()), None)
        if not tid:
            # 宽松匹配（包含关系）
            tid = next((t["id"] for t in transitions
                        if target_status.lower() in t["name"].lower()), None)
        if not tid:
            print(f"[Jira] 找不到流转 '{target_status}'，可用: {[t['name'] for t in transitions]}")
            return False
        resp2 = s.post(
            f"{_base()}/rest/api/2/issue/{issue_key}/transitions",
            json={"transition": {"id": tid}},
            timeout=15,
        )
        return resp2.status_code == 204
    except Exception as e:
        print(f"[Jira] 流转工单失败: {e}")
        return False


def parse_ticket_metadata(description: str) -> dict:
    """从工单描述里解析 dsw_* 和 feishu_* 键值对。"""
    result = {}
    for line in description.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            if k.startswith(("dsw_", "feishu_")):
                result[k] = v.strip()
    return result


def list_jira_projects() -> str:
    """列出当前用户有权访问的 Jira 项目（用于调试/发现 project key）。"""
    s = _s()
    if not s:
        return "❌ Jira 未配置。"
    try:
        resp = s.get(f"{_base()}/rest/api/2/project", timeout=15)
        resp.raise_for_status()
        projects = resp.json()
        if not projects:
            return "没有可访问的 Jira 项目。"
        lines = ["## 可访问的 Jira 项目\n"]
        for p in projects:
            lines.append(f"- **{p['key']}** — {p['name']}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 查询 Jira 项目失败：{e}"


# ── Schema ────────────────────────────────────────────────────────────────────

class JiraSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'list_tickets'   - 列出 GPU 申请工单（可按 status 过滤）\n"
            "  'list_projects'  - 列出所有可访问的 Jira 项目\n"
            "  'add_comment'    - 向工单添加评论（需 issue_key + comment）\n"
            "  'close_ticket'   - 将工单标记为完成（需 issue_key）"
        )
    )
    issue_key: str = Field(default="", description="Jira 工单 Key，如 GPU-42。")
    status: str = Field(default="", description="按状态过滤，如 Open / In Progress / Done。")
    comment: str = Field(default="", description="评论内容（add_comment 时使用）。")


def _manage_jira(action: str, issue_key: str = "", status: str = "", comment: str = "") -> str:
    action = action.strip().lower()
    if action == "list_tickets":
        tickets = get_gpu_tickets(status=status)
        if not tickets:
            return f"没有找到 GPU 申请工单（status={status or '全部'}）。"
        lines = [f"## GPU 申请工单（{len(tickets)} 条）\n"]
        for t in tickets:
            f = t.get("fields", {})
            lines.append(
                f"- **{t['key']}** [{f.get('status', {}).get('name', '-')}] "
                f"{f.get('summary', '-')}"
            )
        return "\n".join(lines)
    if action == "list_projects":
        return list_jira_projects()
    if action == "add_comment":
        if not issue_key or not comment:
            return "❌ add_comment 需要 issue_key 和 comment。"
        ok = add_comment(issue_key, comment)
        return f"✅ 评论已添加到 {issue_key}。" if ok else f"❌ 评论失败。"
    if action == "close_ticket":
        if not issue_key:
            return "❌ close_ticket 需要 issue_key。"
        transition_ticket(issue_key, "完成")
        add_comment(issue_key, "DSW 实例已停止，工单自动关闭。")
        return f"✅ 工单 {issue_key} 已关闭。"
    return f"❓ 未知操作：{action}"


jira_tool = StructuredTool.from_function(
    func=_manage_jira,
    name="manage_jira",
    description=(
        "管理 Jira GPU 资源申请工单。"
        "支持列出工单、查询项目列表、添加评论、关闭工单。"
        "当用户询问 GPU 申请状态、工单信息时调用。"
    ),
    args_schema=JiraSchema,
)
