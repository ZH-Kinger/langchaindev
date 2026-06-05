"""
Jira 工作流查询工具。

专为日常工作流设计（与 jira_tool.py 的 GPU 工单工具互相独立）：
  - standup : 拉昨日活动 + 进行中 Story
  - plan    : 拉当前 Sprint / Epic 列表
  - weekly  : 拉本周完成 + 下周待办
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import requests
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── HTTP 会话 ────────────────────────────────────────────────────────────────

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


# SWD 项目已完结状态（Story 流程末段）
_SWD_DONE_STATUSES = ("产品验收", "QA验收", "已完成", "已关闭", "已解决")


def _default_project() -> str:
    # JIRA_WORKFLOW_PROJECT 优先；未配置时默认 SWD（算法组项目 Key）
    return settings.JIRA_WORKFLOW_PROJECT or "SWD"


# ── 格式化辅助 ────────────────────────────────────────────────────────────────

def _fmt(issue: dict) -> str:
    f       = issue.get("fields", {})
    status  = f.get("status", {}).get("name", "-")
    owner   = (f.get("assignee") or {}).get("displayName", "未分配")
    itype   = f.get("issuetype", {}).get("name", "")
    tag     = f"[{itype}] " if itype else ""
    return f"- **{issue['key']}** {tag}[{status}] {f.get('summary', '-')} @{owner}"


def _short(issue: dict) -> str:
    """工单号 + 摘要截断（≤20字），用于表格单元格。"""
    key     = issue.get("key", "?")
    summary = issue.get("fields", {}).get("summary", "-")
    if len(summary) > 20:
        summary = summary[:19] + "…"
    return f"{key} {summary}"


def _search(jql: str, fields: str = "summary,status,assignee,issuetype",
            max_results: int = 30) -> list[dict]:
    s = _s()
    if not s:
        return []
    resp = s.get(
        f"{_base()}/rest/api/2/search",
        params={"jql": jql, "maxResults": max_results, "fields": fields},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("issues", [])


# ── 业务查询函数 ──────────────────────────────────────────────────────────────

def query_my_activity(username: str, period: str = "yesterday", project: str = "") -> str:
    """
    拉取指定用户在某时间段内的 Jira 活动（更新 / 完成的 Story / Bug）。

    period: 'yesterday' | 'this_week' | 'last_week'
    """
    if not _s():
        return "❌ Jira 未配置（JIRA_URL / JIRA_PAT）。"

    proj    = project or _default_project()
    assignee = f'assignee = "{username}"' if username else "assignee = currentUser()"
    period_jql = {
        "yesterday": "updated >= -1d",
        "this_week": "updated >= startOfWeek()",
        "last_week": "updated >= startOfWeek(-1w) AND updated < startOfWeek()",
    }.get(period, "updated >= -1d")

    jql = f'project = "{proj}" AND {assignee} AND {period_jql} ORDER BY updated DESC'
    try:
        issues = _search(jql, max_results=20)
    except Exception as e:
        return f"❌ Jira 查询失败：{e}"

    if not issues:
        return f"✅ **{period}** 无 Jira 更新记录（{assignee}）。"

    lines = [f"## {period} Jira 活动（{len(issues)} 条）\n"]
    lines += [_fmt(i) for i in issues]
    return "\n".join(lines)


def query_in_progress(username: str = "", project: str = "") -> str:
    """查询当前进行中的 Story / Bug（用于例会"今天计划"部分）。"""
    if not _s():
        return "❌ Jira 未配置。"

    proj    = project or _default_project()
    assignee = f'assignee = "{username}"' if username else "assignee = currentUser()"
    jql = f'project = "{proj}" AND {assignee} AND status = "进行中" ORDER BY priority DESC'

    try:
        issues = _search(jql, max_results=10)
    except Exception as e:
        return f"❌ 查询进行中任务失败：{e}"

    if not issues:
        return "✅ 当前无进行中 Story。"

    lines = [f"## 进行中（{len(issues)} 条）\n"]
    lines += [_fmt(i) for i in issues]
    return "\n".join(lines)


def query_sprint_stories(project: str = "", assignee: str = "") -> str:
    """
    查询当前 Sprint 的所有 Story，按状态分组输出。
    用于 /standup（快速看整个 Sprint 全景）和 /plan。
    """
    if not _s():
        return "❌ Jira 未配置。"

    proj           = project or _default_project()
    assignee_clause = f' AND assignee = "{assignee}"' if assignee else ""
    jql = f'project = "{proj}" AND sprint in openSprints(){assignee_clause} ORDER BY priority DESC'

    try:
        issues = _search(jql, fields="summary,status,assignee,priority,issuetype", max_results=50)
    except Exception as e:
        return f"❌ Sprint 查询失败（可能未启用 Agile Board）：{e}"

    if not issues:
        return "当前 Sprint 无 Story，或该项目未配置 Agile Board。"

    # 按状态分组
    groups: dict[str, list] = {}
    for i in issues:
        st = i["fields"].get("status", {}).get("name", "未知")
        groups.setdefault(st, []).append(i)

    lines = [f"## 当前 Sprint（{len(issues)} 条）\n"]
    for status, items in groups.items():
        lines.append(f"\n### {status}（{len(items)}）")
        lines += [_fmt(i) for i in items]
    return "\n".join(lines)


def query_epics(project: str = "") -> str:
    """列出项目中未完成的 Epic（用于 /plan 立项阶段）。"""
    if not _s():
        return "❌ Jira 未配置。"

    proj = project or _default_project()
    jql  = f'project = "{proj}" AND issuetype = Epic AND status != "完成" ORDER BY created DESC'

    try:
        issues = _search(jql, fields="summary,status,assignee,description", max_results=20)
    except Exception as e:
        return f"❌ Epic 查询失败：{e}"

    if not issues:
        return "当前无进行中的 Epic。"

    lines = [f"## Epic 列表（{len(issues)} 条）\n"]
    lines += [_fmt(i) for i in issues]
    return "\n".join(lines)


# ── Bot 专用原始数据接口（返回 list[dict]，供 feishu_bot 自行格式化）────────────

def query_done_raw(username: str = "", project: str = "") -> list[dict]:
    """返回昨日推进到已完结状态的工单列表（Story + Bug）。"""
    if not _s():
        return []
    proj     = project or _default_project()
    assignee = f'assignee = "{username}"' if username else "assignee = currentUser()"
    statuses = ", ".join(f'"{s}"' for s in _SWD_DONE_STATUSES)
    jql = (
        f'project = "{proj}" AND {assignee} '
        f'AND status in ({statuses}) AND updated >= -1d '
        f'ORDER BY updated DESC'
    )
    try:
        return _search(jql, max_results=15)
    except Exception:
        return []


def query_in_progress_raw(username: str = "", project: str = "") -> list[dict]:
    """返回当前进行中（需求确认 / 进行中）的工单列表。"""
    if not _s():
        return []
    proj     = project or _default_project()
    assignee = f'assignee = "{username}"' if username else "assignee = currentUser()"
    jql = (
        f'project = "{proj}" AND {assignee} '
        f'AND status in ("需求确认", "进行中") '
        f'ORDER BY priority DESC'
    )
    try:
        return _search(jql, max_results=10)
    except Exception:
        return []


def query_done_this_week_raw(username: str = "", project: str = "") -> list[dict]:
    """返回本周进入已完结状态的工单列表（供周报使用）。"""
    if not _s():
        return []
    proj     = project or _default_project()
    assignee = f'assignee = "{username}"' if username else "assignee = currentUser()"
    statuses = ", ".join(f'"{s}"' for s in _SWD_DONE_STATUSES)
    jql = (
        f'project = "{proj}" AND {assignee} '
        f'AND status in ({statuses}) AND updated >= startOfWeek() '
        f'ORDER BY updated DESC'
    )
    try:
        return _search(jql, max_results=30)
    except Exception:
        return []


# ── Schema + Tool ─────────────────────────────────────────────────────────────

class JiraWorkflowSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'my_activity'    - 查指定用户昨日/本周的 Jira 活动，用于例会和周报\n"
            "  'in_progress'    - 查用户当前进行中的 Story\n"
            "  'sprint_stories' - 查当前 Sprint 全部 Story，用于 Sprint 规划\n"
            "  'epics'          - 列出未完成的 Epic，用于立项规划"
        )
    )
    username: str = Field(default="", description="Jira 用户的 displayName 或 username。留空使用 PAT 所有者身份。")
    period:   str = Field(default="yesterday", description="时间范围：yesterday | this_week | last_week")
    project:  str = Field(default="", description="Jira 项目 Key。留空则使用 JIRA_WORKFLOW_PROJECT 配置。")
    assignee: str = Field(default="", description="sprint_stories 时按负责人过滤，留空返回全部。")


def _dispatch(action: str, username: str = "", period: str = "yesterday",
              project: str = "", assignee: str = "") -> str:
    match action.strip().lower():
        case "my_activity":    return query_my_activity(username, period, project)
        case "in_progress":    return query_in_progress(username, project)
        case "sprint_stories": return query_sprint_stories(project, assignee)
        case "epics":          return query_epics(project)
        case _:                return f"❓ 未知操作：{action}"


jira_workflow_tool = StructuredTool.from_function(
    func=_dispatch,
    name="query_jira_workflow",
    description=(
        "查询 Jira 工作流数据，用于生成每日例会草稿、Sprint 规划和周报。"
        "支持：个人昨日/本周 Jira 活动、当前进行中任务、Sprint 全览、Epic 列表。"
        "当用户发送 /standup、/plan、/weekly 或询问「我昨天做了什么」时调用。"
    ),
    args_schema=JiraWorkflowSchema,
)