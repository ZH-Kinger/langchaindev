"""
GitHub 工作流查询工具。

专为日常工作流设计（例会 / 周报 / Sprint 规划）：
  - my_commits : 查询用户在某时间段内的提交记录
  - my_prs     : 查询用户的 PR 列表（open / merged）
  - org_repos  : 列出组织下的仓库（上下文参考）
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import requests
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── HTTP 会话 ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return s


def _s() -> Optional[requests.Session]:
    token = settings.GITHUB_TOKEN
    if not token:
        return None
    return _session(token)


_API = "https://api.github.com"


# ── 日期工具 ──────────────────────────────────────────────────────────────────

def _since_iso(period: str) -> str:
    """将 period 字符串转为 GitHub API 需要的 ISO 8601 日期。"""
    now = datetime.now(tz=timezone.utc)
    if period == "yesterday":
        dt = now - timedelta(days=1)
    elif period == "this_week":
        dt = now - timedelta(days=now.weekday())   # 本周一 00:00
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "last_week":
        dt = now - timedelta(days=now.weekday() + 7)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        dt = now - timedelta(days=1)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 业务查询函数 ──────────────────────────────────────────────────────────────

def query_my_commits(username: str, period: str = "yesterday", org: str = "") -> str:
    """
    跨仓库搜索指定用户在某时间段内的 commits。

    使用 GitHub Search Commits API（需 token 有 repo 权限）。
    """
    s = _s()
    if not s:
        return "❌ GitHub 未配置（GITHUB_TOKEN）。"

    since = _since_iso(period)
    since_date = since[:10]           # YYYY-MM-DD
    org_filter = f" org:{org or settings.GITHUB_ORG}" if (org or settings.GITHUB_ORG) else ""
    q = f"author:{username} committer-date:>={since_date}{org_filter}"

    try:
        resp = s.get(
            f"{_API}/search/commits",
            params={"q": q, "sort": "committer-date", "order": "desc", "per_page": 30},
            headers={"Accept": "application/vnd.github.cloak-preview+json"},
            timeout=15,
        )
        resp.raise_for_status()
        data  = resp.json()
        items = data.get("items", [])
    except Exception as e:
        return f"❌ GitHub commits 查询失败：{e}"

    if not items:
        return f"✅ **{period}** 无 commit 记录（author={username}）。"

    lines = [f"## {period} Commits（{len(items)} 条）\n"]
    for c in items:
        repo   = c.get("repository", {}).get("full_name", "unknown/repo")
        sha    = c.get("sha", "")[:7]
        msg    = (c.get("commit", {}).get("message") or "").splitlines()[0]
        date   = (c.get("commit", {}).get("committer", {}).get("date") or "")[:10]
        url    = c.get("html_url", "")
        lines.append(f"- `{sha}` [{repo}]({url}) {msg}  *{date}*")

    return "\n".join(lines)


def query_my_prs(username: str, org: str = "", state: str = "open") -> str:
    """
    查询用户的 PR 列表。

    state: 'open' | 'merged' | 'all'
    merged 会查询 is:pr is:merged，其余用 state 过滤。
    """
    s = _s()
    if not s:
        return "❌ GitHub 未配置（GITHUB_TOKEN）。"

    org_filter = f" org:{org or settings.GITHUB_ORG}" if (org or settings.GITHUB_ORG) else ""

    if state == "merged":
        q = f"type:pr is:merged author:{username}{org_filter}"
    else:
        state_clause = "is:open" if state == "open" else ""
        q = f"type:pr {state_clause} author:{username}{org_filter}"

    try:
        resp = s.get(
            f"{_API}/search/issues",
            params={"q": q, "sort": "updated", "order": "desc", "per_page": 20},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        return f"❌ GitHub PR 查询失败：{e}"

    if not items:
        return f"✅ 无 {state} PR（author={username}）。"

    label = {"open": "Open PR", "merged": "已合并 PR", "all": "全部 PR"}.get(state, "PR")
    lines = [f"## {label}（{len(items)} 条）\n"]
    for pr in items:
        repo  = _repo_from_url(pr.get("repository_url", ""))
        num   = pr.get("number", "")
        title = pr.get("title", "-")
        url   = pr.get("html_url", "")
        updated = (pr.get("updated_at") or "")[:10]
        lines.append(f"- **#{num}** [{repo}]({url}) {title}  *{updated}*")

    return "\n".join(lines)


def query_org_repos(org: str = "") -> str:
    """列出组织下的仓库（最近活跃排序），用于规划时提供上下文。"""
    s = _s()
    if not s:
        return "❌ GitHub 未配置（GITHUB_TOKEN）。"

    target = org or settings.GITHUB_ORG
    if not target:
        return "❌ 未指定 org，请传入 org 参数或在 .env 设置 GITHUB_ORG。"

    try:
        resp = s.get(
            f"{_API}/orgs/{target}/repos",
            params={"sort": "pushed", "per_page": 20},
            timeout=15,
        )
        resp.raise_for_status()
        repos = resp.json()
    except Exception as e:
        return f"❌ 查询仓库失败：{e}"

    if not repos:
        return f"组织 {target} 下无可见仓库。"

    lines = [f"## {target} 仓库列表（最近活跃）\n"]
    for r in repos:
        pushed = (r.get("pushed_at") or "")[:10]
        desc   = r.get("description") or ""
        lines.append(f"- **{r['name']}** {desc}  *最后推送 {pushed}*")
    return "\n".join(lines)


# ── 内部工具函数 ──────────────────────────────────────────────────────────────

def _repo_from_url(url: str) -> str:
    """从 repository_url 提取 owner/repo 格式。"""
    m = re.search(r"repos/(.+/.+)$", url)
    return m.group(1) if m else url


# ── Schema + Tool ─────────────────────────────────────────────────────────────

class GitHubWorkflowSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'my_commits' - 查询用户昨日/本周的 commit 记录，用于例会和周报\n"
            "  'my_prs'     - 查询用户的 PR 列表（open / merged）\n"
            "  'org_repos'  - 列出组织下的仓库"
        )
    )
    username: str = Field(default="", description="GitHub 用户名（login）。")
    period:   str = Field(default="yesterday", description="时间范围：yesterday | this_week | last_week")
    org:      str = Field(default="", description="GitHub 组织名。留空则使用 GITHUB_ORG 配置。")
    state:    str = Field(default="open", description="PR 状态：open | merged | all（my_prs 时使用）。")


def _dispatch(action: str, username: str = "", period: str = "yesterday",
              org: str = "", state: str = "open") -> str:
    match action.strip().lower():
        case "my_commits": return query_my_commits(username, period, org)
        case "my_prs":     return query_my_prs(username, org, state)
        case "org_repos":  return query_org_repos(org)
        case _:            return f"❓ 未知操作：{action}"


github_workflow_tool = StructuredTool.from_function(
    func=_dispatch,
    name="query_github_workflow",
    description=(
        "查询 GitHub 工作流数据，用于生成每日例会草稿和周报。"
        "支持：跨仓库 commit 记录、PR 列表、组织仓库列表。"
        "当用户发送 /standup、/weekly 或询问「我昨天提交了什么」时调用。"
    ),
    args_schema=GitHubWorkflowSchema,
)