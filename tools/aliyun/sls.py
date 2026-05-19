"""
阿里云 SLS（日志服务）管理工具。

凭证通过 STS AssumeRole 获取（utils.aliyun_client_factory），
按飞书 open_id 隔离权限。

写操作（create_project / create_logstore）必须 confirm=true。
"""
import logging
import time

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings
from utils.aliyun_client_factory import get_sls_client

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class SLSSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'list_projects'    - 列出所有 SLS 项目\n"
            "  'list_logstores'   - 列出项目下的 logstore，需 project\n"
            "  'query'            - 查询日志，需 project + logstore + query\n"
            "  'create_project'   - 创建 SLS 项目，需 project；confirm=true\n"
            "  'create_logstore'  - 创建 logstore，需 project+logstore；confirm=true"
        )
    )
    project: str = Field(default="", description="SLS 项目名")
    logstore: str = Field(default="", description="logstore 名")
    query: str = Field(default="*", description="SLS 查询语句，默认 '*'")
    from_minutes: int = Field(default=60, description="查询过去多少分钟，默认 60")
    region: str = Field(default="", description="SLS region")
    max_lines: int = Field(default=20, description="返回行数上限，默认 20")

    # 写操作参数
    description: str = Field(default="", description="项目/logstore 描述")
    ttl_days: int = Field(default=30, description="logstore 日志保留天数，默认 30")
    shard_count: int = Field(default=2, description="logstore shard 数，默认 2")

    confirm: bool = Field(default=False, description="所有写操作必须显式 true")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def manage_sls(
    action: str,
    project: str = "",
    logstore: str = "",
    query: str = "*",
    from_minutes: int = 60,
    region: str = "",
    max_lines: int = 20,
    description: str = "",
    ttl_days: int = 30,
    shard_count: int = 2,
    confirm: bool = False,
    open_id: str = "",
) -> str:
    action = action.strip().lower()
    client = get_sls_client(open_id=open_id, region=region)
    if client is None:
        return "❌ SLS 凭证不可用：请确认 STS / RAM 映射已配置，且已 pip install aliyun-log-python-sdk。"

    try:
        # ── 只读：list_projects ──────────────────────────────────────────────
        if action == "list_projects":
            resp = client.list_project()
            projects = resp.get_projects() or []
            if not projects:
                return "当前账号下无 SLS 项目（或角色无查询权限）。"
            lines = [f"## SLS 项目（共 {len(projects)} 个）"]
            lines.append("| 项目名 | 区域 | 描述 |")
            lines.append("|--------|------|------|")
            for p in projects:
                lines.append(
                    f"| `{p.get('projectName', '-')}` | "
                    f"{p.get('region', '-')} | {p.get('description', '') or '-'} |"
                )
            return "\n".join(lines)

        # ── 只读：list_logstores ─────────────────────────────────────────────
        if action == "list_logstores":
            if not project:
                return "❌ list_logstores 需要提供 project。"
            resp = client.list_logstore(project)
            stores = resp.get_logstores() or []
            if not stores:
                return f"项目 `{project}` 下无 logstore。"
            lines = [f"## `{project}` 下 logstore（共 {len(stores)} 个）"]
            for s in stores:
                lines.append(f"- `{s}`")
            return "\n".join(lines)

        # ── 只读：query ──────────────────────────────────────────────────────
        if action == "query":
            if not project or not logstore:
                return "❌ query 需要提供 project 和 logstore。"
            end_ts   = int(time.time())
            start_ts = end_ts - max(60, from_minutes * 60)
            resp = client.get_log(project, logstore, start_ts, end_ts,
                                   query=query or "*", line=max_lines)
            logs = resp.get_logs() or []
            if not logs:
                return (
                    f"项目 `{project}` / logstore `{logstore}` "
                    f"过去 {from_minutes} 分钟内查询 `{query}` 无结果。"
                )
            lines = [
                f"## SLS 日志结果",
                f"项目：`{project}`　Logstore：`{logstore}`　过去 {from_minutes} 分钟　"
                f"查询：`{query}`　返回 {len(logs)} 条",
                "",
            ]
            for i, lg in enumerate(logs, 1):
                ts = lg.timestamp
                contents = lg.get_contents() or {}
                preview = " ".join(f"{k}={v}" for k, v in list(contents.items())[:4])
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                lines.append(f"{i}. [{time.strftime('%H:%M:%S', time.localtime(ts))}] {preview}")
            return "\n".join(lines)

        # ── 写操作：统一拦截 confirm ────────────────────────────────────────
        if action in ("create_project", "create_logstore") and not confirm:
            return (
                f"⚠️ SLS `{action}` 是写操作，"
                "请明确传入 `confirm=true` 后再调用。"
            )

        if action == "create_project":
            if not project:
                return "❌ create_project 需要 project 名称。"
            client.create_project(project, description or "")
            logger.info("[SLS] 创建项目 %s 操作者=%s", project, open_id)
            return (
                f"✅ SLS 项目 `{project}` 已创建。\n"
                f"接下来可用 `create_logstore` 在此项目下建 logstore。"
            )

        if action == "create_logstore":
            if not project or not logstore:
                return "❌ create_logstore 需要 project 和 logstore。"
            client.create_logstore(project, logstore, ttl=ttl_days, shard_count=shard_count)
            logger.info("[SLS] 创建 logstore %s/%s ttl=%d shards=%d 操作者=%s",
                        project, logstore, ttl_days, shard_count, open_id)
            return (
                f"✅ logstore `{logstore}` 已在项目 `{project}` 下创建。\n"
                f"- 保留天数：{ttl_days}\n"
                f"- Shard 数：{shard_count}"
            )

        return (
            f"❓ 未知 action：{action}，可选 list_projects / list_logstores / query / "
            "create_project / create_logstore。"
        )

    except Exception as e:
        logger.error("[SLS] 调用失败 action=%s err=%s", action, e, exc_info=True)
        return f"❌ SLS API 调用失败：{e}"


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

sls_tool = StructuredTool.from_function(
    func=manage_sls,
    name="manage_sls",
    description=(
        "管理阿里云 SLS 日志服务。"
        "只读：list_projects 列项目、list_logstores 列 logstore、query 查日志。"
        "写操作（create_project / create_logstore）必须 confirm=true。"
        "query 支持 SLS 查询语法，如 'level:ERROR'、'request_id:abc'。"
    ),
    args_schema=SLSSchema,
)
