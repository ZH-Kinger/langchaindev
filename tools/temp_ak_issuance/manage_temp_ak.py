"""manage_temp_ak 工具：临时 AK/SK 发放的只读预览 / 查询 / 手动吊销。

**没有 issue 动作**——发放只经飞书审批（审批通过才创建），工具绝不提供绕审批发凭证的口子（#55/#56 教训���。
revoke 需管理员（open_id==ADMIN_FEISHU_OPEN_ID）。plan 只算 policy 不调云；status 只读 Redis。
"""
import logging

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

logger = logging.getLogger(__name__)


class TempAkSchema(BaseModel):
    action: str = Field(description="plan / status / revoke")
    bucket: str = Field(default="", description="plan 用：OSS 桶展示名或真实桶名")
    prefix: str = Field(default="", description="plan 用：目录前缀（空=整桶）")
    caps: str = Field(default="", description="plan 用：能力，逗号分隔 read(列)/download(下载)/write(上传)")
    not_before: str = Field(default="", description="plan 用：生效时间 YYYY-MM-DD HH:MM:SS 或时间戳（空=立即）")
    expire: str = Field(default="", description="plan 用：到期时间 YYYY-MM-DD HH:MM:SS 或时间戳")
    grant_id: str = Field(default="", description="status/revoke 用：任务 ID（tak-…）")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


def manage_temp_ak(action: str, bucket: str = "", prefix: str = "", caps: str = "",
                   not_before: str = "", expire: str = "", grant_id: str = "", open_id: str = "") -> str:
    action = (action or "").strip().lower()
    from core.temp_ak_issuance import orchestrator, issuer, cleanup, policy
    import json as _json

    try:
        if action == "plan":
            if not bucket or not expire:
                return "❌ plan 需要 bucket 与 expire（到期时间）。"
            now = _parse_dt(None)  # now
            exp = _parse_dt(expire)
            if not exp:
                return "❌ expire 无法解析（用 YYYY-MM-DD HH:MM:SS 或时间戳）。"
            nb = _parse_dt(not_before) or now
            cap_list = [c.strip().lower() for c in (caps or "").replace(";", ",").split(",")
                        if c.strip().lower() in ("read", "download", "write")]
            grant = {
                "grant_id": "tak-plan", "bucket": bucket,
                "prefix": prefix.strip().lstrip("/"), "caps": cap_list,
                "not_before": nb, "expire": exp, "source_ips": [],
                "mode": issuer.classify_mode(exp, now),
                "user_name": "tempak-plan", "policy_name": policy.POLICY_PREFIX + "tempak-plan",
            }
            p = issuer.plan(grant)
            return (f"计划（dry-run，未发放）：\n"
                    f"- 分流：**{p['mode']}**（{'STS 单发含 Token,到点自灭' if p['mode']=='sts' else '方案B 长期AK+时间窗policy+到期硬删'}）\n"
                    f"- {orchestrator.scope_line(grant)}\n"
                    f"- 有效期：{orchestrator.fmt_window(grant)}\n"
                    f"- policy：\n```json\n{_json.dumps(p['policy'], ensure_ascii=False, indent=2)}\n```\n"
                    f"（真发放只经飞书审批，本工具不发凭证。）")

        if action == "status":
            if not grant_id:
                return "❌ status 需要 grant_id（tak-…）。"
            g = orchestrator.get_grant(grant_id)
            if not g:
                return f"未找到任务 `{grant_id}`（可能已过期）。"
            return (f"任务 `{grant_id}`：状态 **{g['stage']}** 模式 {g.get('mode')}\n"
                    f"- {orchestrator.scope_line(g)}\n"
                    f"- 有效期：{orchestrator.fmt_window(g)}\n"
                    f"- 接收方：{g.get('recipient_email') or '-'}｜RAM 用户：{g.get('user_name') or '-'}"
                    + (f"\n- 错误：{g['error']}" if g.get("error") else ""))

        if action == "revoke":
            from config.settings import settings
            if open_id != settings.ADMIN_FEISHU_OPEN_ID:
                return "❌ 手动吊销需管理员操作。"
            if not grant_id:
                return "❌ revoke 需要 grant_id（tak-…）。"
            g = orchestrator.get_grant(grant_id)
            if not g:
                return f"未找到任务 `{grant_id}`。"
            ok = cleanup.revoke_grant(g)
            return f"{'✅ 已吊销并清理' if ok else '❌ 吊销失败（见日志）'}：`{grant_id}`。"

        return f"❌ 未知 action：{action}，可选 plan / status / revoke。"
    except Exception as e:
        logger.error("[temp_ak] manage_temp_ak failed action=%s", action, exc_info=True)
        return f"❌ 操作失败：{e}"


def _csv(raw: str):
    import re
    return [x.strip().lstrip("/") for x in re.split(r"[,;\n]", raw or "") if x.strip()]


def _parse_dt(s):
    import time
    from datetime import datetime, timedelta, timezone
    if not s:
        return time.time()
    s = str(s).strip()
    if s.isdigit():
        v = float(s)
        return v / 1000 if v > 1e11 else v
    bj = timezone(timedelta(hours=8))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=bj).timestamp()
        except ValueError:
            continue
    return 0.0


temp_ak_tool = StructuredTool.from_function(
    func=manage_temp_ak,
    name="manage_temp_ak",
    description=("临时 AK/SK 发放（外部/卖数据）的预览与管理。plan 预览分流(STS/方案B)+policy(dry-run,不发凭证)；"
                 "status 查任务；revoke 管理员手动吊销。**发放只经飞书审批，本工具不发凭证**。"),
    args_schema=TempAkSchema,
)
