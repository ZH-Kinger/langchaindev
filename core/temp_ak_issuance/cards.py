"""飞书卡片：临时 AK 发放（B 方案：凭证走审批评论，卡片只做脱敏展示）。

- receipt_card：内部群脱敏回执（发放成功）。
- status_card：查询/工具用的脱敏状态卡。
均**不含 secret/token**——凭证明文只在审批评论正文（见 delivery.credential_text）。
"""
from tools.feishu.cards import card, div, fields, note

_MODE_CN = {"sts": "STS 临时凭证（≤12h 自灭）", "ram": "长期 AK（时间窗 policy + 到期硬删）"}
_STAGE_CN = {"NEW": "待发放", "ISSUED": "已发放", "REVOKED": "已吊销/到期清理", "FAILED": "失败"}
_PLATFORM_CN = {"aliyun": "阿里云 OSS", "volcano": "火山云 TOS"}


def receipt_card(grant: dict):
    """内部回执（脱敏，不含 secret/token）。"""
    from . import orchestrator as o
    color = "green" if grant.get("stage") == o.STAGE_ISSUED else "grey"
    return card("✅ 临时 AK 已发放（数据外采）", [
        fields(("任务ID", f"`{grant.get('grant_id')}`"),
               ("平台", _PLATFORM_CN.get(grant.get("platform"), grant.get("platform", ""))),
               ("模式", _MODE_CN.get(grant.get("mode"), grant.get("mode", "")))),
        div(f"**外采企业**：{grant.get('enterprise') or '-'}\n"
            f"**授权**：{o.scope_line(grant)}\n"
            f"**有效期**：{o.fmt_window(grant)}\n"
            + (f"**RAM 用户/AK**：`{grant.get('user_name')}` / `{grant.get('ak_id')}`\n"
               if grant.get("mode") == "ram" else "")
            + f"**申请人**：{grant.get('requester') or '-'}"),
        note("凭证已通过审批评论下发给申请人（只展示一次）；本卡不含密钥。"),
    ], color=color)


def status_card(grant: dict):
    from . import orchestrator as o
    stage = _STAGE_CN.get(grant.get("stage"), grant.get("stage", ""))
    return card(f"临时 AK 任务 · {stage}", [
        fields(("任务ID", f"`{grant.get('grant_id')}`"),
               ("平台", _PLATFORM_CN.get(grant.get("platform"), grant.get("platform", ""))),
               ("模式", _MODE_CN.get(grant.get("mode"), grant.get("mode", "")))),
        div(f"**外采企业**：{grant.get('enterprise') or '-'}\n"
            f"**授权**：{o.scope_line(grant)}\n"
            f"**有效期**：{o.fmt_window(grant)}\n"
            + (f"**失败原因**：{grant.get('error')}" if grant.get("error") else "")),
    ], color="grey")
