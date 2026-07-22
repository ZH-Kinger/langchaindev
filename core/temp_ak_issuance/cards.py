"""飞书卡片：临时 AK 发放。

- credential_card：**含 secret/token**，仅私信发放给审批发起人（一次性展示）。
- receipt_card / status_card：**脱敏**，进内部群，绝不含 secret/token。
"""
from tools.feishu.cards import card, div, fields, note

_MODE_CN = {"sts": "STS 临时凭证（≤12h 自灭）", "ram": "长期 AK（时间窗 policy + 到期硬删）"}
_STAGE_CN = {"NEW": "待发放", "ISSUED": "已发放", "REVOKED": "已吊销/到期清理", "FAILED": "失败"}
_PLATFORM_CN = {"aliyun": "阿里云 OSS", "volcano": "火山云 TOS"}


def credential_card(grant: dict, creds: dict):
    """含密钥，只私信发起人。发起人转交外采企业。"""
    from . import delivery, orchestrator as o
    return card("🔑 数据外采访问凭证（请妥善保存并转交）", [
        div(f"**外采企业**：{grant.get('enterprise') or '-'}\n"
            f"**平台**：{_PLATFORM_CN.get(grant.get('platform'), grant.get('platform', ''))}\n"
            f"**授权**：{o.scope_line(grant)}\n"
            f"**有效期**：{o.fmt_window(grant)}"),
        div("```\n" + delivery.credential_text(grant, creds) + "\n```"),
        note("AccessKey Secret 只展示一次；请立即保存后转交外采企业，勿截图外传。"),
    ], color="red")


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
        note("凭证已私信发放给申请人（只展示一次）；本卡不含密钥。"),
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
