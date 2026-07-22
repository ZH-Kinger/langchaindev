"""凭证下发。

真机审批「数据外采访问凭证申请」无邮箱字段 → 凭证**定向私信审批发起人**（内部申请人，一次性展示），
由其转交外采企业；同时给内部群一张**脱敏**回执卡（不含 secret）。
secret/token 只在发给发起人的那张凭证卡里出现一次，绝不写日志、绝不落 Redis、绝不进群卡。
"""
from __future__ import annotations

from config.settings import settings
from utils.logger import get_logger

from . import orchestrator as o

logger = get_logger(__name__)


def deliver(grant: dict, creds: dict | None) -> None:
    """发放成功后下发：私信发起人（凭证）+ 内部群回执（脱敏）。best-effort。"""
    delivered = True
    if creds:
        try:
            _send_creds_to_requester(grant, creds)
        except Exception:
            delivered = False
            logger.error("[temp_ak] 凭证私信发起人失败 grant=%s", grant.get("grant_id"), exc_info=True)
    try:
        _send_internal_receipt(grant)
    except Exception:
        logger.error("[temp_ak] 内部回执推送失败 grant=%s", grant.get("grant_id"), exc_info=True)
    # 方案 B：AK 已在云上创建、secret 只此一次可得，下发失败即不可恢复 → 醒目告警管理员 revoke+重审。
    if creds and not delivered and grant.get("mode") == "ram":
        _alert_creds_undelivered(grant)


def _send_creds_to_requester(grant: dict, creds: dict) -> None:
    """把含 AK/SK[/Token] 的凭证卡**只**私信发起人（open_id）。无发起人 open_id 时拒发（不进群，防泄漏）。

    凭证下发是关键路径，不能像共享 _send_card 那样静默失败 → 用带响应校验的发送，飞书返非 0 即抛，
    触发上层 _alert_creds_undelivered（否则"以为送达实则没送"，外采企业收不到、无人知晓）。
    """
    target = (grant.get("requester") or "").strip()
    if not target:
        raise RuntimeError("无审批发起人 open_id，凭证无法定向下发（不入群，防泄漏）")
    from . import cards
    _feishu_send_interactive_checked(target, cards.credential_card(grant, creds))
    logger.info("[temp_ak] 凭证已私信发起人 grant=%s", grant.get("grant_id"))


def _feishu_send_interactive_checked(open_id: str, card: dict) -> None:
    """私信一张交互卡并**校验飞书响应 code==0**；失败抛异常。异常信息只含 code/msg，绝不含卡片 body/secret。"""
    import json as _json
    import requests
    from tools.feishu.notify import _get_access_token
    token = _get_access_token()
    resp = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        params={"receive_id_type": "open_id"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"receive_id": open_id, "msg_type": "interactive", "content": _json.dumps(card)},
        timeout=15,
    )
    data = {}
    try:
        data = resp.json()
    except Exception:
        pass
    if resp.status_code != 200 or data.get("code") != 0:
        raise RuntimeError(f"飞书私信失败 status={resp.status_code} code={data.get('code')} msg={data.get('msg')}")


def _send_internal_receipt(grant: dict) -> None:
    """内部回执：脱敏卡（企业/平台/桶目录/权限/有效期/模式），推 TEMP_AK_CHAT_ID（回退 FEISHU_CHAT_ID）。"""
    chat = settings.TEMP_AK_CHAT_ID or settings.FEISHU_CHAT_ID
    if not chat:
        return
    from core.dsw_scheduler import _send_card
    from . import cards
    _send_card("", chat, cards.receipt_card(grant))


def _alert_creds_undelivered(grant: dict) -> None:
    chat = settings.TEMP_AK_CHAT_ID or settings.FEISHU_CHAT_ID
    if not chat:
        return
    try:
        from core.dsw_scheduler import _send_text
        _send_text("", chat,
                   f"⚠️ 临时 AK `{grant.get('grant_id')}`（方案B/长期AK，外采企业 "
                   f"{grant.get('enterprise') or '-'}）已在云上创建，但凭证私信发起人失败——"
                   f"secret 不可恢复。请管理员用 `manage_temp_ak revoke` 或 CLI revoke 吊销后重新发起审批。")
    except Exception:
        logger.error("[temp_ak] creds-undelivered 告警发送失败 grant=%s", grant.get("grant_id"), exc_info=True)


def credential_text(grant: dict, creds: dict) -> str:
    """凭证正文（含 secret/token，仅用于发给发起人的那张卡，出现一次）。"""
    mode_cn = ("STS 临时凭证（含 SecurityToken，到点自动失效）" if creds.get("mode") == "sts"
               else "长期 AccessKey（权限内嵌生效/到期时间，到期后调用被拒并自动清理）")
    lines = [
        f"外采企业：{grant.get('enterprise') or '-'}",
        f"授权范围：{o.scope_line(grant)}",
        f"有效期：{o.fmt_window(grant)}",
        f"凭证类型：{mode_cn}",
        "",
        f"AccessKey ID：{creds.get('access_key_id', '-') or '-'}",
        f"AccessKey Secret：{creds.get('access_key_secret', '-') or '-'}",
    ]
    if creds.get("security_token"):
        lines.append(f"SecurityToken：{creds['security_token']}")
    lines += [
        "",
        "· 仅在生效~到期区间内、且仅对上述桶/目录有效；超时或超范围调用一律被拒。",
        "· Secret 只展示一次，请立即妥善保存后转交外采企业，切勿截图外传或提交代码仓库。",
    ]
    return "\n".join(lines)
