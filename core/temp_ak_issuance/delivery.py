"""凭证下发（方案 B：审批评论）。

用户拍板走**审批评论**下发：发放成功后，把含 AK/SK[/Token] 的凭证文本作为**评论**贴到该审批实例上
（复用 ram_approval._send_approval_comment），审批链参与者可见、申请人转交外采企业。
另给内部群一张**脱敏**回执卡（不含 secret）。
secret/token 只出现在审批评论正文里，绝不写日志、绝不落 Redis、绝不进群卡。
"""
from __future__ import annotations

from config.settings import settings
from utils.logger import get_logger

from . import orchestrator as o

logger = get_logger(__name__)


def deliver(grant: dict, creds: dict | None) -> None:
    """发放成功后下发：凭证走审批评论（发放审批实例）+ 内部群脱敏回执。best-effort。"""
    delivered = True
    if creds:
        try:
            _post_credential_comment(grant, creds, grant.get("instance_code"))
        except Exception:
            delivered = False
            logger.error("[temp_ak] 凭证评论下发失败 grant=%s", grant.get("grant_id"), exc_info=True)
    # 不再推内部回执卡（用户要求：信息只在审批评论里，不单独推群/推人）。
    if creds and not delivered and grant.get("mode") == "ram":
        _alert_creds_undelivered(grant)


def deliver_extend(grant: dict, creds: dict | None) -> None:
    """延期下发：评论贴到**延期审批实例**上。creds(STS 重签发)→新凭证评论；None(方案B 同 AK)→"已延长"通知(无 secret)。"""
    ic = (grant.get("extend_instances") or [grant.get("instance_code")])[-1]
    delivered = True
    try:
        if creds:
            _post_credential_comment(grant, creds, ic)
        else:
            _post_comment(grant, ic, _extended_text(grant))
    except Exception:
        delivered = False
        logger.error("[temp_ak] 延期评论下发失败 grant=%s", grant.get("grant_id"), exc_info=True)
    if creds and not delivered and grant.get("mode") == "ram":
        _alert_creds_undelivered(grant)


# ── 审批评论下发 ──────────────────────────────────────────────────────────────

def _comment_user_id(grant: dict) -> str:
    from core import ram_approval
    return grant.get("requester") or ram_approval._approval_comment_user_id()


def _post_comment(grant: dict, instance_code: str, text: str) -> None:
    from core import ram_approval
    if not instance_code:
        raise RuntimeError("缺审批实例 code，无法评论下发")
    ram_approval._send_approval_comment(instance_code, text, _comment_user_id(grant))


def _post_credential_comment(grant: dict, creds: dict, instance_code: str) -> None:
    """把含 secret 的凭证文本作为评论贴到审批实例（审批链可见，B 方案）。"""
    _post_comment(grant, instance_code, credential_text(grant, creds))
    logger.info("[temp_ak] 凭证已评论下发到审批实例 grant=%s", grant.get("grant_id"))


def _alert_creds_undelivered(grant: dict) -> None:
    chat = settings.TEMP_AK_CHAT_ID or settings.FEISHU_CHAT_ID
    if not chat:
        return
    try:
        from core.dsw_scheduler import _send_text
        _send_text("", chat,
                   f"⚠️ 临时 AK `{grant.get('grant_id')}`（方案B/长期AK，外采企业 "
                   f"{grant.get('enterprise') or '-'}）已在云上创建，但凭证评论下发失败——"
                   f"secret 不可恢复。请管理员用 `manage_temp_ak revoke` 或 CLI revoke 吊销后重新发起审批。")
    except Exception:
        logger.error("[temp_ak] creds-undelivered 告警发送失败 grant=%s", grant.get("grant_id"), exc_info=True)


# ── 文本 ──────────────────────────────────────────────────────────────────────

def credential_text(grant: dict, creds: dict) -> str:
    """凭证正文（含 secret/token，仅贴进审批评论，出现一次）。"""
    mode_cn = ("STS 临时凭证（含 SecurityToken，到点自动失效）" if creds.get("mode") == "sts"
               else "长期 AccessKey（权限内嵌生效/到期时间，到期后调用被拒并自动清理）")
    lines = [
        "数据外采访问凭证（请妥善保存并转交外采企业）",
        f"凭证ID：{grant.get('grant_id')}（延长/撤销时填此 ID）",
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
    if grant.get("source_ips"):
        lines.append(f"出口 IP 限制：仅 {', '.join(grant['source_ips'])} 可用")
    lines += [
        "",
        "· 仅在生效~到期区间内、且仅对上述桶/目录有效；超时或超范围调用一律被拒。",
        "· Secret 请立即保存后转交外采企业，切勿截图外传或提交代码仓库。",
    ]
    return "\n".join(lines)


def _extended_text(grant: dict) -> str:
    """延期通知正文（方案B 同 AK，无 secret）。"""
    return "\n".join([
        "访问凭证有效期已延长",
        f"凭证ID：{grant.get('grant_id')}",
        f"外采企业：{grant.get('enterprise') or '-'}",
        f"授权范围：{o.scope_line(grant)}",
        f"新有效期：{o.fmt_window(grant)}",
        "AccessKey 不变、无需更换；到期后自动失效。",
    ])
