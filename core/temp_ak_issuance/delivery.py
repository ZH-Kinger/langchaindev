"""凭证下发（方案 B：审批评论）。

用户拍板走**审批评论**下发：发放成功后，把含 AK/SK[/Token] 的凭证文本作为**评论**贴到该审批实例上
（复用 ram_approval._send_approval_comment），审批链参与者可见、申请人转交外采企业。
另给内部群一张**脱敏**回执卡（不含 secret）。
secret/token 只出现在审批评论正文里，绝不写日志、绝不落 Redis、绝不进群卡。
"""
from __future__ import annotations

import re

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
    """凭证评论身份**默认与 RAM 建号审批完全一致**——复用 ram_approval._approval_comment_user_id()：
    FEISHU_RAM_APPROVAL_COMMENT_USER_ID → ADMIN_FEISHU_OPEN_ID（以管理员身份发，不冒充申请人/审批人）。
    多级审批下不再用 requester（会解析成审批人、把凭证评论错挂其名下）。

    多账号：某账号档案配了 comment_user_id 才覆盖（如第二主账号想让别人来发凭证评论）。
    注意该 open_id **必须属于发评论用的那个飞书应用**，跨 app 会报 99992361。
    """
    from core import ram_approval
    from . import accounts
    try:
        profile = accounts.by_slug(grant.get("account", ""))
        if profile.comment_user_id:
            return profile.comment_user_id
    except Exception:
        pass          # 档案缺失不该挡下发；退回全局管理员身份
    return ram_approval._approval_comment_user_id()


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
    from . import accounts
    chat = accounts.chat_id_for(grant)      # 按该凭证所属账号取群
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

def _subject_label(grant: dict) -> str:
    """主体那一栏的叫法。单一来源在 accounts.subject_label_for，本函数只是本模块的短别名。"""
    from . import accounts
    return accounts.subject_label_for(grant)


def _access_lines(grant: dict) -> list[str]:
    """连接信息：地域 + 外网 Endpoint + 桶域名。

    **必须给**：桶在哪个地域就得用哪个地域的 endpoint，否则 OSS 直接回 403
    `The bucket you are attempting to access must be addressed using the specified endpoint`
    —— 使用方会以为凭证无效。region 取自 grant（由桶映射解析），缺失时不瞎猜、只提示自查。
    """
    region = (grant.get("region") or "").strip()
    bucket = (grant.get("bucket") or "").strip()
    if not region:
        return ["地域/Endpoint：未知（该桶未配地域映射，请按控制台上该桶的外网 Endpoint 连接）"]
    ep = f"oss-{region}.aliyuncs.com"
    lines = [f"地域：{region}", f"外网 Endpoint：{ep}"]
    if bucket:
        lines.append(f"桶域名：{bucket}.{ep}")
    return lines


def credential_text(grant: dict, creds: dict) -> str:
    """凭证正文（含 secret/token，仅贴进审批评论，出现一次）。"""
    mode_cn = ("STS 临时凭证（含 SecurityToken，到点自动失效）" if creds.get("mode") == "sts"
               else "长期 AccessKey（权限内嵌生效/到期时间，到期后调用被拒并自动清理）")
    label = _subject_label(grant)
    lines = [
        "数据访问凭证（请妥善保存并转交使用方）",
        f"凭证ID：{grant.get('grant_id')}（延长/撤销时填此 ID）",
        f"{label}：{grant.get('enterprise') or '-'}",
        f"授权范围：{o.scope_line(grant)}",
        *_access_lines(grant),
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
    if grant.get("note"):
        # 备注是申请人自由填的文本，会进凭证评论正文。**必须压成一行**：留着换行就能伪造
        # 「AccessKey Secret：…」这样的假行、或塞进误导性的操作指引，而正文是使用方唯一的凭证来源。
        note = re.sub(r"\s+", " ", str(grant["note"])).strip()
        if note:
            lines.append(f"备注：{note[:200]}")
    lines += [
        "",
        "· 仅在生效~到期区间内、且仅对上述桶/目录有效；超时或超范围调用一律被拒。",
        "· Secret 请立即保存后转交使用方，切勿截图外传或提交代码仓库。",
    ]
    return "\n".join(lines)


def _extended_text(grant: dict) -> str:
    """延期通知正文（方案B 同 AK，无 secret）。"""
    return "\n".join([
        "访问凭证有效期已延长",
        f"凭证ID：{grant.get('grant_id')}",
        f"{_subject_label(grant)}：{grant.get('enterprise') or '-'}",
        f"授权范围：{o.scope_line(grant)}",
        f"新有效期：{o.fmt_window(grant)}",
        "AccessKey 不变、无需更换；到期后自动失效。",
    ])
