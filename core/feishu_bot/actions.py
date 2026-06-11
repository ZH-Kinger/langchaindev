"""卡片按钮动作处理（card.action.trigger / card_action 共用）。

_ACTION_HANDLERS 注册表：action 名 → handler(action_val, open_id, chat_id, form_value)。
同步路径须 <3s 内返回（飞书回调超时限制）：耗时操作（Jira / DSW API）一律放后台线程，
同步只做状态更新和 UI 反馈；不在模块顶层引入重依赖（build_mfu_card 等保持 lazy）。
"""
import json
import threading

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)
from tools.feishu.cards import card, div, note
from tools.feishu.notify import _get_access_token
from tools.jira.ticket import create_gpu_ticket, add_comment as jira_comment
from core.dsw_scheduler import (_redis_get, _redis_set, _redis_delete,
                                check_quota, _cost_str, _set_approved)
from . import gpu_flow, messaging


# ── MFU 日报区域切换：返回新卡片原地替换（只读 Redis 缓存，秒回；绝不同步采集）──

def _h_mfu_region(action_val, open_id, chat_id, form_value):
    from tools.aliyun.cluster_mfu import mfu_card_for_callback
    region = action_val.get("region", "") if isinstance(action_val, dict) else ""
    new_card = mfu_card_for_callback(view=region or "summary")
    if new_card is None:   # 缓存全无：后台已开始采集，不替换原卡片
        return {"toast": {"type": "info", "content": "📊 数据采集中（约1分钟），请稍后再点"}}
    # schema 2.0 卡片回调：用 {"card":{"type":"raw","data":...}} 原地更新
    return {"toast": {"type": "info", "content": "已切换"},
            "card": {"type": "raw", "data": new_card}}


# ── AK/SK 表单卡片提交（Fernet 加密存 Redis，资源归属为用户本人）─────────────

def _h_submit_ak_register(action_val, open_id, chat_id, form_value):
    if not form_value:
        return {}
    ak_id     = (form_value.get("ak_id") or "").strip()
    ak_secret = (form_value.get("ak_secret") or "").strip()
    # 可选：用户也可以同时填 ram_user_name 显式指定 RAM 用户（不填则按飞书姓名自动匹配）
    ram_user_name = (form_value.get("ram_user_name") or form_value.get("user_name") or "").strip()

    if not ak_id or not ak_secret:
        return {"toast": {"type": "error", "content": "请填写 AccessKey ID 和 Secret"}}
    if not ak_id.startswith(("LTAI", "ltai")):
        return {"toast": {"type": "error", "content": "AccessKey ID 格式不正确（应以 LTAI 开头）"}}

    # 加密存储 AK/SK
    from utils.aliyun_user_creds import save_user_ak
    if not save_user_ak(open_id, ak_id, ak_secret):
        return {"toast": {"type": "error", "content": "保存失败（Redis 不可达？）"}}

    # 同步建立 RAM 映射：优先用用户填的 ram_user_name，否则尝试自动匹配
    display = ram_user_name or ""
    try:
        from tools.aliyun.ram import list_ram_users_api, save_user_map
        ram_users = list_ram_users_api()
        if ram_user_name:
            matched = next((u for u in ram_users if u["user_name"] == ram_user_name), None)
        else:
            # 按飞书显示名匹配
            from tools.feishu.notify import _get_user_name
            feishu_name = _get_user_name(open_id)
            matched = next((u for u in ram_users
                            if u.get("display_name") == feishu_name), None) if feishu_name else None
        if matched:
            display = matched.get("display_name") or matched["user_name"]
            save_user_map(open_id, display, matched["user_id"])
    except Exception as e:
        logger.warning("[RAM映射] 同步失败（不影响 AK 绑定）: %s", e)

    # 绑定完后如有待处理的 GPU 申请，推送 GPU 卡片
    gpu_state = gpu_flow._get_gpu_state(chat_id, open_id)
    if gpu_state and gpu_state.get("pending_gpu"):
        gpu_flow._clear_gpu_state(chat_id, open_id)
        def _push_gpu_card():
            import time; time.sleep(1)
            messaging._feishu_send(chat_id, "✅ AK 已绑定！正在为你打开 GPU 申请表单...")
            try:
                template_id = settings.FEISHU_GPU_CARD_TEMPLATE_ID
                content = json.dumps({"type": "template", "data": {"template_id": template_id}}) if template_id else json.dumps(gpu_flow._GPU_REQUEST_CARD)
                token = _get_access_token()
                requests.post(
                    "https://open.feishu.cn/open-apis/im/v1/messages",
                    params={"receive_id_type": "chat_id"},
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"receive_id": chat_id, "msg_type": "interactive", "content": content},
                    timeout=15,
                )
            except Exception:
                logger.error("[GPU卡片推送] 失败", exc_info=True)
        threading.Thread(target=_push_gpu_card, daemon=True).start()

    return {
        "toast": {"type": "success", "content": "✅ AK 已加密保存"},
        "card": card("✅ AccessKey 已绑定", [div(
            f"**AccessKey ID：** `{ak_id[:8]}****`（已 Fernet 加密存入 Redis）\n"
            + (f"**RAM 用户：** {display}\n" if display else "")
            + f"**资源归属：** 你的 RAM 用户本人（阿里云控制台可直接看到）\n"
            + f"**自动失效：** {settings.USER_AK_IDLE_TTL_SECONDS // 86400} 天未使用\n\n"
            + "随时发送「解绑AK」可删除，「查看绑定」查状态。")], color="green"),
    }


# ── 表单提交（submit_form behavior）──────────────────────────────────────────

def _h_submit_gpu_request(action_val, open_id, chat_id, form_value):
    if not form_value:
        return {}
    instance_name  = (form_value.get("instance_name") or "").strip()
    gpu_count      = form_value.get("gpu_count", "1")
    duration_hours = form_value.get("duration_hours", "8")
    purpose        = (form_value.get("purpose") or "").strip()
    cpu_cores      = (form_value.get("cpu_cores") or "").strip()
    memory_gb      = (form_value.get("memory_gb") or "").strip()
    priority       = (form_value.get("priority") or "中").strip()
    ssh_public_key = (form_value.get("ssh_public_key") or "").strip()
    image_name     = (form_value.get("image_name") or "").strip()

    if not instance_name:
        return {"toast": {"type": "error", "content": "请填写实例名称"}}
    if not purpose:
        return {"toast": {"type": "error", "content": "请填写申请用途"}}

    # 确保实例名带 wuji- 前缀
    if not instance_name.startswith("wuji-"):
        instance_name = f"wuji-{instance_name}"

    gpu_count_i    = int(gpu_count)   if str(gpu_count).isdigit() else 1
    duration_float = float(duration_hours) if str(duration_hours).replace(".", "").isdigit() else 8.0

    # 配额校验
    within_quota, used, limit = check_quota(open_id, gpu_count_i, duration_float)
    if not within_quota:
        return {"toast": {"type": "error", "content":
            f"月度配额不足（已用 {used:.0f}h / 上限 {limit:.0f}GPU·h），请联系管理员"}}

    # 超大申请（>4 GPU 或 >48h）需管理员审批
    needs_approval = gpu_count_i > 4 or duration_float > 48
    cost_hint      = _cost_str(gpu_count_i, duration_float)

    # 获取飞书真实姓名，用于工单归属
    from tools.feishu.notify import _get_user_name
    reporter_name = _get_user_name(open_id)

    def _do() -> None:
        ticket_key = create_gpu_ticket(
            instance_name=instance_name,
            gpu_count=gpu_count,
            duration_hours=duration_hours,
            purpose=purpose,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            priority=priority,
            ssh_public_key=ssh_public_key,
            image_name=image_name,
            reporter_open_id=open_id,
            reporter_name=reporter_name,
            chat_id=chat_id,
            needs_approval=needs_approval,
        )
        if ticket_key:
            if needs_approval:
                messaging._send_text_to(open_id, chat_id,
                    f"⏳ 工单 {ticket_key} 规格较大（{gpu_count}GPU · {duration_hours}h），"
                    f"已提交管理员审批，审批通过后自动创建实例。\n费用预估：{cost_hint}")
            else:
                messaging._send_text_to(open_id, chat_id,
                    f"✅ 申请已提交！工单：{ticket_key}\n"
                    f"实例 {instance_name}（{gpu_count}GPU · {duration_hours}h）将在 2 分钟内创建，完成后飞书推送。\n"
                    f"费用预估：{cost_hint}")
        else:
            messaging._send_text_to(open_id, chat_id, "❌ Jira 工单创建失败，请联系运维人员。")

    threading.Thread(target=_do, daemon=True).start()
    return {
        "toast": {"type": "info", "content": "申请已提交，处理中..."},
        "card": card("⏳ 申请处理中", [div(
            f"**实例：** {instance_name}　**GPU：** {gpu_count} 卡　**时长：** {duration_hours}h\n"
            f"**CPU：** {cpu_cores or '默认'}　**内存：** {memory_gb or '默认'} GB　**优先级：** {priority}\n\n"
            "正在创建 Jira 工单，完成后飞书通知。")], color="yellow"),
    }


# ── 快选配置（1/4/8 GPU）─────────────────────────────────────────────────────

def _h_quick_gpu(action_val, open_id, chat_id, form_value):
    gpu_count      = action_val.get("gpu_count", "1")
    duration_hours = action_val.get("duration_hours", "8")
    gpu_flow._set_gpu_state(chat_id, open_id, {"gpu_count": gpu_count, "duration_hours": duration_hours})
    return {
        "toast": {"type": "info", "content": f"已选 {gpu_count}GPU · {duration_hours}h，请回复实例名和用途"},
        "card": card("GPU 资源申请", [
            div(f"已选：**{gpu_count} GPU · {duration_hours} 小时**\n\n"
                "请直接回复此消息，补充以下信息：\n"
                "```\n实例名: wzh-train-01\n用途: 大语言模型微调\n```"),
            note("回复后系统自动创建 DSW 实例"),
        ]),
    }


# ── 延续使用 ──────────────────────────────────────────────────────────────────

def _h_extend_dsw(action_val, open_id, chat_id, form_value):
    ticket_key   = action_val.get("ticket_key", "")
    extend_hours = float(action_val.get("extend_hours", "2"))
    state = _redis_get(ticket_key) if ticket_key else None
    if not state:
        return {"toast": {"type": "error", "content": "实例已停止或不存在，无法延续"}}
    state["duration_hours"] = state.get("duration_hours", 8) + extend_hours
    state["warned"]         = False
    state["warn_ts"]        = 0.0
    _redis_set(ticket_key, state)
    threading.Thread(
        target=jira_comment,
        args=(ticket_key, f"用户选择延续使用 {int(extend_hours)} 小时。"),
        daemon=True,
    ).start()
    return {"toast": {"type": "success", "content": f"已延续使用 {int(extend_hours)} 小时"}}


# ── 立即停止 ──────────────────────────────────────────────────────────────────

def _h_stop_dsw(action_val, open_id, chat_id, form_value):
    from tools.aliyun.pai_dsw import manage_pai_dsw
    from tools.jira.ticket import transition_ticket
    ticket_key  = action_val.get("ticket_key", "")
    instance_id = action_val.get("instance_id", "")

    # 幂等保护：state 已删说明已停止过，直接告知
    if ticket_key and not _redis_get(ticket_key):
        return {"toast": {"type": "info", "content": "实例已停止，无需重复操作"}}

    def _do_stop() -> None:
        if instance_id:
            manage_pai_dsw(action="stop", instance_id=instance_id)
        if ticket_key:
            transition_ticket(ticket_key, "完成")
            jira_comment(ticket_key, "用户手动停止实例。")
            _redis_delete(ticket_key)

    threading.Thread(target=_do_stop, daemon=True).start()
    return {
        "toast": {"type": "info", "content": "实例停止中，工单已关闭"},
        "card": card("🛑 实例已停止", [div(
            f"实例 `{instance_id or ticket_key}` 已手动停止，工单已关闭。\n按钮已失效，如需重新使用请提交新工单。")],
            color="red"),
    }


# ── 审批通过 ──────────────────────────────────────────────────────────────────

def _h_approve_gpu(action_val, open_id, chat_id, form_value):
    ticket_key         = action_val.get("ticket_key", "")
    req_open_id        = action_val.get("requester_open_id", "")
    req_chat_id        = action_val.get("requester_chat_id", "")
    if not ticket_key:
        return {"toast": {"type": "error", "content": "缺少工单号"}}
    _set_approved(ticket_key)
    jira_comment(ticket_key, "✅ 管理员已批准，调度器将在 20 秒内自动创建实例。")
    messaging._send_text_to(req_open_id, req_chat_id,
        f"✅ 工单 {ticket_key} 已获批准！调度器将在 20 秒内自动创建 GPU 实例。")
    return {
        "toast": {"type": "success", "content": f"工单 {ticket_key} 已批准"},
        "card": card("✅ 已批准", [div(
            f"工单 **{ticket_key}** 已批准，调度器将在 20 秒内自动创建实例。")], color="green"),
    }


# ── 审批拒绝 ──────────────────────────────────────────────────────────────────

def _h_reject_gpu(action_val, open_id, chat_id, form_value):
    from tools.jira.ticket import transition_ticket
    ticket_key  = action_val.get("ticket_key", "")
    req_open_id = action_val.get("requester_open_id", "")
    req_chat_id = action_val.get("requester_chat_id", "")
    if not ticket_key:
        return {"toast": {"type": "error", "content": "缺少工单号"}}
    jira_comment(ticket_key, "❌ 管理员已拒绝此申请。")
    transition_ticket(ticket_key, "完成")
    messaging._send_text_to(req_open_id, req_chat_id,
        f"❌ 工单 {ticket_key} 的 GPU 申请已被管理员拒绝，如有疑问请联系运维团队。")
    return {
        "toast": {"type": "info", "content": f"工单 {ticket_key} 已拒绝"},
        "card": card("❌ 已拒绝", [div(
            f"工单 **{ticket_key}** 已拒绝，Jira 状态已更新。")], color="red"),
    }


# ── OSS 权限同步：管理员批准 → 后台下发 → 回结果卡片 ─────────────────────────

def _h_approve_oss_perm(action_val, open_id, chat_id, form_value):
    from config.settings import settings as _cfg
    if open_id != _cfg.ADMIN_FEISHU_OPEN_ID:
        return {"toast": {"type": "error", "content": "仅管理员可批准下发"}}

    def _do_oss_perm_apply() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        try:
            from core.oss_perm import permsync
            from core.oss_perm.cards import result_card
            members, combos = permsync.load_members()
            plan = permsync.build_plan(members, combos)
            summary = permsync.apply_all(plan)
            _send_card("", _cfg.FEISHU_CHAT_ID, result_card(summary))
            logger.info("[OSSPerm] 下发完成：成功 %d 失败 %d 无用户 %d",
                        summary["ok"], summary["fail"], summary["no_user"])
        except Exception as e:
            logger.error("[OSSPerm] 下发失败", exc_info=True)
            _send_text("", _cfg.FEISHU_CHAT_ID, f"❌ OSS 权限下发失败：{e}")

    threading.Thread(target=_do_oss_perm_apply, daemon=True).start()
    return {"toast": {"type": "success", "content": "已开始下发，完成后推送结果到群"}}


_ACTION_HANDLERS = {
    "mfu_region":         _h_mfu_region,
    "submit_ak_register": _h_submit_ak_register,
    "submit_gpu_request": _h_submit_gpu_request,
    "quick_gpu":          _h_quick_gpu,
    "extend_dsw":         _h_extend_dsw,
    "stop_dsw":           _h_stop_dsw,
    "approve_gpu":        _h_approve_gpu,
    "reject_gpu":         _h_reject_gpu,
    "approve_oss_perm":   _h_approve_oss_perm,
}


def _process_action(action_name: str, action_val: dict, open_id: str, chat_id: str,
                    form_value: dict | None = None) -> dict:
    """所有卡片动作的共享处理逻辑，返回飞书期待的响应 dict。未知 action 返回 {}。"""
    handler = _ACTION_HANDLERS.get(action_name)
    if handler is None:
        return {}
    return handler(action_val, open_id, chat_id, form_value)


def _handle_card_trigger_sync(data: dict) -> dict:
    """处理经事件订阅路径到达的 card.action.trigger（schema 2.0 结构）。"""
    event      = data.get("event", {})
    action_obj = event.get("action", {})
    action_val = action_obj.get("value") or {}
    form_value = action_obj.get("form_value") or {}
    open_id    = event.get("operator", {}).get("operator_id", {}).get("open_id", "")
    chat_id    = event.get("context", {}).get("open_chat_id", "") or settings.FEISHU_CHAT_ID
    action_name = action_val.get("action", "") if isinstance(action_val, dict) else ""
    if not action_name and form_value:
        # 兼容旧 AK 表单（ak_id/ak_secret）和新 RAM 绑定表单（ram_user_name/user_name）
        if any(k in form_value for k in ("ak_id", "ak_secret", "ram_user_name")):
            action_name = "submit_ak_register"
        else:
            action_name = "submit_gpu_request"
    return _process_action(action_name, action_val, open_id, chat_id, form_value=form_value)
