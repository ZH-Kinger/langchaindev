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

    # 建立 RAM 映射：list_ram_users_api + 取飞书姓名都是网络调用（阿里 RAM 分页 + 飞书 HTTP），
    # 放后台线程，避免占用 3s 回调预算导致飞书重投、成功卡发不出。关联结果异步补推一条文本。
    def _link_ram() -> None:
        try:
            from tools.aliyun.ram import list_ram_users_api, save_user_map
            ram_users = list_ram_users_api()
            if ram_user_name:
                matched = next((u for u in ram_users if u["user_name"] == ram_user_name), None)
            else:
                from tools.feishu.notify import _get_user_name
                feishu_name = _get_user_name(open_id)
                matched = next((u for u in ram_users
                                if u.get("display_name") == feishu_name), None) if feishu_name else None
            if matched:
                name = matched.get("display_name") or matched["user_name"]
                save_user_map(open_id, name, matched["user_id"])
                messaging._feishu_send(chat_id, f"✅ 已关联 RAM 用户：{name}（资源归属你本人）")
        except Exception as e:
            logger.warning("[RAM映射] 同步失败（不影响 AK 绑定）: %s", e)
    threading.Thread(target=_link_ram, daemon=True).start()

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
            + (f"**指定 RAM 用户：** {ram_user_name}（后台关联中…）\n" if ram_user_name
               else "**RAM 用户：** 后台自动关联中…\n")
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

    def _do() -> None:
        # 获取飞书真实姓名（飞书 HTTP）放后台线程，避免占用 3s 回调预算导致重投。
        from tools.feishu.notify import _get_user_name
        reporter_name = _get_user_name(open_id)
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

    level = action_val.get("level", "dir") if isinstance(action_val, dict) else "dir"
    if level not in ("bucket", "dir"):
        level = "dir"
    level_cn = "桶级" if level == "bucket" else "目录级"

    def _do_oss_perm_apply() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        try:
            from core.oss_perm import permsync
            from core.oss_perm.cards import result_card
            members, combos = permsync.load_members()
            plan = permsync.build_plan(members, combos, level=level)
            summary = permsync.apply_all(plan)
            _send_card("", _cfg.FEISHU_CHAT_ID, result_card(summary))
            logger.info("[OSSPerm] 下发完成(%s)：成功 %d 失败 %d 无用户 %d",
                        level, summary["ok"], summary["fail"], summary["no_user"])
        except Exception as e:
            logger.error("[OSSPerm] 下发失败", exc_info=True)
            _send_text("", _cfg.FEISHU_CHAT_ID, f"❌ OSS 权限下发失败：{e}")

    threading.Thread(target=_do_oss_perm_apply, daemon=True).start()
    return {"toast": {"type": "success", "content": f"已开始下发（{level_cn}），完成后推送结果到群"}}


# ── OSS 权限选择性下发：表单卡（粒度单选 + 成员多选）提交 → 仅下发勾选的人 ──────

def _h_approve_oss_perm_selected(action_val, open_id, chat_id, form_value):
    from config.settings import settings as _cfg
    if open_id != _cfg.ADMIN_FEISHU_OPEN_ID:
        return {"toast": {"type": "error", "content": "仅管理员可批准下发"}}

    fv = form_value or {}
    level = fv.get("level") or "dir"
    if level not in ("bucket", "dir"):
        level = "dir"
    selected = fv.get("selected") or []
    if isinstance(selected, str):       # 单选兜底（理论上 multi_select 给 list）
        selected = [selected]
    if not selected:
        return {"toast": {"type": "error", "content": "未选择任何成员，已取消"}}
    sel = set(selected)
    level_cn = "桶级" if level == "bucket" else "目录级"

    def _do_apply() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        try:
            from core.oss_perm import permsync
            from core.oss_perm.cards import result_card
            members, combos = permsync.load_members()
            plan = permsync.build_plan(members, combos, level=level)
            plan = [p for p in plan if p["member"]["username"] in sel]
            if not plan:
                _send_text("", _cfg.FEISHU_CHAT_ID, "⚠️ 勾选的成员均无可解析的下发计划，未执行。")
                return
            summary = permsync.apply_all(plan)
            _send_card("", _cfg.FEISHU_CHAT_ID, result_card(summary))
            logger.info("[OSSPerm] 选择性下发(%s) %d 人：成功 %d 失败 %d 无用户 %d",
                        level, len(plan), summary["ok"], summary["fail"], summary["no_user"])
        except Exception as e:
            logger.error("[OSSPerm] 选择性下发失败", exc_info=True)
            _send_text("", _cfg.FEISHU_CHAT_ID, f"❌ OSS 权限下发失败：{e}")

    threading.Thread(target=_do_apply, daemon=True).start()
    return {"toast": {"type": "success",
                      "content": f"已开始下发（{level_cn} · {len(sel)} 人），完成后推送结果到群"}}


# RAM account query: open form / submit form

def _h_open_ram_query(action_val, open_id, chat_id, form_value):
    from core.ram_query_cards import query_entry_card
    return {
        "toast": {"type": "info", "content": "\u8bf7\u5148\u586b\u5199\u8d26\u53f7\u57fa\u672c\u4fe1\u606f"},
        "card": {"type": "raw", "data": query_entry_card()},
    }


def _h_submit_ram_query(action_val, open_id, chat_id, form_value):
    fv = form_value or {}
    login_name = (fv.get("login_name") or fv.get("user_name") or fv.get("q") or "").strip()
    requested = {
        "login_name": login_name,
        "display_name": (fv.get("display_name") or "").strip(),
        "email": (fv.get("email") or "").strip(),
        "mobile": (fv.get("mobile") or "").strip(),
        "reason": (fv.get("reason") or "").strip(),
    }
    if not login_name:
        return {"toast": {"type": "error", "content": "\u8bf7\u586b\u5199\u767b\u5f55\u540d\u79f0"}}

    from core.ram_query import RamQueryError, RamUserNotFound, query_ram_account
    from core.ram_query_cards import query_error_card, query_result_card
    try:
        user = query_ram_account(login_name)
        return {
            "toast": {"type": "success", "content": "\u67e5\u8be2\u5b8c\u6210"},
            "card": {"type": "raw", "data": query_result_card(user, requested=requested)},
        }
    except RamUserNotFound:
        return {
            "toast": {"type": "info", "content": "\u672a\u627e\u5230\u8d26\u53f7"},
            "card": {"type": "raw", "data": query_result_card(None, requested=requested)},
        }
    except RamQueryError as exc:
        return {
            "toast": {"type": "error", "content": "\u67e5\u8be2\u5931\u8d25"},
            "card": {"type": "raw", "data": query_error_card(login_name, str(exc))},
        }


# \u706b\u5c71\u5f15\u64ce IAM \u8d26\u6237\u67e5\u8be2\uff1a\u5f55\u5165 / \u63d0\u4ea4

def _h_open_volcano_query(action_val, open_id, chat_id, form_value):
    from core.volcano_iam_query_cards import query_entry_card
    return {"toast": {"type": "info", "content": "\u8bf7\u586b\u5199\u7528\u6237\u540d"},
            "card": {"type": "raw", "data": query_entry_card()}}


def _h_submit_volcano_query(action_val, open_id, chat_id, form_value):
    fv = form_value or {}
    user_name = (fv.get("user_name") or fv.get("login_name") or fv.get("q") or "").strip()
    requested = {"user_name": user_name, "display_name": fv.get("display_name", ""),
                 "email": fv.get("email", ""), "mobile": fv.get("mobile", "")}
    from core.volcano_iam_query import (query_volcano_iam_account, VolcanoQueryError,
                                        VolcanoUserNotFound)
    from core.volcano_iam_query_cards import query_result_card, query_error_card
    if not user_name:
        return {"toast": {"type": "error", "content": "\u8bf7\u586b\u5199\u7528\u6237\u540d"}}
    try:
        user = query_volcano_iam_account(user_name)
        return {"toast": {"type": "success", "content": "\u67e5\u8be2\u5b8c\u6210"},
                "card": {"type": "raw", "data": query_result_card(user, requested=requested)}}
    except VolcanoUserNotFound:
        return {"toast": {"type": "info", "content": "\u672a\u627e\u5230\u7528\u6237"},
                "card": {"type": "raw", "data": query_result_card(None, requested=requested)}}
    except VolcanoQueryError as exc:
        return {"toast": {"type": "error", "content": "\u67e5\u8be2\u5931\u8d25"},
                "card": {"type": "raw", "data": query_error_card(user_name, str(exc))}}
    except Exception as exc:
        logger.error("[VolcanoQuery] failed", exc_info=True)
        return {"toast": {"type": "error", "content": "\u67e5\u8be2\u5931\u8d25"},
                "card": {"type": "raw", "data": query_error_card(user_name, str(exc))}}


# Transfer: entry / confirm / retry

def _h_submit_transfer(action_val, open_id, chat_id, form_value):
    """Read source/dest/policy, then parse, estimate, and push confirm card."""
    fv = form_value or {}
    source = (fv.get("source") or "").strip()
    dest = (fv.get("dest") or "").strip()
    same_name_policy = fv.get("same_name_policy") or fv.get("overwrite_policy") or "skip"
    if not source:
        return {"toast": {"type": "error", "content": "\u8bf7\u586b\u5199\u6e90\u5730\u5740"}}

    def _do_prepare() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        from core.transfer import orchestrator
        from core.transfer.cards import confirm_card, progress_card, result_card
        from core.transfer.paths import PathError
        try:
            plan = orchestrator.make_plan(source, dest)
            if plan.engine not in ("mgw", "tos_mig"):
                _send_text(open_id, _cfg_chat(),
                           f"\u26a0\ufe0f \u65b9\u5411 {plan.direction} \u6682\u672a\u652f\u6301\uff08\u4e09\u671f\u6c89\u964d\u6bb5\uff09\u3002")
                return
            job_id = orchestrator._job_id(plan)
            # \u2460 \u8be5\u6e90/\u76ee\u7684\u5f53\u5929\u5df2\u6709\u4efb\u52a1\u4e14\u5df2\u5728\u8dd1/\u5df2\u5b8c\u6210 \u2192 \u76f4\u63a5\u56de\u5b83\u7684\u8fdb\u5ea6/\u7ed3\u679c\u5361\uff0c\u7edd\u4e0d\u518d\u5f39\u201c\u786e\u8ba4\u201d\u5361\u3002
            #    \uff08\u5426\u5219\u7528\u6237\u5bf9\u540c\u4e00\u8def\u5f84\u53cd\u590d\u70b9\u201c\u89e3\u6790\u5e76\u9884\u4f30\u201d\u4f1a\u4e00\u76f4\u6536\u5230\u786e\u8ba4\u63d0\u793a\uff0c\u5373\u4f7f\u4efb\u52a1\u65e9\u5df2\u542f\u52a8\u3002\uff09
            existing = orchestrator.get_job(job_id)
            if existing and existing.get("stage") in (
                    orchestrator.STAGE_CROSSING, orchestrator.STAGE_SINKING, orchestrator.STAGE_DONE):
                live = orchestrator.refresh(job_id) or existing   # \u987a\u4fbf\u5b9e\u65f6\u91cd\u67e5\u4e00\u6b21
                c = (result_card(live) if live["stage"] == orchestrator.STAGE_DONE
                     else progress_card(live))
                _send_card(open_id, _cfg_chat(), c)
                return
            # \u2461 \u4ecd\u8fde\u70b9\u201c\u89e3\u6790\u5e76\u9884\u4f30\u201d\uff08\u4efb\u52a1\u672a\u542f\u52a8\u524d\uff09\uff1a\u6309 job_id \u539f\u5b50\u53bb\u91cd\uff0c30s \u5185\u53ea\u89e3\u6790\u3001\u53ea\u63a8\u4e00\u5f20\u786e\u8ba4\u5361\u3002
            #    \u6162\u9884\u4f30\u8981\u5217\u5927\u76ee\u5f55 10 \u4e07+\u5bf9\u8c61\uff0c\u91cd\u590d\u8dd1\u65e2\u6162\u53c8\u5237\u5c4f\u3002
            try:
                from utils.redis_client import get_redis
                if not get_redis().set(f"transfer:confirmcard:{job_id}", 1, nx=True, ex=30):
                    return
            except Exception:
                pass
            bytes_total, objects_total = orchestrator.estimate_source(plan)
            job = orchestrator.create_job_record(
                plan, open_id=open_id, same_name_policy=same_name_policy,
                bytes_total=bytes_total, objects_total=objects_total)
            need_appr = orchestrator.needs_approval(bytes_total)
            _send_card(open_id, _cfg_chat(), confirm_card(job, need_approval=need_appr))
        except PathError as e:
            _send_text(open_id, _cfg_chat(), f"\u274c \u8def\u5f84\u9519\u8bef\uff1a{e}")
        except Exception as e:
            logger.error("[Transfer] prepare failed", exc_info=True)
            _send_text(open_id, _cfg_chat(), f"\u274c \u8fc1\u79fb\u8bf7\u6c42\u5904\u7406\u5931\u8d25\uff1a{e}")

    threading.Thread(target=_do_prepare, daemon=True).start()
    return {"toast": {"type": "success",
                      "content": "\u6b63\u5728\u540e\u53f0\u89e3\u6790+\u9884\u4f30\uff08\u5927\u76ee\u5f55\u8f83\u6162\uff0c\u7ea6 10-30 \u79d2\uff09\uff0c\u7a0d\u5019\u63a8\u9001\u786e\u8ba4\u5361\uff0c\u8bf7\u52ff\u91cd\u590d\u70b9\u51fb"}}


def _h_confirm_transfer(action_val, open_id, chat_id, form_value, *, reply_v2=True):
    """Start the transfer in background using the selected same-name policy.

    reply_v2: 原地替换用哪张进度卡。确认卡是 schema 2.0 → 回 2.0 展示卡（progress_card_v2，
    同家族替换，避开 200830）。retry 由 1.0 结果卡触发 → 回 1.0 progress_card（1.0→1.0）。
    """
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    if not job_id:
        return {"toast": {"type": "error", "content": "\u7f3a\u5c11\u4efb\u52a1 ID"}}

    from core.transfer import orchestrator
    job = orchestrator.get_job(job_id)
    if not job:
        return {"toast": {"type": "error", "content": "\u4efb\u52a1\u4e0d\u5b58\u5728\u6216\u5df2\u8fc7\u671f"}}
    if job["stage"] not in (orchestrator.STAGE_NEW, orchestrator.STAGE_FAILED):
        return {"toast": {"type": "info", "content": f"\u4efb\u52a1\u5df2\u5728 {job['stage']}\uff0c\u65e0\u9700\u91cd\u590d"}}
    if job.get("launched"):
        return {"toast": {"type": "info", "content": "\u4efb\u52a1\u5df2\u4e0b\u53d1\uff0c\u8bf7\u52ff\u91cd\u590d\u70b9\u51fb"}}

    fv = form_value or {}
    same_name_policy = (
        fv.get("same_name_policy")
        or (action_val.get("same_name_policy") if isinstance(action_val, dict) else "")
        or job.get("same_name_policy")
        or job.get("overwrite_mode")
    )
    job = orchestrator.set_same_name_policy(job, same_name_policy)

    if orchestrator.needs_approval(job.get("bytes_total", 0)) and open_id != settings.ADMIN_FEISHU_OPEN_ID:
        return {"toast": {"type": "error", "content": "\u8d85\u8fc7\u5ba1\u6279\u9608\u503c\uff0c\u4ec5\u7ba1\u7406\u5458\u53ef\u786e\u8ba4\u4e0b\u53d1"}}

    # \u8fde\u70b9\u201c\u786e\u8ba4\u8fc1\u79fb\u201d\uff08\u4e0d\u540c\u786e\u8ba4\u5361 \u2192 \u4e0d\u540c msg_id\uff0c\u5185\u5bb9\u53bb\u91cd\u6321\u4e0d\u4f4f\uff09\u4f1a\u5404\u8d77\u4e00\u4e2a\u8f6e\u8be2\u7ebf\u7a0b\u3001\u5404\u5237\u8fdb\u5ea6/\u7ed3\u679c\u5361\u3002
    # Redis \u539f\u5b50\u9501\uff08\u5feb\u8def\u5f84\uff09+ \u6301\u4e45 launched \u6807\u8bb0\uff08\u8de8\u9501\u8fc7\u671f\u515c\u5e95\uff09\uff1a\u53ea\u6709\u9996\u6b21\u771f\u6b63\u4e0b\u53d1\uff0c\u5176\u4f59\u56de\u201c\u5df2\u4e0b\u53d1\u201d\u3002
    # \u5e95\u5c42 MGW \u5bf9\u540c\u540d\u4efb\u52a1\u5e42\u7b49\uff0c\u6545\u5373\u4fbf\u6f0f\u7f51\u4e5f\u53ea\u6709\u4e00\u4e2a\u771f\u5b9e\u8fc1\u79fb\uff0c\u8fd9\u91cc\u4e3b\u8981\u6b62\u4f4f\u7ebf\u7a0b/\u5237\u5361\u98ce\u66b4\u3002
    try:
        from utils.redis_client import get_redis
        if not get_redis().set(f"transfer:launch:{job_id}", 1, nx=True, ex=120):
            return {"toast": {"type": "info", "content": "\u4efb\u52a1\u5df2\u4e0b\u53d1\uff0c\u8bf7\u52ff\u91cd\u590d\u70b9\u51fb"}}
    except Exception:
        pass
    job["launched"] = True
    orchestrator._save(job)

    def _do_transfer() -> None:
        from core.dsw_scheduler import _send_card, _send_text, _claim_dataflow_notify
        from core.transfer.cards import result_card
        try:
            # \u786e\u8ba4\u5361\u5df2\u5728\u4e0b\u65b9\u539f\u5730\u66ff\u6362\u4e3a\u201c\u8fdb\u884c\u4e2d\u201d\u5361\uff0c\u4e2d\u95f4\u6001\u4e0d\u518d\u53e6\u63a8\uff08\u907f\u514d\u91cd\u590d\u5237\u5361\uff09\uff1b
            # \u53ea\u5728\u7ec8\u6001\u63a8\u4e00\u5f20\u7ed3\u679c\u5361\uff0c\u4e14\u4e0e\u8c03\u5ea6\u5668\u5bf9\u8d26\u5171\u7528 NX \u95f8\u95e8\u53bb\u91cd\uff08\u8c01\u5148\u5230\u8c01\u63a8\uff09\u3002
            def _on_update(j):
                if j["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED) \
                        and _claim_dataflow_notify(j["job_id"]):
                    _send_card("", _cfg_chat(), result_card(j))
            orchestrator.run_to_completion(job, on_update=_on_update)
        except Exception as e:
            logger.error("[Transfer] execute failed job=%s", job_id, exc_info=True)
            _send_text("", _cfg_chat(), f"\u274c \u8fc1\u79fb\u4efb\u52a1 {job_id} \u5931\u8d25\uff1a{e}")

    threading.Thread(target=_do_transfer, daemon=True).start()
    # \u539f\u5730\u628a\u201c\u786e\u8ba4\u5361\u201d\u66ff\u6362\u6210\u201c\u8fdb\u884c\u4e2d\u201d\u5361\uff1a\u786e\u8ba4\u6309\u94ae\u968f\u4e4b\u6d88\u5931 \u2192 \u4e0d\u80fd\u518d\u8fde\u70b9\uff08\u4e0e CPFS \u4e00\u81f4\uff09\u3002
    # \u7528\u5c55\u793a\u526f\u672c\u8bbe\u9636\u6bb5\uff0c\u4e0d\u52a8\u771f job \u7684 stage\uff08run_to_completion \u7684\u6c89\u964d\u6bb5\u4ecd\u6309\u771f stage \u5224\u5b9a\uff09\u3002
    from core.transfer.cards import progress_card, progress_card_v2
    disp = dict(job)
    disp["stage"] = (orchestrator.STAGE_SINKING
                     if job.get("needs_sink") and not job.get("sink_done")
                     else orchestrator.STAGE_CROSSING)
    card_data = progress_card_v2(disp) if reply_v2 else progress_card(disp)
    return {"toast": {"type": "success", "content": "\u5df2\u5f00\u59cb\u8fc1\u79fb\uff0c\u5b8c\u6210\u540e\u63a8\u9001\u7ed3\u679c"},
            "card": {"type": "raw", "data": card_data}}


def _async_refresh_and_push(orch, job_id, progress_card, result_card, chat):
    """后台真正 refresh（会 poll 云端、可能数十秒），阶段推进才推更新卡；终态经
    _claim_dataflow_notify 去重，避免与在线线程/对账重复推。查询回调本身只读 Redis 秒回，
    绝不在 3s 回调里 poll 云端（否则超时→飞书重投→原地卡永远刷不出、只见失败）。"""
    from core.dsw_scheduler import _send_card, _claim_dataflow_notify
    try:
        prev = orch.get_job(job_id)
        prev_stage = prev.get("stage") if prev else None
        job = orch.refresh(job_id)
        if not job:
            return
        terminal = job["stage"] in (orch.STAGE_DONE, orch.STAGE_FAILED)
        if job["stage"] == prev_stage and not terminal:
            return  # 无变化，别刷屏
        if terminal and not _claim_dataflow_notify(job["job_id"]):
            return  # 终态卡已被在线线程/对账推过
        _send_card("", chat, result_card(job) if terminal else progress_card(job))
    except Exception:
        logger.error("[Query] 后台刷新失败 job=%s", job_id, exc_info=True)


def _h_query_transfer_progress(action_val, open_id, chat_id, form_value):
    """点“查询这条”：只读缓存秒回当前卡，后台异步 refresh 后推更新卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.transfer import orchestrator
    from core.transfer.cards import progress_card, result_card
    job = orchestrator.get_job(job_id) if job_id else None   # 只读缓存，不在 3s 回调里 poll 云端
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    terminal = job["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED)
    if not terminal:
        threading.Thread(target=_async_refresh_and_push,
                         args=(orchestrator, job_id, progress_card, result_card, _cfg_chat()),
                         daemon=True).start()
    card = result_card(job) if terminal else progress_card(job)
    return {"toast": {"type": "success",
                      "content": f"当前阶段：{job['stage']}" + ("" if terminal else "（后台刷新中）")},
            "card": {"type": "raw", "data": card}}


def _h_retry_transfer(action_val, open_id, chat_id, form_value):
    """Retry a failed transfer with the same confirmation flow."""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.transfer import orchestrator
    job = orchestrator.get_job(job_id) if job_id else None
    if not job:
        return {"toast": {"type": "error", "content": "\u4efb\u52a1\u4e0d\u5b58\u5728\u6216\u5df2\u8fc7\u671f"}}
    # \u5fc5\u987b\u843d\u76d8\uff1a_h_confirm_transfer \u4f1a\u4ece Redis \u91cd\u65b0\u53d6 job\uff1b\u4e0d save \u5219\u8bfb\u5230\u65e7 launched=True \u88ab\u6321\u3002
    job["stage"] = orchestrator.STAGE_NEW
    job["error"] = ""
    job["launched"] = False
    orchestrator._save(job)
    try:
        from utils.redis_client import get_redis
        # 清 launch 锁（否则 30s 内重试被"已下发"挡）+ 清终态通知闸门
        # （job_id 当天不变，首次失败已占 dataflow:notified，不清则重试成功后结果卡永远不推）。
        get_redis().delete(f"transfer:launch:{job_id}", f"dataflow:notified:{job_id}")
    except Exception:
        pass
    # retry 由 1.0 结果卡触发，回 1.0 progress_card 原地替换（1.0→1.0，同家族）。
    return _h_confirm_transfer({"job_id": job_id}, open_id, chat_id, form_value, reply_v2=False)


def _cfg_chat() -> str:
    return settings.TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID


# CPFS 预热/沉降：录入 / 确认 / 重试

def _cfg_cpfs_chat() -> str:
    return settings.CPFS_CHAT_ID or settings.FEISHU_CHAT_ID


# ── 数据预热/沉降统一向导：选云 → 选地区 → 选文件系统 + 填源/目的地址 ──────────────

def _dataflow_regions(cloud: str, open_id: str = "") -> list:
    """该云有文件系统的地区列表（级联第一步）。"""
    try:
        if cloud == "volcano":
            from core.vepfs_dataflow import discovery
            return discovery.regions()
        from core.cpfs_dataflow import discovery as d
        return d.regions(open_id=open_id)
    except Exception:
        logger.warning("[DATAFLOW] 发现地区失败 cloud=%s", cloud, exc_info=True)
        return []


def _dataflow_fs_options(cloud: str, region: str, open_id: str = "") -> list:
    """某地区的文件系统下拉项 [{value:'fs@region', text}]。"""
    try:
        if cloud == "volcano":
            from core.vepfs_dataflow import discovery
            return discovery.fs_options(region)
        from core.cpfs_dataflow import discovery as d
        return [{"value": f"{f['fs_id']}@{region}", "text": f"{f['fs_id']}（{region}）"}
                for f in d.filesystems_in(region, open_id=open_id)]
    except Exception:
        logger.warning("[DATAFLOW] 发现文件系统失败 cloud=%s region=%s", cloud, region, exc_info=True)
        return []


def _orient_addresses(source: str, dest: str, err_cls):
    """按源/目的地址定方向，返回 (operation, fs_dir, obj_addr)。一侧须是 oss:// 或 tos://。"""
    s_obj = source.lower().startswith(("oss://", "tos://"))
    d_obj = dest.lower().startswith(("oss://", "tos://"))
    if s_obj == d_obj:
        raise err_cls("源和目的须一侧是对象存储（oss:// 或 tos://）、另一侧是文件系统目录 /dir/")
    if s_obj:
        return "preheat", dest, source     # 源=对象存储 → 预热到文件系统(dest)
    return "sink", source, dest            # 源=文件系统 → 沉降到对象存储(dest)


def _guided_plan(cloud: str, fv: dict, region_hint: str = ""):
    """向导 form（fs 下拉/手填 + 源/目的地址）→ 计划。方向由地址自动判断。"""
    fs_raw = (fv.get("fs") or "").strip()
    if "://" in fs_raw:                     # 手动录入 cpfs://fs 或 vepfs://fs
        fs_raw = fs_raw.split("://", 1)[1].strip("/")
    fs_id, _, fs_region = fs_raw.partition("@")
    region = fs_region or region_hint
    source = (fv.get("source") or "").strip()
    dest = (fv.get("dest") or "").strip()
    same = (fv.get("same_name") or "").strip()
    if cloud == "volcano":
        from core.vepfs_dataflow import orchestrator as o
        region = region or settings.VEPFS_REGION
        if not fs_id:
            raise o.DataflowPathError("请选择 vePFS 文件系统")
        op, fs_dir, obj = _orient_addresses(source, dest, o.DataflowPathError)
        return o.make_plan(op, fs_dir, obj, fs_id=fs_id, region=region, same_name=same)
    from core.cpfs_dataflow import orchestrator as o
    region = region or settings.CPFS_REGION
    if not fs_id:
        raise o.DataflowPathError("请选择 CPFS 文件系统")
    op, fs_dir, obj = _orient_addresses(source, dest, o.DataflowPathError)
    return o.make_plan(op, fs_dir, obj, fs_id=fs_id, region=region)


def _h_pick_cloud_aliyun(action_val, open_id, chat_id, form_value):
    """入口点「阿里」→ 选地区卡。"""
    from core.dataflow_cards import region_card
    return {"card": {"type": "raw", "data": region_card("aliyun", _dataflow_regions("aliyun", open_id))}}


def _h_pick_cloud_volcano(action_val, open_id, chat_id, form_value):
    """入口点「火山」→ 选地区卡。"""
    from core.dataflow_cards import region_card
    return {"card": {"type": "raw", "data": region_card("volcano", _dataflow_regions("volcano", open_id))}}


def _h_pick_region_aliyun(action_val, open_id, chat_id, form_value):
    region = (action_val or {}).get("region", "") if isinstance(action_val, dict) else ""
    from core.dataflow_cards import form_card
    return {"card": {"type": "raw",
                     "data": form_card("aliyun", region, _dataflow_fs_options("aliyun", region, open_id))}}


def _h_pick_region_volcano(action_val, open_id, chat_id, form_value):
    region = (action_val or {}).get("region", "") if isinstance(action_val, dict) else ""
    from core.dataflow_cards import form_card
    return {"card": {"type": "raw",
                     "data": form_card("volcano", region, _dataflow_fs_options("volcano", region, open_id))}}


def _h_submit_cpfs_dataflow(action_val, open_id, chat_id, form_value):
    """向导 form（选 fs + 源/目的地址）或旧地址卡 → 后台解析计划，推确认卡。"""
    fv = form_value or {}
    region_hint = (action_val or {}).get("region", "") if isinstance(action_val, dict) else ""
    guided = bool(fv.get("fs"))
    region = (fv.get("region") or "").strip()
    source = (fv.get("source") or "").strip()
    dest = (fv.get("dest") or "").strip()
    if not guided and not (region and source and dest):
        return {"toast": {"type": "error", "content": "请选文件系统并填源、目的地址"}}

    def _do_prepare() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        from core.cpfs_dataflow import orchestrator, engine_nas
        from core.cpfs_dataflow.cards import confirm_card
        from core.cpfs_dataflow.orchestrator import DataflowPathError
        try:
            if guided:
                plan = _guided_plan("aliyun", fv, region_hint)
            else:
                plan = orchestrator.plan_from_addresses(region, source, dest, open_id=open_id)
            job = orchestrator.create_job_record(plan, open_id=open_id)
            # 幂等：飞书对"解析预览"回调也可能重投递/并发 → 会推多张一样的确认卡。
            # 按 job_id（op/fs/dir/oss/当天 的 hash，两次解析同值）原子去重，只推一张。
            try:
                from utils.redis_client import get_redis
                if not get_redis().set(f"cpfs:dataflow:confirmcard:{job['job_id']}", 1, nx=True, ex=30):
                    return
            except Exception:
                pass
            _send_card(open_id, _cfg_cpfs_chat(), confirm_card(job))
        except (DataflowPathError, engine_nas.NasDataflowError) as e:
            _send_text(open_id, _cfg_cpfs_chat(), f"❌ {e}")
        except Exception as e:
            logger.error("[CPFS] prepare failed", exc_info=True)
            _send_text(open_id, _cfg_cpfs_chat(), f"❌ 请求处理失败：{e}")

    threading.Thread(target=_do_prepare, daemon=True).start()
    return {"toast": {"type": "success", "content": "正在解析，稍候推送确认卡"}}


def _h_confirm_cpfs_dataflow(action_val, open_id, chat_id, form_value):
    """后台启动预热/沉降任务，完成后推结果卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    if not job_id:
        return {"toast": {"type": "error", "content": "缺少任务 ID"}}
    from core.cpfs_dataflow import orchestrator
    job = orchestrator.get_job(job_id)
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    if job["stage"] not in (orchestrator.STAGE_NEW, orchestrator.STAGE_FAILED):
        return {"toast": {"type": "info", "content": f"任务已在 {job['stage']}，无需重复"}}

    # 幂等：飞书卡片回调至少投递一次（会重发/并发），原子抢锁，抢不到即重复投递 → 直接返回，
    # 否则两个并发回调都会通过上面的 stage 守卫、各建一个临时 DataFlow + 任务。
    try:
        from utils.redis_client import get_redis
        if not get_redis().set(f"cpfs:dataflow:launch:{job_id}", 1, nx=True, ex=30):
            return {"toast": {"type": "info", "content": "任务已下发，请勿重复点击"}}
    except Exception:
        pass  # Redis 不可用时降级：至少同步置 RUNNING 兜住非并发的重复投递
    # 同步置 RUNNING 落盘，堵住"线程异步改 stage 之前又来一次回调"的窗口
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)

    def _do_run() -> None:
        from core.dsw_scheduler import _send_card, _send_text, _claim_dataflow_notify
        from core.cpfs_dataflow.cards import result_card, progress_card
        try:
            def _on_update(j):
                if j["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED):
                    if _claim_dataflow_notify(j["job_id"]):   # 与对账共用去重闸门
                        _send_card("", _cfg_cpfs_chat(), result_card(j))
                else:
                    _send_card("", _cfg_cpfs_chat(), progress_card(j))
            orchestrator.run_to_completion(job, on_update=_on_update)
        except Exception as e:
            logger.error("[CPFS] execute failed job=%s", job_id, exc_info=True)
            _send_text("", _cfg_cpfs_chat(), f"❌ 任务 {job_id} 失败：{e}")

    threading.Thread(target=_do_run, daemon=True).start()
    from core.cpfs_dataflow.cards import progress_card
    # 直接把确认卡换成带「查询进度」按钮的进度卡，toast 回带任务 ID
    return {"toast": {"type": "success", "content": f"已下发，任务 {job_id}；可点“查询进度”"},
            "card": {"type": "raw", "data": progress_card(job)}}


def _h_query_cpfs_progress(action_val, open_id, chat_id, form_value):
    """点“查询进度”：只读缓存秒回当前卡，后台异步 refresh 后推更新卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.cpfs_dataflow import orchestrator
    from core.cpfs_dataflow.cards import progress_card, result_card
    job = orchestrator.get_job(job_id) if job_id else None   # 只读缓存，不在 3s 回调里 poll 云端
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    terminal = job["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED)
    if not terminal:
        threading.Thread(target=_async_refresh_and_push,
                         args=(orchestrator, job_id, progress_card, result_card, _cfg_cpfs_chat()),
                         daemon=True).start()
    card = result_card(job) if terminal else progress_card(job)
    return {"toast": {"type": "success",
                      "content": f"当前阶段：{job['stage']}" + ("" if terminal else "（后台刷新中）")},
            "card": {"type": "raw", "data": card}}


def _h_retry_cpfs_dataflow(action_val, open_id, chat_id, form_value):
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.cpfs_dataflow import orchestrator
    job = orchestrator.get_job(job_id) if job_id else None
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    # 防御性落盘：_h_confirm_* 会从 Redis 重读 job（FAILED 本就过 guard），落盘保证重读到干净状态。
    job["stage"] = orchestrator.STAGE_NEW
    job["error"] = ""
    orchestrator._save(job)
    try:
        from utils.redis_client import get_redis
        # 清 launch 锁（否则 30s 内重试被"已下发"挡）+ 清终态通知闸门（否则重试成功后结果卡不推）。
        get_redis().delete(f"cpfs:dataflow:launch:{job_id}", f"dataflow:notified:{job_id}")
    except Exception:
        pass
    return _h_confirm_cpfs_dataflow({"job_id": job_id}, open_id, chat_id, form_value)


# 火山 vePFS 预热/沉降：录入 / 确认 / 查询 / 重试（镜像 CPFS，走 core.vepfs_dataflow）

def _cfg_vepfs_chat() -> str:
    return settings.VEPFS_CHAT_ID or settings.FEISHU_CHAT_ID


def _h_submit_vepfs_dataflow(action_val, open_id, chat_id, form_value):
    """向导 form（选 fs + 源/目的地址）或旧地址卡 → 后台解析计划，推确认卡。"""
    fv = form_value or {}
    region_hint = (action_val or {}).get("region", "") if isinstance(action_val, dict) else ""
    guided = bool(fv.get("fs"))
    region = (fv.get("region") or "").strip()
    source = (fv.get("source") or "").strip()
    dest = (fv.get("dest") or "").strip()
    same_name = (fv.get("same_name") or "").strip()
    if not guided and not (region and source and dest):
        return {"toast": {"type": "error", "content": "请选文件系统并填源、目的地址"}}

    def _do_prepare() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        from core.vepfs_dataflow import orchestrator, engine_vepfs
        from core.vepfs_dataflow.cards import confirm_card
        from core.vepfs_dataflow.orchestrator import DataflowPathError
        try:
            if guided:
                plan = _guided_plan("volcano", fv, region_hint)
            else:
                plan = orchestrator.plan_from_addresses(region, source, dest, same_name=same_name)
            job = orchestrator.create_job_record(plan, open_id=open_id)
            try:
                from utils.redis_client import get_redis
                if not get_redis().set(f"vepfs:dataflow:confirmcard:{job['job_id']}", 1, nx=True, ex=30):
                    return
            except Exception:
                pass
            _send_card(open_id, _cfg_vepfs_chat(), confirm_card(job))
        except (DataflowPathError, engine_vepfs.VepfsDataflowError) as e:
            _send_text(open_id, _cfg_vepfs_chat(), f"❌ {e}")
        except Exception as e:
            logger.error("[VEPFS] prepare failed", exc_info=True)
            _send_text(open_id, _cfg_vepfs_chat(), f"❌ 请求处理失败：{e}")

    threading.Thread(target=_do_prepare, daemon=True).start()
    return {"toast": {"type": "success", "content": "正在解析，稍候推送确认卡"}}


def _h_confirm_vepfs_dataflow(action_val, open_id, chat_id, form_value):
    """后台启动预热/沉降任务，完成后推结果卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    if not job_id:
        return {"toast": {"type": "error", "content": "缺少任务 ID"}}
    from core.vepfs_dataflow import orchestrator
    job = orchestrator.get_job(job_id)
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    if job["stage"] not in (orchestrator.STAGE_NEW, orchestrator.STAGE_FAILED):
        return {"toast": {"type": "info", "content": f"任务已在 {job['stage']}，无需重复"}}

    try:
        from utils.redis_client import get_redis
        if not get_redis().set(f"vepfs:dataflow:launch:{job_id}", 1, nx=True, ex=30):
            return {"toast": {"type": "info", "content": "任务已下发，请勿重复点击"}}
    except Exception:
        pass
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)

    def _do_run() -> None:
        from core.dsw_scheduler import _send_card, _send_text, _claim_dataflow_notify
        from core.vepfs_dataflow.cards import result_card, progress_card
        try:
            def _on_update(j):
                if j["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED):
                    if _claim_dataflow_notify(j["job_id"]):   # 与对账共用去重闸门
                        _send_card("", _cfg_vepfs_chat(), result_card(j))
                else:
                    _send_card("", _cfg_vepfs_chat(), progress_card(j))
            orchestrator.run_to_completion(job, on_update=_on_update)
        except Exception as e:
            logger.error("[VEPFS] execute failed job=%s", job_id, exc_info=True)
            _send_text("", _cfg_vepfs_chat(), f"❌ 任务 {job_id} 失败：{e}")

    threading.Thread(target=_do_run, daemon=True).start()
    from core.vepfs_dataflow.cards import progress_card
    return {"toast": {"type": "success", "content": f"已下发，任务 {job_id}；可点“查询进度”"},
            "card": {"type": "raw", "data": progress_card(job)}}


def _h_query_vepfs_progress(action_val, open_id, chat_id, form_value):
    """点“查询进度”：只读缓存秒回当前卡，后台异步 refresh 后推更新卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.vepfs_dataflow import orchestrator
    from core.vepfs_dataflow.cards import progress_card, result_card
    job = orchestrator.get_job(job_id) if job_id else None   # 只读缓存，不在 3s 回调里 poll 云端
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    terminal = job["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED)
    if not terminal:
        threading.Thread(target=_async_refresh_and_push,
                         args=(orchestrator, job_id, progress_card, result_card, _cfg_vepfs_chat()),
                         daemon=True).start()
    card = result_card(job) if terminal else progress_card(job)
    return {"toast": {"type": "success",
                      "content": f"当前阶段：{job['stage']}" + ("" if terminal else "（后台刷新中）")},
            "card": {"type": "raw", "data": card}}


def _h_retry_vepfs_dataflow(action_val, open_id, chat_id, form_value):
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.vepfs_dataflow import orchestrator
    job = orchestrator.get_job(job_id) if job_id else None
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    # 防御性落盘：_h_confirm_* 会从 Redis 重读 job（FAILED 本就过 guard），落盘保证重读到干净状态。
    job["stage"] = orchestrator.STAGE_NEW
    job["error"] = ""
    orchestrator._save(job)
    try:
        from utils.redis_client import get_redis
        # 清 launch 锁（否则 30s 内重试被"已下发"挡）+ 清终态通知闸门（否则重试成功后结果卡不推）。
        get_redis().delete(f"vepfs:dataflow:launch:{job_id}", f"dataflow:notified:{job_id}")
    except Exception:
        pass
    return _h_confirm_vepfs_dataflow({"job_id": job_id}, open_id, chat_id, form_value)


def _h_query_progress_by_id(action_val, open_id, chat_id, form_value):
    """查询进度输入卡提交：读 form_value.job_id，按前缀分发。

    输入卡是 schema 2.0，而进度/结果卡是老式 1.0；用 1.0 卡原地替换 2.0 卡会被飞书拒
    （错误 200830）。故这里改为**推送一张新卡** + 回 toast，不做原地替换。
    """
    jid = ((form_value or {}).get("job_id") or "").strip()
    if not jid:
        return {}
    low = jid.lower()
    if not low.startswith(("vepfs-", "cpfs-", "tr-")):
        return {"toast": {"type": "error", "content": "任务 ID 需以 vepfs- / cpfs- / tr- 开头"}}
    # 飞书对同一次点击会双投递（老式回调 + schema 2.0 事件，两条都带 job_id）。
    # 推送式返回会各推一张卡 → 两张。按 (open_id, job_id) 原子去重，同一次点击只推一张。
    try:
        from utils.redis_client import get_redis
        if not get_redis().set(f"cpfs:query:{open_id}:{jid}", 1, nx=True, ex=8):
            return {"toast": {"type": "info", "content": "查询中…"}}
    except Exception:
        pass

    # refresh 会 poll 云端（可能数十秒）→ 放后台线程，回调只回 toast 秒返，避免超 3s 被飞书重投。
    def _bg():
        from core.dsw_scheduler import _send_card, _send_text
        if low.startswith("vepfs-"):
            from core.vepfs_dataflow import orchestrator as o
            from core.vepfs_dataflow.cards import progress_card, result_card
            chat = chat_id or _cfg_vepfs_chat()
        elif low.startswith("cpfs-"):
            from core.cpfs_dataflow import orchestrator as o
            from core.cpfs_dataflow.cards import progress_card, result_card
            chat = chat_id or _cfg_cpfs_chat()
        else:  # tr-
            from core.transfer import orchestrator as o
            from core.transfer.cards import progress_card, result_card
            chat = chat_id or _cfg_chat()
        try:
            job = o.refresh(jid)
        except Exception:
            logger.error("[Query] 按 ID 刷新失败 %s", jid, exc_info=True)
            job = None
        if not job:
            _send_text(open_id, chat, f"❌ 未找到任务 {jid}（可能已过期）")
            return
        c = (result_card(job) if job["stage"] in (o.STAGE_DONE, o.STAGE_FAILED)
             else progress_card(job))
        _send_card(open_id, chat, c)   # 保持原行为：私信发起人优先，降级到 chat

    threading.Thread(target=_bg, daemon=True).start()
    return {"toast": {"type": "success", "content": f"正在查询 {jid} …"}}


# ── 桶间迁移（同云一次性搬运）──────────────────────────────────────────────────

def _cfg_bkt_chat() -> str:
    return settings.TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID


def _h_submit_bucket_transfer(action_val, open_id, chat_id, form_value):
    """录入卡提交 → 后台解析计划、推确认卡。"""
    fv = form_value or {}
    source = (fv.get("source") or "").strip()
    dest = (fv.get("dest") or "").strip()
    same_name = (fv.get("same_name") or "").strip()
    if not (source and dest):
        return {"toast": {"type": "error", "content": "请填写源、目的地址"}}

    def _do() -> None:
        from core.dsw_scheduler import _send_card, _send_text
        from core.bucket_transfer import orchestrator, paths
        from core.bucket_transfer.cards import confirm_card
        try:
            plan = orchestrator.make_plan(source, dest)
            job = orchestrator.create_job_record(plan, same_name=same_name, open_id=open_id)
            try:
                from utils.redis_client import get_redis
                if not get_redis().set(f"bkt:confirmcard:{job['job_id']}", 1, nx=True, ex=30):
                    return
            except Exception:
                pass
            _send_card(open_id, _cfg_bkt_chat(), confirm_card(job))
        except paths.PathError as e:
            _send_text(open_id, _cfg_bkt_chat(), f"❌ {e}")
        except Exception as e:
            logger.error("[BKT] prepare failed", exc_info=True)
            _send_text(open_id, _cfg_bkt_chat(), f"❌ 请求处理失败：{e}")

    threading.Thread(target=_do, daemon=True).start()
    return {"toast": {"type": "success", "content": "正在解析，稍候推送确认卡"}}


def _h_confirm_bucket_transfer(action_val, open_id, chat_id, form_value):
    """确认下发 → 后台跑迁移，推进度/结果卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    if not job_id:
        return {"toast": {"type": "error", "content": "缺少任务 ID"}}
    from core.bucket_transfer import orchestrator
    job = orchestrator.get_job(job_id)
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    if job["stage"] not in (orchestrator.STAGE_NEW, orchestrator.STAGE_FAILED):
        return {"toast": {"type": "info", "content": f"任务已在 {job['stage']}，无需重复"}}
    try:
        from utils.redis_client import get_redis
        if not get_redis().set(f"bkt:launch:{job_id}", 1, nx=True, ex=30):
            return {"toast": {"type": "info", "content": "任务已下发，请勿重复点击"}}
    except Exception:
        pass
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)

    def _run() -> None:
        from core.dsw_scheduler import _send_card, _send_text, _claim_dataflow_notify
        from core.bucket_transfer import orchestrator as o
        from core.bucket_transfer.cards import result_card, progress_card
        try:
            def _upd(j):
                if j["stage"] in (o.STAGE_DONE, o.STAGE_FAILED):
                    if _claim_dataflow_notify(j["job_id"]):   # 与对账共用去重闸门
                        _send_card("", _cfg_bkt_chat(), result_card(j))
                else:
                    _send_card("", _cfg_bkt_chat(), progress_card(j))
            o.run_to_completion(job, on_update=_upd)
        except Exception as e:
            logger.error("[BKT] execute failed job=%s", job_id, exc_info=True)
            _send_text("", _cfg_bkt_chat(), f"❌ 任务 {job_id} 失败：{e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"toast": {"type": "success", "content": f"已下发，任务 {job_id}；进度卡稍后推送"}}


def _h_query_bucket_transfer(action_val, open_id, chat_id, form_value):
    """点“查询进度”：只读缓存秒回当前卡（均 1.0，原地替换安全），后台异步 refresh 后推更新卡。"""
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.bucket_transfer import orchestrator
    from core.bucket_transfer.cards import progress_card, result_card
    job = orchestrator.get_job(job_id) if job_id else None   # 只读缓存，不在 3s 回调里 poll 云端
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    terminal = job["stage"] in (orchestrator.STAGE_DONE, orchestrator.STAGE_FAILED)
    if not terminal:
        threading.Thread(target=_async_refresh_and_push,
                         args=(orchestrator, job_id, progress_card, result_card, _cfg_bkt_chat()),
                         daemon=True).start()
    card = result_card(job) if terminal else progress_card(job)
    return {"toast": {"type": "success",
                      "content": f"当前阶段：{job['stage']}" + ("" if terminal else "（后台刷新中）")},
            "card": {"type": "raw", "data": card}}


def _h_retry_bucket_transfer(action_val, open_id, chat_id, form_value):
    job_id = action_val.get("job_id", "") if isinstance(action_val, dict) else ""
    from core.bucket_transfer import orchestrator
    job = orchestrator.get_job(job_id) if job_id else None
    if not job:
        return {"toast": {"type": "error", "content": "任务不存在或已过期"}}
    job["stage"] = orchestrator.STAGE_NEW
    job["error"] = ""
    orchestrator._save(job)
    try:
        from utils.redis_client import get_redis
        # 清 launch 锁（否则 30s 内重试被"已下发"挡）+ 清终态通知闸门（否则重试成功后结果卡不推）。
        get_redis().delete(f"bkt:launch:{job_id}", f"dataflow:notified:{job_id}")
    except Exception:
        pass
    return _h_confirm_bucket_transfer({"job_id": job_id}, open_id, chat_id, form_value)


_ACTION_HANDLERS = {
    "mfu_region":         _h_mfu_region,
    "submit_ak_register": _h_submit_ak_register,
    "submit_gpu_request": _h_submit_gpu_request,
    "quick_gpu":          _h_quick_gpu,
    "extend_dsw":         _h_extend_dsw,
    "stop_dsw":           _h_stop_dsw,
    "approve_gpu":        _h_approve_gpu,
    "reject_gpu":         _h_reject_gpu,
    "approve_oss_perm":          _h_approve_oss_perm,
    "approve_oss_perm_selected": _h_approve_oss_perm_selected,
    "open_ram_query":    _h_open_ram_query,
    "submit_ram_query":  _h_submit_ram_query,
    "open_volcano_query":   _h_open_volcano_query,
    "submit_volcano_query": _h_submit_volcano_query,
    "submit_transfer":    _h_submit_transfer,
    "confirm_transfer":   _h_confirm_transfer,
    "query_transfer_progress": _h_query_transfer_progress,
    "retry_transfer":     _h_retry_transfer,
    "pick_cloud_aliyun":     _h_pick_cloud_aliyun,
    "pick_cloud_volcano":    _h_pick_cloud_volcano,
    "pick_region_aliyun":    _h_pick_region_aliyun,
    "pick_region_volcano":   _h_pick_region_volcano,
    "submit_cpfs_dataflow":  _h_submit_cpfs_dataflow,
    "confirm_cpfs_dataflow": _h_confirm_cpfs_dataflow,
    "query_cpfs_progress":   _h_query_cpfs_progress,
    "retry_cpfs_dataflow":   _h_retry_cpfs_dataflow,
    "submit_vepfs_dataflow":  _h_submit_vepfs_dataflow,
    "confirm_vepfs_dataflow": _h_confirm_vepfs_dataflow,
    "query_vepfs_progress":   _h_query_vepfs_progress,
    "retry_vepfs_dataflow":   _h_retry_vepfs_dataflow,
    "query_progress_by_id":  _h_query_progress_by_id,
    "submit_bucket_transfer":  _h_submit_bucket_transfer,
    "confirm_bucket_transfer": _h_confirm_bucket_transfer,
    "query_bucket_transfer":   _h_query_bucket_transfer,
    "retry_bucket_transfer":   _h_retry_bucket_transfer,
}


def _process_action(action_name: str, action_val: dict, open_id: str, chat_id: str,
                    form_value: dict | None = None) -> dict:
    """所有卡片动作的共享处理逻辑，返回飞书期待的响应 dict。未知 action 返回 {}。"""
    handler = _ACTION_HANDLERS.get(action_name)
    if handler is None:
        return {}
    return handler(action_val, open_id, chat_id, form_value)


def card_action_is_duplicate(action_name, open_id, msg_id, form_value, action_val) -> bool:
    """飞书对同一次卡片操作会双投递（老式回调 URL + 事件订阅 card.action.trigger），且各
    通道可能重试。按 (动作+操作人+消息+表单+值) 内容哈希原子去重（SET NX 15s），同一次操作
    只放行一次 → 推送式 handler 不再推多张卡、建流不再重复。Redis 不可用则不去重（降级）。"""
    try:
        import hashlib
        from utils.redis_client import get_redis
        blob = json.dumps(
            {"a": action_name or "", "o": open_id or "", "m": msg_id or "",
             "f": form_value or {}, "v": action_val or {}},
            sort_keys=True, ensure_ascii=False)
        key = "feishu:card_dedup:" + hashlib.md5(blob.encode("utf-8")).hexdigest()
        return not get_redis().set(key, 1, nx=True, ex=15)
    except Exception:
        return False


def _handle_card_trigger_sync(data: dict) -> dict:
    """处理经事件订阅路径到达的 card.action.trigger（schema 2.0 结构）。"""
    event      = data.get("event", {})
    action_obj = event.get("action", {})
    action_val = action_obj.get("value") or {}
    form_value = action_obj.get("form_value") or {}
    open_id    = event.get("operator", {}).get("operator_id", {}).get("open_id", "")
    chat_id    = event.get("context", {}).get("open_chat_id", "") or settings.FEISHU_CHAT_ID
    msg_id     = event.get("context", {}).get("open_message_id", "")
    action_name = action_val.get("action", "") if isinstance(action_val, dict) else ""
    if not action_name and form_value:
        # 兼容旧 AK 表单（ak_id/ak_secret）和新 RAM 绑定表单（ram_user_name/user_name）
        if any(k in form_value for k in ("login_name", "user_name", "q")):
            action_name = "submit_ram_query"
        elif any(k in form_value for k in ("ak_id", "ak_secret", "ram_user_name")):
            action_name = "submit_ak_register"
        else:
            action_name = "submit_gpu_request"
    if card_action_is_duplicate(action_name, open_id, msg_id, form_value, action_val):
        return {}
    return _process_action(action_name, action_val, open_id, chat_id, form_value=form_value)
