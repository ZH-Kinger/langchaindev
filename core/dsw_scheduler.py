"""
DSW 工单调度器。

职责：
  1. 每 2 分钟轮询 Jira，发现新 GPU 申请工单 → 自动创建 DSW 实例
  2. 每 5 分钟检查运行中的实例：
     - 超时前 15 分钟 → 飞书提醒（卡片���「延续/停止」按钮）
     - 警告后 DSW_IDLE_STOP_MINUTES 分钟无响应 → 自动停止实例

状态存储在 Redis：
  dsw:ticket:{ticket_key}  →  JSON 字符串，含：
    instance_id, open_id, chat_id,
    start_ts, duration_hours,
    warned (bool), warn_ts
"""
import json
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from tools.jira_tool import (
    get_gpu_tickets, add_comment, transition_ticket,
    parse_ticket_metadata, update_ticket_field,
)
from tools.pai_dsw_tool import manage_pai_dsw
from utils.redis_client import get_redis


# ── Redis helpers ─────────────────────────────────────────────────────────────

_KEY_PREFIX = "dsw:ticket:"


def _redis_set(ticket_key: str, data: dict) -> None:
    try:
        r = get_redis()
        r.set(_KEY_PREFIX + ticket_key, json.dumps(data), ex=7 * 86400)
    except Exception as e:
        print(f"[Scheduler] Redis set 失败: {e}")


def _redis_get(ticket_key: str) -> Optional[dict]:
    try:
        raw = get_redis().get(_KEY_PREFIX + ticket_key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _redis_delete(ticket_key: str) -> None:
    try:
        get_redis().delete(_KEY_PREFIX + ticket_key)
    except Exception:
        pass


_PENDING_REG_PREFIX = "dsw:pending_reg:"
_PENDING_REG_TTL    = 3600  # 1 小时内不重复提醒


def _mark_pending_reg(ticket_key: str) -> bool:
    """首次标记返回 True（应发通知），已标记返回 False（避免重复推送）。"""
    try:
        r = get_redis()
        key = _PENDING_REG_PREFIX + ticket_key
        if r.exists(key):
            return False
        r.set(key, "1", ex=_PENDING_REG_TTL)
        return True
    except Exception:
        return True


# ── 配额管理 ──────────────────────────────────────────────────────────────────────

_QUOTA_PREFIX = "gpu:quota:"


def _quota_key(open_id: str) -> str:
    from datetime import datetime
    return f"{_QUOTA_PREFIX}{open_id}:{datetime.now().strftime('%Y%m')}"


def get_quota_used(open_id: str) -> float:
    """返回当月已用 GPU·小时数。"""
    try:
        raw = get_redis().get(_quota_key(open_id))
        return float(raw) if raw else 0.0
    except Exception:
        return 0.0


def add_quota_usage(open_id: str, gpu_count: int, hours: float) -> None:
    """记录本次申请消耗的 GPU·小时数（月底自动过期）。"""
    if not open_id:
        return
    try:
        import calendar
        from datetime import datetime as _dt
        now_dt   = _dt.now()
        last_day = calendar.monthrange(now_dt.year, now_dt.month)[1]
        expire   = _dt(now_dt.year, now_dt.month, last_day, 23, 59, 59)
        ttl      = max(86400, int((expire - now_dt).total_seconds()) + 86400)
        r   = get_redis()
        key = _quota_key(open_id)
        r.incrbyfloat(key, gpu_count * hours)
        r.expire(key, ttl)
    except Exception as e:
        print(f"[Quota] 记录使用量失败: {e}")


def check_quota(open_id: str, gpu_count: int, hours: float) -> tuple[bool, float, float]:
    """返回 (是否在配额内, 当月已用GPU·h, 月配额上限)。"""
    used   = get_quota_used(open_id)
    limit  = settings.GPU_QUOTA_HOURS_PER_MONTH
    needed = gpu_count * hours
    return (used + needed <= limit, used, limit)


# ── 费用估算 ──────────────────────────────────────────────────────────────────────

def _cost_str(gpu_count: int, hours: float) -> str:
    total = gpu_count * hours * settings.GPU_PRICE_PER_HOUR
    return f"约 ¥{total:.0f}（{gpu_count}GPU × {hours}h × ¥{settings.GPU_PRICE_PER_HOUR:.0f}/GPU·h）"


# ── 审批流 ────────────────────────────────────────────────────────────────────────

_APPROVED_PREFIX          = "dsw:approved:"
_APPROVAL_NOTIFIED_PREFIX = "dsw:approval_notified:"


def _is_approved(ticket_key: str) -> bool:
    try:
        return bool(get_redis().exists(_APPROVED_PREFIX + ticket_key))
    except Exception:
        return False


def _set_approved(ticket_key: str) -> None:
    try:
        get_redis().set(_APPROVED_PREFIX + ticket_key, "1", ex=7 * 86400)
    except Exception:
        pass


def _mark_approval_notified(ticket_key: str) -> bool:
    """首次返回 True，已通知返回 False。"""
    try:
        r   = get_redis()
        key = _APPROVAL_NOTIFIED_PREFIX + ticket_key
        if r.exists(key):
            return False
        r.set(key, "1", ex=7 * 86400)
        return True
    except Exception:
        return True


def _make_approval_card(ticket_key: str, instance_name: str, gpu_count: str,
                         duration_hours: str, purpose: str, requester_name: str,
                         requester_open_id: str, requester_chat_id: str) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "⏳ GPU 大规模申请待审批"},
                   "template": "orange"},
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**申请人**\n{requester_name}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**工单**\n{ticket_key}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**实例**\n{instance_name}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**规格**\n{gpu_count}GPU · {duration_hours}h"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**费用预估**\n{_cost_str(int(gpu_count), float(duration_hours))}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**用途**\n{purpose}"}},
                ],
            },
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ 批准"},
                        "type": "primary",
                        "value": {"action": "approve_gpu", "ticket_key": ticket_key,
                                  "requester_open_id": requester_open_id,
                                  "requester_chat_id": requester_chat_id},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "❌ 拒绝"},
                        "type": "danger",
                        "value": {"action": "reject_gpu", "ticket_key": ticket_key,
                                  "requester_open_id": requester_open_id,
                                  "requester_chat_id": requester_chat_id},
                    },
                ],
            },
        ],
    }


# ── 实例就绪通知 ──────────────────────────────────────────────────────────────────

def _make_running_card(instance_id: str, instance_name: str,
                        ticket_key: str, gpu_count: int, duration_hours: float) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "🟢 GPU 实例已就绪"},
                   "template": "green"},
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**实例名称**\n{instance_name}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**实例 ID**\n`{instance_id}`"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**工单**\n{ticket_key}"}},
                    {"is_short": True, "text": {"tag": "lark_md",
                     "content": f"**费用预估**\n{_cost_str(gpu_count, duration_hours)}"}},
                ],
            },
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "实例已进入 **Running** 状态，可在 PAI DSW 控制台打开 JupyterLab。\n"
                           "到期前 15 分钟会提醒续期。"}},
        ],
    }


def _poll_until_running(ticket_key: str, instance_id: str, instance_name: str,
                         open_id: str, chat_id: str,
                         gpu_count: int, duration_hours: float,
                         ak_id: str, ak_secret: str) -> None:
    """后台轮询实例状态，Running 后推送就绪通知（最多等 6 分钟）。"""
    for _ in range(36):
        time.sleep(10)
        try:
            result = manage_pai_dsw(action="get", instance_id=instance_id,
                                    _ak_id=ak_id, _ak_secret=ak_secret)
            if "Running" in result:
                card = _make_running_card(instance_id, instance_name, ticket_key,
                                          gpu_count, duration_hours)
                _send_card(open_id, chat_id, card)
                print(f"[Scheduler] 实例 {instance_id} Running，就绪通知已推送。")
                return
            if any(s in result for s in ("Failed", "Stopped", "Deleted")):
                _send_text(open_id, chat_id,
                    f"⚠️ 实例 {instance_name}（{instance_id}）启动异常，请检查 PAI DSW 控制台。")
                return
        except Exception as e:
            print(f"[Scheduler] 轮询实例状态失败: {e}")
    _send_text(open_id, chat_id,
        f"⚠️ 实例 {instance_name}（{instance_id}）启动超时（>6min），请检查 PAI DSW 控制台。")


# ── GPU 空转检测 ──────────────────────────────────────────────────────────────────

def _get_instance_gpu_util(instance_name: str) -> Optional[float]:
    """从 Prometheus 查询 DSW 实例即时 GPU SM 利用率（%），不可达返回 None。"""
    try:
        from tools.prometheus_tool import _query_instant
        promql  = f'avg(AliyunPaidsw_INSTANCE_GPU_SM_UTIL{{instanceName="{instance_name}"}})'
        results = _query_instant(promql)
        if results and results[0].get("value"):
            return float(results[0]["value"][1])
    except Exception:
        pass
    return None


# ── 每日早报 ──────────────────────────────────────────────────────────────────────

def _send_morning_report() -> None:
    """向所有有运行实例的用户推送今日到期汇总。"""
    now          = time.time()
    today_secs   = 24 * 3600
    user_map: dict[str, dict] = {}

    for ticket_key in _all_tracked_keys():
        state = _redis_get(ticket_key)
        if not state:
            continue
        open_id   = state.get("open_id", "")
        chat_id   = state.get("chat_id", "")
        start_ts  = float(state.get("start_ts", now))
        dur_h     = float(state.get("duration_hours", 8))
        expire_ts = start_ts + dur_h * 3600
        remaining = (expire_ts - now) / 3600

        uid = open_id or chat_id
        if uid not in user_map:
            user_map[uid] = {"open_id": open_id, "chat_id": chat_id, "instances": []}
        user_map[uid]["instances"].append({
            "name":         state.get("instance_name", "-"),
            "ticket_key":   ticket_key,
            "remaining_h":  remaining,
            "expires_today": (expire_ts - now) <= today_secs,
        })

    for info in user_map.values():
        insts    = info["instances"]
        expiring = [i for i in insts if i["expires_today"]]
        lines    = [f"☀️ 早安！你有 **{len(insts)}** 个运行中的 GPU 实例：\n"]
        for i in sorted(insts, key=lambda x: x["remaining_h"]):
            tag = " ⚠️ **今日到期**" if i["expires_today"] else ""
            lines.append(
                f"- **{i['name']}**（{i['ticket_key']}）  剩余 {max(0, i['remaining_h']):.1f}h{tag}"
            )
        if expiring:
            lines.append("\n⏰ 今日到期实例请及时续期或停止，避免数据丢失。")
        _send_text(info["open_id"], info["chat_id"], "\n".join(lines))

    print(f"[Scheduler] 早报已发送（{len(user_map)} 位用户）。")


def _all_tracked_keys() -> list[str]:
    try:
        r = get_redis()
        return [k.replace(_KEY_PREFIX, "") for k in r.scan_iter(f"{_KEY_PREFIX}*")]
    except Exception:
        return []


# ── Feishu helpers（延迟导入避免循环依赖）──────────────────────────────────────

def _send_card(open_id: str, chat_id: str, card: dict) -> None:
    """向用户发送飞书交互卡片（优先私信，降级群聊）。"""
    try:
        import requests
        from tools.feishu_tool import _get_access_token
        token = _get_access_token()
        target_id = open_id or chat_id
        id_type   = "open_id" if open_id else "chat_id"
        requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            params={"receive_id_type": id_type},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": target_id,
                "msg_type":   "interactive",
                "content":    json.dumps(card),
            },
            timeout=15,
        )
    except Exception as e:
        print(f"[Scheduler] 飞书推送失败: {e}")


def _send_text(open_id: str, chat_id: str, text: str) -> None:
    try:
        import requests
        from tools.feishu_tool import _get_access_token
        token = _get_access_token()
        target_id = open_id or chat_id
        id_type   = "open_id" if open_id else "chat_id"
        requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            params={"receive_id_type": id_type},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": target_id,
                "msg_type":   "text",
                "content":    json.dumps({"text": text}),
            },
            timeout=15,
        )
    except Exception as e:
        print(f"[Scheduler] 飞书文本推送失败: {e}")


# ── 卡片构造 ──────────────────────────────────────────────────────────────────

def _make_instance_created_card(instance_id: str, instance_name: str,
                                 ticket_key: str, duration_hours: str) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "✅ GPU 实例已创建"},
            "template": "green",
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**实例名称**\n{instance_name}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**实例 ID**\n`{instance_id}`"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**工单**\n{ticket_key}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**有效时长**\n{duration_hours} 小时"}},
                ],
            },
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "实例正在启动，预计 2-3 分钟后可用。到期前 15 分钟将提醒续期。"}},
        ],
    }


def _make_idle_warn_card(instance_id: str, instance_name: str,
                          ticket_key: str, stop_minutes: int) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "⏰ GPU 实例即将超时"},
            "template": "orange",
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md",
                    "content": (
                        f"实例 **{instance_name}** (`{instance_id}`) 使用时长已达上限。\n"
                        f"**{stop_minutes} 分钟内无操作将自动停止实例**（数据保留，可重新启动）。"
                    )},
            },
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "⏩ 延续使用 2 小时"},
                        "type": "primary",
                        "value": {
                            "action": "extend_dsw",
                            "instance_id": instance_id,
                            "ticket_key": ticket_key,
                            "extend_hours": "2",
                        },
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🛑 立即停止"},
                        "type": "danger",
                        "value": {
                            "action": "stop_dsw",
                            "instance_id": instance_id,
                            "ticket_key": ticket_key,
                        },
                    },
                ],
            },
        ],
    }


# ── 核心调度逻辑 ──────────────────────────────────────────────────────────────

def _process_new_ticket(ticket: dict) -> None:
    """处理一张新的 GPU 申请工单：创建实例并更新状态。"""
    key = ticket["key"]
    fields = ticket.get("fields", {})
    description = fields.get("description", "")
    meta = parse_ticket_metadata(description)

    instance_name  = meta.get("dsw_instance_name", f"dsw-{key.lower()}")
    gpu_count      = meta.get("dsw_gpu_count", "1")
    duration_hours = meta.get("dsw_duration_hours", "8")
    _raw_image     = meta.get("dsw_image", "")
    image_name     = _raw_image if _raw_image.startswith(("dsw-registry", "registry", "pai-image", "ego-pretrain")) else settings.PAI_DSW_DEFAULT_IMAGE
    open_id        = meta.get("feishu_open_id", "")
    chat_id        = meta.get("feishu_chat_id", settings.FEISHU_CHAT_ID)

    # 防止重复处理（Redis 里已有记录则跳过）
    if _redis_get(key):
        return

    # ── 强制要求申请人已注册个人 AK/SK（open_id 为空视为管理员手动工单，豁免）──
    if open_id:
        from core.feishu_bot import _is_registered
        if not _is_registered(open_id):
            if _mark_pending_reg(key):
                print(f"[Scheduler] 工单 {key} 申请人未注册 AK/SK，暂缓创建。")
                add_comment(key, "⚠️ 申请人尚未注册阿里云 AK/SK，实例创建暂缓。注册后调度器将在 20 秒内自动重试。")
                _send_text(open_id, chat_id,
                    f"⚠️ 工单 {key} 暂缓处理：请先向 Bot 私信注册你的阿里云 AK/SK，注册成功后自动创建实例。\n\n"
                    "注册格式（请在私聊中发送，保障密钥安全）：\n"
                    "```\n注册AK: AccessKeyId AccessKeySecret\n```")
            else:
                print(f"[Scheduler] 工单 {key} 申请人未注册（已通知，等待注册）。")
            return

    # ── 审批流：大规模申请需管理员确认 ────────────────────────────────────────────
    needs_approval = meta.get("dsw_needs_approval", "").lower() == "true"
    if needs_approval and not _is_approved(key):
        admin_open_id = settings.ADMIN_FEISHU_OPEN_ID
        if admin_open_id:
            if _mark_approval_notified(key):
                print(f"[Scheduler] 工单 {key} 需审批，发审批卡片给管理员。")
                card = _make_approval_card(
                    key, instance_name, gpu_count, duration_hours,
                    meta.get("dsw_purpose", "未说明"),
                    meta.get("dsw_requester_name", open_id),
                    open_id, chat_id,
                )
                _send_card(admin_open_id, "", card)
                add_comment(key, "⏳ 已发送审批通知，等待管理员确认后自动创建实例。")
            else:
                print(f"[Scheduler] 工单 {key} 等待管理员审批中。")
        else:
            # 无管理员配置 → 自动批准
            _set_approved(key)
            print(f"[Scheduler] ADMIN_FEISHU_OPEN_ID 未配置，工单 {key} 自动批准。")
        if not _is_approved(key):
            return

    print(f"[Scheduler] 处理新工单 {key}: {instance_name} x{gpu_count}GPU {duration_hours}h")

    from core.feishu_bot import _get_user_ak
    user_ak_id, user_ak_secret = _get_user_ak(open_id)
    if open_id:
        print(f"[Scheduler] 使用申请人个人 AK 创建实例（open_id={open_id}）")
    else:
        print(f"[Scheduler] open_id 为空（管理员工单），使用全局账号创建实例")

    # 构造 create_json 配置
    cpu_cores  = meta.get("dsw_cpu_cores", "") or str(int(gpu_count) * 8)
    memory_gb  = meta.get("dsw_memory_gb", "") or str(int(gpu_count) * 64)
    create_cfg = json.dumps({
        "instanceName": instance_name,
        "workspaceId": settings.PAI_DSW_WORKSPACE_ID,
        "resourceId": settings.PAI_DSW_RESOURCE_ID,
        "imageUrl": image_name,
        "requestedResource": {
            "gpu": int(gpu_count),
            "cpu": cpu_cores,
            "memory": f"{memory_gb}Gi",
            "gputype": "A10",
        },
    })

    result = manage_pai_dsw(action="create_json", create_config_json=create_cfg,
                            _ak_id=user_ak_id, _ak_secret=user_ak_secret)

    if "实例创建成功" in result or "instance_id" in result.lower():
        # 提取实例 ID
        instance_id = ""
        for line in result.splitlines():
            if "实例 ID" in line or "instance_id" in line.lower():
                parts = line.split("：") if "：" in line else line.split(":")
                if len(parts) > 1:
                    instance_id = parts[-1].strip().strip("`")
                    break

        now_ts      = time.time()
        gpu_count_i = int(gpu_count)
        dur_h_f     = float(duration_hours)
        _redis_set(key, {
            "instance_id":    instance_id,
            "instance_name":  instance_name,
            "open_id":        open_id,
            "chat_id":        chat_id,
            "start_ts":       now_ts,
            "duration_hours": dur_h_f,
            "gpu_count":      gpu_count_i,
            "warned":         False,
            "warn_ts":        0.0,
            "idle_since":     0.0,
            "idle_warned":    False,
        })

        # 记录配额使用量
        add_quota_usage(open_id, gpu_count_i, dur_h_f)

        # 更新 Jira 状态 + 评论
        transition_ticket(key, "完成")
        add_comment(key, f"DSW 实例已自动创建。\n实例 ID：{instance_id}\n{result}")

        # 飞书通知申请人（创建中）
        card = _make_instance_created_card(instance_id, instance_name, key, duration_hours)
        _send_card(open_id, chat_id, card)

        # 后台轮询，Running 后推送就绪通知
        import threading as _th
        _th.Thread(
            target=_poll_until_running,
            args=(key, instance_id, instance_name, open_id, chat_id,
                  gpu_count_i, dur_h_f, user_ak_id, user_ak_secret),
            daemon=True,
        ).start()

        print(f"[Scheduler] 实例已创建 {instance_id}，就绪轮询已启动。")
    else:
        add_comment(key, f"⚠️ 自动创建实例失败，请人工处理。\n错误：{result}")
        _send_text(open_id, chat_id,
                   f"⚠️ 工单 {key} 的 GPU 实例自动创建失败，运维人员会尽快处理。\n详情：{result}")
        print(f"[Scheduler] 工单 {key} 创建实例失败: {result}")


def _check_running_instances() -> None:
    """检查所有跟踪中的实例是否超时，发送警告或自动停止。"""
    now = time.time()
    stop_seconds = settings.DSW_IDLE_STOP_MINUTES * 60
    warn_advance = 15 * 60  # 到期前 15 分钟发警告

    for ticket_key in _all_tracked_keys():
        state = _redis_get(ticket_key)
        if not state:
            continue

        instance_id   = state.get("instance_id", "")
        instance_name = state.get("instance_name", "")
        open_id       = state.get("open_id", "")
        chat_id       = state.get("chat_id", "")
        start_ts      = float(state.get("start_ts", now))
        duration_h    = float(state.get("duration_hours", 8))
        warned        = state.get("warned", False)
        warn_ts       = float(state.get("warn_ts", 0))

        expire_ts = start_ts + duration_h * 3600
        remaining = expire_ts - now

        if not warned and remaining <= warn_advance:
            # 发警告卡片
            print(f"[Scheduler] 实例 {instance_id} 还有 {int(remaining/60)} 分钟到期，发送警告。")
            card = _make_idle_warn_card(instance_id, instance_name,
                                        ticket_key, settings.DSW_IDLE_STOP_MINUTES)
            _send_card(open_id, chat_id, card)
            state["warned"] = True
            state["warn_ts"] = now
            _redis_set(ticket_key, state)

        elif warned and (now - warn_ts) >= stop_seconds:
            # 警告后超时 → 自动停止
            print(f"[Scheduler] 实例 {instance_id} 警告超时，自动停止。")
            stop_result = manage_pai_dsw(action="stop", instance_id=instance_id)
            elapsed_h   = (now - float(state.get("start_ts", now))) / 3600
            gpu_cnt     = int(state.get("gpu_count", 1))
            _send_text(open_id, chat_id,
                       f"🛑 实例 {instance_name}（{instance_id}）已自动停止。\n"
                       f"实际使用 {elapsed_h:.1f}h，费用 {_cost_str(gpu_cnt, elapsed_h)}\n"
                       f"数据已保留，如需继续使用可重新启动。工单：{ticket_key}")
            transition_ticket(ticket_key, "完成")
            add_comment(ticket_key, f"实例已自动停止（超时无响应）。\n{stop_result}")
            _redis_delete(ticket_key)

        # ── GPU 空转检测（仅在超时警告前执行）────────────────────────────────────
        if not warned and instance_name:
            gpu_util = _get_instance_gpu_util(instance_name)
            if gpu_util is not None:
                threshold  = settings.GPU_IDLE_THRESHOLD_PCT
                idle_since = float(state.get("idle_since", 0))
                if gpu_util < threshold:
                    if idle_since == 0:
                        state["idle_since"] = now
                        _redis_set(ticket_key, state)
                    elif (now - idle_since) >= settings.GPU_IDLE_WARN_MINUTES * 60:
                        if not state.get("idle_warned"):
                            print(f"[Scheduler] 实例 {instance_id} GPU 空转，告警。")
                            _send_text(open_id, chat_id,
                                f"⚠️ GPU 空转提醒：实例 **{instance_name}** GPU 利用率 {gpu_util:.1f}%，"
                                f"已持续 {settings.GPU_IDLE_WARN_MINUTES} 分钟低于 "
                                f"{settings.GPU_IDLE_THRESHOLD_PCT:.0f}%。\n"
                                "若任务已完成请及时停止实例以节省费用。")
                            state["idle_warned"] = True
                            _redis_set(ticket_key, state)
                else:
                    if state.get("idle_since") or state.get("idle_warned"):
                        state["idle_since"]  = 0.0
                        state["idle_warned"] = False
                        _redis_set(ticket_key, state)


# ── 调度器主类 ────────────────────────────────────────────────────────────────

class DSWScheduler:
    """后台调度器，随 Flask Bot 一起启动。"""

    TICKET_POLL_INTERVAL = 20    # 秒：轮询 Jira 间隔
    INSTANCE_CHECK_INTERVAL = 300  # 秒：检查实例超时间隔

    def __init__(self):
        self._running = False
        self._ticket_thread:  threading.Thread | None = None
        self._instance_thread: threading.Thread | None = None
        self._morning_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ticket_thread = threading.Thread(
            target=self._ticket_loop, name="jira-poll", daemon=True
        )
        self._instance_thread = threading.Thread(
            target=self._instance_loop, name="dsw-check", daemon=True
        )
        self._morning_thread = threading.Thread(
            target=self._morning_loop, name="morning-report", daemon=True
        )
        self._ticket_thread.start()
        self._instance_thread.start()
        self._morning_thread.start()
        print("[Scheduler] 启动：Jira 轮询 + DSW 超时监控 + GPU 空转检测 + 每日早报")

    def stop(self) -> None:
        self._running = False

    def _ticket_loop(self) -> None:
        time.sleep(3)   # 等 Flask 初始化
        while self._running:
            try:
                tickets = get_gpu_tickets(status="待办")
                for ticket in tickets:
                    try:
                        _process_new_ticket(ticket)
                    except Exception as e:
                        print(f"[Scheduler] 处理工单 {ticket.get('key')} 失败: {e}")
            except Exception as e:
                print(f"[Scheduler] Jira 轮询失败: {e}")
            time.sleep(self.TICKET_POLL_INTERVAL)

    def _instance_loop(self) -> None:
        time.sleep(30)
        while self._running:
            try:
                _check_running_instances()
            except Exception as e:
                print(f"[Scheduler] 实例检查失败: {e}")
            time.sleep(self.INSTANCE_CHECK_INTERVAL)

    def _morning_loop(self) -> None:
        from datetime import datetime, timedelta
        while self._running:
            now_dt = datetime.now()
            target = now_dt.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_dt >= target:
                target += timedelta(days=1)
            wait   = (target - now_dt).total_seconds()
            slept  = 0.0
            while slept < wait and self._running:
                chunk = min(60.0, wait - slept)
                time.sleep(chunk)
                slept += chunk
            if self._running:
                try:
                    _send_morning_report()
                except Exception as e:
                    print(f"[Scheduler] 早报发送失败: {e}")


# 全局单例
scheduler = DSWScheduler()
