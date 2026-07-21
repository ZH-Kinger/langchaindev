"""
DSW 工单调度器。

职责：
  1. 每 2 分钟轮询 Jira，发现新 GPU 申请工单 → 自动创建 DSW 实例
  2. 每 5 分钟检查运行中的实例：
     - GPU 空转（利用率低于阈值）→ 飞书提醒用户手动停止
     - 到期警告 + 到期自动停止：默认屏蔽（DSW_IDLE_STOP_ENABLED=False），
       实例利用率关机交给阿里云工作空间内置设置，不再重复造轮子；
       显式开启才恢复「即将到期」警告卡 + 无响应自动 stop。

状态存储在 Redis：
  dsw:ticket:{ticket_key}  →  JSON 字符串，含：
    instance_id, open_id, chat_id,
    start_ts, duration_hours,
    warned (bool), warn_ts
"""
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.settings import settings
from utils.logger import get_logger
from tools.feishu.cards import btn, buttons, card, div, fields, hr

logger = get_logger(__name__)

# 容器跑在 UTC，但所有「每日 N 点」都按北京时间。中国无夏令时，固定 UTC+8 最稳（不依赖 tzdata）。
_BEIJING = timezone(timedelta(hours=8))


def _bj_now() -> datetime:
    return datetime.now(_BEIJING)


def _sleep_until(target: datetime, running) -> bool:
    """分块睡到 target（北京时区 aware）。中途 running() 变 False 则提前返回 False。"""
    wait, slept = (target - _bj_now()).total_seconds(), 0.0
    while slept < wait and running():
        time.sleep(min(60.0, wait - slept))
        slept += min(60.0, wait - slept)
    return running()
from tools.jira.ticket import (
    get_gpu_tickets, add_comment, transition_ticket,
    parse_ticket_metadata, update_ticket_field,
)
from tools.aliyun.pai_dsw import manage_pai_dsw
from utils.redis_client import get_redis


# ── Redis helpers ─────────────────────────────────────────────────────────────

_KEY_PREFIX = "dsw:ticket:"


def _redis_set(ticket_key: str, data: dict) -> None:
    try:
        r = get_redis()
        r.set(_KEY_PREFIX + ticket_key, json.dumps(data), ex=7 * 86400)
    except Exception as e:
        logger.error("[Scheduler] Redis set 失败", exc_info=True)


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
        logger.warning("[Quota] 记录使用量失败: %s", e)


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
    return card("⏳ GPU 大规模申请待审批", [
        fields(
            ("申请人", requester_name),
            ("工单", ticket_key),
            ("实例", instance_name),
            ("规格", f"{gpu_count}GPU · {duration_hours}h"),
            ("费用预估", _cost_str(int(gpu_count), float(duration_hours))),
            ("用途", purpose),
        ),
        hr(),
        buttons(
            btn("✅ 批准", {"action": "approve_gpu", "ticket_key": ticket_key,
                           "requester_open_id": requester_open_id,
                           "requester_chat_id": requester_chat_id}, "primary"),
            btn("❌ 拒绝", {"action": "reject_gpu", "ticket_key": ticket_key,
                           "requester_open_id": requester_open_id,
                           "requester_chat_id": requester_chat_id}, "danger"),
        ),
    ], color="orange")


# ── 实例就绪通知 ──────────────────────────────────────────────────────────────────

def _make_running_card(instance_id: str, instance_name: str,
                        ticket_key: str, gpu_count: int, duration_hours: float) -> dict:
    return card("🟢 GPU 实例已就绪", [
        fields(
            ("实例名称", instance_name),
            ("实例 ID", f"`{instance_id}`"),
            ("工单", ticket_key),
            ("费用预估", _cost_str(gpu_count, duration_hours)),
        ),
        hr(),
        div("实例已进入 **Running** 状态，可在 PAI DSW 控制台打开 JupyterLab。\n"
            "到期前 15 分钟会提醒续期。"),
    ], color="green")


def _poll_until_running(ticket_key: str, instance_id: str, instance_name: str,
                         open_id: str, chat_id: str,
                         gpu_count: int, duration_hours: float) -> None:
    """后台轮询实例状态，Running 后推送就绪通知（最多等 6 分钟）。"""
    for _ in range(36):
        time.sleep(10)
        try:
            result = manage_pai_dsw(action="get", instance_id=instance_id,
                                    open_id=open_id)
            if "Running" in result:
                card = _make_running_card(instance_id, instance_name, ticket_key,
                                          gpu_count, duration_hours)
                _send_card(open_id, chat_id, card)
                logger.info("[Scheduler] 实例 %s Running，就绪通知已推送", instance_id)
                return
            if any(s in result for s in ("Failed", "Stopped", "Deleted")):
                _send_text(open_id, chat_id,
                    f"⚠️ 实例 {instance_name}（{instance_id}）启动异常，请检查 PAI DSW 控制台。")
                return
        except Exception as e:
            logger.error("[Scheduler] 轮询实例状态失败", exc_info=True)
    _send_text(open_id, chat_id,
        f"⚠️ 实例 {instance_name}（{instance_id}）启动超时（>6min），请检查 PAI DSW 控制台。")


# ── GPU 空转检测 ──────────────────────────────────────────────────────────────────

def _get_instance_gpu_util(instance_name: str) -> Optional[float]:
    """从 Prometheus 查询 DSW 实例即时 GPU SM 利用率（%），不可达返回 None。"""
    try:
        from tools.aliyun.prometheus import _query_instant
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

    logger.info("[Scheduler] 早报已发送（%s 位用户）", len(user_map))


def _send_cluster_morning_report() -> None:
    """每日集群算力效率（MFU）早报：聚合全集群 MFU 并推送到飞书群。"""
    if not settings.CLUSTER_MORNING_REPORT_ENABLED:
        return
    if not settings.PROMETHEUS_URL:
        logger.info("[Scheduler] PROMETHEUS_URL 未配置，跳过集群 MFU 早报")
        return
    chat_id = settings.FEISHU_CHAT_ID
    if not chat_id:
        logger.info("[Scheduler] FEISHU_CHAT_ID 未配置，跳过集群 MFU 早报")
        return
    try:
        from tools.aliyun.cluster_mfu import build_mfu_card
        card = build_mfu_card(view="summary", refresh=True)   # 刷新快照并暖缓存，供卡片按钮切换
        _send_card("", chat_id, card)
        logger.info("[Scheduler] 集群 MFU 早报已推送")
    except Exception:
        logger.error("[Scheduler] 集群 MFU 早报失败", exc_info=True)


def _run_oss_perm_audit_push() -> None:
    """OSS 权限每日对账：有待同步/孤儿则把带「批准」按钮的卡片推到群。"""
    if not settings.OSS_PERM_PUSH_ENABLED:
        return
    try:
        from core.oss_perm import permsync
        from core.oss_perm.cards import audit_form_card
        members, combos = permsync.load_members()
        plan = permsync.build_plan(members, combos)
        active = {m["username"] for m in members if m["username"]}
        diff = permsync.audit_diff(plan, active)
        if diff["n_diff"] == 0:
            logger.info("[OSSPerm] 权限与表格一致，跳过推送")
            return
        _send_card("", settings.FEISHU_CHAT_ID, audit_form_card(diff))
        logger.info("[OSSPerm] 已推送对账表单卡：%d 人待同步", diff["n_diff"])
    except Exception as e:
        logger.error("[OSSPerm] 对账推送失败: %s", e, exc_info=True)


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
        from tools.feishu.notify import _get_access_token
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
        logger.error("[Scheduler] 飞书推送失败", exc_info=True)


def _send_text(open_id: str, chat_id: str, text: str) -> None:
    try:
        import requests
        from tools.feishu.notify import _get_access_token
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
        logger.error("[Scheduler] 飞书文本推送失败", exc_info=True)


# ── 卡片构造 ──────────────────────────────────────────────────────────────────

def _make_instance_created_card(instance_id: str, instance_name: str,
                                 ticket_key: str, duration_hours: str) -> dict:
    return card("✅ GPU 实例已创建", [
        fields(
            ("实例名称", instance_name),
            ("实例 ID", f"`{instance_id}`"),
            ("工单", ticket_key),
            ("有效时长", f"{duration_hours} 小时"),
        ),
        hr(),
        div("实例正在启动，预计 2-3 分钟后可用。到期前 15 分钟将提醒续期。"),
    ], color="green")


def _make_idle_warn_card(instance_id: str, instance_name: str,
                          ticket_key: str, stop_minutes: int) -> dict:
    return card("⏰ GPU 实例即将超时", [
        div(f"实例 **{instance_name}** (`{instance_id}`) 使用时长已达上限。\n"
            f"**{stop_minutes} 分钟内无操作将自动停止实例**（数据保留，可重新启动）。"),
        hr(),
        buttons(
            btn("⏩ 延续使用 2 小时", {"action": "extend_dsw", "instance_id": instance_id,
                                      "ticket_key": ticket_key, "extend_hours": "2"}, "primary"),
            btn("🛑 立即停止", {"action": "stop_dsw", "instance_id": instance_id,
                               "ticket_key": ticket_key}, "danger"),
        ),
    ], color="orange")


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

    # ── 强制要求申请人已建立飞书↔RAM 映射（open_id 为空视为管理员手动工单，豁免）──
    if open_id:
        from core.feishu_bot import _is_registered
        if not _is_registered(open_id):
            if _mark_pending_reg(key):
                logger.info("[Scheduler] 工单 %s 申请人无 RAM 映射，暂缓创建", key)
                add_comment(key, "⚠️ 申请人尚未建立飞书↔RAM 用户映射，实例创建暂缓。映射建立后调度器将在 20 秒内自动重试。")
                _send_text(open_id, chat_id,
                    f"⚠️ 工单 {key} 暂缓处理：未找到你对应的 RAM 用户。\n\n"
                    "可能原因：\n"
                    "  · 飞书显示名与阿里云 RAM 用户的 displayName 不一致\n"
                    "  · 该 RAM 用户不存在或被禁用\n\n"
                    "请联系运维人员手动绑定，或在飞书 Bot 发送：\n"
                    "```\n绑定RAM: <你的RAM用户名>\n```")
            else:
                logger.info("[Scheduler] 工单 %s 申请人无映射（已通知，等待绑定）", key)
            return

    # ── 审批流：大规模申请需管理员确认 ────────────────────────────────────────────
    needs_approval = meta.get("dsw_needs_approval", "").lower() == "true"
    if needs_approval and not _is_approved(key):
        admin_open_id = settings.ADMIN_FEISHU_OPEN_ID
        if admin_open_id:
            if _mark_approval_notified(key):
                logger.info("[Scheduler] 工单 %s 需审批，发审批卡片给管理员", key)
                card = _make_approval_card(
                    key, instance_name, gpu_count, duration_hours,
                    meta.get("dsw_purpose", "未说明"),
                    meta.get("dsw_requester_name", open_id),
                    open_id, chat_id,
                )
                _send_card(admin_open_id, "", card)
                add_comment(key, "⏳ 已发送审批通知，等待管理员确认后自动创建实例。")
            else:
                logger.debug("[Scheduler] 工单 %s 等待管理员审批中", key)
        else:
            # 无管理员配置 → 自动批准
            _set_approved(key)
            logger.warning("[Scheduler] ADMIN_FEISHU_OPEN_ID 未配置，工单 %s 自动批准", key)
        if not _is_approved(key):
            return

    logger.info("[Scheduler] 处理新工单 %s: %s x%sGPU %sh", key, instance_name, gpu_count, duration_hours)

    if open_id:
        logger.info("[Scheduler] 通过 STS AssumeRole 创建实例 open_id=%s", open_id)
    else:
        logger.info("[Scheduler] open_id 为空（管理员工单），使用全局账号创建实例")

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

    # open_id 非空时走 STS（按申请人临时凭证创建，实例归属正确）；为空降级用全局 AK
    result = manage_pai_dsw(action="create_json", create_config_json=create_cfg,
                            open_id=open_id)

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
                  gpu_count_i, dur_h_f),
            daemon=True,
        ).start()

        logger.info("[Scheduler] 实例已创建 %s，就绪轮询已启动", instance_id)
    else:
        add_comment(key, f"⚠️ 自动创建实例失败，请人工处理。\n错误：{result}")
        _send_text(open_id, chat_id,
                   f"⚠️ 工单 {key} 的 GPU 实例自动创建失败，运维人员会尽快处理。\n详情：{result}")
        logger.error("[Scheduler] 工单 %s 创建实例失败: %s", key, result)


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

        # ── 到期警告 + 到期自动关机：默认屏蔽（DSW_IDLE_STOP_ENABLED=False）──────────
        # 实例利用率关机交给阿里云工作空间的内置设置，不再重复造轮子。
        # 只有显式开启才恢复「即将到期」警告卡 + 无响应自动 stop 实例。
        if settings.DSW_IDLE_STOP_ENABLED:
            if not warned and remaining <= warn_advance:
                # 发警告卡片
                logger.info("[Scheduler] 实例 %s 还有 %s 分钟到期，发送警告", instance_id, int(remaining/60))
                card = _make_idle_warn_card(instance_id, instance_name,
                                            ticket_key, settings.DSW_IDLE_STOP_MINUTES)
                _send_card(open_id, chat_id, card)
                state["warned"] = True
                state["warn_ts"] = now
                _redis_set(ticket_key, state)

            elif warned and (now - warn_ts) >= stop_seconds:
                # 警告后超时 → 自动停止
                logger.info("[Scheduler] 实例 %s 警告超时，自动停止", instance_id)
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

        # ── GPU 空转检测（保留：仅提醒、不关机；不受 DSW_IDLE_STOP_ENABLED 影响）──────
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
                            logger.info("[Scheduler] 实例 %s GPU 空转，告警", instance_id)
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
    DATAFLOW_RECONCILE_INTERVAL = 120  # 秒：数据流动/迁移在途任务对账间隔

    def __init__(self):
        self._running = False
        self._ticket_thread:  threading.Thread | None = None
        self._instance_thread: threading.Thread | None = None
        self._morning_thread: threading.Thread | None = None
        self._capacity_thread: threading.Thread | None = None
        self._oss_perm_thread: threading.Thread | None = None
        self._reconcile_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Jira 停用时不启动工单轮询线程（否则会去打已停用的 Jira、并可能重复建 DSW/评论工单）。
        jira_on = settings.JIRA_ENABLED
        if jira_on:
            self._ticket_thread = threading.Thread(
                target=self._ticket_loop, name="jira-poll", daemon=True
            )
        self._instance_thread = threading.Thread(
            target=self._instance_loop, name="dsw-check", daemon=True
        )
        self._morning_thread = threading.Thread(
            target=self._morning_loop, name="morning-report", daemon=True
        )
        if jira_on:
            self._ticket_thread.start()
        self._instance_thread.start()
        self._morning_thread.start()
        extra = ""
        if settings.CAPACITY_MONITOR_ENABLED:
            self._capacity_thread = threading.Thread(
                target=self._capacity_loop, name="capacity-monitor", daemon=True
            )
            self._capacity_thread.start()
            extra = " + 容量巡检"
        if settings.OSS_PERM_PUSH_ENABLED:
            self._oss_perm_thread = threading.Thread(
                target=self._oss_perm_loop, name="oss-perm-push", daemon=True
            )
            self._oss_perm_thread.start()
            extra += " + OSS权限对账推送"
        if getattr(settings, "DATASET_DASHBOARD_ENABLED", False):
            self._dataset_dash_thread = threading.Thread(
                target=self._dataset_dashboard_loop, name="dataset-dashboard", daemon=True
            )
            self._dataset_dash_thread.start()
            extra += " + 数据集大盘维护"
        if getattr(settings, "DATAFLOW_RECONCILE_ENABLED", True):
            self._reconcile_thread = threading.Thread(
                target=self._dataflow_reconcile_loop, name="dataflow-reconcile", daemon=True
            )
            self._reconcile_thread.start()
            extra += " + 数据流动/迁移对账"
        idle_stop = "DSW到期自动停止 + " if settings.DSW_IDLE_STOP_ENABLED else ""
        jira_seg = "Jira 轮询 + " if jira_on else "(Jira 停用) "
        logger.info("[Scheduler] 启动：%s%sGPU 空转提醒 + 每日早报(实例+集群)%s", jira_seg, idle_stop, extra)

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
                        logger.error("[Scheduler] 处理工单 %s 失败", ticket.get("key"), exc_info=True)
            except Exception as e:
                logger.error("[Scheduler] Jira 轮询失败", exc_info=True)
            time.sleep(self.TICKET_POLL_INTERVAL)

    def _instance_loop(self) -> None:
        time.sleep(30)
        while self._running:
            try:
                _check_running_instances()
            except Exception as e:
                logger.error("[Scheduler] 实例检查失败", exc_info=True)
            time.sleep(self.INSTANCE_CHECK_INTERVAL)

    def _morning_loop(self) -> None:
        """每日北京时间 9:00 推送实例 + 集群早报。"""
        while self._running:
            now = _bj_now()
            target = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            if not _sleep_until(target, lambda: self._running):
                break
            try:
                _send_morning_report()
            except Exception:
                logger.error("[Scheduler] 早报发送失败", exc_info=True)
            try:
                _send_cluster_morning_report()
            except Exception:
                logger.error("[Scheduler] 集群监控早报发送失败", exc_info=True)

    def _capacity_loop(self) -> None:
        """容量巡检：对齐到北京整点的 CAPACITY_MONITOR_INTERVAL_HOURS 倍数（如 6h → 0/6/12/18 点）。"""
        from core.capacity_monitor import run_capacity_scan
        step = max(1, min(24, int(settings.CAPACITY_MONITOR_INTERVAL_HOURS)))
        while self._running:
            now = _bj_now()
            nh = ((now.hour // step) + 1) * step          # 下一个对齐整点
            if nh >= 24:
                target = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                target = now.replace(hour=nh, minute=0, second=0, microsecond=0)
            if not _sleep_until(target, lambda: self._running):
                break
            try:
                run_capacity_scan()
            except Exception:
                logger.error("[Scheduler] 容量巡检失败", exc_info=True)

    def _dataset_dashboard_loop(self) -> None:
        """数据集大盘维护：对齐到北京整点的 DATASET_DASHBOARD_INTERVAL_HOURS 倍数（同容量巡检对齐法）。"""
        from core.dataset_dashboard import run_once
        step = max(1, min(24, int(getattr(settings, "DATASET_DASHBOARD_INTERVAL_HOURS", 24))))
        while self._running:
            now = _bj_now()
            nh = ((now.hour // step) + 1) * step
            if nh >= 24:
                target = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                target = now.replace(hour=nh, minute=0, second=0, microsecond=0)
            if not _sleep_until(target, lambda: self._running):
                break
            try:
                run_once()
            except Exception:
                logger.error("[Scheduler] 数据集大盘维护失败", exc_info=True)

    def _dataflow_reconcile_loop(self) -> None:
        """每 2 分钟对账在途数据流动/迁移任务：孤儿(重启后线程已死)完成时自动补推结果卡。
        调度器随容器一起重启复活，故此循环本身抗重启——真正做到“跑完必通知”。"""
        time.sleep(45)   # 等 Flask + 在线线程先起来
        while self._running:
            try:
                _reconcile_dataflow_once()
            except Exception:
                logger.error("[Scheduler] 数据流动/迁移对账失败", exc_info=True)
            time.sleep(self.DATAFLOW_RECONCILE_INTERVAL)

    def _oss_perm_loop(self) -> None:
        """每日北京时间 OSS_PERM_PUSH_HOUR:20 对账，把待同步权限推到群（带批准按钮）。"""
        hour = max(0, min(23, settings.OSS_PERM_PUSH_HOUR))
        while self._running:
            now = _bj_now()
            target = now.replace(hour=hour, minute=20, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            if not _sleep_until(target, lambda: self._running):
                break
            try:
                _run_oss_perm_audit_push()
            except Exception:
                logger.error("[Scheduler] OSS 权限对账推送失败", exc_info=True)


def _dataflow_reconcile_specs() -> list[dict]:
    """在途数据流动/迁移任务的对账规格：命名空间 → (orchestrator, cards, 在途阶段, 目标群, 清理钩子)。

    每个 orchestrator 都有 _KEY_PREFIX / get_job / refresh / _save / result_card。
    refresh 只轮询、绝不重新提交任务，故对账是只读推进、幂等安全。
    **例外：ssh（两段链）** refresh→poll_once 在段1已完成的 STAGE1 孤儿上会自动起段2（一次远端
    rsync launch=写动作）——这正是"重启后续跑"的期望行为；靠 stale 门(180s)与在线线程错开 + NX
    闸门，无并发双起，rsync 增量幂等，安全。
    """
    from core.transfer import orchestrator as tr, cards as trc
    from core.cpfs_dataflow import orchestrator as cp, cards as cpc
    from core.vepfs_dataflow import orchestrator as ve, cards as vec
    from core.bucket_transfer import orchestrator as bk, cards as bkc
    from core.ssh_transfer import orchestrator as sh, cards as shc
    from core.pfs_transfer import orchestrator as pf, cards as pfc
    return [
        {"name": "transfer", "o": tr, "cards": trc, "active": {tr.STAGE_CROSSING},
         "chat": lambda: settings.TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID, "cleanup": None},
        {"name": "cpfs", "o": cp, "cards": cpc, "active": {cp.STAGE_RUNNING},
         "chat": lambda: settings.CPFS_CHAT_ID or settings.FEISHU_CHAT_ID,
         "cleanup": getattr(cp, "_cleanup_ephemeral", None)},   # cpfs 终态删临时 DataFlow
        {"name": "vepfs", "o": ve, "cards": vec, "active": {ve.STAGE_RUNNING},
         "chat": lambda: settings.VEPFS_CHAT_ID or settings.FEISHU_CHAT_ID, "cleanup": None},
        {"name": "bucket", "o": bk, "cards": bkc, "active": {bk.STAGE_RUNNING},
         "chat": lambda: settings.TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID, "cleanup": None},
        {"name": "ssh", "o": sh, "cards": shc, "active": {sh.STAGE_STAGE1, sh.STAGE_STAGE2},
         "chat": lambda: settings.SSH_TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID, "cleanup": None},
        # pfs（三段链）：refresh 定位当前段续推，同 ssh 例外——重启后续跑靠 stale 门 + NX 锁，幂等安全。
        {"name": "pfs", "o": pf, "cards": pfc,
         "active": {pf.STAGE_SINKING, pf.STAGE_CROSSING, pf.STAGE_PREHEATING},
         "chat": lambda: settings.PFS_TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID, "cleanup": None},
    ]


# 只对账“更新时间已过期”的在途任务：活着的后台线程每 60s 轮询一次会刷新 updated_ts，
# 故新鲜(<阈值)的任务必有线程在管、交给它推卡；过期的才是重启后被孤儿化、需要对账接管。
# 这样天然与在线线程错开，happy-path 不会重复推送。
_DATAFLOW_RECONCILE_STALE = 180


def _claim_dataflow_notify(job_id: str) -> bool:
    """终态结果卡“只推一次”的跨线程闸门：在线线程 on_update 与对账循环共用同一把 NX 锁，
    谁先抢到谁推。这样即使 stale 门失准（如单次 poll hang 超阈值、某 orchestrator 漏刷 updated_ts）
    也不会重复推送。Redis 不可用 → 放行（宁可重复也别漏）。TTL 覆盖任务生命周期。"""
    try:
        return bool(get_redis().set(f"dataflow:notified:{job_id}", 1, nx=True, ex=_TTL_NOTIFY))
    except Exception:
        return True


_TTL_NOTIFY = 30 * 86400


def _reconcile_dataflow_once() -> None:
    """扫各命名空间的在途任务，对孤儿(线程已死、updated_ts 过期)实时重查；
    完成/失败则推结果卡并置 notified，保证“任务无论中途重启几次，跑完都会自动通知”。"""
    try:
        r = get_redis()
    except Exception:
        return
    if r is None:
        return
    now = time.time()
    for spec in _dataflow_reconcile_specs():
        o = spec["o"]
        prefix = o._KEY_PREFIX
        try:
            keys = list(r.scan_iter(f"{prefix}*"))
        except Exception:
            continue
        for key in keys:
            job_id = key[len(prefix):]
            try:
                job = o.get_job(job_id)
            except Exception:
                continue
            if not job or job.get("stage") not in spec["active"] or job.get("notified"):
                continue
            if now - float(job.get("updated_ts", 0) or 0) < _DATAFLOW_RECONCILE_STALE:
                continue   # 新鲜 = 有活线程在轮询，交给它，避免重复通知
            try:
                fresh = o.refresh(job_id) or job   # 只轮询、不重提交；顺带接管孤儿轮询
            except Exception:
                logger.warning("[Reconcile] %s 重查失败 job=%s", spec["name"], job_id, exc_info=True)
                continue
            if fresh.get("stage") not in (o.STAGE_DONE, o.STAGE_FAILED):
                continue   # 仍在途：refresh 已落库刷新 updated_ts，下一轮再看
            if spec["cleanup"]:
                try:
                    spec["cleanup"](fresh)
                except Exception:
                    logger.warning("[Reconcile] %s 清理失败 job=%s", spec["name"], job_id, exc_info=True)
            fresh["notified"] = True
            o._save(fresh)
            # 抢到 NX 闸门才推：在线线程若已在终态推过卡，这里就不重复（跨线程去重的最终保证）。
            if not _claim_dataflow_notify(job_id):
                continue
            try:
                # 优先推给发起人本人（created_by），空则降级配置频道；须与在线线程同目标，NX 闸门才一致。
                _send_card(fresh.get("created_by", ""), spec["chat"](), spec["cards"].result_card(fresh))
                logger.info("[Reconcile] %s 任务 %s 完成后自动补通知(%s)",
                            spec["name"], job_id, fresh["stage"])
            except Exception:
                logger.error("[Reconcile] %s 推送结果卡失败 job=%s", spec["name"], job_id, exc_info=True)


# 全局单例
scheduler = DSWScheduler()
