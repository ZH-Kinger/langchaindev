"""
飞书消息 Webhook 服务。
飞书将用户消息事件 POST 到本服务 → Agent 处理 → 回复原消息。

启动方式：
    python main.py --mode bot
或：
    python -m core.feishu_bot

飞书开发者后台配置（事件订阅）：
    请求地址 → http(s)://你的公网IP:8088/feishu/event
    添加事件 → im.message.receive_v1��接收消息）
"""
import sys
import io  # noqa: F401  (保留供其他模块可能用到)

# reconfigure 就地修改编码，不替换文件对象（兼容 Flask/click 的 fileno 检查）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import re
import threading
import time
from collections import deque

import requests
from flask import Flask, request, jsonify
from config.settings import settings
from utils.logger import get_logger, set_trace_id, clear_trace_id, register_error_callback

logger = get_logger(__name__)
from tools.feishu.notify import _get_access_token
from tools.jira.ticket import create_gpu_ticket, add_comment as jira_comment
from core.dsw_scheduler import (scheduler, _redis_get, _redis_set, _redis_delete,
                                _all_tracked_keys, check_quota, _cost_str, _set_approved)
from . import messaging
from .messaging import (_feishu_reply, _feishu_reply_with_chart,  # noqa: F401 (向后兼容再导出)
                        _feishu_send, _send_text_to)

app = Flask(__name__)

# ── 去重：同一 event_id 只处理一次（飞书会重试未 200 的请求）──────��───────────
# 优先用 Redis SET NX + TTL；Redis 不可用时降级为有上限的内存集合
_seen_events_fallback: set = set()
_seen_lock = threading.Lock()
_DEDUP_TTL = 3600  # event_id 保留 1 小时，覆盖飞书最长重试窗口


def _is_duplicate_event(event_id: str) -> bool:
    """已见过返回 True，首次见到返回 False 并记录。"""
    if not event_id:
        return False
    try:
        from utils.redis_client import get_redis, is_redis_available
        if is_redis_available():
            added = get_redis().set(
                f"feishu:event_dedup:{event_id}", 1, nx=True, ex=_DEDUP_TTL
            )
            return not added  # added=None 说明 key 已存在 → 重复
    except Exception:
        pass
    # Redis 不可用：内存兜底，最多保留 500 条（随机淘汰）
    with _seen_lock:
        if event_id in _seen_events_fallback:
            return True
        _seen_events_fallback.add(event_id)
        if len(_seen_events_fallback) > 500:
            _seen_events_fallback.clear()
        return False

# 会话历史统一由 core.agent 的 Redis 函数管理（按 chat_id 隔离）


# ── 飞书 API ─────────────────────────────────────────────────────────────────

# ── GPU 申请卡片 ──────────────────────────────────────────────────────────────

# GPU 申请意图：资源词 + 动作词同时出现，或明确的训练短语
_GPU_RESOURCE_WORDS = ("gpu", "dsw", "显卡", "算力", "实例", "训练资源")
_GPU_ACTION_WORDS   = ("申请", "需要", "开通", "租用")
_GPU_TRAIN_PHRASES  = ("我要训练", "跑模型", "训练模型", "跑训练", "微调模型",
                       "finetune", "fine-tune", "预训练")
# 查询/管理意图词：命中则不触发申请流程
_GPU_QUERY_WORDS    = ("查看", "查询", "列出", "列表", "看一下", "有哪些", "状态",
                       "显示", "我的实例", "已有", "运行中", "停止", "删除", "关闭")


def _is_gpu_intent(text: str) -> bool:
    t = text.lower()
    # 包含查询/管理词 → 交给 Agent 处理，不弹申请卡片
    if any(w in t for w in _GPU_QUERY_WORDS):
        return False
    if any(p in t for p in _GPU_TRAIN_PHRASES):
        return True
    has_resource = any(w in t for w in _GPU_RESOURCE_WORDS)
    has_action   = any(w in t for w in _GPU_ACTION_WORDS)
    return has_resource and has_action


# ── GPU 申请卡片（action buttons，programmatic interactive 消息支持）────────────
# form + submit_form 仅限飞书卡片构建器模板（msg_type: "template"），
# 编程发送的 interactive 消息不渲染 form 块，改用 action buttons 分两步完成申请。
_GPU_REQUEST_CARD = {
    "config": {"wide_screen_mode": True},
    "header": {
        "title": {"tag": "plain_text", "content": "GPU 资源申请"},
        "template": "blue",
    },
    "elements": [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**第一步：选择常用配置**（点击后卡片更新，再补充实例名和用途即可提交）",
            },
        },
        {"tag": "hr"},
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "1 GPU · 8h\n小模型微调"},
                    "type": "default",
                    "value": {"action": "quick_gpu", "gpu_count": "1", "duration_hours": "8"},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "4 GPU · 24h\n7B 模型训练"},
                    "type": "primary",
                    "value": {"action": "quick_gpu", "gpu_count": "4", "duration_hours": "24"},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "8 GPU · 48h\n大模型预训练"},
                    "type": "danger",
                    "value": {"action": "quick_gpu", "gpu_count": "8", "duration_hours": "48"},
                },
            ],
        },
        {"tag": "hr"},
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": (
                    "或直接回复自定义参数：\n"
                    "```\n实例名: wzh-train-01\nGPU数: 4\n时长: 24\n用途: 大语言模型微调\n```"
                ),
            },
        },
    ],
}

# 文字兜底引导（卡片回调不可用时的备用方案）
_GPU_GUIDE_TEXT = (
    "GPU 资源申请\n\n"
    "如卡片无法填写，请直接回复（包含以下四项即可）：\n\n"
    "实例名: wzh-train-01\n"
    "GPU数: 4\n"
    "时长: 24\n"
    "用途: 大语言模型微调\n\n"
    "GPU 数量支持 1 / 2 / 4 / 8，时长单位为小时。"
)

_GPU_STATE_PREFIX = "feishu:gpu_state:"
_GPU_STATE_TTL    = 600   # 10 分钟内等待用户填写


def _send_ak_register_card(message_id: str) -> None:
    """发送 AK/SK 绑定卡片：
    - 优先用飞书卡片构建器模板（FEISHU_AK_REGISTER_TEMPLATE_ID，含密码输入框）
    - 无模板时降级为文字引导，但提示用户应使用模板卡片而非聊天明文
    """
    template_id = settings.FEISHU_AK_REGISTER_TEMPLATE_ID
    if template_id:
        content = json.dumps({"type": "template", "data": {"template_id": template_id}})
        try:
            token = _get_access_token()
            resp = requests.post(
                f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"msg_type": "interactive", "content": content},
                timeout=15,
            )
            if resp.json().get("code") == 0:
                return
        except Exception:
            logger.error("[AK注册卡片] 异常", exc_info=True)

    # 无模板降级：文字引导（明确告诉用户不要把 AK/SK 直接发到聊天里）
    messaging._feishu_reply(message_id,
        "## 🔐 绑定阿里云 AK/SK\n\n"
        "绑定后 Bot 会用你的 RAM 用户身份创建资源，控制台能直接看到归属。\n\n"
        "**推荐方式**：让管理员在飞书卡片构建器配置 `FEISHU_AK_REGISTER_TEMPLATE_ID`\n"
        "  → 之后绑定走表单卡片，AccessKey Secret 输入框为密码遮蔽，不会出现在聊天记录中。\n\n"
        "**临时方式**（不推荐，有泄密风险）：私信 Bot 发送\n"
        "```\n绑定RAM: 你的RAM用户名\n```\n"
        "→ 仅建立 RAM 映射，资源归属是 Bot 角色（不是你本人），但操作审计能查到。\n\n"
        "其他命令：\n"
        "- `查看绑定` — 查看当前绑定状态\n"
        "- `解绑AK` — 删除已加密的 AK/SK")


def _send_gpu_card(message_id: str) -> None:
    """回复 GPU 申请卡片：先发可用镜像参考，再发表单卡片。"""
    try:
        from tools.aliyun.pai_dsw import list_dsw_resources
        images = list_dsw_resources().get("images", [])
        if images:
            lines = ["**可用镜像（复制填入卡片「镜像」字段）：**"]
            for i, img in enumerate(images[:8], 1):
                lines.append(f"{i}. `{img['name']}`")
            messaging._feishu_reply(message_id, "\n".join(lines))
    except Exception as e:
        logger.warning("[GPU卡片] 镜像列表获取失败（已跳过）: %s", e)

    template_id = settings.FEISHU_GPU_CARD_TEMPLATE_ID
    if template_id:
        content = json.dumps({"type": "template", "data": {"template_id": template_id}})
    else:
        content = json.dumps(_GPU_REQUEST_CARD)
    try:
        token = _get_access_token()
        resp = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "interactive", "content": content},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("[GPU卡片] 发送失败: %s", data.get('msg'))
    except Exception as e:
        logger.error("[GPU卡片] 异常", exc_info=True)


def _gpu_state_key(chat_id: str, open_id: str) -> str:
    return f"{_GPU_STATE_PREFIX}{chat_id}:{open_id}"


def _set_gpu_state(chat_id: str, open_id: str, config: dict | None = None) -> None:
    try:
        from utils.redis_client import get_redis
        get_redis().setex(_gpu_state_key(chat_id, open_id), _GPU_STATE_TTL,
                          json.dumps(config or {}))
    except Exception:
        pass


def _get_gpu_state(chat_id: str, open_id: str) -> dict | None:
    """返回已存储的配置 dict；key 不存在时返回 None。"""
    try:
        from utils.redis_client import get_redis
        raw = get_redis().get(_gpu_state_key(chat_id, open_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _clear_gpu_state(chat_id: str, open_id: str) -> None:
    try:
        from utils.redis_client import get_redis
        get_redis().delete(_gpu_state_key(chat_id, open_id))
    except Exception:
        pass


# ── GPU 申请解析 ──────────────────────────────────────────────────────────────

def _parse_gpu_request(text: str) -> dict | None:
    """尝试从用户文本中解析 GPU 申请信息，成功返回 dict，失败返回 None。"""
    t = text.lower()
    result: dict = {}

    for pat in (r"实例名[称]?\s*[：:=]\s*(\S+)", r"名称?\s*[：:=]\s*(\S+)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["instance_name"] = re.sub(r"[，,。.\s]", "", m.group(1))
            break

    for pat in (r"gpu\s*[数量块卡张]?\s*[：:=]\s*(\d+)", r"(\d+)\s*[块卡张]?\s*gpu",
                r"显卡\s*[：:=]\s*(\d+)"):
        m = re.search(pat, t)
        if m:
            result["gpu_count"] = m.group(1)
            break

    for pat in (r"时[长间]\s*[：:=]\s*(\d+)", r"(\d+)\s*小时", r"用\s*(\d+)\s*小时"):
        m = re.search(pat, t)
        if m:
            result["duration_hours"] = m.group(1)
            break

    for pat in (r"用途\s*[：:=]\s*(.+?)(?:[，,。.\n]|$)",
                r"目的\s*[：:=]\s*(.+?)(?:[，,。.\n]|$)",
                r"用于\s*(.+?)(?:[，,。.\n]|$)"):
        m = re.search(pat, text)
        if m:
            result["purpose"] = m.group(1).strip()
            break

    if "instance_name" in result and len(result) >= 2:
        return result
    if len(result) >= 3:
        return result
    return None


def _handle_gpu_request(message_id: str, chat_id: str, open_id: str, parsed: dict) -> None:
    """解析成功后异步创建 Jira 工单，调度器负责后续创建 DSW 实例。"""
    instance_name  = parsed.get("instance_name") or f"dsw-{int(time.time())}"
    gpu_count      = parsed.get("gpu_count", "1")
    duration_hours = parsed.get("duration_hours", "8")
    purpose        = parsed.get("purpose", "未说明")

    messaging._feishu_reply(message_id,
        f"⏳ 正在提交申请...\n实例：{instance_name}  GPU：{gpu_count}卡  时长：{duration_hours}h")

    def _do() -> None:
        ticket_key = create_gpu_ticket(
            instance_name=instance_name,
            gpu_count=gpu_count,
            duration_hours=duration_hours,
            purpose=purpose,
            reporter_open_id=open_id,
            chat_id=chat_id,
        )
        if ticket_key:
            messaging._feishu_reply(message_id,
                f"✅ 申请已提交！工单：{ticket_key}\n"
                f"调度器将在 2 分钟内自动创建实例 {instance_name}（{gpu_count}GPU，{duration_hours}h），"
                f"完成后飞书推送实例详情。")
        else:
            messaging._feishu_reply(message_id, "❌ Jira 工单创建失败，请联系运维人员。")

    threading.Thread(target=_do, daemon=True).start()


# ── 飞书↔阿里云身份绑定 ────────────────────────────────────────────────────
#
# 两条路径并存：
#   A. 用户 AK/SK（推荐）：通过飞书表单卡片填入，加密存 Redis
#      → 资源归属 = RAM 用户本人 ✅
#   B. RAM 映射 + STS（兜底）：根据飞书姓名匹配 RAM 用户，调 STS AssumeRole
#      → 资源归属 = BotRole/session（审计能查到 open_id）

_RAM_BIND_PATTERN = re.compile(r"绑定\s*RAM\s*[:：\s]+(\S+)", re.IGNORECASE)


def _is_registered(open_id: str) -> bool:
    """检查飞书用户是否已建立任意一种凭证关联：
    - 用户 AK/SK 已加密绑定，或
    - RAM 映射已建立（可走 STS 兜底）
    """
    if not open_id:
        return False
    # 用户 AK 优先（不解密、只检查存在）
    try:
        from utils.aliyun_user_creds import has_user_ak
        if has_user_ak(open_id):
            return True
    except Exception:
        pass
    # 退而求其次：RAM 映射存在
    try:
        from tools.aliyun.ram import get_ram_user_by_open_id
        info = get_ram_user_by_open_id(open_id)
        return bool(info and info.get("user_name"))
    except Exception:
        return False


def _handle_ram_bind(message_id: str, open_id: str, text: str) -> bool:
    """处理「绑定RAM」「解绑AK」「查看绑定」「注册AK / 绑定AK」等文本指令。"""
    # 「解绑AK」
    if re.search(r"解绑\s*ak|删除\s*ak|取消\s*绑定", text, re.IGNORECASE):
        from utils.aliyun_user_creds import delete_user_ak
        if delete_user_ak(open_id):
            messaging._feishu_reply(message_id,
                "✅ 已解绑你的阿里云 AK/SK，加密数据已从 Redis 删除。\n"
                "下次 Bot 操作将走 STS 兜底路径（资源归属为 Bot 角色）。")
        else:
            messaging._feishu_reply(message_id, "ℹ️ 你尚未绑定 AK/SK，无需解绑。")
        return True

    # 「查看绑定 / 我的绑定」
    if re.search(r"查看\s*绑定|我的\s*绑定|绑定\s*状态", text, re.IGNORECASE):
        _send_bind_status(message_id, open_id)
        return True

    # 「注册AK / 绑定AK」→ 弹出飞书表单卡片
    if re.search(r"注册\s*ak|绑定\s*ak", text, re.IGNORECASE):
        _send_ak_register_card(message_id)
        return True

    # 「绑定RAM: <ram_user_name>」→ 仅建立 RAM 映射（走 STS 兜底）
    m = _RAM_BIND_PATTERN.search(text)
    if not m:
        return False

    ram_user_name = m.group(1).strip()
    try:
        from tools.aliyun.ram import list_ram_users_api, save_user_map
        ram_users = list_ram_users_api()
        matched = next((u for u in ram_users if u["user_name"] == ram_user_name), None)
        if not matched:
            messaging._feishu_reply(message_id,
                f"❌ 未找到 RAM 用户 `{ram_user_name}`。\n"
                "请确认用户名拼写正确，或联系运维人员协助绑定。")
            return True
        save_user_map(open_id, matched["display_name"] or ram_user_name, matched["user_id"])
        messaging._feishu_reply(message_id,
            f"✅ 已建立 RAM 映射：**{matched['display_name'] or ram_user_name}**。\n"
            "此为兜底路径（走 STS，资源归属为 Bot 角色）。\n"
            "如希望资源归属直接显示你的 RAM 用户名，请发送「绑定AK」走表单卡片。")
    except Exception as e:
        logger.error("[RAM绑定] 失败", exc_info=True)
        messaging._feishu_reply(message_id, f"❌ 绑定失败：{e}")
    return True


def _send_bind_status(message_id: str, open_id: str) -> None:
    """展示当前用户的绑定状态。"""
    lines = ["## 你的阿里云身份绑定"]
    try:
        from utils.aliyun_user_creds import get_user_ak_meta
        from datetime import datetime
        meta = get_user_ak_meta(open_id)
        if meta:
            created = datetime.fromtimestamp(meta["created_ts"]).strftime("%Y-%m-%d %H:%M")
            last_u  = datetime.fromtimestamp(meta["last_used_ts"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- **AK 绑定**：✅  AccessKey ID `{meta['ak_id_masked']}`")
            lines.append(f"  - 绑定时间：{created}")
            lines.append(f"  - 最近使用：{last_u}")
            lines.append(f"  - 资源归属：**你的 RAM 用户**（阿里云控制台可直接看到）")
        else:
            lines.append("- **AK 绑定**：❌ 未绑定（发送「绑定AK」可配置）")
    except Exception:
        lines.append("- **AK 绑定**：查询失败")

    try:
        from tools.aliyun.ram import get_ram_user_by_open_id
        info = get_ram_user_by_open_id(open_id) or {}
        if info.get("user_name"):
            lines.append(f"- **RAM 映射**：✅  {info.get('user_name')}（STS 兜底路径）")
        else:
            lines.append("- **RAM 映射**：❌ 未建立")
    except Exception:
        pass

    lines.append("")
    lines.append("发送「绑定AK」绑定 AK/SK · 「解绑AK」取消绑定")
    messaging._feishu_reply(message_id, "\n".join(lines))


# ── 飞书用户 ↔ RAM 用户自动映射 ──────────────────────────────────────────────

def _auto_map_user(open_id: str) -> None:
    """首次见到某个 open_id 时，自动查飞书姓名并匹配 RAM 用户，写入 Redis。"""
    if not open_id:
        return
    try:
        from tools.aliyun.ram import get_user_map, save_user_map, list_ram_users_api
        if open_id in get_user_map():
            return  # 已有映射，跳过
        # 查飞书姓名
        from tools.feishu.notify import _get_user_name
        feishu_name = _get_user_name(open_id)
        if not feishu_name:
            return
        # 在 RAM 用户里按显示名匹配
        ram_users = list_ram_users_api()
        matched = next((u for u in ram_users if u["display_name"] == feishu_name), None)
        if matched:
            save_user_map(open_id, matched["display_name"], matched["user_id"])
            logger.info("[AutoMap] %s → RAM user_id=%s", feishu_name, matched['user_id'])
        else:
            save_user_map(open_id, feishu_name, "")
            logger.info("[AutoMap] %s 无对应 RAM 用户", feishu_name)
    except Exception as e:
        logger.warning("[AutoMap] 失败: %s", e)


# ── 我的实例快捷查询 ──────────────────────────────────────────────────────────────

def _query_my_instances(message_id: str, open_id: str, chat_id: str) -> None:
    """直接返回该用户通过 Bot 创建的所有运行中实例卡片，不走 Agent。"""
    now = time.time()
    user_tickets = [
        (k, s) for k in _all_tracked_keys()
        if (s := _redis_get(k)) and s.get("open_id") == open_id
    ]
    if not user_tickets:
        messaging._feishu_reply(message_id,
            "你目前没有通过 Bot 运行中的 GPU 实例。\n发送「申请GPU」可提交新申请。")
        return

    elements: list = []
    total_gpu = 0
    for key, state in user_tickets:
        elapsed_h   = (now - float(state.get("start_ts", now))) / 3600
        remaining_h = max(0.0, float(state.get("duration_hours", 8)) - elapsed_h)
        gpu_cnt     = int(state.get("gpu_count", 1))
        total_gpu  += gpu_cnt
        elements.append({
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md",
                 "content": f"**{state.get('instance_name', '-')}**\n工单 {key}"}},
                {"is_short": True, "text": {"tag": "lark_md",
                 "content": f"**剩余时长**\n{remaining_h:.1f} h"}},
                {"is_short": True, "text": {"tag": "lark_md",
                 "content": f"**已使用**\n{elapsed_h:.1f} h"}},
                {"is_short": True, "text": {"tag": "lark_md",
                 "content": f"**GPU 卡数**\n{gpu_cnt} 张"}},
            ],
        })
        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "⏩ 延续 2h"},
                    "type": "primary",
                    "value": {"action": "extend_dsw", "ticket_key": key,
                              "instance_id": state.get("instance_id", ""), "extend_hours": "2"},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🛑 立即停止"},
                    "type": "danger",
                    "value": {"action": "stop_dsw", "ticket_key": key,
                              "instance_id": state.get("instance_id", "")},
                },
            ],
        })
        elements.append({"tag": "hr"})

    card = {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text",
                              "content": f"我的 GPU 实例（{len(user_tickets)} 个）"},
                   "template": "blue"},
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md",
             "content": f"**共占用 GPU：** {total_gpu} 张"}},
            {"tag": "hr"},
        ] + elements,
    }
    try:
        token = _get_access_token()
        requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "interactive", "content": json.dumps(card)},
            timeout=15,
        )
    except Exception as e:
        logger.error("[我的实例] 发送失败", exc_info=True)
        messaging._feishu_reply(message_id, "获取实例信息失败，请稍后重试。")


# ── Agent 异步处理 ────────────────────────────────────────────────────────────

def _process_message(message_id: str, chat_id: str, user_text: str, open_id: str = "") -> None:
    """在后台线程中调用 Agent，将结果回复给用户。"""
    # ① 后台静默建立飞书↔RAM映射（首次见到该用户时触发）
    threading.Thread(target=_auto_map_user, args=(open_id,), daemon=True).start()

    # ② 我的实例快捷查询（不走 Agent）
    if re.search(r"(我的|my)\s*(实例|instance|dsw)", user_text, re.IGNORECASE):
        threading.Thread(
            target=_query_my_instances, args=(message_id, open_id, chat_id), daemon=True
        ).start()
        return

    # ③ GPU 申请意图 → 必须先注册个人 AK/SK
    if _is_gpu_intent(user_text):
        if not _is_registered(open_id):
            _send_ak_register_card(message_id)
            _set_gpu_state(chat_id, open_id, {"pending_gpu": True, "open_id": open_id})
        else:
            _send_gpu_card(message_id)
            _set_gpu_state(chat_id, open_id, {})
        return

    # ④ 处于 GPU 申请状态 → 合并快选配置（gpu_count/duration_hours）和文字解析
    gpu_config = _get_gpu_state(chat_id, open_id)
    if gpu_config is not None:
        parsed = _parse_gpu_request(user_text)
        if parsed or gpu_config:
            merged = {**gpu_config, **(parsed or {})}   # 文字解析优先级更高
            if "instance_name" in merged or len(merged) >= 3:
                _clear_gpu_state(chat_id, open_id)
                _handle_gpu_request(message_id, chat_id, open_id, merged)
                return
        _clear_gpu_state(chat_id, open_id)  # 解析不足，清除状态走 Agent

    # 延迟导入，避免在模块加载时触发 LLM 初始化
    from core.agent import _build_executor, _load_history, _save_turn

    # 从 Redis 加载该 chat_id 的会话历史（Redis 不可用时降级为空列表）
    history = _load_history(chat_id)

    # 先发"思考中"提示，让用户知道 Bot 在处理
    messaging._feishu_reply(message_id, "🤔 正在分析，请稍候...")

    # 将 open_id 注入输入，供 analyze_gpu_training 等需要身份识别的工具使用
    agent_input = f"{user_text}\n[feishu_open_id={open_id}]" if open_id else user_text

    try:
        result = _build_executor().invoke({"input": agent_input, "chat_history": history})
        reply  = result["output"]

        # 清除 LLM 可能在回复文本中虚构的 Markdown 图片占位符
        # 例如 "![图](https://...)" 或 "!https://..." —— 飞书无法渲染这类链接
        reply = re.sub(r"!\[.*?\]\(.*?\)", "", reply)
        reply = re.sub(r"!https?://\S+", "", reply).strip()

        # 写回历史时用原始 user_text（不含注入的 open_id 元数据）
        _save_turn(chat_id, user_text, reply)

        # 每次回复都尝试生成实时指标趋势图，失败时降级为纯文本回复
        try:
            from tools.aliyun.prometheus import fetch_raw_series
            from utils.chart_builder import build_metrics_chart
            from tools.feishu.notify import _upload_image
            series    = fetch_raw_series()
            png_bytes = build_metrics_chart(series)
            token     = _get_access_token()
            image_key = _upload_image(token, png_bytes)
            messaging._feishu_reply_with_chart(message_id, reply, image_key)
        except Exception as chart_err:
            logger.warning("[趋势图] 生成失败（已降级）: %s", chart_err)
            messaging._feishu_reply(message_id, reply)

    except Exception as e:
        logger.error("处理消息出错", exc_info=True)
        messaging._feishu_reply(message_id, f"⚠️ 处理出错：{e}")
    finally:
        clear_trace_id()


# ── 卡片按钮事件处理（card.action.trigger / card_action 共用）────────────────

def _process_action(action_name: str, action_val: dict, open_id: str, chat_id: str,
                    form_value: dict | None = None) -> dict:
    """所有卡片动作的共享处理逻辑，返回飞书期待的响应 dict。
    耗时操作（Jira / DSW API）放后台线程，同步只做状态更新和 UI 反馈。
    """
    # ── MFU 日报区域切换：返回新卡片原地替换（数据走 Redis 缓存，秒级）────────
    if action_name == "mfu_region":
        from tools.aliyun.cluster_mfu import build_mfu_card
        region = action_val.get("region", "") if isinstance(action_val, dict) else ""
        # schema 2.0 卡片回调：用 {"card":{"type":"raw","data":...}} 原地更新
        return {"toast": {"type": "info", "content": "已切换"},
                "card": {"type": "raw", "data": build_mfu_card(view=region or "summary")}}

    # ── AK/SK 表单卡片提交（Fernet 加密存 Redis，资源归属为用户本人）─────────
    if action_name == "submit_ak_register" and form_value:
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
        gpu_state = _get_gpu_state(chat_id, open_id)
        if gpu_state and gpu_state.get("pending_gpu"):
            _clear_gpu_state(chat_id, open_id)
            def _push_gpu_card():
                import time; time.sleep(1)
                messaging._feishu_send(chat_id, "✅ AK 已绑定！正在为你打开 GPU 申请表单...")
                try:
                    template_id = settings.FEISHU_GPU_CARD_TEMPLATE_ID
                    content = json.dumps({"type": "template", "data": {"template_id": template_id}}) if template_id else json.dumps(_GPU_REQUEST_CARD)
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
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "✅ AccessKey 已绑定"}, "template": "green"},
                "elements": [{
                    "tag": "div",
                    "text": {"tag": "lark_md", "content":
                        f"**AccessKey ID：** `{ak_id[:8]}****`（已 Fernet 加密存入 Redis）\n"
                        + (f"**RAM 用户：** {display}\n" if display else "")
                        + f"**资源归属：** 你的 RAM 用户本人（阿里云控制台可直接看到）\n"
                        + f"**自动失效：** {settings.USER_AK_IDLE_TTL_SECONDS // 86400} 天未使用\n\n"
                        + "随时发送「解绑AK」可删除，「查看绑定」查状态。"},
                }],
            },
        }

    # ── 表单提交（submit_form behavior）──────────────────────────────────────
    if action_name == "submit_gpu_request" and form_value:
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
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⏳ 申请处理中"}, "template": "yellow"},
                "elements": [{
                    "tag": "div",
                    "text": {"tag": "lark_md", "content":
                        f"**实例：** {instance_name}　**GPU：** {gpu_count} 卡　**时长：** {duration_hours}h\n"
                        f"**CPU：** {cpu_cores or '默认'}　**内存：** {memory_gb or '默认'} GB　**优先级：** {priority}\n\n"
                        "正在创建 Jira 工单，完成后飞书通知。"},
                }],
            },
        }

    # ── 快选配置（1/4/8 GPU）─────────────────────────────────────────────────
    if action_name == "quick_gpu":
        gpu_count      = action_val.get("gpu_count", "1")
        duration_hours = action_val.get("duration_hours", "8")
        _set_gpu_state(chat_id, open_id, {"gpu_count": gpu_count, "duration_hours": duration_hours})
        return {
            "toast": {"type": "info", "content": f"已选 {gpu_count}GPU · {duration_hours}h，请回复实例名和用途"},
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "GPU 资源申请"}, "template": "blue"},
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": (
                                f"已选：**{gpu_count} GPU · {duration_hours} 小时**\n\n"
                                "请直接回复此消息，补充以下信息：\n"
                                "```\n实例名: wzh-train-01\n用途: 大语言模型微调\n```"
                            ),
                        },
                    },
                    {
                        "tag": "note",
                        "elements": [{"tag": "plain_text", "content": "回复后系统自动创建 DSW 实例"}],
                    },
                ],
            },
        }

    # ── 延续使用 ──────────────────────────────────────────────────────────────
    if action_name == "extend_dsw":
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

    # ── 立即停止 ──────────────────────────────────────────────────────────────
    if action_name == "stop_dsw":
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
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "🛑 实例已停止"},
                    "template": "red",
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md",
                     "content": f"实例 `{instance_id or ticket_key}` 已手动停止，工单已关闭。\n按钮已失效，如需重新使用请提交新工单。"}},
                ],
            },
        }

    # ── 审批通过 ──────────────────────────────────────────────────────────────
    if action_name == "approve_gpu":
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
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "✅ 已批准"}, "template": "green"},
                "elements": [{"tag": "div", "text": {"tag": "lark_md",
                    "content": f"工单 **{ticket_key}** 已批准，调度器将在 20 秒内自动创建实例。"}}],
            },
        }

    # ── 审批拒绝 ──────────────────────────────────────────────────────────────
    if action_name == "reject_gpu":
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
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "❌ 已拒绝"}, "template": "red"},
                "elements": [{"tag": "div", "text": {"tag": "lark_md",
                    "content": f"工单 **{ticket_key}** 已拒绝，Jira 状态已更新。"}}],
            },
        }

    # ── OSS 权限同步：管理员批准 → 后台下发 → 回结果卡片 ───────────────────────
    if action_name == "approve_oss_perm":
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

    return {}


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


# ── Webhook 路由 ──────────────────────────────────────────────────────────────

@app.route("/feishu/event", methods=["GET", "POST"])
def feishu_event():
    data = request.get_json(silent=True) or {}

    # ① URL 验证（首次配置时飞书发送 challenge）
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge", "")})

    header = data.get("header", {})

    # ② 可选：验证 token（在飞书后台 → 事件订阅 → Verification Token 里找）
    verification_token = settings.FEISHU_VERIFICATION_TOKEN
    if verification_token and header.get("token") != verification_token:
        return jsonify({"code": 1, "msg": "invalid token"}), 403

    # ③ 去重
    event_id = header.get("event_id", "")
    if _is_duplicate_event(event_id):
        return jsonify({"code": 0})

    # ④ 卡片按钮点击事件（card.action.trigger）——同步处理，飞书要求 3s 内响应
    event_type = header.get("event_type", "")
    # 已读回执无需处理，提前返回避免日志噪音
    if event_type == "im.message.message_read_v1":
        return jsonify({"code": 0})
    logger.debug("[事件] event_type=%r", event_type)
    if event_type == "card.action.trigger":
        return jsonify(_handle_card_trigger_sync(data))

    # ⑤ 只处理文本消息
    event      = data.get("event", {})
    message    = event.get("message", {})
    msg_type   = message.get("message_type", "")
    if msg_type != "text":
        return jsonify({"code": 0})

    raw_content = message.get("content", "{}")
    try:
        user_text = json.loads(raw_content).get("text", "").strip()
    except Exception:
        user_text = raw_content.strip()

    # 去掉群聊中的 @机器人 标记
    user_text = re.sub(r"@[^\s\u200b]+[\s\u200b]*", "", user_text).strip()
    if not user_text:
        return jsonify({"code": 0})

    message_id = message.get("message_id", "")
    chat_id    = message.get("chat_id", "")
    open_id    = event.get("sender", {}).get("sender_id", {}).get("open_id", "")
    set_trace_id(message_id[-8:] if message_id else "-")

    # ⑤ RAM 绑定指令优先处理（自动映射失败时的兜底）
    if _handle_ram_bind(message_id, open_id, user_text):
        return jsonify({"code": 0})

    # ⑥ 先立即返回 200，再异步调用 Agent（飞书 5s 超时，Agent 可能更慢）
    threading.Thread(
        target=_process_message,
        args=(message_id, chat_id, user_text, open_id),
        daemon=True,
    ).start()

    return jsonify({"code": 0})


@app.route("/feishu/card_action", methods=["GET", "POST"])
def feishu_card_action():
    """飞书卡片请求地址回调：action buttons / 旧式 form 提交均在此处理。"""
    # GET challenge 验证（飞书填写地址时发送）
    challenge = request.args.get("challenge")
    if challenge:
        return jsonify({"challenge": challenge})

    data = request.get_json(silent=True) or {}

    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge", "")})

    # schema 2.0 卡片回调：动作在 data["event"]（与事件订阅 card.action.trigger 同构），
    # 旧解析按 data["action"] 取不到值（action/open_id 全空）→ 统一走 2.0 解析。
    logger.info("[card_action] RAW keys=%s", list(data.keys()))
    if data.get("schema", "").startswith("2") or "event" in data or \
            data.get("header", {}).get("event_type") == "card.action.trigger":
        return jsonify(_handle_card_trigger_sync(data))

    action_obj  = data.get("action", {})
    action_val  = action_obj.get("value") or {}
    form_value  = action_obj.get("form_value") or {}
    open_id = (
        data.get("operator", {}).get("operator_id", {}).get("open_id")
        or data.get("open_id", "")
        or data.get("user_id", "")
    )
    chat_id = data.get("open_chat_id") or settings.FEISHU_CHAT_ID

    action_name = action_val.get("action", "") if isinstance(action_val, dict) else ""
    if not action_name and form_value:
        # 根据表单字段区分注册卡片和 GPU 申请卡片
        if "ak_id" in form_value or "ak_secret" in form_value:
            action_name = "submit_ak_register"
        else:
            action_name = "submit_gpu_request"
    logger.info("[card_action] action=%r open_id=%r", action_name, open_id)

    return jsonify(_process_action(action_name, action_val, open_id, chat_id, form_value=form_value))


@app.route("/health", methods=["GET"])
def health():
    """深度健康检查：探测各依赖服务连通性。"""
    result: dict = {"ts": int(time.time()), "status": "ok"}

    # Redis
    try:
        from utils.redis_client import get_redis
        get_redis().ping()
        result["redis"] = "ok"
    except Exception as e:
        result["redis"] = f"error: {e}"
        result["status"] = "degraded"

    # Prometheus
    try:
        from tools.aliyun.prometheus import _query_instant
        r = _query_instant("up")
        result["prometheus"] = "ok" if r is not None else "error: empty response"
        if r is None:
            result["status"] = "degraded"
    except Exception as e:
        result["prometheus"] = f"error: {e}"
        result["status"] = "degraded"

    # Jira
    try:
        import requests as _req
        from config.settings import settings
        if settings.JIRA_URL and settings.JIRA_PAT:
            resp = _req.get(
                f"{settings.JIRA_URL}/rest/api/2/serverInfo",
                headers={"Authorization": f"Bearer {settings.JIRA_PAT}"},
                timeout=5,
            )
            result["jira"] = "ok" if resp.status_code == 200 else f"error: HTTP {resp.status_code}"
            if resp.status_code != 200:
                result["status"] = "degraded"
        else:
            result["jira"] = "not_configured"
    except Exception as e:
        result["jira"] = f"error: {e}"
        result["status"] = "degraded"

    # PAI DSW API
    try:
        from tools.aliyun.pai_dsw import list_dsw_resources
        list_dsw_resources()
        result["dsw_api"] = "ok"
    except Exception as e:
        result["dsw_api"] = f"error: {e}"
        result["status"] = "degraded"

    # 飞书 Token
    try:
        token = _get_access_token()
        result["feishu"] = "ok" if token else "error: empty token"
        if not token:
            result["status"] = "degraded"
    except Exception as e:
        result["feishu"] = f"error: {e}"
        result["status"] = "degraded"

    # 加密 key
    try:
        from utils.crypto import is_key_configured
        if is_key_configured():
            result["crypto"] = "ok"
        else:
            result["crypto"] = "error: BOT_CREDS_ENCRYPTION_KEY not configured"
            result["status"] = "degraded"
    except Exception as e:
        result["crypto"] = f"error: {e}"
        result["status"] = "degraded"

    return jsonify(result), 200 if result["status"] == "ok" else 207


# ── 入口 ──────────────────────────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = 8088, debug: bool = False):
    # ── 启动前安全校验：加密 key 必须配置 ─────────────────────────────────
    from utils.crypto import is_key_configured
    if not is_key_configured():
        logger.error(
            "[启动校验] BOT_CREDS_ENCRYPTION_KEY 未配置或格式错误！\n"
            "  原因：用户绑定 AK 路径需要 Fernet 加密 key，否则会拒绝写入。\n"
            "  生成命令：python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            "  把输出填到 .env 的 BOT_CREDS_ENCRYPTION_KEY= 行后即可重启。"
        )
        raise SystemExit(2)

    # 在主线程预热 Agent，避免子线程首次导入时 Pydantic v2 对 RunnableParallel
    # 中 lambda 做重新验证导致的 TypeError: got NoneType
    # 注册错误回调：ERROR 级别日志自动推送管理员飞书
    if settings.ADMIN_FEISHU_OPEN_ID:
        register_error_callback(
            lambda msg: messaging._send_text_to(settings.ADMIN_FEISHU_OPEN_ID, settings.FEISHU_CHAT_ID, msg)
        )
        logger.info("错误飞书推送已注册 → %s", settings.ADMIN_FEISHU_OPEN_ID)

    logger.info("正在初始化 Agent（首次加载模型，请稍候）...")
    try:
        from core.agent import _build_executor
        _build_executor()
        logger.info("Agent 初始化完成")
    except Exception as e:
        logger.error("Agent 初始化失败，Bot 将无法回复消息", exc_info=True)

    try:
        scheduler.start()
    except Exception as e:
        logger.error("调度器启动失败", exc_info=True)

    logger.info("飞书 Bot 服务启动 → http://%s:%s/feishu/event", host, port)
    logger.info("卡片回调地址     → http://%s:%s/feishu/card_action", host, port)
    app.run(host=host, port=port, debug=debug)
