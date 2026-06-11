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
from . import gpu_flow, messaging
from .messaging import (_feishu_reply, _feishu_reply_with_chart,  # noqa: F401 (向后兼容再导出)
                        _feishu_send, _send_text_to)
from .gpu_flow import (_GPU_REQUEST_CARD, _GPU_GUIDE_TEXT,  # noqa: F401 (向后兼容再导出)
                       _parse_gpu_request, _handle_gpu_request, _send_gpu_card,
                       _send_ak_register_card, _set_gpu_state, _get_gpu_state,
                       _clear_gpu_state)
from . import actions
from .actions import _process_action, _handle_card_trigger_sync  # noqa: F401 (向后兼容再导出)

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
        gpu_flow._send_ak_register_card(message_id)
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
            gpu_flow._send_ak_register_card(message_id)
            gpu_flow._set_gpu_state(chat_id, open_id, {"pending_gpu": True, "open_id": open_id})
        else:
            gpu_flow._send_gpu_card(message_id)
            gpu_flow._set_gpu_state(chat_id, open_id, {})
        return

    # ④ 处于 GPU 申请状态 → 合并快选配置（gpu_count/duration_hours）和文字解析
    gpu_config = gpu_flow._get_gpu_state(chat_id, open_id)
    if gpu_config is not None:
        parsed = gpu_flow._parse_gpu_request(user_text)
        if parsed or gpu_config:
            merged = {**gpu_config, **(parsed or {})}   # 文字解析优先级更高
            if "instance_name" in merged or len(merged) >= 3:
                gpu_flow._clear_gpu_state(chat_id, open_id)
                gpu_flow._handle_gpu_request(message_id, chat_id, open_id, merged)
                return
        gpu_flow._clear_gpu_state(chat_id, open_id)  # 解析不足，清除状态走 Agent

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


# ── 卡片按钮事件处理：见 actions.py（注册表 + _handle_card_trigger_sync）─────


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
        return jsonify(actions._handle_card_trigger_sync(data))

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
        return jsonify(actions._handle_card_trigger_sync(data))

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

    return jsonify(actions._process_action(action_name, action_val, open_id, chat_id, form_value=form_value))


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
