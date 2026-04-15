"""
飞书消息 Webhook 服务。
飞书将用户消息事件 POST 到本服务 → Agent 处理 → 回复原消息。

启动方式：
    python main.py --mode bot
或：
    python core/feishu_bot.py

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
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import settings
from tools.feishu_tool import _get_access_token

app = Flask(__name__)

# ── 去重：同一 event_id 只处理一次（飞书会重试未 200 的请求）──────────────────
_seen_events: set = set()
_seen_lock   = threading.Lock()

# ── 每个会话独立历史（chat_id → deque of LangChain messages）──────────────────
_histories: dict = {}
_hist_lock  = threading.Lock()
_MAX_HIST   = 20   # 保留最近 20 轮


# ── 飞书 API ─────────────────────────────────────────────────────────────────

def _feishu_reply(message_id: str, text: str) -> None:
    """回复指定消息。"""
    try:
        token = _get_access_token()
        resp  = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "text", "content": json.dumps({"text": text})},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            print(f"[飞书回复] 失败: {data.get('msg')}")
    except Exception as e:
        print(f"[飞书回复] 异常: {e}")


def _feishu_send(chat_id: str, text: str) -> None:
    """向指定会话主动发送消息（用于超时兜底提示）。"""
    try:
        token = _get_access_token()
        requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": chat_id,
                "msg_type":   "text",
                "content":    json.dumps({"text": text}),
            },
            timeout=15,
        )
    except Exception:
        pass


# ── Agent 异步处理 ────────────────────────────────────────────────────────────

def _process_message(message_id: str, chat_id: str, user_text: str) -> None:
    """在后台线程中调用 Agent，将结果回复给用户。"""
    # 延迟导入，避免在模块加载时触发 LLM 初始化
    from core.agent import agent_executor

    with _hist_lock:
        if chat_id not in _histories:
            _histories[chat_id] = deque(maxlen=_MAX_HIST)
        history = list(_histories[chat_id])

    # 先发"思考中"提示，让用户知道 Bot 在处理
    _feishu_reply(message_id, "🤔 正在分析，请稍候...")

    try:
        result = agent_executor.invoke({"input": user_text, "chat_history": history})
        reply  = result["output"]

        with _hist_lock:
            _histories[chat_id].append(HumanMessage(content=user_text))
            _histories[chat_id].append(AIMessage(content=reply))

        _feishu_reply(message_id, reply)

    except Exception as e:
        _feishu_reply(message_id, f"⚠️ 处理出错：{e}")


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
    with _seen_lock:
        if event_id in _seen_events:
            return jsonify({"code": 0})
        _seen_events.add(event_id)
        # 防止集合无限增长，超过 5000 条就清一半
        if len(_seen_events) > 5000:
            remove = list(_seen_events)[:2500]
            for e in remove:
                _seen_events.discard(e)

    # ④ 只处理文本消息
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

    # ⑤ 先立即返回 200，再异步调用 Agent（飞书 5s 超时，Agent 可能更慢）
    threading.Thread(
        target=_process_message,
        args=(message_id, chat_id, user_text),
        daemon=True,
    ).start()

    return jsonify({"code": 0})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ts": int(time.time())})


# ── 入口 ──────────────────────────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = 8088, debug: bool = False):
    # 在主线程预热 Agent，避免子线程首次导入时 Pydantic v2 对 RunnableParallel
    # 中 lambda 做重新验证导致的 TypeError: got NoneType
    print("正在初始化 Agent（首次加载模型，请稍候）...")
    try:
        from core.agent import agent_executor  # noqa: F401
        print("Agent 初始化完成 ✓")
    except Exception as e:
        print(f"⚠️  Agent 初始化失败，Bot 将无法回复消息: {e}")

    print(f"飞书 Bot 服务启动 → http://{host}:{port}/feishu/event")
    print("请在飞书开放平台 → 事件订阅 → 请求地址 填写：http(s)://你的公网IP:{port}/feishu/event")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
