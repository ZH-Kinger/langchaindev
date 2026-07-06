"""Flask Webhook 路由：/feishu/event、/feishu/card_action、/health + run() 入口。"""
import json
import re
import threading
import time

from flask import Flask, request, jsonify

from config.settings import settings
from core import ram_approval
from utils.logger import get_logger, set_trace_id, register_error_callback

logger = get_logger(__name__)
from tools.feishu.notify import _get_access_token
from core.dsw_scheduler import scheduler
from . import actions, messages, messaging

app = Flask(__name__)


@app.route("/feishu/event", methods=["GET", "POST"])
def feishu_event():
    data = request.get_json(silent=True) or {}

    # ① URL 验证（首次配置时飞书发送 challenge）
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge", "")})

    header = data.get("header", {})

    # ② 可选：验证 token（在飞书后台 → 事件订阅 → Verification Token 里找）
    #   token 位置随事件格式而异：schema 2.0 在 header.token；旧版 v1（如 leave_approval
    #   等审批事件、challenge）在顶层 data["token"]。两处都取，否则旧版审批回调（无 header）
    #   会因取不到 token 被 403 误杀。
    verification_token = settings.FEISHU_VERIFICATION_TOKEN
    request_token = header.get("token") or data.get("token")
    if verification_token and request_token != verification_token:
        logger.warning(
            "[feishu_event] invalid token type=%s approval_code=%s instance=%s has_token=%s",
            header.get("event_type") or data.get("type") or "-",
            ram_approval.event_log_summary(data).get("approval_code") or "-",
            ram_approval.event_log_summary(data).get("instance_code") or "-",
            bool(request_token),
        )
        return jsonify({"code": 1, "msg": "invalid token"}), 403

    # ③ 去重
    event_id = header.get("event_id", "")
    if messages._is_duplicate_event(event_id):
        return jsonify({"code": 0})

    # ④ 卡片按钮点击事件（card.action.trigger）——同步处理，飞书要求 3s 内响应
    event_type = header.get("event_type", "")
    # 已读回执无需处理，提前返回避免日志噪音
    if event_type == "im.message.message_read_v1":
        return jsonify({"code": 0})
    logger.debug("[事件] event_type=%r", event_type)
    if event_type == "card.action.trigger":
        return jsonify(actions._handle_card_trigger_sync(data))

    approval_summary = ram_approval.event_log_summary(data)
    is_approval_like = "approval" in (approval_summary.get("event_type") or "").lower() or bool(approval_summary.get("approval_code"))
    should_handle_approval = ram_approval.should_handle_event(data)
    if is_approval_like:
        logger.info(
            "[approval_event] received type=%s approval_code=%s instance=%s status=%s matched=%s target=%s",
            approval_summary.get("event_type") or "-",
            approval_summary.get("approval_code") or "-",
            approval_summary.get("instance_code") or "-",
            approval_summary.get("status") or "-",
            should_handle_approval,
            settings.FEISHU_RAM_APPROVAL_CODE or "-",
        )
    if should_handle_approval:
        threading.Thread(
            target=ram_approval.handle_approval_event,
            args=(data,),
            daemon=True,
        ).start()
        return jsonify({"code": 0})

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
    if messages._handle_ram_bind(message_id, open_id, user_text):
        return jsonify({"code": 0})

    # ⑥ 先立即返回 200，再异步调用 Agent（飞书 5s 超时，Agent 可能更慢）
    threading.Thread(
        target=messages._process_message,
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
        if any(k in form_value for k in ("login_name", "user_name", "q")):
            action_name = "submit_ram_query"
        elif "ak_id" in form_value or "ak_secret" in form_value:
            action_name = "submit_ak_register"
        else:
            action_name = "submit_gpu_request"
    logger.info("[card_action] action=%r open_id=%r", action_name, open_id)

    # 与 2.0 分支共用同一把去重锁：飞书同一次操作双投递（老式回调 + 事件订阅），只放行一次
    msg_id = data.get("open_message_id", "")
    if actions.card_action_is_duplicate(action_name, open_id, msg_id, form_value, action_val):
        return jsonify({})

    return jsonify(actions._process_action(action_name, action_val, open_id, chat_id, form_value=form_value))




def _ram_api_authorized() -> bool:
    expected = getattr(settings, "RAM_QUERY_API_TOKEN", "") or getattr(settings, "FEISHU_VERIFICATION_TOKEN", "")
    if not expected:
        return False
    auth = request.headers.get("Authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    supplied = request.headers.get("X-API-Token", "") or bearer or request.args.get("token", "")
    return supplied == expected


@app.route("/api/ram/user", methods=["GET", "POST"])
def api_ram_user():
    """Read-only RAM account query. Never returns password or AccessKey Secret."""
    if not _ram_api_authorized():
        return jsonify({"ok": False, "error": "unauthorized"}), 403

    body = request.get_json(silent=True) or {}
    login_name = (
        request.args.get("login_name")
        or request.args.get("user_name")
        or request.args.get("q")
        or body.get("login_name")
        or body.get("user_name")
        or body.get("q")
        or ""
    )
    if not str(login_name).strip():
        return jsonify({"ok": False, "error": "missing login_name"}), 400

    from core.ram_query import RamQueryError, RamUserNotFound, query_ram_account
    try:
        user = query_ram_account(str(login_name))
        return jsonify({"ok": True, "exists": True, "user": user})
    except RamUserNotFound:
        return jsonify({"ok": True, "exists": False, "user": None}), 404
    except RamQueryError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

def _gpu_dist_authorized() -> bool:
    """页面 token 门禁：优先 GPU_DIST_TOKEN，回退 RAM_QUERY_API_TOKEN / FEISHU_VERIFICATION_TOKEN。"""
    expected = (getattr(settings, "GPU_DIST_TOKEN", "")
                or getattr(settings, "RAM_QUERY_API_TOKEN", "")
                or getattr(settings, "FEISHU_VERIFICATION_TOKEN", ""))
    if not expected:
        return False
    auth = request.headers.get("Authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    supplied = request.headers.get("X-API-Token", "") or bearer or request.args.get("token", "")
    return supplied == expected


@app.route("/gpu/distribution", methods=["GET"])
def gpu_distribution_page():
    """GPU 卡分布实时页面（自动刷新 HTML）。token 门禁（GPU_DIST_TOKEN 优先）。"""
    if not getattr(settings, "GPU_DIST_ENABLED", True):
        return "gpu distribution disabled", 404
    if not _gpu_dist_authorized():
        return "unauthorized", 403
    from tools.aliyun.gpu_distribution import get_distribution, get_timeseries, build_html
    try:
        refresh = request.args.get("refresh") == "1"
        try:
            hours = int(request.args.get("hours", "24"))
        except (TypeError, ValueError):
            hours = 24
        g = get_distribution(refresh=refresh)
        try:
            series = get_timeseries(hours=hours, refresh=refresh)
        except Exception:
            logger.warning("[gpu_distribution] timeseries failed (charts skipped)", exc_info=True)
            series = {}
        token = request.args.get("token", "") or request.headers.get("X-API-Token", "")
        # 禁止缓存：页面每 15s 自更新，浏览器/飞书 webview 缓存旧页面会导致"按钮点不动"(旧 meta 刷新死循环)
        return build_html(g, series, token=token), 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("[gpu_distribution] render failed: %s", exc, exc_info=True)
        return f"error: {exc}", 500


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
