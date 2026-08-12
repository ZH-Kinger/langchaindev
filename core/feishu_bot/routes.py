"""Flask Webhook 路由：/feishu/event、/feishu/card_action、/health + run() 入口。"""
import hmac
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


def _approval_allowlist() -> set:
    """允许 bot 处理的审批 definitionCode 白名单。其余审批事件（含无 code）一律丢弃——安全硬门，
    防"收到指定之外的审批"。含：RAM 建号审批 + （启用时）临时 AK 发放/延期审批。"""
    codes = set()
    if settings.FEISHU_RAM_APPROVAL_CODE:
        codes.add(settings.FEISHU_RAM_APPROVAL_CODE)
    if settings.TEMP_AK_ENABLED:
        for c in (settings.TEMP_AK_APPROVAL_CODE, settings.TEMP_AK_EXTEND_APPROVAL_CODE):
            if c:
                codes.add(c)
        # 多阿里云主账号：各账号有自己的发放审批 code（延长/撤销复用上面那条共用模板）。
        # 档案未注册（缺 code 或缺该账号 AK）就不会出现在这里，天然不会放行没配好的账号。
        try:
            from core.temp_ak_issuance import accounts as temp_ak_accounts
            codes |= temp_ak_accounts.issue_codes()
        except Exception:
            logger.error("[temp_ak] 账号档案装配失败，白名单仅含基础 code", exc_info=True)
    return codes


def _extract_request_token(data: dict) -> str:
    """取飞书请求里的验证 token（在飞书后台 → 事件订阅 → Verification Token 里找）。

    位置随事件格式而异：schema 2.0 在 `header.token`；旧版 v1（leave_approval 等审批事件、
    challenge、老式卡片回调）在顶层 `data["token"]`。两处都取，否则旧版回调（无 header）会因
    取不到 token 被 403 误杀。
    """
    if not isinstance(data, dict):        # body 是 JSON 数组/标量时 .get 会 AttributeError → 500
        return ""
    header = data.get("header")
    header = header if isinstance(header, dict) else {}
    token = header.get("token") or data.get("token") or ""
    return token if isinstance(token, str) else ""


def _accepted_tokens() -> list:
    """可接受的入站验证 token 集合。

    飞书对**同一次卡片点击双投递**，两条走的是开放平台里两处**不同的配置**，各带各的 token：
      · 事件订阅 `card.action.trigger`  → `header.token`  = `FEISHU_VERIFICATION_TOKEN`
      · 旧式「消息卡片 → 请求网址」      → 顶层 `token`     = `FEISHU_CARD_VERIFICATION_TOKEN`

    线上实测确认过这一点：旧式那条 `has_token=True` 却与事件订阅的 token 不匹配，长度也不同。
    只认前者的话，旧式那条**每次点击都会被拒** —— 功能不受影响（2.0 那条会把动作执行掉，
    且 403 发生在去重之前、不占去重名额），但会持续产生 `invalid token` 噪音，把真正需要
    警觉的攻击信号淹掉；同时也失去了双通道冗余。

    两个都收**不构成放宽**：白名单里仍然只有"本应用在飞书控制台配置过的、由飞书签发的
    secret"，攻击者两个都拿不到。未配置的那个不会进集合（空值被过滤）。
    """
    return [t for t in (settings.FEISHU_VERIFICATION_TOKEN,
                        getattr(settings, "FEISHU_CARD_VERIFICATION_TOKEN", "")) if t]


def _token_source_hint(supplied: str) -> str:
    """判断收到的 token 是不是某个**已知**的 secret，只回名字、绝不回值。

    **保留价值（不是临时诊断）**：配好两套 token 之后，`invalid token` 应当变成罕见事件 ——
    那时每一条都值得看，而 `src=` 能立刻区分「配置漂移/token 轮换忘了同步」与「真有人在打」。
    只回名字不回值，日志里不会留下任何可利用的材料。
    """
    if not supplied:
        return "empty"
    for name, val in (
        ("VERIFICATION_TOKEN",      settings.FEISHU_VERIFICATION_TOKEN),
        ("CARD_VERIFICATION_TOKEN", getattr(settings, "FEISHU_CARD_VERIFICATION_TOKEN", "")),
        ("APP_SECRET",              getattr(settings, "FEISHU_APP_SECRET", "")),
        ("APP_ID",                  getattr(settings, "FEISHU_APP_ID", "")),
    ):
        if val and hmac.compare_digest(str(supplied).encode("utf-8"), str(val).encode("utf-8")):
            return name
    return f"unknown(len={len(supplied)})"


def _token_verified(data: dict) -> bool:
    """入站请求验证 token 校验 —— /feishu/event 与 /feishu/card_action 共用这一把门。

    未配置 `FEISHU_VERIFICATION_TOKEN` 时返回 True（保持既有降级行为，不让未配置的环境直接瘫），
    但 `run()` 启动时会打 ERROR 告警说明此时门是敞的。常数时间比较，避免逐字节比较的时序侧信道。

    **必须先 encode 成 bytes**：`hmac.compare_digest` 对含非 ASCII 字符的 str 会抛
    `TypeError: comparing strings with non-ASCII characters is not supported`。token 来自外部
    请求体，攻击者随手传个中文/西里尔字母就能让视图抛异常 → Flask 500，而 500 会经
    utils/logger 的 ERROR 回调**给管理员刷飞书私信** —— 等于把一道鉴权门变成免鉴权的告警放大器。
    """
    accepted = _accepted_tokens()
    if not accepted:
        return True
    supplied = _extract_request_token(data).encode("utf-8")
    # 逐个比对且**不短路**：any() 本身会短路，但每次比较都是常数时间的，
    # 命中与否只影响比较次数（≤2），不泄漏 token 内容。
    return any(hmac.compare_digest(supplied, str(e).encode("utf-8")) for e in accepted)


@app.route("/feishu/event", methods=["GET", "POST"])
def feishu_event():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):   # body 为 JSON 数组/标量时，后续 .get 会 AttributeError → 500
        data = {}

    # ① URL 验证（首次配置时飞书发送 challenge）
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge", "")})

    header = data.get("header")
    header = header if isinstance(header, dict) else {}   # 同上：header 非 dict 时下面的 .get 会 500

    # ② 可选：验证 token（token 取值位置见 _extract_request_token）
    if not _token_verified(data):
        logger.warning(
            "[feishu_event] invalid token type=%s approval_code=%s instance=%s has_token=%s",
            header.get("event_type") or data.get("type") or "-",
            ram_approval.event_log_summary(data).get("approval_code") or "-",
            ram_approval.event_log_summary(data).get("instance_code") or "-",
            bool(_extract_request_token(data)),
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
        # 审批白名单硬门（用户要求）：只处理明确配置的审批 definitionCode，其余审批事件——**包括未带
        # approval_code 的**——一律记日志丢弃。这也收紧了 ram_approval 的无-code 兜底：凡不在白名单内的
        # 审批一概不进任何处理器，杜绝"收到我给你之外的审批"。
        _allow = _approval_allowlist()
        _ev_code = approval_summary.get("approval_code") or ""
        if _ev_code not in _allow:
            logger.info("[approval_event] 非白名单审批 code=%s 已丢弃（白名单 %d 项）",
                        _ev_code or "-", len(_allow))
            return jsonify({"code": 0})

    # 临时 AK/SK 发放 + 延期审批（独立模板，严格 code 匹配）：排在 RAM 建号审批分发**之前**，
    # temp_ak 严格 code==target 不会误抢 ram 事件。
    if settings.TEMP_AK_ENABLED:
        try:
            from core.temp_ak_issuance import approval as temp_ak_approval
            if temp_ak_approval.should_handle_event(data):
                threading.Thread(
                    target=temp_ak_approval.handle_temp_ak_event, args=(data,), daemon=True,
                ).start()
                return jsonify({"code": 0})
            if temp_ak_approval.should_handle_extend_event(data):
                threading.Thread(
                    target=temp_ak_approval.handle_temp_ak_extend_event, args=(data,), daemon=True,
                ).start()
                return jsonify({"code": 0})
        except Exception:
            logger.error("[temp_ak] 审批事件分发失败", exc_info=True)

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
    if not isinstance(data, dict):   # body 为 JSON 数组/标量时，后续 .get 会 AttributeError → 500
        data = {}

    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge", "")})

    # 验证 token —— 与 /feishu/event 同一把门。**这道校验是安全承重件，别拆**：
    # 本路由把请求体里的 open_id（operator.operator_id.open_id / data.open_id / data.user_id）
    # 当作操作人身份传给 _process_action，而多个 handler 拿它做管理员判定 —— OSS 权限下发
    # (_h_approve_oss_perm[_selected])、跨云/SSH/PFS 迁移的超阈值确认下发。少了这道门，任何人
    # POST 一个把 open_id 填成管理员的请求，就能过掉上述全部管理员门禁（open_id 在组织内可见，
    # 不是秘密）。事件订阅那条 card.action.trigger 走 /feishu/event、本来就过了同一把门，只有
    # 这条老式卡片回调路径此前是敞的。
    if not _token_verified(data):
        logger.warning(
            "[card_action] invalid token has_token=%s src=%s keys=%s",
            bool(_extract_request_token(data)),
            _token_source_hint(_extract_request_token(data)),   # 临时诊断，见该函数注释
            list(data.keys()),
        )
        return jsonify({"code": 1, "msg": "invalid token"}), 403

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




def _api_token_ok(expected: str) -> bool:
    """比对请求里带的 token。常数时间比较；两侧 encode 成 bytes（compare_digest 对非 ASCII 的
    str 会抛 TypeError，token 来自外部请求 → 不 encode 就是个可匿名触发的 500）。"""
    if not expected:
        return False
    auth = request.headers.get("Authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    supplied = request.headers.get("X-API-Token", "") or bearer or request.args.get("token", "")
    return hmac.compare_digest(str(supplied).encode("utf-8"), str(expected).encode("utf-8"))


def _ram_api_authorized() -> bool:
    """`/api/ram/user` 门禁：**只认 `RAM_QUERY_API_TOKEN`，不回退**。

    原实现回退到 `FEISHU_VERIFICATION_TOKEN`，而线上 `RAM_QUERY_API_TOKEN` 恰好为空 → 这个
    对外只读接口一直在拿 webhook 的验证 token 当钥匙。两个面共用一把钥匙意味着任一侧泄漏
    另一侧全开：拿到本接口 token 的人可以伪造 `/feishu/card_action`、把 open_id 填成管理员，
    过掉 OSS 权限下发、PFS 直传确认，以及跨云/SSH 迁移超阈值时的确认门（见 `_token_verified`
    的注释）。故 fail-closed：
    没配专用 token 就一律 403。
    """
    return _api_token_ok(getattr(settings, "RAM_QUERY_API_TOKEN", ""))


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
    """页面 token 门禁：**只认 `GPU_DIST_TOKEN`，不回退**（同 `_ram_api_authorized` 的理由）。

    这个页面的链接会被当按钮推进飞书群，token 明文拼在 URL query 里 —— 一旦回退到
    `FEISHU_VERIFICATION_TOKEN`，等于把 webhook 入站门的钥匙广播给群里每个人。
    """
    return _api_token_ok(getattr(settings, "GPU_DIST_TOKEN", ""))


@app.route("/gpu/distribution", methods=["GET"])
def gpu_distribution_page():
    """GPU 卡分布实时页面（自动刷新 HTML）。token 门禁：**只认 GPU_DIST_TOKEN，无回退**。"""
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
        if not settings.JIRA_ENABLED:
            result["jira"] = "disabled"
        elif settings.JIRA_URL and settings.JIRA_PAT:
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

    # ── 启动校验：验证 token 缺失 = 入站门全敞 ─────────────────────────────
    # 放在错误回调注册之后，好让这条 ERROR 直接推到管理员飞书。不做 SystemExit：未配置 token 的
    # 环境（隔离热备机）应当能起来，但必须让人看见门是敞的。
    if not settings.FEISHU_VERIFICATION_TOKEN:
        logger.error(
            "[启动校验] FEISHU_VERIFICATION_TOKEN 未配置 → /feishu/event 与 /feishu/card_action "
            "的入站校验全部跳过。card_action 的操作人 open_id 取自请求体，此时管理员门禁"
            "（OSS 权限下发 / 跨云·SSH·PFS 迁移下发）可被任意伪造请求绕过。"
            "请在 .env 填 FEISHU_VERIFICATION_TOKEN 后 force-recreate 容器（restart 不重载 env_file）。"
        )

    logger.info("正在初始化 Agent（首次加载模型，请稍候）...")
    try:
        from core.agent import _build_executor
        _build_executor()
        logger.info("Agent 初始化完成")
    except Exception as e:
        logger.error("Agent 初始化失败，Bot 将无法回复消息", exc_info=True)

    if settings.SCHEDULER_ENABLED:
        try:
            scheduler.start()
        except Exception as e:
            logger.error("调度器启动失败", exc_info=True)
    else:
        logger.warning("SCHEDULER_ENABLED=false → 后台调度器未启动（热备/双跑期，避免与 live 机重复推送）")

    logger.info("飞书 Bot 服务启动 → http://%s:%s/feishu/event", host, port)
    logger.info("卡片回调地址     → http://%s:%s/feishu/card_action", host, port)
    app.run(host=host, port=port, debug=debug)
