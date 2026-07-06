"""消息事件处理：事件去重 / GPU 意图识别 / 身份绑定指令 / 我的实例 / Agent 调用。"""
import json
import re
import threading
import time

import requests

from utils.logger import get_logger, clear_trace_id

logger = get_logger(__name__)
from tools.feishu.cards import btn, buttons, card, div, fields, hr
from tools.feishu.notify import _get_access_token
from core.dsw_scheduler import _redis_get, _all_tracked_keys
from . import gpu_flow, messaging

# ── 去重：同一 event_id 只处理一次（飞书会重试未 200 的请求）─────────────────
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


# ── GPU 申请意图 ──────────────────────────────────────────────────────────────

# GPU 申请意图：资源词 + 动作词同时出现，或明确的训练短语
_GPU_RESOURCE_WORDS = ("gpu", "dsw", "显卡", "算力", "实例", "训练资源")
_GPU_ACTION_WORDS   = ("申请", "需要", "开通", "租用")
_GPU_TRAIN_PHRASES  = ("我要训练", "跑模型", "训练模型", "跑训练", "微调模型",
                       "finetune", "fine-tune", "预训练")
# 查询/管理意图词：命中则不触发申请流程
_GPU_QUERY_WORDS    = ("查看", "查询", "列出", "列表", "看一下", "有哪些", "状态",
                       "显示", "我的实例", "已有", "运行中", "停止", "删除", "关闭")


# ── 跨云迁移意图：出现迁移动作词 + 含迁移路径（tos:// 或 oss://）─────────────────
_TRANSFER_SCHEMES = "tos|oss|cpfs|vepfs"
_TRANSFER_PATH_RE = re.compile(rf"\b(?:{_TRANSFER_SCHEMES})://[^\s，,。；;]+", re.IGNORECASE)
_TRANSFER_NEXT_PATH_RE = re.compile(rf"\s*(?:到|至|->|=>|→)\s*(?=(?:{_TRANSFER_SCHEMES})://)", re.IGNORECASE)
_TRANSFER_ACTION_WORDS = ("迁移", "搬", "传输", "转移", "同步到", "拷到", "拷贝到", "搬运", "导入", "导出")
_TRANSFER_DIRECTION_RE = re.compile(r"\b(?:tos|oss|cpfs|vepfs)\s*(?:到|至|->|=>|→)\s*(?:tos|oss|cpfs|vepfs)\b", re.IGNORECASE)


def _extract_transfer_paths(text: str):
    """从文本里抽取迁移路径，返回 (source, dest)；不足返回 (None, None)。"""
    normalized = _TRANSFER_NEXT_PATH_RE.sub(" ", text or "")
    spans = []
    for match in _TRANSFER_PATH_RE.finditer(normalized):
        path = match.group(0).strip().strip("`'\"<>[]()（）")
        path = re.sub(r"(?:到|至|->|=>|→)+$", "", path).rstrip("，,。；;")
        if path:
            spans.append(path)
    if not spans:
        return None, None
    source = spans[0]
    dest = spans[1] if len(spans) > 1 else ""
    return source, dest


def _is_transfer_intent(text: str) -> bool:
    """含迁移动作词且带路径，或同时给出源/目的两条路径。"""
    source, dest = _extract_transfer_paths(text)
    if not source:
        return False
    if dest:
        return True
    return any(w in text for w in _TRANSFER_ACTION_WORDS)


# 录入意图：纯迁移关键词、无路径 → 弹录入卡让用户填地址
_TRANSFER_ENTRY_WORDS = (
    "跨云迁移", "数据迁移", "迁移数据", "新建迁移", "发起迁移",
    "跨平台对象存储迁移", "对象存储迁移", "对象存储传输", "跨平台迁移",
    "tos到oss", "tos 到 oss", "tos->oss", "tos→oss", "oss到tos", "oss 到 tos",
    "迁移卡片", "填写迁移", "填迁移", "发起传输",
)

_SINK_PREHEAT_ENTRY_WORDS = (
    "数据沉降/预热", "数据沉降", "数据预热", "沉降/预热", "沉降预热",
    "cepfs沉降", "cpfs沉降", "cpfs预热", "文件沉降", "数据加载",
)


def _is_sink_preheat_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(word.replace(" ", "").lower() in compact for word in _SINK_PREHEAT_ENTRY_WORDS)


# 火山 vePFS↔TOS 数据流动：排在 CPFS 意图之前，避免"沉降/预热"被 CPFS 抢走
_VEPFS_DATAFLOW_ENTRY_WORDS = (
    "vepfs预热", "vepfs沉降", "vepfs数据流动", "vepfs数据沉降", "vepfs数据预热",
    "火山预热", "火山沉降", "火山数据沉降", "火山数据预热", "火山vepfs",
    "tos预热", "tos沉降",
)


def _is_vepfs_dataflow_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(word.replace(" ", "").lower() in compact for word in _VEPFS_DATAFLOW_ENTRY_WORDS)


# 算力（MFU）日报：点菜单按钮发这些词 / 直接打字 → 回算力效率卡（不必等每日早报）
_MFU_REPORT_RE = re.compile(r"(算力日报|算力效率|算力报告|集群算力|算力监控|mfu)", re.IGNORECASE)


def _is_mfu_report_intent(text: str) -> bool:
    return bool(_MFU_REPORT_RE.search(text or ""))


# 桶间迁移（同云一次性搬运）：发这些词 → 弹桶间迁移卡（排在跨云 transfer 意图之前）
_BUCKET_TRANSFER_ENTRY_WORDS = (
    "桶间迁移", "桶间搬运", "同云迁移", "同区迁移", "桶迁移", "bucket迁移",
    "oss迁移oss", "oss到oss", "tos到tos", "tos迁移tos",
)


def _is_bucket_transfer_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(w.replace(" ", "").lower() in compact for w in _BUCKET_TRANSFER_ENTRY_WORDS)


def _is_transfer_entry_intent(text: str) -> bool:
    """命中录入关键词且没带路径（带路径走 _is_transfer_intent 直接解析）。"""
    if _extract_transfer_paths(text)[0]:
        return False
    compact = re.sub(r"\s+", "", text or "").lower()
    if any(w.replace(" ", "") in compact for w in _TRANSFER_ENTRY_WORDS):
        return True
    if _TRANSFER_DIRECTION_RE.search(text or ""):
        return True
    if "卡片" in compact and any(w in compact for w in ("填", "迁移", "传输", "tos", "oss", "卡片消息")):
        return True
    return False

_TRANSFER_CONFIRM_RE = re.compile(r"^\s*(确认|确定|开始|执行|提交|确认迁移|开始迁移)\s*$")


def _is_transfer_confirm_text(text: str) -> bool:
    return bool(_TRANSFER_CONFIRM_RE.match(text or ""))


# RAM account query intent. Text/menu entry opens a form; actual query runs only after submit.
_RAM_QUERY_ENTRY_WORDS = (
    "queryramaccount", "queryramuser", "ramaccountquery", "ramuserquery",
    "aliyunaccountquery", "aliyunuserquery", "aliyunram", "aliyun ram",
    "\u963f\u91cc\u4e91ram", "\u963f\u91cc\u4e91 ram", "\u963f\u91cc\u4e91RAM", "\u963f\u91cc\u4e91 RAM",
    "ram\u8d26\u53f7\u67e5\u8be2", "ram\u8d26\u6237\u67e5\u8be2", "\u67e5\u8be2ram\u8d26\u53f7", "\u67e5\u8be2ram\u8d26\u6237",
    "\u67e5ram\u8d26\u53f7", "\u67e5ram\u8d26\u6237", "\u67e5\u8be2\u8d26\u53f7", "\u67e5\u8be2\u8d26\u6237",
    "\u67e5\u8d26\u53f7", "\u67e5\u8d26\u6237",
    "\u963f\u91cc\u4e91\u8d26\u53f7\u67e5\u8be2", "\u963f\u91cc\u4e91\u8d26\u6237\u67e5\u8be2",
    "\u67e5\u8be2\u963f\u91cc\u4e91\u8d26\u53f7", "\u67e5\u8be2\u963f\u91cc\u4e91\u8d26\u6237",
)

_VOLCANO_ACCOUNT_QUERY_ENTRY_WORDS = (
    "volcanoaccountquery", "volcanouserquery", "volcengineaccountquery", "volcengineuserquery",
    "volcanoiam", "volcengineiam", "volcano iam", "volcengine iam",
    "\u706b\u5c71\u5f15\u64ceiam", "\u706b\u5c71\u5f15\u64ce iam", "\u706b\u5c71iam", "\u706b\u5c71 iam",
    "\u706b\u5c71\u5f15\u64ce\u8d26\u53f7\u67e5\u8be2", "\u706b\u5c71\u5f15\u64ce\u8d26\u6237\u67e5\u8be2",
    "\u706b\u5c71\u8d26\u53f7\u67e5\u8be2", "\u706b\u5c71\u8d26\u6237\u67e5\u8be2",
    "\u67e5\u8be2\u706b\u5c71\u5f15\u64ce\u8d26\u53f7", "\u67e5\u8be2\u706b\u5c71\u5f15\u64ce\u8d26\u6237",
)


def _is_ram_query_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(word.lower() in compact for word in _RAM_QUERY_ENTRY_WORDS)

def _is_volcano_account_query_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(word.lower() in compact for word in _VOLCANO_ACCOUNT_QUERY_ENTRY_WORDS)


# 进度查询：拦在 Agent 之前，避免 LLM 劫持 + 吃旧会话历史
_PROGRESS_QUERY_RE = re.compile(r"(查询进度|进度查询|查进度|任务进度|查询任务)")
_JOB_ID_RE = re.compile(r"\b(vepfs-[0-9a-fA-F]{6,}|cpfs-[0-9a-fA-F]{6,}|tr-[0-9a-fA-F]{6,})\b")


def _is_progress_query_text(text: str) -> bool:
    return bool(_PROGRESS_QUERY_RE.search(text or ""))


def _handle_progress_query(message_id: str, text: str, open_id: str = "") -> None:
    """按任务ID直接查进度（cpfs-→CPFS 预热/沉降；tr-→跨云迁移）。无ID→弹输入卡让用户填 ID。"""
    m = _JOB_ID_RE.search(text or "")
    if not m:
        from core.cpfs_dataflow.cards import query_input_card
        messaging._feishu_reply_card(message_id, query_input_card())
        return
    jid = m.group(1)
    try:
        if jid.lower().startswith("cpfs-"):
            from core.cpfs_dataflow import orchestrator as o
            job = o.get_job(jid)
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            msg = (f"任务 `{jid}`：**{job['stage']}** {job.get('operation_label','')}\n"
                   f"进度 {job.get('files_done',0)}/{job.get('files_total',0)} 文件，"
                   f"{o.fmt_size(job.get('bytes_done',0))}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        else:  # tr-
            from core.transfer import orchestrator as o
            job = o.get_job(jid)
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            msg = (f"任务 `{jid}`：**{job['stage']}** {job.get('direction','')}\n"
                   f"{o.fmt_size(job.get('bytes_total',0))}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        messaging._feishu_reply(message_id, msg)
    except Exception as e:
        messaging._feishu_reply(message_id, f"查询失败：{e}")

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
        elements.append(fields(
            (state.get("instance_name", "-"), f"工单 {key}"),
            ("剩余时长", f"{remaining_h:.1f} h"),
            ("已使用", f"{elapsed_h:.1f} h"),
            ("GPU 卡数", f"{gpu_cnt} 张"),
        ))
        elements.append(buttons(
            btn("⏩ 延续 2h", {"action": "extend_dsw", "ticket_key": key,
                              "instance_id": state.get("instance_id", ""), "extend_hours": "2"},
                "primary"),
            btn("🛑 立即停止", {"action": "stop_dsw", "ticket_key": key,
                               "instance_id": state.get("instance_id", "")},
                "danger"),
        ))
        elements.append(hr())

    payload = card(f"我的 GPU 实例（{len(user_tickets)} 个）", [
        div(f"**共占用 GPU：** {total_gpu} 张"),
        hr(),
    ] + elements)
    try:
        token = _get_access_token()
        requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "interactive", "content": json.dumps(payload)},
            timeout=15,
        )
    except Exception as e:
        logger.error("[我的实例] 发送失败", exc_info=True)
        messaging._feishu_reply(message_id, "获取实例信息失败，请稍后重试。")


# ── 跨云迁移意图处理 ──────────────────────────────────────────────────────────

def _handle_transfer_intent(message_id: str, chat_id: str, open_id: str,
                            source: str, dest: str) -> None:
    """解析迁移路径 → 预估大小 → 回发确认表单卡。错误回纯文本。"""
    from core.transfer import orchestrator
    from core.transfer.cards import confirm_card
    from core.transfer.paths import PathError
    try:
        plan = orchestrator.make_plan(source, dest or "")
        if plan.engine not in ("mgw", "tos_mig"):
            messaging._feishu_reply(
                message_id,
                f"⚠️ 方向 {plan.direction} 暂未支持（三期沉降段）。")
            return
        bytes_total, objects_total = orchestrator.estimate_source(plan)
        job = orchestrator.create_job_record(
            plan, open_id=open_id, bytes_total=bytes_total, objects_total=objects_total)
        need_appr = orchestrator.needs_approval(bytes_total)
        payload = confirm_card(job, need_approval=need_appr)
        token = _get_access_token()
        requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "interactive", "content": json.dumps(payload)},
            timeout=15,
        )
    except PathError as e:
        messaging._feishu_reply(message_id, f"❌ 路径错误：{e}")
    except Exception as e:
        logger.error("[Transfer] 意图处理失败", exc_info=True)
        messaging._feishu_reply(message_id, f"❌ 迁移请求处理失败：{e}")


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

    # ②.5 进度查询（拦在 Agent 之前，避免 LLM 劫持 + 吃旧会话历史）
    if _is_progress_query_text(user_text):
        _handle_progress_query(message_id, user_text, open_id)
        return

    # ②.6 算力（MFU）日报：点按钮/发关键词���时看（拦在 GPU 意图之前，"算力"是 GPU 资源词会被抢）
    if _is_mfu_report_intent(user_text):
        from tools.aliyun.cluster_mfu import mfu_card_for_callback
        card = mfu_card_for_callback(view="summary")   # 只读缓存秒回；陈旧则后台刷新
        if card is None:                               # 缓存全无 → 已触发后台采集
            messaging._feishu_reply(message_id, "📊 算力数据采集中（约1分钟），请稍后再发一次「算力日报」。")
        else:
            messaging._feishu_reply_card(message_id, card)
        return

    # ③ 跨云迁移录入意图（纯关键词、无路径）→ 弹录入卡让用户填地址
    # 桶间迁移（同云一次性搬运）→ 弹桶间迁移卡（排在跨云迁移意图之前，避免"迁移"被抢）
    if _is_bucket_transfer_entry_intent(user_text):
        from core.bucket_transfer.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    # 火山 vePFS↔TOS 数据流动（排在 CPFS 之前，"vepfs沉降"等由此接管）
    if _is_vepfs_dataflow_entry_intent(user_text):
        from core.vepfs_dataflow.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    if _is_sink_preheat_entry_intent(user_text):
        from core.cpfs_dataflow.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    if _is_volcano_account_query_entry_intent(user_text):
        from core.volcano_iam_query_cards import query_entry_card
        messaging._feishu_reply_card(message_id, query_entry_card())
        return

    if _is_ram_query_entry_intent(user_text):
        from core.ram_query_cards import query_entry_card
        messaging._feishu_reply_card(message_id, query_entry_card())
        return

    if _is_transfer_entry_intent(user_text):
        from core.transfer.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    # ③.5 跨云迁移意图（含迁移路径 + 动作词）→ 解析、预估、弹确认卡
    if _is_transfer_intent(user_text):
        source, dest = _extract_transfer_paths(user_text)
        threading.Thread(
            target=_handle_transfer_intent,
            args=(message_id, chat_id, open_id, source, dest), daemon=True
        ).start()
        return

    # ③.6 纯文本确认不提交迁移，避免 Agent 编造任务 ID；必须点击确认卡按钮。
    if _is_transfer_confirm_text(user_text):
        messaging._feishu_reply(message_id, "请点击上一张迁移确认卡里的“确认迁移”按钮；只回复文字不会提交迁移任务。")
        return

    # ④ GPU 申请意图 → 必须先注册个人 AK/SK
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
