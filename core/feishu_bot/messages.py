"""消息事件处理：事件去重 / GPU 意图识别 / 身份绑定指令 / 我的实例 / Agent 调用。"""
import json
import re
import threading
import time

import requests

from config.settings import settings
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

# 数据预热/沉降统一入口：阿里 CPFS↔OSS + 火山 vePFS↔TOS 共用一张录入卡（卡内选云平台）。
_SINK_PREHEAT_ENTRY_WORDS = (
    "数据沉降/预热", "数据沉降", "数据预热", "沉降/预热", "沉降预热",
    "cepfs沉降", "cpfs沉降", "cpfs预热", "文件沉降", "数据加载",
    # 火山 vePFS↔TOS 关键词也进入同一张卡（用户在卡里选火山）
    "vepfs预热", "vepfs沉降", "vepfs数据流动", "vepfs数据沉降", "vepfs数据预热",
    "火山预热", "火山沉降", "火山数据沉降", "火山数据预热", "火山vepfs",
    "tos预热", "tos沉降",
)


def _is_sink_preheat_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(word.replace(" ", "").lower() in compact for word in _SINK_PREHEAT_ENTRY_WORDS)


# 算力（MFU）日报：点菜单按钮发这些词 / 直接打字 → 回算力效率卡（不必等每日早报）
_MFU_REPORT_RE = re.compile(r"(算力日报|算力效率|算力报告|集群算力|算力监控|mfu)", re.IGNORECASE)


def _is_mfu_report_intent(text: str) -> bool:
    return bool(_MFU_REPORT_RE.search(text or ""))


# 帮助/能力菜单意图：即时回能力清单，不走 LLM
_HELP_RE = re.compile(r"(帮助|help|能做什么|会做什么|你会什么|有什么功能|功能列表|使用说明|怎么用你|菜单)", re.IGNORECASE)


def _is_help_intent(text: str) -> bool:
    return bool(_HELP_RE.search(text or ""))


# 指标趋势图意图：仅当问题确与监控/指标相关时，才在 Agent 回复后附一张实时趋势图。
# 否则「知识库/工单/迁移」等无关回复也会被塞一张 CPU/内存曲线——又慢又费又误导。
# 命中一次省掉：fetch_raw_series(云查询) + matplotlib 渲染 + 图片上传 + 取 token。
# 注意：不放裸 `gpu`——"查我的 gpu 工单"是工单类问句、不该附监控图；真监控问句会命中
# `利用率/使用率/显存/监控/趋势/mfu/集群状态` 等词或走 monitor/cluster intent。
_METRICS_CHART_RE = re.compile(
    r"(cpu|内存|memory|显存|利用率|使用率|指标|趋势|监控|负载|水位|"
    r"prometheus|grafana|mfu|算力|集群状态|metrics)", re.IGNORECASE)
_METRICS_CHART_INTENTS = {"monitor", "cluster"}


def _wants_metrics_chart(text: str, intents=None) -> bool:
    if intents and _METRICS_CHART_INTENTS.intersection(intents):
        return True
    return bool(_METRICS_CHART_RE.search(text or ""))


def _capability_menu() -> str:
    """用户面能力清单：帮助意图直接回它；未知意图兜底也复用它引导用户。"""
    return (
        "🤖 我能帮你做这些（直接用自然语言说需求即可）：\n"
        "• GPU 卡分布 / 谁在用卡 —— 发「卡分布」\n"
        "• 集群算力效率日报 MFU —— 发「算力日报」\n"
        "• GPU 申请 / DSW 实例开通、启停、查询\n"
        "• OSS / TOS 对象存储：用量、各子目录大小、目录树\n"
        "• 跨云 / 桶间数据迁移（tos ↔ oss，直接发路径）\n"
        "• CPFS / vePFS 数据预热与沉降\n"
        "• RAM 账号查询、OSS 权限对账下发\n"
        "• Jira 工单 / GPU 申请记录、GitHub 工作流\n"
        "• K8s/Pod 运维、Prometheus 指标趋势、知识库问答\n"
        "\n例：「看卡分布」「算力日报」「杭州 OSS 用量」「把 tos://a/ 迁到 oss://b/」"
    )


# GPU 卡分布：地区×卡型 + 每用户在算卡数（拦在 GPU 申请意图之前，"卡"是 GPU 资源词会被抢）
_GPU_DIST_RE = re.compile(
    r"(卡分布|显卡分布|gpu分布|卡用量|用卡情况|卡统计|卡占用|卡的分布|谁在用卡|用了多少卡|卡使用情况)",
    re.IGNORECASE)


def _is_gpu_dist_intent(text: str) -> bool:
    return bool(_GPU_DIST_RE.search(text or ""))


# 桶间迁移（同云一次性搬运）：发这些词 → 弹桶间迁移卡（排在跨云 transfer 意图之前）
_BUCKET_TRANSFER_ENTRY_WORDS = (
    "桶间迁移", "桶间搬运", "同云迁移", "同区迁移", "桶迁移", "bucket迁移",
    "oss迁移oss", "oss到oss", "tos到tos", "tos迁移tos",
)


def _is_bucket_transfer_entry_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(w.replace(" ", "").lower() in compact for w in _BUCKET_TRANSFER_ENTRY_WORDS)


# SSH 迁移链（杭州OSS→新加坡→泰国服务器）：触发词「数据迁移（泰国H200）」及变体。
# 注意：含「迁移」会与跨云 transfer 意图撞，必须在 _process_message 里排在 transfer 意图之前判定。
_SSH_TRANSFER_WORDS = (
    "数据迁移泰国", "泰国h200", "泰国迁移", "迁移泰国", "迁到泰国", "传到泰国",
    "h200迁移", "泰国数据迁移", "迁移到泰国", "sshtransfer",
)


def _is_ssh_transfer_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower().replace("（", "(").replace("）", ")")
    if any(w in compact for w in _SSH_TRANSFER_WORDS):
        return True
    return "泰国" in compact and ("迁移" in compact or "h200" in compact)


# PFS 跨云直传（vePFS↔CPFS）：同时提到两种 PFS 名 = 明确 PFS↔PFS（区别于只提一个 PFS+对象存储的
# 预热/沉降）；或含明确直传词。须排在预热/沉降入口之前，避免被 _is_sink_preheat_entry_intent 吃掉。
_PFS_TRANSFER_WORDS = ("pfs直传", "pfs跨云", "跨云pfs", "pfs互传", "pfs之间")


def _is_pfs_transfer_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    if "vepfs" in compact and "cpfs" in compact:
        return True
    return any(w in compact for w in _PFS_TRANSFER_WORDS)


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
_JOB_ID_RE = re.compile(r"\b(vepfs-[0-9a-fA-F]{6,}|cpfs-[0-9a-fA-F]{6,}|tr-[0-9a-fA-F]{6,}|sgp-[0-9a-fA-F]{6,}|xpfs-[0-9a-fA-F]{6,})\b")


def _is_progress_query_text(text: str) -> bool:
    return bool(_PROGRESS_QUERY_RE.search(text or ""))


def _push_terminal_result_card(chain: str, jid: str, job: dict) -> None:
    """按 ID 查询首次把任务从在途刷成终态时，除文本外补推一张富结果卡（带重试按钮）到发起人/
    配置频道，经 `_claim_dataflow_notify` NX 闸门与在线线程/对账去重（谁先到谁推、只推一次）。

    必要性：refresh 会把 Redis 写成 DONE/FAILED，而对账 `_reconcile_dataflow_once` 的 active 集
    只含在途态→之后永久跳过该 job；在线轮询线程又随容器重启已死。若这里不补推，重启场景下富
    结果卡（尤其 FAILED 的「重试」按钮）+ 频道完成/失败通知会永远丢。跑在消息后台线程，可同步推。"""
    try:
        from core.dsw_scheduler import _claim_dataflow_notify, _send_card
        from core.feishu_bot import actions
        if chain == "vepfs":
            from core.vepfs_dataflow.cards import result_card
            chat = actions._cfg_vepfs_chat()
        elif chain == "cpfs":
            from core.cpfs_dataflow.cards import result_card
            chat = actions._cfg_cpfs_chat()
        elif chain == "ssh":
            from core.ssh_transfer.cards import result_card
            chat = actions._cfg_ssh_chat()
        elif chain == "pfs":
            from core.pfs_transfer.cards import result_card
            chat = actions._cfg_pfs_chat()
        else:
            from core.transfer.cards import result_card
            chat = actions._cfg_chat()
        if _claim_dataflow_notify(jid):
            _send_card(job.get("created_by", ""), chat, result_card(job))
    except Exception:
        logger.error("[进度查询] 终态补推结果卡失败 job=%s", jid, exc_info=True)


def _handle_progress_query(message_id: str, text: str, open_id: str = "") -> None:
    """按任务ID直接查进度（cpfs-→CPFS 预热/沉降；tr-→跨云迁移）。无ID→弹输入卡让用户填 ID。"""
    m = _JOB_ID_RE.search(text or "")
    if not m:
        from core.cpfs_dataflow.cards import query_input_card
        messaging._feishu_reply_card(message_id, query_input_card())
        return
    jid = m.group(1)
    try:
        if jid.lower().startswith("vepfs-"):
            from core.vepfs_dataflow import orchestrator as o
            chain = "vepfs"
            job = o.refresh(jid) or o.get_job(jid)   # 实时重查云端（自愈重启后死掉的轮询线程）
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            msg = (f"任务 `{jid}`：**{job['stage']}** {job.get('operation_label','')}\n"
                   f"进度 {job.get('files_done',0)}/{job.get('files_total',0)} 文件，"
                   f"{o.fmt_size(job.get('bytes_done',0))}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        elif jid.lower().startswith("cpfs-"):
            from core.cpfs_dataflow import orchestrator as o
            chain = "cpfs"
            job = o.refresh(jid) or o.get_job(jid)   # 实时重查云端（自愈重启后死掉的轮询线程）
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            msg = (f"任务 `{jid}`：**{job['stage']}** {job.get('operation_label','')}\n"
                   f"进度 {job.get('files_done',0)}/{job.get('files_total',0)} 文件，"
                   f"{o.fmt_size(job.get('bytes_done',0))}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        elif jid.lower().startswith("sgp-"):
            from core.ssh_transfer import orchestrator as o
            chain = "ssh"
            job = o.refresh(jid) or o.get_job(jid)   # 实时重查（自愈重启后死掉的轮询线程）
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            msg = (f"任务 `{jid}`：**{job['stage']}** {o.stage_label(job['stage'])}\n"
                   f"进度 {o.progress_line(job)}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        elif jid.lower().startswith("xpfs-"):
            from core.pfs_transfer import orchestrator as o
            chain = "pfs"
            job = o.refresh(jid) or o.get_job(jid)   # 实时重查（定位当前段续推，自愈重启死掉的线程）
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            msg = (f"任务 `{jid}`：**{job['stage']}** {job.get('direction','')}\n"
                   f"段完成 沉降={job.get('sink_done')} 跨云={job.get('cross_done')} 预热={job.get('preheat_done')}\n"
                   f"进度 {o.progress_line(job)}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        else:  # tr-
            from core.transfer import orchestrator as o
            chain = "transfer"
            job = o.refresh(jid) or o.get_job(jid)   # 实时重查云端（自愈重启后死掉的轮询线程）
            if not job:
                messaging._feishu_reply(message_id, f"未找到任务 `{jid}`（可能已过期）。")
                return
            done = o.fmt_size(job.get("bytes_done", 0)) if job.get("bytes_done") else ""
            total = o.fmt_size(job.get("bytes_total", 0))
            prog = f"{done} / {total}" if done else total
            msg = (f"任务 `{jid}`：**{job['stage']}** {job.get('direction','')}\n"
                   f"进度 {prog}"
                   + (f"\n错误：{job['error']}" if job.get('error') else ""))
        messaging._feishu_reply(message_id, msg)
        # 若本次查询正是那次把任务刷成终态的观测：补推富结果卡（对账此后不再管终态 job）。
        if job["stage"] in (o.STAGE_DONE, o.STAGE_FAILED):
            _push_terminal_result_card(chain, jid, job)
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
        if card is not None:
            messaging._feishu_reply_card(message_id, card)
        else:
            # 缓存全无（早报没跑/Redis 重启）：先即时反馈，再后台采集(约1分钟)，完成后**自动把日报卡发给用户**，
            # 不再要求"稍后再发一次"（消息路径不受卡片回调 3s 限制，可放心后台采集）。
            messaging._feishu_reply(message_id, "📊 算力数据采集中（约1分钟），完成后自动发给你…")

            def _build_and_push():
                try:
                    from tools.aliyun.cluster_mfu import build_mfu_card
                    c = build_mfu_card(view="summary", refresh=True)
                    messaging._feishu_reply_card(message_id, c)
                except Exception:
                    logger.error("[MFU] 后台采集日报失败", exc_info=True)
                    messaging._feishu_reply(message_id, "❌ 算力日报采集失败，请稍后重试。")

            threading.Thread(target=_build_and_push, daemon=True).start()
        return

    # ②.7 GPU 卡分布（地区×卡型 + 每用户在算卡数）：回摘要卡 + 实时页面链接（拦在 GPU 申请意图之前）
    if _is_gpu_dist_intent(user_text):
        def _reply_dist():
            try:
                from tools.aliyun.gpu_distribution import get_distribution, summary_card, dist_url
                g = get_distribution()
                messaging._feishu_reply_card(message_id, summary_card(g, dist_url()))
            except Exception:
                logger.error("[GPUDIST] 回复卡分布失败", exc_info=True)
                messaging._feishu_reply(message_id, "❌ 卡分布查询失败，请稍后重试。")

        threading.Thread(target=_reply_dist, daemon=True).start()
        return

    # ③ 跨云迁移录入意图（纯关键词、无路径）→ 弹录入卡让用户填地址
    # PFS 跨云直传（vePFS↔CPFS）：排在预热/沉降与 transfer/ssh 意图之前——「同时提到两种 PFS」
    # 是 PFS↔PFS 直传的明确信号，须先拦，避免被单 PFS 的预热/沉降或泛「迁移」词抢走。
    if settings.PFS_TRANSFER_ENABLED and _is_pfs_transfer_intent(user_text):
        from core.pfs_transfer.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    # 桶间迁移（同云一次性搬运）→ 弹桶间迁移卡（排在跨云迁移意图之前，避免"迁移"被抢）
    # SSH 迁移链（泰国H200）：排在跨云 transfer 意图之前——「数据迁移」会被 transfer 入口误抢。
    if _is_ssh_transfer_intent(user_text):
        from core.ssh_transfer.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    if _is_bucket_transfer_entry_intent(user_text):
        from core.bucket_transfer.cards import entry_card
        messaging._feishu_reply_card(message_id, entry_card())
        return

    # 数据预热/沉降统一入口（选云 → 选地区 → 选文件系统 + 填源/目的地址）
    if _is_sink_preheat_entry_intent(user_text):
        from core.dataflow_cards import entry_card
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
        # Jira 停用时优雅停用 GPU 申请（依赖 Jira 建工单→调度器建 DSW）：即时回提示、不弹卡、不置 state。
        if not settings.JIRA_ENABLED:
            messaging._feishu_reply(message_id, "GPU 申请暂停（工单系统 Jira 停用中），请联系运维人员。")
            return
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

    # 帮助/能力菜单：即时回清单，不进 Agent
    if _is_help_intent(user_text):
        messaging._feishu_reply(message_id, _capability_menu())
        return

    # 延迟导入，避免在模块加载时触发 LLM 初始化
    from core.agent import (_build_executor, _load_history, _load_summary, _save_turn,
                            select_tools_scoped, _names_to_tools)
    from tools import TOOL_GROUPS

    # 会话历史按【用户 open_id】隔离，而非群聊 chat_id——群里不同用户共用一个 chat_id 会
    # 串用户/串话题（曾导致把别人的迁移任务当成本条消息的答案）。无 open_id（私聊/异常）回退 chat_id。
    session_id = open_id or chat_id
    history = _load_history(session_id)

    # 先发"思考中"提示，让用户知道 Bot 在处理
    messaging._feishu_reply(message_id, "🤔 正在分析，请稍候...")

    # 按当前消息缩小工具集：不相关消息拿不到 manage_transfer 等，从源头杜绝乱调
    tools, intents = select_tools_scoped(user_text)
    base_input = f"{user_text}\n[feishu_open_id={open_id}]" if open_id else user_text
    if tools is None:
        # 未知意图：只给知识库(RAG)，并约束"从知识库答 / 列能力并反问，绝不臆造或乱调工具"
        tools = _names_to_tools(TOOL_GROUPS["knowledge"])
        agent_input = (
            "【路由提示】未能识别用户想做的具体运维操作。请：先用知识库工具尝试回答；"
            "若知识库无相关答案，就用下面的能力清单告诉用户你能做什么、并反问他具体想做什么；"
            "禁止调用任何其它工具、禁止编造任务/迁移/实例的状态或进度。\n"
            f"能力清单：\n{_capability_menu()}\n\n用户消息：{base_input}"
        )
    else:
        agent_input = base_input
    # 滚动摘要（更早对话的压缩）作前缀拼进输入，给长对话连贯性；不作独立 system 消息（GLM 兼容更稳）。
    _summary = _load_summary(session_id)
    if _summary:
        agent_input = ("（以下为你与该用户更早对话的摘要，仅供连贯理解、请勿主动复述）：\n"
                       f"{_summary}\n\n———\n{agent_input}")
    logger.info("[对话] 意图=%s 工具数=%d 摘要=%s", intents or "unknown", len(tools), bool(_summary))

    try:
        result = _build_executor(tools).invoke({"input": agent_input, "chat_history": history})
        reply  = result["output"]

        # 清除 LLM 可能在回复文本中虚构的 Markdown 图片占位符
        # 例如 "![图](https://...)" 或 "!https://..." —— 飞书无法渲染这类链接
        reply = re.sub(r"!\[.*?\]\(.*?\)", "", reply)
        reply = re.sub(r"!https?://\S+", "", reply).strip()

        # 写回历史时用原始 user_text（不含注入的 open_id 元数据）；按用户隔离
        _save_turn(session_id, user_text, reply)

        # 仅在问题与监控/指标相关时才附实时趋势图；无关问题直接纯文本回复
        # （省掉云查询+渲染+上传，且不给知识库/工单/迁移等回复塞无关曲线）。
        if _wants_metrics_chart(user_text, intents):
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
        else:
            messaging._feishu_reply(message_id, reply)

    except Exception as e:
        logger.error("处理消息出错", exc_info=True)
        messaging._feishu_reply(message_id, f"⚠️ 处理出错：{e}")
    finally:
        clear_trace_id()
