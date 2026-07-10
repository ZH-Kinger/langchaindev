"""GPU 申请流程：申请卡片 / AK 注册卡片 / 申请状态（Redis）/ 文本解析 / 工单提交。"""
import json
import re
import threading
import time

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)
from tools.feishu.cards import btn, buttons, card, div, hr
from tools.feishu.notify import _get_access_token
from tools.jira.ticket import create_gpu_ticket
from . import messaging


# ── GPU 申请卡片（action buttons，programmatic interactive 消息支持）────────────
# form + submit_form 仅限飞书卡片构建器模板（msg_type: "template"），
# 编程发送的 interactive 消息不渲染 form 块，改用 action buttons 分两步完成申请。
_GPU_REQUEST_CARD = card("GPU 资源申请", [
    div("**第一步：选择常用配置**（点击后卡片更新，再补充实例名和用途即可提交）"),
    hr(),
    buttons(
        btn("1 GPU · 8h\n小模型微调",
            {"action": "quick_gpu", "gpu_count": "1", "duration_hours": "8"}),
        btn("4 GPU · 24h\n7B 模型训练",
            {"action": "quick_gpu", "gpu_count": "4", "duration_hours": "24"}, "primary"),
        btn("8 GPU · 48h\n大模型预训练",
            {"action": "quick_gpu", "gpu_count": "8", "duration_hours": "48"}, "danger"),
    ),
    hr(),
    div("或直接回复自定义参数：\n"
        "```\n实例名: wzh-train-01\nGPU数: 4\n时长: 24\n用途: 大语言模型微调\n```"),
])

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
    # Jira 停用兜底：文本解析路径也可能到这里建工单，堵住以防建孤儿工单+假成功回执。
    if not settings.JIRA_ENABLED:
        messaging._feishu_reply(message_id, "GPU 申请暂停（工单系统 Jira 停用中），请联系运维人员。")
        return
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
