"""飞书消息发送原语：回复 / 图表卡片回复 / 主动发送。

包内其他模块统一 `from . import messaging` 后用 `messaging._feishu_reply(...)`
模块属性方式调用 —— 测试只需 patch 本模块即可覆盖所有调用点。
"""
import json

import requests

from utils.logger import get_logger

logger = get_logger(__name__)
from tools.feishu.cards import card, div, hr, img, note
from tools.feishu.notify import _get_access_token


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
            logger.warning("[飞书回复] 失败: %s", data.get('msg'))
    except Exception as e:
        logger.error("[飞书回复] 异常", exc_info=True)


def _feishu_reply_with_chart(message_id: str, text: str, image_key: str) -> None:
    """以交互卡片形式回复：文本回答 + 实时指标趋势图"""
    from datetime import datetime
    from tools.feishu.notify import _upload_image  # noqa: F401 (确保路径可用)
    payload = card(None, [
        div(text),
        hr(),
        img(image_key, "实时指标趋势图"),
        hr(),
        note(f"实时指标快照 · {datetime.now().strftime('%H:%M:%S')}"),
    ])
    try:
        token = _get_access_token()
        resp  = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "interactive", "content": json.dumps(payload)},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("[飞书卡片回复] 失败: %s", data.get('msg'))
    except Exception as e:
        logger.error("[飞书卡片回复] 异常", exc_info=True)


def _feishu_reply_card(message_id: str, card_payload: dict) -> None:
    """以交互卡片形式回复指定消息（card_payload 为完整卡片 dict）。"""
    try:
        token = _get_access_token()
        resp = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "interactive", "content": json.dumps(card_payload)},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("[飞书卡片回复] 失败: %s", data.get('msg'))
    except Exception:
        logger.error("[飞书卡片回复] 异常", exc_info=True)


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


def _send_text_to(open_id: str, chat_id: str, text: str) -> None:
    """向用户发文本消息（优先私信，降级群聊）。"""
    try:
        token     = _get_access_token()
        target_id = open_id or chat_id
        id_type   = "open_id" if open_id else "chat_id"
        requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            params={"receive_id_type": id_type},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"receive_id": target_id, "msg_type": "text",
                  "content": json.dumps({"text": text})},
            timeout=15,
        )
    except Exception as e:
        logger.error("[发消息] 失败", exc_info=True)
