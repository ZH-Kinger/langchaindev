"""
飞书消息 Webhook 服务。
飞书将用户消息事件 POST 到本服务 → Agent 处理 → 回复原消息。

启动方式：
    python main.py --mode bot
或：
    python -m core.feishu_bot

飞书开发者后台配置（事件订阅）：
    请求地址 → http(s)://你的公网IP:8088/feishu/event
    添加事件 → im.message.receive_v1（接收消息）

包结构：
    messaging.py  发送原语（reply / 图表卡片 / 主动发送）
    gpu_flow.py   GPU 申请流程（卡片 / 状态 / 解析 / 提交）
    actions.py    卡片按钮动作注册表 + _handle_card_trigger_sync
    messages.py   消息事件处理（去重 / 意图 / 绑定 / Agent 调用）
    routes.py     Flask app + 三个路由 + run() 入口

本 __init__ 仅做向后兼容再导出；包内跨模块调用一律
`from . import <mod>` + `<mod>.func(...)` 模块属性方式（patch 单点生效）。
"""
import sys
import io  # noqa: F401  (保留供其他模块可能用到)

# reconfigure 就地修改编码，不替换文件对象（兼容 Flask/click 的 fileno 检查）
# 必须保持在所有子模块导入之前（Windows 下中文日志依赖）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from . import actions, gpu_flow, messages, messaging, routes  # noqa: E402
from .routes import app, run  # noqa: F401,E402
from .actions import _process_action, _handle_card_trigger_sync  # noqa: F401,E402
from .messages import (_is_duplicate_event, _is_gpu_intent, _is_registered,  # noqa: F401,E402
                       _handle_ram_bind, _send_bind_status, _auto_map_user,
                       _query_my_instances, _process_message)
from .gpu_flow import (_GPU_REQUEST_CARD, _GPU_GUIDE_TEXT,  # noqa: F401,E402
                       _parse_gpu_request, _handle_gpu_request, _send_gpu_card,
                       _send_ak_register_card, _set_gpu_state, _get_gpu_state,
                       _clear_gpu_state)
from .messaging import (_feishu_reply, _feishu_reply_with_chart,  # noqa: F401,E402
                        _feishu_send, _send_text_to)
