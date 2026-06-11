"""即将迁移到 cards.py 原语的卡片 builder：迁移前锁住完整 dict 形状（characterization）。

期望值按当前源码字面量手抄；迁移后必须逐字相等 —— 形状漂移 = 回归。
oss_perm / cluster_mfu / capacity 配色已有专测兜着，此处补 dsw_scheduler 4 卡、
capacity 合并卡、notify Grafana 按钮、_GPU_REQUEST_CARD。
"""
import pytest


def _f(label, value):
    return {"is_short": True, "text": {"tag": "lark_md", "content": f"**{label}**\n{value}"}}


def _btn(text, value, type="default"):
    return {"tag": "button", "text": {"tag": "plain_text", "content": text},
            "type": type, "value": value}


# ── dsw_scheduler 4 卡 ───────────────────────────────────────────────────────

def test_approval_card_shape():
    from core.dsw_scheduler import _make_approval_card, _cost_str
    c = _make_approval_card("GPU-7", "wuji-x", "8", "48", "预训练", "张三", "ou_u", "oc_c")
    assert c == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "⏳ GPU 大规模申请待审批"},
                   "template": "orange"},
        "elements": [
            {"tag": "div", "fields": [
                _f("申请人", "张三"), _f("工单", "GPU-7"), _f("实例", "wuji-x"),
                _f("规格", "8GPU · 48h"), _f("费用预估", _cost_str(8, 48.0)), _f("用途", "预训练"),
            ]},
            {"tag": "hr"},
            {"tag": "action", "actions": [
                _btn("✅ 批准", {"action": "approve_gpu", "ticket_key": "GPU-7",
                               "requester_open_id": "ou_u", "requester_chat_id": "oc_c"}, "primary"),
                _btn("❌ 拒绝", {"action": "reject_gpu", "ticket_key": "GPU-7",
                               "requester_open_id": "ou_u", "requester_chat_id": "oc_c"}, "danger"),
            ]},
        ],
    }


def test_running_card_shape():
    from core.dsw_scheduler import _make_running_card, _cost_str
    c = _make_running_card("dsw-1", "wuji-x", "GPU-7", 4, 24.0)
    assert c == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "🟢 GPU 实例已就绪"},
                   "template": "green"},
        "elements": [
            {"tag": "div", "fields": [
                _f("实例名称", "wuji-x"), _f("实例 ID", "`dsw-1`"),
                _f("工单", "GPU-7"), _f("费用预估", _cost_str(4, 24.0)),
            ]},
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "实例已进入 **Running** 状态，可在 PAI DSW 控制台打开 JupyterLab。\n"
                           "到期前 15 分钟会提醒续期。"}},
        ],
    }


def test_instance_created_card_shape():
    from core.dsw_scheduler import _make_instance_created_card
    c = _make_instance_created_card("dsw-1", "wuji-x", "GPU-7", "24")
    assert c == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "✅ GPU 实例已创建"},
                   "template": "green"},
        "elements": [
            {"tag": "div", "fields": [
                _f("实例名称", "wuji-x"), _f("实例 ID", "`dsw-1`"),
                _f("工单", "GPU-7"), _f("有效时长", "24 小时"),
            ]},
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "实例正在启动，预计 2-3 分钟后可用。到期前 15 分钟将提醒续期。"}},
        ],
    }


def test_idle_warn_card_shape():
    from core.dsw_scheduler import _make_idle_warn_card
    c = _make_idle_warn_card("dsw-1", "wuji-x", "GPU-7", 30)
    assert c == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "⏰ GPU 实例即将超时"},
                   "template": "orange"},
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "实例 **wuji-x** (`dsw-1`) 使用时长已达上限。\n"
                           "**30 分钟内无操作将自动停止实例**（数据保留，可重新启动）。"}},
            {"tag": "hr"},
            {"tag": "action", "actions": [
                _btn("⏩ 延续使用 2 小时", {"action": "extend_dsw", "instance_id": "dsw-1",
                                          "ticket_key": "GPU-7", "extend_hours": "2"}, "primary"),
                _btn("🛑 立即停止", {"action": "stop_dsw", "instance_id": "dsw-1",
                                   "ticket_key": "GPU-7"}, "danger"),
            ]},
        ],
    }


# ── capacity_monitor 合并卡 ──────────────────────────────────────────────────

def test_combined_capacity_card_shape():
    from core.capacity_monitor import _build_combined_card, _fmt_size
    sec1 = [{"tag": "div", "text": {"tag": "lark_md", "content": "A"}}]
    sec2 = [{"tag": "div", "text": {"tag": "lark_md", "content": "B"}}]
    total = 5 * 1024 ** 4
    c = _build_combined_card([sec1, sec2], any_over=False, grand_total=total)
    assert c == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text",
                             "content": f"📦 容量巡检（OSS + TOS，合计 {_fmt_size(total)}）"},
                   "template": "blue"},
        "elements": [sec1[0], {"tag": "hr"}, sec2[0]],
    }
    assert _build_combined_card([sec1], True, total)["header"]["template"] == "red"


# ── notify Grafana 按钮 ──────────────────────────────────────────────────────

def test_grafana_buttons_shape(monkeypatch):
    from config.settings import settings
    from tools.feishu import notify
    monkeypatch.setattr(settings, "GRAFANA_URL", "https://g.example")
    monkeypatch.setattr(settings, "GRAFANA_DATASOURCE_UID", "ds-1")
    monkeypatch.setattr("tools.aliyun.prometheus.grafana_shortcuts",
                        lambda node_filter="": [("CPU", "up")])
    out = notify._grafana_buttons()
    assert out[0] == {"tag": "hr"}
    acts = out[1]["actions"]
    assert acts[0]["tag"] == "button"
    assert acts[0]["text"] == {"tag": "plain_text", "content": "CPU"}
    assert acts[0]["type"] == "default"
    assert acts[0]["url"].startswith("https://g.example/explore?orgId=1&left=")
    assert list(acts[0].keys()) == ["tag", "text", "type", "url"]
    assert acts[-1] == {"tag": "button", "text": {"tag": "plain_text", "content": "🔍 打开 Explore"},
                        "type": "primary", "url": "https://g.example/explore"}


def test_grafana_buttons_empty_without_url(monkeypatch):
    from config.settings import settings
    from tools.feishu import notify
    monkeypatch.setattr(settings, "GRAFANA_URL", "")
    assert notify._grafana_buttons() == []


# ── feishu_bot GPU 申请卡 ────────────────────────────────────────────────────

def test_gpu_request_card_shape():
    from core.feishu_bot import _GPU_REQUEST_CARD
    assert _GPU_REQUEST_CARD == {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "GPU 资源申请"},
                   "template": "blue"},
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "**第一步：选择常用配置**（点击后卡片更新，再补充实例名和用途即可提交）"}},
            {"tag": "hr"},
            {"tag": "action", "actions": [
                _btn("1 GPU · 8h\n小模型微调",
                     {"action": "quick_gpu", "gpu_count": "1", "duration_hours": "8"}),
                _btn("4 GPU · 24h\n7B 模型训练",
                     {"action": "quick_gpu", "gpu_count": "4", "duration_hours": "24"}, "primary"),
                _btn("8 GPU · 48h\n大模型预训练",
                     {"action": "quick_gpu", "gpu_count": "8", "duration_hours": "48"}, "danger"),
            ]},
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md",
                "content": "或直接回复自定义参数：\n"
                           "```\n实例名: wzh-train-01\nGPU数: 4\n时长: 24\n用途: 大语言模型微调\n```"}},
        ],
    }
