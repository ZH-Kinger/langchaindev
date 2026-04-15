"""
飞书企业自建应用消息推送工具。
流程：用 app_id + app_secret 换取 app_access_token → 调 IM API 发送富文本卡片。
"""
import re
import requests
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── 颜色 / 状态映射 ─────────────────────────────────────────────────────────────

_COLOR_HEADER = {
    "red":    "red",
    "yellow": "yellow",
    "green":  "green",
    "gray":   "grey",
}

_STATUS_LABEL = {
    "red":    "🔴 严重告警",
    "yellow": "⚠️  存在警告",
    "green":  "✅ 运行正常",
    "gray":   "⚪ 数据不可达",
}

_TREND_ICON = {
    "上升 ↑": "📈",
    "下降 ↓": "📉",
    "平稳 →": "➡️",
}


# ── 飞书 API 调用 ───────────────────────────────────────────────────────────────

def _get_access_token() -> str:
    """
    用 app_id + app_secret 换取 app_access_token。
    文档：https://open.feishu.cn/document/server-docs/authentication-management/access-token/app_access_token_internal
    """
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={"app_id": settings.FEISHU_APP_ID, "app_secret": settings.FEISHU_APP_SECRET},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取飞书 Token 失败：{data.get('msg')}")
    return data["app_access_token"]


def _upload_image(token: str, png_bytes: bytes) -> str:
    """
    将 PNG bytes 上传到飞书图片 API，返回 image_key。
    文档：https://open.feishu.cn/document/server-docs/im-v1/image/create
    """
    resp = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/images",
        headers={"Authorization": f"Bearer {token}"},
        data={"image_type": "message"},
        files={"image": ("chart.png", png_bytes, "image/png")},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"图片上传失败: {data.get('msg')}")
    return data["data"]["image_key"]


def _send_card(token: str, card: dict) -> dict:
    """
    调用飞书 IM API 发送消息卡片到指定群。
    文档：https://open.feishu.cn/document/server-docs/im-v1/message/create
    """
    import json
    resp = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        params={"receive_id_type": "chat_id"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "receive_id": settings.FEISHU_CHAT_ID,
            "msg_type":   "interactive",
            "content":    json.dumps(card),
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── 报告解析 ────────────────────────────────────────────────────────────────────

def _section_title(key: str) -> str:
    return {
        "CPU":     "🖥️ CPU 使用率",
        "MEMORY":  "🧠 内存使用率",
        "DISK":    "💾 磁盘 I/O",
        "NETWORK": "🌐 网络流量",
        "GPU":     "🎮 GPU 使用率",
    }.get(key.upper(), key)


def _smart_detail(detail: str) -> str:
    """
    对节点明细做智能压缩：
    - 节点列表（>5 个形如 "name: value%" 的条目）：
        * 若全部 ≤0.1%，返回 "全部 N 个节点正常"
        * 否则显示前 3 个活跃节点（缩短名称）+ 总计
    - 摘要型明细（GPU / 磁盘 / 网络）：原样返回
    """
    entries = [e.strip() for e in detail.split(" | ") if e.strip()]
    if len(entries) <= 5:
        return detail  # 摘要型，直接返回

    # 尝试解析为 "nodename: value" 格式
    node_vals = []
    for entry in entries:
        m = re.match(r"^(.+?):\s*([\d.]+)", entry)
        if m:
            node_vals.append((m.group(1).strip(), float(m.group(2))))
    if len(node_vals) != len(entries):
        return detail  # 解析失败，原样返回

    total   = len(node_vals)
    active  = sorted([(n, v) for n, v in node_vals if v > 0.1], key=lambda x: -x[1])
    if not active:
        return f"全部 {total} 个节点正常"

    # 缩短节点名：取 "-" 分割后最后两段，再截断到 12 个字符
    def shorten(name: str) -> str:
        parts = name.split("-")
        short = "-".join(parts[-2:]) if len(parts) >= 2 else name
        return short[-12:] if len(short) > 12 else short

    top3   = active[:3]
    listed = " · ".join(f"{shorten(n)}: {v:.1f}%" for n, v in top3)
    suffix = f"(+{total - len(top3)} 个节点正常)" if total > len(top3) else ""
    return f"活跃({len(active)}/{total})：{listed} {suffix}".strip()


def _parse_report(content: str) -> dict:
    """
    解析 prometheus_tool 输出的 [REPORT_START]...[REPORT_END] 格式。
    每个 section 提取独立字段，供 _build_card 做精细排版。
    解析失败时返回 fallback 结构。
    """
    if "[REPORT_START]" not in content:
        return {
            "overall_color": "green",
            "report_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections":      [{"title": "报告内容", "raw": content}],
            "is_raw":        True,
        }

    overall_color = re.search(r"OVERALL_COLOR:\s*(\w+)", content)
    report_time   = re.search(r"REPORT_TIME:\s*(.+)",    content)

    sections = []
    for match in re.finditer(r"---(\w+)---\n(.+?)\n明细:\s*(.+?)(?=\n---|$)", content, re.DOTALL):
        key, stat_line, raw_detail = match.group(1), match.group(2).strip(), match.group(3).strip()

        def _field(label: str) -> str:
            m = re.search(rf"{label}:\s*([^|]+)", stat_line)
            return m.group(1).strip() if m else ""

        color  = _field("颜色")  or "green"
        status = _field("状态")
        avg    = _field("均值")
        peak   = _field("峰值")
        trend  = _field("趋势")

        # 趋势加 emoji
        for trend_text, icon in _TREND_ICON.items():
            trend = trend.replace(trend_text, f"{icon} {trend_text}")

        sections.append({
            "title":  _section_title(key),
            "color":  color,
            "status": status,
            "avg":    avg,
            "peak":   peak,
            "trend":  trend,
            "detail": _smart_detail(raw_detail),
        })

    return {
        "overall_color": overall_color.group(1) if overall_color else "green",
        "report_time":   report_time.group(1).strip() if report_time else "",
        "sections":      sections,
        "is_raw":        False,
    }


# ── 卡片构建 ────────────────────────────────────────────────────────────────────

def _grafana_buttons(node_filter: str = "") -> list:
    """
    从 prometheus_tool.grafana_shortcuts() 动态读取 PromQL，生成 Grafana Explore 按钮。
    GRAFANA_URL 未配置时返回空列表。
    新增/修改指标只需改 prometheus_tool._CHART_PROMQL，按钮自动更新。
    """
    import json
    import urllib.parse
    from tools.prometheus_tool import grafana_shortcuts

    if not settings.GRAFANA_URL:
        return []

    base = settings.GRAFANA_URL.rstrip("/")
    ds   = settings.GRAFANA_DATASOURCE_UID or "Prometheus"

    def _explore_url(promql: str, from_time: str = "now-1h") -> str:
        payload = {
            "datasource": ds,
            "queries":    [{"refId": "A", "expr": promql, "range": True}],
            "range":      {"from": from_time, "to": "now"},
        }
        return f"{base}/explore?orgId=1&left={urllib.parse.quote(json.dumps(payload))}"

    # 从 prometheus_tool 取 PromQL，label 也在那里统一维护
    metric_buttons = [
        {
            "tag":  "button",
            "text": {"tag": "plain_text", "content": label},
            "type": "default",
            "url":  _explore_url(promql),
        }
        for label, promql in grafana_shortcuts(node_filter)
    ]

    # 最后追加一个"打开 Explore"入口按钮
    metric_buttons.append({
        "tag":  "button",
        "text": {"tag": "plain_text", "content": "🔍 打开 Explore"},
        "type": "primary",
        "url":  f"{base}/explore",
    })

    return [{"tag": "hr"}, {"tag": "action", "actions": metric_buttons}]


def _build_card(title: str, parsed: dict,
                image_key: str = None, chart_error: str = None) -> dict:
    """构建飞书 Interactive Card JSON。"""
    overall  = parsed["overall_color"]
    elements = []

    # ── 顶部摘要 ──
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": (
                f"**报告时间：** {parsed['report_time']}\n"
                f"**整体状态：** {_STATUS_LABEL.get(overall, overall)}"
            ),
        },
    })

    # ── 趋势图或提示 ──
    elements.append({"tag": "hr"})
    if image_key:
        elements.append({
            "tag":     "img",
            "img_key": image_key,
            "alt":     {"tag": "plain_text", "content": "指标趋势图"},
            "mode":    "fit_horizontal",
        })
    elif chart_error:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": chart_error},
        })

    if parsed.get("is_raw"):
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": parsed["sections"][0]["raw"]},
        })
    else:
        _ICON = {"red": "🔴", "yellow": "⚠️", "green": "✅", "gray": "⚪"}
        for sec in parsed["sections"]:
            elements.append({"tag": "hr"})
            icon = _ICON.get(sec["color"], "")
            body = (
                f"**{sec['title']}**　{icon} {sec['status']}\n"
                f"均值 **{sec['avg']}** · 峰值 **{sec['peak']}** · {sec['trend']}\n"
                f"{sec['detail']}"
            )
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": body},
            })

    # ── Grafana 快捷按钮 ──
    elements.extend(_grafana_buttons())

    # ── 底部注脚 ──
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "note",
        "elements": [{
            "tag":     "plain_text",
            "content": f"由 AIOps Agent 自动生成 · Powered by Qwen-Max · {datetime.now().strftime('%H:%M:%S')}",
        }],
    })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title":    {"tag": "plain_text", "content": title},
            "template": _COLOR_HEADER.get(overall, "blue"),
        },
        "elements": elements,
    }


# ── 对外主函数 ──────────────────────────────────────────────────────────────────

def send_feishu_report(
    report_content: str,
    title: str = "AIOps 基础设施健康报告",
) -> str:
    """Agent 工具执行函数。"""
    # 配置校验
    if not settings.FEISHU_APP_ID or not settings.FEISHU_APP_SECRET:
        return f"⚠️ 飞书 App ID / Secret 未配置，报告内容如下：\n{report_content}"
    if not settings.FEISHU_CHAT_ID:
        return "❌ FEISHU_CHAT_ID 未配置，无法确定推送目标群。"

    try:
        token  = _get_access_token()
        parsed = _parse_report(report_content)

        # ── 生成并上传趋势图 ──
        image_key   = None
        chart_error = None
        try:
            from tools.prometheus_tool import fetch_raw_series
            from utils.chart_builder import build_metrics_chart
            series    = fetch_raw_series()
            png_bytes = build_metrics_chart(series)
            image_key = _upload_image(token, png_bytes)
        except Exception as e:
            err_str = str(e)
            if "99991672" in err_str or "im:resource" in err_str:
                chart_error = (
                    "📷 趋势图未显示 — 飞书应用缺少 **im:resource** 权限\n"
                    "开通路径：飞书开放平台 → 我的应用 → "
                    f"`{settings.FEISHU_APP_ID}` → 权限管理 → 搜索 `im:resource` 并开通"
                )
            else:
                chart_error = f"📷 趋势图生成失败：{err_str[:80]}"

        card   = _build_card(title, parsed, image_key=image_key, chart_error=chart_error)
        result = _send_card(token, card)

        if result.get("code") != 0:
            return f"❌ 飞书 API 返回错误：{result.get('msg')}（code={result.get('code')}）"

        return f"✅ 报告已成功推送到飞书群（消息 ID：{result.get('data', {}).get('message_id', '未知')}）"

    except requests.exceptions.ConnectionError:
        return "❌ 网络连接失败，请检查飞书 API 是否可达。"
    except requests.exceptions.Timeout:
        return "❌ 飞书 API 请求超时。"
    except Exception as e:
        return f"❌ 推送失败：{e}"


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

class FeishuReportSchema(BaseModel):
    report_content: str = Field(
        description=(
            "要推送的报告内容。通常是 query_infrastructure_metrics 工具"
            "（query_type='report'）的输出结果，也可传入任意纯文本。"
        )
    )
    title: str = Field(
        default="AIOps 基础设施健康报告",
        description="飞书卡片标题。"
    )


feishu_tool = StructuredTool.from_function(
    func=send_feishu_report,
    name="push_report_to_feishu",
    description=(
        "将运维报告以富文本卡片形式推送到飞书群。"
        "推送前请先调用 query_infrastructure_metrics(query_type='report') 获取报告内容，"
        "再将其传入本工具的 report_content 参数。"
    ),
    args_schema=FeishuReportSchema,
)
