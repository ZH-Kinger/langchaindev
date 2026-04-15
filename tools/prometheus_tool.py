"""
Prometheus 指标查询与分析工具。
数据源：阿里云 PAI Prometheus（Basic Auth，AK/SK 认证）。
支持：CPU / 内存 / 磁盘 I/O / 网络 / GPU 五类指标的趋势分析和阈值判断。
"""
import time
from datetime import datetime

import numpy as np
import requests
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── PromQL 配置字典（阿里云 PAI 节点级指标）────────────────────────────────────
# {node_sel2} 占位符在运行时替换为 label selector 内容，如 `nodeId=~"node01.*"`
# 空字符串时渲染为 `{}` — 合法的"不过滤"写法
METRIC_QUERIES: dict = {
    "cpu": {
        "label":  "CPU 使用率",
        "unit":   "%",
        "warn":   80.0,
        "crit":   90.0,
        "promql": "avg by (nodeId)(AliyunPaiquota_NODE_CPU_UTIL{{{node_sel2}}})",
    },
    "memory": {
        "label":  "内存使用率",
        "unit":   "%",
        "warn":   85.0,
        "crit":   95.0,
        "promql": "avg by (nodeId)(AliyunPaiquota_NODE_MEMORY_UTIL{{{node_sel2}}})",
    },
    "disk": {
        "label":     "磁盘 I/O",
        "unit":      "MB/s",
        "warn":      200.0,
        "crit":      500.0,
        "promql_read": (
            "avg(rate(AliyunPaiquota_NODE_DISK_READ_BYTES_TOTAL{{{node_sel2}}}[5m]))"
            " / 1048576"
        ),
        "promql_write": (
            "avg(rate(AliyunPaiquota_NODE_DISK_WRITE_BYTES_TOTAL{{{node_sel2}}}[5m]))"
            " / 1048576"
        ),
    },
    "network": {
        "label":     "网络流量",
        "unit":      "MB/s",
        "warn":      100.0,
        "crit":      500.0,
        "promql_rx": (
            "avg(rate(AliyunPaiquota_NODE_NETWORK_RECEVICE_BYTES_TOTAL{{{node_sel2}}}[5m]))"
            " / 1048576"
        ),
        "promql_tx": (
            "avg(rate(AliyunPaiquota_NODE_NETWORK_TRANSMIT_BYTES_TOTAL{{{node_sel2}}}[5m]))"
            " / 1048576"
        ),
    },
    "gpu": {
        "label":      "GPU 使用率",
        "unit":       "%",
        "warn":       95.0,
        "crit":       99.0,
        "warn_temp":  80.0,
        "crit_temp":  90.0,
        "promql_util": "avg(AliyunPaiquota_NODE_GPU_SM_UTIL{{{node_sel2}}})",
        "promql_temp": "avg(AliyunPaiquota_NODE_GPU_TEMPERATURE{{{node_sel2}}})",
    },
}

_COLOR_PRIORITY = {"red": 3, "yellow": 2, "green": 1, "gray": 0}


# ── 内部工具函数 ────────────────────────────────────────────────────────────────

def _node_selector(node_filter: str) -> str:
    """将节点 ID 转为 PromQL label selector 内容（不含花括号）。"""
    if not node_filter:
        return ""
    return f'nodeId=~"{node_filter}.*"'


def _auth() -> tuple:
    """返回 Basic Auth 元组（AK ID, AK Secret）。"""
    return (settings.ALIYUN_ACCESS_KEY_ID, settings.ALIYUN_ACCESS_KEY_SECRET)


def _query_range(promql: str, start: float, end: float, step: str = "60s") -> list:
    """
    调用 Prometheus /api/v1/query_range。
    返回 result 列表；任何异常静默处理并返回空列表。
    """
    if not settings.PROMETHEUS_URL:
        return []
    url = f"{settings.PROMETHEUS_URL}/api/v1/query_range"
    try:
        resp = requests.get(
            url,
            params={"query": promql, "start": start, "end": end, "step": step},
            auth=_auth(),
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("data", {}).get("result", [])
    except Exception:
        return []


def _query_instant(promql: str) -> list:
    """调用 Prometheus /api/v1/query（即时查询）。"""
    if not settings.PROMETHEUS_URL:
        return []
    url = f"{settings.PROMETHEUS_URL}/api/v1/query"
    try:
        resp = requests.get(url, params={"query": promql}, auth=_auth(), timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("result", [])
    except Exception:
        return []


def _flatten_values(results: list) -> list:
    """将多条 series 的所有数值点合并为一个 float 列表。"""
    values = []
    for series in results:
        for _, v in series.get("values", []):
            try:
                values.append(float(v))
            except (ValueError, TypeError):
                pass
    return values


def _trend_analysis(values: list) -> str:
    """用线性回归判断趋势方向。"""
    if len(values) < 3:
        return "数据不足"
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    if abs(slope) < 0.05:
        return "平稳 →"
    return "上升 ↑" if slope > 0 else "下降 ↓"


def _threshold_check(avg: float, max_val: float, warn: float, crit: float) -> tuple:
    """返回 (状态文字, 颜色关键词)。"""
    if max_val >= crit or avg >= crit * 0.9:
        return "严重", "red"
    if max_val >= warn or avg >= warn * 0.9:
        return "警告", "yellow"
    return "正常", "green"


def _analyze_single(key: str, node_filter: str, start: float, end: float) -> dict:
    """
    分析单类指标，返回结构化 dict。
    降级：查询失败时 color='gray'，不影响其他指标。
    """
    cfg = METRIC_QUERIES[key]
    node_sel2 = _node_selector(node_filter)

    _GRAY = lambda detail: {
        "label": cfg["label"], "unit": cfg["unit"],
        "avg": 0, "max": 0, "trend": "无数据", "status": "不可达", "color": "gray",
        "detail": detail,
    }

    # ── 磁盘 I/O（读写分别查询）──────────────────────────────────────────────
    if key == "disk":
        read_vals  = _flatten_values(_query_range(cfg["promql_read"].format(node_sel2=node_sel2), start, end))
        write_vals = _flatten_values(_query_range(cfg["promql_write"].format(node_sel2=node_sel2), start, end))
        all_vals   = read_vals + write_vals
        if not all_vals:
            return _GRAY("读: 不可达 | 写: 不可达")
        read_avg  = sum(read_vals)  / len(read_vals)  if read_vals  else 0.0
        write_avg = sum(write_vals) / len(write_vals) if write_vals else 0.0
        avg       = (read_avg + write_avg) / 2
        max_val   = max(all_vals)
        status, color = _threshold_check(avg, max_val, cfg["warn"], cfg["crit"])
        return {
            "label": cfg["label"], "unit": cfg["unit"],
            "avg": round(avg, 2), "max": round(max_val, 2),
            "trend": _trend_analysis(all_vals), "status": status, "color": color,
            "detail": f"读均值: {read_avg:.2f} MB/s | 写均值: {write_avg:.2f} MB/s",
        }

    # ── 网络（收发分别查询）──────────────────────────────────────────────────
    if key == "network":
        rx_vals = _flatten_values(_query_range(cfg["promql_rx"].format(node_sel2=node_sel2), start, end))
        tx_vals = _flatten_values(_query_range(cfg["promql_tx"].format(node_sel2=node_sel2), start, end))
        all_vals = rx_vals + tx_vals
        if not all_vals:
            return _GRAY("接收: 不可达 | 发送: 不可达")
        rx_avg = sum(rx_vals) / len(rx_vals) if rx_vals else 0.0
        tx_avg = sum(tx_vals) / len(tx_vals) if tx_vals else 0.0
        avg    = (rx_avg + tx_avg) / 2
        max_val = max(all_vals)
        status, color = _threshold_check(avg, max_val, cfg["warn"], cfg["crit"])
        return {
            "label": cfg["label"], "unit": cfg["unit"],
            "avg": round(avg, 2), "max": round(max_val, 2),
            "trend": _trend_analysis(all_vals), "status": status, "color": color,
            "detail": f"接收均值: {rx_avg:.2f} MB/s | 发送均值: {tx_avg:.2f} MB/s",
        }

    # ── GPU（SM 利用率 + 温度双指标）────────────────────────────────────────
    if key == "gpu":
        util_vals = _flatten_values(_query_range(cfg["promql_util"].format(node_sel2=node_sel2), start, end))
        temp_vals = _flatten_values(_query_range(cfg["promql_temp"].format(node_sel2=node_sel2), start, end))
        if not util_vals and not temp_vals:
            return _GRAY("GPU 数据不可达")
        util_avg = sum(util_vals) / len(util_vals) if util_vals else 0.0
        util_max = max(util_vals) if util_vals else 0.0
        temp_avg = sum(temp_vals) / len(temp_vals) if temp_vals else 0.0
        temp_max = max(temp_vals) if temp_vals else 0.0
        _, util_color = _threshold_check(util_avg, util_max, cfg["warn"], cfg["crit"])
        _, temp_color = _threshold_check(temp_avg, temp_max, cfg["warn_temp"], cfg["crit_temp"])
        # 取更严重的颜色
        color = util_color if _COLOR_PRIORITY.get(util_color, 0) >= _COLOR_PRIORITY.get(temp_color, 0) else temp_color
        status = "严重" if color == "red" else ("警告" if color == "yellow" else "正常")
        return {
            "label": cfg["label"], "unit": cfg["unit"],
            "avg": round(util_avg, 2), "max": round(util_max, 2),
            "trend": _trend_analysis(util_vals) if util_vals else "无数据",
            "status": status, "color": color,
            "detail": (
                f"SM利用率均值: {util_avg:.1f}% | 峰值: {util_max:.1f}% | "
                f"温度均值: {temp_avg:.1f}°C | 最高温: {temp_max:.1f}°C"
            ),
        }

    # ── CPU / 内存（单 PromQL）──────────────────────────────────────────────
    promql  = cfg["promql"].format(node_sel2=node_sel2)
    results = _query_range(promql, start, end)
    values  = _flatten_values(results)
    if not values:
        return _GRAY("Prometheus 数据不可达")

    avg     = sum(values) / len(values)
    max_val = max(values)
    status, color = _threshold_check(avg, max_val, cfg["warn"], cfg["crit"])

    # 各节点明细（取每条 series 最后一个采样点）
    node_details = []
    for series in results:
        node_id  = series["metric"].get("nodeId", series["metric"].get("instance", "unknown"))
        last_val = float(series["values"][-1][1]) if series.get("values") else 0.0
        node_details.append(f"{node_id}: {last_val:.1f}{cfg['unit']}")

    return {
        "label":  cfg["label"], "unit": cfg["unit"],
        "avg":    round(avg, 2), "max": round(max_val, 2),
        "trend":  _trend_analysis(values), "status": status, "color": color,
        "detail": " | ".join(node_details) if node_details else "无节点数据",
    }


# ── 图表数据（单条聚合序列，供 chart_builder 使用）──────────────────────────────

_CHART_PROMQL = {
    "cpu":     "avg(AliyunPaiquota_NODE_CPU_UTIL{{{sel}}})",
    "memory":  "avg(AliyunPaiquota_NODE_MEMORY_UTIL{{{sel}}})",
    "disk":    (
        "avg(rate(AliyunPaiquota_NODE_DISK_READ_BYTES_TOTAL{{{sel}}}[5m])) / 1048576"
        " + avg(rate(AliyunPaiquota_NODE_DISK_WRITE_BYTES_TOTAL{{{sel}}}[5m])) / 1048576"
    ),
    "network": (
        "avg(rate(AliyunPaiquota_NODE_NETWORK_RECEVICE_BYTES_TOTAL{{{sel}}}[5m])) / 1048576"
        " + avg(rate(AliyunPaiquota_NODE_NETWORK_TRANSMIT_BYTES_TOTAL{{{sel}}}[5m])) / 1048576"
    ),
    "gpu":     "avg(AliyunPaiquota_NODE_GPU_SM_UTIL{{{sel}}})",
}


# Grafana Explore 按钮的标签顺序（PromQL 从 _CHART_PROMQL 取，改一处全局生效）
_GRAFANA_LABELS: dict = {
    "cpu":     "📊 CPU",
    "memory":  "🧠 内存",
    "gpu":     "🎮 GPU",
    "disk":    "💾 磁盘",
    "network": "🌐 网络",
}


def grafana_shortcuts(node_filter: str = "") -> list:
    """
    返回 [(button_label, promql)] 列表，供 feishu_tool 动态生成 Grafana Explore 按钮。
    PromQL 统一来自 _CHART_PROMQL，node_filter 可选用于过滤特定节点。
    """
    sel = _node_selector(node_filter)
    return [
        (label, _CHART_PROMQL[key].format(sel=sel))
        for key, label in _GRAFANA_LABELS.items()
        if key in _CHART_PROMQL
    ]


def fetch_raw_series(node_filter: str = "", duration_minutes: int = 60) -> dict:
    """
    返回各指标的原始时序数据，供图表生成使用。
    格式: {key: {"timestamps": [...], "values": [...], "unit": str}}
    """
    sel   = _node_selector(node_filter)
    end   = time.time()
    start = end - duration_minutes * 60
    # 步长：目标约 60 个数据点
    step  = max(60, (duration_minutes * 60) // 60)

    _UNITS = {"cpu": "%", "memory": "%", "gpu": "%", "disk": "MB/s", "network": "MB/s"}
    result = {}
    for key, tmpl in _CHART_PROMQL.items():
        promql = tmpl.format(sel=sel)
        series = _query_range(promql, start, end, step=f"{step}s")
        if series and series[0].get("values"):
            vals = series[0]["values"]
            result[key] = {
                "timestamps": [v[0] for v in vals],
                "values":     [float(v[1]) for v in vals],
                "unit":       _UNITS[key],
            }
        else:
            result[key] = {"timestamps": [], "values": [], "unit": _UNITS[key]}
    return result


def _query_user_usage(ts: float) -> str:
    """
    查询每个用户当前的资源占用情况（实例级指标）。
    指标前缀 AliyunPaidsw_INSTANCE_* 含 username / instanceName / status 标签。
    """
    def _instant(promql: str) -> list:
        return _query_instant(promql)

    # 各用户维度的即时聚合
    cpu_results  = _instant("avg by (username, instanceName, status)(AliyunPaidsw_INSTANCE_CPU_UTIL)")
    gpu_results  = _instant("avg by (username, instanceName)(AliyunPaidsw_INSTANCE_GPU_SM_UTIL)")
    gmem_used    = _instant("avg by (username, instanceName)(AliyunPaidsw_INSTANCE_GPU_MEMORY_USED)")
    gmem_total   = _instant("avg by (username, instanceName)(AliyunPaidsw_INSTANCE_GPU_MEMORY_TOTAL)")

    if not cpu_results:
        return "⚠️ 当前无运行中的实例，或实例级指标暂不可达。"

    # 按实例整合数据
    instances: dict[str, dict] = {}
    for r in cpu_results:
        key  = r["metric"].get("instanceName", "unknown")
        user = r["metric"].get("username",     "unknown")
        stat = r["metric"].get("status",       "unknown")
        instances.setdefault(key, {"user": user, "status": stat,
                                   "cpu": 0.0, "gpu": 0.0,
                                   "gmem_used": 0.0, "gmem_total": 0.0})
        try:
            instances[key]["cpu"] = float(r["value"][1])
        except Exception:
            pass

    for r in gpu_results:
        key = r["metric"].get("instanceName", "unknown")
        if key in instances:
            try:
                instances[key]["gpu"] = float(r["value"][1])
            except Exception:
                pass

    for r in gmem_used:
        key = r["metric"].get("instanceName", "unknown")
        if key in instances:
            try:
                instances[key]["gmem_used"] = float(r["value"][1]) / 1024
            except Exception:
                pass

    for r in gmem_total:
        key = r["metric"].get("instanceName", "unknown")
        if key in instances:
            try:
                instances[key]["gmem_total"] = float(r["value"][1]) / 1024
            except Exception:
                pass

    # 按用户分组汇总
    user_map: dict[str, list] = {}
    for inst_name, d in instances.items():
        user_map.setdefault(d["user"], []).append({"inst": inst_name, **d})

    lines = [f"👥 用户资源占用（共 {len(instances)} 个实例 · {len(user_map)} 位用户）",
             "━" * 48]

    for user, insts in sorted(user_map.items()):
        avg_cpu = sum(i["cpu"] for i in insts) / len(insts)
        avg_gpu = sum(i["gpu"] for i in insts) / len(insts)
        total_gmem_used  = sum(i["gmem_used"]  for i in insts)
        total_gmem_total = sum(i["gmem_total"] for i in insts)
        gmem_str = (f"{total_gmem_used:.1f}/{total_gmem_total:.0f} GiB"
                    if total_gmem_total > 0 else "N/A")
        lines.append(
            f"👤 {user}  ({len(insts)} 个实例)\n"
            f"   CPU {avg_cpu:.1f}%  GPU {avg_gpu:.1f}%  显存 {gmem_str}"
        )
        for i in sorted(insts, key=lambda x: -x["gpu"]):
            gmem = (f"{i['gmem_used']:.1f}/{i['gmem_total']:.0f}G"
                    if i["gmem_total"] > 0 else "")
            lines.append(
                f"   └─ {i['inst']}  [{i['status']}]"
                f"  CPU {i['cpu']:.1f}%  GPU {i['gpu']:.1f}%  {gmem}"
            )

    return "\n".join(lines)


def _build_report_block(metrics: dict, report_time: str) -> str:
    """
    将各指标分析结果拼成带标记的报告字符串，供 feishu_tool 解析。
    """
    overall_color = max(
        metrics.values(),
        key=lambda m: _COLOR_PRIORITY.get(m["color"], 0),
    )["color"]

    lines = [
        "[REPORT_START]",
        f"OVERALL_COLOR: {overall_color}",
        f"REPORT_TIME: {report_time}",
    ]

    for key in ("cpu", "memory", "disk", "network", "gpu"):
        m = metrics[key]
        lines.append(f"---{key.upper()}---")
        lines.append(
            f"标题: {m['label']} | 状态: {m['status']} | "
            f"均值: {m['avg']}{m['unit']} | 峰值: {m['max']}{m['unit']} | "
            f"趋势: {m['trend']} | 颜色: {m['color']}"
        )
        lines.append(f"明细: {m['detail']}")

    lines.append("[REPORT_END]")
    return "\n".join(lines)


# ── 对外主函数 ──────────────────────────────────────────────────────────────────

def query_prometheus_metrics(
    query_type: str,
    node_filter: str = "",
    duration_minutes: int = 60,
) -> str:
    """Agent 工具执行函数。"""
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，请在 .env 中填写阿里云 Prometheus 地址。"

    end   = time.time()
    start = end - duration_minutes * 60

    # 指标发现模式
    if query_type == "discover":
        url = f"{settings.PROMETHEUS_URL}/api/v1/label/__name__/values"
        try:
            resp = requests.get(url, auth=_auth(), timeout=10)
            resp.raise_for_status()
            names = resp.json().get("data", [])
            return f"发现 {len(names)} 个指标：\n" + "\n".join(names[:100])
        except Exception as e:
            return f"❌ 指标发现失败：{e}"

    # 完整报告模式
    if query_type == "report":
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics = {
            key: _analyze_single(key, node_filter, start, end)
            for key in ("cpu", "memory", "disk", "network", "gpu")
        }
        return _build_report_block(metrics, report_time)

    # 按用户聚合（实例级指标）
    if query_type == "users":
        return _query_user_usage(end)

    # 单类指标模式
    if query_type in METRIC_QUERIES:
        m = _analyze_single(query_type, node_filter, start, end)
        detail_key = "详情" if query_type == "gpu" else "节点明细"
        return (
            f"📊 {m['label']} 分析（过去 {duration_minutes} 分钟）\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"状态：{m['status']} | 均值：{m['avg']}{m['unit']} | "
            f"峰值：{m['max']}{m['unit']} | 趋势：{m['trend']}\n"
            f"{detail_key}：{m['detail']}"
        )

    return (
        f"❓ 未知 query_type：{query_type}，"
        f"可选值：report / cpu / memory / disk / network / gpu / users / discover"
    )


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

class PrometheusQuerySchema(BaseModel):
    query_type: str = Field(
        description=(
            "查询类型：\n"
            "  'report'   - 生成完整五合一报告（CPU+内存+磁盘+网络+GPU，推飞书前必须先调此项）\n"
            "  'cpu'      - 仅分析 CPU 使用率趋势\n"
            "  'memory'   - 仅分析内存使用率趋势\n"
            "  'disk'     - 仅分析磁盘 I/O 吞吐（读+写）\n"
            "  'network'  - 仅分析网络收发流量\n"
            "  'gpu'      - 仅分析 GPU SM 利用率和温度\n"
            "  'users'    - 按用户展示各实例 CPU/GPU/显存占用（实例级明细）\n"
            "  'discover' - 列出 Prometheus 中所有可用指标名称"
        )
    )
    node_filter: str = Field(
        default="",
        description="节点过滤关键词（nodeId 前缀），如 'node-001'。留空则聚合所有节点。"
    )
    duration_minutes: int = Field(
        default=60,
        description="查询过去多少分钟的数据，默认 60 分钟。"
    )


prometheus_tool = StructuredTool.from_function(
    func=query_prometheus_metrics,
    name="query_infrastructure_metrics",
    description=(
        "从阿里云 PAI Prometheus 查询 GPU 集群指标（CPU/内存/磁盘 I/O/网络/GPU），"
        "进行趋势分析和阈值告警判断。"
        "需要推送飞书报告时，先用 query_type='report' 生成报告，"
        "再将输出传给 push_report_to_feishu 工具。"
    ),
    args_schema=PrometheusQuerySchema,
)
