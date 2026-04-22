"""
GPU 集群全局监控看板工具。

并行巡检所有 Running GPU 实例，输出：
  - 全局概览：实例总数、空转数、总费用、网络告警数
  - 实例健康排行表（评分低排前面）
  - 优先操作列表
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings
from tools.dsw_instance_inspector import (
    _fetch_instant,
    _fetch_gpu_trend,
    _classify_trend,
    _health_score,
)


def _probe_instance(name: str, gpu_count: int) -> dict:
    """并行拉取单个实例的即时指标 + 30 分钟趋势，返回汇总 dict。"""
    with ThreadPoolExecutor(max_workers=2) as p:
        fi = p.submit(_fetch_instant, name)
        ft = p.submit(_fetch_gpu_trend, name)
        m          = fi.result(timeout=30)
        trend_vals = ft.result(timeout=30)
    trend        = _classify_trend(trend_vals)
    score, level = _health_score(m, trend)
    return {
        "name":       name,
        "gpu_count":  gpu_count,
        "gpu_util":   m.get("gpu_util"),
        "net_retran": m.get("net_retran") or 0,
        "temp":       m.get("gpu_temp"),
        "score":      score,
        "level":      level,
        "trend":      trend,
        "avg30":      float(np.mean(trend_vals)) if trend_vals else None,
        "m":          m,
    }


def cluster_health_report(top_n: int = 20) -> str:
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，无法读取指标。"

    try:
        from tools.pai_dsw_tool import list_dsw_resources
        raw = list_dsw_resources()
        all_instances = raw.get("instances", [])
    except Exception as e:
        return f"❌ 获取实例列表失败：{e}"

    running = [i for i in all_instances
               if i.get("status") == "Running" and i.get("gpu_count", 0) > 0]
    if not running:
        return "当前没有运行中的 GPU 实例。"

    targets = running[:top_n]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(_probe_instance, inst["name"], inst.get("gpu_count", 1)): inst
            for inst in targets
        }
        for fut in as_completed(futs):
            try:
                results.append(fut.result(timeout=35))
            except Exception:
                inst = futs[fut]
                results.append({
                    "name":       inst["name"],
                    "gpu_count":  inst.get("gpu_count", 1),
                    "gpu_util":   None,
                    "net_retran": 0,
                    "temp":       None,
                    "score":      50,
                    "level":      "⚪ 数据不可达",
                    "trend":      "无数据",
                    "avg30":      None,
                    "m":          {},
                })

    results.sort(key=lambda x: x["score"])

    idle_count    = sum(1 for r in results if r["avg30"] is not None and r["avg30"] < 5)
    net_warn      = sum(1 for r in results if r["net_retran"] > 5)
    hot_count     = sum(1 for r in results if r["temp"] and r["temp"] > 85)
    idle_gpu_h    = sum(r["gpu_count"] for r in results
                        if r["avg30"] is not None and r["avg30"] < 5)
    idle_cost_est = idle_gpu_h * settings.GPU_PRICE_PER_HOUR

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "## GPU 集群监控看板",
        f"生成时间：{now_str}　Running 实例：{len(results)} 个",
        "",
        "### 全局概览",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| Running GPU 实例 | {len(results)} 个 |",
        f"| GPU 空转实例 | {'🔴 ' if idle_count else ''}{idle_count} 个 |",
        f"| 空转费用浪费 | ≈ ¥{idle_cost_est:.0f} / 小时 |",
        f"| 网络重传异常 | {'🔴 ' if net_warn else ''}{net_warn} 个实例 (>5%) |",
        f"| 高温告警 | {'🔴 ' if hot_count else ''}{hot_count} 个实例 (>85°C) |",
        "",
        "### 实例健康排行（评分低 = 需优先关注）",
        "",
        "| # | 实例名 | GPU卡 | GPU利用率 | 30m均值 | 趋势 | 网络重传 | 温度 | 健康评分 |",
        "|---|--------|-------|-----------|---------|------|----------|------|----------|",
    ]

    def _fv(v, unit="", fmt=".1f"):
        return f"{v:{fmt}}{unit}" if v is not None else "—"

    for i, r in enumerate(results, 1):
        gpu_str  = _fv(r["gpu_util"], "%")
        avg_str  = _fv(r["avg30"], "%")
        ret_str  = f"{'🔴' if r['net_retran']>5 else '🟡' if r['net_retran']>1 else '🟢'} {_fv(r['net_retran'], '%')}"
        temp_str = f"{'🔴' if r['temp'] and r['temp']>85 else ''}{_fv(r['temp'], '°C')}"
        lines.append(
            f"| {i} | `{r['name']}` | {r['gpu_count']} | {gpu_str} | {avg_str} "
            f"| {r['trend']} | {ret_str} | {temp_str} | {r['score']}/100 {r['level'].split()[0]} |"
        )

    lines.append("")

    actions: list[str] = []

    idle_names = [r["name"] for r in results if r["avg30"] is not None and r["avg30"] < 5]
    if idle_names:
        actions.append(
            f"💸 **立即核查空转实例**（共 {len(idle_names)} 个，每小时浪费约 ¥{idle_cost_est:.0f}）："
            f" `{'`、`'.join(idle_names[:5])}`"
            + ("等" if len(idle_names) > 5 else "")
        )

    net_names = [r["name"] for r in results if r["net_retran"] > 5]
    if net_names:
        actions.append(
            f"🌐 **网络异常需运维介入**（重传率 >5%，影响 DDP 梯度同步）："
            f" `{'`、`'.join(net_names)}`"
        )

    hot_names = [r["name"] for r in results if r["temp"] and r["temp"] > 85]
    if hot_names:
        actions.append(
            f"🌡️ **高温风险**（>85°C，可能触发降频）：`{'`、`'.join(hot_names)}`"
        )

    low_score = [r for r in results if r["score"] < 60]
    if low_score and not (idle_names or net_names or hot_names):
        actions.append(
            f"⚠️ **{len(low_score)} 个实例健康评分低于 60**，建议逐一用 inspect_dsw_instance 深查。"
        )

    if not actions:
        actions.append("✅ 所有实例运行状态良好，无需立即操作。")

    lines.append("### 优先操作")
    lines.append("")
    for a in actions:
        lines.append(f"- {a}")

    return "\n".join(lines)


class ClusterReportSchema(BaseModel):
    top_n: int = Field(
        default=20,
        description="最多巡检多少个实例，默认 20。",
    )


cluster_health_report_tool = StructuredTool.from_function(
    func=cluster_health_report,
    name="cluster_health_report",
    description=(
        "GPU 集群全局监控看板：并行巡检所有 Running GPU 实例，输出健康排行表、"
        "空转费用浪费估算、网络异常汇总、优先操作列表。"
        "当用户问'集群状态'、'所有实例'、'全局监控'、'哪些在空转'、'总费用'时调用。"
    ),
    args_schema=ClusterReportSchema,
)