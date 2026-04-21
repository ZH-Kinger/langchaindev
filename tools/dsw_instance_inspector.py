"""
DSW 实例运行健康巡检工具。

对单个或多个 DSW 实例做即时健康快照，输出：
  - 基本信息：状态、已运行时长、费用估算、申请人
  - 资源快照：GPU 利用率 / 显存 / CPU / 系统内存（当前值）
  - 健康状态：温度、功耗、GPU 健康标志、网络重传
  - 短期趋势：过去 30 分钟 GPU 利用率走势（空转/活跃/间歇/上升）
  - 空转检测：连续低利用率告警（可配置阈值）
  - 操作建议：续期 / 停止 / 释放 / 关注
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── 即时指标 PromQL（instanceName 维度）──────────────────────────────────────────

_INSTANT_PROMQL: dict[str, str] = {
    "gpu_util":      'avg(AliyunPaidsw_INSTANCE_GPU_SM_UTIL{{instanceName="{n}"}})',
    "gpu_mem_used":  'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_USED{{instanceName="{n}"}}) / 1024',
    "gpu_mem_total": 'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_TOTAL{{instanceName="{n}"}}) / 1024',
    "gpu_temp":      'avg(AliyunPaidsw_INSTANCE_GPU_TEMPERATURE{{instanceName="{n}"}})',
    "gpu_power":     'avg(AliyunPaidsw_INSTANCE_GPU_POWER_USAGE{{instanceName="{n}"}})',
    "gpu_health":    'min(AliyunPaidsw_INSTANCE_GPU_HEALTH{{instanceName="{n}"}})',
    "cpu_util":      'avg(AliyunPaidsw_INSTANCE_CPU_UTIL{{instanceName="{n}"}})',
    "sys_mem":       'avg(AliyunPaidsw_INSTANCE_MEMORY_UTIL{{instanceName="{n}"}})',
    "net_retran":    'avg(AliyunPaidsw_INSTANCE_NETWORK_RETRAN_UTIL{{instanceName="{n}"}})',
    "disk_read":     'avg(rate(AliyunPaidsw_INSTANCE_DISK_READ_BYTES_TOTAL{{instanceName="{n}"}}[5m])) / 1048576',
    "cpfs_lat":      'avg(AliyunPaidsw_INSTANCE_CPFS_READ_LATENCY{{instanceName="{n}"}})',
}

# 趋势窗口用 range query（最近 30 分钟，2 分钟步长 = 15 个点）
_TREND_MINUTES = 30
_TREND_STEP    = "120s"


# ── 并行拉取即时值 ────────────────────────────────────────────────────────────────

def _fetch_instant(name: str) -> dict[str, float | None]:
    from tools.prometheus_tool import _query_instant

    def _one(key: str, promql: str) -> tuple[str, float | None]:
        try:
            result = _query_instant(promql.format(n=name))
            if result:
                raw = result[0].get("value", [None, None])[1]
                if raw not in (None, "NaN", "Inf", "+Inf", "-Inf"):
                    return key, float(raw)
        except Exception:
            pass
        return key, None

    metrics: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_one, k, q): k for k, q in _INSTANT_PROMQL.items()}
        for fut in as_completed(futs):
            k, v = fut.result(timeout=15)
            metrics[k] = v
    return metrics


def _fetch_gpu_trend(name: str) -> list[float]:
    """拉取最近 30 分钟的 GPU SM 利用率时序。"""
    from tools.prometheus_tool import _query_range
    end   = time.time()
    start = end - _TREND_MINUTES * 60
    promql = f'avg(AliyunPaidsw_INSTANCE_GPU_SM_UTIL{{instanceName="{name}"}})'
    try:
        series = _query_range(promql, start, end, step=_TREND_STEP)
        return [float(v) for s in series for _, v in s.get("values", [])
                if v not in ("NaN", "Inf", "+Inf", "-Inf")]
    except Exception:
        return []


# ── 趋势分类 ─────────────────────────────────────────────────────────────────────

def _classify_trend(vals: list[float]) -> str:
    if not vals:
        return "无数据"
    avg = float(np.mean(vals))
    p90 = float(np.percentile(vals, 90))
    p10 = float(np.percentile(vals, 10))
    if avg < 5:
        return "持续空转"
    if p10 < 5 and p90 > 30:
        return "间歇活跃（训练循环有阻塞）"
    if avg > 60:
        slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0]) if len(vals) >= 3 else 0
        if slope > 2:
            return "活跃且上升 ↑"
        if slope < -2:
            return "活跃但下降 ↓"
        return "稳定活跃"
    if avg > 20:
        return "中等利用"
    return "低利用"


# ── 健康评分 ─────────────────────────────────────────────────────────────────────

def _health_score(m: dict, trend: str) -> tuple[int, str]:
    """返回 (0-100 分, 等级文字)。分越高越健康。"""
    score = 100
    reasons: list[str] = []

    if m.get("gpu_health") is not None and m["gpu_health"] < 1:
        score -= 40; reasons.append("GPU硬件异常")
    if m.get("gpu_temp") is not None and m["gpu_temp"] > 85:
        score -= 20; reasons.append(f"高温{m['gpu_temp']:.0f}°C")
    if m.get("net_retran") is not None and m["net_retran"] > 5:
        score -= 15; reasons.append(f"网络重传{m['net_retran']:.1f}%")
    if m.get("cpfs_lat") is not None and m["cpfs_lat"] > 10:
        score -= 10; reasons.append(f"CPFS延迟{m['cpfs_lat']:.1f}ms")
    if "空转" in trend:
        score -= 5; reasons.append("GPU空转")

    score = max(0, score)
    if score >= 85:
        level = "✅ 健康"
    elif score >= 60:
        level = "🟡 注意"
    elif score >= 40:
        level = "🟠 警告"
    else:
        level = "🔴 异常"
    return score, level + (f"（{' / '.join(reasons)}）" if reasons else "")


# ── 从 Redis / DSW API 获取实例元数据 ────────────────────────────────────────────

def _get_instance_meta(name: str) -> dict:
    """尝试从 Redis 获取工单元数据；回退到 DSW API 查询当前状态。"""
    meta = {"instance_name": name, "status": "unknown",
            "gpu_count": 1, "elapsed_h": None, "cost": None,
            "owner": "", "purpose": "", "instance_id": ""}

    # ① Redis（Bot 创建的实例有完整工单信息）
    try:
        from core.dsw_scheduler import _all_tracked_keys, _redis_get
        for key in _all_tracked_keys():
            state = _redis_get(key)
            if state and state.get("instance_name") == name:
                now = time.time()
                start_ts = float(state.get("start_ts", now))
                elapsed  = (now - start_ts) / 3600
                gpu_cnt  = int(state.get("gpu_count", 1))
                meta.update({
                    "status":      "Running",
                    "gpu_count":   gpu_cnt,
                    "elapsed_h":   elapsed,
                    "cost":        elapsed * gpu_cnt * settings.GPU_PRICE_PER_HOUR,
                    "owner":       state.get("open_id", ""),
                    "purpose":     state.get("purpose", ""),
                    "instance_id": state.get("instance_id", ""),
                    "ticket_key":  key,
                })
                return meta
    except Exception:
        pass

    # ② DSW API（兜底，获取真实状态）
    try:
        from tools.pai_dsw_tool import _list_instances
        instances = _list_instances()
        for inst in instances:
            if inst.get("name") == name:
                meta["status"]      = inst.get("status", "unknown")
                meta["gpu_count"]   = inst.get("gpu_count", 1)
                meta["instance_id"] = inst.get("instance_id", "")
                meta["owner"]       = inst.get("user_name", "")
                break
    except Exception:
        pass

    return meta


# ── 操作建议 ─────────────────────────────────────────────────────────────────────

def _build_ops_advice(meta: dict, m: dict, trend: str, score: int) -> list[str]:
    advice: list[str] = []
    gpu_util  = m.get("gpu_util") or 0
    gpu_temp  = m.get("gpu_temp")
    net_ret   = m.get("net_retran") or 0
    elapsed   = meta.get("elapsed_h")
    gpu_count = meta.get("gpu_count", 1)

    if m.get("gpu_health") is not None and m["gpu_health"] < 1:
        advice.append("🚨 **立即检查**：GPU 健康指标异常，运行 `nvidia-smi` 确认是否有 ECC 错误，必要时申请换卡。")

    if "持续空转" in trend and elapsed is not None and elapsed > 1:
        advice.append(
            f"💸 **释放建议**：实例已运行 {elapsed:.1f}h，GPU 持续空转，"
            f"已产生费用约 ¥{meta.get('cost', 0):.0f}。"
            "确认任务是否已完成，完成则立即停止以节省费用。"
        )

    if gpu_temp is not None and gpu_temp > 85:
        advice.append(f"🌡️ **降温处理**：当前温度 {gpu_temp:.0f}°C，建议临时降低 batch_size 或暂停部分任务，联系运维检查散热。")

    if net_ret > 10:
        advice.append(f"🌐 **网络告警**：重传率 {net_ret:.1f}%（正常 <1%），多节点 DDP 梯度同步受损，联系运维检查网络链路或切换 InfiniBand。")
    elif net_ret > 1:
        advice.append(f"🌐 **网络注意**：重传率 {net_ret:.1f}%，建议观察，若持续升高请联系运维。")

    if gpu_util > 80:
        mem_pct = None
        if m.get("gpu_mem_used") and m.get("gpu_mem_total") and m["gpu_mem_total"] > 0:
            mem_pct = m["gpu_mem_used"] / m["gpu_mem_total"] * 100
        if mem_pct and mem_pct > 90:
            advice.append(f"⚠️ **显存告急**：当前显存占用 {mem_pct:.0f}%，OOM 风险高。考虑梯度检查点或混合精度。")
        else:
            advice.append("✅ GPU 利用率良好，可考虑增大 batch_size 进一步提升吞吐量。")

    if not advice:
        if score >= 85:
            advice.append("✅ 实例运行状态正常，无需操作。")
        else:
            advice.append("🔍 指标轻微异常，建议持续观察。")

    return advice


# ── 主函数 ────────────────────────────────────────────────────────────────────────

def inspect_dsw_instance(
    instance_name: str = "",
    open_id: str = "",
) -> str:
    """对 DSW 实例做健康巡检，输出运行状态快照和操作建议。"""
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，无法读取指标。"

    # 确定巡检目标
    targets: list[str] = []
    if instance_name:
        targets = [instance_name]
    elif open_id:
        try:
            from core.dsw_scheduler import _all_tracked_keys, _redis_get
            for key in _all_tracked_keys():
                state = _redis_get(key)
                if state and state.get("open_id") == open_id:
                    n = state.get("instance_name", "")
                    if n:
                        targets.append(n)
        except Exception:
            pass
        if not targets:
            return "未找到该用户通过 Bot 创建的运行中实例。可指定实例名：`inspect_instance(instance_name='xxx')`"
    else:
        return "❌ 请提供 instance_name 或 open_id。"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines   = [
        "## DSW 实例健康巡检",
        f"生成时间：{now_str}　实例数：{len(targets)}",
        "",
    ]

    for name in targets:
        # 并行拉取：即时指标 + 趋势
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_instant = pool.submit(_fetch_instant, name)
            f_trend   = pool.submit(_fetch_gpu_trend, name)
            m     = f_instant.result(timeout=30)
            trend_vals = f_trend.result(timeout=30)

        meta  = _get_instance_meta(name)
        trend = _classify_trend(trend_vals)
        score, level = _health_score(m, trend)
        advice = _build_ops_advice(meta, m, trend, score)

        def _v(val, unit="", fmt=".1f"):
            return f"{val:{fmt}}{unit}" if val is not None else "—"

        gpu_util = m.get("gpu_util")
        gpu_mu   = m.get("gpu_mem_used")
        gpu_mt   = m.get("gpu_mem_total")
        mem_pct  = (gpu_mu / gpu_mt * 100) if gpu_mu and gpu_mt and gpu_mt > 0 else None

        elapsed_str = ""
        if meta.get("elapsed_h") is not None:
            h = meta["elapsed_h"]
            elapsed_str = f"{int(h)}h {int((h % 1)*60)}m"

        lines.append(f"### 实例：`{name}`")
        lines.append("")

        # ── 基本信息 ─────────────────────────────────────────���───────────────
        lines.append("**基本信息**")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 运行状态 | {meta.get('status', '—')} |")
        lines.append(f"| GPU 卡数 | {meta.get('gpu_count', '—')} 卡 |")
        if elapsed_str:
            lines.append(f"| 已运行时长 | {elapsed_str} |")
        if meta.get("cost") is not None:
            lines.append(f"| 当前费用估算 | ≈ ¥{meta['cost']:.0f} |")
        if meta.get("purpose"):
            lines.append(f"| 申请用途 | {meta['purpose']} |")
        if meta.get("ticket_key"):
            lines.append(f"| 关联工单 | {meta['ticket_key']} |")
        lines.append("")

        # ── 健康总评 ─────────────────────────────────────────────────────────
        lines.append(f"**健康评分：{score}/100　{level}**")
        lines.append("")

        # ── 资源快照 ─────────────────────────────────────────────────────────
        lines.append("**资源快照（当前值）**")
        lines.append("")
        lines.append("| 指标 | 当前值 | 状态 |")
        lines.append("|------|--------|------|")

        def _gpu_status(v):
            if v is None: return "—"
            if v < 5:  return "🔴 空转"
            if v < 30: return "🟡 低"
            if v < 70: return "🟢 中等"
            return "✅ 高"

        lines.append(f"| GPU SM 利用率 | **{_v(gpu_util, '%')}** | {_gpu_status(gpu_util)} |")
        lines.append(f"| GPU 显存占用 | {_v(gpu_mu, ' GB')} / {_v(gpu_mt, ' GB')} | {_v(mem_pct, '%', '.0f')} |")
        lines.append(f"| GPU 温度 | {_v(m.get('gpu_temp'), '°C')} | {'🔴 过热' if m.get('gpu_temp') and m['gpu_temp'] > 85 else '🟢 正常' if m.get('gpu_temp') else '—'} |")
        lines.append(f"| GPU 功耗 | {_v(m.get('gpu_power'), ' W')} | — |")
        lines.append(f"| GPU 硬件健康 | {'✅ 正常' if m.get('gpu_health') == 1 else ('🔴 异常' if m.get('gpu_health') == 0 else '—')} | — |")
        lines.append(f"| CPU 利用率 | {_v(m.get('cpu_util'), '%')} | — |")
        lines.append(f"| 系统内存 | {_v(m.get('sys_mem'), '%')} | — |")
        lines.append(f"| 网络重传率 | {_v(m.get('net_retran'), '%')} | {'🔴 异常' if (m.get('net_retran') or 0) > 5 else ('🟡 偏高' if (m.get('net_retran') or 0) > 1 else '🟢 正常') if m.get('net_retran') is not None else '—'} |")
        lines.append(f"| 磁盘读取 | {_v(m.get('disk_read'), ' MB/s')} | — |")
        lines.append(f"| CPFS 读延迟 | {_v(m.get('cpfs_lat'), ' ms')} | {'🟠 慢' if m.get('cpfs_lat') and m['cpfs_lat'] > 5 else '🟢 正常' if m.get('cpfs_lat') is not None else '—'} |")
        lines.append("")

        # ── 30 分钟趋势 ──────────────────────────────────────────────────────
        if trend_vals:
            avg30 = float(np.mean(trend_vals))
            max30 = float(np.max(trend_vals))
            lines.append(f"**近 {_TREND_MINUTES} 分钟 GPU 趋势**：{trend}　"
                         f"均值 {avg30:.1f}%　峰值 {max30:.1f}%")
        else:
            lines.append(f"**近 {_TREND_MINUTES} 分钟 GPU 趋势**：暂无数据")
        lines.append("")

        # ── 操作建议 ─────────────────────────────────────────────────────────
        lines.append("**操作建议**")
        lines.append("")
        for a in advice:
            lines.append(f"- {a}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── LangChain 工具封装 ────────────────────────────────────────────────────────────

class DSWInspectorSchema(BaseModel):
    instance_name: str = Field(
        default="",
        description="指定单个 DSW 实例名（如 'wuji-train-01'）。",
    )
    open_id: str = Field(
        default="",
        description="飞书用户 open_id，用于查找该用户通过 Bot 创建的所有实例。",
    )


dsw_instance_inspector_tool = StructuredTool.from_function(
    func=inspect_dsw_instance,
    name="inspect_dsw_instance",
    description=(
        "对 DSW 实例做运行健康巡检：查看当前 GPU/CPU/显存/温度/网络的即时快照，"
        "评估实例健康评分，检测 GPU 空转，估算当前费用，给出续期/停止/关注等操作建议。"
        "与 analyze_gpu_training 的区别：本工具是运维视角的实例状态总览，后者是算法优化建议。"
        "当用户问'实例状态怎么样'、'健康吗'、'在跑吗'、'费用多少'、'要不要停'时调用。"
    ),
    args_schema=DSWInspectorSchema,
)
