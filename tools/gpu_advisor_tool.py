"""
GPU 集群智能顾问工具。
基于实时 Prometheus 指标对 GPU 集群进行全面诊断，
给出负载均衡、散热管理、任务调度、成本优化等针对性建议。
"""
import re
import time

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── Schema ────────────────────────────────────────────────────────────────────

class GPUAdvisorSchema(BaseModel):
    focus: str = Field(
        default="all",
        description=(
            "建议聚焦方向：\n"
            "  'all'         - 全面诊断（默认）\n"
            "  'utilization' - GPU 利用率与负载均衡\n"
            "  'thermal'     - 散热与温度管理\n"
            "  'scheduling'  - 任务调度与资源分配\n"
            "  'cost'        - 资源成本优化"
        )
    )
    node_filter: str = Field(
        default="",
        description="节点 ID 前缀过滤，留空则分析全部节点。"
    )
    duration_minutes: int = Field(
        default=60,
        description="分析过去多少分钟的数据，默认 60 分钟。"
    )


# ── 建议生成函数 ───────────────────────────────────────────────────────────────

def _advice_utilization(gpu: dict) -> list:
    items = []
    avg   = gpu.get("avg",  0)
    peak  = gpu.get("max",  0)
    trend = gpu.get("trend", "")

    if avg < 15:
        items.append(
            "🔵 **GPU 严重空闲**（SM 利用率均值 {:.1f}%）：大量算力处于浪费状态。\n"
            "建议：① 合并小批量训练任务，增大 batch_size；"
            "② 启用 NVIDIA MIG / MPS 让多任务共享单 GPU；"
            "③ 低负载时段可暂停部分节点以降低成本。".format(avg)
        )
    elif avg < 40:
        items.append(
            "🟡 **GPU 利用率偏低**（均值 {:.1f}%）：资源未被充分利用。\n"
            "建议：① 检查 DataLoader num_workers 是否过小，导致 GPU 等待数据；"
            "② 尝试将多个小模型实验合并到同节点不同 GPU 上并行执行；"
            "③ 使用 torch.compile() 或 XLA 提升单任务 GPU 吞吐。".format(avg)
        )
    elif avg > 92:
        items.append(
            "🔴 **GPU 负载过高**（均值 {:.1f}%，峰值 {:.1f}%）：接近饱和，存在任务排队风险。\n"
            "建议：① 暂停低优先级任务释放算力；"
            "② 对大模型开启梯度累积（gradient_accumulation_steps）；"
            "③ 评估横向扩容（增加节点）时机。".format(avg, peak)
        )
    else:
        items.append(
            "✅ **GPU 利用率健康**（均值 {:.1f}%，峰值 {:.1f}%）：负载处于理想区间（40%–92%），"
            "当前算力分配合理。".format(avg, peak)
        )

    if "上升" in trend and avg > 65:
        items.append(
            "📈 **负载持续上升**：建议提前预留扩容资源，"
            "并在 Prometheus 中设置 SM 利用率 > 95% 的飞书自动告警。"
        )
    if peak - avg > 35:
        items.append(
            "⚡ **负载波动剧烈**（峰均差 {:.1f}%）：可能由 epoch 切换或数据预处理抖动引起。\n"
            "建议：① 启用数据预取（prefetch_factor=2）；"
            "② 检查数据集分片是否均匀（每 shard 大小相近）。".format(peak - avg)
        )
    return items


def _advice_thermal(gpu: dict) -> list:
    items  = []
    detail = gpu.get("detail", "")

    m_ta = re.search(r"温度均值:\s*([\d.]+)", detail)
    m_tm = re.search(r"最高温:\s*([\d.]+)", detail)
    temp_avg = float(m_ta.group(1)) if m_ta else None
    temp_max = float(m_tm.group(1)) if m_tm else None

    if temp_avg is None:
        items.append(
            "⚪ **温度数据不可达**：请确认 GPU 温度指标"
            "（AliyunPaiquota_NODE_GPU_TEMPERATURE）已正确接入 Prometheus。"
        )
        return items

    if temp_max >= 90:
        items.append(
            "☢️ **GPU 温度危险**（最高 {:.1f}°C）：极端高温将触发硬件热保护降频，严重影响训练速度。\n"
            "立即处理：① 确认机房冷通道畅通、冷热隔离完好；"
            "② 立即降低训练任务负载 20%~30%；"
            "③ 用 nvidia-smi 检查风扇转速是否正常。".format(temp_max)
        )
    elif temp_max >= 80:
        items.append(
            "🔥 **GPU 温度偏高**（最高 {:.1f}°C，均值 {:.1f}°C）：逼近降频阈值，需尽快介入。\n"
            "建议：① 检查机柜散热间距（推荐 ≥1U 间隔）；"
            "② 确认 HVAC 制冷功率满足当前满载需求；"
            "③ 设置温度 > 83°C 的飞书自动告警。".format(temp_max, temp_avg)
        )
    elif temp_max >= 70:
        items.append(
            "🌡️ **GPU 温度适中**（最高 {:.1f}°C，均值 {:.1f}°C）：运行正常，"
            "高负载任务上线前建议确认散热余量。".format(temp_max, temp_avg)
        )
    else:
        items.append(
            "❄️ **GPU 散热优良**（最高 {:.1f}°C，均值 {:.1f}°C）：温度处于安全范围，散热系统工作正常。"
            .format(temp_max, temp_avg)
        )
    return items


def _advice_scheduling(gpu: dict, cpu: dict, mem: dict) -> list:
    items   = []
    gpu_avg = gpu.get("avg", 0)
    cpu_avg = cpu.get("avg", 0)
    mem_avg = mem.get("avg", 0)

    # CPU-GPU 数据管线瓶颈
    if gpu_avg > 55 and cpu_avg < 25:
        items.append(
            "🔍 **数据管线瓶颈**：GPU 利用率 {:.1f}% 但 CPU 仅 {:.1f}%，"
            "GPU 很可能在等待 CPU 完成数据预处理。\n"
            "建议：① DataLoader num_workers 设为 CPU 逻辑核数的 50%；"
            "② 使用 pin_memory=True 加速 CPU→GPU 传输；"
            "③ 将数据预处理提前离线完成（预先生成 .pt 缓存文件）。"
            .format(gpu_avg, cpu_avg)
        )

    # 系统内存压力
    if mem_avg > 80:
        items.append(
            "💾 **系统内存吃紧**（使用率 {:.1f}%）：可能影响数据集缓存，甚至引发 OOM-Killer。\n"
            "建议：① 减少同时运行的训练实例数；"
            "② 启用梯度检查点（gradient_checkpointing=True）以时间换内存；"
            "③ 对超大数据集改用内存映射（torch mmap_mode='r'）。".format(mem_avg)
        )

    # 空闲时段调度建议
    if gpu_avg < 30:
        items.append(
            "📅 **空闲算力利用建议**：当前 GPU 利用率仅 {:.1f}%，可充分利用空闲窗口：\n"
            "  · 启动超参数搜索（Optuna / Ray Tune）提升模型性能；\n"
            "  · 执行离线数据增强、特征工程等 CPU 密集型预处理；\n"
            "  · 安排推理评估任务（显存占用比训练低约 40%）。".format(gpu_avg)
        )
    elif gpu_avg > 85:
        items.append(
            "📋 **排队调度建议**：集群负载高（{:.1f}%），新任务应排队等待：\n"
            "  · 为任务设置 Kubernetes PriorityClass，避免抢占生产训练；\n"
            "  · 将非紧急任务延迟到夜间低峰时段（00:00–06:00）提交；\n"
            "  · 考虑启用 GPU Time-Slicing 让小任务见缝插针执行。".format(gpu_avg)
        )
    return items


def _advice_cost(gpu: dict, cpu: dict) -> list:
    items = []
    avg   = gpu.get("avg", 0)
    trend = gpu.get("trend", "")

    if avg < 20:
        waste_pct = 100 - avg
        items.append(
            "💰 **高成本浪费预警**：GPU 平均利用率 {:.1f}%，约 {:.0f}% 的费用产生于空闲期。\n"
            "建议：① 设置空闲自动暂停策略（连续 30 min SM < 10% 则挂起实例）；"
            "② 将可中断训练改用抢占式实例（Spot），价格约为按需价格的 30%–70%；"
            "③ 每周定期审查长期运行但低负载的实例。".format(avg, waste_pct)
        )
    elif avg < 50:
        items.append(
            "💡 **成本优化空间**：GPU 利用率 {:.1f}%，存在一定优化余地。\n"
            "建议评估是否可降低实例规格或将多个小任务整合到更少节点上运行。".format(avg)
        )
    else:
        items.append(
            "✅ **成本效益良好**：GPU 利用率 {:.1f}%，资源使用充分，当前配置性价比合理。".format(avg)
        )

    if "下降" in trend:
        items.append(
            "📉 **负载下降趋势**：若趋势持续，建议逐步缩减节点规模，"
            "避免持续产生空闲资源费用。"
        )
    return items


# ── 主函数 ────────────────────────────────────────────────────────────────────

def advise_gpu_cluster(
    focus: str = "all",
    node_filter: str = "",
    duration_minutes: int = 60,
) -> str:
    """智能分析 GPU 集群状态并输出优化建议报告。"""
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，无法获取集群指标。"

    from tools.prometheus_tool import _analyze_single

    end   = time.time()
    start = end - duration_minutes * 60

    gpu_m = _analyze_single("gpu",    node_filter, start, end)
    cpu_m = _analyze_single("cpu",    node_filter, start, end)
    mem_m = _analyze_single("memory", node_filter, start, end)

    if gpu_m["color"] == "gray" and cpu_m["color"] == "gray":
        return (
            "⚠️ Prometheus 指标不可达，无法生成建议。\n"
            "请检查 PROMETHEUS_URL、ALIYUN_ACCESS_KEY_ID/SECRET 配置是否正确。"
        )

    _VALID_FOCUS = {"all", "utilization", "thermal", "scheduling", "cost"}
    if focus not in _VALID_FOCUS:
        return (
            f"❓ 未知 focus 参数：{focus}\n"
            f"可选值：{' / '.join(sorted(_VALID_FOCUS))}"
        )

    # ── 按 focus 收集建议条目 ──────────────────────────────────────────────────
    items = []
    if focus in ("all", "utilization"):
        items += _advice_utilization(gpu_m)
    if focus in ("all", "thermal"):
        items += _advice_thermal(gpu_m)
    if focus in ("all", "scheduling"):
        items += _advice_scheduling(gpu_m, cpu_m, mem_m)
    if focus in ("all", "cost"):
        items += _advice_cost(gpu_m, cpu_m)

    # ── 拼装报告 ───────────────────────────────────────────────────────────────
    from datetime import datetime
    scope = f"节点 `{node_filter}*`" if node_filter else "全部节点"
    header = [
        f"## GPU 集群优化建议（{datetime.now().strftime('%H:%M')}）",
        f"**分析范围**：{scope}　**周期**：近 {duration_minutes} 分钟",
        "",
        "**当前状态**",
        f"- GPU：均值 {gpu_m['avg']:.1f}%，峰值 {gpu_m['max']:.1f}%，{gpu_m['status']}",
        f"- CPU：均值 {cpu_m['avg']:.1f}%",
        f"- 内存：均值 {mem_m['avg']:.1f}%",
        "",
        "---",
        "",
    ]

    body = []
    for i, item in enumerate(items, 1):
        body.append(f"**建议 {i}**\n{item}\n")

    # ── 优先级摘要 ─────────────────────────────────────────────────────────────
    if gpu_m["color"] == "red":
        cta = "🚨 **优先处理**：GPU 存在严重风险，请立即按上述建议排查。"
    elif gpu_m["color"] == "yellow":
        cta = "⚠️ **近期跟进**：存在潜在风险，建议在本周内完成优化。"
    else:
        cta = "📌 **持续优化**：集群运行稳定，可按优先级逐步落地以上建议。"

    return "\n".join(header + body) + cta


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

gpu_advisor_tool = StructuredTool.from_function(
    func=advise_gpu_cluster,
    name="advise_gpu_cluster",
    description=(
        "分析 GPU 集群的实时指标，给出利用率优化、散热管理、任务调度、成本优化等方面的针对性建议。"
        "当用户询问'如何优化集群'、'GPU 使用建议'、'集群效率'等问题时调用。"
    ),
    args_schema=GPUAdvisorSchema,
)