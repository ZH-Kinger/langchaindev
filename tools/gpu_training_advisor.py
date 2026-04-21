"""
GPU 训练行为深度分析工具。

并行查询 DSW 实例的 24 项 Prometheus 指标，覆盖：
  - GPU 计算层：SM 利用率、Tensor Core 活跃率、SM 占用率、实际 TFLOPS
  - 显存层：    显存占用、显存带宽利用率（DRAM Active）
  - 热管理层：  温度、功耗
  - CPU 层：    CPU 利用率、CFS 配额节流（DataLoader 饥饿）
  - 存储层：    磁盘 I/O 速率、CPFS 读取延迟
  - 互联层：    NVLink 带宽、PCIe 带宽、网络重传率、RDMA 错误率

基于 Roofline 模型分析计算瓶颈（计算受限 vs 显存带宽受限），
识别 Tensor Core 是否空置，输出带代码示例的针对性建议。
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── 24 项 DSW 实例级指标（并行拉取）─────────────────────────────────────────────

_INSTANCE_PROMQL: dict[str, str] = {
    # ── GPU 计算 ────────────────────────────────────────────────────────────────
    "gpu_util":       'avg(AliyunPaidsw_INSTANCE_GPU_SM_UTIL{{instanceName="{n}"}})',
    "tensor_active":  'avg(AliyunPaidsw_INSTANCE_GPU_PIP_TENSOR_ACTIVE{{instanceName="{n}"}})',
    "dram_active":    'avg(AliyunPaidsw_INSTANCE_GPU_DRAM_ACTIVE_UTIL{{instanceName="{n}"}})',
    "sm_occupancy":   'avg(AliyunPaidsw_INSTANCE_GPU_SM_OCCUPANCY{{instanceName="{n}"}})',
    "tensor_tflops":  'avg(AliyunPaidsw_INSTANCE_GPU_TENSORTFLOPS_USED{{instanceName="{n}"}})',
    # ── 显存 ────────────────────────────────────────────────────────────────────
    "gpu_mem_used":   'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_USED{{instanceName="{n}"}}) / 1024',
    "gpu_mem_total":  'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_TOTAL{{instanceName="{n}"}}) / 1024',
    "mem_bandwidth":  'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_BANDWIDTH_USED{{instanceName="{n}"}})',
    # ── 温度 & 功耗 ─────────────────────────────────────────────────────────────
    "gpu_temp":       'avg(AliyunPaidsw_INSTANCE_GPU_TEMPERATURE{{instanceName="{n}"}})',
    "gpu_power":      'avg(AliyunPaidsw_INSTANCE_GPU_POWER_USAGE{{instanceName="{n}"}})',
    "gpu_health":     'min(AliyunPaidsw_INSTANCE_GPU_HEALTH{{instanceName="{n}"}})',
    # ── CPU & 系统内存 ──────────────────────────────────────────────────────────
    "cpu_util":       'avg(AliyunPaidsw_INSTANCE_CPU_UTIL{{instanceName="{n}"}})',
    "cpu_throttled":  'avg(AliyunPaidsw_INSTANCE_CPU_CFS_THROTTLED{{instanceName="{n}"}})',
    "sys_mem_util":   'avg(AliyunPaidsw_INSTANCE_MEMORY_UTIL{{instanceName="{n}"}})',
    # ── 磁盘 I/O ────────────────────────────────────────────────────────────────
    "disk_read_mb":   'avg(rate(AliyunPaidsw_INSTANCE_DISK_READ_BYTES_TOTAL{{instanceName="{n}"}}[5m])) / 1048576',
    "disk_write_mb":  'avg(rate(AliyunPaidsw_INSTANCE_DISK_WRITE_BYTES_TOTAL{{instanceName="{n}"}}[5m])) / 1048576',
    "cpfs_read_lat":  'avg(AliyunPaidsw_INSTANCE_CPFS_READ_LATENCY{{instanceName="{n}"}})',
    # ── 网络（DDP 通信）────────────────────────────────────────────────────────
    "net_rx_mb":      'avg(rate(AliyunPaidsw_INSTANCE_NETWORK_RECEIVE_BYTES_TOTAL{{instanceName="{n}"}}[5m])) / 1048576',
    "net_tx_mb":      'avg(rate(AliyunPaidsw_INSTANCE_NETWORK_TRANSMIT_BYTES_TOTAL{{instanceName="{n}"}}[5m])) / 1048576',
    "net_retran":     'avg(AliyunPaidsw_INSTANCE_NETWORK_RETRAN_UTIL{{instanceName="{n}"}})',
    # ── NVLink（同节点多卡互联）────────────────────────────────────────────────
    "nvlink_rx":      'avg(AliyunPaidsw_INSTANCE_GPU_NVLINK_RECEIVE{{instanceName="{n}"}})',
    "nvlink_tx":      'avg(AliyunPaidsw_INSTANCE_GPU_NVLINK_TRANSMIT{{instanceName="{n}"}})',
    # ── PCIe（CPU ↔ GPU 数据传输）──────────────────────────────────────────────
    "pcie_rx":        'avg(AliyunPaidsw_INSTANCE_GPU_PCIE_RECEIVE{{instanceName="{n}"}})',
    "pcie_tx":        'avg(AliyunPaidsw_INSTANCE_GPU_PCIE_TRANSMIT{{instanceName="{n}"}})',
    # ── RDMA（多节点训练）──────────────────────────────────────────────────────
    "rdma_seq_err":   'avg(rate(AliyunPaidsw_INSTANCE_RDMA_PACKET_SEQUENCE_ERROR{{instanceName="{n}"}}[5m]))',
}


# ── 并行拉取 ─────────────────────────────────────────────────────────────────────

def _fetch_all_metrics(name: str, start: float, end: float) -> dict[str, list]:
    """并行查询所有指标，返回 {key: [float...]}，缺失指标为空列表。"""
    from tools.prometheus_tool import _query_range

    def _fetch_one(key: str, tmpl: str) -> tuple[str, list]:
        promql = tmpl.format(n=name)
        try:
            series = _query_range(promql, start, end, step="120s")
            vals = [float(v) for s in series for _, v in s.get("values", [])
                    if v not in ("NaN", "Inf", "+Inf", "-Inf")]
        except Exception:
            vals = []
        return key, vals

    result: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, k, q): k for k, q in _INSTANCE_PROMQL.items()}
        for fut in as_completed(futures):
            try:
                k, v = fut.result(timeout=30)
                result[k] = v
            except Exception:
                result[futures[fut]] = []
    return result


# ── 统计量 ───────────────────────────────────────────────────────────────────────

def _s(vals: list) -> dict:
    """基础统计，空列表全返回 None。"""
    if not vals:
        return {"avg": None, "max": None, "min": None, "std": None, "p10": None, "p90": None}
    a = np.array(vals, dtype=float)
    return {
        "avg": float(np.mean(a)),
        "max": float(np.max(a)),
        "min": float(np.min(a)),
        "std": float(np.std(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def _build_feature_vec(raw: dict[str, list], gpu_count: int) -> dict:
    """将原始时序数据聚合为特征向量。"""
    st = {k: _s(v) for k, v in raw.items()}

    gpu_mem_used  = st["gpu_mem_used"]["avg"]
    gpu_mem_total = st["gpu_mem_total"]["avg"]
    mem_pct = (gpu_mem_used / gpu_mem_total * 100
               if gpu_mem_used and gpu_mem_total and gpu_mem_total > 0 else None)

    # Tensor Core 参与度：tensor_active vs gpu_util 比值
    ga = st["gpu_util"]["avg"]
    ta = st["tensor_active"]["avg"]
    tensor_ratio = (ta / ga if ga and ga > 5 and ta is not None else None)

    # 显存带宽强度：dram_active vs gpu_util 比值
    da = st["dram_active"]["avg"]
    dram_intensity = (da / ga if ga and ga > 5 and da is not None else None)

    # NVLink 活跃（任意方向 > 0）
    nvlink_active = bool(
        (st["nvlink_rx"]["avg"] or 0) + (st["nvlink_tx"]["avg"] or 0) > 0
    )

    # RDMA 错误率
    rdma_err = st["rdma_seq_err"]["avg"]

    return {
        # GPU 计算
        "gpu_avg":       ga,
        "gpu_max":       st["gpu_util"]["max"],
        "gpu_std":       st["gpu_util"]["std"],
        "gpu_p10":       st["gpu_util"]["p10"],
        "gpu_p90":       st["gpu_util"]["p90"],
        "tensor_avg":    ta,
        "tensor_ratio":  tensor_ratio,     # Tensor Core 参与比
        "dram_avg":      da,
        "dram_intensity": dram_intensity,  # 显存带宽强度比
        "sm_occ_avg":    st["sm_occupancy"]["avg"],
        "tflops_avg":    st["tensor_tflops"]["avg"],
        # 显存
        "mem_pct":       mem_pct,
        "mem_bw_avg":    st["mem_bandwidth"]["avg"],
        # 热管理
        "temp_avg":      st["gpu_temp"]["avg"],
        "temp_max":      st["gpu_temp"]["max"],
        "power_avg":     st["gpu_power"]["avg"],
        "power_max":     st["gpu_power"]["max"],
        "gpu_health":    st["gpu_health"]["min"],
        # CPU & 系统
        "cpu_avg":       st["cpu_util"]["avg"],
        "cpu_throttled": st["cpu_throttled"]["avg"],
        "sys_mem_avg":   st["sys_mem_util"]["avg"],
        # 存储
        "disk_read_mb":  st["disk_read_mb"]["avg"],
        "disk_write_mb": st["disk_write_mb"]["avg"],
        "cpfs_lat_avg":  st["cpfs_read_lat"]["avg"],
        # 网络
        "net_rx_mb":     st["net_rx_mb"]["avg"],
        "net_tx_mb":     st["net_tx_mb"]["avg"],
        "net_retran":    st["net_retran"]["avg"],
        # 互联
        "nvlink_active": nvlink_active,
        "pcie_rx_avg":   st["pcie_rx"]["avg"],
        "pcie_tx_avg":   st["pcie_tx"]["avg"],
        "rdma_err":      rdma_err,
        # 元信息
        "gpu_count":     gpu_count,
        "data_points":   len(raw.get("gpu_util", [])),
    }


# ── Redis 实例查询 ────────────────────────────────────────────────────────────────

def _get_user_instances(open_id: str) -> list[dict]:
    try:
        from core.dsw_scheduler import _all_tracked_keys, _redis_get
        result = []
        for key in _all_tracked_keys():
            state = _redis_get(key)
            if state and state.get("open_id") == open_id:
                state["ticket_key"] = key
                result.append(state)
        return result
    except Exception:
        return []


def _get_purpose_from_jira(ticket_key: str) -> str:
    if not ticket_key:
        return ""
    try:
        from tools.jira_tool import _s as _js, _base, parse_ticket_metadata
        s = _js()
        if not s:
            return ""
        resp = s.get(f"{_base()}/rest/api/2/issue/{ticket_key}",
                     params={"fields": "description"}, timeout=10)
        if resp.status_code == 200:
            desc = resp.json().get("fields", {}).get("description", "")
            return parse_ticket_metadata(desc).get("dsw_purpose", "")
    except Exception:
        pass
    return ""


# ── 模式识别（13 种标签）────────────────────────────────────────────────────────

def _classify(fv: dict) -> list[str]:
    """基于特征向量返回瓶颈模式标签列表（可多选）。"""
    tags: list[str] = []

    gpu_avg      = fv.get("gpu_avg")
    if gpu_avg is None:
        return ["no_data"]

    cpu_avg      = fv.get("cpu_avg") or 0
    mem_pct      = fv.get("mem_pct") or 0
    dram_int     = fv.get("dram_intensity")
    tensor_ratio = fv.get("tensor_ratio")
    sm_occ       = fv.get("sm_occ_avg")
    temp_max     = fv.get("temp_max")
    power_max    = fv.get("power_max")
    gpu_health   = fv.get("gpu_health")
    cpu_thr      = fv.get("cpu_throttled") or 0
    disk_read    = fv.get("disk_read_mb") or 0
    cpfs_lat     = fv.get("cpfs_lat_avg")
    net_retran   = fv.get("net_retran") or 0
    nvlink_active= fv.get("nvlink_active", False)
    rdma_err     = fv.get("rdma_err") or 0
    gpu_count    = fv.get("gpu_count", 1)
    gpu_p10      = fv.get("gpu_p10") or 0
    gpu_p90      = fv.get("gpu_p90") or 0
    gpu_std      = fv.get("gpu_std") or 0

    # ── GPU 硬件故障 ──────────────────────────────────────────────────────────
    if gpu_health is not None and gpu_health < 1:
        tags.append("gpu_unhealthy")

    # ── 利用率级别 ────────────────────────────────────────────────────────────
    if gpu_avg < 15:
        tags.append("severe_idle")
    elif gpu_avg < 35:
        tags.append("low_util")
    elif gpu_avg > 80:
        tags.append("compute_bound")

    # ── Roofline：显存带宽受限（DRAM 高 + SM 利用率中等）────────────────────
    if dram_int is not None and dram_int > 1.3 and gpu_avg < 75:
        tags.append("mem_bandwidth_bound")

    # ── Tensor Core 空置（GPU 忙但没有用 Tensor Core）────────────────────────
    if gpu_avg > 40 and tensor_ratio is not None and tensor_ratio < 0.25:
        tags.append("tensor_core_idle")

    # ── SM 占用率低（kernel 太小，GPU 并行度不足）────────────────────────────
    if sm_occ is not None and sm_occ < 40 and gpu_avg > 20:
        tags.append("low_sm_occupancy")

    # ── 显存压力 ──────────────────────────────────────────────────────────────
    if mem_pct > 90:
        tags.append("mem_critical")
    elif mem_pct > 75:
        tags.append("mem_high")

    # ── IO 瓶颈：GPU 等数据（GPU 中高但 CPU 低 + 无 CPU 节流）──────────────
    if gpu_avg > 40 and cpu_avg < 30 and cpu_thr < 5:
        tags.append("io_bound")

    # ── CPU CFS 节流（DataLoader worker 被 cgroup 限速）─────────────────────
    if cpu_thr > 10:
        tags.append("cpu_cfs_throttled")

    # ── 磁盘 I/O 瓶颈（读取速率高 + GPU 低利用）─────────────────────────────
    if disk_read > 200 and gpu_avg < 40:
        tags.append("disk_io_bound")

    # ── CPFS 延迟高（共享存储访问慢）─────────────────────────────────────────
    if cpfs_lat is not None and cpfs_lat > 5:
        tags.append("cpfs_slow")

    # ── 热管理风险（高温 or 高功耗）─────────────────────────────────────────
    if (temp_max is not None and temp_max > 85) or (power_max is not None and power_max > 145):
        tags.append("thermal_risk")

    # ── 多卡：未用 NVLink（梯度同步走 PCIe，效率低）─────────────────────────
    if gpu_count >= 2 and not nvlink_active and gpu_avg > 20:
        tags.append("nvlink_inactive")

    # ── 多卡整体效率低 ───────────────────────────────────────────────────────
    if gpu_count >= 4 and gpu_avg < 60:
        tags.append("multi_gpu_inefficient")

    # ── 网络重传（DDP 通信损耗）─────────────────────────────────────────────
    if net_retran > 1.0:
        tags.append("net_retransmit")

    # ── RDMA 错误（多节点 InfiniBand 问题）──────────────────────────────────
    if rdma_err is not None and rdma_err > 0.1:
        tags.append("rdma_error")

    # ── 负载间歇（训练循环有阻塞点）────────────────────────────────────────
    if gpu_p10 < 5 and gpu_p90 > 40:
        tags.append("intermittent")

    # ── 负载剧烈抖动 ────────────────────────────────────────────────────────
    if gpu_std > 25 and gpu_avg > 20 and "intermittent" not in tags:
        tags.append("spiky")

    return tags or ["normal"]


# ── 任务类型推断 ──────────────────────────────────────────────────────────────────

_PURPOSE_MAP = {
    "finetune":  ["微调", "finetune", "fine-tune", "sft", "rlhf", "lora", "qlora", "peft", "指令"],
    "pretrain":  ["预训练", "pretrain", "pre-train", "from scratch"],
    "inference": ["推理", "inference", "serving", "部署", "vllm", "tensorrt"],
    "distill":   ["蒸馏", "distill", "知识蒸馏"],
    "multimodal":["多模态", "multimodal", "vlm", "clip", "diffusion", "图像生成"],
}

_TASK_LABEL = {
    "finetune": "微调", "pretrain": "预训练",
    "inference": "推理", "distill": "蒸馏",
    "multimodal": "多模态", "general": "通用训练",
}


def _infer_task(purpose: str) -> str:
    p = purpose.lower()
    for task, kws in _PURPOSE_MAP.items():
        if any(k in p for k in kws):
            return task
    return "general"


# ── 建议生成 ──────────────────────────────────────────────────────────────────────

def _build_advices(tags: list, task: str, fv: dict) -> list[dict]:
    """根据模式标签 + 任务类型生成建议列表（含代码示例）。"""
    advs: list[dict] = []
    gpu_avg   = fv.get("gpu_avg", 0) or 0
    mem_pct   = fv.get("mem_pct") or 0
    gpu_count = fv.get("gpu_count", 1)
    temp_max  = fv.get("temp_max")
    power_max = fv.get("power_max")
    cpfs_lat  = fv.get("cpfs_lat_avg")
    dram_avg  = fv.get("dram_avg")
    sm_occ    = fv.get("sm_occ_avg")
    tensor_r  = fv.get("tensor_ratio")
    disk_read = fv.get("disk_read_mb") or 0
    net_retran= fv.get("net_retran") or 0

    # ── GPU 硬件故障（最高优先级）────────────────────────────────────────────
    if "gpu_unhealthy" in tags:
        advs.append({"pri": "紧急", "title": "GPU 硬件异常，需立即介入",
            "body": (
                "Prometheus `GPU_HEALTH` 指标返回 0，GPU 处于不健康状态。\n"
                "**立即检查**：\n"
                "```bash\nnvidia-smi  # 查看 GPU 状态和错误计数\nnvidia-smi -q | grep -i 'error\\|ecc\\|retired'\n```\n"
                "若出现 ECC 错误或显存页面退休（Retired Pages），请联系运维申请换卡或迁移实例。"
            )
        })

    # ── 热管理风险 ────────────────────────────────────────────────────────────
    if "thermal_risk" in tags:
        temp_str  = f"{temp_max:.0f}°C" if temp_max else "—"
        power_str = f"{power_max:.0f}W" if power_max else "—"
        advs.append({"pri": "高", "title": f"热管理风险：温度 {temp_str} / 功耗 {power_str}",
            "body": (
                "GPU 温度或功耗接近阈值，硬件可能触发降频（Thermal Throttling），导致训练速度下降。\n"
                "**短期**：\n"
                "① 临时降低 batch_size 或暂停部分任务，降低热功耗\n"
                "② 用 nvidia-smi 确认是否已降频：\n"
                "```bash\nnvidia-smi -q -d CLOCK | grep 'SM Clock'\n# 若当前时钟 << 最大时钟，已在降频\n```\n"
                "**根本解决**：联系运维检查机柜散热 / 检查 TDP 配置是否超出机房功率预算。"
            )
        })

    # ── 显存告急 ─────────────────────────────────────────────────────────────
    if "mem_critical" in tags:
        advs.append({"pri": "高", "title": f"显存占用 {mem_pct:.0f}%，OOM 风险极高",
            "body": (
                "**三种立即可用的方案（任选一或组合）：**\n\n"
                "① **混合精度训练**（显存节省 30–50%，A10/A100 推荐 bf16）\n"
                "```python\nfrom torch.cuda.amp import autocast, GradScaler\nscaler = GradScaler()\nwith autocast(dtype=torch.bfloat16):\n    loss = model(inputs)\nscaler.scale(loss).backward()\nscaler.step(optimizer); scaler.update()\n```\n"
                "② **梯度检查点**（以时间换显存，节省 50–70%，训练时长增加 ~30%）\n"
                "```python\nmodel.gradient_checkpointing_enable()  # HuggingFace 一行搞定\n```\n"
                "③ **梯度累积**（缩小 batch_size，等效 batch 不变）\n"
                "```python\naccum_steps = 4  # 等效 batch = micro_batch × 4\nfor i, batch in enumerate(loader):\n    with autocast():\n        loss = model(batch) / accum_steps\n    scaler.scale(loss).backward()\n    if (i + 1) % accum_steps == 0:\n        scaler.step(optimizer); scaler.update()\n        optimizer.zero_grad()\n```"
            )
        })

    elif "mem_high" in tags and task == "finetune":
        advs.append({"pri": "中", "title": f"显存 {mem_pct:.0f}%，LoRA 可降低 60–80% 显存占用",
            "body": (
                "全量微调显存消耗高且收益递减，推荐 LoRA 或 QLoRA：\n"
                "```python\nfrom peft import LoraConfig, get_peft_model, TaskType\nconfig = LoraConfig(\n    r=16,                    # 秩，越大参数越多（通常 8–64）\n    lora_alpha=32,           # 缩放因子，通常设为 2×r\n    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],\n    lora_dropout=0.05,\n    task_type=TaskType.CAUSAL_LM,\n)\nmodel = get_peft_model(model, config)\nmodel.print_trainable_parameters()  # 通常只有 0.1%–1%\n```\n"
                "**QLoRA**（4bit 量化 + LoRA，70B 模型可在 2× A10 上微调）：\n"
                "```python\nfrom transformers import BitsAndBytesConfig\nquant_config = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_compute_dtype=torch.bfloat16,\n    bnb_4bit_use_double_quant=True,\n)\nmodel = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config)\n```"
            )
        })

    # ── Tensor Core 空置（最重要的计算效率问题）─────────────────────────────
    if "tensor_core_idle" in tags:
        ratio_str = f"{(tensor_r or 0)*100:.0f}%" if tensor_r else "< 25%"
        advs.append({"pri": "高", "title": f"Tensor Core 严重闲置（参与率 {ratio_str}），算力损失巨大",
            "body": (
                f"GPU SM 利用率 {gpu_avg:.0f}%，但 Tensor Core 参与率仅 {ratio_str}。"
                "A10 的 Tensor Core 峰值算力是 FP32 的 8×，闲置意味着算力严重浪费。\n\n"
                "**根因 1：未使用混合精度**（最常见）\n"
                "```python\n# PyTorch 原生 AMP（推荐 bf16）\nwith torch.autocast('cuda', dtype=torch.bfloat16):\n    output = model(input)\n\n# HuggingFace Trainer\nTrainingArguments(bf16=True)  # 或 fp16=True\n```\n"
                "**根因 2：未使用 Flash Attention**（注意力机制大量使用 Tensor Core）\n"
                "```python\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_id,\n    attn_implementation='flash_attention_2',  # 需安装 flash-attn\n    torch_dtype=torch.bfloat16,\n)\n```\n"
                "**根因 3：矩阵维度不对齐**（维度不是 8 的倍数时 Tensor Core 无法调度）\n"
                "```python\n# hidden_size、vocab_size、batch_size 尽量设为 64 或 128 的倍数\n# 如 hidden_size=4096（✓），hidden_size=4100（✗）\n```"
            )
        })

    # ── 显存带宽受限（Roofline：内存 bound，非计算 bound）───────────────────
    if "mem_bandwidth_bound" in tags:
        dram_str = f"{dram_avg:.0f}%" if dram_avg else "—"
        advs.append({"pri": "高", "title": f"显存带宽受限（DRAM Active {dram_str}），内存访问是瓶颈",
            "body": (
                "GPU 显存总线处于饱和状态，而计算单元相对空闲，说明模型操作是**内存密集型**的。\n\n"
                "**根因 & 解决**：\n"
                "① **算子融合**：将连续的 element-wise 操作合并成一个 kernel，减少显存往返：\n"
                "```python\n# 使用 torch.compile() 自动融合（PyTorch 2.0+）\nmodel = torch.compile(model, mode='reduce-overhead')\n# 或启用 XFormers 的 memory_efficient_attention\n```\n"
                "② **避免频繁的小 tensor 操作**：\n"
                "```python\n# ❌ 多次读写显存\na = x + 1\nb = a * 2\nc = b.relu()\n\n# ✅ 用 fused kernel（如 torch.nn.functional.silu 比手动 x * sigmoid(x) 快 3×）\n```\n"
                "③ **增大 batch_size**：更大 batch 能更好地摊薄显存访问延迟，提升显存带宽利用效率"
            )
        })

    # ── SM 占用率低 ───────────────────────────────────────────────────────────
    if "low_sm_occupancy" in tags:
        occ_str = f"{sm_occ:.0f}%" if sm_occ else "—"
        advs.append({"pri": "中", "title": f"SM 占用率低（{occ_str}），GPU 并行度不足",
            "body": (
                "SM Occupancy 反映 GPU warp 调度器的利用率。占用率低意味着：\n"
                "• kernel 太小（batch_size 太小 / 序列太短）\n"
                "• 每个 thread block 寄存器用量过大导致并发数下降\n\n"
                "**优化方向**：\n"
                "① 增大 batch_size，让 GPU 有足够的 warp 填满 SM\n"
                "② 对于推理场景：启用 continuous batching（vLLM 内置）\n"
                "③ 使用 `torch.profiler` 找到占用率低的具体算子：\n"
                "```python\nwith torch.profiler.profile(\n    activities=[torch.profiler.ProfilerActivity.CUDA],\n    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),\n) as prof:\n    model(input)\nprint(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))\n```"
            )
        })

    # ── IO 瓶颈（GPU 等 DataLoader）─────────────────────────────────────────
    if "io_bound" in tags:
        advs.append({"pri": "高", "title": "DataLoader 是瓶颈，GPU 在等待 CPU 处理数据",
            "body": (
                f"GPU 利用率 {gpu_avg:.0f}%，但 CPU 负载低，数据管线是性能天花板。\n\n"
                "**DataLoader 调优**：\n"
                "```python\nDataLoader(\n    dataset,\n    num_workers=min(os.cpu_count(), 16),  # 不要超过 CPU 核数\n    pin_memory=True,                      # 锁页内存，CPU→GPU 速度提升 2×\n    prefetch_factor=3,                    # 每个 worker 预取 3 个 batch\n    persistent_workers=True,              # 避免每 epoch 重建进程（节省 5–10s/epoch）\n)\n```\n"
                "**离线预处理**（治本）：\n"
                "```python\n# 一次处理，训练时直接读缓存（HuggingFace datasets）\nds = ds.map(\n    tokenize_fn,\n    batched=True, num_proc=8,\n    cache_file_name='tokenized_cache.arrow',\n)\n```\n"
                "**Iterable Dataset + 流式加载**（超大数据集）：\n"
                "```python\nfrom torch.utils.data import IterableDataset\n# 避免一次性加载全部数据到内存\n```"
            )
        })

    # ── CPU CFS 节流 ─────────────────────────────────────────────────────────
    if "cpu_cfs_throttled" in tags:
        advs.append({"pri": "高", "title": "CPU 被 cgroup 配额节流，DataLoader worker 被限速",
            "body": (
                "实例的 CPU 使用量超出了分配的 CFS 配额，导致 DataLoader 进程被系统强制暂停。\n\n"
                "**快速验证**：\n"
                "```bash\n# 查看容器 CPU 节流比例\ncat /sys/fs/cgroup/cpu/cpu.stat | grep throttled\n```\n"
                "**解决方案**：\n"
                "① **申请更多 CPU 核数**（在 GPU 申请卡片中填写 cpu_cores = GPU数 × 16）\n"
                "② **减少 DataLoader num_workers**（降低并发 CPU 使用）\n"
                "③ **将数据预处理移到训练前**，训练时只做简单的 tensor 操作"
            )
        })

    # ── 磁盘 I/O 瓶颈 ────────────────────────────────────────────────────────
    if "disk_io_bound" in tags:
        advs.append({"pri": "高", "title": f"磁盘读取速率高（{disk_read:.0f} MB/s），存储是瓶颈",
            "body": (
                "高磁盘读取速率 + 低 GPU 利用率，说明数据读取是训练瓶颈。\n\n"
                "**按存储位置分层解决**：\n\n"
                "① **本地 SSD**（最快，无需额外操作）：数据已在 SSD，无需优化\n"
                "② **NAS / 对象存储**：\n"
                "```bash\n# 将训练数据复制到本地 SSD（推荐 /mnt/data 目录）\ncp -r /nfs/dataset /mnt/data/\n```\n"
                "③ **内存文件系统**（最快，适合 <100GB 的小数据集）：\n"
                "```bash\n# /dev/shm 是内存，速度接近显存带宽\ncp -r /mnt/data/dataset /dev/shm/\n# 更新代码中的数据路径\n```\n"
                "④ **TFRecord / Arrow 格式**（减少随机读取，顺序读取 2–5× 更快）：\n"
                "```python\n# 转换为 HuggingFace Arrow 格式\nds.save_to_disk('/mnt/data/arrow_cache')\n```"
            )
        })

    # ── CPFS 延迟高 ──────────────────────────────────────────────────────────
    if "cpfs_slow" in tags:
        lat_str = f"{cpfs_lat:.1f}ms" if cpfs_lat else "—"
        advs.append({"pri": "高", "title": f"CPFS 共享存储延迟高（{lat_str}），NAS 访问是瓶颈",
            "body": (
                "云并行文件系统（CPFS）读取延迟超过 5ms，对小文件或随机读取影响明显。\n\n"
                "① **预热缓存**：训练开始前先顺序遍历一次数据集（pagecache 预热）\n"
                "② **用 Arrow / MMap 格式存储**（大文件顺序读取，对 CPFS 友好）：\n"
                "```python\n# HuggingFace datasets 默认使用 Arrow 格式 + mmap\ndataset = load_from_disk('/cpfs/dataset/')  # 自动启用内存映射\n```\n"
                "③ **数据分片预取**：使用多个 DataLoader worker 并发读取不同分片\n"
                "④ **挂载本地 NVMe 缓存**：联系运维挂载高速本地盘作为 CPFS 缓存层"
            )
        })

    # ── 热管理未到风险但需关注 ───────────────────────────────────────────────
    # (already handled above for "thermal_risk", skip)

    # ── 多卡：NVLink 未激活 ──────────────────────────────────────────────────
    if "nvlink_inactive" in tags:
        advs.append({"pri": "中", "title": f"{gpu_count} 卡训练未走 NVLink，梯度同步通过 PCIe（慢 5–10×）",
            "body": (
                "NVLink 带宽可达 600 GB/s，而 PCIe 只有约 32 GB/s。梯度同步走 PCIe 会显著拖慢多卡训练。\n\n"
                "**确认 NVLink 连通性**：\n"
                "```bash\nnvidia-smi topo -m\n# 若卡间显示 NV1/NV2 则支持 NVLink，PIX/PHB 则走 PCIe\n```\n"
                "**强制 DDP 通过 NVLink**：\n"
                "```python\nimport os\nos.environ['NCCL_P2P_DISABLE'] = '0'      # 确保 P2P 传输开启\nos.environ['NCCL_NET_GDR_LEVEL'] = '5'   # 启用 GPUDirect RDMA\n\nimport torch.distributed as dist\ndist.init_process_group(backend='nccl')   # NCCL 自动使用 NVLink\n```\n"
                "**梯度桶优化**（减少通信次数）：\n"
                "```python\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nmodel = DDP(model, bucket_cap_mb=200)  # 合并更多梯度再通信\n```"
            )
        })

    # ── 多卡整体效率低 ───────────────────────────────────────────────────────
    if "multi_gpu_inefficient" in tags:
        advs.append({"pri": "中", "title": f"{gpu_count} 卡利用率 {gpu_avg:.0f}%，多卡并行效率低",
            "body": (
                "**诊断方向**：\n"
                "① **通信占比过高**：增大 batch_size，让计算时间 >> 通信时间\n"
                "② **负载不均衡**：数据集分片不均，部分 GPU 先跑完等其他 GPU\n"
                "③ **考虑 FSDP 替代 DDP**（7B+ 大模型，节省显存且通信更高效）：\n"
                "```python\nfrom torch.distributed.fsdp import FullyShardedDataParallel as FSDP\nfrom torch.distributed.fsdp import ShardingStrategy, MixedPrecision\n\nfsdp_config = dict(\n    sharding_strategy=ShardingStrategy.FULL_SHARD,\n    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),\n    device_id=torch.cuda.current_device(),\n)\nmodel = FSDP(model, **fsdp_config)\n```\n"
                "④ **性能分析**（找出每步通信 vs 计算时间占比）：\n"
                "```python\nimport torch.distributed as dist\n# 在 DDP 训练中开启 NCCL 日志\nos.environ['NCCL_DEBUG'] = 'INFO'\nos.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'\n```"
            )
        })

    # ── 网络重传 ─────────────────────────────────────────────────────────────
    if "net_retransmit" in tags:
        advs.append({"pri": "中", "title": f"网络重传率 {net_retran:.1f}%，DDP 梯度同步通信受损",
            "body": (
                "高重传率说明网络存在丢包或拥塞，多节点 DDP 训练受此影响最大。\n"
                "① 联系运维检查网络链路质量（`sar -n DEV 1` 查看包错误率）\n"
                "② 若有 InfiniBand，改用 RDMA 通信（避开 TCP 重传问题）：\n"
                "```bash\nexport NCCL_IB_DISABLE=0\nexport NCCL_NET=IB\n```\n"
                "③ 临时缓解：增大梯度累积步数，减少通信频率"
            )
        })

    # ── RDMA 错误 ────────────────────────────────────────────────────────────
    if "rdma_error" in tags:
        advs.append({"pri": "高", "title": "RDMA 包序列错误，多节点训练通信不稳��",
            "body": (
                "RDMA（InfiniBand）通信出现包序列错误，可能导致多节点训练挂起或梯度错误。\n"
                "① 检查 IB 链路状态：\n"
                "```bash\nibstat      # 查看 IB 端口状态\nibstatus    # 检查 Active/Initializing\n```\n"
                "② 联系运维检查 IB 交换机和 HCA 驱动版本\n"
                "③ 临时切换到 TCP 通信绕过问题（速度会下降）：\n"
                "```bash\nexport NCCL_IB_DISABLE=1\nexport NCCL_NET=Socket\n```"
            )
        })

    # ── 训练循环阻塞 ─────────────────────────────────────────────────────────
    if "intermittent" in tags:
        advs.append({"pri": "中", "title": "GPU 利用率间歇性下降，训练循环有 CPU 同步阻塞",
            "body": (
                "GPU 利用率在高低之间频繁切换，常见原因是训练循环中有强制 CPU 同步的操作：\n"
                "```python\n# ❌ 常见问题\nloss.item()        # 强制 CPU 同步\nacc = (pred == label).float().mean().item()  # 同上\ntorch.cuda.synchronize()  # 手动同步\n\n# ✅ 只在需要打印时才同步\nif step % 100 == 0:\n    print(f'loss={loss.item():.4f}')\n```\n"
                "**用 PyTorch Profiler 定位具体阻塞位置**：\n"
                "```python\nwith torch.profiler.profile(activities=[\n    torch.profiler.ProfilerActivity.CPU,\n    torch.profiler.ProfilerActivity.CUDA,\n]) as prof:\n    for step, batch in enumerate(train_loader):\n        train_step(batch)\n        if step == 10: break\n# 在 TensorBoard 中查看 CPU/GPU 时间轴，找 gap\nprof.export_chrome_trace('trace.json')\n```"
            )
        })

    # ── 负载抖动 ─────────────────────────────────────────────────────────────
    if "spiky" in tags:
        advs.append({"pri": "低", "title": "GPU 利用率波动大，batch 之间计算量不均",
            "body": (
                "① NLP 任务：按序列长度分桶减少 padding 开销（动态 padding）：\n"
                "```python\nfrom transformers import DataCollatorWithPadding\ncollator = DataCollatorWithPadding(tokenizer, padding='longest')\n# 训练集按长度排序 + shuffle bucket（减少同 batch 长度差异）\n```\n"
                "② 图像任务：检查多尺度增强，固定输入尺寸可提升稳定性\n"
                "③ 验证集评估时 GPU 利用率下降属正常，可用 `model.eval() + torch.no_grad()` 确认"
            )
        })

    # ── 低利用率 ─────────────────────────────────────────────────────────────
    if "severe_idle" in tags or "low_util" in tags:
        lv = "高" if "severe_idle" in tags else "中"
        advs.append({"pri": lv, "title": f"GPU 严重低利用（均值 {gpu_avg:.0f}%），算力大量浪费",
            "body": (
                "**快速诊断（终端执行）**：\n"
                "```bash\nwatch -n 1 nvidia-smi  # 观察 GPU Util 和 Mem Usage 实时变化\n```\n"
                "**常见根因**：\n"
                "① batch_size 太小 → 增大直到显存占用 ~75%\n"
                "② 模型太小、序列太短 → 考虑同卡并行跑多个实验\n"
                "③ 频繁写 checkpoint → 减少 save_steps，或异步写盘\n"
                "④ 数据集在 HDD/NAS → 移到 SSD 或 /dev/shm"
            )
        })

    # ── 任务专项建议 ─────────────────────────────────────────────────────────
    if task == "finetune" and "mem_critical" not in tags and "tensor_core_idle" not in tags:
        advs.append({"pri": "低", "title": "微调专项：学习率与训练稳定性",
            "body": (
                "① Cosine Warmup（防止初始大 LR 破坏预训练权重）：\n"
                "```python\nfrom transformers import get_cosine_schedule_with_warmup\nscheduler = get_cosine_schedule_with_warmup(\n    optimizer,\n    num_warmup_steps=total_steps // 10,\n    num_training_steps=total_steps,\n)\n```\n"
                "② 全量微调推荐 LR `1e-5~5e-5`，LoRA 推荐 `1e-4~3e-4`\n"
                "③ 指令微调只对 response 部分计算 loss，忽略 prompt：\n"
                "```python\nlabels = batch['input_ids'].clone()\nlabels[:, :prompt_length] = -100  # 忽略 prompt 部分\n```"
            )
        })

    elif task == "pretrain":
        advs.append({"pri": "低", "title": "预训练专项：最大化吞吐量",
            "body": (
                "① **Flash Attention 2** + **bf16** 是预训练的标配组合\n"
                "② **torch.compile()** 开启算子融合（首次编译约 1–2 min，后续加速 20–30%）\n"
                "③ 全局 batch size 目标 **1M–4M tokens**，梯度累积按需调整\n"
                "④ 数据加载：使用 **WebDataset** 或 MosaicML 的 **Streaming** 库处理 TB 级数据\n"
                "```python\nfrom streaming import StreamingDataset\ndataset = StreamingDataset(local='/cache', remote='s3://bucket/data')\n```"
            )
        })

    elif task == "inference":
        advs.append({"pri": "低", "title": "推理专项：最大化吞吐与最小化延迟",
            "body": (
                "① **vLLM**（PagedAttention，吞吐提升 3–10×，生产首选）：\n"
                "```python\nfrom vllm import LLM, SamplingParams\nllm = LLM(model='your-model', tensor_parallel_size=4, dtype='bfloat16')\n```\n"
                "② **AWQ/GPTQ 量化**（INT4，速度≈FP16，显存降低 75%）：\n"
                "```python\nfrom awq import AutoAWQForCausalLM\nmodel = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True)\n```\n"
                "③ 推理最优 batch：让显存占用约 70% 时，调整 `max_num_seqs` 参数"
            )
        })

    # ── 高效运行（进阶优化）──────────────────────────────────────────────────
    if "compute_bound" in tags and "tensor_core_idle" not in tags:
        advs.append({"pri": "低", "title": "利用率优秀，可尝试进一步提速",
            "body": (
                f"GPU 均值利用率 {gpu_avg:.0f}%，处于理想区间，进阶优化：\n"
                "① `torch.compile(model, mode='max-autotune')` — 更激进的算子融合\n"
                "② `torch.backends.cudnn.benchmark = True` — 输入尺寸固定时有效\n"
                "③ 检查是否有冗余的 `.cpu()` / `.item()` 在热路径上"
            )
        })

    return advs


# ── 主分析函数 ────────────────────────────────────────────────────────────────────

def analyze_user_gpu_training(
    open_id: str = "",
    instance_name: str = "",
    hours: int = 24,
) -> str:
    """并行拉取 24 项指标，深度分析用户 GPU 训练模式，给出针对性优化建议。"""
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，无法读取 GPU 指标。"

    end   = time.time()
    start = end - hours * 3600

    # ── 确定分析目标 ──────────────────────────────────────────────────────────
    targets: list[dict] = []
    if instance_name:
        targets = [{"instance_name": instance_name, "gpu_count": 1,
                    "start_ts": start, "ticket_key": "", "open_id": open_id}]
    elif open_id:
        targets = _get_user_instances(open_id)
        if not targets:
            return (
                "未找到你通过 Bot 创建的运行中实例记录。\n"
                "可直接指定实例名：**分析实例 `<instance_name>`**"
            )
    else:
        return "❌ 请提供 open_id 或 instance_name 参数。"

    # ── 并行拉取各实例指标 ────────────────────────────────────────────────────
    profiles: list[dict] = []
    for inst in targets:
        name = inst.get("instance_name", "")
        if not name:
            continue
        inst_start = max(start, float(inst.get("start_ts", start)))
        raw        = _fetch_all_metrics(name, inst_start, end)
        gpu_count  = int(inst.get("gpu_count", 1))
        fv         = _build_feature_vec(raw, gpu_count)
        purpose    = inst.get("purpose", "") or _get_purpose_from_jira(inst.get("ticket_key", ""))
        task       = _infer_task(purpose)
        tags       = _classify(fv)
        profiles.append({"name": name, "fv": fv, "tags": tags,
                          "task": task, "purpose": purpose})

    if not profiles:
        return "⚠️ Prometheus 中未查到实例指标（实例可能已停止或指标延迟 2–3 分钟）。"

    # ── 拼装报告 ─────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines   = [
        "## GPU 训练行为深度分析",
        f"分析周期：近 {hours}h　生成时间：{now_str}　实例数：{len(profiles)}",
        "",
    ]

    _TAG_LABEL = {
        "gpu_unhealthy":       "🚨 GPU 硬件异常",
        "thermal_risk":        "🔴 热管理风险",
        "mem_critical":        "🔴 显存告急",
        "tensor_core_idle":    "🔴 Tensor Core 空置",
        "mem_bandwidth_bound": "🟠 显存带宽受限",
        "low_util":            "🟡 利用率偏低",
        "severe_idle":         "🔴 严重空转",
        "io_bound":            "🟠 DataLoader 瓶颈",
        "cpu_cfs_throttled":   "🟠 CPU 配额节流",
        "disk_io_bound":       "🟠 磁盘 I/O 瓶颈",
        "cpfs_slow":           "🟠 CPFS 延迟高",
        "low_sm_occupancy":    "🟡 SM 占用率低",
        "mem_high":            "🟡 显存压力大",
        "nvlink_inactive":     "🟡 NVLink 未激活",
        "multi_gpu_inefficient":"🟡 多卡效率低",
        "net_retransmit":      "🟡 网络重传",
        "rdma_error":          "🔴 RDMA 错误",
        "intermittent":        "🟡 训练阻塞",
        "spiky":               "🟡 负载波动",
        "compute_bound":       "✅ 计算密集（良好）",
        "normal":              "✅ 运行正常",
        "no_data":             "⚪ 指标不可达",
    }

    all_advices: list[dict] = []

    for p in profiles:
        fv      = p["fv"]
        tags    = p["tags"]
        name    = p["name"]
        purpose = p["purpose"]
        task    = p["task"]

        def _fmt(v, unit="", fmt=".1f"):
            return f"{v:{fmt}}{unit}" if v is not None else "—"

        lines.append(f"### 实例：`{name}`")
        lines.append(
            f"GPU 卡数：{fv['gpu_count']} 卡　"
            f"任务类型：{_TASK_LABEL.get(task, '通用训练')}　"
            f"采样点数：{fv['data_points']}"
        )
        if purpose:
            lines.append(f"申请用途：{purpose}")
        lines.append("")

        # ── 指标汇总表 ────────────────────────────────────────────────────────
        lines.append("**指标汇总**")
        lines.append("")
        lines.append("| 类别 | 指标 | 数值 | 说明 |")
        lines.append("|------|------|------|------|")

        ga = fv.get("gpu_avg")
        ta = fv.get("tensor_avg")
        da = fv.get("dram_avg")
        so = fv.get("sm_occ_avg")
        tr = fv.get("tensor_ratio")
        mp = fv.get("mem_pct")

        lines.append(f"| GPU计算 | SM 利用率 | **{_fmt(ga, '%')}** | 峰值 {_fmt(fv.get('gpu_max'), '%')}，抖动 σ={_fmt(fv.get('gpu_std'), '%')} |")
        lines.append(f"| GPU计算 | Tensor Core 活跃率 | {_fmt(ta, '%')} | 参与比 {_fmt(tr, '', '.0%') if tr else '—'} |")
        lines.append(f"| GPU计算 | 显存带宽利用（DRAM） | {_fmt(da, '%')} | >80% 为带宽受限 |")
        lines.append(f"| GPU计算 | SM 占用率 | {_fmt(so, '%')} | <40% 说明并行度不足 |")
        lines.append(f"| GPU计算 | Tensor TFLOPS | {_fmt(fv.get('tflops_avg'), ' TFLOPS')} | — |")
        lines.append(f"| 显存 | 显存占用率 | {_fmt(mp, '%', '.0f')} | >90% OOM 风险 |")
        lines.append(f"| 显存 | 显存带宽 | {_fmt(fv.get('mem_bw_avg'), ' GB/s')} | — |")
        lines.append(f"| 热管理 | GPU 温度 | {_fmt(fv.get('temp_avg'), '°C')} | 最高 {_fmt(fv.get('temp_max'), '°C')} |")
        lines.append(f"| 热管理 | GPU 功耗 | {_fmt(fv.get('power_avg'), ' W')} | 峰值 {_fmt(fv.get('power_max'), ' W')} |")
        lines.append(f"| CPU | CPU 利用率 | {_fmt(fv.get('cpu_avg'), '%')} | — |")
        lines.append(f"| CPU | CFS 节流率 | {_fmt(fv.get('cpu_throttled'), '%')} | >10% 影响 DataLoader |")
        lines.append(f"| 系统 | 系统内存 | {_fmt(fv.get('sys_mem_avg'), '%')} | — |")
        lines.append(f"| 存储 | 磁盘读取 | {_fmt(fv.get('disk_read_mb'), ' MB/s')} | — |")
        lines.append(f"| 存储 | CPFS 读延迟 | {_fmt(fv.get('cpfs_lat_avg'), ' ms')} | >5ms 影响数据加载 |")
        lines.append(f"| 网络 | 网络重传率 | {_fmt(fv.get('net_retran'), '%')} | >1% 影响 DDP |")
        lines.append(f"| 互联 | NVLink 状态 | {'活跃' if fv.get('nvlink_active') else '未激活'} | — |")
        lines.append("")

        # ── 模式标签 ──────────────────────────────────────────────────────────
        tag_str = " · ".join(_TAG_LABEL.get(t, t) for t in tags)
        lines.append(f"**识别模式**：{tag_str}")
        lines.append("")

        if "no_data" not in tags:
            advices = _build_advices(tags, task, fv)
            all_advices.extend(advices)

    # ── 汇总优化建议 ─────────────────────────────────────────────────────────
    if not all_advices:
        lines.append("✅ 所有实例运行状态良好，暂无高优先级优化项。")
    else:
        pri_order = {"紧急": 0, "高": 1, "中": 2, "低": 3}
        all_advices.sort(key=lambda x: pri_order.get(x["pri"], 4))

        seen, unique = set(), []
        for a in all_advices:
            if a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)

        lines.append("---")
        lines.append("## 优化建议（按优先级排序）")
        lines.append("")
        emoji_map = {"紧急": "🚨", "高": "🔴", "中": "🟡", "低": "🔵"}
        for i, a in enumerate(unique, 1):
            e = emoji_map.get(a["pri"], "⚪")
            lines.append(f"### 建议 {i}  {e} 优先级「{a['pri']}」· {a['title']}")
            lines.append("")
            lines.append(a["body"])
            lines.append("")

    return "\n".join(lines)


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

class GPUTrainingAdvisorSchema(BaseModel):
    open_id: str = Field(
        default="",
        description=(
            "飞书用户 open_id，用于查找该用户通过 Bot 创建的所有实例。"
            "飞书场景下系统会自动注入当前用户的 open_id，直接传入即可。"
        ),
    )
    instance_name: str = Field(
        default="",
        description="指定单个 DSW 实例名（如 'wuji-train-01'），提供时优先分析该实例。",
    )
    hours: int = Field(
        default=24,
        description="分析过去多少小时的数据，建议设为实例已运行时长，默认 24h。",
    )


gpu_training_advisor_tool = StructuredTool.from_function(
    func=analyze_user_gpu_training,
    name="analyze_gpu_training",
    description=(
        "深度分析用户 GPU 训练行为：并行查询 24 项 Prometheus 指标（SM 利用率、"
        "Tensor Core 活跃率、显存带宽、SM 占用率、实际 TFLOPS、温度、功耗、CPU CFS 节流、"
        "磁盘 I/O、CPFS 延迟、NVLink、网络重传、RDMA 错误），基于 Roofline 模型识别瓶颈，"
        "给出带代码示例的针对性建议。"
        "当用户问'分析我的GPU'、'训练建议'、'利用率低'、'怎么优化训练'、'Tensor Core' 时调用。"
    ),
    args_schema=GPUTrainingAdvisorSchema,
)
