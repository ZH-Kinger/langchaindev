"""
GPU 训练行为分析与算法优化建议工具。

从 Prometheus 读取用户 DSW 实例的 GPU/CPU/显存时序指标，识别训练瓶颈模式
（IO 瓶颈、显存压力、低利用率、多卡通信损耗等），给出模型算法层面的针对性建议。
"""
import time
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings


# ── DSW 实例级 Prometheus 指标模板 ──────────────────────────────────────────────

_INSTANCE_PROMQL = {
    "gpu_util":      'avg(AliyunPaidsw_INSTANCE_GPU_SM_UTIL{{instanceName="{n}"}})',
    "gpu_mem_used":  'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_USED{{instanceName="{n}"}}) / 1024',
    "gpu_mem_total": 'avg(AliyunPaidsw_INSTANCE_GPU_MEMORY_TOTAL{{instanceName="{n}"}}) / 1024',
    "cpu_util":      'avg(AliyunPaidsw_INSTANCE_CPU_UTIL{{instanceName="{n}"}})',
}


def _query_range_for_instance(name: str, start: float, end: float) -> dict:
    """查询单个 DSW 实例的时序指标，返回 {key: [float...]} 字典。"""
    from tools.prometheus_tool import _query_range
    result = {}
    for key, tmpl in _INSTANCE_PROMQL.items():
        promql = tmpl.format(n=name)
        series = _query_range(promql, start, end, step="120s")
        vals = []
        for s in series:
            for _, v in s.get("values", []):
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        result[key] = vals
    return result


def _stats(vals: list) -> dict:
    """基础统计量，空列表时全返回 None。"""
    if not vals:
        return {"avg": None, "max": None, "std": None, "p10": None, "p90": None}
    arr = np.array(vals, dtype=float)
    return {
        "avg": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


# ── Redis 实例查询 ───────────────────────────────────────────────────────────────

def _get_user_instances(open_id: str) -> list[dict]:
    """从 Redis 获取该用户通过 Bot 创建的所有跟踪实例。"""
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
    """从 Jira 工单 description 中提取 dsw_purpose 字段。"""
    if not ticket_key:
        return ""
    try:
        from tools.jira_tool import _s, _base, parse_ticket_metadata
        s = _s()
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


# ── 模式识别 ────────────────────────────────────────────────────────────────────

def _classify(s: dict) -> list[str]:
    """根据统计特征返回瓶颈模式标签列表（可多选）。"""
    gpu_avg = s.get("gpu_avg")
    if gpu_avg is None:
        return ["no_data"]

    gpu_std = s.get("gpu_std") or 0
    gpu_p10 = s.get("gpu_p10") or 0
    gpu_p90 = s.get("gpu_p90") or 0
    cpu_avg = s.get("cpu_avg")
    mem_pct = s.get("mem_pct")
    gpu_count = s.get("gpu_count", 1)

    tags = []

    # 利用率级别
    if gpu_avg < 15:
        tags.append("severe_idle")
    elif gpu_avg < 35:
        tags.append("low_util")
    elif gpu_avg > 80:
        tags.append("compute_bound")

    # IO 瓶颈：GPU 等数据（GPU 中高但 CPU 低）
    if gpu_avg > 40 and cpu_avg is not None and cpu_avg < 30:
        tags.append("io_bound")

    # CPU 饱和（预处理抢占训练进程）
    if cpu_avg is not None and cpu_avg > 75:
        tags.append("cpu_saturated")

    # 显存压力
    if mem_pct is not None:
        if mem_pct > 90:
            tags.append("mem_critical")
        elif mem_pct > 75:
            tags.append("mem_high")

    # 间歇性使用（p10≈0 但 p90 较高 → 训练循环有阻塞）
    if gpu_p10 < 5 and gpu_p90 > 40:
        tags.append("intermittent")

    # 负载剧烈抖动
    if gpu_std > 25 and gpu_avg > 20:
        tags.append("spiky")

    # 多卡效率不足
    if gpu_count >= 4 and gpu_avg < 60:
        tags.append("multi_gpu_inefficient")

    return tags or ["normal"]


# ── 任务类型推断 ─────────────────────────────────────────────────────────────────

_PURPOSE_MAP = {
    "finetune":     ["微调", "finetune", "fine-tune", "sft", "rlhf", "lora", "qlora", "peft", "指令"],
    "pretrain":     ["预训练", "pretrain", "pre-train", "from scratch", "预训"],
    "inference":    ["推理", "inference", "serving", "部署", "vllm", "tensorrt"],
    "distillation": ["蒸馏", "distill", "知识蒸馏"],
    "multimodal":   ["多模态", "multimodal", "vlm", "clip", "diffusion", "图像生成"],
}

_TASK_LABEL = {
    "finetune": "微调", "pretrain": "预训练",
    "inference": "推理", "distillation": "蒸馏",
    "multimodal": "多模态", "general": "通用训练",
}


def _infer_task(purpose: str) -> str:
    p = purpose.lower()
    for task, kws in _PURPOSE_MAP.items():
        if any(k in p for k in kws):
            return task
    return "general"


# ── 建议生成（规则引擎）──────────────────────────────────────────────────────────

def _build_advices(tags: list, task: str, s: dict, gpu_count: int) -> list[dict]:
    """返回 [{"pri": "高/中/低", "title": str, "body": str}, ...]"""
    advs = []
    gpu_avg = s.get("gpu_avg", 0) or 0
    mem_pct = s.get("mem_pct") or 0

    # ── 显存告急 ────────────────────────────────────────────────────────────────
    if "mem_critical" in tags:
        advs.append({"pri": "高", "title": f"显存占用 {mem_pct:.0f}%，OOM 风险高",
            "body": (
                "**立即可做（三选一或组合）：**\n"
                "① 混合精度（节省显存 30–50%）：\n"
                "```python\nfrom torch.cuda.amp import autocast, GradScaler\nscaler = GradScaler()\nwith autocast():\n    loss = model(inputs)\nscaler.scale(loss).backward()\n```\n"
                "② 梯度检查点（以重算换显存）：\n"
                "```python\nmodel.gradient_checkpointing_enable()  # HuggingFace 模型一行搞定\n```\n"
                "③ 缩小 batch_size + 增大梯度累积步数，等效 batch 不变：\n"
                "```python\n# 原 batch=32 → 改为 batch=8, accumulation_steps=4\noptimizer.zero_grad()\nfor i, batch in enumerate(loader):\n    loss = model(batch) / accumulation_steps\n    loss.backward()\n    if (i+1) % accumulation_steps == 0:\n        optimizer.step(); optimizer.zero_grad()\n```"
            )
        })

    elif "mem_high" in tags:
        if task == "finetune":
            advs.append({"pri": "中", "title": "显存偏高，推荐 LoRA 代替全量微调",
                "body": (
                    "LoRA 只训练 0.1%–1% 的参数，显存可降低 60–80%，精度损失极小：\n"
                    "```python\nfrom peft import get_peft_model, LoraConfig, TaskType\nconfig = LoraConfig(\n    r=16, lora_alpha=32,\n    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],\n    task_type=TaskType.CAUSAL_LM,\n)\nmodel = get_peft_model(model, config)\nmodel.print_trainable_parameters()  # 查看可训练参数量\n```"
                )
            })
        else:
            advs.append({"pri": "中", "title": "显存偏高，建议开启 bf16 训练",
                "body": (
                    "A10/A100 原生支持 bf16，比 fp16 更稳定（无需 loss scaling）：\n"
                    "```python\n# HuggingFace Trainer\nTrainingArguments(bf16=True, ...)\n# 原生 PyTorch\nmodel = model.to(torch.bfloat16)\noptimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)\n```"
                )
            })

    # ── IO 瓶颈 ─────────────────────────────────────────────────────────────────
    if "io_bound" in tags:
        advs.append({"pri": "高", "title": "DataLoader 成为瓶颈，GPU 在等待数据",
            "body": (
                f"GPU 利用率 {gpu_avg:.0f}% 但 CPU 较低，数据管线是性能天花板。\n"
                "**DataLoader 参数优化**：\n"
                "```python\ntrain_loader = DataLoader(\n    dataset,\n    batch_size=64,\n    num_workers=8,          # 设为 CPU 核数的 50%\n    pin_memory=True,        # 锁页内存，加速 CPU→GPU 传输\n    prefetch_factor=2,      # 每个 worker 预取 2 个 batch\n    persistent_workers=True,# 避免每 epoch 重建进程\n)\n```\n"
                "**彻底解决：离线预处理数据**\n"
                "```python\n# HuggingFace datasets - 一次处理，永久缓存\nds = ds.map(\n    tokenize_fn,\n    batched=True,\n    num_proc=8,\n    cache_file_name='data/tokenized_cache'\n)\n```\n"
                "若使用图像数据，将数据集复制到 `/dev/shm`（内存文件系统）可再提速 2–3×。"
            )
        })

    # ── CPU 饱和 ─────────────────────────────────────────────────────────────────
    if "cpu_saturated" in tags:
        advs.append({"pri": "中", "title": "CPU 持续高负载，预处理与训练互相抢占",
            "body": (
                "① 将数据增强/tokenize 提前离线完成，训练时只做 `__getitem__` 读取\n"
                "② 图像增强改用 Albumentations + OpenCV 后端（比 PIL 快 3×）\n"
                "```python\nimport albumentations as A\ntransform = A.Compose([\n    A.RandomCrop(224, 224),\n    A.HorizontalFlip(),\n    A.Normalize(mean, std),\n], additional_targets={'image': 'image'})\n```\n"
                "③ 避免在 `collate_fn` 中做复杂计算，仅做 padding 和 stacking"
            )
        })

    # ── 间歇性使用 ───────────────────────────────────────────────────────────────
    if "intermittent" in tags:
        advs.append({"pri": "中", "title": "训练循环中存在 CPU 同步阻塞",
            "body": (
                "GPU 利用率忽高忽低，常见原因是训练循环中有强制 CPU 同步的操作：\n"
                "```python\n# ❌ 每步都触发 CPU 同步（GPU 停下来等）\nprint(f'loss: {loss.item()}')  # .item() 会同步\n\n# ✅ 改为每 100 步打印一次\nif step % 100 == 0:\n    print(f'loss: {loss.item()}')\n```\n"
                "其他常见阻塞点：\n"
                "• 每步写 TensorBoard（改为 `log_every_n_steps=50`）\n"
                "• 每步做验证集评估（改为每 epoch 或每 N 步）\n"
                "• 变长序列未做长度分组，导致 padding 开销大 → 使用 BucketSampler"
            )
        })

    # ── 负载抖动 ─────────────────────────────────────────────────────────────────
    if "spiky" in tags and "intermittent" not in tags:
        advs.append({"pri": "低", "title": "GPU 利用率波动大，影响吞吐稳定性",
            "body": (
                "① NLP 任务：按序列长度分桶，减少同一 batch 内长度差异：\n"
                "```python\nfrom transformers import DataCollatorWithPadding\ncollator = DataCollatorWithPadding(tokenizer, padding='longest')  # 动态 padding\n```\n"
                "② 确保数据集充分 shuffle（`shuffle=True` + `seed` 固定）\n"
                "③ 验证阶段利用率下降属正常，确保用 `torch.no_grad()` + `model.eval()`"
            )
        })

    # ── 低利用率 ─────────────────────────────────────────────────────────────────
    if "severe_idle" in tags or "low_util" in tags:
        level = "高" if "severe_idle" in tags else "中"
        advs.append({"pri": level, "title": f"GPU 严重低利用（均值 {gpu_avg:.0f}%），算力大量浪费",
            "body": (
                "**快速诊断（在实例终端运行）**：\n"
                "```bash\n# 实时观察 GPU 状态\nwatch -n 1 nvidia-smi\n\n# 分析每步训练速度（samples/sec）\n# 如果速度正常但利用率低，说明模型计算量太小\n```\n"
                "**常见原因及修复**：\n"
                f"① `batch_size` 太小 → 当前显存允许的最大值（目标 70%–85% 显存占用）\n"
                "② 频繁写 checkpoint / 大量 I/O → 减少保存频率\n"
                "③ 数据在 HDD → 将数据集移到 SSD 或 `/dev/shm`\n"
                "④ 模型太小但 GPU 太多 → 单 GPU 跑完全足够，不必多卡"
            )
        })

    # ── 多卡效率 ─────────────────────────────────────────────────────────────────
    if "multi_gpu_inefficient" in tags:
        advs.append({"pri": "中", "title": f"{gpu_count} 卡场景下利用率偏低，分布式策略需调整",
            "body": (
                "多卡 GPU 利用率低通常是通信开销过大导致的：\n"
                "① **增大 batch_size**：通信量与 batch 成反比，batch 越大通信占比越小\n"
                "② **DDP 梯度分桶**：减少通信次数\n"
                "```python\nfrom torch.nn.parallel import DistributedDataParallel as DDP\nmodel = DDP(model, bucket_cap_mb=25)  # 默认 25MB，可适当调大\n```\n"
                "③ **7B+ 大模型改用 FSDP**（比 DDP 更省显存 + 支持更大 batch）：\n"
                "```python\nfrom torch.distributed.fsdp import FullyShardedDataParallel as FSDP\nfrom torch.distributed.fsdp import ShardingStrategy\nmodel = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)\n```\n"
                "④ 若通信带宽不足（NVLink/InfiniBand 未启用），联系运维开启 NVLink"
            )
        })

    # ── 任务专项建议 ──────────────────────────────────────────────────────────────
    if task == "finetune" and "mem_critical" not in tags:
        advs.append({"pri": "低", "title": "微调专项：学习率策略与训练稳定性",
            "body": (
                "① **Cosine Warmup 调度**（防止初始大 LR 破坏预训练权重）：\n"
                "```python\nfrom transformers import get_cosine_schedule_with_warmup\nscheduler = get_cosine_schedule_with_warmup(\n    optimizer,\n    num_warmup_steps=total_steps // 10,  # 前 10% 步骤 warmup\n    num_training_steps=total_steps,\n)\n```\n"
                "② 学习率建议：全量微调 `1e-5~5e-5`，LoRA `1e-4~3e-4`\n"
                "③ 开启梯度裁剪防止梯度爆炸：`clip_grad_norm_(model.parameters(), 1.0)`\n"
                "④ 指令微调注意 **仅对 response 部分计算 loss**（忽略 prompt 部分）"
            )
        })

    elif task == "pretrain":
        advs.append({"pri": "低", "title": "预训练专项：吞吐与效率最大化",
            "body": (
                "① **Flash Attention 2**（注意力计算减少约 50% 显存和时间，预训练必备）：\n"
                "```python\nfrom transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_id,\n    attn_implementation='flash_attention_2',\n    torch_dtype=torch.bfloat16,\n)\n```\n"
                "② **torch.compile()**（首次编译 1–2 min，后续加速 20–30%）：\n"
                "```python\nmodel = torch.compile(model)  # PyTorch 2.0+\n```\n"
                "③ 全局 batch size 目标 1M–4M tokens，不足时用 gradient_accumulation\n"
                "④ DeepSpeed ZeRO-2/3 支持跨节点显存聚合，突破单节点限制"
            )
        })

    elif task == "inference":
        advs.append({"pri": "低", "title": "推理专项：吞吐与延迟优化",
            "body": (
                "① **vLLM**（生产推理首选，PagedAttention，吞吐提升 3–10×）：\n"
                "```python\nfrom vllm import LLM, SamplingParams\nllm = LLM(model='your-model-path', tensor_parallel_size=4)\noutputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=256))\n```\n"
                "② **INT4/AWQ 量化**（显存减半，速度基本不变）：\n"
                "```python\nfrom awq import AutoAWQForCausalLM\nmodel = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True)\n```\n"
                "③ 静态 batch 推理时，batch_size 填满约 70% 显存即可，更大边际收益降低"
            )
        })

    # ── 高效运行补充建议 ─────────────────────────────────────────────────────────
    if "compute_bound" in tags:
        advs.append({"pri": "低", "title": "利用率优秀，可尝试进一步提速",
            "body": (
                f"当前 GPU 均值利用率 {gpu_avg:.0f}%，处于理想区间，进阶优化：\n"
                "① `torch.compile(model)` — 算子融合，通常加速 15–30%\n"
                "② Flash Attention 2 — 若尚未开启\n"
                "③ `torch.backends.cudnn.benchmark = True` — 输入尺寸固定时有效\n"
                "④ 检查是否有冗余的 `.cpu()` / `.numpy()` 调用拖慢热路径"
            )
        })

    return advs


# ── 主分析函数 ────────────────────────────────────────────────────────────────────

def analyze_user_gpu_training(
    open_id: str = "",
    instance_name: str = "",
    hours: int = 24,
) -> str:
    """
    分析用户 GPU 训练使用模式，识别瓶颈，给出模型/算法层面的优化建议。

    参数优先级：instance_name > open_id（通过 Bot 申请的实例列表）。
    """
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，无法读取 GPU 指标。"

    end   = time.time()
    start = end - hours * 3600

    # ── 确定分析目标 ──────────────────────────────────────────────────────────────
    targets: list[dict] = []

    if instance_name:
        targets = [{
            "instance_name": instance_name,
            "gpu_count":     1,
            "start_ts":      start,
            "ticket_key":    "",
            "open_id":       open_id,
        }]
    elif open_id:
        targets = _get_user_instances(open_id)
        if not targets:
            return (
                "未找到你通过 Bot 创建的运行中实例记录。\n"
                "若实例正在运行，可直接指定实例名：**分析实例 `<实例名>`**\n"
                "（Redis 中的实例记录保留 7 天）"
            )
    else:
        return "❌ 请提供 open_id 或 instance_name 参数。"

    # ── 逐实例拉取指标 ────────────────────────────────────────────────────────────
    profiles: list[dict] = []

    for inst in targets:
        name = inst.get("instance_name", "")
        if not name:
            continue

        inst_start = max(start, float(inst.get("start_ts", start)))
        raw = _query_range_for_instance(name, inst_start, end)

        gpu_util_vals = raw.get("gpu_util", [])
        gpu_mem_used  = raw.get("gpu_mem_used", [])
        gpu_mem_total = raw.get("gpu_mem_total", [])
        cpu_util_vals = raw.get("cpu_util", [])

        gpu_s = _stats(gpu_util_vals)
        cpu_s = _stats(cpu_util_vals)

        # 显存占用率
        mem_pct = None
        if gpu_mem_used and gpu_mem_total:
            avg_used  = sum(gpu_mem_used)  / len(gpu_mem_used)
            avg_total = sum(gpu_mem_total) / len(gpu_mem_total)
            if avg_total > 0:
                mem_pct = avg_used / avg_total * 100

        gpu_count = int(inst.get("gpu_count", 1))
        feature_vec = {
            "gpu_avg": gpu_s["avg"], "gpu_max": gpu_s["max"],
            "gpu_std": gpu_s["std"], "gpu_p10": gpu_s["p10"], "gpu_p90": gpu_s["p90"],
            "cpu_avg": cpu_s["avg"],
            "mem_pct": mem_pct,
            "gpu_count": gpu_count,
            "data_points": len(gpu_util_vals),
        }

        purpose   = inst.get("purpose", "") or _get_purpose_from_jira(inst.get("ticket_key", ""))
        task_type = _infer_task(purpose)
        tags      = _classify(feature_vec)

        profiles.append({
            "name":     name,
            "fv":       feature_vec,
            "tags":     tags,
            "task":     task_type,
            "purpose":  purpose,
            "gpu_count": gpu_count,
        })

    if not profiles:
        return "⚠️ Prometheus 中未查到实例指标数据（实例可能已停止或指标延迟 2–3 分钟）。"

    # ── 拼装报告 ─────────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"## GPU 训练行为分析",
        f"分析周期：近 {hours}h　生成时间：{now_str}　实例数：{len(profiles)}",
        "",
    ]

    _TAG_LABEL = {
        "severe_idle":          "🔴 严重空转",
        "low_util":             "🟡 利用率偏低",
        "io_bound":             "🟠 数据管线瓶颈",
        "cpu_saturated":        "🟠 CPU 饱和",
        "mem_critical":         "🔴 显存告急",
        "mem_high":             "🟡 显存压力大",
        "intermittent":         "🟡 训练阻塞",
        "spiky":                "🟡 负载波动",
        "multi_gpu_inefficient":"🟡 多卡效率低",
        "compute_bound":        "✅ 计算密集（良好）",
        "normal":               "✅ 运行正常",
        "no_data":              "⚪ 指标不可达",
    }

    all_advices: list[dict] = []

    for p in profiles:
        fv      = p["fv"]
        tags    = p["tags"]
        name    = p["name"]
        purpose = p["purpose"]
        task    = p["task"]
        pts     = fv.get("data_points", 0)

        g_avg = fv.get("gpu_avg")
        g_str = f"{g_avg:.1f}%" if g_avg is not None else "无数据"
        c_str = f"{fv['cpu_avg']:.1f}%" if fv.get("cpu_avg") is not None else "—"
        m_str = f"{fv['mem_pct']:.0f}%" if fv.get("mem_pct") is not None else "—"

        lines.append(f"### 实例：`{name}`")
        lines.append(
            f"GPU 卡数：{p['gpu_count']} 卡　"
            f"任务类型：{_TASK_LABEL.get(task, '通用训练')}　"
            f"采样点：{pts} 个"
        )
        if purpose:
            lines.append(f"申请用途：{purpose}")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| GPU SM 利用率均值 | **{g_str}** |")
        if fv.get("gpu_max") is not None:
            lines.append(f"| GPU SM 利用率峰值 | {fv['gpu_max']:.1f}% |")
        if fv.get("gpu_std") is not None:
            lines.append(f"| 利用率标准差（波动） | {fv['gpu_std']:.1f}% |")
        lines.append(f"| CPU 利用率均值 | {c_str} |")
        lines.append(f"| GPU 显存占用率 | {m_str} |")
        lines.append("")

        tag_strs = " · ".join(_TAG_LABEL.get(t, t) for t in tags)
        lines.append(f"**识别模式**：{tag_strs}")
        lines.append("")

        if "no_data" not in tags:
            advices = _build_advices(tags, task, fv, p["gpu_count"])
            all_advices.extend(advices)

    # ── 汇总优化建议 ─────────────────────────────────────────────────────────────
    if not all_advices:
        lines.append("✅ 所有实例运行状态良好，暂无高优先级优化项。")
    else:
        pri_order = {"高": 0, "中": 1, "低": 2}
        all_advices.sort(key=lambda x: pri_order.get(x["pri"], 3))

        # 去重
        seen, unique = set(), []
        for a in all_advices:
            if a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)

        lines.append("---")
        lines.append("## 优化建议")
        lines.append("")
        pri_emoji = {"高": "🔴", "中": "🟡", "低": "🔵"}
        for i, a in enumerate(unique, 1):
            lines.append(f"### 建议 {i}  {pri_emoji.get(a['pri'], '⚪')} 优先级{a['pri']} · {a['title']}")
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
            "飞书场景下系统会自动注入当前用户的 open_id，直接使用即可。"
        ),
    )
    instance_name: str = Field(
        default="",
        description=(
            "指定单个 DSW 实例名称（如 'wuji-train-01'）。"
            "提供此参数时优先分析该实例，忽略 open_id 的实例列表查询。"
        ),
    )
    hours: int = Field(
        default=24,
        description="分析过去多少小时的数据，默认 24 小时，建议设为实例已运行时长。",
    )


gpu_training_advisor_tool = StructuredTool.from_function(
    func=analyze_user_gpu_training,
    name="analyze_gpu_training",
    description=(
        "分析用户 GPU 实例的训练行为（SM 利用率、显存、CPU-GPU 关系），"
        "识别 IO 瓶颈、显存压力、低利用率、多卡效率不足等问题，"
        "给出针对微调/预训练/推理等不同任务类型的模型算法优化建议（含代码示例）。"
        "当用户问'分析我的GPU使用'、'训练建议'、'为什么利用率低'、'怎么优化训练' 时调用。"
        "若用户在飞书中，open_id 已注入到输入中，直接传入即可。"
    ),
    args_schema=GPUTrainingAdvisorSchema,
)
