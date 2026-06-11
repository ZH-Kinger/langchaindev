"""
多区域集群算力效率（MFU）+ 资源容量 + 节点调度/碎片化 日报工具（交互式卡片）。

自动发现所有区域（杭州/北京/新加坡…），每区按自己的 GPU 卡型取单卡峰值。
飞书卡片带区域切换按钮：默认「全局汇总」，点区域名原地切到该区详情面板（保持面板简洁）。

每区面板分两块：
  配额（当前）：总卡 / 已申请（申请率）+ CPU核 / 内存 / 显存(已分配/总)。
  昨日效率（自然日昨天 00:00–24:00 北京，range）：MFU 均值/峰值/谷值、张量活跃、显存带宽活跃、
    在算卡、实测TFLOPS、NVLink/PCIe 卡均带宽。
  节点调度：每节点容量/已分配/空闲 → 整空节点/碎片卡/分布 + 拖后腿节点；诊断整机训练为何排不进。

坑/口径：在算卡用 PIP_TENSOR_ACTIVE 存在性（弃用漏报的 GPU_HEALTH）；按 regionId 区分；
单卡峰值/显存按卡型映射（settings.GPU_PEAK_TFLOPS_BY_TYPE / GPU_MEM_GB_BY_TYPE）。
浪费 = 空占卡·时(昨日) + 碎片卡·时(当前) × 单价。卡片快照缓存 Redis 15min，供按钮秒级切换。
"""
import json
from datetime import datetime, timezone, timedelta

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings
from tools.feishu.cards import btn, buttons, card, div, fields, hr

_BJ = timezone(timedelta(hours=8))
_REGION_DISPLAY = {"cn-hangzhou": "杭州", "cn-beijing": "北京", "ap-southeast-1": "新加坡"}
_CACHE_KEY = "mfu:snapshot"
_CACHE_TTL = 900   # 15 分钟，供卡片按钮切换复用，避免每次点击重算


def _region_name(r): return _REGION_DISPLAY.get(r, r)
def _gpu_name(t): return settings.GPU_TYPE_DISPLAY.get(t, t or "?")
def _peak_for(t): return float(settings.GPU_PEAK_TFLOPS_BY_TYPE.get(t, settings.GPU_PEAK_TFLOPS))
def _mem_for(t): return float(settings.GPU_MEM_GB_BY_TYPE.get(t, 0))
def _sel(region): return f'{{regionId="{region}"}}' if region else "{}"


def _yesterday_window():
    today_bj = datetime.now(_BJ).replace(hour=0, minute=0, second=0, microsecond=0)
    start = today_bj - timedelta(days=1)
    return start.timestamp(), today_bj.timestamp(), start.strftime("%Y-%m-%d")


# ── PromQL ──────────────────────────────────────────────────────────────────────

def _scalar(promql, default=0.0):
    from tools.aliyun.prometheus import _query_instant
    try:
        r = _query_instant(promql)
        if r:
            v = r[0].get("value", [None, None])[1]
            if v not in (None, "NaN", "Inf", "+Inf", "-Inf"):
                return float(v)
    except Exception:
        pass
    return default


def _grouped(promql, keys):
    from tools.aliyun.prometheus import _query_instant
    out = {}
    try:
        for x in _query_instant(promql):
            k = tuple(x["metric"].get(kk, "") for kk in keys)
            v = x.get("value", [None, None])[1]
            if v not in (None, "NaN", "Inf", "+Inf", "-Inf"):
                out[k] = float(v)
    except Exception:
        pass
    return out


def _range_series(promql, start, end, step="3600s"):
    """返回 {ts: val}（用于按时间对齐算逐时 MFU）。"""
    from tools.aliyun.prometheus import _query_range
    out = {}
    try:
        for s in _query_range(promql, start, end, step=step):
            for ts, v in s.get("values", []):
                if v not in (None, "NaN", "Inf", "+Inf", "-Inf"):
                    out[ts] = out.get(ts, 0.0) + float(v)
    except Exception:
        pass
    return out


def _mean(d):
    return sum(d.values()) / len(d) if d else 0.0


# ── 区域发现 ────────────────────────────────────────────────────────────────────

def _discover_regions():
    from tools.aliyun.prometheus import _query_instant
    allow = [x.strip() for x in (settings.CLUSTER_REGION or "").split(",") if x.strip()]
    gtype = {}
    try:
        for x in _query_instant("AliyunPaiquota_NODE_GPU_ACCELERATOR_TOTAL"):
            m = x["metric"]; r = m.get("regionId")
            if r and not gtype.get(r):
                gtype[r] = m.get("nodeGpuType", "")
    except Exception:
        pass
    try:
        for x in _query_instant("AliyunPaiquota_QUOTA_GPU_ACCELERATOR_TOTAL"):
            r = x["metric"].get("regionId")
            if r and r not in gtype:
                gtype[r] = ""
    except Exception:
        pass
    regions = [(r, t) for r, t in gtype.items() if (not allow or r in allow)]
    regions.sort()
    return regions


# ── 单区域采集 ──────────────────────────────────────────────────────────────────

def _capacity(region, gpu_type):
    s = _sel(region)
    total = int(_scalar(f"sum(AliyunPaiquota_QUOTA_GPU_ACCELERATOR_TOTAL{s})"))
    mem_alloc = _scalar(f"sum(AliyunPaiquota_QUOTA_GPU_MEMORY_ALLOCATED{s})")
    mem_total = total * _mem_for(gpu_type)
    return {
        "gpu_total":     total,
        "gpu_request":   int(_scalar(f"sum(AliyunPaiquota_QUOTA_GPU_ACCELERATOR_REQUEST{s})")),
        "cpu_total":     _scalar(f"sum(AliyunPaiquota_QUOTA_CPU_TOTAL{s})"),
        "cpu_request":   _scalar(f"sum(AliyunPaiquota_QUOTA_CPU_REQUEST{s})"),
        "mem_total":     _scalar(f"sum(AliyunPaiquota_QUOTA_MEMORY_TOTAL{s})"),
        "mem_request":   _scalar(f"sum(AliyunPaiquota_QUOTA_MEMORY_REQUEST{s})"),
        "gpu_mem_alloc": mem_alloc,
        "gpu_mem_total": mem_total,
        "gpu_mem_util":  (mem_alloc / mem_total * 100) if mem_total else 0.0,
    }


def _yesterday(region, peak):
    s, thr = _sel(region), settings.GPU_ACTIVE_THRESHOLD_PCT
    start, end, date = _yesterday_window()
    alloc = _mean(_range_series(f"sum(AliyunPaiquota_QUOTA_GPU_ACCELERATOR_REQUEST{s})", start, end))
    tf_s  = _range_series(f"sum(AliyunPaidlc_CARD_GPU_TENSORTFLOPS_USED{s})", start, end)
    tf_s2 = _range_series(f"sum(AliyunPaidsw_CARD_GPU_TENSORTFLOPS_USED{s})", start, end)
    ac_s  = _range_series(f"count(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE{s} > {thr})", start, end)
    ac_s2 = _range_series(f"count(AliyunPaidsw_CARD_GPU_PIP_TENSOR_ACTIVE{s} > {thr})", start, end)
    # 逐时对齐算 MFU 峰/谷
    tflops = {t: tf_s.get(t, 0) + tf_s2.get(t, 0) for t in set(tf_s) | set(tf_s2)}
    active = {t: ac_s.get(t, 0) + ac_s2.get(t, 0) for t in set(ac_s) | set(ac_s2)}
    mfus = [tflops.get(t, 0) / (active[t] * peak) * 100
            for t in active if active[t] > 0 and peak]
    tflops_avg, active_avg = _mean(tflops), _mean(active)
    mfu_avg = (tflops_avg / (active_avg * peak) * 100) if (active_avg and peak) else 0.0
    return {
        "alloc_avg": alloc, "active_avg": active_avg, "tflops_avg": tflops_avg,
        "gpu_util":      _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_DUTY_CYCLE{s})", start, end)),
        "sm_util":       _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_SM_UTIL{s})", start, end)),
        "tensor_active": _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE{s})", start, end)),
        "dram_active":   _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_DRAM_ACTIVE_UTIL{s})", start, end)),
        # NVLink/PCIe 指标为卡均吞吐(MiB/s)，展示时 /1024 → GiB/s
        "nvlink_rx": _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_NVLINK_RECEIVE{s})", start, end)),
        "nvlink_tx": _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_NVLINK_TRANSMIT{s})", start, end)),
        "pcie_rx":   _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_PCIE_RECEIVE{s})", start, end)),
        "pcie_tx":   _mean(_range_series(f"avg(AliyunPaidlc_CARD_GPU_PCIE_TRANSMIT{s})", start, end)),
        "mfu": mfu_avg, "mfu_peak": max(mfus) if mfus else 0.0, "mfu_low": min(mfus) if mfus else 0.0,
        "idle_avg": max(0.0, alloc - active_avg), "idle_card_hours": max(0.0, alloc - active_avg) * 24,
        "date": date,
    }


def _nodes(region):
    s = _sel(region)
    cap = _grouped(f"AliyunPaiquota_NODE_GPU_ACCELERATOR_TOTAL{s}", ("nodeId",))
    req = _grouped(f"AliyunPaiquota_NODE_GPU_ACCELERATOR_REQUEST{s}", ("nodeId",))
    ta  = _grouped(f"avg by (nodeName)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE{s})", ("nodeName",))
    nodes = [{"node": k[0], "cap": int(c), "alloc": int(req.get(k, 0)),
              "free": max(0, int(c) - int(req.get(k, 0)))} for k, c in cap.items()]
    node_cap = nodes[0]["cap"] if nodes else 8
    dist = {}
    for n in nodes:
        dist[n["free"]] = dist.get(n["free"], 0) + 1
    # 拖后腿节点：在算节点里张量活跃明显低于中位数的
    acts = sorted(v for v in ta.values() if v > 0)
    med = acts[len(acts) // 2] if acts else 0.0
    stragglers = sorted(((k[0], v) for k, v in ta.items() if 0 < v < med * 0.5),
                        key=lambda x: x[1])[:3]
    return {
        "node_count": len(nodes), "node_cap": node_cap,
        "free_total": sum(n["free"] for n in nodes),
        "fully_free_nodes": sum(1 for n in nodes if n["cap"] > 0 and n["free"] == n["cap"]),
        "frag_free": sum(n["free"] for n in nodes if 0 < n["free"] < n["cap"]),
        "free_dist": dict(sorted(dist.items())),
        "active_nodes": sum(1 for v in ta.values() if v > 0),
        "ta_median": med,
        "stragglers": [{"node": n, "ta": v} for n, v in stragglers],
    }


def _dlc_jobs(region, peak):
    s, keys = _sel(region), ("displayName", "username")
    ta = _grouped(f"avg by (displayName, username)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE{s})", keys)
    tf = _grouped(f"sum by (displayName, username)(AliyunPaidlc_CARD_GPU_TENSORTFLOPS_USED{s})", keys)
    cd = _grouped(f"count by (displayName, username)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE{s})", keys)
    jobs = []
    for k, n in cd.items():
        n = int(n); used = tf.get(k, 0.0)
        jobs.append({"name": k[0] or "(未命名)", "user": k[1] or "—", "cards": n,
                     "tensor_active": ta.get(k, 0.0),
                     "mfu": (used / (n * peak) * 100) if (n and peak) else 0.0})
    jobs.sort(key=lambda j: -j["cards"])
    return jobs


def gather_region(region, gpu_type):
    peak, price = _peak_for(gpu_type), settings.GPU_PRICE_PER_HOUR
    y, cap, nd = _yesterday(region, peak), _capacity(region, gpu_type), _nodes(region)
    return {
        "region": region, "region_name": _region_name(region),
        "gpu_type": gpu_type, "gpu_name": _gpu_name(gpu_type), "peak_tflops": peak,
        "yesterday": y, **cap, "nodes": nd,
        "alloc_rate": (cap["gpu_request"] / cap["gpu_total"] * 100) if cap["gpu_total"] else 0.0,
        "cpu_rate":   (cap["cpu_request"] / cap["cpu_total"] * 100) if cap["cpu_total"] else 0.0,
        "mem_rate":   (cap["mem_request"] / cap["mem_total"] * 100) if cap["mem_total"] else 0.0,
        "waste_idle": y["idle_avg"] * 24 * price,
        "waste_frag": nd["frag_free"] * 24 * price,
        "waste_total": (y["idle_avg"] + nd["frag_free"]) * 24 * price,
        "jobs": _dlc_jobs(region, peak),
    }


def gather_all():
    rs = [gather_region(r, t) for r, t in _discover_regions()]
    _, _, date = _yesterday_window()
    return {
        "date": date, "regions": rs,
        "gpu_total": sum(d["gpu_total"] for d in rs),
        "gpu_request": sum(d["gpu_request"] for d in rs),
        "tflops": sum(d["yesterday"]["tflops_avg"] for d in rs),
        "waste_total": sum(d["waste_total"] for d in rs),
    }


# ── 缓存（供卡片按钮秒级切换）────────────────────────────────────────────────────

def _load_or_gather(refresh=False):
    from utils import redis_client
    if not refresh:
        try:
            raw = redis_client.get_redis().get(_CACHE_KEY)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    g = gather_all()
    try:
        redis_client.get_redis().set(_CACHE_KEY, json.dumps(g), ex=_CACHE_TTL)
    except Exception:
        pass
    return g


# ── 渲染 ────────────────────────────────────────────────────────────────────────

def _pct(v): return f"{v:.1f}%"


def _fmt_dist(nd):
    """空闲卡分布 {0:26,2:1,8:1} → '26满·1剩2·1整空'（key 可能是 JSON 字符串）。"""
    cap = nd["node_cap"]
    parts = []
    for free, cnt in sorted(nd["free_dist"].items(), key=lambda kv: int(kv[0])):
        free = int(free)
        if free == 0:
            parts.append(f"{cnt}满")
        elif free == cap:
            parts.append(f"{cnt}整空")
        else:
            parts.append(f"{cnt}剩{free}")
    return " · ".join(parts) if parts else "—"


def _region_elements(d):
    """单区域面板（飞书卡片：fields 栅格对齐 + 诊断 bullets）。"""
    low = settings.MFU_LOW_THRESHOLD_PCT
    y, nd = d["yesterday"], d["nodes"]
    mfu = "—" if y["active_avg"] == 0 else f"{_pct(y['mfu'])}（峰{_pct(y['mfu_peak'])}/谷{_pct(y['mfu_low'])}）"

    els = [
        div(f"**▍{d['region_name']} · {d['gpu_name']}**　峰值 {d['peak_tflops']:.0f} TFLOPS"
            f"　{nd['node_count']}×{nd['node_cap']} 卡　空闲分布 {_fmt_dist(nd)}"),
        fields(
            ("GPU 卡 总/申请", f"{d['gpu_total']}/{d['gpu_request']}（{_pct(d['alloc_rate'])}）"),
            ("显存 已分配/总", f"{d['gpu_mem_alloc']:.0f}/{d['gpu_mem_total']:.0f}G（{_pct(d['gpu_mem_util'])}）"),
            ("CPU 核 总/申请", f"{d['cpu_total']:.0f}/{d['cpu_request']:.0f}"),
            ("内存 总/申请", f"{d['mem_total']:.0f}/{d['mem_request']:.0f}G"),
            ("昨日 MFU", mfu),
            ("GPU使用率 / SM使用率", f"{_pct(y.get('gpu_util', 0))} / {_pct(y.get('sm_util', 0))}"),
            ("张量活跃 / 显存带宽", f"{_pct(y['tensor_active'])} / {_pct(y['dram_active'])}"),
            ("在算 / 算力", f"{y['active_avg']:.0f}卡 / {y['tflops_avg']:.0f}TF"),
            ("NVLink 收/发", f"{y['nvlink_rx']/1024:.1f}/{y['nvlink_tx']/1024:.1f} GiB/s"),
            ("PCIe 收/发", f"{y['pcie_rx']/1024:.1f}/{y['pcie_tx']/1024:.1f} GiB/s"),
            ("节点 空闲/整空/碎片", f"{nd['free_total']}/{nd['fully_free_nodes']}/{nd['frag_free']}"),
        ),
    ]
    bullets = []
    full = nd["fully_free_nodes"]
    if nd["free_total"] > 0 and full < 2 and nd["frag_free"] > 0:
        bullets.append(f"🚧 **训练排不进**：{nd['free_total']} 空闲但仅 {full} 整空节点，"
                       f"{nd['frag_free']} 碎片卡凑不出整机 → ≥{max(2, full+1)} 整机训练无法调度")
    if d["gpu_total"] > 0 and d["gpu_request"] == 0:
        bullets.append(f"💤 **整区闲置**：{d['gpu_total']} 张 {d['gpu_name']} 一张没申请，{full} 整空节点可直接接")
    if nd["stragglers"]:
        s = "、".join(f"`{x['node'][-10:]}` {_pct(x['ta'])}" for x in nd["stragglers"])
        bullets.append(f"🐢 **拖后腿节点**（远低于中位 {_pct(nd['ta_median'])}）：{s}")
    jobs = [j for j in d["jobs"] if j["cards"] > 0]
    if jobs:
        tops = "；".join(f"{j['user']} {j['cards']}卡 {'🔴' if j['mfu'] < low else ''}{_pct(j['mfu'])}" for j in jobs[:4])
        bullets.append(f"🏃 **在跑**：{tops}")
    if bullets:
        els.append(div("\n".join(bullets)))
    return els


def _summary_lines(g):
    """跨区域汇总：markdown 表格。"""
    lines = ["**🌏 跨区域汇总**", "",
             "| 区域 | 卡型 | 总/申请 | 昨日MFU | 在算 | TFLOPS | 整空 |",
             "|---|---|---|---|---|---|---|"]
    for d in g["regions"]:
        y = d["yesterday"]
        mfu = "—" if y["active_avg"] == 0 else _pct(y["mfu"])
        lines.append(f"| {d['region_name']} | {d['gpu_name']} | {d['gpu_total']}/{d['gpu_request']} "
                     f"| {mfu} | {y['active_avg']:.0f} | {y['tflops_avg']:.0f} | {d['nodes']['fully_free_nodes']} |")
    lines.append(f"| **合计** | | **{g['gpu_total']}/{g['gpu_request']}** | | | **{g['tflops']:.0f}** | |")
    acts = []
    for d in g["regions"]:
        if d["gpu_total"] > 0 and d["gpu_request"] == 0:
            acts.append(f"💤 **{d['region_name']} {d['gpu_total']} 张 {d['gpu_name']} 全闲**")
    if g["waste_total"] > 0:
        acts.append(f"💸 全域日浪费 ≈ **¥{g['waste_total']:.0f}**（空占+碎片）")
    acts.append("👇 点下方按钮看各区详情")
    lines += ["", "　".join(acts)]
    return lines


def _region_lines(d):
    low = settings.MFU_LOW_THRESHOLD_PCT
    y, nd = d["yesterday"], d["nodes"]
    mfu = "—" if y["active_avg"] == 0 else f"{_pct(y['mfu'])}（峰 {_pct(y['mfu_peak'])}/谷 {_pct(y['mfu_low'])}）"
    lines = [
        f"**▍{d['region_name']} · {d['gpu_name']}**（峰值 {d['peak_tflops']:.0f} TFLOPS · {nd['node_count']}×{nd['node_cap']}卡）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| GPU 卡 总/申请 | {d['gpu_total']}/{d['gpu_request']}（{_pct(d['alloc_rate'])}）|",
        f"| 显存 已分配/总 | {d['gpu_mem_alloc']:.0f}/{d['gpu_mem_total']:.0f} GiB（{_pct(d['gpu_mem_util'])}）|",
        f"| CPU 核 总/申请 | {d['cpu_total']:.0f}/{d['cpu_request']:.0f} |",
        f"| 内存 总/申请 | {d['mem_total']:.0f}/{d['mem_request']:.0f} GiB |",
        f"| 昨日 MFU | {mfu} |",
        f"| GPU使用率 / SM使用率 | {_pct(y.get('gpu_util', 0))} / {_pct(y.get('sm_util', 0))} |",
        f"| 张量活跃 / 显存带宽 | {_pct(y['tensor_active'])} / {_pct(y['dram_active'])} |",
        f"| 在算 / 算力 | {y['active_avg']:.0f} 卡 / {y['tflops_avg']:.0f} TF |",
        f"| NVLink 收/发 | {y['nvlink_rx']/1024:.1f}/{y['nvlink_tx']/1024:.1f} GiB/s |",
        f"| PCIe 收/发 | {y['pcie_rx']/1024:.1f}/{y['pcie_tx']/1024:.1f} GiB/s |",
        f"| 节点 空闲/整空/碎片 | {nd['free_total']}/{nd['fully_free_nodes']}/{nd['frag_free']} |",
        f"| 空闲分布 | {_fmt_dist(nd)} |",
    ]
    full = nd["fully_free_nodes"]
    bullets = []
    if nd["free_total"] > 0 and full < 2 and nd["frag_free"] > 0:
        bullets.append(f"🚧 **训练排不进**：{nd['free_total']} 空闲卡但仅 {full} 整空节点，"
                       f"{nd['frag_free']} 碎片卡凑不出整机 → ≥{max(2, full+1)} 整机训练无法调度")
    if d["gpu_total"] > 0 and d["gpu_request"] == 0:
        bullets.append(f"💤 **整区闲置**：{d['gpu_total']} 张 {d['gpu_name']} 一张没申请，{full} 整空节点可直接接")
    if nd["stragglers"]:
        s = "、".join(f"`{x['node'][-10:]}` {_pct(x['ta'])}" for x in nd["stragglers"])
        bullets.append(f"🐢 **拖后腿节点**（远低于中位 {_pct(nd['ta_median'])}）：{s}")
    jobs = [j for j in d["jobs"] if j["cards"] > 0]
    if jobs:
        tops = "；".join(f"{j['user']} {j['cards']}卡 {'🔴' if j['mfu']<low else ''}{_pct(j['mfu'])}" for j in jobs[:4])
        bullets.append(f"🏃 **在跑**：{tops}")
    if bullets:
        lines += [""] + [f"- {b}" for b in bullets]
    return lines


def _realtime_lines(g):
    """⚡ 当前瞬时（instant 查询，非昨日均值）：各区此刻 GPU/SM/张量/MFU/在算。

    用 `by (regionId)` 分组一次查全区域（7 条 query），避免逐区 18 条拖慢回调（飞书需<3s）。
    """
    thr = settings.GPU_ACTIVE_THRESHOLD_PCT
    ts = datetime.now(_BJ).strftime("%H:%M:%S")
    K = ("regionId",)
    a_dlc = _grouped(f"count by (regionId)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE > {thr})", K)
    a_dsw = _grouped(f"count by (regionId)(AliyunPaidsw_CARD_GPU_PIP_TENSOR_ACTIVE > {thr})", K)
    tf_dlc = _grouped("sum by (regionId)(AliyunPaidlc_CARD_GPU_TENSORTFLOPS_USED)", K)
    tf_dsw = _grouped("sum by (regionId)(AliyunPaidsw_CARD_GPU_TENSORTFLOPS_USED)", K)
    gpu = _grouped("avg by (regionId)(AliyunPaidlc_CARD_GPU_DUTY_CYCLE)", K)
    sm = _grouped("avg by (regionId)(AliyunPaidlc_CARD_GPU_SM_UTIL)", K)
    ta = _grouped("avg by (regionId)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE)", K)
    lines = [f"**⚡ 实时快照 · {ts} 北京**", "",
             "| 区域 | 卡型 | GPU% | SM% | 张量% | MFU | 在算 | TFLOPS |",
             "|---|---|---|---|---|---|---|---|"]
    for d in g["regions"]:
        rk, peak = (d["region"],), d["peak_tflops"]
        active = a_dlc.get(rk, 0) + a_dsw.get(rk, 0)
        tflops = tf_dlc.get(rk, 0) + tf_dsw.get(rk, 0)
        mfu = "—" if not active else _pct(tflops / (active * peak) * 100 if peak else 0)
        lines.append(f"| {d['region_name']} | {d['gpu_name']} | {_pct(gpu.get(rk, 0))} | {_pct(sm.get(rk, 0))} "
                     f"| {_pct(ta.get(rk, 0))} | {mfu} | {active:.0f} | {tflops:.0f} |")
    lines += ["", "_此为当前瞬时值；昨日全天均值见「📊 全局」_"]
    return lines


def _buttons(g, view):
    btns = [btn("📊 全局", {"action": "mfu_region", "region": ""},
                "primary" if view == "summary" else "default")]
    for d in g["regions"]:
        btns.append(btn(d["region_name"], {"action": "mfu_region", "region": d["region"]},
                        "primary" if view == d["region"] else "default"))
    btns.append(btn("⚡ 实时", {"action": "mfu_region", "region": "__rt__"},
                    "primary" if view == "__rt__" else "default"))
    return buttons(*btns)


def _card(g, view="summary"):
    cur = next((d for d in g["regions"] if d["region"] == view), None)
    elements = [_buttons(g, view), hr()]
    if view == "__rt__":
        elements.append(div("\n".join(_realtime_lines(g))))
        alarm = False
        sub = "⚡ 实时"
    elif cur:
        elements += _region_elements(cur)
        alarm = cur["waste_total"] > 0
        sub = f"{cur['region_name']} · {cur['gpu_name']}"
    else:
        elements.append(div("\n".join(_summary_lines(g))))
        alarm = g["waste_total"] > 0 or any(d["gpu_total"] > 0 and d["gpu_request"] == 0 for d in g["regions"])
        sub = f"{len(g['regions'])}区 {g['gpu_request']}/{g['gpu_total']} 卡在用"
    # 共享卡片需 update_multi 才能被回调原地替换
    return card(f"🧮 多区域算力效率日报 · {g['date']} · {sub}", elements,
                color="red" if alarm else "blue", update_multi=True)


def build_mfu_card(view="summary", refresh=False):
    g = _load_or_gather(refresh=refresh)
    if not g.get("regions"):
        return {"config": {}, "header": {"title": {"tag": "plain_text", "content": "🧮 算力效率日报"}},
                "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": "未发现 GPU 区域/配额。"}}]}
    return _card(g, view)


def cluster_mfu_report():
    """文本版（agent 工具）：汇总 + 各区详情全列出。"""
    if not settings.PROMETHEUS_URL:
        return "❌ PROMETHEUS_URL 未配置，无法读取指标。"
    g = _load_or_gather(refresh=True)
    if not g["regions"]:
        return "未发现任何 GPU 区域/配额（确认 CLUSTER_REGION 与 PAI exporter 接入）。"
    lines = [f"## 多区域集群算力效率日报 · {g['date']}", ""] + _summary_lines(g) + ["", "---", ""]
    for d in g["regions"]:
        lines += _region_lines(d) + [""]
    return "\n".join(lines)


# ── LangChain 工具 ──────────────────────────────────────────────────────────────

class ClusterMFUSchema(BaseModel):
    dummy: str = Field(default="", description="无需参数")


cluster_mfu_report_tool = StructuredTool.from_function(
    func=lambda dummy="": cluster_mfu_report(),
    name="cluster_mfu_report",
    description=(
        "多区域集群算力效率日报：各区域(杭州/北京/新加坡)按卡型取峰值，汇总昨日MFU(均值/峰谷)、"
        "张量活跃、显存带宽、NVLink/PCIe、在算/实测TFLOPS + 当前容量(GPU/CPU/内存/显存) + "
        "节点碎片化/拖后腿节点/调度诊断 + 在跑任务。问 'MFU/算力利用率/各区域使用/碎片化/"
        "为什么训练起不来/谁在空占卡/节点卡分布' 时调用。"
    ),
    args_schema=ClusterMFUSchema,
)
