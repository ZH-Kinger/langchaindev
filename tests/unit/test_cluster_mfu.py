"""tools/aliyun/cluster_mfu.py 多区域交互卡片测试。

mock _query_instant（发现/容量/节点/任务）+ _query_range（昨日均值，含通信/显存带宽）：
  - 区域发现 + 按卡型取峰值/显存
  - 容量(含显存利用率)、昨日(MFU 峰谷/张量活跃/显存带宽/NVLink-PCIe)、节点(碎片+拖后腿)
  - 卡片：汇总视图带区域按钮；点区域切到该区面板
"""
import re
import pytest


def _pt(val, **labels):
    return {"metric": labels, "value": [0, str(val)]}


def _series(val):
    return {"metric": {}, "values": [[0, str(val)], [3600, str(val)]]}


def _reg(q):
    m = re.search(r'regionId="([^"]+)"', q)
    return m.group(1) if m else None


NODES = {
    "cn-hangzhou":    {"gtype": "GU8T", "allocs": [8] * 26 + [5] + [4, 4] + [2, 2] + [0],
                       "node_ta": [15, 15, 15, 15, 2, 2]},          # 2 个拖后腿
    "ap-southeast-1": {"gtype": "L20X", "allocs": [8, 8, 8, 8, 3, 0], "node_ta": [5]},
}
CAP = {
    "cn-hangzhou":    {"GPU_ACCELERATOR_TOTAL": 256, "GPU_ACCELERATOR_REQUEST": 225, "CPU_TOTAL": 5888,
                       "CPU_REQUEST": 5454, "MEMORY_TOTAL": 57600, "MEMORY_REQUEST": 50264,
                       "GPU_MEMORY_ALLOCATED": 21600},
    "ap-southeast-1": {"GPU_ACCELERATOR_TOTAL": 48, "GPU_ACCELERATOR_REQUEST": 35, "CPU_TOTAL": 1104,
                       "CPU_REQUEST": 805, "MEMORY_TOTAL": 10800, "MEMORY_REQUEST": 7000,
                       "GPU_MEMORY_ALLOCATED": 4935},
}
YEST = {
    "cn-hangzhou":    {"alloc": 225, "act_dlc": 190, "act_dsw": 2, "tf_dlc": 11000, "tf_dsw": 4,
                       "ta": 15.0, "dram": 11.0, "nv_rx": 4652, "nv_tx": 3827, "pc_rx": 2696, "pc_tx": 651},
    "ap-southeast-1": {"alloc": 35, "act_dlc": 35, "act_dsw": 0, "tf_dlc": 793, "tf_dsw": 0,
                       "ta": 2.5, "dram": 8.6, "nv_rx": 25332, "nv_tx": 25954, "pc_rx": 4519, "pc_tx": 3428},
}
JOBS = {
    "cn-hangzhou":    [("jobA", "u1", 150, 45.0, 150 * 70.0), ("jobB", "u2", 42, 8.0, 42 * 12.0)],
    "ap-southeast-1": [("jobS", "u3", 35, 5.0, 793.0)],
}
_CAP_KEYS = ["GPU_ACCELERATOR_TOTAL", "GPU_ACCELERATOR_REQUEST", "CPU_TOTAL", "CPU_REQUEST",
             "MEMORY_TOTAL", "MEMORY_REQUEST", "GPU_MEMORY_ALLOCATED"]


@pytest.fixture
def patch_prom(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CLUSTER_REGION", "")
    monkeypatch.setattr(settings, "GPU_PEAK_TFLOPS", 148.0)
    monkeypatch.setattr(settings, "GPU_PEAK_TFLOPS_BY_TYPE", {"GU8T": 148, "L20X": 989})
    monkeypatch.setattr(settings, "GPU_MEM_GB_BY_TYPE", {"GU8T": 96, "L20X": 141})
    monkeypatch.setattr(settings, "GPU_TYPE_DISPLAY", {"GU8T": "H20", "L20X": "H200"})
    monkeypatch.setattr(settings, "GPU_ACTIVE_THRESHOLD_PCT", 1.0)
    monkeypatch.setattr(settings, "MFU_LOW_THRESHOLD_PCT", 30.0)
    monkeypatch.setattr(settings, "GPU_PRICE_PER_HOUR", 10.0)
    monkeypatch.setattr(settings, "PROMETHEUS_URL", "https://prom.example")

    def fake_instant(q):
        reg = _reg(q)
        if "NODE_GPU_ACCELERATOR_TOTAL" in q and reg is None:
            return [_pt(8, regionId=r, nodeGpuType=NODES[r]["gtype"]) for r in NODES]
        if "QUOTA_GPU_ACCELERATOR_TOTAL" in q and reg is None and not q.startswith("sum"):
            return [_pt(CAP[r]["GPU_ACCELERATOR_TOTAL"], regionId=r) for r in CAP]
        if reg is None:
            return []
        if q.startswith("sum(AliyunPaiquota_QUOTA_"):
            for k in _CAP_KEYS:
                if k in q:
                    return [_pt(CAP[reg][k])]
        if "NODE_GPU_ACCELERATOR_TOTAL" in q:
            return [_pt(8, nodeId=f"{reg}-n{i}") for i in range(len(NODES[reg]["allocs"]))]
        if "NODE_GPU_ACCELERATOR_REQUEST" in q:
            return [_pt(a, nodeId=f"{reg}-n{i}") for i, a in enumerate(NODES[reg]["allocs"])]
        if "PIP_TENSOR_ACTIVE" in q and q.startswith("avg by (nodeName)"):
            return [_pt(v, nodeName=f"{reg}-node{i}") for i, v in enumerate(NODES[reg]["node_ta"])]
        js = JOBS.get(reg, [])
        if "PIP_TENSOR_ACTIVE" in q and q.startswith("count by"):
            return [_pt(c, displayName=n, username=u) for n, u, c, ta, tf in js]
        if "PIP_TENSOR_ACTIVE" in q and q.startswith("avg by (displayName"):
            return [_pt(ta, displayName=n, username=u) for n, u, c, ta, tf in js]
        if "TENSORTFLOPS_USED" in q and q.startswith("sum by"):
            return [_pt(tf, displayName=n, username=u) for n, u, c, ta, tf in js]
        return []

    def fake_range(q, start, end, step="3600s"):
        reg = _reg(q); y = YEST.get(reg)
        if not y:
            return []
        if "QUOTA_GPU_ACCELERATOR_REQUEST" in q:                        return [_series(y["alloc"])]
        if "AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE" in q and ">" in q: return [_series(y["act_dlc"])]
        if "AliyunPaidsw_CARD_GPU_PIP_TENSOR_ACTIVE" in q and ">" in q: return [_series(y["act_dsw"])]
        if "AliyunPaidlc_CARD_GPU_TENSORTFLOPS_USED" in q:              return [_series(y["tf_dlc"])]
        if "AliyunPaidsw_CARD_GPU_TENSORTFLOPS_USED" in q:              return [_series(y["tf_dsw"])]
        if "DRAM_ACTIVE_UTIL" in q:                                     return [_series(y["dram"])]
        if "NVLINK_RECEIVE" in q:                                       return [_series(y["nv_rx"])]
        if "NVLINK_TRANSMIT" in q:                                      return [_series(y["nv_tx"])]
        if "PCIE_RECEIVE" in q:                                         return [_series(y["pc_rx"])]
        if "PCIE_TRANSMIT" in q:                                        return [_series(y["pc_tx"])]
        if q.startswith("avg(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE"): return [_series(y["ta"])]
        return []

    from tools.aliyun import prometheus
    monkeypatch.setattr(prometheus, "_query_instant", fake_instant)
    monkeypatch.setattr(prometheus, "_query_range", fake_range)
    from tools.aliyun import cluster_mfu
    return cluster_mfu


def _by(g):
    return {d["region"]: d for d in g["regions"]}


def test_discovery_and_peak(patch_prom):
    by = _by(patch_prom.gather_all())
    assert by["cn-hangzhou"]["gpu_name"] == "H20" and by["cn-hangzhou"]["peak_tflops"] == 148
    assert by["ap-southeast-1"]["gpu_name"] == "H200" and by["ap-southeast-1"]["peak_tflops"] == 989


def test_capacity_gpu_mem_util(patch_prom):
    d = _by(patch_prom.gather_all())["cn-hangzhou"]
    assert d["gpu_mem_alloc"] == 21600
    assert d["gpu_mem_total"] == 256 * 96
    assert d["gpu_mem_util"] == pytest.approx(21600 / (256 * 96) * 100, rel=1e-3)


def test_yesterday_fields(patch_prom):
    y = _by(patch_prom.gather_all())["cn-hangzhou"]["yesterday"]
    assert y["active_avg"] == 192 and y["tflops_avg"] == pytest.approx(11004)
    assert y["mfu"] == pytest.approx(11004 / (192 * 148) * 100, rel=1e-3)
    assert y["mfu_peak"] == pytest.approx(y["mfu"], rel=1e-3)      # 常量序列 → 峰=谷=均
    assert y["dram_active"] == pytest.approx(11.0)
    assert y["nvlink_rx"] == pytest.approx(4652 / 60, rel=1e-3)    # MiB/s


def test_node_fragmentation_and_stragglers(patch_prom):
    nd = _by(patch_prom.gather_all())["cn-hangzhou"]["nodes"]
    assert nd["fully_free_nodes"] == 1 and nd["frag_free"] == 23
    # node_ta 中位 15，两个 2% 节点 < 0.5×中位 → 拖后腿
    assert len(nd["stragglers"]) == 2
    assert all(s["ta"] == 2 for s in nd["stragglers"])


def test_card_summary_has_region_buttons(patch_prom):
    card = patch_prom.build_mfu_card(view="summary", refresh=True)
    actions = card["elements"][0]
    assert actions["tag"] == "action"
    labels = [b["text"]["content"] for b in actions["actions"]]
    assert "📊 全局" in labels and "杭州" in labels and "新加坡" in labels
    vals = [b["value"] for b in actions["actions"]]
    assert {"action": "mfu_region", "region": "cn-hangzhou"} in vals
    assert "跨区域汇总" in card["elements"][-1]["text"]["content"]


def test_card_region_view(patch_prom):
    card = patch_prom.build_mfu_card(view="cn-hangzhou", refresh=True)
    flds = [e for e in card["elements"] if "fields" in e]
    assert flds, "区域面板应有 fields 栅格"
    labels = " ".join(f["text"]["content"] for f in flds[0]["fields"])
    assert "GPU 卡" in labels and "显存" in labels and "MFU" in labels and "NVLink" in labels
    assert "杭州" in card["header"]["title"]["content"]
    btn = next(b for b in card["elements"][0]["actions"] if b["value"]["region"] == "cn-hangzhou")
    assert btn["type"] == "primary"


def test_fmt_dist_readable(patch_prom):
    nd = {"node_cap": 8, "free_dist": {0: 26, "2": 1, 8: 1}}   # 混 int/str key（JSON 缓存后）
    assert patch_prom._fmt_dist(nd) == "26满 · 1剩2 · 1整空"


def test_text_report(patch_prom):
    rep = patch_prom.cluster_mfu_report()
    assert "跨区域汇总" in rep and "杭州" in rep and "新加坡" in rep and "H200" in rep


def test_skips_without_prometheus(patch_prom, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "PROMETHEUS_URL", "")
    assert "PROMETHEUS_URL" in patch_prom.cluster_mfu_report()
