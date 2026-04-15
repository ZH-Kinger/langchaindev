"""
AIOps 测试数据生成器
运行方式：python data/generate_data.py
生成文件：
  - k8s_events_200.csv       K8s 事件日志（OOM / CrashLoop 等）
  - network_metrics_200.csv  网络指标（延迟 / 丢包 / 带宽）
  - memory_metrics_200.csv   内存指标（内存使用率 / Swap）
  - alarms_sample.json       JSON 格式告警（与 raw_alarms 同 schema）
  - metrics_sample.xlsx      Excel 格式 CPU 指标
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)   # 固定随机种子，每次生成结果一致


def _timestamps(n: int, interval_sec: int = 10) -> list:
    start = datetime(2026, 3, 31, 9, 0, 0)
    return [start + timedelta(seconds=i * interval_sec) for i in range(n)]


# ── 1. K8s 事件日志 ────────────────────────────────────────────────────────────
def gen_k8s_events(n: int = 200) -> str:
    namespaces = ["default", "kube-system", "monitoring", "prod"]
    pod_bases  = ["order-svc", "payment-svc", "user-svc", "nginx-ingress", "redis-master"]
    events = [
        ("Warning", "OOMKilled"),
        ("Warning", "CrashLoopBackOff"),
        ("Warning", "ImagePullBackOff"),
        ("Warning", "Unhealthy"),
        ("Warning", "NodeNotReady"),
        ("Normal",  "Pulled"),
        ("Normal",  "Started"),
    ]
    rows = []
    for ts in _timestamps(n):
        ns      = rng.choice(namespaces)
        pod     = rng.choice(pod_bases) + "-" + format(rng.integers(10000, 99999), "05d")
        ev_type, msg = events[rng.integers(0, len(events))]
        count   = int(rng.integers(1, 30))
        rows.append([ts, ns, pod, ev_type, msg, count])

    df = pd.DataFrame(rows, columns=["timestamp", "namespace", "pod_name", "event_type", "message", "count"])
    path = os.path.join(BASE_DIR, "k8s_events_200.csv")
    df.to_csv(path, index=False)
    print(f"  k8s_events_200.csv       ({len(df)} 行)")
    return path


# ── 2. 网络指标 ────────────────────────────────────────────────────────────────
def gen_network_metrics(n: int = 200) -> str:
    nodes = ["edge-node-01", "edge-node-02", "core-switch-01"]
    rows  = []
    for ts in _timestamps(n, interval_sec=60):
        node    = rng.choice(nodes)
        latency = round(float(max(0.5, rng.normal(20, 15))), 2)
        loss    = round(float(max(0.0, rng.normal(0.5, 2))), 3)
        bw      = round(float(max(10,  rng.normal(800, 200))), 1)
        rows.append([ts, node, latency, loss, bw])

    df = pd.DataFrame(rows, columns=["timestamp", "node", "latency_ms", "packet_loss_pct", "bandwidth_mbps"])
    path = os.path.join(BASE_DIR, "network_metrics_200.csv")
    df.to_csv(path, index=False)
    print(f"  network_metrics_200.csv  ({len(df)} 行)")
    return path


# ── 3. 内存指标 ────────────────────────────────────────────────────────────────
def gen_memory_metrics(n: int = 200) -> str:
    nodes = ["web-server-01", "db-server-01", "cache-server-01"]
    rows  = []
    for ts in _timestamps(n, interval_sec=60):
        node = rng.choice(nodes)
        mem  = round(float(np.clip(rng.normal(65, 20), 0, 100)), 2)
        swap = round(float(np.clip(rng.normal(10, 15), 0, 100)), 2)
        rows.append([ts, node, mem, swap])

    df = pd.DataFrame(rows, columns=["timestamp", "node", "mem_util", "swap_util"])
    path = os.path.join(BASE_DIR, "memory_metrics_200.csv")
    df.to_csv(path, index=False)
    print(f"  memory_metrics_200.csv   ({len(df)} 行)")
    return path


# ── 4. JSON 格式告警 ───────────────────────────────────────────────────────────
def gen_alarms_json(n: int = 50) -> str:
    nodes    = ["node-10", "node-11", "node-12"]
    messages = ["CPU spike > 90%", "Memory > 95%", "Disk full /data", "Network timeout"]
    records  = []
    for ts in _timestamps(n, interval_sec=60):
        records.append({
            "timestamp": str(ts),
            "node":      str(rng.choice(nodes)),
            "message":   str(rng.choice(messages)),
            "severity":  str(rng.choice(["Critical", "Warning", "Info"])),
        })

    path = os.path.join(BASE_DIR, "alarms_sample.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  alarms_sample.json       ({len(records)} 条)")
    return path


# ── 5. Excel 格式 CPU 指标 ─────────────────────────────────────────────────────
def gen_metrics_excel(n: int = 100) -> str:
    nodes = ["web-server-01", "web-server-02"]
    rows  = []
    for ts in _timestamps(n, interval_sec=60):
        node = rng.choice(nodes)
        cpu  = round(float(np.clip(rng.normal(40, 25), 0, 100)), 2)
        rows.append([ts, node, cpu])

    df   = pd.DataFrame(rows, columns=["timestamp", "node", "cpu_util"])
    path = os.path.join(BASE_DIR, "metrics_sample.xlsx")
    df.to_excel(path, index=False)
    print(f"  metrics_sample.xlsx      ({len(df)} 行)")
    return path


if __name__ == "__main__":
    print("正在生成测试数据集...\n")
    gen_k8s_events()
    gen_network_metrics()
    gen_memory_metrics()
    gen_alarms_json()
    gen_metrics_excel()
    print("\n全部生成完毕！")
