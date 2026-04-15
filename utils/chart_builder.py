"""
指标趋势图生成工具。
接收 fetch_raw_series() 返回的时序数据，生成 PNG 并返回 bytes。
"""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 字体 / 样式配置 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus":   False,
    "figure.facecolor":     "#f0f2f5",
    "axes.facecolor":       "#ffffff",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#e0e0e0",
    "grid.linewidth":       0.5,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
})

# ── 每类指标的显示配置 ────────────────────────────────────────────────────────────
_META = {
    "cpu":     {"label": "CPU 使用率",  "unit": "%",    "line": "#4e79a7", "warn": 80,  "crit": 90},
    "memory":  {"label": "内存使用率",  "unit": "%",    "line": "#f28e2b", "warn": 85,  "crit": 95},
    "gpu":     {"label": "GPU SM 利用", "unit": "%",    "line": "#76b7b2", "warn": 95,  "crit": 99},
    "disk":    {"label": "磁盘 I/O",   "unit": "MB/s", "line": "#e15759", "warn": 200, "crit": 500},
    "network": {"label": "网络流量",    "unit": "MB/s", "line": "#59a14f", "warn": 100, "crit": 500},
}

_ORDER = ("cpu", "memory", "gpu", "disk", "network")


def build_metrics_chart(series_data: dict) -> bytes:
    """
    生成 5 格趋势图（1 行 × 5 列），返回 PNG bytes。
    series_data 格式: {key: {"timestamps": [...], "values": [...], "unit": str}}
    """
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    fig.subplots_adjust(wspace=0.42, left=0.05, right=0.98, top=0.80, bottom=0.22)

    for ax, key in zip(axes, _ORDER):
        meta  = _META[key]
        data  = series_data.get(key, {})
        ts    = data.get("timestamps", [])
        vs    = data.get("values",     [])
        unit  = meta["unit"]

        ax.set_title(meta["label"], fontsize=9, fontweight="bold", pad=5, color="#333333")
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        if ts and vs:
            dt_ts   = [datetime.fromtimestamp(t) for t in ts]
            max_v   = max(vs)
            avg_v   = sum(vs) / len(vs)

            # 根据峰值选线色
            line_color = (
                "#e74c3c" if max_v >= meta["crit"] else
                "#f39c12" if max_v >= meta["warn"] else
                meta["line"]
            )

            ax.plot(dt_ts, vs, color=line_color, linewidth=1.6, zorder=3)
            ax.fill_between(dt_ts, vs, alpha=0.18, color=line_color, zorder=2)

            # 阈值参考线
            y_max = max(max_v * 1.25, meta["warn"] * 0.2)
            ax.set_ylim(bottom=0, top=y_max if y_max > 0 else 1)
            ax.axhline(meta["warn"], color="#f39c12", linewidth=0.8,
                       linestyle="--", alpha=0.75, label=f"警告 {meta['warn']}{unit}")
            ax.axhline(meta["crit"], color="#e74c3c", linewidth=0.8,
                       linestyle=":",  alpha=0.75, label=f"严重 {meta['crit']}{unit}")

            # y 轴标签
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0f}{unit}")
            )
            # x 轴时间格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

            # 右上角标注均值
            ax.text(0.98, 0.97, f"均 {avg_v:.2f}{unit}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=line_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "暂无数据", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#aaaaaa")
            ax.set_xticks([])
            ax.set_yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
