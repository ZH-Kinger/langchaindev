import os
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from tools.base_tool import BaseOpsTool, logger
from utils.redis_client import cache_analysis, load_seen_alarms, save_seen_alarms, count_seen_alarms

base = BaseOpsTool()


class DataAnalysisSchema(BaseModel):
    file_name: str = Field(
        description="data 目录下的文件名。支持 .csv / .json / .xlsx 格式。"
                    "可分析：告警日志、K8s 事件、网络指标、内存指标、CPU 指标等。"
    )


def _load_file(file_path: str) -> pd.DataFrame:
    """根据扩展名自动选择读取方式"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        return pd.read_json(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)


def _analyze_alarms(df: pd.DataFrame, file_name: str) -> str:
    """场景 A：告警风暴降噪（含 timestamp + node + message）+ Redis 持久化去重"""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    original_count = len(df)

    # C. 持久化去重：从 Redis 加载已处理的告警 key
    seen = load_seen_alarms(file_name)
    current_keys = set(df["node"] + ":" + df["message"])
    new_keys = current_keys - seen

    # 本次去重（pandas 去重 + 过滤掉 Redis 中已见过的）
    cleaned_df = df.drop_duplicates(subset=["node", "message"], keep="last")
    cleaned_df = cleaned_df[
        (cleaned_df["node"] + ":" + cleaned_df["message"]).isin(new_keys)
    ] if seen else cleaned_df

    reduced_count = len(cleaned_df)
    reduction_rate = (1 - reduced_count / original_count) * 100

    # 将本次新告警写入 Redis
    save_seen_alarms(file_name, new_keys)
    redis_seen_total = count_seen_alarms(file_name)

    top_alarm = df["message"].value_counts().idxmax()
    top_node  = df[df["message"] == top_alarm]["node"].iloc[0]

    base.log_operation("AlarmReduction", {"file": file_name}, "Success")
    return (
        f"✨ **告警降噪完成** ({file_name})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 原始 {original_count} 条 → 本次新增 {reduced_count} 条（压缩率 {reduction_rate:.1f}%）\n"
        f"🗄️ Redis 累计已去重告警类型：{redis_seen_total} 种\n"
        f"🔍 高频告警：`{top_alarm}` 集中于节点 `{top_node}`\n"
        f"💡 建议：该节点疑似告警风暴，请优先检查磁盘 IO 或日志滚动配置。"
    )


def _analyze_k8s_events(df: pd.DataFrame, file_name: str) -> str:
    """场景 B：K8s 事件分析（含 namespace + pod_name + event_type）"""
    total = len(df)
    warnings = df[df["event_type"] == "Warning"]
    warning_count = len(warnings)

    top_msg    = warnings["message"].value_counts().idxmax() if warning_count else "无"
    top_ns     = warnings["namespace"].value_counts().idxmax() if warning_count else "无"
    crash_pods = warnings[warnings["message"] == "CrashLoopBackOff"]["pod_name"].unique().tolist()

    status = "🚨 存在严重事件" if warning_count > total * 0.3 else "✅ 整体健康"

    base.log_operation("K8sEventAnalysis", {"file": file_name}, "Success")
    return (
        f"☸️ **K8s 事件分析完成** ({file_name})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 总事件 {total} 条，Warning {warning_count} 条（占比 {warning_count/total*100:.1f}%）\n"
        f"🔥 高频异常：`{top_msg}`，集中在命名空间 `{top_ns}`\n"
        f"💀 CrashLoop Pod：{crash_pods[:5] if crash_pods else '无'}\n"
        f"⚖️ 判定：{status}\n"
        f"💡 建议：{'立即检查 ' + top_ns + ' 命名空间的资源限制配置。' if warning_count else '继续观察。'}"
    )


def _analyze_network_metrics(df: pd.DataFrame, file_name: str) -> str:
    """场景 C：网络指标分析（含 latency_ms / packet_loss_pct / bandwidth_mbps）"""
    stats = {}
    if "latency_ms" in df.columns:
        stats["avg_latency"] = df["latency_ms"].mean()
        stats["max_latency"] = df["latency_ms"].max()
    if "packet_loss_pct" in df.columns:
        stats["avg_loss"] = df["packet_loss_pct"].mean()
        stats["max_loss"] = df["packet_loss_pct"].max()
    if "bandwidth_mbps" in df.columns:
        stats["avg_bw"]  = df["bandwidth_mbps"].mean()
        stats["min_bw"]  = df["bandwidth_mbps"].min()

    latency_bad = stats.get("max_latency", 0) > 100
    loss_bad    = stats.get("max_loss", 0) > 5

    base.log_operation("NetworkAnalysis", {"file": file_name}, "Success")
    return (
        f"🌐 **网络指标分析完成** ({file_name})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 延迟：均值 {stats.get('avg_latency',0):.1f}ms / 峰值 {stats.get('max_latency',0):.1f}ms"
        f"  {'🚨 延迟过高' if latency_bad else '✅ 正常'}\n"
        f"📉 丢包：均值 {stats.get('avg_loss',0):.2f}% / 峰值 {stats.get('max_loss',0):.2f}%"
        f"  {'🚨 丢包严重' if loss_bad else '✅ 可控'}\n"
        f"📶 带宽：均值 {stats.get('avg_bw',0):.0f}Mbps / 最低 {stats.get('min_bw',0):.0f}Mbps\n"
        f"💡 建议：{'检查网络设备或 QoS 配置。' if latency_bad or loss_bad else '网络状态良好，继续监控。'}"
    )


def _analyze_numeric_metrics(df: pd.DataFrame, file_name: str) -> str:
    """场景 D：通用数值指标统计（兜底逻辑）"""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats    = df[numeric_cols].describe()
    max_val  = df[numeric_cols[0]].max()
    col_name = numeric_cols[0]
    status   = "🚨 严重异常" if max_val > 90 else ("⚠️ 需要关注" if max_val > 70 else "✅ 状态平稳")

    base.log_operation("MetricAnalysis", {"file": file_name}, "Success")
    return (
        f"📊 **指标统计完成** ({file_name})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📉 指标概览：\n{stats.to_string()}\n\n"
        f"⚖️ 判定（以 `{col_name}` 为基准）：{status}（峰值 {max_val:.2f}）\n"
        f"💡 建议：关注峰值时段的任务调度。"
    )


@cache_analysis
def smart_data_analysis(file_name: str) -> str:
    """
    智能路由分析：自动识别文件格式（csv/json/xlsx）和数据 Schema，
    分别执行告警降噪、K8s 事件分析、网络指标分析或通用指标统计。
    """
    file_path = base.get_data_path(file_name)
    if not os.path.exists(file_path):
        return f"❌ 找不到文件：{file_path}"

    try:
        df   = _load_file(file_path)
        cols = df.columns.tolist()

        if all(c in cols for c in ["timestamp", "node", "message"]):
            return _analyze_alarms(df, file_name)

        if all(c in cols for c in ["namespace", "pod_name", "event_type"]):
            return _analyze_k8s_events(df, file_name)

        if any(c in cols for c in ["latency_ms", "packet_loss_pct", "bandwidth_mbps"]):
            return _analyze_network_metrics(df, file_name)

        if df.select_dtypes(include=["number"]).columns.tolist():
            return _analyze_numeric_metrics(df, file_name)

        return f"❓ 未知数据格式，包含列：{cols}"

    except Exception as e:
        logger.error(f"分析异常: {e}")
        return f"❌ 分析过程中发生错误：{e}"


alarm_reduction_tool = StructuredTool.from_function(
    func=smart_data_analysis,
    name="compress_system_alarms",
    description=(
        "对 data 目录下的文件进行深度分析。"
        "支持格式：.csv / .json / .xlsx。"
        "支持类型：告警日志降噪、K8s 事件诊断、网络指标分析、CPU/内存等通用指标统计。"
    ),
    args_schema=DataAnalysisSchema,
)
