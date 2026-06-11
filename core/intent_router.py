"""
LLM 意图路由：用 cloud_llm 在低温度下识别用户意图，输出 TOOL_GROUPS 名称列表。

调用层（core/agent._select_tools）使用顺序：
  1. 快通道（含明确专有名词 dsw/jira/ecs/oss/sls 等）→ 跳过本模块
  2. 本模块 `route(user_input)` → 返回意图名列表
  3. 兜底（老关键词路由）

关键设计：
  - lru_cache 按文本缓存路由结果（避免同一句话反复调 LLM）
  - 强制 JSON 输出 + 解析容错（兼容 markdown 包裹的代码块）
  - 未知意图名自动过滤
  - LLM 调用失败、超时、空输出统一返回空列表，让上层降级
"""
import json
import re
from functools import lru_cache
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── 意图描述（喂给 LLM，每行一条意图）─────────────────────────────────────────
# key 必须存在于 tools/__init__.py 的 TOOL_GROUPS 字典中
INTENT_DESCRIPTIONS: dict[str, str] = {
    "knowledge": "查询知识库 / 文档 / 规程 / 操作手册",
    "ops":       "K8s / Pod / 容器运维：重启、降噪、修复故障",
    "pai_dsw":   "PAI DSW 数据科学工作站实例的查询、启停、删除",
    "jira":      "Jira 工单、GPU 申请记录、工作项查询",
    "ecs":       "ECS 云服务器：列出、详情、启停、重启、释放",
    "oss":       "OSS 对象存储：列 bucket / 对象、上传、删除、统计用量、各子目录大小、目录树",
    "tos":       "火山引擎 TOS 对象存储：各子目录大小、容量盘点、列对象",
    "sls":       "SLS 日志服务：列项目、查日志、创建 logstore",
    "monitor":   "查 Prometheus 指标 / CPU / 内存 / 趋势 / 系统资源",
    "advisor":   "GPU 集群优化建议：成本、效率、散热、调度、瓶颈分析",
    "cluster":   "集群整体状态、所有实例、空转检测、健康巡检、算力效率日报（MFU / GPU日报 / 各区域利用率）",
    "training":  "GPU 训练深度分析：利用率、Tensor Core、显存、DataLoader 瓶颈",
    "inspect":   "单个 DSW 实例的健康/费用/超时巡检",
    "workflow":  "Jira / GitHub 工作流：commits、PR、Sprint、昨日活动",
    "notify":    "主动推送飞书消息 / 报告",
}


def _build_prompt(user_input: str) -> str:
    """构造一次性 prompt：列出所有意图，让 LLM 选 1-2 个。"""
    intent_lines = "\n".join(f"  - {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())
    return (
        "你是一个意图路由器。给定用户的运维查询，从下列意图中选出 1-2 个最相关的，"
        "只返回 JSON 数组，不要其他文字、不要 markdown 代码块包裹。\n\n"
        f"可选意图：\n{intent_lines}\n\n"
        f"用户输入：{user_input}\n\n"
        '示例输出（必须是合法 JSON 数组）：["advisor"] 或 ["cluster", "monitor"]'
    )


# ── JSON 解析容错 ────────────────────────────────────────────────────────────

_JSON_ARRAY_PATTERN = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def _parse_intents(llm_output: str) -> list[str]:
    """从 LLM 输出中提取意图名列表。容忍 markdown 包裹、附加文字。"""
    if not llm_output:
        return []
    text = llm_output.strip()

    # 去掉 markdown 代码块包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 第一轮：直接 JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if isinstance(x, str)]
    except Exception:
        pass

    # 第二轮：抓第一个 [...] 片段再 parse
    m = _JSON_ARRAY_PATTERN.search(text)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return [str(x).strip().lower() for x in result if isinstance(x, str)]
        except Exception:
            pass

    return []


def _filter_known_intents(intents: list[str]) -> list[str]:
    """过滤掉不在 INTENT_DESCRIPTIONS 中的项；保留顺序、去重。"""
    seen = set()
    valid = []
    for name in intents:
        if name in INTENT_DESCRIPTIONS and name not in seen:
            seen.add(name)
            valid.append(name)
    return valid[:2]  # 最多 2 个


# ── LLM 调用（带缓存）────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _cached_route(user_input: str) -> tuple[str, ...]:
    """同一文本的 LLM 路由结果缓存。tuple 是为了 hashable。"""
    return tuple(_invoke_llm_for_route(user_input))


def _invoke_llm_for_route(user_input: str) -> list[str]:
    """实际调 cloud_llm 做意图分类。失败返回空列表。"""
    try:
        from core.llm_factory import get_cloud_llm
        llm = get_cloud_llm(temperature=0)
        prompt = _build_prompt(user_input)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        intents = _parse_intents(content)
        valid = _filter_known_intents(intents)
        if not valid:
            logger.warning("[IntentRouter] LLM 输出无可用意图 raw=%.200s", content)
        else:
            logger.info("[IntentRouter] 路由结果 input=%.50s → %s", user_input, valid)
        return valid
    except Exception as e:
        logger.warning("[IntentRouter] LLM 调用失败，降级到关键词路由：%s", e)
        return []


# ── 对外主接口 ──────────────────────────────────────────────────────────────

def route(user_input: str) -> list[str]:
    """
    返回意图名列表（最多 2 个，已校验在 INTENT_DESCRIPTIONS 中）。
    空列表表示路由失败，调用方应降级到老关键词路由或全量工具。
    """
    if not user_input or not user_input.strip():
        return []
    return list(_cached_route(user_input.strip()))


def clear_cache() -> None:
    """供测试或运行时手动清除路由缓存。"""
    _cached_route.cache_clear()
