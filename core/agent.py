import sys
import json
import logging
from utils.logger import get_logger

logger = get_logger(__name__)

# Windows GBK 终端下强制 UTF-8，reconfigure 就地修改不替换对象，不影响 Flask/click
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from functools import lru_cache
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from core.llm_factory import get_cloud_llm
from utils.redis_client import get_redis, is_redis_available
from core.prompts_agent import get_agent_prompt
from tools import ALL_TOOLS, TOOL_GROUPS
from core.llm_factory import clear_llm_cache
from config.settings import settings
import os

# ── Redis 对话记忆 key 前缀 ───────────────────────────────────────────────────
SESSION_KEY_PREFIX = "agent:chat_history:"
MAX_HISTORY = 20   # Redis 里最多保留最近 20 条消息（10 轮对话）
HISTORY_TTL_SECONDS = 7 * 86400   # 会话历史空闲 7 天自动过期，避免 Redis 里无限堆积

# AgentExecutor 兜底：限制工具调用轮数与总执行时长，防止 LLM 反复调工具把用户挂住/烧钱
AGENT_MAX_ITERATIONS = 8
AGENT_MAX_EXECUTION_TIME = 60   # 秒


def _build_executor(tools: list = None) -> AgentExecutor:
    """构建 AgentExecutor（非流式）。tools 留空=全量（向后兼容旧调用与启动预热）；
    飞书 Bot 传入按消息缩小后的工具子集，避免不相关消息拿到全量工具而乱调。

    不再缓存：工具子集按消息变化，缓存无意义；LLM client 仍由 get_cloud_llm 缓存，
    每次仅重建 agent+executor 包装（内存操作，开销很小）。"""
    tools = tools or ALL_TOOLS
    llm = get_cloud_llm()
    prompt = get_agent_prompt()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,
                         max_iterations=AGENT_MAX_ITERATIONS,
                         max_execution_time=AGENT_MAX_EXECUTION_TIME)


# ── 工具路由：三层结构 ──────────────────────────────────────────────────────
# ① 快通道：含明确专有名词时直接关键词路由（零延迟、零成本）
# ② LLM 路由：调 cloud_llm 做语义意图分类（带缓存，约 80% 命中率）
# ③ 兜底：老关键词路由（LLM 失败 / 网络断时仍能用）

# 快通道触发词：含这些专有名词就跳过 LLM，直接走精确关键词路由
_FAST_PATH_TOKENS = (
    "dsw", "pai dsw", "工作站",
    "jira", "工单", "gpu申请",
    "ecs", "云服务器",
    "oss", "bucket", "对象存储",
    "tos", "火山", "volcano",
    "迁移", "传输", "跨云", "transfer", "migration",
    "sls", "logstore", "日志服务",
    "k8s", "pod",
    "grafana", "prometheus",
    "mfu", "日报", "算力效率",
)

# 知识库触发词：和上面快通道冲突时优先让 LLM 决定。
# 例如「看下知识库怎么部署 K8s」含 k8s，但用户意图明显是查文档。
_KNOWLEDGE_OVERRIDE_TOKENS = (
    "知识库", "文档", "操作手册", "查文档", "看文档", "规程",
)


def _names_to_tools(names: set) -> list:
    """把 TOOL_GROUPS 名集合转为 tool 实例列表，空则返回 ALL_TOOLS。"""
    matched = [t for t in ALL_TOOLS if t.name in names]
    return matched if matched else ALL_TOOLS


def _intents_to_tools(intents: list) -> list:
    """把 LLM 路由返回的意图组名列表转为 tool 实例列表。"""
    if not intents:
        return []
    names: set = set()
    for intent in intents:
        if intent in TOOL_GROUPS:
            names |= TOOL_GROUPS[intent]
    return _names_to_tools(names) if names else []


def _select_tools(user_input: str) -> list:
    """三层路由：快通道 → LLM → 老关键词兜底。"""
    text = user_input.lower()

    # ── 知识库覆盖：含「知识库/文档/手册」时让 LLM 决定，不走快通道 ──────────
    # 解决「看下知识库怎么部署 K8s」被 k8s 抢路由的过度触发问题
    knowledge_intent = any(tok in text for tok in _KNOWLEDGE_OVERRIDE_TOKENS)

    # ── ① 快通道：含明确专有名词，直接关键词路由 ────────────────────────────
    if not knowledge_intent and any(tok in text for tok in _FAST_PATH_TOKENS):
        return _legacy_keyword_route(text)

    # ── ② LLM 语义路由（带 lru_cache） ──────────────────────────────────────
    try:
        from core.intent_router import route as llm_route
        intents = llm_route(user_input)
        if intents:
            tools = _intents_to_tools(intents)
            if tools:
                return tools
    except Exception as e:
        # LLM 路由模块本身异常（不应该发生），降级
        from utils.logger import get_logger
        get_logger(__name__).warning("[Router] LLM 路由模块异常，降级关键词：%s", e)

    # ── ③ 兜底：老关键词路由 ────────────────────────────────────────────────
    return _legacy_keyword_route(text)



_TRANSFER_ROUTE_TOKENS = (
    "迁移", "传输", "同步", "复制", "拷贝", "搬迁", "跨云",
    "transfer", "migration",
)
_TRANSFER_STORAGE_TOKENS = (
    "tos", "oss", "cpfs", "vepfs", "对象存储", "bucket", "存储桶", "火山",
)
_TRANSFER_URI_TOKENS = ("tos://", "oss://", "cpfs://", "vepfs://")
_TRANSFER_DIRECTION_TOKENS = (
    "tos到oss", "oss到tos", "tos 到 oss", "oss 到 tos",
    "tos->oss", "oss->tos", "tos -> oss", "oss -> tos",
)


def _looks_like_transfer(text: str) -> bool:
    if any(k in text for k in _TRANSFER_DIRECTION_TOKENS):
        return True
    uri_hits = sum(1 for k in _TRANSFER_URI_TOKENS if k in text)
    if uri_hits >= 2:
        return True
    has_route_action = any(k in text for k in _TRANSFER_ROUTE_TOKENS)
    has_storage_ref = any(k in text for k in _TRANSFER_STORAGE_TOKENS)
    return has_route_action and has_storage_ref


def _keyword_route_names(text: str) -> "set | None":
    """老关键词路由的名集合版：命中返回 TOOL_GROUPS 名集合，**未命中返回 None**。

    text 必须已经是 lower() 后的。分支顺序：先强意图（带专有名词），后弱意图（自然语言泛词）。
    knowledge 分支放最后做兜底，避免「怎么/如何」抢占其他意图。
    """
    if any(k in text for k in ("重启", "pod", "k8s", "告警", "降噪", "修复")):
        return TOOL_GROUPS["ops"] | TOOL_GROUPS["monitor"]
    if any(k in text for k in ("dsw", "实例", "工作站", "启动实例", "停止实例", "删除实例", "pai dsw", "数据科学")):
        return TOOL_GROUPS["pai_dsw"]
    if any(k in text for k in ("jira", "工单", "gpu申请", "申请记录", "工作项")):
        return TOOL_GROUPS["jira"] | TOOL_GROUPS["pai_dsw"]
    if any(k in text for k in ("ecs", "云服务器", "重启服务器", "服务器列表", "服务器实例")):
        return TOOL_GROUPS["ecs"]
    if _looks_like_transfer(text):
        return TOOL_GROUPS["transfer"]
    if any(k in text for k in ("tos", "火山", "volcano")):
        return TOOL_GROUPS["tos"]
    if any(k in text for k in ("oss", "对象存储", "bucket", "存储桶", "存储空间",
                               "目录大小", "目录结构", "目录树")):
        return TOOL_GROUPS["oss"]
    if any(k in text for k in ("sls", "日志服务", "查日志", "log service", "logstore", "日志查询")):
        return TOOL_GROUPS["sls"]
    # 日报/MFU 必须先于「算力→monitor」「效率→advisor」两个泛词分支，
    # 否则「算力日报」「效率日报」会被抢走，永远到不了 cluster_mfu_report
    if any(k in text for k in ("mfu", "日报", "算力效率", "算力利用率", "效率报告")):
        return TOOL_GROUPS["cluster"] | TOOL_GROUPS["advisor"] | TOOL_GROUPS["notify"]
    if any(k in text for k in ("dlc", "eas", "产品", "算力", "训练任务", "推理", "开发环境", "分布")):
        return TOOL_GROUPS["monitor"]
    if any(k in text for k in ("集群状态", "所有实例", "全局监控", "哪些在空转",
                               "总费用", "集群监控", "cluster", "全部实例",
                               "集群优化", "集群建议", "集群效率", "集群成本",
                               "集群散热", "集群调度", "集群")):
        return TOOL_GROUPS["cluster"] | TOOL_GROUPS["advisor"]
    if any(k in text for k in ("训练建议", "算法建议", "训练分析", "利用率低", "怎么优化训练",
                               "分析我的gpu", "分析实例", "训练瓶颈", "显存优化", "dataloader",
                               "tensor core", "sm利用率", "训练慢", "为什么慢")):
        return TOOL_GROUPS["training"]
    if any(k in text for k in ("健康", "巡检", "状态怎么样", "在跑吗", "费用多少",
                               "要不要停", "实例状态", "跑了多久", "inspect")):
        return TOOL_GROUPS["inspect"]
    if any(k in text for k in ("建议", "优化", "效率", "成本", "散热", "调度", "空闲", "瓶颈")):
        return TOOL_GROUPS["advisor"]
    if any(k in text for k in ("cpu", "内存", "memory", "prometheus", "指标", "趋势", "监控")):
        return TOOL_GROUPS["monitor"]
    if any(k in text for k in ("飞书", "通知", "发送", "消息", "推送")):
        return TOOL_GROUPS["notify"]
    if any(k in text for k in ("文档", "规程", "知识库", "操作手册", "查文档", "看文档")):
        return TOOL_GROUPS["knowledge"] | TOOL_GROUPS["monitor"]
    return None


def _legacy_keyword_route(text: str) -> list:
    """老关键词路由（薄封装，行为不变）：命中→对应工具子集；未命中→全量兜底。"""
    names = _keyword_route_names(text)
    return _names_to_tools(names) if names else ALL_TOOLS


def select_tools_scoped(user_input: str) -> "tuple[list | None, list[str]]":
    """按当前消息选工具，能区分「未知」。返回 (tools, intents)；tools=None 表示未知意图。

    供飞书 Bot 用（interactive 路径仍用 _select_tools 的全量兜底语义，本函数不改它）。
    逻辑镜像 _select_tools 三级路由，但把「兜底全量」换成「明确的未知信号」。
    """
    text = user_input.lower()
    knowledge_intent = any(tok in text for tok in _KNOWLEDGE_OVERRIDE_TOKENS)

    # ① 快通道（非知识库覆盖 且 含专有名词）
    if not knowledge_intent and any(tok in text for tok in _FAST_PATH_TOKENS):
        names = _keyword_route_names(text)
        if names:
            return _names_to_tools(names), []

    # ② LLM 语义路由（带 lru_cache）
    try:
        from core.intent_router import route as llm_route
        intents = llm_route(user_input)
        if intents:
            tools = _intents_to_tools(intents)
            if tools:
                return tools, intents
    except Exception as e:
        from utils.logger import get_logger
        get_logger(__name__).warning("[Router] LLM 路由模块异常，降级关键词：%s", e)

    # ③ 关键词兜底
    names = _keyword_route_names(text)
    if names:
        return _names_to_tools(names), []

    # 全落空 → 未知
    return None, []


@lru_cache(maxsize=8)
def _get_streaming_executor(tool_names: frozenset) -> AgentExecutor:
    """按工具集缓存流式 Executor，相同工具组合复用同一实例"""
    tools = [t for t in ALL_TOOLS if t.name in tool_names]
    llm = get_cloud_llm(streaming=True)
    agent = create_tool_calling_agent(llm, tools, get_agent_prompt())
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,
                         max_iterations=AGENT_MAX_ITERATIONS,
                         max_execution_time=AGENT_MAX_EXECUTION_TIME)


def _run_with_stream(user_input: str, chat_history: list) -> str:
    """CLI 专用：动态工具选择 + 流式输出，实时展示推理过程与最终回答"""
    tools = _select_tools(user_input)
    executor = _get_streaming_executor(frozenset(t.name for t in tools))

    ai_output = ""
    for chunk in executor.stream({"input": user_input, "chat_history": chat_history}):
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"\n  [→ 调用: {action.tool}]", flush=True)
        elif "steps" in chunk:
            obs = str(chunk["steps"][0].observation)
            preview = obs[:120] + ("..." if len(obs) > 120 else "")
            print(f"  [← {preview}]", flush=True)
        elif "output" in chunk:
            ai_output = chunk["output"]
            print(f"\n{ai_output}", flush=True)
    return ai_output


def _session_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


def _load_history(session_id: str) -> list:
    """从 Redis 加载指定会话的历史消息，降级为空列表"""
    if not is_redis_available():
        return []
    try:
        r = get_redis()
        items = r.lrange(_session_key(session_id), 0, -1)
        history = []
        for item in items:
            data = json.loads(item)
            cls = HumanMessage if data["role"] == "human" else AIMessage
            history.append(cls(content=data["content"]))
        return history
    except Exception as e:
        logger.warning("Redis 加载历史失败: %s", e)
        return []


def _save_turn(session_id: str, human: str, ai: str) -> None:
    """追加一轮对话到 Redis，超出 MAX_HISTORY 时截断旧消息"""
    if not is_redis_available():
        return
    try:
        r = get_redis()
        key = _session_key(session_id)
        with r.pipeline() as pipe:
            pipe.rpush(key,
                       json.dumps({"role": "human", "content": human}),
                       json.dumps({"role": "ai",    "content": ai}))
            pipe.ltrim(key, -MAX_HISTORY, -1)
            pipe.expire(key, HISTORY_TTL_SECONDS)   # 每轮续期：空闲 7 天后自动清理，不无限堆积
            pipe.execute()
    except Exception as e:
        logger.warning("Redis 保存历史失败: %s", e)


def _clear_history(session_id: str) -> None:
    if is_redis_available():
        try:
            get_redis().delete(_session_key(session_id))
        except Exception:
            pass


def _switch_model(model_name: str) -> str:
    """切换模型并清除缓存"""
    # 更新环境变量和设置
    os.environ["MODEL_NAME"] = model_name
    settings.MODEL_NAME = model_name
    
    # 清除 LLM 缓存，使新配置生效
    clear_llm_cache()
    
    # 清除 agent executor 缓存（_build_executor 已不缓存，无需清；流式执行器仍缓存）
    _get_streaming_executor.cache_clear()
    
    return f"模型已切换为: {model_name}"


def run(session_id: str = "cli-default"):
    redis_ok = is_redis_available()
    storage = "Redis（持久化）" if redis_ok else "内存（本次有效）"
    print(f"AIOps Agent 已启动 | 会话：{session_id} | 对话记忆：{storage}")
    print("输入 'exit' 退出，'clear' 清空记忆，'history' 查看历史记录数。")

    # 启动时从 Redis 恢复历史
    chat_history = _load_history(session_id)
    if chat_history:
        print(f"[已恢复 {len(chat_history)//2} 轮历史对话]")

    while True:
        try:
            user_input = input("\n运维指令 >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
            if user_input.lower() == "clear":
                chat_history = []
                _clear_history(session_id)
                print("记忆已清空（Redis + 本地）。")
                continue
            if user_input.lower() == "history":
                print(f"当前记忆：{len(chat_history)//2} 轮对话")
                continue

            ai_output = _run_with_stream(user_input, chat_history)

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_output))
            _save_turn(session_id, user_input, ai_output)

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n已退出。")
            break
        except UnicodeEncodeError as e:
            sys.stderr.write(f"编码错误（已跳过）：{e}\n")
        except Exception as e:
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    run()
