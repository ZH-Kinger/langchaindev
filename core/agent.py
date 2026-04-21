import sys
import json
import logging

# Windows GBK 终端下强制 UTF-8，reconfigure 就地修改不替换对象，不影响 Flask/click
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from functools import lru_cache
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm_factory import get_cloud_llm
from utils.redis_client import get_redis, is_redis_available
from core.prompts import get_agent_prompt
from tools import ALL_TOOLS

# ── Redis 对话记忆 key 前缀 ───────────────────────────────────────────────────
SESSION_KEY_PREFIX = "agent:chat_history:"
MAX_HISTORY = 20   # Redis 里最多保留最近 20 条消息（10 轮对话）


@lru_cache(maxsize=1)
def _build_executor() -> AgentExecutor:
    """首次调用时构建 AgentExecutor（全量工具，非流式），供飞书 Bot 复用"""
    llm = get_cloud_llm()
    prompt = get_agent_prompt()
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=True, handle_parsing_errors=True)


# ── 工具路由（按关键词缩减工具列表，减少 token 消耗）──────────────────────────
_TOOL_GROUPS = {
    "knowledge": {"query_knowledge"},
    "monitor":   {"system_data_manager", "analyze_node_cpu_trend", "query_infrastructure_metrics"},
    "ops":       {"compress_system_alarms", "restart_k8s_service"},
    "notify":    {"push_report_to_feishu"},
    "advisor":   {"advise_gpu_cluster", "query_infrastructure_metrics"},
    "pai_dsw":   {"manage_pai_dsw"},
    "jira":      {"manage_jira"},
    "training":  {"analyze_gpu_training", "query_infrastructure_metrics"},
    "inspect":   {"inspect_dsw_instance", "manage_pai_dsw"},
}

# 启动时校验 _TOOL_GROUPS 中的名字与 ALL_TOOLS 一致，防止静默路由失效
_ALL_TOOL_NAMES = {t.name for t in ALL_TOOLS}
for _g, _ns in _TOOL_GROUPS.items():
    assert _ns <= _ALL_TOOL_NAMES, f"_TOOL_GROUPS[{_g}] 含未知工具名: {_ns - _ALL_TOOL_NAMES}"


def _select_tools(user_input: str) -> list:
    """按关键词路由工具，匹配失败时安全回退到全量工具列表"""
    text = user_input.lower()
    if any(k in text for k in ("文档", "规程", "知识库", "怎么", "如何")):
        names = _TOOL_GROUPS["knowledge"] | _TOOL_GROUPS["monitor"]
    elif any(k in text for k in ("重启", "pod", "k8s", "告警", "降噪", "修复")):
        names = _TOOL_GROUPS["ops"] | _TOOL_GROUPS["monitor"]
    elif any(k in text for k in ("dsw", "实例", "工作站", "启动实例", "停止实例", "删除实例", "pai dsw", "数据科学")):
        names = _TOOL_GROUPS["pai_dsw"]
    elif any(k in text for k in ("jira", "工单", "gpu申请", "申请记录", "工作项")):
        names = _TOOL_GROUPS["jira"] | _TOOL_GROUPS["pai_dsw"]
    elif any(k in text for k in ("dlc", "eas", "产品", "算力", "训练任务", "推理", "开发环境", "分布")):
        names = _TOOL_GROUPS["monitor"]
    elif any(k in text for k in ("训练建议", "算法建议", "训练分析", "利用率低", "怎么优化训练",
                                  "分析我的gpu", "分析实例", "训练瓶颈", "显存优化", "dataloader")):
        names = _TOOL_GROUPS["training"]
    elif any(k in text for k in ("健康", "巡检", "状态怎么样", "在跑吗", "费用多少",
                                  "要不要停", "实例状态", "跑了多久", "inspect")):
        names = _TOOL_GROUPS["inspect"]
    elif any(k in text for k in ("建议", "优化", "效率", "成本", "散热", "调度", "空闲", "瓶颈")):
        names = _TOOL_GROUPS["advisor"]
    elif any(k in text for k in ("cpu", "内存", "memory", "prometheus", "指标", "趋势", "监控")):
        names = _TOOL_GROUPS["monitor"]
    elif any(k in text for k in ("飞书", "通知", "发送", "消息", "推送")):
        names = _TOOL_GROUPS["notify"]
    else:
        return ALL_TOOLS
    matched = [t for t in ALL_TOOLS if t.name in names]
    return matched if matched else ALL_TOOLS


@lru_cache(maxsize=8)
def _get_streaming_executor(tool_names: frozenset) -> AgentExecutor:
    """按工具集缓存流式 Executor，相同工具组合复用同一实例"""
    tools = [t for t in ALL_TOOLS if t.name in tool_names]
    llm = get_cloud_llm(streaming=True)
    agent = create_tool_calling_agent(llm, tools, get_agent_prompt())
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


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
        logging.warning("Redis 加载历史失败: %s", e)
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
            pipe.execute()
    except Exception as e:
        logging.warning("Redis 保存历史失败: %s", e)


def _clear_history(session_id: str) -> None:
    if is_redis_available():
        try:
            get_redis().delete(_session_key(session_id))
        except Exception:
            pass


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
