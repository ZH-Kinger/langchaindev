import sys

# Windows GBK 终端下强制 UTF-8，reconfigure 就地修改不替换对象，不影响 Flask/click
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm_factory import get_cloud_llm
from utils.redis_client import get_redis, is_redis_available
from core.prompts import get_agent_prompt
from tools import ALL_TOOLS

# ── Redis 对话记忆 key 前缀 ───────────────────────────────────────────────────
SESSION_KEY = "agent:chat_history"
MAX_HISTORY = 20   # Redis 里最多保留最近 20 条消息（10 轮对话）

llm = get_cloud_llm()
tools = ALL_TOOLS
prompt = get_agent_prompt()

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


def _load_history() -> list:
    """从 Redis 加载历史消息，降级为空列表"""
    if not is_redis_available():
        return []
    try:
        r = get_redis()
        items = r.lrange(SESSION_KEY, 0, -1)
        history = []
        for item in items:
            role, content = item.split("||", 1)
            if role == "human":
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))
        return history
    except Exception:
        return []


def _save_turn(human: str, ai: str) -> None:
    """追加一轮对话到 Redis，超出 MAX_HISTORY 时截断旧消息"""
    if not is_redis_available():
        return
    try:
        r = get_redis()
        r.rpush(SESSION_KEY, f"human||{human}", f"ai||{ai}")
        # 只保留最近 MAX_HISTORY 条
        total = r.llen(SESSION_KEY)
        if total > MAX_HISTORY:
            r.ltrim(SESSION_KEY, total - MAX_HISTORY, -1)
    except Exception:
        pass


def _clear_history() -> None:
    if is_redis_available():
        try:
            get_redis().delete(SESSION_KEY)
        except Exception:
            pass


def run():
    redis_ok = is_redis_available()
    storage = "Redis（持久化）" if redis_ok else "内存（本次有效）"
    print(f"AIOps Agent 已启动 | 对话记忆：{storage}")
    print("输入 'exit' 退出，'clear' 清空记忆，'history' 查看历史记录数。")

    # 启动时从 Redis 恢复历史
    chat_history = _load_history()
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
                _clear_history()
                print("记忆已清空（Redis + 本地）。")
                continue
            if user_input.lower() == "history":
                print(f"当前记忆：{len(chat_history)//2} 轮对话")
                continue

            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            ai_output = response["output"]

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_output))
            _save_turn(user_input, ai_output)

            print(f"\n{ai_output}\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\n已退出。")
            break
        except UnicodeEncodeError as e:
            sys.stderr.write(f"编码错误（已跳过）：{e}\n")
        except Exception as e:
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    run()
