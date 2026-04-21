from functools import lru_cache
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools.system_tool import system_stats_tool
from tools.analysis_skills import alarm_reduction_tool
from tools.ops_skills import k8s_restart_tool
from utils.llm_factory import get_cloud_llm, get_edge_llm
from core.prompts import get_watcher_prompt, get_manager_prompt

_watcher_tools = [system_stats_tool]
_manager_tools = [alarm_reduction_tool, k8s_restart_tool]


@lru_cache(maxsize=1)
def _build_executors() -> tuple:
    """首次调用时构建两个 AgentExecutor，后续复用缓存实例。"""
    edge_llm  = get_edge_llm()
    cloud_llm = get_cloud_llm(temperature=0)

    watcher_executor = AgentExecutor(
        agent=create_tool_calling_agent(edge_llm, _watcher_tools, get_watcher_prompt()),
        tools=_watcher_tools, verbose=True,
    )

    manager_executor = AgentExecutor(
        agent=create_tool_calling_agent(cloud_llm, _manager_tools, get_manager_prompt()),
        tools=_manager_tools, verbose=True,
    )

    return watcher_executor, manager_executor


# --- 跨模型协同逻辑 ---

def run_hybrid_workflow(user_query):
    watcher_executor, manager_executor = _build_executors()
    print(f"🚩 任务开始: {user_query}")

    # 第一步：4B 模型先去"探路"
    print("\n[Step 1] 边缘 Agent (Qwen3-4B) 正在进行现场勘察...")
    watcher_result = watcher_executor.invoke({"input": f"检查环境，看看有哪些数据文件需要处理：{user_query}"})
    observation = watcher_result["output"]

    # 第二步：云端模型接手"决策"
    print("\n[Step 2] 云端 Agent 正在根据边缘汇报进行深度决策...")
    final_decision = manager_executor.invoke({
        "input": f"边缘 Agent 汇报如下：'{observation}'。请针对此情况进行告警降噪，并决定是否需要修复。用户原始指令：{user_query}"
    })

    return final_decision["output"]


def run():
    while True:
        try:
            user_input = input("\n任务描述 >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            result = run_hybrid_workflow(user_input)
            print(f"\n{result}\n" + "-" * 30)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    run()