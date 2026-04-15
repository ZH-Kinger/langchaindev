from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools.system_tool import system_stats_tool
from tools.analysis_skills import alarm_reduction_tool
from tools.ops_skills import k8s_restart_tool
from utils.llm_factory import get_cloud_llm, get_edge_llm

# --- 1. 配置两个物理隔离的 LLM ---
edge_llm = get_edge_llm()            # 边缘大脑：Qwen3-4B (负责感知)
cloud_llm = get_cloud_llm(temperature=0)  # 云端大脑：Qwen-Max (负责决策)

# --- 2. 定义角色与工具分配 ---


# 感知 Agent：使用 4B 模型，只给它查看权限
watcher_tools = [system_stats_tool]
watcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是边缘感知 Agent，运行在受限资源环境中。你负责监控系统状态和扫描文件。请简洁地汇报你发现的异常。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
watcher_executor = AgentExecutor(agent=create_tool_calling_agent(edge_llm, watcher_tools, watcher_prompt),
                                 tools=watcher_tools, verbose=True)

# 决策 Agent：使用云端模型，拥有降噪分析和执行权限
manager_tools = [alarm_reduction_tool, k8s_restart_tool]
manager_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是高级运维主管。你会收到边缘 Agent 的汇报，请据此进行深度分析并决定是否修复。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
manager_executor = AgentExecutor(agent=create_tool_calling_agent(cloud_llm, manager_tools, manager_prompt),
                                 tools=manager_tools, verbose=True)


# --- 3. 实现真正的跨模型协同逻辑 ---

def run_hybrid_workflow(user_query):
    print(f"🚩 任务开始: {user_query}")

    # 第一步：4B 模型先去“探路”
    print("\n[Step 1] 边缘 Agent (Qwen3-4B) 正在进行现场勘察...")
    watcher_result = watcher_executor.invoke({"input": f"检查环境，看看有哪些数据文件需要处理：{user_query}"})
    observation = watcher_result["output"]

    # 第二步：云端模型接手“决策”
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