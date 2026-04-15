from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import system_stats_tool, alarm_reduction_tool, k8s_restart_tool
from utils.llm_factory import get_cloud_llm

# 初始化统一的 LLM（temperature=0 保证决策确定性）
llm = get_cloud_llm(temperature=0)

# 1. 定义【感知与诊断专家】 (Diagnostic Agent)
# 它负责扫描文件和降噪分析
diag_tools = [system_stats_tool, alarm_reduction_tool]
diag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名诊断专家。你的任务是发现异常文件，并利用 Pandas 工具进行深度降噪分析。你只负责产出分析报告，不执行任何修复动作。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
diag_agent = create_tool_calling_agent(llm, diag_tools, diag_prompt)
diag_executor = AgentExecutor(agent=diag_agent, tools=diag_tools, verbose=True)

# 2. 定义【自动化执行官】 (Ops Executor Agent)
# 它负责看报告并决定是否重启 K8s
ops_tools = [k8s_restart_tool]
ops_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名运维执行官。你会收到一份诊断报告。你的职责是评估报告中的风险，如果确实需要，则执行重启。如果报告建议不重启，你必须遵守。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
ops_agent = create_tool_calling_agent(llm, ops_tools, ops_prompt)
ops_executor = AgentExecutor(agent=ops_agent, tools=ops_tools, verbose=True)


def run_collaborative_ops(task_description: str):
    print(f"\n🚀 [中控台] 启动协作流程: {task_description}")

    # 阶段 1: 派诊断专家去拿结果
    print("\n--- 🕵️ 阶段 1: 专家诊断 ---")
    diag_response = diag_executor.invoke({
        "input": f"针对用户的要求 '{task_description}'，请先扫描 data 目录并给出分析结论。"
    })
    report = diag_response["output"]
    print(f"📄 诊断报告产出: \n{report}")

    # 阶段 2: 派执行官去根据报告操作
    print("\n--- 🛠️ 阶段 2: 动作执行 ---")
    final_res = ops_executor.invoke({
        "input": f"这是诊断报告：'{report}'。请根据报告内容决定是否需要执行运维操作。用户原始要求是：'{task_description}'"
    })

    return final_res["output"]


def run():
    while True:
        try:
            user_input = input("\n任务描述 >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            result = run_collaborative_ops(user_input)
            print(f"\n{result}\n" + "-" * 30)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    run()