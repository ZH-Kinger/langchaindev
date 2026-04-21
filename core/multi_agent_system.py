from functools import lru_cache
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import system_stats_tool, alarm_reduction_tool, k8s_restart_tool
from utils.llm_factory import get_cloud_llm
from core.prompts import get_diag_prompt, get_ops_prompt

_diag_tools = [system_stats_tool, alarm_reduction_tool]
_ops_tools  = [k8s_restart_tool]


@lru_cache(maxsize=1)
def _build_executors() -> tuple:
    """首次调用时构建两个 AgentExecutor，后续复用缓存实例"""
    llm = get_cloud_llm(temperature=0)

    diag_executor = AgentExecutor(
        agent=create_tool_calling_agent(llm, _diag_tools, get_diag_prompt()),
        tools=_diag_tools, verbose=True,
    )

    ops_executor = AgentExecutor(
        agent=create_tool_calling_agent(llm, _ops_tools, get_ops_prompt()),
        tools=_ops_tools, verbose=True,
    )

    return diag_executor, ops_executor


def run_collaborative_ops(task_description: str):
    diag_executor, ops_executor = _build_executors()
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