from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
# 直接从你的配置文件导入
from config.settings import settings
from tools import query_knowledge,get_system_health
from core.prompts import get_agent_prompt

# 1. 直接在这里实例化云端 LLM
llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
        temperature=settings.TEMPERATURE,
    )

# 2. 准备工具和提示词
tools = [query_knowledge,get_system_health]
prompt = get_agent_prompt()

# 3. 构造 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. 构造执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🤖 AIOps 智能体已上线！(Agent 交互模式)")
    print("输入 'exit' 或 'quit' 退出，输入 'clear' 清空记忆")
    print("=" * 50)

    # 初始化一个列表来存放当前会话的记忆
    # 注意：如果想永久保存，可以对接你之前的 FileChatMessageHistory
    chat_history = []

    while True:
        try:
            # 1. 获取 CLI 输入
            user_input = input("\n👨‍💻 运维指令 >> ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("👋 再见，ZH-Kinger！系统已安全退出。")
                break
            if user_input.lower() == "clear":
                chat_history = []
                print("🧹 记忆已清空。")
                continue

            print("\n🧠 思考中...")

            # 2. 调用 Agent 执行器
            # 传入当前的 chat_history，Agent 会根据历史记录进行推理
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # 3. 更新记忆（将本次对话存入历史）
            # 注意：create_tool_calling_agent 期望的消息格式
            from langchain_core.messages import HumanMessage, AIMessage

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response["output"]))

            # 4. 打印最终结果
            print(f"\n💡 助手方案：\n{response['output']}")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n强制退出...")
            break
        except Exception as e:
            print(f"\n❌ 运行时出错: {e}")