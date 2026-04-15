import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

from config.settings import settings
from utils.llm_factory import get_cloud_llm
from utils.vector_store import get_retriever
from core.prompts import aiops_prompt
from core.chains import create_rag_chain

# 1. 环境初始化
settings.setup_env()

# 2. 组件初始化
def get_session_history(session_id: str):
    file_path = os.path.join(settings.SESSION_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path)

def chat():
    # 初始化 LLM 和 检索器
    llm = get_cloud_llm()
    retriever = get_retriever()

    # 组装基础 Chain
    base_chain = create_rag_chain(llm, retriever, aiops_prompt)

    # 包装记忆功能
    with_message_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("AIOps 助手已上线！(模块化重构完成)")
    session_config = {"configurable": {"session_id": "zh_kinger_ops_001"}}

    while True:
        user_input = input("\n运维问题 >> ").strip()
        if not user_input: continue
        if user_input.lower() == "exit": break

        print("\n💡 建议方案：", end="", flush=True)
        try:
            for chunk in with_message_history.stream({"input": user_input}, config=session_config):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
        print("\n" + "-" * 30)

if __name__ == "__main__":
    chat()