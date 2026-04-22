import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

from config.settings import settings
from core.llm_factory import get_cloud_llm
from core.vector_store import get_retriever
from core.prompts_rag import aiops_prompt
from core.chains import create_rag_chain


def _get_session_history(session_id: str):
    file_path = os.path.join(settings.SESSION_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path)


def _ask_session_id() -> str:
    """交互式获取会话 ID，直接回车使用默认值。"""
    try:
        sid = input("会话 ID（直接回车使用 'default'）: ").strip()
        return sid or "default"
    except (KeyboardInterrupt, EOFError):
        return "default"


def run(session_id: str = "") -> None:
    if not session_id:
        session_id = _ask_session_id()

    llm = get_cloud_llm()
    retriever = get_retriever()
    base_chain = create_rag_chain(llm, retriever, aiops_prompt)
    chain = RunnableWithMessageHistory(
        base_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print(f"AIOps 助手已上线（RAG 模式 | 会话：{session_id}）")
    session_config = {"configurable": {"session_id": session_id}}

    while True:
        try:
            user_input = input("\n运维问题 >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        print("\n建议方案：", end="", flush=True)
        try:
            for chunk in chain.stream({"input": user_input}, config=session_config):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n出错了: {e}")
        print("\n" + "-" * 30)