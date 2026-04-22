from functools import lru_cache
from langchain.tools import tool


@lru_cache(maxsize=1)
def _get_rag_chain():
    """首次调用时构建 RAG 链（加载 Embedding 模型），后续复用缓存实例"""
    from core.llm_factory import get_cloud_llm
    from core.chains import create_rag_chain
    from core.vector_store import get_retriever
    from core.prompts_rag import aiops_prompt
    return create_rag_chain(get_cloud_llm(), get_retriever(), aiops_prompt())


@tool
def query_knowledge(query: str) -> str:
    """当需要查询 K8s、Kafka 或运维规程时，调用此工具。输入应为具体问题。"""
    # 工具调用无对话上下文，传空列表而非空字符串，与 MessagesPlaceholder 类型一致
    response = _get_rag_chain().invoke({
        "input": query,
        "chat_history": []
    })
    return response