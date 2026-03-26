from langchain.tools import tool
from langchain_openai import ChatOpenAI
from config.settings import settings  # 统一从 settings 引入
from core.chains import create_rag_chain
from utils.vector_store import get_retriever
from core.prompts import aiops_prompt

# --- 第一步：直接在本地实例化零件 ---

# 1. 实例化 LLM (替代掉原来的 get_qwen_model)
llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.API_KEY,
    base_url=settings.BASE_URL,
    temperature=settings.TEMPERATURE,
)

# 2. 获取检索器和 Prompt
retriever = get_retriever()
rag_prompt = aiops_prompt()

# 3. 组装成真正的 rag_chain
# 注意：这个 rag_chain 是给工具调用的“内部逻辑”
rag_chain = create_rag_chain(llm, retriever, rag_prompt)

# --- 第二步：定义工具 ---
@tool
def query_knowledge(query: str) -> str:
    """当需要查询 K8s、Kafka 或运维规程时，调用此工具。输入应为具体问题。"""
    # 强制补齐 rag_chain 需要的字典结构（input, chat_history）
    response = rag_chain.invoke({
        "input": query,
        "chat_history": ""
    })
    return response