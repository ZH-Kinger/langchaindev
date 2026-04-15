from langchain_openai import ChatOpenAI
from config.settings import settings


def get_cloud_llm(temperature: float = None) -> ChatOpenAI:
    """
    返回云端大脑实例（Qwen-Max / DashScope）。
    temperature 默认使用 settings.TEMPERATURE，传入 0 可用于需要确定性输出的 Agent。
    """
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
    )


def get_edge_llm(temperature: float = 0) -> ChatOpenAI:
    """
    返回边缘大脑实例（Qwen3-4B / ModelScope）。
    轻量感知任务专用，自带重试与超时配置。
    """
    return ChatOpenAI(
        model=settings.EDGE_MODEL_NAME,
        api_key=settings.EDGE_API_KEY,
        base_url=settings.EDGE_BASE_URL,
        temperature=temperature,
        max_retries=5,
        timeout=60,
    )