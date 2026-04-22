from functools import lru_cache
from langchain_openai import ChatOpenAI
from config.settings import settings


@lru_cache(maxsize=8)
def get_cloud_llm(temperature: float = None, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        streaming=streaming,
    )


@lru_cache(maxsize=4)
def get_edge_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.EDGE_MODEL_NAME,
        api_key=settings.EDGE_API_KEY,
        base_url=settings.EDGE_BASE_URL,
        temperature=temperature,
        max_retries=5,
        timeout=60,
    )