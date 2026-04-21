from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """加载本地 Embedding 模型，进程内只加载一次"""
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        cache_folder=settings.MODEL_CACHE
    )


@lru_cache(maxsize=1)
def get_retriever():
    """初始化 ChromaDB 检索器，进程内只创建一次"""
    client_settings = ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True
    )
    vectorstore = Chroma(
        persist_directory=settings.VECTOR_DB,
        embedding_function=get_embedding_model(),
        client_settings=client_settings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})