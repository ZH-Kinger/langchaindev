from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from chromadb.config import Settings as ChromaSettings
from config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        cache_folder=settings.MODEL_CACHE,
    )


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    vectorstore = Chroma(
        persist_directory=settings.VECTOR_DB,
        embedding_function=get_embedding_model(),
        client_settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})