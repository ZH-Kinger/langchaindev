from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings  # 1. 导入配置类
from config.settings import settings


def get_embedding_model():
    """统一获取 Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        cache_folder=settings.MODEL_CACHE
    )


def get_retriever():
    """初始化并返回检索器"""
    embeddings = get_embedding_model()

    # 2. 显式定义 Chroma 配置，强制关闭匿名埋点
    client_settings = ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True
    )

    # 3. 在实例化时传入 client_settings
    vectorstore = Chroma(
        persist_directory=settings.VECTOR_DB,
        embedding_function=embeddings,
        client_settings=client_settings  # 核心：在这里把红字“掐死”
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})