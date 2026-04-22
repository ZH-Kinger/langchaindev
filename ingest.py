import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config.settings import settings
from core.vector_store import get_embedding_model


def run_ingest():
    # 自动初始化环境（离线开关、路径创建等）
    settings.setup_env()

    # 1. 加载文件 (从 config 获取路径)
    print(f"📂 正在从 {settings.DATA_PATH} 加载文档...")
    loader = DirectoryLoader(
        settings.DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    # 2. 切分文档 (参数进 config)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 文档切分完成，共 {len(chunks)} 个片段。")

    # 3. 获取 Embedding 模型 (统一调用 utils 里的方法)
    embeddings = get_embedding_model()

    # 4. 创建并持久化向量数据库
    print(f"📦 正在写入 ChromaDB 至 {settings.VECTOR_DB}...")

    # 💡 提示：如果你想每次运行都清空旧数据重来，可以取消下面这行的注释：
    # if os.path.exists(settings.VECTOR_DB): import shutil; shutil.rmtree(settings.VECTOR_DB)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.VECTOR_DB
    )
    print(f"✅ 数据库构建完成！位置：{settings.VECTOR_DB}")


if __name__ == "__main__":
    run_ingest()