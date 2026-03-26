# core/loaders.py

def format_docs(docs):
    """
    将检索到的文档列表合并成一个长字符串，方便喂给 LLM。
    """
    return "\n\n".join(doc.page_content for doc in docs)