from operator import itemgetter
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from core.loaders import format_docs

def create_rag_chain(llm, retriever, prompt_input):
    # --- 核心修复：智能判断 ---
    # 如果 prompt_input 是一个函数，就调用它获取对象
    # 如果它已经是一个对象了（比如在 tools.py 里直接传的对象），就直接使用
    actual_prompt = prompt_input() if callable(prompt_input) else prompt_input

    return (
        RunnableParallel({
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        })
        | actual_prompt  # 使用处理后的 prompt
        | llm
        | StrOutputParser()
    )