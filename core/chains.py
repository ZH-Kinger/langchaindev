from operator import itemgetter
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(llm, retriever, prompt_input):
    actual_prompt = prompt_input() if callable(prompt_input) else prompt_input

    return (
        RunnableParallel({
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        })
        | actual_prompt
        | llm
        | StrOutputParser()
    )