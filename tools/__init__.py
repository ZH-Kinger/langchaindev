# tools/__init__.py
from .system_tool import get_system_health
from .rag_tool import query_knowledge

# 统一暴露给 Agent 的工具箱
ops_tools = [
    get_system_health,
    query_knowledge
]