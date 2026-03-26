import psutil
from langchain_core.tools import tool

@tool
def get_system_health():
    """查询服务器的实时 CPU 和内存使用率。"""
    return f"CPU: {psutil.cpu_percent()}% | Memory: {psutil.virtual_memory().percent}%"