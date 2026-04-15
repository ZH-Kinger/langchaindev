# tools/system_tool.py
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import psutil
import os
from tools.base_tool import BaseOpsTool  # 引入你的基类

# 实例化基类，获取路径管理能力
base = BaseOpsTool()


# --- 部件 1: 参数模型 (升级版) ---
class SystemCheckSchema(BaseModel):
    mode: str = Field(
        description="模式选择：'realtime' (查看实时系统负载), 'list_data' (主动扫描 data 目录下的测试文件)"
    )
    item: str = Field(
        default="cpu",
        description="当模式为 realtime 时可选：'cpu', 'memory'。当模式为 list_data 时此项可忽略。"
    )


# --- 部件 2: 核心逻辑 (增加主动拉取逻辑) ---
def get_system_stats_v2(mode: str, item: str = "cpu") -> str:
    # 模式 1：主动扫描 data 目录 (Agent 的眼睛)
    if mode == 'list_data':
        try:
            files = os.listdir(base.data_dir)
            if not files:
                return "⚠️ [主动扫描] data 目录目前为空，没有可拉取的数据集。"
            supported = ('.csv', '.json', '.xlsx', '.xls', '.log')
            data_files = [f for f in files if f.endswith(supported)]
            if not data_files:
                return "⚠️ [主动扫描] data 目录中没有可分析的数据文件（支持 .csv / .json / .xlsx / .log）。"
            summary = ", ".join(data_files)
            return f"🔍 [主动扫描] 发现 {len(data_files)} 个可分析文件：{summary}"
        except Exception as e:
            return f"❌ 扫描目录失败: {str(e)}"

    # 模式 2：原有的实时状态查询
    if mode == 'realtime':
        if item == 'cpu':
            return f"📈 当前 CPU 使用率: {psutil.cpu_percent()}%"
        elif item == 'memory':
            return f"🧠 当前内存使用率: {psutil.virtual_memory().percent}%"

    return "❓ 未知操作模式，请选择 realtime 或 list_data"


# --- 部件 3: 工具封装 ---
system_stats_tool = StructuredTool.from_function(
    func=get_system_stats_v2,
    name="system_data_manager",  # 改个更有“管理感”的名字
    description="主动管理系统数据。可以实时查看 CPU/内存负载，也可以主动扫描 data 目录下的测试文件列表。",
    args_schema=SystemCheckSchema
)