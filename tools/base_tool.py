import os
import logging
from datetime import datetime
from abc import ABC, abstractmethod

# 配置运维审计日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [9527-OPS] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AIOps-Base")


class BaseOpsTool(ABC):
    """所有运维工具的基类"""

    def __init__(self):
        # 自动定位项目根目录下的 data 文件夹
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")

    def get_data_path(self, filename: str) -> str:
        """安全地获取 data 目录下的文件路径"""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"文件未找到: {path}")
        return path

    def log_operation(self, tool_name: str, params: dict, result: str):
        """统一的审计日志记录"""
        logger.info(f"工具: {tool_name} | 参数: {params} | 状态: 运行完成")
        # 你甚至可以把这个记录写进 sessions/ 下的 json 里