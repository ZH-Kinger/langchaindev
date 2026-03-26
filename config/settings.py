import os
from pathlib import Path

class Config:
    # 基础配置 (大写是规范)
    MODEL_NAME = "qwen-max"
    TEMPERATURE = 0.3
    MAX_TOKENS = 1000

    # API 配置
    API_KEY = os.environ.get("OPENAI_API_KEY", "你的api")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 业务相关的配置
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    LOG_LEVEL = "INFO"

    # 路径配置
    # 1. 动态获取项目根目录 (settings.py 的上两级)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 2. 所有的路径都基于 BASE_DIR 拼接
    VECTOR_DB = str(BASE_DIR / "vector_db")
    MODEL_CACHE = str(BASE_DIR / "core" / "models" / "model_cache")
    SESSION_DIR = str(BASE_DIR / "sessions")

    # RAG 切分参数
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    def setup_env(self):
        # 1. 离线开关
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        # 2. 关键点：这里使用的是 self.MODEL_CACHE (必须和上面定义的名字完全对齐)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.MODEL_CACHE
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        # 3. 自动创建必要的文件夹
        for path in [self.SESSION_DIR, self.MODEL_CACHE, self.VECTOR_DB]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"📁 自动创建目录: {path}")


# 实例化，方便外部直接导入
settings = Config()