import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

class Config:
    # 基础配置
    MODEL_NAME = "qwen-max"
    TEMPERATURE = 0.3
    MAX_TOKENS = 1000

    # 云端 API 配置（从 .env 读取，不在代码里写死密钥）
    API_KEY = os.environ.get("OPENAI_API_KEY")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 边缘模型配置（hybrid_agents.py 使用）
    EDGE_MODEL_NAME = "Qwen/Qwen3-4B"
    EDGE_API_KEY = os.environ.get("EDGE_API_KEY")
    EDGE_BASE_URL = os.environ.get("EDGE_BASE_URL")

    # 阿里云 Prometheus（SLS 托管）
    PROMETHEUS_URL           = os.environ.get("PROMETHEUS_URL", "")
    ALIYUN_ACCESS_KEY_ID     = os.environ.get("ALIYUN_ACCESS_KEY_ID", "")
    ALIYUN_ACCESS_KEY_SECRET = os.environ.get("ALIYUN_ACCESS_KEY_SECRET", "")

    # 飞书企业自建应用
    FEISHU_APP_ID            = os.environ.get("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET        = os.environ.get("FEISHU_APP_SECRET", "")
    FEISHU_CHAT_ID           = os.environ.get("FEISHU_CHAT_ID", "")
    FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")

    # 阿里云 PAI DSW（Python SDK，无需 Node.js）
    PAI_DSW_ACCESS_KEY_ID     = os.environ.get("PAI_DSW_ACCESS_KEY_ID", "")
    PAI_DSW_ACCESS_KEY_SECRET = os.environ.get("PAI_DSW_ACCESS_KEY_SECRET", "")
    PAI_DSW_REGION_ID         = os.environ.get("PAI_DSW_REGION_ID", "cn-hangzhou")
    PAI_DSW_WORKSPACE_ID      = os.environ.get("PAI_DSW_WORKSPACE_ID", "")
    PAI_DSW_RESOURCE_ID       = os.environ.get("PAI_DSW_RESOURCE_ID", "")

    # Grafana
    GRAFANA_URL            = os.environ.get("GRAFANA_URL", "")           # e.g. https://grafana.example.com
    GRAFANA_API_KEY        = os.environ.get("GRAFANA_API_KEY", "")       # Bearer token
    GRAFANA_DATASOURCE_UID = os.environ.get("GRAFANA_DATASOURCE_UID", "") # Prometheus 数据源 UID

    # Redis
    REDIS_HOST     = os.environ.get("REDIS_HOST", "127.0.0.1")
    REDIS_PORT     = int(os.environ.get("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    REDIS_DB       = int(os.environ.get("REDIS_DB", 0))

    # 业务相关配置
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    LOG_LEVEL = "INFO"

    # 路径配置
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH   = str(BASE_DIR / "data")
    VECTOR_DB   = str(BASE_DIR / "vector_db")
    MODEL_CACHE = str(BASE_DIR / "models" / "model_cache")
    SESSION_DIR = str(BASE_DIR / "sessions")

    # RAG 切分参数
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    def setup_env(self):
        # 离线开关（防止 HuggingFace 联网）
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.MODEL_CACHE
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        # 自动创建必要目录
        for path in [self.SESSION_DIR, self.MODEL_CACHE, self.VECTOR_DB, self.DATA_PATH]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"自动创建目录: {path}")


settings = Config()