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
    FEISHU_APP_ID             = os.environ.get("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET         = os.environ.get("FEISHU_APP_SECRET", "")
    FEISHU_CHAT_ID            = os.environ.get("FEISHU_CHAT_ID", "")
    FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
    # GPU 申请表单卡片模板 ID（飞书卡片构建器发布后获取，留空则降级为 action buttons 卡片）
    FEISHU_GPU_CARD_TEMPLATE_ID    = os.environ.get("FEISHU_GPU_CARD_TEMPLATE_ID", "")
    FEISHU_AK_REGISTER_TEMPLATE_ID = os.environ.get("FEISHU_AK_REGISTER_TEMPLATE_ID", "")

    # 阿里云 PAI DSW（Python SDK，无需 Node.js）
    PAI_DSW_ACCESS_KEY_ID     = os.environ.get("PAI_DSW_ACCESS_KEY_ID", "")
    PAI_DSW_ACCESS_KEY_SECRET = os.environ.get("PAI_DSW_ACCESS_KEY_SECRET", "")
    PAI_DSW_REGION_ID         = os.environ.get("PAI_DSW_REGION_ID", "cn-hangzhou")
    PAI_DSW_WORKSPACE_ID      = os.environ.get("PAI_DSW_WORKSPACE_ID", "")
    PAI_DSW_RESOURCE_ID       = os.environ.get("PAI_DSW_RESOURCE_ID", "")
    PAI_DSW_DEFAULT_IMAGE     = os.environ.get("PAI_DSW_DEFAULT_IMAGE", "")

    # 阿里云 Bot Master 账号（用于 STS AssumeRole 和 RAM 查询）
    # Master 账号只挂 AliyunSTSAssumeRoleAccess + AliyunRAMReadOnlyAccess
    # 真正的资源操作权限挂在 BotRole-* 角色上，按用户/用户组动态选择
    ALIYUN_BOT_MASTER_AK_ID     = os.environ.get("ALIYUN_BOT_MASTER_AK_ID", "")
    ALIYUN_BOT_MASTER_AK_SECRET = os.environ.get("ALIYUN_BOT_MASTER_AK_SECRET", "")
    ALIYUN_BOT_ACCOUNT_UID      = os.environ.get("ALIYUN_BOT_ACCOUNT_UID", "")
    ALIYUN_STS_REGION_ID        = os.environ.get("ALIYUN_STS_REGION_ID", "cn-hangzhou")
    ALIYUN_STS_DURATION_SECONDS = int(os.environ.get("ALIYUN_STS_DURATION_SECONDS", "3600"))

    # 用户组 → 角色 ARN 映射（按声明顺序优先匹配，留空表示走 default）
    # 格式：JSON 数组，例：[{"group":"algo-team","role":"acs:ram::UID:role/BotRole-Algo"}]
    ALIYUN_BOT_ROLE_MAPPING_RAW = os.environ.get("ALIYUN_BOT_ROLE_MAPPING", "[]")
    ALIYUN_BOT_ROLE_DEFAULT     = os.environ.get(
        "ALIYUN_BOT_ROLE_DEFAULT",
        f"acs:ram::{os.environ.get('ALIYUN_BOT_ACCOUNT_UID', '')}:role/BotRole-Default"
        if os.environ.get("ALIYUN_BOT_ACCOUNT_UID") else "",
    )

    # 用户 AK/SK 加密 key（Fernet）— 用户在飞书表单卡片填入的 AK 用此 key 加密后存 Redis
    # 生成命令：python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # 缺失时 utils/crypto.py 会降级为透明传递并告警（建议生产必填）
    BOT_CREDS_ENCRYPTION_KEY = os.environ.get("BOT_CREDS_ENCRYPTION_KEY", "")
    # 用户 AK 30 天未使用自动清理（秒）
    USER_AK_IDLE_TTL_SECONDS = int(os.environ.get("USER_AK_IDLE_TTL_SECONDS", str(30 * 86400)))

    # Jira（GPU 工单系统）
    JIRA_URL              = os.environ.get("JIRA_URL", "")               # e.g. https://jira.example.com
    JIRA_PAT              = os.environ.get("JIRA_PAT", "")               # Personal Access Token
    JIRA_PROJECT_KEY      = os.environ.get("JIRA_PROJECT_KEY", "GPU")    # GPU 工单项目 Key
    JIRA_ISSUE_TYPE       = os.environ.get("JIRA_ISSUE_TYPE", "Task")    # 工单类型
    JIRA_WORKFLOW_PROJECT = os.environ.get("JIRA_WORKFLOW_PROJECT", "")  # 算法组工作流项目 Key（如 ALGO）

    # GitHub（工作流查询）
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")   # Personal Access Token（repo 权限）
    GITHUB_ORG   = os.environ.get("GITHUB_ORG", "")     # 组织名，如 wuji-technology
    DSW_IDLE_WARN_HOURS   = float(os.environ.get("DSW_IDLE_WARN_HOURS", "1"))   # 空闲多久发警告（小时）
    DSW_IDLE_STOP_MINUTES = int(os.environ.get("DSW_IDLE_STOP_MINUTES", "30"))  # 警告后多久自动停止（分钟）

    # Grafana
    GRAFANA_URL            = os.environ.get("GRAFANA_URL", "")           # e.g. https://grafana.example.com
    GRAFANA_API_KEY        = os.environ.get("GRAFANA_API_KEY", "")       # Bearer token
    GRAFANA_DATASOURCE_UID = os.environ.get("GRAFANA_DATASOURCE_UID", "") # Prometheus 数据源 UID

    # GPU 配额与运营
    ADMIN_FEISHU_OPEN_ID      = os.environ.get("ADMIN_FEISHU_OPEN_ID", "")
    GPU_PRICE_PER_HOUR        = float(os.environ.get("GPU_PRICE_PER_HOUR", "35.0"))
    GPU_QUOTA_HOURS_PER_MONTH = float(os.environ.get("GPU_QUOTA_HOURS_PER_MONTH", "200.0"))
    GPU_IDLE_THRESHOLD_PCT    = float(os.environ.get("GPU_IDLE_THRESHOLD_PCT", "5.0"))
    GPU_IDLE_WARN_MINUTES     = int(os.environ.get("GPU_IDLE_WARN_MINUTES", "30"))

    # 火山引擎 TOS（对象存储，静态 AK，不走阿里云 STS）
    TOS_ACCESS_KEY = os.environ.get("TOS_ACCESS_KEY", "")
    TOS_SECRET_KEY = os.environ.get("TOS_SECRET_KEY", "")
    TOS_ENDPOINT   = os.environ.get("TOS_ENDPOINT", "tos-cn-shanghai.volces.com")
    TOS_REGION     = os.environ.get("TOS_REGION", "cn-shanghai")

    # 容量巡检（OSS + TOS 目录大小定时盘点 → 飞书主动推送）
    # 默认关闭，opt-in；TARGETS 为 JSON 数组，每项 {vendor,bucket,prefix[,region]}
    CAPACITY_MONITOR_ENABLED = os.environ.get("CAPACITY_MONITOR_ENABLED", "false").lower() == "true"
    CAPACITY_MONITOR_TARGETS_RAW = os.environ.get(
        "CAPACITY_MONITOR_TARGETS",
        '[{"vendor":"oss","bucket":"wuji-bucket-hangzhou","prefix":"third-party-data/"},'
        '{"vendor":"tos","bucket":"wuji-egocentric-data","prefix":"third-party-data/"}]',
    )
    CAPACITY_MONITOR_INTERVAL_HOURS = float(os.environ.get("CAPACITY_MONITOR_INTERVAL_HOURS", "6"))
    CAPACITY_ALERT_THRESHOLD_TB     = float(os.environ.get("CAPACITY_ALERT_THRESHOLD_TB", "0"))  # 0=不告警
    CAPACITY_MONITOR_CHAT_ID        = os.environ.get("CAPACITY_MONITOR_CHAT_ID", "")  # 留空回退 FEISHU_CHAT_ID
    # 增量缓存：已扫过的交付批次目录不再重扫（批次写完即不变），大幅提速；false=每次全量重算
    CAPACITY_BATCH_CACHE_ENABLED    = os.environ.get("CAPACITY_BATCH_CACHE_ENABLED", "true").lower() == "true"

    # 容量巡检结果写入飞书多维表格（Bitable）：快照→厂家总量→批次明细 三表关联
    CAPACITY_BITABLE_ENABLED        = os.environ.get("CAPACITY_BITABLE_ENABLED", "false").lower() == "true"
    CAPACITY_BITABLE_APP_TOKEN      = os.environ.get("CAPACITY_BITABLE_APP_TOKEN", "")
    CAPACITY_BITABLE_TABLE_SNAPSHOT = os.environ.get("CAPACITY_BITABLE_TABLE_SNAPSHOT", "")  # 巡检快照表
    CAPACITY_BITABLE_TABLE_VENDOR   = os.environ.get("CAPACITY_BITABLE_TABLE_VENDOR", "")    # 厂家总量表
    CAPACITY_BITABLE_TABLE_BATCH    = os.environ.get("CAPACITY_BITABLE_TABLE_BATCH", "")     # 批次明细表

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

    # ── 配置项描述（用于自检报告）────────────────────────────────────────────────
    _REQUIRED_FIELDS = [
        ("API_KEY",                "LLM 不可用，Agent 无法工作"),
        ("PROMETHEUS_URL",         "监控分析 / GPU 训练建议不可用"),
        ("FEISHU_APP_ID",          "飞书 Bot 不可用"),
        ("FEISHU_APP_SECRET",      "飞书 Bot 不可用"),
        ("FEISHU_VERIFICATION_TOKEN", "飞书事件验证跳过（存在安全风险）"),
        ("JIRA_URL",               "GPU 工单系统不可用"),
        ("JIRA_PAT",               "GPU 工单系统不可用"),
        ("PAI_DSW_ACCESS_KEY_ID",  "DSW 实例管理不可用"),
        ("PAI_DSW_WORKSPACE_ID",   "DSW 实例管理不可用"),
        ("REDIS_HOST",             "会话记忆 / 配额管理不可用（降级到内存）"),
        ("ADMIN_FEISHU_OPEN_ID",   "大规格申请审批流不可用（将自动批准）"),
        ("TOS_ACCESS_KEY",         "火山 TOS 容量统计 / 巡检不可用"),
    ]

    def validate(self) -> list[tuple[str, str]]:
        """检查配置完整性，返回 [(缺失字段, 影响说明), ...] 列表。"""
        missing = []
        for field, impact in self._REQUIRED_FIELDS:
            val = getattr(self, field, None)
            if not val:
                missing.append((field, impact))
        return missing

    def print_validate(self) -> None:
        """启动时打印配置自检报告（有缺失才输出）。"""
        from utils.logger import get_logger
        log = get_logger("config")
        missing = self.validate()
        if not missing:
            log.info("配置自检通过，所有关键配置已设置 ✓")
            return
        log.warning("以下配置未设置，相关功能不可用：")
        for field, impact in missing:
            log.warning("  %-35s → %s", field, impact)

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

        # 配置自检
        self.print_validate()


settings = Config()