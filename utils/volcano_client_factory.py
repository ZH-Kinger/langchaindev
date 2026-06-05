"""
火山引擎 TOS 客户端工厂。

TOS 不走阿里云 STS，凭静态 AK（settings.TOS_*）。与 aliyun_client_factory 保持
同样的降级风格：未配置或未安装 SDK 时返回 None，由调用方给出友好提示。

用法：
    client = get_tos_client()
    if client is None:
        return "TOS 凭证未配置或 SDK 未安装"
    ...
    client.close()
"""
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def get_tos_client():
    """返回 tos.TosClientV2；未配置 AK / 未安装 tos 时返回 None。"""
    if not settings.TOS_ACCESS_KEY or not settings.TOS_SECRET_KEY:
        logger.warning("[VolcanoFactory] TOS_ACCESS_KEY / TOS_SECRET_KEY 未配置")
        return None
    try:
        import tos
    except ImportError:
        logger.error("[VolcanoFactory] tos 未安装，请 pip install tos")
        return None
    return tos.TosClientV2(
        settings.TOS_ACCESS_KEY,
        settings.TOS_SECRET_KEY,
        settings.TOS_ENDPOINT,
        settings.TOS_REGION,
    )
