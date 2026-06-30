"""
阿里云 SDK Client 工厂。

为飞书用户构造已注入临时 STS 凭证的 SDK client，自动续签。
所有云 API 调用都应走这里，不要直接读 settings.PAI_DSW_ACCESS_KEY_*。

用法：
    client = get_pai_dsw_client(open_id)
    if client is None:
        return "无法获取临时凭证"
    resp = client.list_instances(...)
"""
from functools import lru_cache
from typing import Optional

from config.settings import settings
from utils.aliyun_sts import assume_role_for_user
from utils.logger import get_logger

logger = get_logger(__name__)


# ── PAI DSW ──────────────────────────────────────────────────────────────────

def get_pai_dsw_client(open_id: str = ""):
    """
    返回 PAI DSW Client。
    - open_id 非空：走 STS AssumeRole（推荐）
    - open_id 为空：用全局 PAI_DSW_ACCESS_KEY_*（兼容旧调用）
    """
    cred = _resolve_cred(open_id)
    if not cred:
        return None
    return _build_pai_dsw_client(
        ak=cred["ak"], sk=cred["sk"], token=cred["token"],
        region=settings.PAI_DSW_REGION_ID or "cn-hangzhou",
    )


@lru_cache(maxsize=32)
def _build_pai_dsw_client(ak: str, sk: str, token: str, region: str):
    """按 (ak, token) 缓存：临时凭证刷新后会自动产出新 client。"""
    from alibabacloud_pai_dsw20220101.client import Client
    from alibabacloud_tea_openapi import models as open_api_models

    cfg = open_api_models.Config(
        access_key_id=ak,
        access_key_secret=sk,
        security_token=token or None,
        region_id=region,
        endpoint=f"pai-dsw.{region}.aliyuncs.com",
    )
    return Client(cfg)


# ── ECS ──────────────────────────────────────────────────────────────────────

def get_ecs_client(open_id: str = "", region: str = ""):
    """返回 ECS Client（走 STS）。region 缺省取 PAI_DSW_REGION_ID。"""
    region = region or settings.PAI_DSW_REGION_ID or "cn-hangzhou"
    cred = _resolve_cred(open_id)
    if not cred:
        return None
    return _build_ecs_client(cred["ak"], cred["sk"], cred["token"], region)


@lru_cache(maxsize=32)
def _build_ecs_client(ak: str, sk: str, token: str, region: str):
    from alibabacloud_ecs20140526.client import Client
    from alibabacloud_tea_openapi import models as open_api_models

    cfg = open_api_models.Config(
        access_key_id=ak,
        access_key_secret=sk,
        security_token=token or None,
        region_id=region,
        endpoint=f"ecs.{region}.aliyuncs.com",
    )
    return Client(cfg)


# ── OSS ──────────────────────────────────────────────────────────────────────

def get_oss_auth(open_id: str = ""):
    """返回 (auth_obj, region_endpoint)。OSS SDK 用 oss2，不是 alibabacloud 系列。"""
    cred = _resolve_cred(open_id)
    if not cred:
        return None, None
    try:
        import oss2
        auth = (oss2.StsAuth(cred["ak"], cred["sk"], cred["token"])
                if cred["token"] else oss2.Auth(cred["ak"], cred["sk"]))
        region = settings.PAI_DSW_REGION_ID or "cn-hangzhou"
        return auth, f"https://oss-{region}.aliyuncs.com"
    except ImportError:
        logger.error("[ClientFactory] oss2 未安装，请 pip install oss2")
        return None, None


def get_oss_service(open_id: str = ""):
    """返回 oss2.Service，用于列 bucket。"""
    auth, endpoint = get_oss_auth(open_id)
    if not auth:
        return None
    import oss2
    return oss2.Service(auth, endpoint)


def get_oss_bucket(open_id: str, bucket_name: str, region: str = ""):
    """返回 oss2.Bucket，用于列对象 / 上传下载等。"""
    auth, _ = get_oss_auth(open_id)
    if not auth:
        return None
    import oss2
    region = region or settings.PAI_DSW_REGION_ID or "cn-hangzhou"
    return oss2.Bucket(auth, f"https://oss-{region}.aliyuncs.com", bucket_name)


# ── SLS ──────────────────────────────────────────────────────────────────────

def get_sls_client(open_id: str = "", region: str = ""):
    """返回 SLS Client（aliyun-log-python-sdk）。"""
    region = region or settings.PAI_DSW_REGION_ID or "cn-hangzhou"
    cred = _resolve_cred(open_id)
    if not cred:
        return None
    try:
        from aliyun.log import LogClient
        endpoint = f"{region}.log.aliyuncs.com"
        client = LogClient(endpoint, cred["ak"], cred["sk"], token=cred["token"] or None)
        return client
    except ImportError:
        logger.error("[ClientFactory] aliyun-log-python-sdk 未安装，请 pip install aliyun-log-python-sdk")
        return None


# ── 数据在线迁移服务（hcs_mgw，跨云 TOS→OSS）─────────────────────────────────

def get_mgw_client(open_id: str = ""):
    """返回数据在线迁移服务 Client（alibabacloud_hcs_mgw20240626）。

    迁移服务需 mgw:* 写权限，通常用主账号/全局 AK（背景任务无用户上下文）。
    open_id 非空时仍走 STS，但需保证角色有迁移权限。
    """
    cred = _resolve_cred(open_id)
    if not cred:
        return None
    return _build_mgw_client(
        cred["ak"], cred["sk"], cred["token"],
        endpoint=settings.MGW_ENDPOINT or "cn-beijing.mgw.aliyuncs.com",
        region=settings.MGW_REGION or "cn-beijing",
    )


@lru_cache(maxsize=8)
def _build_mgw_client(ak: str, sk: str, token: str, endpoint: str, region: str):
    try:
        from alibabacloud_hcs_mgw20240626.client import Client
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError:
        logger.error("[ClientFactory] alibabacloud_hcs_mgw20240626 未安装，"
                     "请 pip install alibabacloud_hcs_mgw20240626")
        return None
    cfg = open_api_models.Config(
        access_key_id=ak,
        access_key_secret=sk,
        security_token=token or None,
        region_id=region,
        endpoint=endpoint,
    )
    return Client(cfg)


# ── NAS DataFlow（CPFS 预热/沉降，RPC 通用 OpenAPI）──────────────────────────

def get_nas_openapi_client(open_id: str = "", region: str = ""):
    """返回调用 NAS(2017-06-26) RPC 接口的通用 OpenAPI Client。

    NAS DataFlow 没有现成的 alibabacloud_nas SDK 装在本项目里，改用通用
    alibabacloud-tea-openapi 的 call_api（同 core.ram_approval._make_ims_client 范式），
    免新增依赖、部署免 rebuild。CreateDataFlowTask 等需 nas:* 写权限，默认走全局 AK。
    """
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    cred = _resolve_cred(open_id)
    if not cred:
        return None
    return _build_nas_openapi_client(
        cred["ak"], cred["sk"], cred["token"], f"nas.{region}.aliyuncs.com",
    )


@lru_cache(maxsize=8)
def _build_nas_openapi_client(ak: str, sk: str, token: str, endpoint: str):
    try:
        from alibabacloud_tea_openapi.client import Client as OpenApiClient
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError:
        logger.error("[ClientFactory] alibabacloud-tea-openapi 未安装")
        return None
    cfg = open_api_models.Config(
        access_key_id=ak,
        access_key_secret=sk,
        security_token=token or None,
        endpoint=endpoint,
    )
    return OpenApiClient(cfg)


# ── 凭证统一解析 ─────────────────────────────────────────────────────────────

def _resolve_cred(open_id: str) -> Optional[dict]:
    """统一凭证解析。

    优先级（高 → 低）：
      1. 用户绑定的 AK/SK（飞书表单卡片）— 资源归属为 RAM 用户本人 ✅
      2. STS AssumeRole — 资源归属为 BotRole/session
      3. 全局 PAI_DSW_ACCESS_KEY_* — 兜底（管理员场景或开发期）

    返回 {ak, sk, token}，失败返回 None。
    """
    # ── ① 用户绑定的 AK/SK（最理想，资源归属正确）────────────────────────
    if open_id:
        try:
            from utils.aliyun_user_creds import get_user_ak
            user_ak = get_user_ak(open_id)
            if user_ak:
                ak, sk = user_ak
                logger.debug("[ClientFactory] open_id=%s 走用户 AK 路径", open_id)
                return {"ak": ak, "sk": sk, "token": ""}
        except Exception as e:
            logger.warning("[ClientFactory] 用户 AK 读取失败 open_id=%s: %s", open_id, e)

        # ── ② STS AssumeRole（兜底，资源归属为角色）──────────────────────
        cred = assume_role_for_user(open_id)
        if cred:
            return {
                "ak":    cred["access_key_id"],
                "sk":    cred["access_key_secret"],
                "token": cred["security_token"],
            }
        logger.warning("[ClientFactory] open_id=%s STS 失败，降级全局 AK", open_id)

    # ── ③ 全局 AK（管理员工单或开发期）─────────────────────────────────
    if settings.PAI_DSW_ACCESS_KEY_ID and settings.PAI_DSW_ACCESS_KEY_SECRET:
        return {
            "ak":    settings.PAI_DSW_ACCESS_KEY_ID,
            "sk":    settings.PAI_DSW_ACCESS_KEY_SECRET,
            "token": "",
        }
    return None