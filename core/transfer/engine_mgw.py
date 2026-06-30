"""
阿里云「数据在线迁移服务」(hcs_mgw) 调用链封装 —— 跨云 TOS→OSS 引擎（一期）。

本质是 ImportJob：目的端只能是 OSS，源端支持第三方对象存储（火山 TOS 用 address_type="tos"）。
调用链（照搬 wuji_il 控制台任务）：
    create_address(源 tos) → verify_address
    create_address(目的 oss, 用 RAM role) → verify_address
    create_job(transfer_mode, overwrite_mode, src, dest)
    update_job(status="IMPORT_JOB_LAUNCHING")           # 启动（异步）
    get_job 轮询至 IMPORT_JOB_FINISHED / IMPORT_JOB_INTERRUPTED

凭证：源 TOS 的 access_id/secret 由调用方注入（来自加密存储/配置）；
目的 OSS 用 settings.TRANSFER_OSS_ROLE（RAM role），不落 AK。

SDK 字段名以 alibabacloud_hcs_mgw20240626 为准；模型构造用 setattr 容错（字段命名按官方示例）。
"""
import logging

from config.settings import settings
from utils.aliyun_client_factory import get_mgw_client

logger = logging.getLogger(__name__)

# 任务终态
STATUS_FINISHED    = "IMPORT_JOB_FINISHED"
STATUS_INTERRUPTED = "IMPORT_JOB_INTERRUPTED"
STATUS_LAUNCHING   = "IMPORT_JOB_LAUNCHING"


class MgwError(RuntimeError):
    """迁移服务调用失败，消息面向用户。"""


def _models():
    """惰性导入 SDK models；未安装抛 MgwError（友好提示）。"""
    try:
        from alibabacloud_hcs_mgw20240626 import models as m
        return m
    except ImportError:
        raise MgwError("alibabacloud_hcs_mgw20240626 未安装，无法执行跨云迁移。"
                       "请 pip install alibabacloud_hcs_mgw20240626")


def _tos_domain(region: str) -> str:
    """火山 TOS 的 S3 兼容 endpoint（在线迁移源用 domain 字段）。"""
    region = region or settings.TOS_REGION or "cn-beijing"
    return f"tos-s3-{region}.volces.com"


def _put_address(client, name: str, detail) -> None:
    """创建数据地址；已存在则幂等忽略（重试时不撞名报错）。

    迁移服务对同名地址再建会报错（code 含 AlreadyExist/Exist/Duplicate）。
    地址内容由 (name, detail) 决定，name 带 job_id 唯一，复用旧地址安全。
    """
    m = _models()
    info = m.CreateAddressInfo(name=name, address_detail=detail)
    try:
        # 真实 SDK：CreateAddressRequest(import_address=CreateAddressInfo)
        client.create_address(settings.MGW_USER_ID, m.CreateAddressRequest(import_address=info))
    except Exception as e:
        code = (getattr(e, "code", "") or "") + " " + str(e)
        if any(k in code for k in ("AlreadyExist", "Exist", "Duplicate", "已存在")):
            logger.info("[MGW] 数据地址 %s 已存在，复用", name)
            return
        raise


def create_tos_source_address(client, name: str, *, bucket: str, prefix: str,
                              access_id: str, access_secret: str,
                              region: str = "") -> None:
    """创建源数据地址（火山 TOS，第三方对象存储）。幂等。"""
    m = _models()
    detail = m.AddressDetail()
    detail.address_type = "tos"
    detail.access_id = access_id
    detail.access_secret = access_secret
    detail.bucket = bucket
    detail.prefix = prefix
    detail.domain = _tos_domain(region)
    _put_address(client, name, detail)


def _oss_domain(region: str, internal: bool) -> str:
    """目的 OSS endpoint。internal=True 走内网（同区免流量费，wuji_il 即此）。"""
    suffix = "-internal" if internal else ""
    return f"oss-{region}{suffix}.aliyuncs.com"


def create_oss_dest_address(client, name: str, *, bucket: str, prefix: str,
                            region: str, role: str, internal: bool = True) -> None:
    """创建目的数据地址（阿里 OSS，用 RAM role 授权，不落 AK）。幂等。

    region 形如 cn-hangzhou；region_id 用 oss-<region>，domain 按内/外网拼。
    """
    m = _models()
    detail = m.AddressDetail()
    detail.address_type = "oss"
    detail.region_id = f"oss-{region}"
    detail.bucket = bucket
    detail.prefix = prefix
    detail.role = role
    detail.domain = _oss_domain(region, internal)   # wuji_il: oss-cn-hangzhou-internal.aliyuncs.com
    _put_address(client, name, detail)


def _verify_once(client, name: str):
    """调一次 verify，返回 (status, error_message)。

    真实 SDK：verify_address(userid, address_name) 两参；
    返回体 body.verify_address_response.{status, error_message}（status 异步，刚建为空）。
    """
    resp = client.verify_address(settings.MGW_USER_ID, name)
    inner = getattr(getattr(resp, "body", None), "verify_address_response", None)
    if inner is None:
        return "", ""
    return (getattr(inner, "status", "") or ""), (getattr(inner, "error_message", "") or "")


def verify_address(client, name: str, *, retries: int = 10, interval: float = 3.0) -> bool:
    """轮询校验数据地址连通性，status=='available' 返回 True。

    status 是异步结果：刚建地址时为空，需轮询等它出 available / failed。
    超时（仍空）按通过处理——建任务时迁移服务会再校验一次，不在此处硬卡。
    """
    import time
    last_msg = ""
    for i in range(max(1, retries)):
        status, msg = _verify_once(client, name)
        last_msg = msg or last_msg
        s = status.lower()
        if s == "available":
            return True
        if s in ("failed", "rejected", "unavailable"):
            raise MgwError(f"数据地址 `{name}` 校验失败：{msg or status}")
        # status 为空 = 还在校验中，等下一轮
        if i < retries - 1:
            time.sleep(interval)
    logger.warning("[MGW] 地址 %s 校验未在 %d 次内返回 available，继续建任务（服务端会复检）",
                   name, retries)
    return True


def create_job(client, *, name: str, src_address: str, dest_address: str,
               transfer_mode: str, overwrite_mode: str,
               max_band_width: int = 0) -> None:
    """创建迁移任务。"""
    m = _models()
    info = m.CreateJobInfo(
        name=name,
        transfer_mode=transfer_mode,
        overwrite_mode=overwrite_mode,
        src_address=src_address,
        dest_address=dest_address,
    )
    if max_band_width > 0:
        try:
            info.import_qos = m.ImportQos(max_band_width=max_band_width)
        except Exception:
            logger.warning("[MGW] ImportQos 构造失败，忽略限速")
    # 真实 SDK：CreateJobRequest(import_job=CreateJobInfo)
    client.create_job(settings.MGW_USER_ID, m.CreateJobRequest(import_job=info))


def launch_job(client, name: str) -> None:
    """启动迁移任务（update_job → IMPORT_JOB_LAUNCHING）。异步。"""
    m = _models()
    info = m.UpdateJobInfo(status=STATUS_LAUNCHING)
    # 真实 SDK：UpdateJobRequest(import_job=UpdateJobInfo)
    client.update_job(settings.MGW_USER_ID, name, m.UpdateJobRequest(import_job=info))


def get_job_status(client, name: str) -> dict:
    """查询任务状态，返回 {status, bytes, objects, error}。

    真机实测（非文档）：
      - 状态在 get_job().body.import_job.status
      - 进度在 list_job_history().body.job_history_list.job_history[0]
        （CopiedCount/CopiedSize/FailedCount/SkippedCount，最新一条 CommitId 最大）；
        get_job_result 需 runtime_id，麻烦且无必要，故走 history。
    """
    m = _models()
    resp = client.get_job(settings.MGW_USER_ID, name, m.GetJobRequest())
    job = getattr(getattr(resp, "body", None), "import_job", None)
    status = getattr(job, "status", "") or "" if job is not None else ""

    copied_bytes = copied_objs = failed = 0
    try:
        h = client.list_job_history(settings.MGW_USER_ID, name, m.ListJobHistoryRequest())
        lst = getattr(getattr(h, "body", None), "job_history_list", None)
        hist = getattr(lst, "job_history", None) or []
        # 取有效计数的最新一条（-1 表示该 commit 无统计）
        for jh in hist:
            cc = getattr(jh, "copied_count", -1)
            if cc is not None and cc >= 0:
                copied_objs = cc
                copied_bytes = getattr(jh, "copied_size", 0) or 0
                failed = getattr(jh, "failed_count", 0) or 0
                break
    except Exception:
        pass   # 任务刚建/未跑时无 history，正常

    return {"status": status, "bytes": copied_bytes, "objects": copied_objs,
            "error": f"{failed} 个对象失败" if failed else ""}


# ── 高层编排：建址→校验→建任务→启动 ──────────────────────────────────────────

def submit_cross_job(*, job_name: str,
                     src_bucket: str, src_prefix: str,
                     src_access_id: str, src_access_secret: str, src_region: str,
                     dest_bucket: str, dest_prefix: str, dest_region: str,
                     dest_internal: bool = True,
                     transfer_mode: str = "", overwrite_mode: str = "",
                     open_id: str = "") -> str:
    """一次性提交 TOS→OSS 迁移任务并启动，返回 job_name（用于轮询）。

    幂等性由调用方（orchestrator 的 job_id）保证；此处不重复建址检查，
    地址名带 job_name 后缀避免撞名。失败抛 MgwError。
    """
    client = get_mgw_client(open_id)
    if client is None:
        raise MgwError("无法构造在线迁移 Client：检查 AK 与 alibabacloud_hcs_mgw20240626 是否安装。")
    if not settings.MGW_USER_ID:
        raise MgwError("未配置 MGW_USER_ID（在线迁移服务 userid）。")
    if not settings.TRANSFER_OSS_ROLE:
        raise MgwError("未配置 TRANSFER_OSS_ROLE（目的 OSS 的 RAM 角色名）。")

    transfer_mode = transfer_mode or settings.TRANSFER_MODE_DEFAULT
    overwrite_mode = overwrite_mode or settings.TRANSFER_OVERWRITE_DEFAULT
    src_addr = f"{job_name}-src"
    dest_addr = f"{job_name}-dst"

    try:
        create_tos_source_address(client, src_addr, bucket=src_bucket, prefix=src_prefix,
                                  access_id=src_access_id, access_secret=src_access_secret,
                                  region=src_region)
        verify_address(client, src_addr)
        create_oss_dest_address(client, dest_addr, bucket=dest_bucket, prefix=dest_prefix,
                                region=dest_region, role=settings.TRANSFER_OSS_ROLE,
                                internal=dest_internal)
        verify_address(client, dest_addr)
        create_job(client, name=job_name, src_address=src_addr, dest_address=dest_addr,
                   transfer_mode=transfer_mode, overwrite_mode=overwrite_mode)
        launch_job(client, job_name)
        logger.info("[MGW] 迁移任务已提交并启动 job=%s %s/%s → %s/%s",
                    job_name, src_bucket, src_prefix, dest_bucket, dest_prefix)
        return job_name
    except MgwError:
        raise
    except Exception as e:
        logger.error("[MGW] 提交迁移任务失败 job=%s", job_name, exc_info=True)
        raise MgwError(f"提交迁移任务失败：{e}")


def poll_job(job_name: str, open_id: str = "") -> dict:
    """查询一次任务状态（供 orchestrator 轮询）。失败返回 error 字段。"""
    try:
        client = get_mgw_client(open_id)
        if client is None:
            return {"status": "", "error": "迁移 Client 不可用"}
        return get_job_status(client, job_name)
    except Exception as e:
        logger.error("[MGW] 轮询任务失败 job=%s", job_name, exc_info=True)
        return {"status": "", "error": str(e)}
