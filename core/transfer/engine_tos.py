"""
火山引擎「存储迁移服务」DMS 1.0 调用链封装 —— 跨云 OSS→TOS 引擎（二期）。

模型：单接口 create_data_migrate_task，源/目的配置内联在任务里（无独立地址、无代理组，
对应控制台「公共网络」任务）。用官方 SDK volcenginesdkdms（DMSApi），非手搓签名。

所有字段值来自真机 query 账号内 `oss-tos`(task_id=552548) 任务的金标准配置：
  source_type=StorageTypeObject, 源 vendor=StorageVendorOSS,
  endpoint=https://oss-<region>.aliyuncs.com, region=oss-<region>,
  prefix_list=[...]（列表）, storage_class=InheritSource。
唯一偏离：overwrite_policy 用 "None"（同名跳过、永不覆盖）而非样板的 "Force"，
守住"永不覆盖"安全约束。overwrite_policy 合法枚举：Force / None / LastModify。

client：volcenginesdkcore.Configuration(ak/sk/region) + DMSApi；region 用目的 TOS 区域。
"""
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# 同名策略：None=同名不覆盖（跳过），守住安全约束。Force=全量覆盖（禁用）。
OVERWRITE_POLICY = "None"
SOURCE_TYPE      = "StorageTypeObject"
SOURCE_VENDOR    = "StorageVendorOSS"     # 阿里 OSS 源的 vendor 枚举（真机确认）
SOURCE_VENDOR_TOS = "StorageVendorTOS"    # 火山 TOS 源（桶间 TOS→TOS），真机验证 task 553276
STORAGE_CLASS    = "InheritSource"        # 保持原存储属性

# 任务状态（真机实测全枚举）：
#   进行中：Transferring（传输中）/ ResultGenerating（生成报告中）/ Listing 等
#   终态：Success（成功）/ Failure（失败，非 Failed）/ Stopped（停止）
STATUS_SUCCESS = "Success"
STATUS_FAILURE = "Failure"
STATUS_STOPPED = "Stopped"
_DONE_STATES   = {STATUS_SUCCESS}
_FAIL_STATES   = {STATUS_FAILURE, STATUS_STOPPED}
# 其余状态（Transferring/ResultGenerating/Listing…）一律视为进行中，继续轮询。


class TosMigError(RuntimeError):
    """火山 DMS 调用失败，消息面向用户。"""


def _overwrite_policy(value: str = "") -> str:
    v = str(value or "").strip().lower()
    if v in ("overwrite", "always", "force", "all"):
        return "Force"
    return "None"


def _api(region: str):
    """构造 DMS 1.0 API client。region=目的 TOS 区域。未装 SDK/缺凭证抛 TosMigError。"""
    ak = settings.TRANSFER_TOS_ACCESS_KEY or settings.TOS_ACCESS_KEY
    sk = settings.TRANSFER_TOS_SECRET_KEY or settings.TOS_SECRET_KEY
    if not ak or not sk:
        raise TosMigError("未配置火山 AK/SK（TRANSFER_TOS_ACCESS_KEY 或 TOS_ACCESS_KEY）。")
    try:
        import volcenginesdkcore as core
        import volcenginesdkdms as dms1
    except ImportError:
        raise TosMigError("volcengine-python-sdk 未安装，请 pip install volcengine-python-sdk")
    cfg = core.Configuration()
    cfg.ak, cfg.sk, cfg.region = ak, sk, region
    return dms1.DMSApi(core.ApiClient(cfg)), dms1


def create_migrate_task(*, task_name: str,
                        src_bucket: str, src_prefix: str, src_region: str,
                        src_access_id: str, src_access_secret: str,
                        dest_bucket: str, dest_region: str,
                        overwrite_policy: str = OVERWRITE_POLICY,
                        src_is_tos: bool = False) -> int:
    """建迁移任务，返回 task_id。目的=火山 TOS。

    src_is_tos=False（默认）：源=阿里 OSS（跨云 OSS→TOS），endpoint=oss-<region>.aliyuncs.com，
        region=oss-<region>，源 AK/SK 用传入的阿里凭证。
    src_is_tos=True：源=火山 TOS（桶间 TOS→TOS，同账号），vendor=StorageVendorTOS，
        endpoint=tos-<region>.volces.com，region=<region>（不加 oss- 前缀），源 AK/SK 用火山凭证。
    源前缀走 prefix_list（列表，is_excluded=False 表示"仅迁这些前缀"）。同名一律跳过（None）。
    """
    api, dms1 = _api(dest_region)
    if src_is_tos:
        src_vendor = SOURCE_VENDOR_TOS
        src_endpoint = f"https://tos-{src_region}.volces.com"
        src_reg = src_region
    else:
        src_vendor = SOURCE_VENDOR
        src_endpoint = f"https://oss-{src_region}.aliyuncs.com"
        src_reg = f"oss-{src_region}" if not src_region.startswith("oss-") else src_region

    src_bac = dms1.BucketAccessConfigForCreateDataMigrateTaskInput(
        vendor=src_vendor, endpoint=src_endpoint, region=src_reg,
        bucket_name=src_bucket, ak=src_access_id, sk=src_access_secret)
    source = dms1.SourceForCreateDataMigrateTaskInput(
        object_source_config=dms1.ObjectSourceConfigForCreateDataMigrateTaskInput(
            bucket_access_config=src_bac, prefix_list=[src_prefix],
            is_excluded=False, scan_with_delimiter=False))
    target = dms1.TargetForCreateDataMigrateTaskInput(
        ak=settings.TRANSFER_TOS_ACCESS_KEY or settings.TOS_ACCESS_KEY,
        sk=settings.TRANSFER_TOS_SECRET_KEY or settings.TOS_SECRET_KEY,
        bucket_name=dest_bucket)
    basic = dms1.BasicConfigForCreateDataMigrateTaskInput(
        task_name=task_name, source_type=SOURCE_TYPE,
        overwrite_policy=_overwrite_policy(overwrite_policy), storage_class=STORAGE_CLASS,
        failed_num_to_abort=0, enable_range_check=False)
    body = dms1.CreateDataMigrateTaskRequest(
        basic_config=basic, source=source, target=target)
    resp = api.create_data_migrate_task(body)
    task_id = getattr(resp, "task_id", None)
    if task_id is None:
        raise TosMigError("create_data_migrate_task 未返回 task_id")
    logger.info("[DMS1] %s 任务已建 task_id=%s %s/%s → %s",
                "TOS→TOS" if src_is_tos else "OSS→TOS", task_id, src_bucket, src_prefix, dest_bucket)
    return int(task_id)


def query_task(task_id: int, dest_region: str) -> dict:
    """查任务状态+进度，返回 {status, bytes, objects, error}。"""
    api, dms1 = _api(dest_region)
    r = api.query_data_migrate_task(dms1.QueryDataMigrateTaskRequest(task_id=int(task_id)))
    prog = getattr(r, "task_progress", None)
    return {
        "status":  getattr(r, "task_status", "") or "",
        "bytes":   getattr(prog, "transferred_bytes", 0) or 0 if prog else 0,
        "objects": getattr(prog, "transferred_objects", 0) or 0 if prog else 0,
        "error":   "",
    }


# ── 高层编排：建任务即启动（DMS 1.0 建任务后自动跑）────────────────────────────

def submit_cross_job(*, job_name: str,
                     src_bucket: str, src_prefix: str,
                     src_access_id: str = "", src_access_secret: str = "", src_region: str = "",
                     dest_bucket: str, dest_prefix: str, dest_region: str,
                     dest_internal: bool = True,            # 火山侧无内外网区分，留参数兼容
                     transfer_mode: str = "", overwrite_mode: str = "",
                     src_is_tos: bool = False, open_id: str = "") -> str:
    """提交迁移任务，返回 task_id 字符串（存进 job 的 cross_job_name 供轮询）。目的=火山 TOS。

    src_is_tos=False（默认）：源=阿里 OSS，跨云 OSS→TOS（一期）。
    src_is_tos=True：源=火山 TOS，桶间 TOS→TOS（同账号，源 AK/SK 用火山凭证）。
    DMS 1.0 建任务后自动开始迁移，无需单独 start。
    ⚠️ 落盘规则（真机 task 552876）：对象**保持完整源 key 原样**落到目的桶，不去前缀、不加前缀。
       目的只给 bucket，无法指定子目录。dest_prefix 仅记录用；落盘路径由源 key 结构决定。
    """
    # 桶间 TOS→TOS 同账号：源 AK/SK 就是火山账号凭证（与目的同一把）
    if src_is_tos and not (src_access_id and src_access_secret):
        src_access_id = settings.TRANSFER_TOS_ACCESS_KEY or settings.TOS_ACCESS_KEY
        src_access_secret = settings.TRANSFER_TOS_SECRET_KEY or settings.TOS_SECRET_KEY
    try:
        task_id = create_migrate_task(
            task_name=job_name,
            src_bucket=src_bucket, src_prefix=src_prefix, src_region=src_region,
            src_access_id=src_access_id, src_access_secret=src_access_secret,
            dest_bucket=dest_bucket, dest_region=dest_region,
            overwrite_policy=overwrite_mode or transfer_mode or OVERWRITE_POLICY,
            src_is_tos=src_is_tos)
        return str(task_id)
    except TosMigError:
        raise
    except Exception as e:
        logger.error("[DMS1] 提交迁移任务失败 job=%s", job_name, exc_info=True)
        raise TosMigError(f"提交火山迁移任务失败：{e}")


def poll_job(job_name_or_task_id: str, open_id: str = "",
             dest_region: str = "") -> dict:
    """查询一次任务状态（供 orchestrator 轮询）。job 存的是 task_id。"""
    region = dest_region or settings.TRANSFER_TOS_REGION or settings.TOS_REGION or "cn-shanghai"
    try:
        return query_task(int(job_name_or_task_id), region)
    except Exception as e:
        logger.error("[DMS1] 轮询任务失败 id=%s", job_name_or_task_id, exc_info=True)
        return {"status": "", "error": str(e)}
