"""
火山引擎 vePFS「数据流动」调用封装 —— vePFS 数据预热(Import)/沉降(Export) 引擎。

产品「文件存储 vePFS」，service `vepfs`，version `2022-01-01`，静态 AK（复用 TOS_ACCESS_KEY/
SECRET，无 STS）。官方 Python 子包 volcenginesdkvepfs（VEPFSApi），含在已装的
volcengine-python-sdk（同 core.transfer.engine_tos 用的 volcenginesdkdms 是同一个大包）。

client 构造与 engine_tos._api 完全一致：Configuration(ak/sk/region) + ApiClient + VEPFSApi。
region 须与 vePFS 文件系统所在区域一致。

只用三个动作：
    CreateDataFlowTask     TaskAction=Import 预热(TOS→vePFS) / Export 沉降(vePFS→TOS)
    DescribeDataFlowTasks  轮询 DataFlowTaskId 状态/进度
    CancelDataFlowTask     取消（备用）

与阿里 NAS DataFlow 的关键差异：火山**没有** CreateDataFlow 持久绑定对象，任务参数里直接带
TOS 桶/前缀 + vePFS 侧 SubPath/FilesetId，方向只由 TaskAction 决定，不反转源/目的字段。

请求字段名来自 SDK models（snake_case）。少数取值待真机反查（已在注释标注）：
  - DataStorage 桶串格式（tos://bucket vs 裸桶名）
  - task status 终态枚举串（SDK 里是自由 str）
"""
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

VEPFS_VERSION = "2022-01-01"

# TaskAction
ACTION_IMPORT = "Import"   # TOS → vePFS（预热）
ACTION_EXPORT = "Export"   # vePFS → TOS（沉降）

DATA_TYPE_DEFAULT = "MetaAndData"   # 元数据+数据；仅元数据用 "Metadata"

# 同名冲突策略：Skip=跳过(默认，守"永不覆盖") / KeepLatest=比最后修改时间保新 / OverWrite=覆盖
POLICY_SKIP       = "Skip"
POLICY_KEEPLATEST = "KeepLatest"
POLICY_OVERWRITE  = "OverWrite"

# task status：SDK 未把 status 做成受限枚举（自由 str），故用「子串归类」判终态，
# 兼容 Running/Success/Finished/Failed/Canceled 等未知大小写变体。上线前真机反查确认。
_DONE_HINTS = ("success", "finished", "complete", "done")
_FAIL_HINTS = ("fail", "error", "cancel", "stopped", "abort")


class VepfsDataflowError(RuntimeError):
    """vePFS 数据流动调用失败，消息面向用户。"""


def _overwrite_policy(value: str = "") -> str:
    """把用户/配置的同名策略归一到 vePFS 枚举。默认 Skip（永不覆盖）。"""
    v = str(value or "").strip().lower()
    if v in ("overwrite", "force", "always", "all", "overwrite_existing"):
        return POLICY_OVERWRITE
    if v in ("keeplatest", "keep_latest", "latest", "newest"):
        return POLICY_KEEPLATEST
    return POLICY_SKIP


def is_done(status: str) -> bool:
    s = (status or "").lower()
    return any(h in s for h in _DONE_HINTS)


def is_failed(status: str) -> bool:
    s = (status or "").lower()
    return any(h in s for h in _FAIL_HINTS)


def _api(region: str):
    """构造 vePFS API client。region=vePFS 文件系统所在区域。未装 SDK/缺凭证抛错。"""
    ak = settings.TOS_ACCESS_KEY
    sk = settings.TOS_SECRET_KEY
    if not ak or not sk:
        raise VepfsDataflowError("未配置火山 AK/SK（TOS_ACCESS_KEY / TOS_SECRET_KEY）。")
    try:
        import volcenginesdkcore as core
        import volcenginesdkvepfs as vepfs
    except ImportError:
        raise VepfsDataflowError("volcengine-python-sdk 未安装，请 pip install volcengine-python-sdk")
    cfg = core.Configuration()
    cfg.ak, cfg.sk, cfg.region = ak, sk, region
    return vepfs.VEPFSApi(core.ApiClient(cfg)), vepfs


def _data_storage(tos_bucket: str) -> str:
    """DataStorage 桶串。默认 tos://<bucket>（若真机报参数错，改成裸桶名）。"""
    b = (tos_bucket or "").strip()
    if b.startswith("tos://"):
        return b
    return f"tos://{b}"


def submit_task(*, fs_id: str, task_action: str, tos_bucket: str, tos_prefix: str = "",
                sub_path: str = "", fileset_id: str = "", region: str,
                same_name_policy: str = "", data_type: str = DATA_TYPE_DEFAULT) -> str:
    """CreateDataFlowTask，返回 DataFlowTaskId。task_action ∈ {Import, Export}。

    TOS 侧 = DataStorage(桶) + DataStoragePath(前缀)；vePFS 侧 = FileSystemId + SubPath(/FilesetId)。
    方向由 task_action 决定，字段不反转。
    """
    if task_action not in (ACTION_IMPORT, ACTION_EXPORT):
        raise VepfsDataflowError(f"非法 TaskAction：{task_action}")
    api, vepfs = _api(region)
    kwargs = dict(
        file_system_id=fs_id,
        task_action=task_action,
        data_type=data_type or DATA_TYPE_DEFAULT,
        data_storage=_data_storage(tos_bucket),
        data_storage_path=tos_prefix or "",
        same_name_file_policy=_overwrite_policy(same_name_policy),
    )
    # vePFS 侧目录：Fileset 优先（智算/配额目录），否则子目录 SubPath
    if fileset_id:
        kwargs["fileset_id"] = fileset_id
    if sub_path:
        kwargs["sub_path"] = sub_path
    req = vepfs.CreateDataFlowTaskRequest(**kwargs)
    try:
        resp = api.create_data_flow_task(req)
    except Exception as e:
        raise VepfsDataflowError(f"CreateDataFlowTask 调用失败：{_err(e)}")
    task_id = getattr(resp, "data_flow_task_id", None) or getattr(resp, "DataFlowTaskId", None)
    if not task_id:
        raise VepfsDataflowError(f"CreateDataFlowTask 未返回 DataFlowTaskId：{resp}")
    logger.info("[VEPFS] %s 任务已建 task=%s fs=%s tos=%s/%s sub=%s",
                task_action, task_id, fs_id, tos_bucket, tos_prefix, sub_path or fileset_id)
    return str(task_id)


def query_task(task_id: str, fs_id: str, region: str) -> dict:
    """DescribeDataFlowTasks 查单任务，返回 {status, bytes_total, bytes_done,
    files_total, files_done, error}。字段容错 getattr（SDK 命名可能有出入）。"""
    api, vepfs = _api(region)
    req = vepfs.DescribeDataFlowTasksRequest(
        file_system_id=fs_id, data_flow_task_ids=str(task_id))
    try:
        resp = api.describe_data_flow_tasks(req)
    except Exception as e:
        raise VepfsDataflowError(f"DescribeDataFlowTasks 调用失败：{_err(e)}")
    tasks = getattr(resp, "data_flow_tasks", None) or getattr(resp, "DataFlowTasks", None) or []
    task = None
    for t in tasks:
        tid = getattr(t, "data_flow_task_id", None) or getattr(t, "DataFlowTaskId", None)
        if str(tid) == str(task_id):
            task = t
            break
    if task is None and tasks:
        task = tasks[0]
    if task is None:
        return {"status": "", "bytes_total": 0, "bytes_done": 0,
                "files_total": 0, "files_done": 0, "error": ""}

    def _g(*names):
        for n in names:
            v = getattr(task, n, None)
            if v not in (None, ""):
                return v
        return None

    def _int(*names):
        try:
            return int(_g(*names) or 0)
        except (TypeError, ValueError):
            return 0

    status = _g("status", "Status") or ""
    failed = _int("failed_count", "FailedCount")
    parts = []
    if is_failed(status):
        if failed:
            parts.append(f"{failed} 个对象处理失败")
        rep = _g("reports", "Reports")
        if rep and not parts:
            parts.append(str(rep)[:200])
        if not parts:
            parts.append(f"任务 {status}")
    return {
        "status": status,
        "bytes_total": _int("total_size", "TotalSize"),
        "bytes_done": _int("exec_size", "ExecSize"),
        "files_total": _int("total_count", "TotalCount"),
        "files_done": _int("exec_count", "ExecCount"),
        "error": "；".join(parts),
    }


def cancel_task(task_id: str, fs_id: str, region: str) -> None:
    api, vepfs = _api(region)
    try:
        api.cancel_data_flow_task(vepfs.CancelDataFlowTaskRequest(
            file_system_id=fs_id, data_flow_task_id=str(task_id)))
    except Exception as e:
        raise VepfsDataflowError(f"CancelDataFlowTask 调用失败：{_err(e)}")


def _err(e) -> str:
    """从火山 SDK 异常里抽可读消息。"""
    for attr in ("body", "message", "reason"):
        v = getattr(e, attr, "")
        if v:
            return str(v)[:300]
    return str(e)[:300]
