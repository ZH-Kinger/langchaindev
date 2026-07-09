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

请求字段名来自 SDK models（snake_case）。cn-shanghai 真机反查已确认：
  - DataStorage = **裸桶名**（带 tos:// 会 InvalidParameter.BucketName）
  - DataStoragePath / SubPath 非空须 **首尾都带斜杠** /a/b/（内部名 SourceStoragePrefix / SubPath）
  - DataStoragePath 允许空串（= 整桶根）
待反查：task status 终态枚举串（SDK 里是自由 str）
"""
import json
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
# 含 "unsuccess"：捕获 "Unsuccessful" 这类既含 success(DONE命中) 又表失败的歧义串；
# 配合 poll_once 先判 failed，确保它判为 FAILED 而非 DONE。
_FAIL_HINTS = ("unsuccess", "fail", "error", "cancel", "stopped", "abort")


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
    """DataStorage = 裸桶名（cn-shanghai 真机反查：带 tos:// 前缀会被判 InvalidParameter.BucketName；
    裸桶名才被正确解析为 BucketName）。"""
    b = (tos_bucket or "").strip()
    if b.startswith("tos://"):
        b = b[len("tos://"):]
    return b.strip("/")


def _norm_slash_dir(p: str) -> str:
    """vePFS DataFlow 路径格式（cn-shanghai 真机反查）：SubPath 与 DataStoragePath 非空时必须
    首尾都带 '/'（形如 /a/b/）；无首斜杠、仅首斜杠、仅尾斜杠都会被判 InvalidParameter.*。
    空串保留（DataStoragePath 空 = 整桶根，合法）。"""
    p = (p or "").strip()
    if not p:
        return ""
    if not p.startswith("/"):
        p = "/" + p
    if not p.endswith("/"):
        p += "/"
    return p


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
        data_storage_path=_norm_slash_dir(tos_prefix),   # 首尾带斜杠 /a/b/ 或空
        same_name_file_policy=_overwrite_policy(same_name_policy),
    )
    # vePFS 侧目录：Fileset 优先（智算/配额目录），否则子目录 SubPath（首尾带斜杠 /a/b/）
    if fileset_id:
        kwargs["fileset_id"] = fileset_id
    if sub_path:
        kwargs["sub_path"] = _norm_slash_dir(sub_path)
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
    # 必须带分页！真机反查确认：不传 page_number/page_size 时火山返回 total_count>0 但
    # data_flow_tasks=[]（空列表）→ 永远拿不到任务 → 空 status → 任务永远卡 RUNNING。
    req = vepfs.DescribeDataFlowTasksRequest(
        file_system_id=fs_id, data_flow_task_ids=str(task_id),
        page_number=1, page_size=100)
    try:
        resp = api.describe_data_flow_tasks(req)
    except Exception as e:
        raise VepfsDataflowError(f"DescribeDataFlowTasks 调用失败：{_err(e)}")
    tasks = getattr(resp, "data_flow_tasks", None) or getattr(resp, "DataFlowTasks", None) or []
    if not tasks and (getattr(resp, "total_count", 0) or getattr(resp, "TotalCount", 0) or 0):
        logger.warning("[VEPFS] DescribeDataFlowTasks 命中 total_count 但列表空——检查分页参数 (task=%s)", task_id)
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


def list_filesystems(region: str) -> list[dict]:
    """DescribeFileSystems 列出某 region 的 vePFS 文件系统 [{fs_id, region, name, status}]。

    需 AK 有 `vepfs:DescribeFileSystems`。仅返回有 file_system_id 的项，附带名称/状态供消歧展示。
    """
    api, vepfs = _api(region)
    try:
        resp = api.describe_file_systems(vepfs.DescribeFileSystemsRequest(page_number=1, page_size=100))
    except Exception as e:
        raise VepfsDataflowError(f"DescribeFileSystems 调用失败：{_err(e)}")
    out = []
    for fs in (getattr(resp, "file_systems", None) or getattr(resp, "FileSystems", None) or []):
        fid = getattr(fs, "file_system_id", "") or getattr(fs, "FileSystemId", "")
        if fid:
            out.append({
                "fs_id": fid,
                "region": getattr(fs, "region_id", "") or getattr(fs, "RegionId", "") or region,
                "name": getattr(fs, "file_system_name", "") or getattr(fs, "FileSystemName", ""),
                "status": getattr(fs, "status", "") or getattr(fs, "Status", ""),
            })
    return out


def cancel_task(task_id: str, fs_id: str, region: str) -> None:
    api, vepfs = _api(region)
    try:
        api.cancel_data_flow_task(vepfs.CancelDataFlowTaskRequest(
            file_system_id=fs_id, data_flow_task_id=str(task_id)))
    except Exception as e:
        raise VepfsDataflowError(f"CancelDataFlowTask 调用失败：{_err(e)}")


# 常见火山错误码 → 人话（面向用户，替代甩原始 JSON）
_FRIENDLY_CODES = {
    "InvalidParameter.BucketMaybeNotExist":
        "指定的 TOS 桶不存在，请检查是否漏填桶名（完整格式：tos://<桶>/<路径>/）",
    "InvalidParameter.SourceStoragePrefix":
        "源/目的路径前缀非法（子目录需形如 /a/b/、首尾带斜杠）",
    "InvalidParameter.BucketName":
        "TOS 桶名格式非法（应是裸桶名，别带 tos:// 前缀或多余斜杠）",
}


def _err(e) -> str:
    """从火山 SDK 异常里抽可读消息：优先解析错误 JSON 的 Error.Code/Message，
    命中已知码给人话，否则回退 Code：Message，再退回原始文本（截断）。"""
    raw = ""
    for attr in ("body", "message", "reason"):
        v = getattr(e, attr, "")
        if v:
            raw = str(v)
            break
    if not raw:
        raw = str(e)
    try:
        obj = json.loads(raw)
        err = (obj.get("ResponseMetadata") or {}).get("Error") or obj.get("Error") or {}
        code = err.get("Code") or ""
        msg = err.get("Message") or ""
        if code:
            friendly = _FRIENDLY_CODES.get(code)
            if friendly:
                return f"{friendly}（{code}）"
            return f"{code}：{msg}" if msg else code
    except Exception:
        pass
    return raw[:300]
