"""
阿里云 NAS DataFlow 调用封装 —— CPFS 数据预热(Import)/沉降(Export) 引擎。

产品 NAS，版本 2017-06-26，RPC 风格。用通用 alibabacloud-tea-openapi 的 call_api
发请求（同 core.ram_approval._call_ims_api 范式），免新增 alibabacloud_nas 依赖。

只用到四个动作：
    DescribeDataFlows      查现有绑定 → 解析 DataFlowId
    CreateDataFlowTask     TaskAction=Import 预热 / Export 沉降
    DescribeDataFlowTasks  轮询 TaskId 状态/进度
    CancelDataFlowTask     取消（备用）

不调 CreateDataFlow/DeleteDataFlow（会清空 Fileset，危险）。

凭证：默认全局主账号 AK（需 nas:DescribeDataFlows / CreateDataFlowTask / DescribeDataFlowTasks）。
响应字段嵌套未在文档固定，故用容错深搜解析（兼容大小写/层级差异）。
"""
import json
import logging

from config.settings import settings
from utils.aliyun_client_factory import get_nas_openapi_client

logger = logging.getLogger(__name__)

NAS_VERSION = "2017-06-26"

# TaskAction
ACTION_IMPORT = "Import"   # OSS → CPFS（预热）
ACTION_EXPORT = "Export"   # CPFS → OSS（沉降）

# 任务状态
STATUS_PENDING   = "Pending"
STATUS_EXECUTING = "Executing"
STATUS_COMPLETED = "Completed"
STATUS_FAILED    = "Failed"
STATUS_CANCELING = "Canceling"
STATUS_CANCELED  = "Canceled"
_DONE_STATES = {STATUS_COMPLETED}
_FAIL_STATES = {STATUS_FAILED, STATUS_CANCELED}


class NasDataflowError(RuntimeError):
    """NAS DataFlow 调用失败，消息面向用户。"""


# ── 工具 ─────────────────────────────────────────────────────────────────────

def edition(fs_id: str) -> str:
    """按 FileSystemId 前缀判版本：bmcpfs-*=智算版(computing) / 其它=通用版(general)。"""
    return "computing" if (fs_id or "").startswith("bmcpfs") else "general"


def normalize_dir(path: str) -> str:
    """目录规整为以 '/' 开头和结尾（NAS DataFlow Directory 约束）。"""
    p = (path or "").strip()
    if not p:
        return "/"
    if not p.startswith("/"):
        p = "/" + p
    if not p.endswith("/"):
        p += "/"
    return p


def safe_description(desc: str) -> str:
    """NAS CreateDataFlow 的 Description 约束：仅允许 字母/中文/数字/下划线/短划线/半角冒号，
    须以字母或中文开头，2~128 字符，不能以 http:// 开头。空格等非法字符会报 IllegalCharacters。"""
    import re
    d = (desc or "").strip()
    # 非法字符（含空格）替换为短划线
    d = re.sub(r"[^0-9A-Za-z_\-:一-鿿]", "-", d)
    # 去掉开头非字母/中文的字符
    d = re.sub(r"^[^A-Za-z一-鿿]+", "", d)
    if len(d) < 2:
        d = "aiops-auto"
    return d[:128]


def _dataflow_map() -> dict:
    try:
        return json.loads(getattr(settings, "CPFS_DATAFLOW_MAP_RAW", "") or "{}")
    except Exception as e:
        logger.error("[CPFS] CPFS_DATAFLOW_MAP 解析失败: %s", e)
        return {}


def _client(open_id: str, region: str):
    client = get_nas_openapi_client(open_id, region)
    if client is None:
        raise NasDataflowError("无法构造 NAS OpenAPI Client：检查 AK 与 alibabacloud-tea-openapi。")
    return client


def _call(client, action: str, query: dict) -> dict:
    """发一次 NAS RPC 调用，返回响应 body(dict)。失败抛 NasDataflowError。"""
    from alibabacloud_tea_openapi import utils_models as om
    from alibabacloud_tea_openapi.utils import Utils
    from darabonba.runtime import RuntimeOptions

    clean = {k: v for k, v in query.items() if v is not None and v != ""}
    req = om.OpenApiRequest(query=Utils.query(clean))
    params = om.Params(
        action=action, version=NAS_VERSION, protocol="HTTPS", pathname="/",
        method="POST", auth_type="AK", style="RPC",
        req_body_type="formData", body_type="json",
    )
    try:
        resp = client.call_api(params, req, RuntimeOptions())
    except Exception as e:
        code = getattr(e, "code", "") or ""
        msg = getattr(e, "message", "") or str(e)
        raise NasDataflowError(f"NAS {action} 调用失败：{code} {msg}".strip())
    body = resp.get("body") if isinstance(resp, dict) else resp
    return body or {}


def _iter_dicts(obj):
    """深度遍历，产出所有 dict 节点。"""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _dicts_with(obj, key: str) -> list:
    """深搜含指定 key 的 dict（兼容响应层级差异）。"""
    return [d for d in _iter_dicts(obj) if key in d]


def _get(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


# ── DataFlow 解析 ────────────────────────────────────────────────────────────

def list_dataflows(fs_id: str, region: str = "", *, open_id: str = "") -> list[dict]:
    """DescribeDataFlows → 规整为 [{data_flow_id, source_storage, source_storage_path,
    fs_path, fset_id, status}]。"""
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    client = _client(open_id, region)
    body = _call(client, "DescribeDataFlows", {"RegionId": region, "FileSystemId": fs_id})
    out = []
    for d in _dicts_with(body, "DataFlowId"):
        out.append({
            "data_flow_id": _get(d, "DataFlowId"),
            "source_storage": _get(d, "SourceStorage"),
            "source_storage_path": _get(d, "SourceStoragePath"),
            "fs_path": _get(d, "FileSystemPath"),
            "fset_id": _get(d, "FsetId"),
            "status": _get(d, "Status"),
        })
    # 去重（深搜可能命中嵌套重复）
    seen, uniq = set(), []
    for d in out:
        if d["data_flow_id"] and d["data_flow_id"] not in seen:
            seen.add(d["data_flow_id"])
            uniq.append(d)
    return uniq


def create_dataflow(fs_id: str, region: str = "", *, oss_bucket: str, oss_path: str = "/",
                    fs_path: str = "/", description: str = "", open_id: str = "") -> str:
    """CreateDataFlow：把 CPFS FileSystemPath 与 oss://bucket/SourceStoragePath 绑定，返回 DataFlowId。

    目前实现 CPFS 智算版(bmcpfs-)参数（SourceStoragePath + FileSystemPath）。通用版需 FsetId/Throughput，
    另行支持。前置：OSS 桶需打标签 cpfs-dataflow=true + 服务关联角色。异步，需 wait_dataflow_running。
    """
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    if edition(fs_id) != "computing":
        raise NasDataflowError("目前仅支持 CPFS 智算版(bmcpfs-)自动创建 DataFlow；通用版需 FsetId/Throughput。")
    client = _client(open_id, region)
    body = _call(client, "CreateDataFlow", {
        "RegionId": region,
        "FileSystemId": fs_id,
        "SourceStorage": f"oss://{oss_bucket}",
        "SourceStoragePath": normalize_dir(oss_path),
        "FileSystemPath": normalize_dir(fs_path),
        "Description": safe_description(description or "aiops-auto-dataflow"),
    })
    for d in _iter_dicts(body):
        dfid = _get(d, "DataFlowId")
        if dfid:
            logger.info("[CPFS] CreateDataFlow ok fs=%s %s ↔ oss://%s%s → %s",
                        fs_id, normalize_dir(fs_path), oss_bucket, normalize_dir(oss_path), dfid)
            return dfid
    raise NasDataflowError(f"CreateDataFlow 未返回 DataFlowId：{body}")


def describe_dataflow(fs_id: str, region: str, data_flow_id: str, *, open_id: str = "") -> dict:
    client = _client(open_id, region)
    body = _call(client, "DescribeDataFlows", {
        "RegionId": region, "FileSystemId": fs_id,
        "Filters": [{"Key": "DataFlowIds", "Value": data_flow_id}],
    })
    for d in _dicts_with(body, "DataFlowId"):
        if _get(d, "DataFlowId") == data_flow_id:
            return {"status": _get(d, "Status"), "error": _get(d, "ErrorMessage")}
    return {"status": "", "error": ""}


def wait_dataflow_running(fs_id: str, region: str, data_flow_id: str, *,
                          open_id: str = "", retries: int = 30, interval: float = 4.0) -> None:
    """轮询 DataFlow 到 Running。Misconfigured/Deleting/Stopped 抛错；超时放行（建任务时服务端复检）。"""
    import time
    for _ in range(max(1, retries)):
        st = describe_dataflow(fs_id, region, data_flow_id, open_id=open_id)
        s = (st.get("status") or "").lower()
        if s == "running":
            return
        if s in ("misconfigured", "deleting", "stopped"):
            raise NasDataflowError(f"DataFlow {data_flow_id} 状态 {st.get('status')}：{st.get('error') or ''}")
        time.sleep(interval)
    logger.warning("[CPFS] DataFlow %s 未在轮询内 Running，继续建任务", data_flow_id)


def delete_dataflow(fs_id: str, region: str, data_flow_id: str, *, open_id: str = "") -> None:
    """DeleteDataFlow：删除绑定关系（仅删绑定，不动已迁移的数据）。仅 Running/Stopped 可删。"""
    client = _client(open_id, region)
    _call(client, "DeleteDataFlow", {
        "RegionId": region, "FileSystemId": fs_id, "DataFlowId": data_flow_id,
    })
    logger.info("[CPFS] DeleteDataFlow ok fs=%s df=%s", fs_id, data_flow_id)


def list_filesystems(region: str, *, open_id: str = "") -> list[str]:
    """DescribeFileSystems 列出某地域的 CPFS/CPFS 智算版文件系统 ID（cpfs-/bmcpfs- 前缀）。

    需 nas:DescribeFileSystems（AliyunNASFullAccess 含）。响应字段嵌套容错深搜。
    注：CPFS 智算版(bmcpfs)是否出现在 DescribeFileSystems 取决于账号/产品，未出现时
    用 CPFS_FILE_SYSTEM_IDS 显式补充。
    """
    client = _client(open_id, region)
    body = _call(client, "DescribeFileSystems", {"RegionId": region, "PageSize": 100, "PageNumber": 1})
    ids: list[str] = []
    for d in _dicts_with(body, "FileSystemId"):
        fid = _get(d, "FileSystemId")
        if fid and (fid.startswith("cpfs") or fid.startswith("bmcpfs")) and fid not in ids:
            ids.append(fid)
    return ids


def filesystem_names(region: str, *, open_id: str = "") -> dict:
    """{fs_id: 名称} —— 名称取 DescribeFileSystems 的 Description（阿里云控制台里的“文件系统名称”）。
    仅含 CPFS/CPFS 智算版；空名回退为空串。用于下拉展示，帮用户分辨同区多个 CPFS。"""
    client = _client(open_id, region)
    body = _call(client, "DescribeFileSystems", {"RegionId": region, "PageSize": 100, "PageNumber": 1})
    names: dict = {}
    for d in _dicts_with(body, "FileSystemId"):
        fid = _get(d, "FileSystemId")
        if fid and (str(fid).startswith("cpfs") or str(fid).startswith("bmcpfs")):
            names[fid] = _get(d, "Description", "FileSystemName", default="") or ""
    return names


def resolve_dataflow(fs_id: str, region: str = "", *, oss_bucket: str = "",
                     fs_path: str = "", open_id: str = "") -> dict:
    """选定要用的 DataFlow。优先 CPFS_DATAFLOW_MAP 显式覆盖，否则按 OSS bucket /
    FileSystemPath 在现有 DataFlow 中匹配。返回选中的 dataflow dict。找不到抛 NasDataflowError。"""
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    # ① 显式映射覆盖
    dfmap = _dataflow_map()
    for key in (f"oss://{oss_bucket}" if oss_bucket else "", fs_path or ""):
        if key and key in dfmap:
            return {"data_flow_id": dfmap[key], "source_storage": f"oss://{oss_bucket}",
                    "fs_path": fs_path, "source_storage_path": "", "fset_id": "", "status": ""}

    flows = list_dataflows(fs_id, region, open_id=open_id)
    if not flows:
        raise NasDataflowError(
            f"文件系统 `{fs_id}` 下没有可用 DataFlow。请先在控制台创建 CPFS↔OSS DataFlow，"
            f"或在 CPFS_DATAFLOW_MAP 中配置。")
    # 先按 OSS bucket 过滤候选
    cand = [f for f in flows if oss_bucket in (f.get("source_storage") or "")] if oss_bucket else list(flows)
    # 再按 CPFS 路径取最长祖先前缀消歧（一个桶可能绑多个 CPFS 路径）
    if fs_path:
        norm = normalize_dir(fs_path)
        best, best_len = None, -1
        for f in cand:
            fp = normalize_dir(f.get("fs_path") or "/")
            if norm.startswith(fp) and len(fp) > best_len:
                best, best_len = f, len(fp)
        if best:
            return best
    if oss_bucket and cand:
        if len(cand) == 1:
            return cand[0]
        raise NasDataflowError(
            f"`{fs_id}` 与 OSS `{oss_bucket}` 间有 {len(cand)} 条 DataFlow，"
            f"请把 CPFS 目录填得更具体以消歧。")
    if oss_bucket and not cand:
        raise NasDataflowError(
            f"`{fs_id}` 与 OSS `{oss_bucket}` 之间没有 DataFlow 绑定，无法预热/沉降。"
            f"请确认两者已建数据流动，或换一个桶。")
    if len(flows) == 1:
        return flows[0]
    raise NasDataflowError(
        f"文件系统 `{fs_id}` 下有 {len(flows)} 个 DataFlow，无法唯一确定，请指定 OSS 桶或更具体的 CPFS 目录。")


# ── 任务提交 / 轮询 ──────────────────────────────────────────────────────────

def submit_task(*, fs_id: str, data_flow_id: str, action: str, directory: str = "",
                region: str = "", dst_directory: str = "", conflict_policy: str = "",
                entry_list: str = "", data_type: str = "MetaAndData",
                create_dir: bool = True, open_id: str = "") -> str:
    """CreateDataFlowTask，返回 TaskId。action ∈ {Import, Export}。

    Directory/EntryList 二选一（都给则用 EntryList）。智算版自动补 ConflictPolicy。
    """
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    if action not in (ACTION_IMPORT, ACTION_EXPORT):
        raise NasDataflowError(f"非法 TaskAction：{action}")
    client = _client(open_id, region)
    query = {
        "RegionId": region,
        "FileSystemId": fs_id,
        "DataFlowId": data_flow_id,
        "TaskAction": action,
        "DataType": data_type or "MetaAndData",
    }
    if entry_list:
        query["EntryList"] = entry_list
    else:
        query["Directory"] = normalize_dir(directory)
    if dst_directory:
        query["DstDirectory"] = normalize_dir(dst_directory)
    # 智算版任务必填 ConflictPolicy；通用版不下发（避免被拒）
    if edition(fs_id) == "computing":
        query["ConflictPolicy"] = conflict_policy or settings.CPFS_CONFLICT_POLICY_DEFAULT or "SKIP_THE_FILE"
        if action == ACTION_IMPORT and create_dir:
            query["CreateDirIfNotExist"] = True
    body = _call(client, "CreateDataFlowTask", query)
    task_id = ""
    for d in _iter_dicts(body):
        task_id = _get(d, "TaskId")
        if task_id:
            break
    if not task_id:
        raise NasDataflowError(f"CreateDataFlowTask 未返回 TaskId：{body}")
    logger.info("[CPFS] %s 任务已提交 fs=%s df=%s dir=%s → task=%s",
                action, fs_id, data_flow_id, directory or entry_list, task_id)
    return task_id


def query_task(fs_id: str, task_id: str, region: str = "", *, open_id: str = "") -> dict:
    """DescribeDataFlowTasks 查单任务，返回 {status, files_total, files_done,
    bytes_total, bytes_done, error}。"""
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    client = _client(open_id, region)
    body = _call(client, "DescribeDataFlowTasks", {
        "RegionId": region,
        "FileSystemId": fs_id,
        "Filters": [{"Key": "TaskIds", "Value": task_id}],
        "WithReports": False,
    })
    task = None
    for d in _dicts_with(body, "TaskId"):
        if str(_get(d, "TaskId")) == str(task_id):
            task = d
            break
    if task is None:
        # 找不到具体任务时返回任一含 Status 的节点，避免误判失败
        cands = _dicts_with(body, "Status")
        task = cands[0] if cands else {}
    def _int(*keys):
        try:
            return int(_get(task, *keys, default=0) or 0)
        except (TypeError, ValueError):
            return 0
    return {
        "status": _get(task, "Status"),
        "files_total": _int("FilesTotal"),
        "files_done": _int("FilesDone", "ActualFiles"),
        "bytes_total": _int("BytesTotal"),
        "bytes_done": _int("BytesDone", "ActualBytes"),
        "error": _get(task, "ErrorMessage", "Message"),
    }


def cancel_task(fs_id: str, data_flow_id: str, task_id: str, region: str = "",
                *, open_id: str = "") -> None:
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    client = _client(open_id, region)
    _call(client, "CancelDataFlowTask", {
        "RegionId": region, "FileSystemId": fs_id,
        "DataFlowId": data_flow_id, "TaskId": task_id,
    })


# ── transfer 三期沉降复用的薄封装 ────────────────────────────────────────────

def submit_sink_job(*, fs_id: str, cpfs_dir: str, oss_bucket: str, oss_prefix: str = "",
                    region: str = "", conflict_policy: str = "", open_id: str = "") -> str:
    """对 cpfs_dir 做 Export（沉降到 oss_bucket/oss_prefix），返回 'TaskId@DataFlowId'。

    transfer 三期 SINKING 段调用：源全闪、目的同厂 OSS。
    """
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    df = resolve_dataflow(fs_id, region, oss_bucket=oss_bucket, fs_path=cpfs_dir, open_id=open_id)
    task_id = submit_task(
        fs_id=fs_id, data_flow_id=df["data_flow_id"], action=ACTION_EXPORT,
        directory=cpfs_dir, dst_directory=oss_prefix, region=region,
        conflict_policy=conflict_policy, open_id=open_id,
    )
    return f"{task_id}@{df['data_flow_id']}"


def poll_sink(fs_id: str, sink_ref: str, region: str = "", *, open_id: str = "") -> dict:
    """轮询沉降任务（sink_ref = 'TaskId@DataFlowId'）。返回 query_task 结构 + bytes/objects 别名。"""
    task_id = (sink_ref or "").split("@", 1)[0]
    st = query_task(fs_id, task_id, region, open_id=open_id)
    st["bytes"] = st.get("bytes_done") or st.get("bytes_total") or 0
    st["objects"] = st.get("files_done") or st.get("files_total") or 0
    return st
