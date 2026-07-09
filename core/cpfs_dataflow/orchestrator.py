"""
CPFS 预热/沉降编排：状态机 + Redis 任务记录 + 轮询推进。

状态机：NEW → RUNNING → DONE | FAILED
Redis: cpfs:dataflow:job:{job_id}  30 天 TTL。
幂等：job_id = hash(operation, fs_id, directory, oss, 当天)。

operation:
    preheat  TaskAction=Import   OSS → CPFS（加载）
    sink     TaskAction=Export   CPFS → OSS（沉降）

输入约定：cpfs_path 是 CPFS 侧目录（可带 cpfs:// 前缀，否则用 CPFS_FILE_SYSTEM_ID）；
oss 是可选的 OSS 路径（oss://bucket/prefix/），用于定位 DataFlow 与目标/源目录。
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from config.settings import settings
from utils.redis_client import get_redis
from core.cpfs_dataflow import engine_nas
from core.transfer.orchestrator import fmt_size, fmt_ts, fmt_duration  # 复用格式化

logger = logging.getLogger(__name__)

_KEY_PREFIX = "cpfs:dataflow:job:"
_TTL = 30 * 86400
_BJ = timezone(timedelta(hours=8))

STAGE_NEW     = "NEW"
STAGE_RUNNING = "RUNNING"
STAGE_DONE    = "DONE"
STAGE_FAILED  = "FAILED"

OP_PREHEAT = "preheat"
OP_SINK    = "sink"
_OP_ALIASES = {
    "preheat": OP_PREHEAT, "import": OP_PREHEAT, "load": OP_PREHEAT,
    "预热": OP_PREHEAT, "加载": OP_PREHEAT,
    "sink": OP_SINK, "export": OP_SINK, "flush": OP_SINK,
    "沉降": OP_SINK, "导出": OP_SINK,
}
_OP_ACTION = {OP_PREHEAT: engine_nas.ACTION_IMPORT, OP_SINK: engine_nas.ACTION_EXPORT}
_OP_LABEL = {OP_PREHEAT: "预热(OSS→CPFS)", OP_SINK: "沉降(CPFS→OSS)"}


class DataflowPathError(ValueError):
    """输入路径/操作错误，消息面向用户。"""


def normalize_operation(value: str) -> str:
    op = _OP_ALIASES.get(str(value or "").strip().lower())
    if not op:
        raise DataflowPathError(f"未知操作 `{value}`，应为 预热/preheat 或 沉降/sink。")
    return op


def operation_label(op: str) -> str:
    return _OP_LABEL.get(op, op)


@dataclass
class DataflowPlan:
    operation: str        # preheat / sink
    action: str           # Import / Export
    fs_id: str
    region: str
    cpfs_dir: str         # CPFS 侧目录（/.../ ）
    oss_bucket: str       # 可选
    oss_prefix: str       # 可选
    directory: str        # 实际下发 Directory（按 action 定向）
    dst_directory: str    # 实际下发 DstDirectory（可空）
    data_flow_id: str = ""  # 选择项直达时已知，跳过 resolve

    @property
    def edition(self) -> str:
        return engine_nas.edition(self.fs_id)


def _strip_mount(raw: str) -> str:
    """剥掉 CPFS 挂载前缀（CPFS_MOUNT_PREFIX，默认 /cpfs）。

    真机上看到的是挂载路径 /cpfs/<dir>，但 DataFlow 的 Directory/DstDirectory 是**文件系统内
    相对路径**；不剥前缀就把 `cpfs` 当成真实目录名拼进去 → 数据多套一层 cpfs/
    （落 <mount>/cpfs/<dir> 而非 <mount>/<dir>）。两条解析路径都必须剥。
    """
    raw = (raw or "").strip()
    mount = (settings.CPFS_MOUNT_PREFIX or "").rstrip("/")
    if mount and (raw == mount or raw.startswith(mount + "/")):
        return raw[len(mount):] or "/"
    return raw


def _parse_cpfs(raw: str) -> tuple[str, str]:
    """返回 (fs_id, cpfs_dir)。支持 cpfs://<fs>/<dir>/ 或裸目录（用默认 fs）。"""
    raw = (raw or "").strip()
    if not raw:
        raise DataflowPathError("缺少 CPFS 目录。")
    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        if scheme.lower() not in ("cpfs", "bmcpfs", "cepfs"):
            raise DataflowPathError(f"CPFS 路径 scheme 应为 cpfs://，收到 `{scheme}://`。")
        rest = rest.lstrip("/")
        fs, d = (rest.split("/", 1) + [""])[:2] if "/" in rest else (rest, "")
        if not fs:
            raise DataflowPathError(f"路径缺少文件系统 ID：`{raw}`")
        return fs, engine_nas.normalize_dir(d)
    # 用户给的完整路径，如 /cpfs/cwr/third_party_data/label → 去挂载前缀 → /cwr/third_party_data/label
    fs_path = _strip_mount(raw)
    fs = settings.CPFS_FILE_SYSTEM_ID
    if not fs:
        raise DataflowPathError("未配置 CPFS_FILE_SYSTEM_ID，请用 cpfs://<fs-id>/<dir>/ 显式指定。")
    return fs, engine_nas.normalize_dir(fs_path)


def _parse_oss(raw: str) -> tuple[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return "", ""
    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        if scheme.lower() != "oss":
            raise DataflowPathError(f"OSS 路径 scheme 应为 oss://，收到 `{scheme}://`。")
    else:
        rest = raw
    rest = rest.lstrip("/")
    bucket, prefix = (rest.split("/", 1) + [""])[:2] if "/" in rest else (rest, "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def make_plan(operation: str, cpfs_path: str = "", oss: str = "", *,
              fs_id: str = "", region: str = "", data_flow_id: str = "") -> DataflowPlan:
    """解析输入为预热/沉降计划。抛 DataflowPathError。

    显式给 fs_id（卡片选择直达）时，cpfs_path 当作该 fs 下的目录，跳过路径推断 fs；
    否则从 cpfs_path（cpfs://<fs>/<dir>/ 或完整本地路径）解析 fs 与目录。
    """
    op = normalize_operation(operation)
    if fs_id:
        # 卡片直达已给 fs_id，但用户填的目录仍可能带 /cpfs 挂载前缀 → 必须同样剥，
        # 否则数据多套一层 cpfs/（落 <mount>/cpfs/<dir> 而非 <mount>/<dir>）。
        cpfs_dir = engine_nas.normalize_dir(_strip_mount(cpfs_path) or "/")
    else:
        fs_id, cpfs_dir = _parse_cpfs(cpfs_path)
    oss_bucket, oss_prefix = _parse_oss(oss)
    region = region or settings.CPFS_REGION or "cn-hangzhou"
    action = _OP_ACTION[op]
    # Directory 定向：Import 处理 OSS 侧目录→落 CPFS；Export 处理 CPFS 侧目录→落 OSS
    if op == OP_PREHEAT:
        directory = oss_prefix or cpfs_dir
        dst_directory = cpfs_dir if oss_prefix else ""
    else:  # sink
        directory = cpfs_dir
        dst_directory = oss_prefix
    return DataflowPlan(
        operation=op, action=action, fs_id=fs_id, region=region,
        cpfs_dir=cpfs_dir, oss_bucket=oss_bucket, oss_prefix=oss_prefix,
        directory=engine_nas.normalize_dir(directory),
        dst_directory=engine_nas.normalize_dir(dst_directory) if dst_directory else "",
        data_flow_id=data_flow_id,
    )


def _classify(addr: str) -> str:
    a = (addr or "").strip()
    low = a.lower()
    if low.startswith("oss://"):
        return "oss"
    if low.startswith(("cpfs://", "bmcpfs://", "cepfs://")) or a.startswith("/"):
        return "cpfs"
    return ""


def _region_fs(region: str, open_id: str = "") -> str:
    fss = engine_nas.list_filesystems(region, open_id=open_id)
    if not fss:
        raise DataflowPathError(f"地区 {region} 下没有 CPFS 文件系统")
    if len(fss) > 1:
        # 多个 CPFS 时不能盲选第一个（可能选错 fs → 路径对不上/操作错文件系统）。
        # 让用户用 cpfs://<fs-id>/<目录>/ 明确指定是哪一个。
        lst = "、".join(fss)
        raise DataflowPathError(
            f"地区 {region} 下有多个 CPFS 文件系统（{lst}），无法自动判断用哪个。\n"
            f"请把源/目的的 CPFS 地址写成 `cpfs://<fs-id>/<目录>/` 指定，例如 "
            f"`cpfs://{fss[0]}/wzh/`。")
    return fss[0]


def plan_from_addresses(region: str, source: str, dest: str, *, open_id: str = "") -> DataflowPlan:
    """按 源/目的地址 定方向、解析计划（不建 DataFlow；建流延后到下发时做）。

    源是 CPFS、目的是 OSS → 沉降(Export)；源是 OSS、目的是 CPFS → 预热(Import)。
    地区由卡片给定；该地区唯一的 CPFS 文件系统自动选中（cpfs://<fs> 显式则用之）。
    DataFlow 在 start_task 时临时创建、任务终态后自动删除（用完即删，不占 10 个/CPFS 上限）。
    """
    region = (region or settings.CPFS_REGION or "").strip()
    if not region:
        raise DataflowPathError("请先选择地区")
    s_kind, d_kind = _classify(source), _classify(dest)
    if {s_kind, d_kind} != {"cpfs", "oss"}:
        raise DataflowPathError("源/目的必须一个是 CPFS 路径、另一个是 oss:// 路径")
    if s_kind == "cpfs":
        operation, cpfs_addr, oss_addr = OP_SINK, source, dest
    else:
        operation, cpfs_addr, oss_addr = OP_PREHEAT, dest, source

    if "://" in cpfs_addr:
        fs_id, fs_path = _parse_cpfs(cpfs_addr)
    else:
        fs_id = _region_fs(region, open_id=open_id)
        mount = (settings.CPFS_MOUNT_PREFIX or "").rstrip("/")
        p = cpfs_addr.strip()
        if mount and (p == mount or p.startswith(mount + "/")):
            p = p[len(mount):] or "/"
        fs_path = engine_nas.normalize_dir(p)
    oss_bucket, oss_prefix = _parse_oss(oss_addr)
    if not oss_bucket:
        raise DataflowPathError("OSS 地址缺少 bucket")

    return make_plan(operation, fs_path, f"oss://{oss_bucket}/{oss_prefix}",
                     fs_id=fs_id, region=region, data_flow_id="")


def _job_id(plan: DataflowPlan) -> str:
    day = datetime.now(_BJ).strftime("%Y%m%d")
    raw = f"{plan.operation}|{plan.fs_id}|{plan.cpfs_dir}|{plan.oss_bucket}/{plan.oss_prefix}|{day}"
    return "cpfs-" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}{job_id}"


def get_job(job_id: str) -> dict | None:
    try:
        raw = get_redis().get(_key(job_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _save(job: dict) -> None:
    job["updated_ts"] = time.time()
    try:
        get_redis().setex(_key(job["job_id"]), _TTL, json.dumps(job, ensure_ascii=False))
    except Exception:
        logger.warning("[CPFS] 写 Redis 失败 job=%s", job.get("job_id"))


def create_job_record(plan: DataflowPlan, *, open_id: str = "") -> dict:
    """落库任务记录（幂等：同 job_id 未失败则返回旧记录）。"""
    job_id = _job_id(plan)
    existing = get_job(job_id)
    if existing and existing.get("stage") not in (STAGE_FAILED,):
        return existing
    job = {
        "job_id": job_id,
        "operation": plan.operation,
        "operation_label": operation_label(plan.operation),
        "action": plan.action,
        "fs_id": plan.fs_id,
        "region": plan.region,
        "edition": plan.edition,
        "cpfs_dir": plan.cpfs_dir,
        "oss_bucket": plan.oss_bucket,
        "oss_prefix": plan.oss_prefix,
        "directory": plan.directory,
        "dst_directory": plan.dst_directory,
        "data_flow_id": plan.data_flow_id or "",
        "task_id": "",
        "stage": STAGE_NEW,
        "files_total": 0,
        "files_done": 0,
        "bytes_total": 0,
        "bytes_done": 0,
        "created_by": open_id,
        "created_ts": time.time(),
        "updated_ts": time.time(),
        "error": "",
    }
    _save(job)
    return job


def _relative_dir(full: str, base: str) -> str:
    """把 full 目录表示成相对 base 的目录（都以 / 开头结尾）。base 是 full 的祖先时截掉前缀，
    否则原样返回。用于智算版任务 Directory/DstDirectory（相对 DataFlow 的绑定路径）。"""
    full = engine_nas.normalize_dir(full)
    base = engine_nas.normalize_dir(base)
    if base == "/" or not base:
        return full
    if full == base:
        return "/"
    if full.startswith(base):
        return engine_nas.normalize_dir(full[len(base):])
    return full


def start_task(job: dict) -> dict:
    """选定 DataFlow（优先复用能覆盖该目录的已有流，否则临时建）+ 提交任务，置 RUNNING。失败置 FAILED。"""
    try:
        if not job.get("data_flow_id") or job.get("dataflow_deleted"):
            # ① 先找能覆盖该 CPFS 目录的**已有** DataFlow（按最长祖先前缀）。有就复用——不建、不删，
            #    避免与已有绑定冲突（CreateDataFlow 报 InvalidStatus.DataFlowConflicted 403）。
            flow = None
            try:
                flow = engine_nas.resolve_dataflow(
                    job["fs_id"], job["region"],
                    oss_bucket=job.get("oss_bucket", ""), fs_path=job.get("cpfs_dir", ""),
                    open_id=job.get("created_by", ""))
            except engine_nas.NasDataflowError:
                flow = None
            if flow and flow.get("data_flow_id"):
                job["data_flow_id"] = flow["data_flow_id"]
                job["dataflow_ephemeral"] = False
                job["df_fs_path"] = flow.get("fs_path") or "/"
                job["df_oss_path"] = flow.get("source_storage_path") or "/"
            else:
                # ② 没有可复用的 → 临时建（用完即删），绑定就在 cpfs_dir↔oss_prefix 上
                job["data_flow_id"] = engine_nas.create_dataflow(
                    job["fs_id"], job["region"],
                    oss_bucket=job.get("oss_bucket", ""), oss_path=job.get("oss_prefix", "") or "/",
                    fs_path=job.get("cpfs_dir", ""), open_id=job.get("created_by", ""),
                )
                job["dataflow_ephemeral"] = True
                job["df_fs_path"] = job.get("cpfs_dir", "")
                job["df_oss_path"] = job.get("oss_prefix", "") or "/"
            job["dataflow_deleted"] = False
            _save(job)
            engine_nas.wait_dataflow_running(
                job["fs_id"], job["region"], job["data_flow_id"], open_id=job.get("created_by", ""))
        # 智算版任务 Directory/DstDirectory 相对 DataFlow 的 FileSystemPath/SourceStoragePath。
        # 统一用相对路径：临时流绑在 cpfs_dir↔oss_prefix → 相对根 "/"；复用的祖先流 → 相对其绑定的子目录。
        task_dir = _relative_dir(job.get("cpfs_dir", ""), job.get("df_fs_path", "/"))
        task_dst = _relative_dir(job.get("oss_prefix", ""), job.get("df_oss_path", "/"))
        task_id = engine_nas.submit_task(
            fs_id=job["fs_id"], data_flow_id=job["data_flow_id"], action=job["action"],
            directory=task_dir, dst_directory=task_dst,
            region=job["region"], open_id=job.get("created_by", ""),
        )
        job["task_id"] = task_id
        job["stage"] = STAGE_RUNNING
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = str(e)
        job["finished_ts"] = time.time()
        logger.error("[CPFS] 任务启动失败 job=%s", job.get("job_id"), exc_info=True)
    _save(job)
    return job


_ERR_HINTS = {
    "PathNotAccessible": "CPFS 目录不存在或不可访问；确认路径正确、区分大小写、且该目录确实有数据",
    "NoSuchKey": "OSS 源前缀下没有对象",
    "AccessDenied": "无权限访问，检查 RAM 角色 / OSS 桶授权",
}


def _friendly_error(err: str) -> str:
    """给已知阿里错误码补一句中文说明，让失败卡可操作。未知码原样返回。"""
    for code, hint in _ERR_HINTS.items():
        if code in (err or ""):
            return f"{err}（{hint}）"
    return err or "任务失败"


def poll_once(job: dict) -> dict:
    """轮询一次任务状态，更新阶段/进度。"""
    if job["stage"] != STAGE_RUNNING or not job.get("task_id"):
        return job
    st = engine_nas.query_task(job["fs_id"], job["task_id"], job["region"],
                               open_id=job.get("created_by", ""))
    for k in ("files_total", "files_done", "bytes_total", "bytes_done"):
        if st.get(k):
            job[k] = st[k]
    status = st.get("status", "")
    if status in engine_nas._DONE_STATES:
        job["stage"] = STAGE_DONE
        job["finished_ts"] = time.time()
    elif status in engine_nas._FAIL_STATES:
        job["stage"] = STAGE_FAILED
        job["finished_ts"] = time.time()
        job["error"] = _friendly_error(st.get("error", "") or f"任务{status}")
    _save(job)
    return job


def _cleanup_ephemeral(job: dict) -> None:
    """任务终态后删除临时 DataFlow（用完即删，不占 10 个/CPFS 上限）。best-effort。"""
    if not job.get("dataflow_ephemeral") or job.get("dataflow_deleted"):
        return
    dfid = job.get("data_flow_id")
    if not dfid:
        return
    try:
        engine_nas.delete_dataflow(job["fs_id"], job["region"], dfid, open_id=job.get("created_by", ""))
        job["dataflow_deleted"] = True
        _save(job)
    except Exception:
        logger.warning("[CPFS] 清理临时 DataFlow 失败 df=%s", dfid, exc_info=True)


def refresh(job_id: str):
    """查进度前实时重查云端：后台轮询线程会随容器重启而死，只读 Redis 会停在旧 RUNNING。
    stage 仍 RUNNING 且有 task_id → poll_once 重查并落库，点“查询”即可看到真实终态（自愈）。"""
    job = get_job(job_id)
    if job and job.get("stage") == STAGE_RUNNING and job.get("task_id"):
        try:
            job = poll_once(job)
        except Exception:
            pass
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 1440) -> dict:
    """启动任务并阻塞轮询至终态（供后台线程调用）；终态后删临时 DataFlow。"""
    job = start_task(job)
    if on_update:
        on_update(job)
    if job["stage"] != STAGE_FAILED:
        for _ in range(max_polls):
            time.sleep(poll_interval)
            prev = job["stage"]
            job = poll_once(job)
            if job["stage"] != prev and on_update:
                on_update(job)
            if job["stage"] in (STAGE_DONE, STAGE_FAILED):
                break
        else:
            job["stage"] = STAGE_FAILED
            job["error"] = f"轮询超时（>{max_polls * poll_interval // 3600}h 未完成）"
            job["finished_ts"] = time.time()
            _save(job)
            if on_update:
                on_update(job)
    _cleanup_ephemeral(job)   # 无论成功/失败/超时，跑完删临时 DataFlow
    return job


def needs_approval(bytes_total: int) -> bool:
    """超过 CPFS_APPROVAL_GB 阈值需管理员审批（任务启动前通常为 0，靠人工确认卡兜底）。"""
    gb = bytes_total / (1024 ** 3)
    return gb > settings.CPFS_APPROVAL_GB
