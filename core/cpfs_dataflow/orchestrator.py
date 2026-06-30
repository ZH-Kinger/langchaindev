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
    mount = (settings.CPFS_MOUNT_PREFIX or "").rstrip("/")
    fs_path = raw
    if mount and (raw == mount or raw.startswith(mount + "/")):
        fs_path = raw[len(mount):] or "/"
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
        cpfs_dir = engine_nas.normalize_dir(cpfs_path or "/")
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


def start_task(job: dict) -> dict:
    """解析 DataFlow + 提交任务，置 RUNNING。失败置 FAILED。"""
    try:
        if not job.get("data_flow_id"):
            df = engine_nas.resolve_dataflow(
                job["fs_id"], job["region"],
                oss_bucket=job.get("oss_bucket", ""), fs_path=job.get("cpfs_dir", ""),
                open_id=job.get("created_by", ""),
            )
            job["data_flow_id"] = df["data_flow_id"]
        task_id = engine_nas.submit_task(
            fs_id=job["fs_id"], data_flow_id=job["data_flow_id"], action=job["action"],
            directory=job.get("directory", ""), dst_directory=job.get("dst_directory", ""),
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
        job["error"] = st.get("error", "") or f"任务{status}"
    _save(job)
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 1440) -> dict:
    """启动任务并阻塞轮询至终态（供后台线程调用）。"""
    job = start_task(job)
    if on_update:
        on_update(job)
    if job["stage"] == STAGE_FAILED:
        return job
    for _ in range(max_polls):
        time.sleep(poll_interval)
        prev = job["stage"]
        job = poll_once(job)
        if job["stage"] != prev and on_update:
            on_update(job)
        if job["stage"] in (STAGE_DONE, STAGE_FAILED):
            return job
    job["stage"] = STAGE_FAILED
    job["error"] = f"轮询超时（>{max_polls * poll_interval // 3600}h 未完成）"
    job["finished_ts"] = time.time()
    _save(job)
    if on_update:
        on_update(job)
    return job


def needs_approval(bytes_total: int) -> bool:
    """超过 CPFS_APPROVAL_GB 阈值需管理员审批（任务启动前通常为 0，靠人工确认卡兜底）。"""
    gb = bytes_total / (1024 ** 3)
    return gb > settings.CPFS_APPROVAL_GB
