"""
vePFS 预热/沉降编排：状态机 + Redis 任务记录 + 轮询推进。

状态机：NEW → RUNNING → DONE | FAILED
Redis: vepfs:dataflow:job:{job_id}  30 天 TTL。
幂等：job_id = hash(operation, fs_id, sub_path, tos, 当天)。

operation:
    preheat  TaskAction=Import   TOS → vePFS（加载）
    sink     TaskAction=Export   vePFS → TOS（沉降）

比 core.cpfs_dataflow 精简：火山无 CreateDataFlow 绑定 → start_task 直接 submit_task，
无 resolve/临建/临删 DataFlow。

输入约定：vepfs_addr 是 vePFS 侧地址（vepfs://<fs>/<subpath>/ 或裸子目录，裸时用 VEPFS_FILE_SYSTEM_ID）；
tos_addr 是 tos://bucket/prefix/。方向由 源/目的 地址类型自动判断。
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from config.settings import settings
from utils.redis_client import get_redis
from core.vepfs_dataflow import engine_vepfs
from core.transfer.orchestrator import fmt_size, fmt_ts, fmt_duration  # 复用格式化

logger = logging.getLogger(__name__)

_KEY_PREFIX = "vepfs:dataflow:job:"
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
_OP_ACTION = {OP_PREHEAT: engine_vepfs.ACTION_IMPORT, OP_SINK: engine_vepfs.ACTION_EXPORT}
_OP_LABEL = {OP_PREHEAT: "预热(TOS→vePFS)", OP_SINK: "沉降(vePFS→TOS)"}


class DataflowPathError(ValueError):
    """输入路径/操作错误，消息面向用户。"""


def normalize_operation(value: str) -> str:
    op = _OP_ALIASES.get(str(value or "").strip().lower())
    if not op:
        raise DataflowPathError(f"未知操作 `{value}`，应为 预热/preheat 或 沉降/sink。")
    return op


def operation_label(op: str) -> str:
    return _OP_LABEL.get(op, op)


_SAME_NAME_LABELS = {
    "skip": "跳过同名（永不覆盖）", "keeplatest": "保留最新", "overwrite": "覆盖同名",
}


def _same_name_label(value: str) -> str:
    return _SAME_NAME_LABELS.get(str(value or "").strip().lower(), value or "跳过同名（默认）")


def _norm_dir(path: str) -> str:
    """子目录规整为以 '/' 开头结尾。空→'/'。"""
    p = (path or "").strip()
    if not p:
        return "/"
    if not p.startswith("/"):
        p = "/" + p
    if not p.endswith("/"):
        p += "/"
    return p


@dataclass
class DataflowPlan:
    operation: str        # preheat / sink
    action: str           # Import / Export
    fs_id: str
    region: str
    sub_path: str         # vePFS 侧子目录（/.../ ）
    tos_bucket: str
    tos_prefix: str
    same_name: str = ""


def _parse_vepfs(raw: str) -> tuple[str, str]:
    """返回 (fs_id, sub_path)。支持 vepfs://<fs>/<dir>/ 或裸目录（用默认 fs）。"""
    raw = (raw or "").strip()
    if not raw:
        raise DataflowPathError("缺少 vePFS 目录。")
    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        if scheme.lower() not in ("vepfs", "pfs"):
            raise DataflowPathError(f"vePFS 路径 scheme 应为 vepfs://，收到 `{scheme}://`。")
        rest = rest.lstrip("/")
        fs, d = (rest.split("/", 1) + [""])[:2] if "/" in rest else (rest, "")
        if not fs:
            raise DataflowPathError(f"路径缺少文件系统 ID：`{raw}`")
        return fs, _norm_dir(d)
    fs = settings.VEPFS_FILE_SYSTEM_ID
    if not fs:
        raise DataflowPathError("未配置 VEPFS_FILE_SYSTEM_ID，请用 vepfs://<fs-id>/<dir>/ 显式指定。")
    return fs, _norm_dir(raw)


def _parse_tos(raw: str) -> tuple[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return "", ""
    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        if scheme.lower() != "tos":
            raise DataflowPathError(f"TOS 路径 scheme 应为 tos://，收到 `{scheme}://`。")
    else:
        rest = raw
    rest = rest.lstrip("/")
    bucket, prefix = (rest.split("/", 1) + [""])[:2] if "/" in rest else (rest, "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _classify(addr: str) -> str:
    a = (addr or "").strip().lower()
    if a.startswith("tos://"):
        return "tos"
    if a.startswith(("vepfs://", "pfs://")) or a.startswith("/"):
        return "vepfs"
    return ""


def make_plan(operation: str, vepfs_addr: str = "", tos: str = "", *,
              fs_id: str = "", region: str = "", same_name: str = "") -> DataflowPlan:
    """解析输入为预热/沉降计划。抛 DataflowPathError。

    显式给 fs_id（卡片直达）时 vepfs_addr 当作该 fs 下子目录；否则从 vepfs_addr 解析 fs+目录。
    """
    op = normalize_operation(operation)
    if fs_id:
        sub_path = _norm_dir(vepfs_addr or "/")
    else:
        fs_id, sub_path = _parse_vepfs(vepfs_addr)
    tos_bucket, tos_prefix = _parse_tos(tos)
    region = region or settings.VEPFS_REGION or "cn-beijing"
    return DataflowPlan(
        operation=op, action=_OP_ACTION[op], fs_id=fs_id, region=region,
        sub_path=sub_path, tos_bucket=tos_bucket, tos_prefix=tos_prefix,
        same_name=same_name or "",
    )


def _region_fs(region: str) -> str:
    """某地区唯一的 vePFS 文件系统自动选中；多个时报错让用户用 vepfs://<fs>/ 指定（不盲选）。"""
    fss = engine_vepfs.list_filesystems(region)
    if not fss:
        raise DataflowPathError(f"地区 {region} 下没有 vePFS 文件系统（或 AK 无 DescribeFileSystems 权限）。")
    if len(fss) > 1:
        lst = "、".join(f"{f['fs_id']}({f['name']})" for f in fss)
        raise DataflowPathError(
            f"地区 {region} 下有多个 vePFS 文件系统（{lst}），无法自动判断用哪个。\n"
            f"请把 vePFS 地址写成 `vepfs://<fs-id>/<目录>/` 指定，例如 `vepfs://{fss[0]['fs_id']}/wzh/`。")
    return fss[0]["fs_id"]


def plan_from_addresses(region: str, source: str, dest: str, *, same_name: str = "") -> DataflowPlan:
    """按 源/目的地址 定方向、解析计划。

    源 vePFS、目的 TOS → 沉降(Export)；源 TOS、目的 vePFS → 预热(Import)。
    地区由卡片给定（vePFS 文件系统所在区域）。vePFS 地址给裸目录时：优先 VEPFS_FILE_SYSTEM_ID，
    否则按地区自动解析该地区唯一 fs（多个则要求 vepfs://<fs>/ 显式指定）。
    """
    region = (region or settings.VEPFS_REGION or "").strip()
    if not region:
        raise DataflowPathError("请先选择地区")
    s_kind, d_kind = _classify(source), _classify(dest)
    if {s_kind, d_kind} != {"vepfs", "tos"}:
        raise DataflowPathError("源/目的必须一个是 vePFS 路径（vepfs://... 或 /...）、另一个是 tos:// 路径")
    if s_kind == "vepfs":
        operation, vepfs_addr, tos_addr = OP_SINK, source, dest
    else:
        operation, vepfs_addr, tos_addr = OP_PREHEAT, dest, source

    if "://" in vepfs_addr:
        fs_id, sub_path = _parse_vepfs(vepfs_addr)
    else:
        fs_id = settings.VEPFS_FILE_SYSTEM_ID or _region_fs(region)
        sub_path = _norm_dir(vepfs_addr)
    tos_bucket, tos_prefix = _parse_tos(tos_addr)
    if not tos_bucket:
        raise DataflowPathError("TOS 地址缺少 bucket")
    return make_plan(operation, sub_path, f"tos://{tos_bucket}/{tos_prefix}",
                     fs_id=fs_id, region=region, same_name=same_name)


def _job_id(plan: DataflowPlan) -> str:
    day = datetime.now(_BJ).strftime("%Y%m%d")
    raw = f"{plan.operation}|{plan.fs_id}|{plan.sub_path}|{plan.tos_bucket}/{plan.tos_prefix}|{day}"
    return "vepfs-" + hashlib.sha1(raw.encode()).hexdigest()[:12]


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
        logger.warning("[VEPFS] 写 Redis 失败 job=%s", job.get("job_id"))


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
        "sub_path": plan.sub_path,
        "tos_bucket": plan.tos_bucket,
        "tos_prefix": plan.tos_prefix,
        "same_name": plan.same_name or settings.VEPFS_CONFLICT_POLICY_DEFAULT or "Skip",
        "task_id": "",
        "stage": STAGE_NEW,
        "files_total": 0,
        "files_done": 0,
        "bytes_total": 0,
        "bytes_done": 0,
        "created_by": open_id,
        "created_ts": time.time(),
        "updated_ts": time.time(),
        "finished_ts": 0,
        "error": "",
    }
    _save(job)
    return job


def start_task(job: dict) -> dict:
    """提交任务，置 RUNNING。失败置 FAILED。"""
    try:
        task_id = engine_vepfs.submit_task(
            fs_id=job["fs_id"], task_action=job["action"],
            tos_bucket=job["tos_bucket"], tos_prefix=job.get("tos_prefix", ""),
            sub_path=job.get("sub_path", ""), region=job["region"],
            same_name_policy=job.get("same_name", ""),
        )
        job["task_id"] = task_id
        job["stage"] = STAGE_RUNNING
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = str(e)
        job["finished_ts"] = time.time()
        logger.error("[VEPFS] 任务启动失败 job=%s", job.get("job_id"), exc_info=True)
    _save(job)
    return job


def poll_once(job: dict) -> dict:
    """轮询一次任务状态，更新阶段/进度。"""
    if job["stage"] != STAGE_RUNNING or not job.get("task_id"):
        return job
    st = engine_vepfs.query_task(job["task_id"], job["fs_id"], job["region"])
    for k in ("files_total", "files_done", "bytes_total", "bytes_done"):
        if st.get(k):
            job[k] = st[k]
    status = st.get("status", "")
    if engine_vepfs.is_done(status):
        job["stage"] = STAGE_DONE
        job["finished_ts"] = time.time()
    elif engine_vepfs.is_failed(status):
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
    return job
