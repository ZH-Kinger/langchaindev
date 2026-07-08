"""桶间迁移编排：状态机 + Redis 任务记录 + 后台轮询。

状态机：NEW → RUNNING → DONE | FAILED
Redis： bkt:transfer:job:{job_id}  30 天 TTL
幂等：  job_id = hash(cloud, source, dest, 当天)

阿里 OSS→OSS：engine_mgw.submit_cross_job(src_scheme="oss") + poll_job
火山 TOS→TOS：engine_tos.submit_cross_job(src_is_tos=True) + poll_job
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta

from config.settings import settings
from utils.redis_client import get_redis
from core.bucket_transfer import paths
from core.bucket_transfer.paths import PathError
# 只读复用 transfer 的同名策略 / MGW 模式映射（不修改 transfer 逻辑）
from core.transfer.orchestrator import normalize_same_name_policy, same_name_policy_label, _mgw_modes

logger = logging.getLogger(__name__)

_KEY_PREFIX = "bkt:transfer:job:"
_TTL = 30 * 86400

STAGE_NEW     = "NEW"
STAGE_RUNNING = "RUNNING"
STAGE_DONE    = "DONE"
STAGE_FAILED  = "FAILED"

_BJ = timezone(timedelta(hours=8))


def _now() -> float:
    return time.time()


def _today() -> str:
    return datetime.now(_BJ).strftime("%Y%m%d")


def make_plan(source: str, dest: str) -> paths.BucketPlan:
    return paths.build_plan(source, dest)


def _job_id(plan: paths.BucketPlan) -> str:
    raw = f"{plan.cloud}|{plan.src.uri()}|{plan.dest.uri()}|{_today()}"
    return "bkt-" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _key(job_id: str) -> str:
    return _KEY_PREFIX + job_id


def get_job(job_id: str) -> dict | None:
    try:
        raw = get_redis().get(_key(job_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _save(job: dict) -> None:
    try:
        get_redis().setex(_key(job["job_id"]), _TTL, json.dumps(job, ensure_ascii=False))
    except Exception:
        logger.warning("[BKT] 保存任务失败 job=%s", job.get("job_id"), exc_info=True)


def create_job_record(plan: paths.BucketPlan, *, same_name: str = "", open_id: str = "") -> dict:
    """建/复用任务记录（当天同源目的幂等）。已在跑/已完成的直接返回。"""
    job_id = _job_id(plan)
    existing = get_job(job_id)
    if existing and existing.get("stage") not in (STAGE_FAILED,):
        return existing
    job = {
        "job_id": job_id,
        "cloud": plan.cloud,
        "engine": plan.engine,
        "direction": plan.direction,
        "source": plan.src.uri(),
        "dest": plan.dest.uri(),
        "src_bucket": plan.src.bucket, "src_prefix": plan.src.prefix,
        "dest_bucket": plan.dest.bucket, "dest_prefix": plan.dest.prefix,
        "same_name": normalize_same_name_policy(same_name),
        "stage": STAGE_NEW,
        "cross_job_name": "",
        "bytes": 0, "objects": 0, "error": "",
        "created_ts": _now(), "finished_ts": 0,
        "created_by": open_id,
    }
    _save(job)
    return job


def start_cross(job: dict) -> dict:
    """提交迁移任务，置 RUNNING。失败置 FAILED。"""
    try:
        policy = job.get("same_name", "skip")
        if job["engine"] == "mgw":
            # 阿里 OSS→OSS：源目的 region 自动探测（跨 region 源用公网 domain）
            from core.transfer import engine_mgw
            from tools.aliyun.oss import detect_bucket_region
            oid = job.get("created_by", "")
            src_region = detect_bucket_region(oid, job["src_bucket"])
            dest_region = detect_bucket_region(oid, job["dest_bucket"])
            deploy_region = settings.TRANSFER_OSS_REGION or src_region
            transfer_mode, overwrite_mode = _mgw_modes(policy)
            ref = engine_mgw.submit_cross_job(
                job_name=job["job_id"],
                src_bucket=job["src_bucket"], src_prefix=job["src_prefix"],
                src_scheme="oss", src_region=src_region,
                src_internal=(src_region == deploy_region),   # 同区内网、异区公网
                dest_bucket=job["dest_bucket"], dest_prefix=job["dest_prefix"],
                dest_region=dest_region, dest_internal=(dest_region == deploy_region),
                transfer_mode=transfer_mode, overwrite_mode=overwrite_mode,
                open_id=oid,
            )
        else:  # dms：火山 TOS→TOS
            from core.transfer import engine_tos
            _def = settings.TRANSFER_TOS_REGION or settings.TOS_REGION or "cn-beijing"
            # tos:// 带不出 region → 用 list_buckets 自动探测源/目的桶的 region，探测不到回退默认
            src_region = job.get("src_region") or engine_tos.detect_tos_bucket_region(job["src_bucket"], _def)
            dest_region = job.get("dest_region") or engine_tos.detect_tos_bucket_region(job["dest_bucket"], src_region)
            ref = engine_tos.submit_cross_job(
                job_name=job["job_id"],
                src_bucket=job["src_bucket"], src_prefix=job["src_prefix"], src_region=src_region,
                dest_bucket=job["dest_bucket"], dest_prefix=job["dest_prefix"], dest_region=dest_region,
                src_is_tos=True, overwrite_mode=policy, open_id=job.get("created_by", ""),
            )
            job["dest_region"] = dest_region
        job["cross_job_name"] = ref or job["job_id"]
        job["stage"] = STAGE_RUNNING
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = str(e)
        job["finished_ts"] = _now()
        logger.error("[BKT] 提交失败 job=%s", job.get("job_id"), exc_info=True)
    _save(job)
    return job


def poll_once(job: dict) -> dict:
    if job["stage"] != STAGE_RUNNING or not job.get("cross_job_name"):
        return job
    if job["engine"] == "mgw":
        from core.transfer import engine_mgw
        st = engine_mgw.poll_job(job["cross_job_name"], open_id=job.get("created_by", ""))
        status = st.get("status", "")
        done = status == engine_mgw.STATUS_FINISHED
        failed = status == engine_mgw.STATUS_INTERRUPTED
    else:
        from core.transfer import engine_tos
        st = engine_tos.poll_job(job["cross_job_name"], open_id=job.get("created_by", ""),
                                 dest_region=job.get("dest_region", ""))
        status = st.get("status", "")
        done = status in engine_tos._DONE_STATES
        failed = status in engine_tos._FAIL_STATES
    for k in ("bytes", "objects"):
        if st.get(k):
            job[k] = st[k]
    if done:
        job["stage"] = STAGE_DONE
        job["finished_ts"] = _now()
    elif failed:
        job["stage"] = STAGE_FAILED
        job["finished_ts"] = _now()
        job["error"] = st.get("error", "") or f"任务{status}"
    _save(job)
    return job


def refresh(job_id: str):
    """查进度前实时重查云端：后台轮询线程随容器重启而死，只读 Redis 会停在旧 RUNNING。
    stage 仍 RUNNING 且有 cross_job_name → poll_once 重查并落库（点“查询”/对账即可自愈）。"""
    job = get_job(job_id)
    if job and job.get("stage") == STAGE_RUNNING and job.get("cross_job_name"):
        try:
            job = poll_once(job)
        except Exception:
            pass
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 1440) -> dict:
    """提交并阻塞轮询至终态（供后台线程调用）。"""
    job = start_cross(job)
    if on_update:
        on_update(job)
    if job["stage"] == STAGE_RUNNING:
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
            job["finished_ts"] = _now()
            _save(job)
            if on_update:
                on_update(job)
    return job


def fmt_size(n: int) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.1f} {unit}"
        n /= 1024


def fmt_ts(ts: float) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, _BJ).strftime("%Y-%m-%d %H:%M:%S")
