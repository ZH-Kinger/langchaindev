"""SSH 迁移链编排：状态机 + Redis job + 轮询推进。

状态机：NEW → STAGE1(ossutil 杭州→SGP) → STAGE2(rsync SGP→泰国) → DONE | FAILED
Redis: ssh:transfer:job:{job_id}  30 天 TTL。job_id = sgp-hash(源, 泰国目的根, 当天)。
段1 成功(stage1_rc=0 落盘)后才进 STAGE2；retry 时段1已成功则只重跑段2（省 CEN 流量）。
"""
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from config.settings import settings
from utils.redis_client import get_redis
from utils.logger import get_logger
from core.ssh_transfer import paths, engine_ssh
from core.ssh_transfer.engine_ssh import STAGE1, STAGE2
from core.transfer.orchestrator import fmt_size, fmt_ts, fmt_duration  # 复用格式化（cards 用 ts/duration）

__all__ = ["fmt_size", "fmt_ts", "fmt_duration"]  # 供 cards/cli 从本模块取用

logger = get_logger(__name__)

_KEY_PREFIX = "ssh:transfer:job:"
_TTL = 30 * 86400
_BJ = timezone(timedelta(hours=8))

STAGE_NEW    = "NEW"
STAGE_STAGE1 = "STAGE1"
STAGE_STAGE2 = "STAGE2"
STAGE_DONE   = "DONE"
STAGE_FAILED = "FAILED"

_ACTIVE = (STAGE_STAGE1, STAGE_STAGE2)
_STAGE_LABEL = {
    STAGE_NEW: "待下发", STAGE_STAGE1: "段1 杭州→新加坡(ossutil)",
    STAGE_STAGE2: "段2 新加坡→泰国(rsync)", STAGE_DONE: "完成", STAGE_FAILED: "失败",
}


def stage_label(stage: str) -> str:
    return _STAGE_LABEL.get(stage, stage)


# ── job_id / 存取 ─────────────────────────────────────────────────────────────

def _dest_root() -> str:
    return (settings.THAI_DEST_ROOT or "").rstrip("/")


def _job_id(plan: paths.Plan) -> str:
    day = datetime.now(_BJ).strftime("%Y%m%d")
    raw = f"{plan.source_uri()}|{_dest_root()}/{plan.dest_rel()}|{day}"
    return "sgp-" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}{job_id}"


def get_job(job_id: str) -> dict | None:
    try:
        raw = get_redis().get(_key(job_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _save(job: dict) -> None:
    job["updated_ts"] = time.time()   # 每次刷（对账 stale 门依赖它，auditor 曾抓 bucket 漏刷）
    try:
        get_redis().setex(_key(job["job_id"]), _TTL, json.dumps(job, ensure_ascii=False))
    except Exception:
        logger.warning("[SSHT] 写 Redis 失败 job=%s", job.get("job_id"))


def create_job_record(plan: paths.Plan, *, open_id: str = "",
                      bytes_total: int = 0, objects_total: int = 0,
                      size_known: bool = True) -> dict:
    """落库任务记录（幂等：同 job_id 未失败则返回旧记录，并回填缺失的 created_by，照 #45）。"""
    job_id = _job_id(plan)
    existing = get_job(job_id)
    if existing and existing.get("stage") not in (STAGE_FAILED,):
        if open_id and not existing.get("created_by"):
            existing["created_by"] = open_id
            _save(existing)
        return existing
    job = {
        "job_id": job_id,
        "source_bucket": plan.source_bucket,
        "source_prefix": plan.source_prefix,
        "source_uri": plan.source_uri(),
        "dest_root": _dest_root(),
        "dest_subdir": plan.dest_subdir,
        "dest_rel": plan.dest_rel(),
        "dest_uri": f"{settings.THAI_USER or 'wuji'}@{settings.THAI_HOST}:{_dest_root()}/{plan.dest_rel()}",
        "stage": STAGE_NEW,
        "stage1_rc": None,          # 段1 退出码；None=未完成，0=成功
        "bytes_total": bytes_total,
        "objects_total": objects_total,
        "estimate_ok": size_known,
        "created_by": open_id,
        "created_ts": time.time(),
        "updated_ts": time.time(),
        "finished_ts": 0,
        "error": "",
        "launched": False,
    }
    _save(job)
    return job


# ── 估算 / 审批 ───────────────────────────────────────────────────────────────

def estimate_source(plan: paths.Plan) -> tuple[int, int, bool]:
    """估算源前缀大小（字节, 对象数, ok）。走 SGP 上已配好的 ossutil du；SSH 不通/解析失败→ok=False。"""
    try:
        return engine_ssh.estimate_source(plan.source_bucket, plan.source_prefix)
    except Exception:
        logger.warning("[SSHT] 估算源大小失败 %s", plan.source_uri(), exc_info=True)
        return 0, 0, False


def needs_approval(bytes_total: int, size_known: bool = True) -> bool:
    """超阈值需审批。**估算未知(size_known=False)时 fail-safe 当作需审批**，不放行未知大小的大迁移。"""
    if not size_known:
        return True
    try:
        tb = float(settings.SSH_TRANSFER_APPROVAL_TB or 1)
    except (TypeError, ValueError):
        tb = 1.0
    return bytes_total > tb * (1024 ** 4)


# ── 推进 ──────────────────────────────────────────────────────────────────────

def _start_stage(job: dict, stage: str) -> None:
    try:
        if stage == STAGE_STAGE1:
            engine_ssh.start_stage1(job["job_id"], source_bucket=job["source_bucket"],
                                    source_prefix=job["source_prefix"])
        else:
            engine_ssh.start_stage2(job["job_id"], source_prefix=job["source_prefix"],
                                    dest_rel=job.get("dest_rel", ""))
        job["stage"] = stage
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = str(e)
        job["finished_ts"] = time.time()
        logger.error("[SSHT] 起 %s 失败 job=%s", stage, job.get("job_id"), exc_info=True)
    _save(job)


def _fmt_eta(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def _sample_progress(job: dict, eng_stage: str) -> None:
    """采样已传字节 + 速率写进 job（best-effort，失败不动）。段2 rsync 自带；段1 用日志速率，
    没直接给速率时用相邻两次字节采样差算。"""
    try:
        prog = engine_ssh.stage_progress(job["job_id"], eng_stage)
    except Exception:
        return
    now = time.time()
    bd, spd = prog.get("bytes_done"), prog.get("speed_bps")
    if bd is not None:
        prev_bd, prev_ts = job.get("_bd_sample"), job.get("_bd_ts")
        if spd is None and prev_bd is not None and prev_ts and now > prev_ts and bd >= prev_bd:
            spd = int((bd - prev_bd) / (now - prev_ts))
        job["bytes_done"] = bd
        job["_bd_sample"] = bd
        job["_bd_ts"] = now
        if prog.get("pct") is not None:
            job["pct"] = prog["pct"]
    if spd is not None:
        job["speed_bps"] = spd


def progress_line(job: dict) -> str:
    """给进度卡/CLI 的一行进度：`已传 X/Y (nn%) · 速率 nn/s · 剩余约 mm`。缺项自动省略。"""
    bd = int(job.get("bytes_done") or 0)
    bt = int(job.get("bytes_total") or 0)
    spd = int(job.get("speed_bps") or 0)
    parts = []
    pct = job.get("pct")
    if pct is None and bt and bd:
        pct = int(bd * 100 / bt)
    if bt:
        parts.append(f"已传 {fmt_size(bd)}/{fmt_size(bt)}" + (f" ({pct}%)" if pct is not None else ""))
    elif bd:
        parts.append(f"已传 {fmt_size(bd)}")
    if spd:
        parts.append(f"速率 {fmt_size(spd)}/s")
        if bt and bd < bt:
            parts.append(f"剩余约 {_fmt_eta((bt - bd) / spd)}")
    return " · ".join(parts) if parts else "进度采集中…"


def poll_once(job: dict) -> dict:
    """按当前 stage 轮询对应段一次，更新 job。"""
    stage = job.get("stage")
    if stage not in _ACTIVE:
        return job
    eng_stage = STAGE1 if stage == STAGE_STAGE1 else STAGE2
    try:
        st = engine_ssh.poll_stage(job["job_id"], eng_stage)
    except Exception:
        logger.warning("[SSHT] 轮询 %s 失败 job=%s（暂保持在途）", stage, job.get("job_id"))
        return job
    _sample_progress(job, eng_stage)   # 已传字节 + 速率（供进度卡显示）
    status = st.get("status")
    if status == "FAILED":
        job["stage"] = STAGE_FAILED
        job["error"] = st.get("error", "") or f"{stage} 失败"
        job["finished_ts"] = time.time()
    elif status == "DONE":
        if stage == STAGE_STAGE1:
            job["stage1_rc"] = st.get("rc", 0)   # 记段1成功，retry 可跳过段1
            _save(job)
            _start_stage(job, STAGE_STAGE2)       # 段1 完成 → 立即起段2
            return job
        else:  # 段2 完成 = 全链路完成
            job["stage"] = STAGE_DONE
            job["finished_ts"] = time.time()
    _save(job)
    return job


def refresh(job_id: str):
    """查进度前实时重查云端（后台轮询线程随容器重启而死，只读 Redis 会停在旧在途态）。"""
    job = get_job(job_id)
    if job and job.get("stage") in _ACTIVE:
        try:
            job = poll_once(job)
        except Exception:
            pass
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 2880) -> dict:
    """启动并阻塞轮询至终态（后台线程调用）。段1成功才进段2；retry 时段1已成功直接从段2起。

    max_polls*poll_interval 默认上限 48h（跨云 + rsync 大数据可能很久）。
    """
    resume_stage2 = job.get("stage1_rc") == 0 and job.get("stage") in (STAGE_NEW, STAGE_FAILED, STAGE_STAGE2)
    _start_stage(job, STAGE_STAGE2 if resume_stage2 else STAGE_STAGE1)
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
            break
    else:
        job["stage"] = STAGE_FAILED
        job["error"] = f"轮询超时（>{max_polls * poll_interval // 3600}h 未完成）"
        job["finished_ts"] = time.time()
        _save(job)
        if on_update:
            on_update(job)
    return job
