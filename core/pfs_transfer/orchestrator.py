"""PFS↔PFS 跨云直传 链式编排器（第 6 条搬运链）。

三段顺序驱动，每段复用对应现成子 orchestrator 的 run_to_completion（自包含、阻塞到该段终态）：
    SINKING(源 PFS→源云对象存储) → CROSSING(跨云对象存储→对象存储) → PREHEATING(目的云对象存储→目的 PFS)

    P1 vePFS→CPFS：vepfs.Export → transfer(TOS→OSS) → cpfs.Import
    P2 CPFS→vePFS：cpfs.Export  → transfer(OSS→TOS) → vepfs.Import   （P2 跨云段未真机验，前置阻塞）

段级"跳过已成功段"续跑（照 ssh_transfer stage1_rc）：链 job 记 sink_done/cross_done/preheat_done +
各子 job_id；retry/重启续跑跳过已 DONE 段。不改任何现有引擎，纯加法式编排。
"""
import hashlib
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from config.settings import settings
from utils.redis_client import get_redis

from . import paths
from core.vepfs_dataflow import orchestrator as vepfs_o
from core.cpfs_dataflow import orchestrator as cpfs_o
from core.transfer import orchestrator as transfer_o

logger = logging.getLogger(__name__)

_KEY_PREFIX = "pfs:transfer:job:"
_LAUNCH_PREFIX = "pfs:transfer:launch:"
_TTL = 30 * 86400
_BJ = timezone(timedelta(hours=8))

STAGE_NEW        = "NEW"
STAGE_SINKING    = "SINKING"
STAGE_CROSSING   = "CROSSING"
STAGE_PREHEATING = "PREHEATING"
STAGE_DONE       = "DONE"
STAGE_FAILED     = "FAILED"

_ACTIVE = (STAGE_SINKING, STAGE_CROSSING, STAGE_PREHEATING)

# 段 → (done 标记键, 子 job_id 键, 展示标签)
_STAGES = (
    (STAGE_SINKING,    "sink_done",    "sink_job_id",    "沉降(源PFS→源云对象存储)"),
    (STAGE_CROSSING,   "cross_done",   "cross_job_id",   "跨云(对象存储→对象存储)"),
    (STAGE_PREHEATING, "preheat_done", "preheat_job_id", "预热(目的云对象存储→目的PFS)"),
)
_STAGE_LABEL = {s: lbl for s, _d, _j, lbl in _STAGES}


class PfsTransferError(RuntimeError):
    """PFS 直传编排失败，消息面向用户。"""


# ── job 存取 ────────────────────────────────────────────────────────────────

def _job_id(plan: paths.Plan) -> str:
    day = datetime.now(_BJ).strftime("%Y%m%d")
    raw = (f"{plan.direction}|{plan.src_pfs.scheme}://{plan.src_pfs.fs_id}/{plan.src_pfs.sub_path}|"
           f"{plan.dst_pfs.scheme}://{plan.dst_pfs.fs_id}/{plan.dst_pfs.sub_path}|{day}")
    return "xpfs-" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}{job_id}"


def get_job(job_id: str) -> dict | None:
    try:
        import json
        raw = get_redis().get(_key(job_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _save(job: dict) -> None:
    try:
        import json
        job["updated_ts"] = time.time()
        get_redis().set(_key(job["job_id"]), json.dumps(job, ensure_ascii=False), ex=_TTL)
    except Exception:
        logger.warning("[PFS] save job failed id=%s", job.get("job_id"), exc_info=True)


def _staging_prefix(base_prefix: str, chain_id: str) -> str:
    """chain 级中转前缀：<基前缀><chain_id>/ —— 用 chain_id 隔离，多链不撞、便于清理定位、
    且同一链（当天幂等同 id）retry/续跑复用同一 staging。"""
    return f"{base_prefix}{chain_id}/"


def create_job_record(plan: paths.Plan, *, open_id: str = "", same_name: str = "") -> dict:
    """建/取链 job。当天幂等（同方向+源+目的复用旧记录，回填 created_by 照 #45）。"""
    job_id = _job_id(plan)
    existing = get_job(job_id)
    if existing and existing.get("stage") not in (None, STAGE_FAILED):
        if open_id and not existing.get("created_by"):
            existing["created_by"] = open_id
            _save(existing)
        return existing

    src_prefix = _staging_prefix(plan.src_staging.base_prefix, job_id)
    dst_prefix = _staging_prefix(plan.dst_staging.base_prefix, job_id)
    job = {
        "job_id": job_id,
        "chain": "pfs",
        "direction": plan.direction,
        "same_name": same_name or "",
        "created_by": open_id or (existing or {}).get("created_by", ""),
        "created_ts": time.time(),
        "updated_ts": time.time(),
        "finished_ts": 0,
        "stage": STAGE_NEW,
        "error": "",
        "launched": False,
        "src_pfs": {"scheme": plan.src_pfs.scheme, "fs_id": plan.src_pfs.fs_id,
                    "sub_path": plan.src_pfs.sub_path, "region": plan.src_pfs.region},
        "dst_pfs": {"scheme": plan.dst_pfs.scheme, "fs_id": plan.dst_pfs.fs_id,
                    "sub_path": plan.dst_pfs.sub_path, "region": plan.dst_pfs.region},
        "src_staging": {"scheme": plan.src_staging.scheme, "bucket": plan.src_staging.bucket,
                        "prefix": src_prefix, "region": plan.src_staging.region,
                        "dataflow_id": plan.src_staging.dataflow_id},
        "dst_staging": {"scheme": plan.dst_staging.scheme, "bucket": plan.dst_staging.bucket,
                        "prefix": dst_prefix, "region": plan.dst_staging.region,
                        "dataflow_id": plan.dst_staging.dataflow_id},
        # 段级续跑标记 + 子 job 引用
        "sink_done": False, "cross_done": False, "preheat_done": False,
        "sink_job_id": "", "cross_job_id": "", "preheat_job_id": "",
        # 当前段进度透传
        "bytes_done": 0, "bytes_total": 0,
    }
    _save(job)
    return job


# ── 审批门（入口一次）───────────────────────────────────────────────────────

def estimate_source(plan: paths.Plan) -> tuple[int, bool]:
    """PFS 目录暂无现成测大小工具（transfer.estimate_source 只测对象存储）。
    返回 (字节, size_known)；恒 (0, False) → 上层 fail-safe 当作需审批。用户预估量在入口另行给。"""
    return 0, False


def needs_approval(bytes_total: int, size_known: bool = True) -> bool:
    """未知大小 → 恒需审批（fail-safe，照 ssh_transfer）；已知则按阈值。"""
    if not size_known:
        return True
    tb = float(getattr(settings, "PFS_TRANSFER_APPROVAL_TB", 1) or 1)
    return bytes_total > tb * (1024 ** 4)


# ── 每段建子 job（复用子 orchestrator）─────────────────────────────────────

def _sub_uri(staging: dict) -> str:
    return f"{staging['scheme']}://{staging['bucket']}/{staging['prefix']}"


def _make_sub(job: dict, stage: str):
    """按方向+段构造子 orchestrator 的 job。返回 (sub_orchestrator, sub_job)。"""
    direction = job["direction"]
    same = job.get("same_name", "")
    src, dst = job["src_pfs"], job["dst_pfs"]
    ss, ds = job["src_staging"], job["dst_staging"]
    oid = job.get("created_by", "")

    if stage == STAGE_SINKING:
        if direction == paths.DIRECTION_P1:      # vePFS Export → TOS
            plan = vepfs_o.make_plan(vepfs_o.OP_SINK, src["sub_path"], _sub_uri(ss),
                                     fs_id=src["fs_id"], region=ss["region"], same_name=same)
            return vepfs_o, vepfs_o.create_job_record(plan, open_id=oid)
        plan = cpfs_o.make_plan(cpfs_o.OP_SINK, src["sub_path"], _sub_uri(ss),      # CPFS Export → OSS
                                fs_id=src["fs_id"], region=ss["region"], data_flow_id=ss.get("dataflow_id", ""))
        return cpfs_o, cpfs_o.create_job_record(plan, open_id=oid)

    if stage == STAGE_CROSSING:                   # 对象存储 → 对象存储（跨云）
        plan = transfer_o.make_plan(_sub_uri(ss), _sub_uri(ds))
        return transfer_o, transfer_o.create_job_record(plan, open_id=oid)

    if stage == STAGE_PREHEATING:
        if direction == paths.DIRECTION_P1:       # OSS → CPFS Import
            plan = cpfs_o.make_plan(cpfs_o.OP_PREHEAT, dst["sub_path"], _sub_uri(ds),
                                    fs_id=dst["fs_id"], region=ds["region"], data_flow_id=ds.get("dataflow_id", ""))
            return cpfs_o, cpfs_o.create_job_record(plan, open_id=oid)
        plan = vepfs_o.make_plan(vepfs_o.OP_PREHEAT, dst["sub_path"], _sub_uri(ds),  # TOS → vePFS Import
                                 fs_id=dst["fs_id"], region=ds["region"], same_name=same)
        return vepfs_o, vepfs_o.create_job_record(plan, open_id=oid)

    raise PfsTransferError(f"未知段：{stage}")


def _stage_dispatch(stage: str, direction: str):
    """段 + 方向 → 该段用的子 orchestrator（refresh 时定位子 job 归属）。"""
    if stage == STAGE_CROSSING:
        return transfer_o
    p1 = direction == paths.DIRECTION_P1
    if stage == STAGE_SINKING:
        return vepfs_o if p1 else cpfs_o
    if stage == STAGE_PREHEATING:
        return cpfs_o if p1 else vepfs_o
    return None


def _propagate(job: dict, sub_job: dict, on_update) -> None:
    """把子 job 的进度透传到链 job 并落盘（不改链 stage）。"""
    bd = sub_job.get("bytes_done") or sub_job.get("exec_size") or 0
    bt = sub_job.get("bytes_total") or sub_job.get("total_size") or 0
    changed = (bd != job.get("bytes_done")) or (bt != job.get("bytes_total"))
    job["bytes_done"], job["bytes_total"] = bd, bt
    _save(job)
    if changed and on_update:
        try:
            on_update(job)
        except Exception:
            logger.warning("[PFS] on_update failed", exc_info=True)


def progress_line(job: dict) -> str:
    stage = job.get("stage", "")
    lbl = _STAGE_LABEL.get(stage, stage)
    bd, bt = job.get("bytes_done", 0), job.get("bytes_total", 0)
    if bt:
        pct = int(bd * 100 / bt) if bt else 0
        return f"{lbl}：{transfer_o.fmt_size(bd)}/{transfer_o.fmt_size(bt)}（{pct}%）"
    if bd:
        return f"{lbl}：已传 {transfer_o.fmt_size(bd)}"
    return f"{lbl}：进行中…"


# ── 驱动 ────────────────────────────────────────────────────────────────────

def _fail(job: dict, stage: str, err: str, on_update) -> dict:
    job["stage"] = STAGE_FAILED
    job["error"] = err or f"{_STAGE_LABEL.get(stage, stage)}失败"
    job["finished_ts"] = time.time()
    _save(job)
    if on_update:
        try:
            on_update(job)
        except Exception:
            pass
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 1440) -> dict:
    """顺序驱动三段至终态（供后台线程调用）。已 DONE 段跳过（段级续跑）。"""
    job["launched"] = True
    for stage, done_key, jid_key, _lbl in _STAGES:
        if job.get(done_key):
            continue
        job["stage"] = stage
        _save(job)
        if on_update:
            try:
                on_update(job)
            except Exception:
                pass
        try:
            sub_o, sub_job = _make_sub(job, stage)
        except Exception as e:
            return _fail(job, stage, f"{_STAGE_LABEL[stage]}构造失败：{e}", on_update)
        job[jid_key] = sub_job.get("job_id", "")
        _save(job)
        try:
            sub_job = sub_o.run_to_completion(
                sub_job, on_update=lambda sj: _propagate(job, sj, on_update),
                poll_interval=poll_interval, max_polls=max_polls)
        except Exception as e:
            return _fail(job, stage, f"{_STAGE_LABEL[stage]}执行异常：{e}", on_update)
        if sub_job.get("stage") != "DONE":
            return _fail(job, stage, sub_job.get("error") or f"{_STAGE_LABEL[stage]}未成功", on_update)
        job[done_key] = True
        job["bytes_done"] = job["bytes_total"] = 0   # 进入下一段重置进度
        _save(job)

    job["stage"] = STAGE_DONE
    job["finished_ts"] = time.time()
    _save(job)
    if on_update:
        try:
            on_update(job)
        except Exception:
            pass
    return job


def _resume_async(job: dict) -> None:
    """refresh 检测到当前段刚完成且仍有剩余段时，NX 锁保护下后台续跑（重启续跑；
    NX 防 refresh 与手动查询窄窗并发双起，照 ssh L6）。

    **安全守卫**：只对已 launched（=经 confirm 审批门下发过）的 job 续跑。否则"查询进度"文本
    （gate-free、无 admin 校验）会在一条 stage=NEW/launched=False 的未确认 job 上触发 refresh→
    续跑，完全绕过 confirm + 审批门。未 launched 一律不起跑。"""
    if not job.get("launched"):
        return
    if job.get("stage") in (STAGE_DONE, STAGE_FAILED):
        return
    if all(job.get(dk) for _s, dk, _j, _l in _STAGES):
        return
    try:
        ok = get_redis().set(f"{_LAUNCH_PREFIX}{job['job_id']}", "1", nx=True, ex=120)
    except Exception:
        ok = None
    if not ok:
        return
    def _bg():
        try:
            run_to_completion(get_job(job["job_id"]) or job)
        finally:
            try:
                get_redis().delete(f"{_LAUNCH_PREFIX}{job['job_id']}")
            except Exception:
                pass
    threading.Thread(target=_bg, daemon=True).start()


def refresh(job_id: str):
    """重启自愈：定位当前段 → refresh 该段子 job（单次 poll 云端）→ 反映到链；
    当前段刚 DONE 且有剩余段 → 后台续跑。返回最新链 job。"""
    job = get_job(job_id)
    if not job or job.get("stage") in (STAGE_DONE, STAGE_FAILED, None):
        return job
    stage = job.get("stage")
    # NEW 但已 launched → 尝试从第一个未完成段续跑
    if stage in (STAGE_NEW, None):
        _resume_async(job)
        return get_job(job_id)
    sub_o = _stage_dispatch(stage, job.get("direction", ""))
    key = {STAGE_SINKING: "sink", STAGE_CROSSING: "cross", STAGE_PREHEATING: "preheat"}.get(stage)
    if not sub_o or not key:
        return job
    sub_id = job.get(f"{key}_job_id")
    if not sub_id:
        return job
    try:
        sub_o.refresh(sub_id)
        sub_job = sub_o.get_job(sub_id)
    except Exception:
        sub_job = None
    if not sub_job:
        return job
    _propagate(job, sub_job, None)
    sst = sub_job.get("stage")
    if sst == "DONE":
        job[f"{key}_done"] = True
        _save(job)
        _resume_async(job)
    elif sst == "FAILED":
        _fail(job, stage, sub_job.get("error") or f"{_STAGE_LABEL[stage]}失败", None)
    return get_job(job_id)


# 供 cards / cli 复用（对齐 transfer）
fmt_size = transfer_o.fmt_size
fmt_ts = transfer_o.fmt_ts
fmt_duration = transfer_o.fmt_duration
