"""九章迁移编排：单段状态机 + Redis 记录 + 轮询推进。

    NEW → PULLING → VERIFYING → DONE | FAILED

比泰国那条（两段）简单：单跳直连，没有中转、没有段间切换、没有段下发锁的竞态窗口。
Redis `jz:transfer:job:{id}`，30 天 TTL，job 前缀 `jz-`，当天幂等。

刻意复用 `core.ssh_transfer.paths`（安全边界，防注入白名单）——那部分只能有一份实现。
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from config.settings import settings
from utils.redis_client import get_redis

from core.ssh_transfer import paths          # 安全白名单：复用，绝不复制
from . import engine

logger = logging.getLogger(__name__)

_KEY_PREFIX = "jz:transfer:job:"
_LAUNCH_PREFIX = "jz:transfer:launch:"
_TTL = 30 * 86400
_BJ = timezone(timedelta(hours=8))

STAGE_NEW = "NEW"
STAGE_PULLING = "PULLING"
STAGE_VERIFYING = "VERIFYING"
STAGE_DONE = "DONE"
STAGE_FAILED = "FAILED"
_ACTIVE = (STAGE_PULLING, STAGE_VERIFYING)


class JiuzhangError(engine.JiuzhangError):
    """向后兼容的别名，便于上层只 import 本模块。"""


def build_plan(source_raw: str, dest_subdir: str = ""):
    """复用泰国链的路径解析 —— 同样的 `oss://桶/前缀/` 语义、同样的严格白名单。"""
    return paths.build_plan(source_raw, dest_subdir)


def _job_id(plan) -> str:
    day = datetime.now(_BJ).strftime("%Y%m%d")
    raw = f"{plan.source_uri()}|{plan.dest_rel()}|{day}"
    return "jz-" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def _key(job_id: str) -> str:
    return _KEY_PREFIX + job_id


def get_job(job_id: str):
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
        logger.warning("[JZ] 写 Redis 失败 job=%s", job.get("job_id"))


def create_job_record(plan, *, open_id: str = "") -> dict:
    """建记录。已存在且非 FAILED 则原样返回（幂等：同一天同一对路径不会起两个任务）。"""
    jid = _job_id(plan)
    existing = get_job(jid)
    if existing and existing.get("stage") not in (None, STAGE_FAILED):
        return existing
    job = {
        "job_id": jid,
        "source_bucket": plan.source_bucket,
        "source_prefix": plan.source_prefix,
        "dest_rel": plan.dest_rel(),
        "dest_dir": engine.dest_dir(plan.source_prefix, plan.dest_rel()),
        "stage": STAGE_NEW,
        "created_ts": time.time(),
        "created_by": open_id,
        "launched": False,
        "notified": False,
        "bytes_total": 0, "bytes_done": 0, "objects_total": 0,
        "estimate_ok": True,
        "error": "", "error_detail": "",
    }
    _save(job)
    return job


def estimate_source(plan):
    try:
        # 与泰国链共用同一条 API 估算路径（见 tools.aliyun.oss.estimate_prefix 的说明）。
        # engine.estimate_source（远端 ossutil du）保留作为回退，未接线。
        from tools.aliyun.oss import estimate_prefix
        return estimate_prefix(plan.source_bucket, plan.source_prefix,
                               endpoint=getattr(settings, "JIUZHANG_OSS_ENDPOINT", ""),
                               max_seconds=int(getattr(settings, "JIUZHANG_ESTIMATE_TIMEOUT", 240) or 240))
    except Exception:
        logger.warning("[JZ] 估算失败 %s", plan.source_uri(), exc_info=True)
        return 0, 0, False


def needs_approval(bytes_total: int, size_known: bool = True) -> bool:
    """超阈值需管理员。**估算失败 fail-safe 当作需审批** —— 不放行未知大小的迁移。"""
    if not size_known:
        return True
    try:
        tb = float(getattr(settings, "JIUZHANG_APPROVAL_TB", 0) or
                   getattr(settings, "SSH_TRANSFER_APPROVAL_TB", 1) or 1)
    except (TypeError, ValueError):
        tb = 1.0
    return bytes_total > tb * (1024 ** 4)


def _claim_launch(job_id: str) -> bool:
    """下发 NX 锁：卡片回调会重复投递，没有它两个并发回调会各起一个 ossutil。"""
    try:
        return bool(get_redis().set(f"{_LAUNCH_PREFIX}{job_id}", 1, nx=True, ex=180))
    except Exception:
        return True     # Redis 不可用时放行：宁可偶尔重复下发，也别完全起不来


def progress_line(job: dict) -> str:
    stage = job.get("stage", "?")
    if stage in (STAGE_DONE, STAGE_FAILED):
        return f"{stage} {job.get('error', '')}".strip()
    if stage == STAGE_VERIFYING:
        return "正在做端到端校验…"
    done, total = job.get("bytes_done") or 0, job.get("bytes_total") or 0
    spd = job.get("speed_bps")
    parts = [f"已传 {_fmt(done)}"]
    if total:
        parts.append(f"/ {_fmt(total)}（{done / total * 100:.1f}%）")
    if spd:
        parts.append(f"@ {_fmt(spd)}/s")
    return " ".join(parts)


def _fmt(n) -> str:
    n = float(n or 0)
    for u in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if n < 1024 or u == "PiB":
            return f"{n:.2f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} PiB"


def poll_once(job: dict) -> dict:
    """查一次并推进。不重新提交任务。"""
    if job.get("stage") == STAGE_VERIFYING:
        return _do_verify(job)
    st = engine.poll(job["job_id"])
    if st["status"] == "RUNNING":
        p = engine.progress(job["job_id"])
        for k_src, k_dst in (("bytes_done", "bytes_done"), ("objects_done", "objects_done"),
                             ("speed_bps", "speed_bps")):
            if p.get(k_src) is not None:
                job[k_dst] = p[k_src]
        job["stage"] = STAGE_PULLING
        return job
    if st["status"] == "FAILED":
        job["stage"] = STAGE_FAILED
        job["error"] = st.get("error") or "拉取失败"
        job["error_detail"] = engine.failure_detail(job["job_id"])
        job["finished_ts"] = time.time()
        return job
    # DONE
    job["stage"] = STAGE_VERIFYING if _verify_enabled() else STAGE_DONE
    if job["stage"] == STAGE_DONE:
        job["finished_ts"] = time.time()
        job["verify_summary"] = "（已按配置跳过校验）"
    return job


def _verify_enabled() -> bool:
    return str(getattr(settings, "JIUZHANG_VERIFY", "true")).strip().lower() != "false"


def _do_verify(job: dict) -> dict:
    """端到端校验。**fail-closed**：不过、或校验本身崩掉，都判 FAILED。

    并发闸门：一趟校验几分钟到几十分钟、期间不刷 updated_ts，对账会判失联再跑一次；
    多份 job dict last-write-wins 可能把 FAILED 覆盖成 DONE（真 fail-open）。
    抢不到锁 → 保持 VERIFYING、不给任何结论。
    """
    try:
        r = get_redis()
        if not r.set(f"jz:transfer:verify:{job['job_id']}", 1, nx=True, ex=7200):
            logger.info("[JZ] %s 校验已在进行，本轮不给结论", job["job_id"])
            return job
    except Exception:
        pass
    from . import verify
    try:
        res = verify.verify_pull(job)
    except Exception as e:
        logger.error("[JZ] %s 校验异常", job["job_id"], exc_info=True)
        job["stage"] = STAGE_FAILED
        job["error"] = f"校验没能完成，本次不给通过结论：{e}"
        job["finished_ts"] = time.time()
        return job
    job["verify"] = res
    job["verify_summary"] = res.get("summary", "")
    job["finished_ts"] = time.time()
    job["stage"] = STAGE_DONE if res.get("passed") else STAGE_FAILED
    if not res.get("passed"):
        job["error"] = ("校验环境有问题（不是数据不一致），修好后重跑校验即可。"
                        if res.get("env_issue") else
                        "端到端校验未通过——数据可能不完整，先别清理源数据。")
    return job


def refresh(job_id: str):
    """**只轮询、不重新提交**。查询进度与对账共用。"""
    job = get_job(job_id)
    if not job or job.get("stage") in (None, STAGE_DONE, STAGE_FAILED):
        return job
    if not job.get("launched"):
        return job          # 没确认过的任务不能被「查进度」带起来跑（审批门守卫）
    try:
        job = poll_once(job)
    except Exception as e:
        logger.warning("[JZ] refresh 查不到状态(保持在途) job=%s: %s", job_id, e)
        return job
    _save(job)
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 10080) -> dict:
    """起任务 + 阻塞轮询到终态。**在后台线程里调。**

    max_polls 默认 7 天（同泰国链）：大数据集的拉取可能几十小时，24h 上限会在任务
    正常运行时误判超时。真失败仍由 rc marker 立刻判定，不依赖这个上限。
    """
    job["launched"] = True
    if job.get("stage") in (None, STAGE_NEW, STAGE_FAILED):
        if not _claim_launch(job["job_id"]):
            logger.info("[JZ] %s 已有并发下发，跳过", job["job_id"])
        else:
            try:
                engine.start_pull(job["job_id"], source_bucket=job["source_bucket"],
                                  source_prefix=job["source_prefix"],
                                  dest_rel=job.get("dest_rel", ""))
            except Exception as e:
                job["stage"] = STAGE_FAILED
                job["error"] = str(e)
                job["error_detail"] = ""
                job["finished_ts"] = time.time()
                _save(job)
                _notify(job, on_update)
                return job
        job["stage"] = STAGE_PULLING
        job["started_ts"] = time.time()
        _save(job)
        _notify(job, on_update)

    for _ in range(max_polls):
        time.sleep(poll_interval)
        try:
            job = poll_once(job)
        except Exception as e:
            logger.warning("[JZ] 轮询失败(将重试) job=%s: %s", job.get("job_id"), e)
            continue
        _save(job)
        _notify(job, on_update)
        if job.get("stage") in (STAGE_DONE, STAGE_FAILED):
            return job

    job["stage"] = STAGE_FAILED
    job["error"] = (f"轮询超时（>{max_polls * poll_interval // 3600}h）——"
                    f"任务可能仍在九章上运行，确认后再决定是否重试。")
    _save(job)
    _notify(job, on_update)
    return job


def _notify(job, on_update):
    if not on_update:
        return
    try:
        on_update(job)
    except Exception:
        logger.warning("[JZ] on_update 回调异常 job=%s", job.get("job_id"), exc_info=True)
