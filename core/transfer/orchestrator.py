"""
跨云迁移编排：状态机 + Redis 任务记录 + 轮询推进。

任务状态机（一期只有跨云段；三期前置沉降段）：
    NEW → CROSSING → DONE | FAILED
    (三期) NEW → SINKING → CROSSING → DONE | FAILED

Redis: transfer:job:{job_id}  存整段任务上下文，30 天 TTL。
幂等：job_id = hash(source, dest, 当天)，重复提交命中同一 job。
断点续跑：未完成 job 由 dsw_scheduler._transfer_loop 扫描续轮询（三期接入）。

凭证：源火山 TOS 的 access_id/secret 从 settings 读（静态 AK），与 TOS 工具一致。
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta

from config.settings import settings
from utils.redis_client import get_redis
from core.transfer import paths
from core.transfer.paths import PathError

logger = logging.getLogger(__name__)

_KEY_PREFIX = "transfer:job:"
_TTL = 30 * 86400        # 30 天

# 阶段
STAGE_NEW      = "NEW"
STAGE_SINKING  = "SINKING"
STAGE_CROSSING = "CROSSING"
STAGE_DONE     = "DONE"
STAGE_FAILED   = "FAILED"

# User-facing same-name policies. Keep skip as the safe default; map to
# provider-specific API knobs only at the engine boundary.
SAME_NAME_SKIP = "skip"
SAME_NAME_OVERWRITE = "overwrite"
_SAME_NAME_ALIASES = {
    "": SAME_NAME_SKIP,
    "skip": SAME_NAME_SKIP,
    "no": SAME_NAME_SKIP,
    "none": SAME_NAME_SKIP,
    "never": SAME_NAME_SKIP,
    "lastmodified": SAME_NAME_SKIP,
    "overwrite": SAME_NAME_OVERWRITE,
    "always": SAME_NAME_OVERWRITE,
    "force": SAME_NAME_OVERWRITE,
    "all": SAME_NAME_OVERWRITE,
}

_BJ = timezone(timedelta(hours=8))


def normalize_same_name_policy(value: str = "") -> str:
    """Normalize UI/CLI/API aliases to skip or overwrite."""
    return _SAME_NAME_ALIASES.get(str(value or "").strip().lower(), SAME_NAME_SKIP)


def same_name_policy_label(value: str = "") -> str:
    policy = normalize_same_name_policy(value)
    overwrite = "\u8986\u76d6\u540c\u540d\u6587\u4ef6"
    skip = "\u8df3\u8fc7\u540c\u540d\u6587\u4ef6"
    return overwrite if policy == SAME_NAME_OVERWRITE else skip


def _mgw_modes(policy: str) -> tuple[str, str]:
    """Return (transfer_mode, overwrite_mode) for Alibaba MGW."""
    if normalize_same_name_policy(policy) == SAME_NAME_OVERWRITE:
        return "all", "always"
    # Real-service fallback: MGW rejects overwrite_mode=never; lastmodified is
    # the closest supported non-full-copy behavior.
    return "lastmodified", "always"


def _bucket_map() -> dict:
    try:
        return json.loads(settings.TRANSFER_BUCKET_MAP_RAW or "{}")
    except Exception as e:
        logger.error("[Transfer] TRANSFER_BUCKET_MAP 解析失败: %s", e)
        return {}


def make_plan(source_raw: str, dest_raw: str = "") -> paths.Plan:
    """解析路径为迁移计划（薄封装，统一注入 bucket_map）。抛 PathError。"""
    return paths.build_plan(source_raw, dest_raw, bucket_map=_bucket_map())


def _job_id(plan: paths.Plan) -> str:
    day = datetime.now(_BJ).strftime("%Y%m%d")
    raw = f"{plan.source.uri()}|{plan.dest.uri()}|{day}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"tr-{h}"


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
        logger.warning("[Transfer] 写 Redis 失败 job=%s", job.get("job_id"))


def estimate_source(plan: paths.Plan) -> tuple[int, int]:
    """探测源目录大小/对象数，用于确认卡回显与审批阈值判断。失败返回 (0,0)。

    源 TOS：复用 tools.volcano.tos._prefix_size；
    源 OSS：复用 tools.aliyun.oss._prefix_size（OSS→TOS 方向）。
    全闪源（三期）暂不探测。
    """
    src = plan.source if plan.source.is_object else plan.sink_target
    if src is None:
        return 0, 0
    try:
        if src.scheme == "tos":
            from utils.volcano_client_factory import get_tos_client
            from tools.volcano.tos import _prefix_size
            client = get_tos_client()
            if client is None:
                return 0, 0
            try:
                return _prefix_size(client, src.bucket, src.prefix)
            finally:
                client.close()
        elif src.scheme == "oss":
            from utils.aliyun_client_factory import get_oss_bucket
            from tools.aliyun.oss import _prefix_size as oss_prefix_size
            region = settings.TRANSFER_OSS_REGION or "cn-hangzhou"
            bucket_obj = get_oss_bucket("", src.bucket, region)
            if bucket_obj is None:
                return 0, 0
            return oss_prefix_size(bucket_obj, src.prefix)
    except Exception:
        logger.warning("[Transfer] 源大小探测失败", exc_info=True)
    return 0, 0


def needs_approval(bytes_total: int) -> bool:
    """超过 TRANSFER_APPROVAL_TB 阈值需管理员审批。"""
    tb = bytes_total / (1024 ** 4)
    return tb > settings.TRANSFER_APPROVAL_TB


# 同名文件优先跳过、不覆盖——硬安全约束。
# 真机探测（非文档）：服务端只接受 overwrite_mode=always；"跳过同名"靠 transfer_mode 区分：
#   lastmodified + always = 增量，同名且未变化的跳过（= 优先跳过同名，安全默认）
#   all + always          = 全量强制覆盖（禁用）
# 文档称 never+all=不覆盖，但服务端实际拒收 never，故用 lastmodified+always 实现"不覆盖"。
OVERWRITE_MODE = "always"        # 服务端唯一接受值
TRANSFER_MODE = "lastmodified"   # 增量：同名未变则跳过，绝不全量覆盖


def create_job_record(plan: paths.Plan, *, open_id: str = "",
                      same_name_policy: str = "", overwrite_mode: str = "",
                      bytes_total: int = 0, objects_total: int = 0) -> dict:
    """落库一条任务记录（幂等：同 job_id 已存在则返回旧记录）。

    overwrite_mode 参数保留签名兼容，但一律强制 no（永不覆盖），忽略传入值。
    """
    policy = normalize_same_name_policy(same_name_policy or overwrite_mode)
    transfer_mode, overwrite_mode_actual = _mgw_modes(policy)
    job_id = _job_id(plan)
    existing = get_job(job_id)
    if existing and existing.get("stage") not in (STAGE_FAILED,):
        if existing.get("stage") == STAGE_NEW:
            existing["same_name_policy"] = policy
            existing["transfer_mode"] = transfer_mode
            existing["overwrite_mode"] = overwrite_mode_actual
            _save(existing)
        return existing
    job = {
        "job_id": job_id,
        "source": plan.source.uri(),
        "dest": plan.dest.uri(),
        "direction": plan.direction,
        "engine": plan.engine,
        "needs_sink": plan.needs_sink,
        "same_name_policy": policy,
        "transfer_mode": transfer_mode,
        "overwrite_mode": overwrite_mode_actual,
        "stage": STAGE_NEW,
        "sink_task_id": "",
        "cross_job_name": "",
        "bytes_total": bytes_total,
        "objects_total": objects_total,
        "created_by": open_id,
        "created_ts": time.time(),
        "updated_ts": time.time(),
        "error": "",
    }
    _save(job)
    return job


def set_same_name_policy(job: dict, policy: str) -> dict:
    """Persist the user-selected same-name policy before launching."""
    normalized = normalize_same_name_policy(policy)
    transfer_mode, overwrite_mode_actual = _mgw_modes(normalized)
    job["same_name_policy"] = normalized
    job["transfer_mode"] = transfer_mode
    job["overwrite_mode"] = overwrite_mode_actual
    _save(job)
    return job


def start_cross(job: dict) -> dict:
    """启动跨云段，按引擎分派。提交任务并置 CROSSING。失败置 FAILED。

    engine=mgw     → 阿里在线迁移，进 OSS（一期，真机验证）
    engine=tos_mig → 火山 DMS，进 TOS（二期，字段待真机校准）
    """
    engine = job["engine"]
    if engine not in ("mgw", "tos_mig"):
        job["stage"] = STAGE_FAILED
        job["error"] = f"跨云引擎 {engine} 尚未实现"
        _save(job)
        return job

    policy = normalize_same_name_policy(
        job.get("same_name_policy") or job.get("overwrite_policy") or job.get("overwrite_mode")
    )
    transfer_mode, overwrite_mode_actual = _mgw_modes(policy)
    job["same_name_policy"] = policy
    job["transfer_mode"] = transfer_mode
    job["overwrite_mode"] = overwrite_mode_actual
    plan = make_plan(job["source"], job["dest"])
    src = plan.source if plan.source.is_object else plan.sink_target
    cross_job_name = job["job_id"]
    try:
        if engine == "mgw":
            from core.transfer import engine_mgw
            cross_job_name = engine_mgw.submit_cross_job(
                job_name=job["job_id"],
                src_bucket=src.bucket, src_prefix=src.prefix,
                # 迁移专用 TOS 凭证/区域，留空回退容量巡检的 TOS_*
                src_access_id=settings.TRANSFER_TOS_ACCESS_KEY or settings.TOS_ACCESS_KEY,
                src_access_secret=settings.TRANSFER_TOS_SECRET_KEY or settings.TOS_SECRET_KEY,
                src_region=settings.TRANSFER_TOS_REGION or settings.TOS_REGION,
                dest_bucket=plan.dest.bucket, dest_prefix=plan.dest.prefix,
                dest_region=settings.TRANSFER_OSS_REGION or "cn-hangzhou",
                dest_internal=settings.TRANSFER_OSS_INTERNAL,
                transfer_mode=transfer_mode,
                overwrite_mode=overwrite_mode_actual,
                open_id=job.get("created_by", ""),
            )
        else:   # tos_mig：OSS→TOS，源=阿里 OSS，目的=火山 TOS
            from core.transfer import engine_tos
            cross_job_name = engine_tos.submit_cross_job(
                job_name=job["job_id"],
                src_bucket=src.bucket, src_prefix=src.prefix,
                src_access_id=settings.PAI_DSW_ACCESS_KEY_ID,
                src_access_secret=settings.PAI_DSW_ACCESS_KEY_SECRET,
                src_region=settings.TRANSFER_OSS_REGION or "cn-hangzhou",
                dest_bucket=plan.dest.bucket, dest_prefix=plan.dest.prefix,
                dest_region=settings.TRANSFER_TOS_REGION or settings.TOS_REGION,
                transfer_mode=transfer_mode,
                overwrite_mode=policy,
                open_id=job.get("created_by", ""),
            )
        job["cross_job_name"] = cross_job_name or job["job_id"]
        job["stage"] = STAGE_CROSSING
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = str(e)
        job["finished_ts"] = time.time()
        logger.error("[Transfer] 跨云段启动失败 job=%s", job["job_id"], exc_info=True)
    _save(job)
    return job


def start_sinking(job: dict) -> dict:
    """全闪源沉降到同厂对象存储（三期，目前仅支持阿里 CPFS→OSS）。置 SINKING。

    复用 core.cpfs_dataflow.engine_nas（NAS DataFlow Export）。vepfs→tos（火山）尚未实现。
    """
    plan = make_plan(job["source"], job["dest"])
    if not plan.needs_sink:
        return job
    src = plan.source
    if src.scheme != "cpfs":
        job["stage"] = STAGE_FAILED
        job["error"] = f"{src.scheme} 沉降暂未实现（目前仅支持阿里 CPFS→OSS）"
        job["finished_ts"] = time.time()
        _save(job)
        return job
    try:
        from core.cpfs_dataflow import engine_nas
        sink_ref = engine_nas.submit_sink_job(
            fs_id=src.bucket, cpfs_dir=src.prefix,
            oss_bucket=plan.sink_target.bucket, oss_prefix=plan.sink_target.prefix,
            region=settings.CPFS_REGION, open_id=job.get("created_by", ""),
        )
        job["sink_task_id"] = sink_ref
        job["stage"] = STAGE_SINKING
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = f"沉降启动失败：{e}"
        job["finished_ts"] = time.time()
        logger.error("[Transfer] 沉降启动失败 job=%s", job["job_id"], exc_info=True)
    _save(job)
    return job


def poll_sink_once(job: dict) -> dict:
    """轮询一次沉降任务（SINKING）。完成置 sink_done=True；失败置 FAILED。"""
    if job["stage"] != STAGE_SINKING:
        return job
    from core.cpfs_dataflow import engine_nas
    plan = make_plan(job["source"], job["dest"])
    st = engine_nas.poll_sink(plan.source.bucket, job.get("sink_task_id", ""),
                              region=settings.CPFS_REGION, open_id=job.get("created_by", ""))
    if st.get("bytes"):
        job["bytes_total"] = st["bytes"]
    if st.get("objects"):
        job["objects_total"] = st["objects"]
    status = st.get("status", "")
    if status in engine_nas._DONE_STATES:
        job["sink_done"] = True
    elif status in engine_nas._FAIL_STATES:
        job["stage"] = STAGE_FAILED
        job["error"] = st.get("error", "") or "沉降失败"
        job["finished_ts"] = time.time()
    _save(job)
    return job


def poll_once(job: dict) -> dict:
    """轮询一次跨云任务，更新阶段。按引擎分派 poll/终态判定。返回更新后的 job。"""
    if job["stage"] != STAGE_CROSSING:
        return job
    if job["engine"] == "tos_mig":
        from core.transfer import engine_tos as eng
        done_states, fail_states = eng._DONE_STATES, eng._FAIL_STATES
        st = eng.poll_job(job["cross_job_name"], job.get("created_by", ""),
                          dest_region=settings.TRANSFER_TOS_REGION or settings.TOS_REGION or "cn-shanghai")
    else:
        from core.transfer import engine_mgw as eng
        done_states = {eng.STATUS_FINISHED}
        fail_states = {eng.STATUS_INTERRUPTED}
        st = eng.poll_job(job["cross_job_name"], job.get("created_by", ""))
    status = st.get("status", "")
    if st.get("bytes"):
        job["bytes_total"] = st["bytes"]
    if st.get("objects"):
        job["objects_total"] = st["objects"]
    if status in done_states:
        job["stage"] = STAGE_DONE
        job["finished_ts"] = time.time()
    elif status in fail_states:
        job["stage"] = STAGE_FAILED
        job["finished_ts"] = time.time()
        job["error"] = st.get("error", "") or "任务中断"
    _save(job)
    return job


def refresh(job_id: str):
    """查进度前实时重查云端：后台轮询线程会随容器重启而死，只读 Redis 会停在旧 CROSSING。
    stage 仍 CROSSING 且有 cross_job_name → poll_once 重查并落库（点“查询这条”即可自愈）。"""
    job = get_job(job_id)
    if job and job.get("stage") == STAGE_CROSSING and job.get("cross_job_name"):
        try:
            job = poll_once(job)
        except Exception:
            pass
    return job


def run_to_completion(job: dict, *, on_update=None, poll_interval: int = 60,
                      max_polls: int = 1440) -> dict:
    """启动跨云段并阻塞轮询至终态（供后台线程调用）。

    on_update(job) 在每次阶段变化时回调（推送进度卡）。
    max_polls*poll_interval 默认上限 24h。
    三期：全闪源先沉降（SINKING）到同厂对象存储，完成后再跨云（CROSSING）。
    """
    # ── 三期前置：全闪源沉降 ──────────────────────────────────────────────
    if job.get("needs_sink") and not job.get("sink_done") and job["stage"] in (STAGE_NEW, STAGE_FAILED):
        job = start_sinking(job)
        if on_update:
            on_update(job)
        if job["stage"] == STAGE_FAILED:
            return job
        for _ in range(max_polls):
            time.sleep(poll_interval)
            job = poll_sink_once(job)
            if job["stage"] == STAGE_FAILED:
                if on_update:
                    on_update(job)
                return job
            if job.get("sink_done"):
                break
        else:
            job["stage"] = STAGE_FAILED
            job["error"] = f"沉降轮询超时（>{max_polls * poll_interval // 3600}h）"
            job["finished_ts"] = time.time()
            _save(job)
            if on_update:
                on_update(job)
            return job

    job = start_cross(job)
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


def fmt_size(num_bytes: int) -> str:
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_ts(ts: float) -> str:
    """epoch 秒 → 北京时间 'YYYY-MM-DD HH:MM:SS'。无值返回 '-'。"""
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, _BJ).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(start_ts: float, end_ts: float) -> str:
    """两个 epoch 秒之间的耗时,人类可读(如 '2分13秒')。"""
    if not start_ts or not end_ts or end_ts < start_ts:
        return "-"
    sec = int(end_ts - start_ts)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"
