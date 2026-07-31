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
# verify 顶层 import：三个模块都不 import orchestrator，不存在循环依赖（我原先写的
# 「避开循环依赖」注释是错的）。放顶层还有个好处 —— import 错误在启动时就暴露，
# 而不是等段2 跑完几十小时才在 poll_once 里炸出来、把 job 卡在非终态。
from core.ssh_transfer import paths, engine_ssh, engine_ossutil, verify as _verify
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

def _stage2_direct() -> bool:
    """段2 是否走「泰国 ossutil 直拉」（默认是；设 SSH_STAGE2_MODE=rsync 回退旧的 SGP 转发）。"""
    return str(getattr(settings, "SSH_STAGE2_MODE", "ossutil")).strip().lower() != "rsync"


STAGE2_MODE_OSSUTIL = "ossutil"
STAGE2_MODE_RSYNC = "rsync"
# 人工接管时写进 job["handoff"]["engine"] 的标识（见下方 stage2_mode_of 的说明）
_HANDOFF_OSSUTIL = "ossutil_thai_direct"


def stage2_mode_of(job: dict) -> str:
    """判定这个 job 的段2 **实际**跑在哪个引擎上。

    优先级刻意如此：
    1. `job["stage2_mode"]` —— 由 `_start_stage` 写入，是最权威的事实。
    2. `job["handoff"]["engine"]` —— **人工接管**留下的标记。2026-07-31 切换那天，我先手工在
       泰国起了 ossutil、再写这套代码；那一单的记录里只有 handoff、没有 stage2_mode。不认它的
       话，新代码会拿 rsync 探针去查 SGP 上根本不存在的 marker → 误判「进程异常退出」→
       把一个正常跑着的 19 小时任务标成失败、还可能触发重传 19.5TiB。
    3. 都没有 → rsync。**绝不能默认 ossutil**：本次改动之前建的所有 job 都是 rsync 跑的，
       默认成 ossutil 会把它们全部误判失败。

    注意：**不按当前 `SSH_STAGE2_MODE` 配置分派**。任务起来后有人改配置（或容器 recreate
    读到新值），按配置分派就会拿错引擎的探针去查 marker，同样是误判失败。
    """
    mode = str(job.get("stage2_mode") or "").strip().lower()
    if mode in (STAGE2_MODE_OSSUTIL, STAGE2_MODE_RSYNC):
        return mode
    handoff = job.get("handoff")
    if isinstance(handoff, dict) and str(handoff.get("engine") or "") == _HANDOFF_OSSUTIL:
        return STAGE2_MODE_OSSUTIL
    return STAGE2_MODE_RSYNC


def _stage2_engine_of(job: dict):
    """按 stage2_mode_of 的判定取引擎模块。"""
    return engine_ossutil if stage2_mode_of(job) == STAGE2_MODE_OSSUTIL else engine_ssh


def _claim_stage_launch(job_id: str, stage: str) -> bool:
    """抢「起某一段」的下发权（Redis NX）。抢不到说明别人正在起，本次必须放手。

    为什么必须有：段1 rc 落盘后、到 `stage=STAGE2` 写回 Redis 之前有个最长 60s 的窗口
    （在线线程 60s 才轮一次）。这期间任何一次 refresh()（用户发「查询进度」、点按钮、
    或在线线程死后对账兜底）都会独立跑一遍 poll_once → 看到 rc=0 → 再起一次段2。
    并行 rsync 之前这是良性的（两条 rsync 幂等自纠），但现在段2 起手会先清 unit marker：
    第二次下发的清理会删掉第一次那 50 个 rc → 第一次 verify 时看到「有 50 个分片没留下
    退出码」→ **明明全传完了却报失败**，19.5TB 白跑几天。且 16 条 ssh 并发握手会撞破
    泰国 sshd 的 MaxStartups 10 → 随机 rc=255。
    Redis 不可用时**放行**（返回 True）：宁可退回改动前的良性重复，也不能因为缓存挂了
    就永久起不了段2。
    """
    try:
        r = get_redis()
        if r is None:
            return True
        return bool(r.set(f"ssh:transfer:stagelaunch:{job_id}:{stage}", 1, nx=True, ex=180))
    except Exception:
        logger.warning("[SSHT] 抢 %s 下发锁失败（放行）job=%s", stage, job_id, exc_info=True)
        return True


def _start_stage(job: dict, stage: str) -> None:
    if not _claim_stage_launch(job["job_id"], stage):
        # 别人正在起这一段。**不要写 Redis**：赢家会把 stage 写回去，这里擅自改会覆盖它。
        # 但要把赢家已落库的状态**读回本地 job dict** —— 否则调用方（run_to_completion）
        # 手里这份永远停在旧 stage，轮询每轮空转，7 天后用这份陈旧内存对象写 FAILED，
        # 把赢家的 DONE 覆盖掉（「成功任务 7 天后变失败」）。
        logger.info("[SSHT] %s 已有并发下发在进行，本次跳过 job=%s", stage, job.get("job_id"))
        try:
            fresh = get_job(job["job_id"])
            if fresh:
                job.update(fresh)
        except Exception:
            logger.warning("[SSHT] 下发锁输家回读 job 失败 job=%s", job.get("job_id"), exc_info=True)
        return
    try:
        if stage == STAGE_STAGE1:
            engine_ssh.start_stage1(job["job_id"], source_bucket=job["source_bucket"],
                                    source_prefix=job["source_prefix"])
        elif _stage2_direct():
            engine_ossutil.start_stage2(job["job_id"], source_prefix=job["source_prefix"],
                                        dest_rel=job.get("dest_rel", ""))
            job["stage2_mode"] = STAGE2_MODE_OSSUTIL   # 记进 job：查/重试/排障都要知道它在哪台机上
        else:
            engine_ssh.start_stage2(job["job_id"], source_prefix=job["source_prefix"],
                                    dest_rel=job.get("dest_rel", ""))
            job["stage2_mode"] = STAGE2_MODE_RSYNC
        job["stage"] = stage
        job["error"] = ""
    except Exception as e:
        job["stage"] = STAGE_FAILED
        job["error"] = str(e)
        # 起任务就挂时没有新日志可摘，必须清掉上一轮的明细：否则「起 stage1 失败(rc=1)」会配上
        # 上一轮那条「失败对象 42366 个 / ossfs2 单分片说明」，自相矛盾、把排障带偏。
        job["error_detail"] = ""
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
        if eng_stage == STAGE2 and _stage2_engine_of(job) is engine_ossutil:
            prog = engine_ossutil.stage_progress(job["job_id"])
        else:
            prog = engine_ssh.stage_progress(job["job_id"], eng_stage)
    except Exception:
        return
    now = time.time()
    bd, spd = prog.get("bytes_done"), prog.get("speed_bps")
    if bd is not None:
        prev_bd, prev_ts = job.get("_bd_sample"), job.get("_bd_ts")
        if spd is None and prev_bd is not None and prev_ts and now > prev_ts and bd >= prev_bd:
            spd = int((bd - prev_bd) / (now - prev_ts))
        elif prev_bd is not None and bd < prev_bd:
            # 已传字节回退 = 重跑/切换了执行方式，上一轮的速率已无意义。不清的话它会被拿去算
            # ETA，给出一个凭空的「剩余 3h」。清成 0，下一轮采样自然会重新算出真实速率。
            job["speed_bps"] = 0
        job["bytes_done"] = bd
        job["_bd_sample"] = bd
        job["_bd_ts"] = now
        # 直接覆盖（含 None）：并行段2 恒不给 pct，若只在非 None 时写，曾单流跑过的 job 会一直
        # 挂着上一轮的陈旧百分比，而 progress_line 优先用它 → 卡片上的进度永久定格在旧值。
        job["pct"] = prog.get("pct")
    if spd is not None:
        job["speed_bps"] = spd
    # 并行段2 的分片计数：done 与 total 的差额是「有分片没跑完」的唯一可见信号，收尾时一眼看出。
    for k in ("units_total", "units_done"):
        if prog.get(k) is not None:
            job[k] = prog[k]


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
    # 并行段2：把分片完成数摆出来。收尾时 done<total 就是「有分片没跑完」的直接证据，
    # 不然这种情况在卡片上完全看不出来。
    ut, ud = job.get("units_total"), job.get("units_done")
    if ut:
        parts.append(f"分片 {ud or 0}/{ut}")
        # 分片全完成后还没到终态 = 正在跑收尾全量核对。那一趟是纯元数据扫描（7.5 万文件
        # 走一遍 FUSE），字节数不再增长、可能几十分钟不动 —— 不标出来就像卡死了，会被误 kill。
        if (ud or 0) >= ut and job.get("stage") in _ACTIVE:
            parts.append("收尾核对中")
    return " · ".join(parts) if parts else "进度采集中…"


def _claim_verify(job_id: str) -> bool:
    """抢「跑校验」的独占权（NX, TTL 2h）。抢不到 → 本次不给任何结论。

    为什么必须有：一趟校验要列 7.5 万对象 + 泰国 find 7.5 万文件 + 5 次整对象重下 cmp，
    几分钟到几十分钟；期间**不刷 updated_ts**，180s 后对账就判本 job 失联 → refresh →
    poll_once → 又见 DONE → **再跑一次校验**；用户查询/点按钮还能起第三次。多份 job dict
    各写各的，last-write-wins —— 一份 passed=False、一份 True 时，可能把 FAILED 覆盖成
    DONE。**那才是真正的 fail-open**（校验本来就是为了防这个）。
    TTL 给足 2h：宁可持有者崩了之后卡 2h（job 留在 STAGE2、之后自然重试），
    也不能让第二个校验并行进来。
    """
    try:
        r = get_redis()
        if r is None:
            return True          # 同 _claim_stage_launch：缓存挂了不能让链路永久停住
        return bool(r.set(f"ssh:transfer:verify:{job_id}", 1, nx=True, ex=7200))
    except Exception:
        logger.warning("[SSHT] 抢校验锁失败（放行）job=%s", job_id, exc_info=True)
        return True


def _release_verify(job_id: str) -> None:
    try:
        r = get_redis()
        if r is not None:
            r.delete(f"ssh:transfer:verify:{job_id}")
    except Exception:
        pass


def _run_stage2_verify(job: dict) -> bool:
    """段2 传输器报成功后跑端到端校验。

    返回值是**给调用方的三态信号**，必须用它决定要不要晋级 DONE：
      True  = 本次给出了结论（校验通过 → 调用方可判 DONE；不过 → 本函数已置 FAILED）
      False = **本次不给结论**（校验被别人占着），调用方必须保持在途、绝不能判 DONE

    为什么必须返回 bool：只看 `stage != FAILED` 的话，抢不到校验锁的那一方（stage 仍是
    STAGE2）会被当成「校验通过」→ 在**零校验**的情况下宣布 DONE 并推成功卡。那正是校验层
    要防的 fail-open，只是从「两份结论互相覆盖」换成了「输家跳过校验宣布成功」，而且更确定
    会发生：对账的 180s stale 门 + 一趟校验十几分钟 ⇒ 输家几乎必然出现。

    为什么不能只信传输器的退出码：21TB 迁移最坏的结局不是失败，而是「报成功但少数据/内容
    截断」—— 运维毫无察觉，几个月后训练读到坏文件才发现，那时源可能已经清理了。
    校验自身出错（列不到清单、SSH 挂）也判失败：这时候「不知道对不对」必须当作「不对」，
    fail-open 会把一个可能缺数据的迁移标成成功，正是要防的那种事故。
    """
    jid = job.get("job_id", "")
    if not getattr(settings, "SSH_STAGE2_VERIFY", True):
        logger.warning("[SSHT] %s 已按配置跳过端到端校验（SSH_STAGE2_VERIFY=false）", jid)
        job["verify"] = {"passed": None, "summary": "⚠ 本次未做端到端校验（SSH_STAGE2_VERIFY=false）"}
        return True                 # 明确放行（配置说了不校验），算给了结论
    if not _claim_verify(jid):
        # 别人正在校验。结论只能由持锁那一方给出。这里回读赢家已落库的状态（否则调用方
        # 末尾的 _save 会用我们这份陈旧 stage 覆盖掉赢家的 DONE/FAILED —— 与 _start_stage
        # 输家分支同一个坑），然后返回 False 让调用方保持在途。
        logger.info("[SSHT] %s 校验已在进行，本次跳过（保持在途）", jid)
        try:
            fresh = get_job(jid)
            if fresh:
                job.update(fresh)
        except Exception:
            logger.warning("[SSHT] %s 校验锁输家回读失败", jid, exc_info=True)
        # 赢家已经落了结论 → 直接采用，省掉下一轮整趟重跑（一趟 10~30 分钟）。
        if isinstance(job.get("verify"), dict) and job["verify"].get("passed") is not None:
            return True
        return False
    try:
        samples = getattr(settings, "SSH_STAGE2_VERIFY_SAMPLES", 5)
        try:
            samples = max(1, int(str(samples).strip() or 5))
        except (TypeError, ValueError):
            samples = 5
        res = _verify.verify_stage2(job, samples=samples)
    except Exception as e:
        logger.error("[SSHT] %s 端到端校验执行失败", jid, exc_info=True)
        job["verify"] = {"passed": False, "summary": f"校验无法完成：{e}"}
        job["stage"] = STAGE_FAILED
        job["error"] = f"段2 传输已结束，但端到端校验无法完成：{e}"
        job["error_detail"] = ""
        job["finished_ts"] = time.time()
        return True                 # 给了结论（失败），调用方不要再动 stage
    finally:
        _release_verify(jid)
    job["verify"] = res
    if not res.get("passed"):
        job["stage"] = STAGE_FAILED
        job["error"] = "段2 传输器报成功，但端到端校验未通过（数据可能缺失/被截断）"
        job["error_detail"] = res.get("summary", "")
        job["finished_ts"] = time.time()
    return True


def poll_once(job: dict) -> dict:
    """按当前 stage 轮询对应段一次，更新 job。"""
    stage = job.get("stage")
    if stage not in _ACTIVE:
        return job
    eng_stage = STAGE1 if stage == STAGE_STAGE1 else STAGE2
    direct2 = eng_stage == STAGE2 and _stage2_engine_of(job) is engine_ossutil
    try:
        st = (engine_ossutil.poll_stage(job["job_id"]) if direct2
              else engine_ssh.poll_stage(job["job_id"], eng_stage))
    except Exception:
        logger.warning("[SSHT] 轮询 %s 失败 job=%s（暂保持在途）", stage, job.get("job_id"))
        return job
    _sample_progress(job, eng_stage)   # 已传字节 + 速率（供进度卡显示）
    status = st.get("status")
    if status == "FAILED":
        job["stage"] = STAGE_FAILED
        job["error"] = st.get("error", "") or f"{stage} 失败"
        job["finished_ts"] = time.time()
        # 先落库再去摘明细：摘明细要走 SSH（最坏几十秒），期间 updated_ts 若还停在上一轮，
        # 对账的 stale 门会认为本 job 失联、再触发一次 refresh → 白跑一遍远端 grep。
        _save(job)
        # 只在终态多花一次 SSH 摘真原因：光给「退出码 N」排障要手工翻十几 MB 日志（本次踩过）。
        try:
            job["error_detail"] = (engine_ossutil.failure_detail(job["job_id"]) if direct2
                                  else engine_ssh.failure_detail(job["job_id"], eng_stage))
        except Exception:
            job["error_detail"] = ""
    elif status == "DONE":
        if stage == STAGE_STAGE1:
            job["stage1_rc"] = st.get("rc", 0)   # 记段1成功，retry 可跳过段1
            _save(job)
            _start_stage(job, STAGE_STAGE2)       # 段1 完成 → 立即起段2
            return job
        else:  # 段2 传输器自称完成 —— 还要过端到端校验才算全链路完成
            job["stage2_rc"] = st.get("rc", 0)
            # 必须用返回值判：`stage != FAILED` 会把「校验被别人占着、本次没校验」也当成
            # 「校验通过」→ 零校验宣布 DONE + 推成功卡。见 _run_stage2_verify 的三态说明。
            if _run_stage2_verify(job) and job.get("stage") != STAGE_FAILED:
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
                      max_polls: int = 10080) -> dict:
    """启动并阻塞轮询至终态（后台线程调用）。段1成功才进段2；retry 时段1已成功直接从段2起。

    max_polls*poll_interval 默认上限 7 天。**原来 48h 不够**：真机实测段1 约 148MiB/s，
    19.5TiB 的单子光段1 就要 ~38h，加段2 必然超 48h → 会在任务其实还在正常跑的时候被判「轮询超时」
    并推失败卡。按 7 天给足余量（真失败由 rc marker 立刻判定，不依赖这个上限）。
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
        hours = max_polls * poll_interval // 3600
        span = f"{hours // 24}天" if hours >= 48 else f"{hours}h"
        job["stage"] = STAGE_FAILED
        job["error"] = f"轮询超时（>{span} 未完成）"
        job["error_detail"] = ""   # 同 _start_stage：别让上一轮的明细配上「轮询超时」这个新原因
        job["finished_ts"] = time.time()
        _save(job)
        if on_update:
            on_update(job)
    return job
