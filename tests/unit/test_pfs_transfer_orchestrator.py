"""PFS 跨云直传 —— orchestrator：job_id 幂等 / create_job_record / needs_approval /
run_to_completion 三段推进 + 段级跳过 / _make_sub 方向×段派发 / refresh 定位段续推。

三个子 orchestrator（vepfs/cpfs/transfer）的 make_plan/create_job_record/run_to_completion/
refresh/get_job 全部用哨兵桩探测调用；PFS_STAGING_MAP 用 monkeypatch settings.PFS_STAGING_MAP_RAW；
Redis 走 fakeredis（conftest）。
"""
import json

import pytest

from core.pfs_transfer import paths
from core.pfs_transfer import orchestrator as o
from config.settings import settings

VEPFS_FS = "vepfs-abc"
CPFS_FS = "cpfs-xyz"
_MAP = {
    f"vepfs://{VEPFS_FS}": {"region": "cn-beijing", "tos_bucket": "wuji-dc-bj", "tos_prefix": "pfs-staging/"},
    f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "wuji-il-hz",
                          "oss_prefix": "pfs-staging/", "dataflow_id": "df-123"},
}


@pytest.fixture(autouse=True)
def _map(monkeypatch):
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(_MAP))


def _plan_p1(sub="wzh/"):
    return paths.build_plan(f"vepfs://{VEPFS_FS}/{sub}", f"cpfs://{CPFS_FS}/team/")


def _plan_p2(sub="team/"):
    return paths.build_plan(f"cpfs://{CPFS_FS}/{sub}", f"vepfs://{VEPFS_FS}/wzh/")


# ══════════════════════════════════════════════════════════════════════════════
# _job_id
# ══════════════════════════════════════════════════════════════════════════════

def test_job_id_prefix_and_deterministic():
    p = _plan_p1()
    j1, j2 = o._job_id(p), o._job_id(p)
    assert j1 == j2
    assert j1.startswith("xpfs-")


def test_job_id_differs_by_plan():
    assert o._job_id(_plan_p1("a/")) != o._job_id(_plan_p1("b/"))


def test_job_id_direction_differs():
    """P1 与 P2（源目的互换）→ 异 id。"""
    assert o._job_id(_plan_p1()) != o._job_id(_plan_p2())


# ══════════════════════════════════════════════════════════════════════════════
# create_job_record
# ══════════════════════════════════════════════════════════════════════════════

def test_create_first_build():
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    assert job["stage"] == o.STAGE_NEW
    assert job["chain"] == "pfs"
    assert job["direction"] == paths.DIRECTION_P1
    assert job["created_by"] == "ou1"
    assert job["sink_done"] is False and job["cross_done"] is False and job["preheat_done"] is False
    assert job["sink_job_id"] == "" and job["launched"] is False


def test_create_staging_prefix_contains_chain_id():
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    jid = job["job_id"]
    for st in ("src_staging", "dst_staging"):
        pref = job[st]["prefix"]
        assert pref.startswith("pfs-staging/")
        assert pref.endswith(jid + "/")
        assert jid in pref


def test_create_idempotent_reuse_same_id():
    p = _plan_p1()
    j1 = o.create_job_record(p, open_id="")
    j2 = o.create_job_record(p, open_id="")
    assert j1["job_id"] == j2["job_id"]
    assert j2["stage"] == o.STAGE_NEW      # 复用旧记录，未复位


def test_create_backfills_empty_created_by():
    p = _plan_p1()
    o.create_job_record(p, open_id="")             # 先建，created_by 空
    j2 = o.create_job_record(p, open_id="ou2")     # 回填
    assert j2["created_by"] == "ou2"


def test_create_does_not_overwrite_created_by():
    p = _plan_p1()
    o.create_job_record(p, open_id="ou1")
    j2 = o.create_job_record(p, open_id="ou2")
    assert j2["created_by"] == "ou1"


def test_create_failed_existing_rebuilt():
    """既存记录已 FAILED → 新建复位（同 job_id）。"""
    p = _plan_p1()
    j1 = o.create_job_record(p, open_id="ou1")
    j1["stage"] = o.STAGE_FAILED
    j1["sink_done"] = True
    o._save(j1)
    j2 = o.create_job_record(p, open_id="ou1")
    assert j2["job_id"] == j1["job_id"]
    assert j2["stage"] == o.STAGE_NEW
    assert j2["sink_done"] is False       # 复位


# ══════════════════════════════════════════════════════════════════════════════
# needs_approval（fail-safe）
# ══════════════════════════════════════════════════════════════════════════════

def test_needs_approval_size_unknown_always_true(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_APPROVAL_TB", 1)
    assert o.needs_approval(0, size_known=False) is True
    assert o.needs_approval(10 ** 18, size_known=False) is True


def test_needs_approval_known_under_threshold_false(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_APPROVAL_TB", 1)
    assert o.needs_approval(500 * 1024 ** 3, size_known=True) is False


def test_needs_approval_boundary(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_APPROVAL_TB", 1)
    assert o.needs_approval(1 * 1024 ** 4, size_known=True) is False       # == 阈值，不超
    assert o.needs_approval(1 * 1024 ** 4 + 1, size_known=True) is True    # 超一字节


def test_needs_approval_custom_threshold(monkeypatch):
    monkeypatch.setattr(settings, "PFS_TRANSFER_APPROVAL_TB", 5)
    assert o.needs_approval(4 * 1024 ** 4, size_known=True) is False
    assert o.needs_approval(6 * 1024 ** 4, size_known=True) is True


# ══════════════════════════════════════════════════════════════════════════════
# 子 orchestrator 哨兵桩
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def subs(monkeypatch):
    """把 vepfs/cpfs/transfer 三个子 orchestrator 的 make_plan/create_job_record/
    run_to_completion 换成哨兵桩，记录调用；run_to_completion 返回的 stage 由 run_stage 控。"""
    rec = {"make_plan": [], "create": [], "run": [], "counter": {}}
    rec["run_stage"] = {}     # modname -> "DONE"/"FAILED"

    def _mk(modname, mod):
        def make_plan(*a, **k):
            rec["make_plan"].append({"mod": modname, "args": a, "kwargs": k})
            return {"_plan": modname}

        def create_job_record(plan, *, open_id="", **k):
            n = rec["counter"].get(modname, 0) + 1
            rec["counter"][modname] = n
            jid = f"{modname}-{n}"
            rec["create"].append({"mod": modname, "open_id": open_id, "job_id": jid})
            return {"job_id": jid, "stage": "NEW"}

        def run_to_completion(sub_job, *, on_update=None, poll_interval=60, max_polls=1440):
            rec["run"].append({"mod": modname, "job_id": sub_job["job_id"]})
            sub_job["stage"] = rec["run_stage"].get(modname, "DONE")
            if on_update:
                try:
                    on_update(sub_job)
                except Exception:
                    pass
            return sub_job

        monkeypatch.setattr(mod, "make_plan", make_plan)
        monkeypatch.setattr(mod, "create_job_record", create_job_record)
        monkeypatch.setattr(mod, "run_to_completion", run_to_completion)

    _mk("vepfs", o.vepfs_o)
    _mk("cpfs", o.cpfs_o)
    _mk("transfer", o.transfer_o)
    return rec


# ── _make_sub 派发 + 参数 ─────────────────────────────────────────────────────

def test_make_sub_p1_dispatch_and_ops(subs):
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    # 沉降 → vepfs Export
    sub_o, _ = o._make_sub(job, o.STAGE_SINKING)
    assert sub_o is o.vepfs_o
    mp = subs["make_plan"][-1]
    assert mp["mod"] == "vepfs" and mp["args"][0] == o.vepfs_o.OP_SINK
    assert mp["kwargs"]["fs_id"] == VEPFS_FS
    assert mp["args"][2].startswith("tos://wuji-dc-bj/")    # 沉降落点 = TOS staging
    # 跨云 → transfer
    sub_o, _ = o._make_sub(job, o.STAGE_CROSSING)
    assert sub_o is o.transfer_o
    mp = subs["make_plan"][-1]
    assert mp["mod"] == "transfer"
    assert mp["args"][0].startswith("tos://wuji-dc-bj/")    # 源 = TOS
    assert mp["args"][1].startswith("oss://wuji-il-hz/")    # 目的 = OSS
    # 预热 → cpfs Import
    sub_o, _ = o._make_sub(job, o.STAGE_PREHEATING)
    assert sub_o is o.cpfs_o
    mp = subs["make_plan"][-1]
    assert mp["mod"] == "cpfs" and mp["args"][0] == o.cpfs_o.OP_PREHEAT
    assert mp["kwargs"]["fs_id"] == CPFS_FS


def test_make_sub_p2_dispatch(subs):
    job = o.create_job_record(_plan_p2(), open_id="ou1")
    assert o._make_sub(job, o.STAGE_SINKING)[0] is o.cpfs_o       # CPFS Export
    assert o._make_sub(job, o.STAGE_CROSSING)[0] is o.transfer_o
    assert o._make_sub(job, o.STAGE_PREHEATING)[0] is o.vepfs_o   # vePFS Import
    ops = [m["args"][0] for m in subs["make_plan"]]
    assert ops[0] == o.cpfs_o.OP_SINK and ops[2] == o.vepfs_o.OP_PREHEAT


def test_make_sub_propagates_created_by(subs):
    job = o.create_job_record(_plan_p1(), open_id="ou_creator")
    o._make_sub(job, o.STAGE_SINKING)
    assert subs["create"][-1]["open_id"] == "ou_creator"


# ══════════════════════════════════════════════════════════════════════════════
# run_to_completion —— 三段推进 / FAILED 不进下一段 / 段级跳过
# ══════════════════════════════════════════════════════════════════════════════

def test_run_all_three_stages_done(subs):
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    stages_seen = []
    res = o.run_to_completion(job, on_update=lambda j: stages_seen.append(j["stage"]))
    assert res["stage"] == o.STAGE_DONE
    # 三段按序、用对子 orchestrator
    assert [r["mod"] for r in subs["run"]] == ["vepfs", "transfer", "cpfs"]
    assert job["sink_done"] and job["cross_done"] and job["preheat_done"]
    assert o.STAGE_SINKING in stages_seen and o.STAGE_DONE in stages_seen


def test_run_p2_uses_reversed_orchestrators(subs):
    job = o.create_job_record(_plan_p2(), open_id="ou1")
    o.run_to_completion(job)
    assert [r["mod"] for r in subs["run"]] == ["cpfs", "transfer", "vepfs"]


def test_run_stage_failed_stops_chain(subs):
    """跨云段 FAILED → 链 FAILED，不进预热段。"""
    subs["run_stage"]["transfer"] = "FAILED"
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    res = o.run_to_completion(job)
    assert res["stage"] == o.STAGE_FAILED
    assert [r["mod"] for r in subs["run"]] == ["vepfs", "transfer"]   # cpfs 未跑
    assert job["sink_done"] is True and job["cross_done"] is False
    assert "cpfs" not in [r["mod"] for r in subs["run"]]
    assert job["error"]


def test_run_first_stage_failed_stops_immediately(subs):
    subs["run_stage"]["vepfs"] = "FAILED"
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    res = o.run_to_completion(job)
    assert res["stage"] == o.STAGE_FAILED
    assert [r["mod"] for r in subs["run"]] == ["vepfs"]


def test_run_stage_skip_already_done(subs):
    """sink_done=True → 跳沉降，从跨云段起。"""
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    job["sink_done"] = True
    o._save(job)
    res = o.run_to_completion(job)
    assert res["stage"] == o.STAGE_DONE
    assert [r["mod"] for r in subs["run"]] == ["transfer", "cpfs"]    # 沉降跳过


def test_run_skip_two_stages(subs):
    """sink_done + cross_done → 只跑预热。"""
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    job["sink_done"] = True
    job["cross_done"] = True
    o._save(job)
    o.run_to_completion(job)
    assert [r["mod"] for r in subs["run"]] == ["cpfs"]


def test_run_sets_launched(subs):
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    o.run_to_completion(job)
    assert job["launched"] is True


def test_run_sub_run_raises_marks_failed(subs, monkeypatch):
    """子 run_to_completion 抛异常 → 链 FAILED（不外泄）。"""
    def boom(sub_job, **k):
        raise RuntimeError("net down")
    monkeypatch.setattr(o.vepfs_o, "run_to_completion", boom)
    job = o.create_job_record(_plan_p1(), open_id="ou1")
    res = o.run_to_completion(job)
    assert res["stage"] == o.STAGE_FAILED and "net down" in res["error"]


# ══════════════════════════════════════════════════════════════════════════════
# refresh —— 定位段 + launched 守卫（HIGH-1）+ 续跑 + FAILED 反映
# ══════════════════════════════════════════════════════════════════════════════

class _SyncThread:
    def __init__(self, target=None, daemon=None, **k):
        self._t = target

    def start(self):
        if self._t:
            self._t()


@pytest.fixture
def resume_spy(monkeypatch):
    """桩 pfs run_to_completion（探测续跑）+ 同步线程。"""
    calls = []
    monkeypatch.setattr(o, "run_to_completion",
                        lambda job, **k: calls.append(job.get("job_id")) or job)
    monkeypatch.setattr(o.threading, "Thread", _SyncThread)
    return calls


def _mk_inflight(stage, direction=paths.DIRECTION_P1, **over):
    plan = _plan_p1() if direction == paths.DIRECTION_P1 else _plan_p2()
    job = o.create_job_record(plan, open_id="ou1")
    job["stage"] = stage
    job.update(over)
    o._save(job)
    return job


def test_refresh_crossing_done_triggers_resume(subs, resume_spy, monkeypatch):
    """在途 CROSSING、launched：子 job DONE → cross_done + 后台续跑。"""
    job = _mk_inflight(o.STAGE_CROSSING, launched=True, sink_done=True, cross_job_id="transfer-1")
    monkeypatch.setattr(o.transfer_o, "refresh", lambda jid: None)
    monkeypatch.setattr(o.transfer_o, "get_job", lambda jid: {"stage": "DONE"})
    out = o.refresh(job["job_id"])
    assert out["cross_done"] is True
    assert resume_spy == [job["job_id"]]        # 续跑被触发


def test_refresh_stage_still_running_no_resume(subs, resume_spy, monkeypatch):
    """子 job 仍 RUNNING → 不置 done、不续跑。"""
    job = _mk_inflight(o.STAGE_SINKING, launched=True, sink_job_id="vepfs-1")
    monkeypatch.setattr(o.vepfs_o, "refresh", lambda jid: None)
    monkeypatch.setattr(o.vepfs_o, "get_job", lambda jid: {"stage": "RUNNING"})
    o.refresh(job["job_id"])
    assert resume_spy == []


def test_refresh_sub_failed_reflects_to_chain(subs, resume_spy, monkeypatch):
    """子 job FAILED → 链 FAILED，不续跑。"""
    job = _mk_inflight(o.STAGE_CROSSING, launched=True, sink_done=True, cross_job_id="transfer-1")
    monkeypatch.setattr(o.transfer_o, "refresh", lambda jid: None)
    monkeypatch.setattr(o.transfer_o, "get_job", lambda jid: {"stage": "FAILED", "error": "boom"})
    out = o.refresh(job["job_id"])
    assert out["stage"] == o.STAGE_FAILED
    assert "boom" in out["error"]
    assert resume_spy == []


def test_refresh_terminal_noop(subs, resume_spy, monkeypatch):
    """已 DONE 的链 → refresh 直接返回、不查子、不续跑。"""
    job = _mk_inflight(o.STAGE_DONE, launched=True)
    called = []
    monkeypatch.setattr(o.transfer_o, "refresh", lambda jid: called.append(jid))
    out = o.refresh(job["job_id"])
    assert out["stage"] == o.STAGE_DONE
    assert called == [] and resume_spy == []


def test_refresh_missing_job_returns_none(resume_spy):
    assert o.refresh("xpfs-doesnotexist") is None


# ── HIGH-1 锁死回归：未 launched 的 NEW job 不能靠 refresh 触发续跑（绕过审批门）─────

def test_refresh_new_not_launched_no_resume(resume_spy):
    """未 launched（未经 confirm 审批门）的 NEW job → refresh 不起后台线程、不续跑。"""
    job = _mk_inflight(o.STAGE_NEW, launched=False)
    o.refresh(job["job_id"])
    assert resume_spy == []      # 绝不续跑


def test_resume_async_guard_blocks_unlaunched(resume_spy):
    """直测 _resume_async 守卫：launched=False → 立即返回，零续跑。"""
    job = _mk_inflight(o.STAGE_NEW, launched=False)
    o._resume_async(job)
    assert resume_spy == []


def test_refresh_new_launched_resumes(resume_spy):
    """launched=True 的 NEW（崩溃续跑场景）→ 守卫放行，续跑。"""
    job = _mk_inflight(o.STAGE_NEW, launched=True)
    o.refresh(job["job_id"])
    assert resume_spy == [job["job_id"]]


def test_resume_async_all_done_no_resume(resume_spy):
    """三段全 done 但 stage 卡在途 → 无剩余段可跑，不续跑。"""
    job = _mk_inflight(o.STAGE_PREHEATING, launched=True,
                       sink_done=True, cross_done=True, preheat_done=True)
    o._resume_async(job)
    assert resume_spy == []


# ══════════════════════════════════════════════════════════════════════════════
# progress_line
# ══════════════════════════════════════════════════════════════════════════════

def test_progress_line_with_bytes():
    job = {"stage": o.STAGE_CROSSING, "bytes_done": 50, "bytes_total": 100}
    line = o.progress_line(job)
    assert "50" in line or "%" in line


def test_progress_line_no_total():
    job = {"stage": o.STAGE_SINKING, "bytes_done": 0, "bytes_total": 0}
    assert "进行中" in o.progress_line(job)
