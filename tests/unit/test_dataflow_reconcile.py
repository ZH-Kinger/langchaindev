"""调度器数据流动/迁移对账：孤儿在途任务完成后自动补推结果卡、且不与在线线程重复。

覆盖三条关键语义：
1. 过期(孤儿)且完成 → refresh 后推结果卡 + 置 notified；
2. 新鲜(有活线程在轮询) → 跳过，绝不重复推；
3. 已 notified / 非在途 → 跳过。
"""
import time
import pytest


@pytest.fixture
def sched(monkeypatch):
    from core import dsw_scheduler as s
    sent = []
    monkeypatch.setattr(s, "_send_card", lambda o, c, card: sent.append((c, card)))
    return s, sent


def _spec(o, cards, active, name="x"):
    return {"name": name, "o": o, "cards": cards, "active": active,
            "chat": lambda: "oc_chat", "cleanup": None}


class _Cards:
    @staticmethod
    def result_card(job):
        return {"RESULT": job["job_id"], "stage": job["stage"]}


class _Orch:
    _KEY_PREFIX = "test:job:"
    STAGE_RUNNING = "RUNNING"
    STAGE_DONE = "DONE"
    STAGE_FAILED = "FAILED"

    def __init__(self, store, refresh_to=None):
        self.store = store
        self.refresh_to = refresh_to
        self.refresh_calls = []

    def get_job(self, jid):
        return self.store.get(jid)

    def refresh(self, jid):
        self.refresh_calls.append(jid)
        job = self.store.get(jid)
        if self.refresh_to and job:
            job = {**job, "stage": self.refresh_to}
            self.store[jid] = job
        return job

    def _save(self, job):
        self.store[job["job_id"]] = job


def _run(monkeypatch, s, orch, cards, active):
    monkeypatch.setattr(s, "_dataflow_reconcile_specs",
                        lambda: [_spec(orch, cards, active)])
    # scan_iter over the fake orch store
    class _R:
        def scan_iter(self, pat):
            return [orch._KEY_PREFIX + jid for jid in list(orch.store)]
    monkeypatch.setattr(s, "get_redis", lambda: _R())
    s._reconcile_dataflow_once()


def test_orphan_done_gets_notified(sched, monkeypatch):
    s, sent = sched
    old = time.time() - 10_000     # 过期 → 孤儿
    store = {"j1": {"job_id": "j1", "stage": "RUNNING", "updated_ts": old, "task_id": "t"}}
    orch = _Orch(store, refresh_to="DONE")
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert orch.refresh_calls == ["j1"]              # 对孤儿实时重查
    assert sent and sent[0][1] == {"RESULT": "j1", "stage": "DONE"}
    assert store["j1"]["notified"] is True           # 置标记，下轮不再推


def test_fresh_job_skipped_no_double_push(sched, monkeypatch):
    s, sent = sched
    fresh = time.time() - 5         # 新鲜 → 有活线程在管
    store = {"j2": {"job_id": "j2", "stage": "RUNNING", "updated_ts": fresh, "task_id": "t"}}
    orch = _Orch(store, refresh_to="DONE")
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert orch.refresh_calls == []                  # 不碰新鲜任务
    assert sent == []                                # 绝不重复推送


def test_already_notified_and_terminal_skipped(sched, monkeypatch):
    s, sent = sched
    old = time.time() - 10_000
    store = {
        "n": {"job_id": "n", "stage": "RUNNING", "updated_ts": old, "notified": True},  # 已通知
        "d": {"job_id": "d", "stage": "DONE", "updated_ts": old},                       # 非在途
    }
    orch = _Orch(store, refresh_to="DONE")
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert orch.refresh_calls == []
    assert sent == []


def test_orphan_still_running_no_push(sched, monkeypatch):
    s, sent = sched
    old = time.time() - 10_000
    store = {"j": {"job_id": "j", "stage": "RUNNING", "updated_ts": old, "task_id": "t"}}
    orch = _Orch(store, refresh_to="RUNNING")   # 重查后仍在跑
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert orch.refresh_calls == ["j"]          # 接管轮询
    assert sent == []                           # 未完成 → 不推
    assert "notified" not in store["j"]


def test_orphan_failed_gets_notified(sched, monkeypatch):
    """孤儿重查为 FAILED 也要补推结果卡（此前只测了 DONE）。"""
    s, sent = sched
    old = time.time() - 10_000
    store = {"f": {"job_id": "f", "stage": "RUNNING", "updated_ts": old}}
    orch = _Orch(store, refresh_to="FAILED")
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert sent and sent[0][1] == {"RESULT": "f", "stage": "FAILED"}
    assert store["f"]["notified"] is True


def test_cleanup_hook_called_on_terminal(sched, monkeypatch):
    """终态先跑 cleanup 钩子（cpfs 删临时 DataFlow）再推卡。"""
    s, sent = sched
    old = time.time() - 10_000
    store = {"c": {"job_id": "c", "stage": "RUNNING", "updated_ts": old}}
    orch = _Orch(store, refresh_to="DONE")
    cleaned = []
    spec = {"name": "cpfs", "o": orch, "cards": _Cards, "active": {"RUNNING"},
            "chat": lambda: "oc_chat", "cleanup": lambda j: cleaned.append(j["job_id"])}
    monkeypatch.setattr(s, "_dataflow_reconcile_specs", lambda: [spec])

    class _R:
        def scan_iter(self, pat):
            return [orch._KEY_PREFIX + jid for jid in list(orch.store)]
    monkeypatch.setattr(s, "get_redis", lambda: _R())
    s._reconcile_dataflow_once()
    assert cleaned == ["c"]
    assert sent and sent[0][1] == {"RESULT": "c", "stage": "DONE"}


# ── 跨线程 NX 去重闸门（2e5f01c）──────────────────────────────────────────────────

def _run_with_gate(monkeypatch, s, orch, cards, active, gate_result):
    """像 _run，但给 fake redis 一个返回 gate_result 的 .set（模拟 NX 抢闸门成/败）。"""
    monkeypatch.setattr(s, "_dataflow_reconcile_specs",
                        lambda: [_spec(orch, cards, active)])

    class _R:
        def scan_iter(self, pat):
            return [orch._KEY_PREFIX + jid for jid in list(orch.store)]
        def set(self, *a, **k):
            return gate_result
    monkeypatch.setattr(s, "get_redis", lambda: _R())
    s._reconcile_dataflow_once()


def test_reconcile_skips_push_when_gate_taken(sched, monkeypatch):
    """在线线程已抢到 NX 闸门（.set→False）→ 对账仍落 notified 但不重复推卡。"""
    s, sent = sched
    old = time.time() - 10_000
    store = {"g": {"job_id": "g", "stage": "RUNNING", "updated_ts": old}}
    orch = _Orch(store, refresh_to="DONE")
    _run_with_gate(monkeypatch, s, orch, _Cards, {"RUNNING"}, gate_result=False)
    assert store["g"]["notified"] is True   # 仍落盘 notified（幂等标记）
    assert sent == []                        # 闸门被在线线程占 → 不重复推


def test_reconcile_pushes_when_gate_open(sched, monkeypatch):
    """无人抢闸门（.set→True）→ 对账抢到，正常补推。"""
    s, sent = sched
    old = time.time() - 10_000
    store = {"g": {"job_id": "g", "stage": "RUNNING", "updated_ts": old}}
    orch = _Orch(store, refresh_to="DONE")
    _run_with_gate(monkeypatch, s, orch, _Cards, {"RUNNING"}, gate_result=True)
    assert sent and sent[0][1] == {"RESULT": "g", "stage": "DONE"}


def test_claim_dataflow_notify_once_then_blocks():
    """终态“只推一次”闸门：首次 True、同 job_id 再抢 False（NX，fakeredis 实测）。"""
    from core import dsw_scheduler as s
    assert s._claim_dataflow_notify("j-gate-1") is True
    assert s._claim_dataflow_notify("j-gate-1") is False
    assert s._claim_dataflow_notify("j-gate-2") is True   # 不同 job 互不影响


def test_claim_dataflow_notify_redis_down_passthrough(monkeypatch):
    """Redis 不可用 → 放行（宁可重复也别漏通知）。"""
    from core import dsw_scheduler as s
    def _boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr(s, "get_redis", _boom)
    assert s._claim_dataflow_notify("j-any") is True


# ── 真实 specs 契约：四个命名空间必须齐备对账所依赖的属性 ─────────────────────────

# ── #41 对账终态推送优先发起人（created_by），空则降级配置频道 ────────────────────

@pytest.fixture
def sched_full(monkeypatch):
    """像 sched，但捕获 _send_card 的完整三参 (target, chat, card) 以断言首参=created_by。"""
    from core import dsw_scheduler as s
    sent = []
    monkeypatch.setattr(s, "_send_card", lambda o, c, card: sent.append((o, c, card)))
    return s, sent


def test_reconcile_pushes_to_created_by(sched_full, monkeypatch):
    """孤儿完成补推：_send_card 首参=job.created_by（发起人 open_id），而非空串。"""
    s, sent = sched_full
    old = time.time() - 10_000
    store = {"j1": {"job_id": "j1", "stage": "RUNNING", "updated_ts": old,
                    "created_by": "ou_creator"}}
    orch = _Orch(store, refresh_to="DONE")
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert sent and sent[0][0] == "ou_creator"     # 目标=发起人
    assert sent[0][1] == "oc_chat"                  # 降级频道仍作第二参（target or chat）


def test_reconcile_missing_created_by_falls_back_to_empty(sched_full, monkeypatch):
    """无 created_by → 首参为 ""（_send_card 内部 target or chat 会回落配置频道）。"""
    s, sent = sched_full
    old = time.time() - 10_000
    store = {"j2": {"job_id": "j2", "stage": "RUNNING", "updated_ts": old}}
    orch = _Orch(store, refresh_to="DONE")
    _run(monkeypatch, s, orch, _Cards, {"RUNNING"})
    assert sent and sent[0][0] == ""                # 降级：空目标
    assert sent[0][1] == "oc_chat"


def test_real_reconcile_specs_contract():
    """对账把 _dataflow_reconcile_specs 全 mock 掉了；这里断言真实 specs 的四个命名空间
    (transfer/cpfs/vepfs/bucket) 各自 orchestrator/cards 都真有对账依赖的属性——
    任一重命名（如 refresh/_KEY_PREFIX/result_card/STAGE_DONE）会静默拖垮对账，此测钉死。"""
    from core import dsw_scheduler as s
    specs = s._dataflow_reconcile_specs()
    names = {sp["name"] for sp in specs}
    assert {"transfer", "cpfs", "vepfs", "bucket"} <= names
    for sp in specs:
        o = sp["o"]
        for attr in ("_KEY_PREFIX", "get_job", "refresh", "_save", "STAGE_DONE", "STAGE_FAILED"):
            assert hasattr(o, attr), f"{sp['name']} orchestrator 缺 {attr}"
        assert isinstance(o._KEY_PREFIX, str) and o._KEY_PREFIX
        assert hasattr(sp["cards"], "result_card"), f"{sp['name']} cards 缺 result_card"
        assert callable(sp["chat"])
        assert isinstance(sp["active"], set) and sp["active"]
        assert sp["cleanup"] is None or callable(sp["cleanup"])
