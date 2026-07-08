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
