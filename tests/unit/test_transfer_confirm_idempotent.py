"""跨云迁移确认幂等：连点“确认迁移”只真正下发一次，其余回“已下发”。

复现的线上问题：transfer 的 confirm handler 早先没有 launched 标记/锁（不像 cpfs/bucket），
用户连点不同确认卡（msg_id 不同、内容去重挡不住）→ 每次都起后台线程刷卡。
"""
import pytest


@pytest.fixture
def A(monkeypatch):
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    # run_to_completion 置空：避免线程里真的建 MGW 任务/推卡
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b: False)
    monkeypatch.setattr(orchestrator, "set_same_name_policy", lambda job, p: job)
    return actions, orchestrator


def _new_job():
    return {"job_id": "tr-abc123", "stage": "NEW", "bytes_total": 1, "source": "tos://a/x/",
            "dest": "oss://b/y/", "same_name_policy": "skip"}


def test_confirm_launches_once_then_blocks(A, monkeypatch):
    actions, orch = A
    store = {"tr-abc123": _new_job()}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda job: store.__setitem__(job["job_id"], job))

    r1 = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r1["toast"]["type"] == "success"
    assert store["tr-abc123"].get("launched") is True
    # 确认后原地替换为“进行中”卡（确认按钮消失，无法再连点）
    assert r1.get("card", {}).get("type") == "raw"
    assert "进行中" in str(r1["card"]["data"]) or "迁移中" in str(r1["card"]["data"])

    # 第二次连点：launched 已置 → 直接挡回
    r2 = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r2["toast"]["type"] == "info"
    assert "已下发" in r2["toast"]["content"]


def test_retry_clears_launched_and_relaunches(A, monkeypatch):
    actions, orch = A
    failed = _new_job()
    failed["stage"] = "FAILED"
    failed["launched"] = True
    store = {"tr-abc123": failed}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda job: store.__setitem__(job["job_id"], job))
    # 先把锁占上，模拟上次下发残留
    from utils.redis_client import get_redis
    get_redis().set("transfer:launch:tr-abc123", 1)

    r = actions._h_retry_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success"          # 重试成功再次下发
    assert store["tr-abc123"]["stage"] != "FAILED"
    assert store["tr-abc123"]["launched"] is True    # 重新下发后再次置位


def test_reparse_running_path_shows_progress_not_confirm(monkeypatch):
    """对已在跑的同一源/目的再点“解析并预估”→ 回进度卡，不再弹确认卡、也不重复慢预估。"""
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    import core.transfer.cards as cards
    import core.dsw_scheduler as sched

    class _Sync:   # 让后台线程同步执行，便于断言
        def __init__(self, target=None, daemon=None, **k): self._t = target
        def start(self): self._t()
    monkeypatch.setattr(actions.threading, "Thread", _Sync)

    class _Plan:
        engine = "mgw"
        direction = "tos→oss"
    monkeypatch.setattr(orchestrator, "make_plan", lambda s, d="": _Plan())
    monkeypatch.setattr(orchestrator, "_job_id", lambda plan: "tr-run1")
    crossing = {"job_id": "tr-run1", "stage": orchestrator.STAGE_CROSSING}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: crossing)
    monkeypatch.setattr(orchestrator, "refresh", lambda jid: crossing)

    def _boom(*a, **k):
        raise AssertionError("已在跑不应再跑慢预估")
    monkeypatch.setattr(orchestrator, "estimate_source", _boom)
    monkeypatch.setattr(cards, "progress_card", lambda j: {"PROGRESS": j["job_id"]})

    def _no_confirm(*a, **k):
        raise AssertionError("已在跑不应再弹确认卡")
    monkeypatch.setattr(cards, "confirm_card", _no_confirm)
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append(card))

    actions._h_submit_transfer({"action": "submit_transfer"}, "ou_x", "chat",
                               {"source": "tos://a/x/", "dest": "oss://b/y/"})
    assert sent == [{"PROGRESS": "tr-run1"}]
