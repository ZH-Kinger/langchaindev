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
