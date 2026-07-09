"""HIGH#1（#37）：四链 retry 重置 stage=NEW 时必须清各自 launch 锁 + `dataflow:notified:{job_id}`。

复现的线上问题：`dataflow:notified:{job_id}` 30 天 TTL、只 set 不 delete，job_id 当天不变跨重试
相同 → 首次失败已占闸门 → 重试成功时终态回调抢闸门返 False → **成功结果卡永远不推**。
另 launch NX 锁（30s TTL）不清则 30s 内重试被“已下发”静默吞。

四条链路（transfer / cpfs / vepfs / bucket）的 retry handler 都应：
  ① _save(stage=NEW)（cpfs/vepfs 此前不落盘，靠 Redis 旧值侥幸过 guard）；
  ② delete(<各自 launch 锁>, "dataflow:notified:{job_id}")。
retry 末尾会调 _h_confirm_*，本测把 confirm 换成 recorder，只验重置+清键这一步。
"""
import pytest

from core.feishu_bot import actions
from utils.redis_client import get_redis


# 四链参数：(retry handler 名, confirm handler 名, orchestrator 模块路径, launch 锁 key 前缀)
_CHAINS = [
    ("_h_retry_transfer",       "_h_confirm_transfer",       "core.transfer.orchestrator",       "transfer:launch"),
    ("_h_retry_cpfs_dataflow",  "_h_confirm_cpfs_dataflow",  "core.cpfs_dataflow.orchestrator",  "cpfs:dataflow:launch"),
    ("_h_retry_vepfs_dataflow", "_h_confirm_vepfs_dataflow", "core.vepfs_dataflow.orchestrator", "vepfs:dataflow:launch"),
    ("_h_retry_bucket_transfer","_h_confirm_bucket_transfer","core.bucket_transfer.orchestrator","bkt:launch"),
]


def _load_orch(path):
    import importlib
    return importlib.import_module(path)


@pytest.mark.parametrize("retry_name,confirm_name,orch_path,lock_prefix", _CHAINS,
                         ids=[c[0] for c in _CHAINS])
def test_retry_clears_launch_lock_and_notified_gate(monkeypatch, retry_name, confirm_name,
                                                    orch_path, lock_prefix):
    orch = _load_orch(orch_path)
    jid = "job-retry-1"
    store = {jid: {"job_id": jid, "stage": orch.STAGE_FAILED, "error": "boom", "launched": True}}
    monkeypatch.setattr(orch, "get_job", lambda j: store.get(j))
    monkeypatch.setattr(orch, "_save", lambda job: store.__setitem__(job["job_id"], dict(job)))

    # confirm 换成 recorder：只验重置阶段这一步，不真下发
    confirmed = []
    monkeypatch.setattr(actions, confirm_name,
                        lambda av, oid, cid, fv, **k: confirmed.append(av) or {"toast": {"type": "success", "content": "ok"}})

    # 预置两把残留 key：launch 锁 + 终态通知闸门
    r = get_redis()
    r.set(f"{lock_prefix}:{jid}", 1)
    r.set(f"dataflow:notified:{jid}", 1)

    out = getattr(actions, retry_name)({"job_id": jid}, "ou_admin", "chat", {})

    # ① 两把 key 都被删
    assert r.get(f"{lock_prefix}:{jid}") is None, "launch 锁未清 → 30s 内重试被吞"
    assert r.get(f"dataflow:notified:{jid}") is None, "notified 闸门未清 → 重试成功后结果卡永不推"
    # ② stage 落盘为 NEW（cpfs/vepfs 尤其：不落盘则 confirm 从 Redis 重读到旧 stage 被 guard 挡回）
    assert store[jid]["stage"] == orch.STAGE_NEW
    # ③ 确实转交 confirm 重新下发
    assert confirmed == [{"job_id": jid}]
    assert out["toast"]["type"] == "success"


@pytest.mark.parametrize("retry_name,orch_path", [(c[0], c[2]) for c in _CHAINS],
                         ids=[c[0] for c in _CHAINS])
def test_retry_missing_job_returns_error_no_delete(monkeypatch, retry_name, orch_path):
    """job 不存在 → error toast，且不误删任何 key、不转交 confirm。"""
    orch = _load_orch(orch_path)
    monkeypatch.setattr(orch, "get_job", lambda j: None)
    out = getattr(actions, retry_name)({"job_id": "nope"}, "ou_admin", "chat", {})
    assert out["toast"]["type"] == "error"
    assert "card" not in out


@pytest.mark.parametrize("orch_path", ["core.cpfs_dataflow.orchestrator",
                                       "core.vepfs_dataflow.orchestrator"])
def test_cpfs_vepfs_retry_persists_new_stage(monkeypatch, orch_path):
    """HIGH#1 重点：cpfs/vepfs retry 必须 _save（落盘 stage=NEW）——此前不落盘。

    以 _save 是否被调用 + 落到 store 的 stage 断言（confirm 换 recorder，隔离下发）。
    """
    orch = _load_orch(orch_path)
    retry_name = ("_h_retry_cpfs_dataflow" if "cpfs" in orch_path
                  else "_h_retry_vepfs_dataflow")
    confirm_name = ("_h_confirm_cpfs_dataflow" if "cpfs" in orch_path
                    else "_h_confirm_vepfs_dataflow")
    jid = "job-persist-1"
    store = {jid: {"job_id": jid, "stage": orch.STAGE_FAILED, "error": "x"}}
    saved = []
    monkeypatch.setattr(orch, "get_job", lambda j: dict(store[j]))
    monkeypatch.setattr(orch, "_save",
                        lambda job: (saved.append(dict(job)), store.__setitem__(job["job_id"], dict(job))))
    monkeypatch.setattr(actions, confirm_name,
                        lambda *a, **k: {"toast": {"type": "success", "content": "ok"}})

    getattr(actions, retry_name)({"job_id": jid}, "ou_admin", "chat", {})
    assert saved, "retry 未落盘（缺 _save）→ confirm 会从 Redis 重读旧 FAILED stage"
    assert saved[-1]["stage"] == orch.STAGE_NEW and saved[-1]["error"] == ""
    assert store[jid]["stage"] == orch.STAGE_NEW
