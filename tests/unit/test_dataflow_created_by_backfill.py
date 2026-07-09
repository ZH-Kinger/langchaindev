"""#45 P5：向导流程 created_by 丢失回填（CPFS + vePFS 对称）。

线上根因：`create_job_record` 幂等分支原样返回旧记录、从不回填 created_by → 同日同
(op/fs/子目录/tos) 的 job_id 被无 open_id 入口（Agent 工具 LLM 没注入 / CLI 无 open_id）
先建成空 created_by → 永久为空 → 进度/结果卡 `_send_card("",chat)` 回落配置频道（管理员），
不私信发起人；CPFS 额外：created_by 驱动云调用 STS 身份，空则退回 master AK（越权+审计错人）。

修①（治本）：orchestrator 幂等分支带真 open_id 时回填空 created_by 并 _save。
修②（兜底）：`_h_confirm_*` 在置 RUNNING 前用确认者 open_id 回填空 created_by。
"""
import importlib
import pytest


# (orchestrator 模块, 构造一个合法 plan 的工厂) —— 两条链路对称
_CHAINS = [
    ("core.cpfs_dataflow.orchestrator",
     lambda o: o.make_plan("sink", "/cwr/o/", "oss://bk/e/", fs_id="bmcpfs-x", region="cn-hangzhou")),
    ("core.vepfs_dataflow.orchestrator",
     lambda o: o.make_plan("sink", "/wzh/", "tos://bk/e/", fs_id="vepfs-x", region="cn-beijing")),
]
_IDS = ["cpfs", "vepfs"]


def _orch(path):
    return importlib.import_module(path)


# ── 修① orchestrator.create_job_record 幂等分支回填 ─────────────────────────────

@pytest.mark.parametrize("path,mkplan", _CHAINS, ids=_IDS)
def test_idempotent_backfills_empty_created_by(monkeypatch, path, mkplan):
    """旧记录 created_by 空 + 本次带真 open_id → 回填并 _save（返回同一旧记录）。"""
    orch = _orch(path)
    plan = mkplan(orch)
    existing = {"job_id": "j-1", "stage": orch.STAGE_RUNNING, "created_by": ""}
    saved = []
    monkeypatch.setattr(orch, "get_job", lambda jid: existing)
    monkeypatch.setattr(orch, "_save", lambda job: saved.append(dict(job)))

    out = orch.create_job_record(plan, open_id="ou_real")
    assert out is existing                         # 幂等复用旧记录（未走新建）
    assert out["created_by"] == "ou_real"          # 回填真值
    assert saved and saved[-1]["created_by"] == "ou_real"   # 落盘


@pytest.mark.parametrize("path,mkplan", _CHAINS, ids=_IDS)
def test_idempotent_does_not_overwrite_existing_created_by(monkeypatch, path, mkplan):
    """旧记录 created_by 已有非空 → 绝不覆盖，且不触发 _save。"""
    orch = _orch(path)
    plan = mkplan(orch)
    existing = {"job_id": "j-1", "stage": orch.STAGE_RUNNING, "created_by": "ou_original"}
    saved = []
    monkeypatch.setattr(orch, "get_job", lambda jid: existing)
    monkeypatch.setattr(orch, "_save", lambda job: saved.append(dict(job)))

    out = orch.create_job_record(plan, open_id="ou_new_confirmer")
    assert out["created_by"] == "ou_original"      # 不被后来者覆盖
    assert saved == []                             # 条件不满足，未落盘


@pytest.mark.parametrize("path,mkplan", _CHAINS, ids=_IDS)
def test_idempotent_empty_open_id_does_not_write_empty(monkeypatch, path, mkplan):
    """本次 open_id 空（Agent/CLI 入口）→ 不写空、不 _save，保持旧记录原样。"""
    orch = _orch(path)
    plan = mkplan(orch)
    existing = {"job_id": "j-1", "stage": orch.STAGE_RUNNING, "created_by": ""}
    saved = []
    monkeypatch.setattr(orch, "get_job", lambda jid: existing)
    monkeypatch.setattr(orch, "_save", lambda job: saved.append(dict(job)))

    out = orch.create_job_record(plan, open_id="")
    assert out["created_by"] == ""                 # 空 open_id 不覆盖
    assert saved == []


@pytest.mark.parametrize("path,mkplan", _CHAINS, ids=_IDS)
def test_failed_record_goes_to_new_branch(monkeypatch, path, mkplan):
    """旧记录 FAILED → 不走幂等分支，走新建（created_by=本次 open_id，stage=NEW）。"""
    orch = _orch(path)
    plan = mkplan(orch)
    existing = {"job_id": "j-1", "stage": orch.STAGE_FAILED, "created_by": "ou_stale"}
    saved = []
    monkeypatch.setattr(orch, "get_job", lambda jid: existing)
    monkeypatch.setattr(orch, "_save", lambda job: saved.append(dict(job)))

    out = orch.create_job_record(plan, open_id="ou_fresh")
    assert out is not existing                     # 新建了记录
    assert out["stage"] == orch.STAGE_NEW
    assert out["created_by"] == "ou_fresh"


# ── 修② _h_confirm_* 确认 handler 回填 ─────────────────────────────────────────

# (confirm handler 名, orchestrator 模块路径)
_CONFIRMS = [
    ("_h_confirm_cpfs_dataflow", "core.cpfs_dataflow.orchestrator"),
    ("_h_confirm_vepfs_dataflow", "core.vepfs_dataflow.orchestrator"),
]


def _new_running_job(orch):
    # confirm 守卫要求 stage in (NEW, FAILED)
    return {"job_id": "j-c1", "stage": orch.STAGE_NEW, "created_by": "",
            "operation_label": "沉降", "cpfs_dir": "/cwr/o/", "oss_bucket": "bk",
            "oss_prefix": "e/", "sub_path": "/wzh/", "tos_bucket": "bk", "tos_prefix": "e/"}


@pytest.mark.parametrize("confirm_name,orch_path", _CONFIRMS, ids=_IDS)
def test_confirm_backfills_empty_created_by(monkeypatch, confirm_name, orch_path):
    """job.created_by 空 + 确认者 open_id 非空 → 置真值后 _save(RUNNING)。"""
    from core.feishu_bot import actions
    orch = _orch(orch_path)
    job = _new_running_job(orch)
    store = {job["job_id"]: job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(orch, "run_to_completion", lambda j, **k: j)
    # 不真起后台线程
    monkeypatch.setattr(actions.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda s: None})())

    out = getattr(actions, confirm_name)({"job_id": job["job_id"]}, "ou_confirmer", "chat", {})
    assert out["toast"]["type"] == "success"
    assert store[job["job_id"]]["created_by"] == "ou_confirmer"   # 回填
    assert store[job["job_id"]]["stage"] == orch.STAGE_RUNNING


@pytest.mark.parametrize("confirm_name,orch_path", _CONFIRMS, ids=_IDS)
def test_confirm_keeps_existing_created_by(monkeypatch, confirm_name, orch_path):
    """job.created_by 已有非空 → 确认不改（发起人优先于确认者）。"""
    from core.feishu_bot import actions
    orch = _orch(orch_path)
    job = _new_running_job(orch)
    job["created_by"] = "ou_creator"
    store = {job["job_id"]: job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(orch, "run_to_completion", lambda j, **k: j)
    monkeypatch.setattr(actions.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda s: None})())

    getattr(actions, confirm_name)({"job_id": job["job_id"]}, "ou_admin_confirmer", "chat", {})
    assert store[job["job_id"]]["created_by"] == "ou_creator"     # 不被确认者覆盖
