"""火山 vePFS 预热/沉降单测：方向解析 / 引擎字段映射 / query 提取 / orchestrator 状态机 / 意图卡片。"""
import pytest

from core.vepfs_dataflow import engine_vepfs, orchestrator as orch


# ── 计划 / 方向解析 ─────────────────────────────────────────────────────────────

def test_make_plan_preheat_import():
    p = orch.make_plan("preheat", "vepfs://vepfs-a/data/", "tos://bk/prefix/")
    assert p.operation == "preheat" and p.action == engine_vepfs.ACTION_IMPORT
    assert p.fs_id == "vepfs-a" and p.sub_path == "/data/"
    assert p.tos_bucket == "bk" and p.tos_prefix == "prefix/"


def test_make_plan_sink_export():
    p = orch.make_plan("sink", "vepfs://vepfs-a/ckpt/", "tos://bk/out/")
    assert p.operation == "sink" and p.action == engine_vepfs.ACTION_EXPORT


def test_plan_from_addresses_direction():
    # 源 vePFS → 目的 TOS = 沉降
    p1 = orch.plan_from_addresses("cn-beijing", "vepfs://fs-x/a/", "tos://bk/a/")
    assert p1.operation == "sink" and p1.fs_id == "fs-x"
    # 源 TOS → 目的 vePFS = 预热
    p2 = orch.plan_from_addresses("cn-beijing", "tos://bk/a/", "vepfs://fs-x/a/")
    assert p2.operation == "preheat"


def test_plan_from_addresses_both_tos_rejected():
    with pytest.raises(orch.DataflowPathError):
        orch.plan_from_addresses("cn-beijing", "tos://a/x/", "tos://b/x/")


def test_region_fs_multiple_requires_explicit(monkeypatch):
    """地区多个 vePFS → 裸路径报错，提示用 vepfs://<fs>/ 指定。"""
    monkeypatch.setattr(orch.settings, "VEPFS_FILE_SYSTEM_ID", "")
    monkeypatch.setattr(engine_vepfs, "list_filesystems", lambda region: [
        {"fs_id": "vepfs-1", "region": region, "name": "wuji-vepfs", "status": "Running"},
        {"fs_id": "vepfs-2", "region": region, "name": "data-D", "status": "Running"}])
    with pytest.raises(orch.DataflowPathError):
        orch.plan_from_addresses("cn-shanghai", "/label/", "tos://bk/label/")


def test_region_fs_single_auto(monkeypatch):
    monkeypatch.setattr(orch.settings, "VEPFS_FILE_SYSTEM_ID", "")
    monkeypatch.setattr(engine_vepfs, "list_filesystems", lambda region: [
        {"fs_id": "vepfs-only", "region": region, "name": "x", "status": "Running"}])
    p = orch.plan_from_addresses("cn-shanghai", "/label/", "tos://bk/label/")
    assert p.fs_id == "vepfs-only" and p.operation == "sink"


def test_default_fs_id_skips_discovery(monkeypatch):
    """配了 VEPFS_FILE_SYSTEM_ID → 裸路径直接用它，不触发 discovery。"""
    monkeypatch.setattr(orch.settings, "VEPFS_FILE_SYSTEM_ID", "vepfs-def")
    def _boom(region):
        raise AssertionError("不应调用 discovery")
    monkeypatch.setattr(engine_vepfs, "list_filesystems", _boom)
    p = orch.plan_from_addresses("cn-shanghai", "tos://bk/a/", "/label/")
    assert p.fs_id == "vepfs-def" and p.operation == "preheat"


def test_make_plan_bare_vepfs_needs_fs(monkeypatch):
    monkeypatch.setattr(orch.settings, "VEPFS_FILE_SYSTEM_ID", "")
    with pytest.raises(orch.DataflowPathError):
        orch.make_plan("sink", "/data/", "tos://bk/x/")


def test_make_plan_bare_vepfs_uses_default_fs(monkeypatch):
    monkeypatch.setattr(orch.settings, "VEPFS_FILE_SYSTEM_ID", "vepfs-def")
    p = orch.make_plan("sink", "/data/", "tos://bk/x/")
    assert p.fs_id == "vepfs-def" and p.sub_path == "/data/"


# ── 同名策略 / 状态归类 ─────────────────────────────────────────────────────────

def test_overwrite_policy_mapping():
    assert engine_vepfs._overwrite_policy("overwrite") == "OverWrite"
    assert engine_vepfs._overwrite_policy("KeepLatest") == "KeepLatest"
    assert engine_vepfs._overwrite_policy("") == "Skip"
    assert engine_vepfs._overwrite_policy("skip") == "Skip"


def test_status_classification():
    assert engine_vepfs.is_done("Success") and engine_vepfs.is_done("Finished")
    assert engine_vepfs.is_failed("Failed") and engine_vepfs.is_failed("Canceled")
    assert not engine_vepfs.is_done("Running") and not engine_vepfs.is_failed("Running")


# ── 引擎字段映射 ────────────────────────────────────────────────────────────────

def test_submit_task_field_mapping(monkeypatch):
    cap = {}
    fake = _FakeVepfs(cap)
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))
    tid = engine_vepfs.submit_task(
        fs_id="vepfs-a", task_action=engine_vepfs.ACTION_EXPORT,
        tos_bucket="bk", tos_prefix="out/", sub_path="/ckpt/", region="cn-beijing",
        same_name_policy="overwrite")
    assert tid == "555"
    req = cap["create_req"]
    assert req["file_system_id"] == "vepfs-a"
    assert req["task_action"] == "Export"
    assert req["data_storage"] == "tos://bk"          # 默认加 tos:// 前缀
    assert req["data_storage_path"] == "out/"
    assert req["sub_path"] == "/ckpt/"
    assert req["same_name_file_policy"] == "OverWrite"
    assert req["data_type"] == "MetaAndData"


def test_submit_task_rejects_bad_action():
    with pytest.raises(engine_vepfs.VepfsDataflowError):
        engine_vepfs.submit_task(fs_id="f", task_action="Bogus", tos_bucket="b", region="r")


def test_query_task_extracts_progress_and_error(monkeypatch):
    class _Task:
        data_flow_task_id = 555
        status = "Failed"
        total_size = 1024
        exec_size = 512
        total_count = 10
        exec_count = 4
        failed_count = 3
    class _Resp:
        data_flow_tasks = [_Task()]
    fake = _FakeVepfs({}, query_resp=_Resp())
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))
    st = engine_vepfs.query_task("555", "vepfs-a", "cn-beijing")
    assert st["status"] == "Failed"
    assert st["bytes_total"] == 1024 and st["bytes_done"] == 512
    assert st["files_total"] == 10 and st["files_done"] == 4
    assert "3 个对象处理失败" in st["error"]


# ── orchestrator 状态机 ────────────────────────────────────────────────────────

def test_orch_run_to_completion_done(monkeypatch):
    plan = orch.make_plan("preheat", "vepfs://fs-a/d/", "tos://bk/d/")
    job = orch.create_job_record(plan, open_id="ou_1")
    assert job["stage"] == orch.STAGE_NEW
    monkeypatch.setattr(engine_vepfs, "submit_task", lambda **k: "777")
    monkeypatch.setattr(engine_vepfs, "query_task",
                        lambda *a, **k: {"status": "Success", "bytes_done": 10, "files_done": 2})
    job = orch.run_to_completion(job, poll_interval=0, max_polls=3)
    assert job["stage"] == orch.STAGE_DONE and job["files_done"] == 2 and job["task_id"] == "777"


def test_orch_run_to_completion_failure(monkeypatch):
    plan = orch.make_plan("sink", "vepfs://fs-a/d/", "tos://bk/d/")
    job = orch.create_job_record(plan, open_id="ou_1")
    monkeypatch.setattr(engine_vepfs, "submit_task", lambda **k: "778")
    monkeypatch.setattr(engine_vepfs, "query_task",
                        lambda *a, **k: {"status": "Failed", "error": "boom"})
    job = orch.run_to_completion(job, poll_interval=0, max_polls=3)
    assert job["stage"] == orch.STAGE_FAILED and "boom" in job["error"]


def test_orch_submit_failure_sets_failed(monkeypatch):
    plan = orch.make_plan("sink", "vepfs://fs-a/d/", "tos://bk/d/")
    job = orch.create_job_record(plan, open_id="ou_1")
    def boom(**k):
        raise engine_vepfs.VepfsDataflowError("提交失败")
    monkeypatch.setattr(engine_vepfs, "submit_task", boom)
    job = orch.run_to_completion(job, poll_interval=0, max_polls=2)
    assert job["stage"] == orch.STAGE_FAILED and "提交失败" in job["error"]


# ── 意图 + 卡片 ─────────────────────────────────────────────────────────────────

def test_unified_entry_card_has_cloud_selector():
    """统一「数据预热/沉降」卡含云平台(阿里/火山)下拉 + 源/目的/同名，submit=submit_cpfs_dataflow。"""
    from core.cpfs_dataflow import cards
    ec = cards.entry_card()
    form = ec["body"]["elements"][1]["elements"]
    names = {e.get("name") for e in form}
    assert {"cloud", "region", "source", "dest", "same_name"} <= names
    cloud = next(e for e in form if e.get("name") == "cloud")
    vals = {o["value"] for o in cloud["options"]}
    assert vals == {"aliyun", "volcano"}
    btn = [e for e in form if e.get("tag") == "button"][0]
    assert btn["behaviors"][0]["value"]["action"] == "submit_cpfs_dataflow"


def test_vepfs_keywords_open_unified_card():
    """火山/vepfs 关键词命中统一的 sink/preheat 入口意图（卡内选火山）。"""
    from core.feishu_bot import messages
    assert messages._is_sink_preheat_entry_intent("vepfs沉降")
    assert messages._is_sink_preheat_entry_intent("火山数据预热")
    assert messages._is_sink_preheat_entry_intent("数据沉降")   # 阿里入口不受影响
    assert not messages._is_sink_preheat_entry_intent("查一下 GPU")


def test_submit_routes_volcano_to_vepfs(monkeypatch):
    """统一卡提交：cloud=volcano → 委派给 vePFS 处理器。"""
    from core.feishu_bot import actions
    called = {}
    monkeypatch.setattr(actions, "_h_submit_vepfs_dataflow",
                        lambda av, oid, cid, fv: called.setdefault("vepfs", True) or {"toast": {}})
    actions._h_submit_cpfs_dataflow({}, "ou_1", "oc_1",
                                    {"cloud": "volcano", "region": "cn-beijing",
                                     "source": "vepfs://fs/a/", "dest": "tos://bk/a/"})
    assert called.get("vepfs")


# ── fakes ──────────────────────────────────────────────────────────────────────

class _FakeVepfs:
    """同时充当 vepfs 模块（构造 request）和 api（create/query/cancel）。"""
    def __init__(self, cap, query_resp=None):
        self.cap = cap
        self._query_resp = query_resp

    # request 构造器
    def CreateDataFlowTaskRequest(self, **kw):
        self.cap["create_req"] = kw
        return kw

    def DescribeDataFlowTasksRequest(self, **kw):
        self.cap["query_req"] = kw
        return kw

    def CancelDataFlowTaskRequest(self, **kw):
        return kw

    # api
    def create_data_flow_task(self, req):
        return type("R", (), {"data_flow_task_id": 555})()

    def describe_data_flow_tasks(self, req):
        return self._query_resp
