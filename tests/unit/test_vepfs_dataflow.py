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
    # cn-shanghai 真机反查：DataStorage 须裸桶名；DataStoragePath/SubPath 须首尾带斜杠
    assert req["data_storage"] == "bk"                 # 裸桶名（tos:// 会 InvalidParameter.BucketName）
    assert req["data_storage_path"] == "/out/"         # 补首斜杠 → 内部 SourceStoragePrefix 合法
    assert req["sub_path"] == "/ckpt/"
    assert req["same_name_file_policy"] == "OverWrite"
    assert req["data_type"] == "MetaAndData"


def test_data_storage_strips_scheme():
    assert engine_vepfs._data_storage("tos://bk") == "bk"
    assert engine_vepfs._data_storage("bk") == "bk"
    assert engine_vepfs._data_storage("tos://bk/") == "bk"


def test_norm_slash_dir_rules():
    # 非空 → 首尾都带斜杠；空 → 保留空（整桶根合法）
    assert engine_vepfs._norm_slash_dir("a/b") == "/a/b/"
    assert engine_vepfs._norm_slash_dir("/a/b/") == "/a/b/"
    assert engine_vepfs._norm_slash_dir("a/b/") == "/a/b/"
    assert engine_vepfs._norm_slash_dir("") == ""


def test_submit_task_normalizes_prefix_with_tos_scheme(monkeypatch):
    cap = {}
    fake = _FakeVepfs(cap)
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))
    engine_vepfs.submit_task(
        fs_id="vepfs-a", task_action=engine_vepfs.ACTION_IMPORT,
        tos_bucket="tos://bk", tos_prefix="wuji/x", sub_path="wuji/x", region="cn-shanghai")
    req = cap["create_req"]
    assert req["data_storage"] == "bk"
    assert req["data_storage_path"] == "/wuji/x/"
    assert req["sub_path"] == "/wuji/x/"


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

def test_entry_card_cloud_picker():
    """统一入口卡=两个云平台按钮（阿里/火山）。"""
    from core.dataflow_cards import entry_card
    acts = set()
    def _walk(o):
        if isinstance(o, dict):
            v = o.get("value")
            if isinstance(v, dict) and v.get("action"):
                acts.add(v["action"])
            for x in o.values():
                _walk(x)
        elif isinstance(o, list):
            for x in o:
                _walk(x)
    _walk(entry_card())
    assert {"pick_cloud_aliyun", "pick_cloud_volcano"} <= acts


def test_form_card_volcano_dropdown():
    """火山向导表单卡：fs 下拉 + 源/目的地址；提交 submit_vepfs_dataflow；无操作/桶选择器。"""
    from core.dataflow_cards import form_card
    fs = [{"value": "vepfs-a@cn-shanghai", "text": "wuji（cn-shanghai）"}]
    ec = form_card("volcano", "cn-shanghai", fs)
    form = ec["body"]["elements"][1]["elements"]
    names = {e.get("name") for e in form}
    assert {"fs", "source", "dest", "same_name"} <= names
    assert "operation" not in names and "bucket" not in names
    btn = [e for e in form if e.get("tag") == "button"][0]
    assert btn["behaviors"][0]["value"]["action"] == "submit_vepfs_dataflow"
    assert btn["behaviors"][0]["value"]["region"] == "cn-shanghai"


def test_guided_plan_direction_from_addresses():
    """_guided_plan：源=文件系统→沉降；源=对象存储→预热；两侧同类报错。"""
    from core.feishu_bot import actions
    # 源=vePFS目录、目的=TOS → 沉降(Export)
    p1 = actions._guided_plan("volcano",
        {"fs": "vepfs-a@cn-shanghai", "source": "/label/", "dest": "tos://bk/arch/", "same_name": "skip"})
    assert p1.operation == "sink" and p1.fs_id == "vepfs-a" and p1.region == "cn-shanghai"
    assert p1.tos_bucket == "bk" and p1.tos_prefix == "arch/"
    # 源=TOS、目的=vePFS目录 → 预热(Import)
    p2 = actions._guided_plan("volcano",
        {"fs": "vepfs-a@cn-shanghai", "source": "tos://bk/arch/", "dest": "/label/"})
    assert p2.operation == "preheat"
    # 两侧都是对象存储 → 报错
    with pytest.raises(orch.DataflowPathError):
        actions._guided_plan("volcano",
            {"fs": "vepfs-a@cn-shanghai", "source": "tos://a/x/", "dest": "tos://b/y/"})


def test_guided_plan_region_from_hint(monkeypatch):
    """fs 手填无 @region 时，用按钮带来的 region_hint。"""
    from core.feishu_bot import actions
    p = actions._guided_plan("volcano",
        {"fs": "vepfs-x", "source": "/d/", "dest": "tos://bk/d/"}, region_hint="cn-shanghai")
    assert p.fs_id == "vepfs-x" and p.region == "cn-shanghai"


def test_vepfs_keywords_open_unified_card():
    """火山/vepfs 关键词命中统一的 sink/preheat 入口意图（卡内选火山）。"""
    from core.feishu_bot import messages
    assert messages._is_sink_preheat_entry_intent("vepfs沉降")
    assert messages._is_sink_preheat_entry_intent("火山数据预热")
    assert messages._is_sink_preheat_entry_intent("数据沉降")   # 阿里入口不受影响
    assert not messages._is_sink_preheat_entry_intent("查一下 GPU")




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
