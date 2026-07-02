"""orchestrator 单测：路径解析 / 计划定向 / 状态机（fakeredis + mock engine）。"""
import pytest

from core.cpfs_dataflow import orchestrator as orch
from core.cpfs_dataflow import engine_nas


def test_make_plan_sink_orientation():
    plan = orch.make_plan("sink", "cpfs://bmcpfs-x/outputs/", "oss://bk/exports/")
    assert plan.operation == "sink" and plan.action == "Export"
    assert plan.fs_id == "bmcpfs-x" and plan.cpfs_dir == "/outputs/"
    assert plan.oss_bucket == "bk" and plan.oss_prefix == "exports/"
    assert plan.directory == "/outputs/"        # Export: CPFS 侧
    assert plan.dst_directory == "/exports/"     # → OSS 侧


def test_make_plan_preheat_orientation():
    plan = orch.make_plan("预热", "cpfs://bmcpfs-x/dataset/", "oss://bk/raw/")
    assert plan.action == "Import"
    assert plan.directory == "/raw/"             # Import: OSS 侧
    assert plan.dst_directory == "/dataset/"     # → CPFS 侧


def test_make_plan_bare_dir_needs_fs(monkeypatch):
    monkeypatch.setattr(orch.settings, "CPFS_FILE_SYSTEM_ID", "")
    with pytest.raises(orch.DataflowPathError):
        orch.make_plan("sink", "/outputs/")
    monkeypatch.setattr(orch.settings, "CPFS_FILE_SYSTEM_ID", "bmcpfs-default")
    plan = orch.make_plan("sink", "/outputs/")
    assert plan.fs_id == "bmcpfs-default"


def test_make_plan_full_path_strips_mount(monkeypatch):
    monkeypatch.setattr(orch.settings, "CPFS_FILE_SYSTEM_ID", "bmcpfs-default")
    monkeypatch.setattr(orch.settings, "CPFS_MOUNT_PREFIX", "/cpfs")
    plan = orch.make_plan("preheat", "/cpfs/cwr/third_party_data/label",
                          "oss://wuji-bucket-hangzhou/wuji_il")
    assert plan.fs_id == "bmcpfs-default"
    assert plan.cpfs_dir == "/cwr/third_party_data/label/"   # 去掉 /cpfs 挂载前缀
    assert plan.oss_bucket == "wuji-bucket-hangzhou" and plan.oss_prefix == "wuji_il/"
    # 预热(Import)：Directory=OSS 侧，DstDirectory=CPFS 侧
    assert plan.directory == "/wuji_il/"
    assert plan.dst_directory == "/cwr/third_party_data/label/"


def test_plan_from_addresses_sink(monkeypatch):
    monkeypatch.setattr(engine_nas, "list_filesystems", lambda region, **k: ["bmcpfs-a"])
    # plan 阶段不建流：create_dataflow 不应被调用
    monkeypatch.setattr(engine_nas, "create_dataflow",
                        lambda *a, **k: pytest.fail("plan 阶段不应建流"))
    plan = orch.plan_from_addresses("cn-hangzhou", "/cpfs/cwr/label/", "oss://bk/wuji_il/")
    assert plan.operation == "sink" and plan.action == "Export"
    assert plan.fs_id == "bmcpfs-a" and plan.data_flow_id == ""     # 延后到 start_task 建
    assert plan.cpfs_dir == "/cwr/label/"          # 去 /cpfs 挂载前缀
    assert plan.oss_bucket == "bk" and plan.oss_prefix == "wuji_il/"


def test_plan_from_addresses_preheat(monkeypatch):
    monkeypatch.setattr(engine_nas, "list_filesystems", lambda region, **k: ["bmcpfs-a"])
    plan = orch.plan_from_addresses("cn-hangzhou", "oss://bk/raw/", "/cpfs/cwr/dataset/")
    assert plan.operation == "preheat" and plan.action == "Import"
    assert plan.cpfs_dir == "/cwr/dataset/" and plan.oss_bucket == "bk"


def test_start_task_creates_ephemeral_dataflow(monkeypatch):
    monkeypatch.setattr(engine_nas, "list_filesystems", lambda region, **k: ["bmcpfs-a"])
    plan = orch.plan_from_addresses("cn-hangzhou", "/cpfs/cwr/o/", "oss://bk/e/")
    job = orch.create_job_record(plan)
    monkeypatch.setattr(engine_nas, "create_dataflow", lambda *a, **k: "df-tmp")
    monkeypatch.setattr(engine_nas, "wait_dataflow_running", lambda *a, **k: None)
    monkeypatch.setattr(engine_nas, "submit_task", lambda **k: "task-1")
    job = orch.start_task(job)
    assert job["stage"] == orch.STAGE_RUNNING
    assert job["data_flow_id"] == "df-tmp" and job["dataflow_ephemeral"] is True
    assert job["task_id"] == "task-1"


def test_run_to_completion_deletes_ephemeral_dataflow(monkeypatch):
    monkeypatch.setattr(engine_nas, "list_filesystems", lambda region, **k: ["bmcpfs-a"])
    plan = orch.plan_from_addresses("cn-hangzhou", "/cpfs/cwr/o/", "oss://bk/e/")
    job = orch.create_job_record(plan)
    deleted = {}
    monkeypatch.setattr(engine_nas, "create_dataflow", lambda *a, **k: "df-tmp")
    monkeypatch.setattr(engine_nas, "wait_dataflow_running", lambda *a, **k: None)
    monkeypatch.setattr(engine_nas, "submit_task", lambda **k: "task-1")
    monkeypatch.setattr(engine_nas, "query_task",
                        lambda *a, **k: {"status": "Completed", "files_done": 1, "bytes_done": 9,
                                         "files_total": 1, "bytes_total": 9, "error": ""})
    monkeypatch.setattr(engine_nas, "delete_dataflow",
                        lambda fs, region, dfid, **k: deleted.update(df=dfid))
    job = orch.run_to_completion(job, poll_interval=0, max_polls=3)
    assert job["stage"] == orch.STAGE_DONE
    assert deleted.get("df") == "df-tmp" and job.get("dataflow_deleted") is True


def test_plan_from_addresses_bad_pair():
    with pytest.raises(orch.DataflowPathError):
        orch.plan_from_addresses("cn-hangzhou", "oss://a/", "oss://b/")   # 两个都是 oss


def test_plan_from_addresses_needs_region(monkeypatch):
    monkeypatch.setattr(orch.settings, "CPFS_REGION", "")
    with pytest.raises(orch.DataflowPathError):
        orch.plan_from_addresses("", "/cpfs/x/", "oss://b/")


def test_plan_from_addresses_multi_fs_requires_explicit(monkeypatch):
    """地区有多个 CPFS 且路径没写 cpfs://<fs>/ → 报错要求明确指定，不盲选。"""
    monkeypatch.setattr(engine_nas, "list_filesystems", lambda region, **k: ["bmcpfs-a", "bmcpfs-b"])
    with pytest.raises(orch.DataflowPathError):
        orch.plan_from_addresses("cn-hangzhou", "/cpfs/x/", "oss://b/p/")


def test_plan_from_addresses_multi_fs_explicit_ok(monkeypatch):
    """多 CPFS 时，显式 cpfs://<fs>/ 直接用该 fs，不触发歧义报错。"""
    monkeypatch.setattr(engine_nas, "list_filesystems", lambda region, **k: ["bmcpfs-a", "bmcpfs-b"])
    plan = orch.plan_from_addresses("cn-hangzhou", "cpfs://bmcpfs-b/x/", "oss://b/p/")
    assert plan.fs_id == "bmcpfs-b"


def test_make_plan_bad_operation():
    with pytest.raises(orch.DataflowPathError):
        orch.make_plan("evict", "cpfs://bmcpfs-x/d/")


def test_create_job_record_idempotent(monkeypatch):
    plan = orch.make_plan("sink", "cpfs://bmcpfs-x/o/", "oss://bk/e/")
    j1 = orch.create_job_record(plan, open_id="ou_1")
    j2 = orch.create_job_record(plan, open_id="ou_1")
    assert j1["job_id"] == j2["job_id"]
    assert j1["stage"] == orch.STAGE_NEW


def test_start_task_sets_running(monkeypatch):
    plan = orch.make_plan("sink", "cpfs://bmcpfs-x/o/", "oss://bk/e/")
    job = orch.create_job_record(plan)
    monkeypatch.setattr(engine_nas, "create_dataflow", lambda *a, **k: "df-1")
    monkeypatch.setattr(engine_nas, "wait_dataflow_running", lambda *a, **k: None)
    monkeypatch.setattr(engine_nas, "submit_task", lambda **k: "task-1")
    job = orch.start_task(job)
    assert job["stage"] == orch.STAGE_RUNNING
    assert job["data_flow_id"] == "df-1" and job["task_id"] == "task-1"


def test_start_task_failure(monkeypatch):
    plan = orch.make_plan("sink", "cpfs://bmcpfs-x/o/", "oss://bk/e/")
    job = orch.create_job_record(plan)
    def boom(*a, **k):
        raise engine_nas.NasDataflowError("create failed")
    monkeypatch.setattr(engine_nas, "create_dataflow", boom)
    job = orch.start_task(job)
    assert job["stage"] == orch.STAGE_FAILED and "create failed" in job["error"]


def test_poll_once_done(monkeypatch):
    plan = orch.make_plan("preheat", "cpfs://bmcpfs-x/d/", "oss://bk/r/")
    job = orch.create_job_record(plan)
    job["stage"], job["task_id"] = orch.STAGE_RUNNING, "task-1"
    monkeypatch.setattr(engine_nas, "query_task", lambda *a, **k: {
        "status": "Completed", "files_done": 5, "bytes_done": 100,
        "files_total": 5, "bytes_total": 100, "error": ""})
    job = orch.poll_once(job)
    assert job["stage"] == orch.STAGE_DONE and job["bytes_done"] == 100


def test_poll_once_failed(monkeypatch):
    plan = orch.make_plan("preheat", "cpfs://bmcpfs-x/d/", "oss://bk/r/")
    job = orch.create_job_record(plan)
    job["stage"], job["task_id"] = orch.STAGE_RUNNING, "task-1"
    monkeypatch.setattr(engine_nas, "query_task", lambda *a, **k: {
        "status": "Failed", "error": "boom"})
    job = orch.poll_once(job)
    assert job["stage"] == orch.STAGE_FAILED and job["error"] == "boom"


def test_run_to_completion(monkeypatch):
    plan = orch.make_plan("sink", "cpfs://bmcpfs-x/o/", "oss://bk/e/")
    job = orch.create_job_record(plan)
    monkeypatch.setattr(engine_nas, "create_dataflow", lambda *a, **k: "df")
    monkeypatch.setattr(engine_nas, "wait_dataflow_running", lambda *a, **k: None)
    monkeypatch.setattr(engine_nas, "submit_task", lambda **k: "task-1")
    monkeypatch.setattr(engine_nas, "query_task", lambda *a, **k: {
        "status": "Completed", "files_done": 1, "bytes_done": 10,
        "files_total": 1, "bytes_total": 10, "error": ""})
    monkeypatch.setattr(engine_nas, "delete_dataflow", lambda *a, **k: None)
    seen = []
    job = orch.run_to_completion(job, on_update=lambda j: seen.append(j["stage"]),
                                 poll_interval=0, max_polls=3)
    assert job["stage"] == orch.STAGE_DONE
    assert orch.STAGE_RUNNING in seen and orch.STAGE_DONE in seen
