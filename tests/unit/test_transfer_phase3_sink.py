"""transfer 三期：全闪源 CPFS 先沉降(SINKING)再跨云(CROSSING)。"""
import json

from core.transfer import orchestrator as torch
from core.transfer import engine_tos
from core.cpfs_dataflow import engine_nas


def _setup_map(monkeypatch):
    monkeypatch.setattr(torch.settings, "TRANSFER_BUCKET_MAP_RAW",
                        json.dumps({"cpfs://fs1": "sinkoss", "oss://sinkoss": "desttos"}))
    monkeypatch.setattr(torch.settings, "CPFS_REGION", "cn-hangzhou")


def test_plan_cpfs_needs_sink(monkeypatch):
    _setup_map(monkeypatch)
    plan = torch.make_plan("cpfs://fs1/data/", "")
    assert plan.needs_sink
    assert plan.sink_target.scheme == "oss" and plan.sink_target.bucket == "sinkoss"
    assert plan.dest.scheme == "tos" and plan.dest.bucket == "desttos"
    assert plan.engine == "tos_mig"


def test_run_to_completion_sink_then_cross(monkeypatch):
    _setup_map(monkeypatch)
    plan = torch.make_plan("cpfs://fs1/data/", "")
    job = torch.create_job_record(plan)

    monkeypatch.setattr(engine_nas, "submit_sink_job", lambda **k: "task@df")
    monkeypatch.setattr(engine_nas, "poll_sink",
                        lambda *a, **k: {"status": "Completed", "bytes": 100, "objects": 2})
    monkeypatch.setattr(engine_tos, "submit_cross_job", lambda **k: "cross-1")
    done = list(engine_tos._DONE_STATES)[0]
    monkeypatch.setattr(engine_tos, "poll_job",
                        lambda *a, **k: {"status": done, "bytes": 100, "objects": 2})

    stages = []
    job = torch.run_to_completion(job, on_update=lambda j: stages.append(j["stage"]),
                                  poll_interval=0, max_polls=5)
    assert job["stage"] == torch.STAGE_DONE
    assert torch.STAGE_SINKING in stages and torch.STAGE_CROSSING in stages
    assert job.get("sink_done") and job["sink_task_id"] == "task@df"


def test_sink_failure_short_circuits(monkeypatch):
    _setup_map(monkeypatch)
    plan = torch.make_plan("cpfs://fs1/data/", "")
    job = torch.create_job_record(plan)
    monkeypatch.setattr(engine_nas, "submit_sink_job", lambda **k: "task@df")
    monkeypatch.setattr(engine_nas, "poll_sink",
                        lambda *a, **k: {"status": "Failed", "error": "export boom"})
    # 跨云不应被触发
    monkeypatch.setattr(engine_tos, "submit_cross_job",
                        lambda **k: (_ for _ in ()).throw(AssertionError("cross should not run")))
    job = torch.run_to_completion(job, poll_interval=0, max_polls=5)
    assert job["stage"] == torch.STAGE_FAILED and "export boom" in job["error"]


def test_vepfs_sink_not_implemented(monkeypatch):
    monkeypatch.setattr(torch.settings, "TRANSFER_BUCKET_MAP_RAW",
                        json.dumps({"vepfs://vfs": "sinktos", "tos://sinktos": "destoss"}))
    plan = torch.make_plan("vepfs://vfs/data/", "")
    job = torch.create_job_record(plan)
    job = torch.start_sinking(job)
    assert job["stage"] == torch.STAGE_FAILED and "vepfs" in job["error"]
