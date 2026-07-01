"""engine_nas 单测：patch _call 喂 canned 响应，验证解析 / 版本分支 / DataFlow 解析。"""
import json

import pytest

from core.cpfs_dataflow import engine_nas as eng


def test_edition_by_prefix():
    assert eng.edition("bmcpfs-abc") == "computing"
    assert eng.edition("cpfs-abc") == "general"
    assert eng.edition("") == "general"


def test_normalize_dir():
    assert eng.normalize_dir("dataset") == "/dataset/"
    assert eng.normalize_dir("/dataset") == "/dataset/"
    assert eng.normalize_dir("/dataset/") == "/dataset/"
    assert eng.normalize_dir("") == "/"


def test_list_dataflows_parses_nested(monkeypatch):
    body = {"DataFlowInfo": {"DataFlow": [
        {"DataFlowId": "df-1", "SourceStorage": "oss://bk-a", "FileSystemPath": "/a/", "Status": "Running"},
        {"DataFlowId": "df-2", "SourceStorage": "oss://bk-b", "FileSystemPath": "/b/", "Status": "Stopped"},
    ]}}
    monkeypatch.setattr(eng, "_call", lambda *a, **k: body)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    flows = eng.list_dataflows("bmcpfs-x", "cn-hangzhou")
    assert [f["data_flow_id"] for f in flows] == ["df-1", "df-2"]
    assert flows[0]["source_storage"] == "oss://bk-a"


def test_resolve_dataflow_by_bucket(monkeypatch):
    body = {"DataFlow": [
        {"DataFlowId": "df-1", "SourceStorage": "oss://bk-a", "FileSystemPath": "/a/"},
        {"DataFlowId": "df-2", "SourceStorage": "oss://bk-b", "FileSystemPath": "/b/"},
    ]}
    monkeypatch.setattr(eng, "_call", lambda *a, **k: body)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    df = eng.resolve_dataflow("bmcpfs-x", "cn-hangzhou", oss_bucket="bk-b")
    assert df["data_flow_id"] == "df-2"


def test_resolve_dataflow_map_override(monkeypatch):
    monkeypatch.setattr(eng.settings, "CPFS_DATAFLOW_MAP_RAW", json.dumps({"oss://bk-z": "df-z"}))
    # 不应触发 list_dataflows
    monkeypatch.setattr(eng, "_call", lambda *a, **k: pytest.fail("should not call API"))
    df = eng.resolve_dataflow("bmcpfs-x", "cn-hangzhou", oss_bucket="bk-z")
    assert df["data_flow_id"] == "df-z"


def test_resolve_dataflow_ambiguous_raises(monkeypatch):
    body = {"DataFlow": [
        {"DataFlowId": "df-1", "SourceStorage": "oss://bk-a"},
        {"DataFlowId": "df-2", "SourceStorage": "oss://bk-b"},
    ]}
    monkeypatch.setattr(eng, "_call", lambda *a, **k: body)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    monkeypatch.setattr(eng.settings, "CPFS_DATAFLOW_MAP_RAW", "{}")
    with pytest.raises(eng.NasDataflowError):
        eng.resolve_dataflow("bmcpfs-x", "cn-hangzhou")  # 两条且无法区分


def test_submit_task_computing_adds_conflict_policy(monkeypatch):
    captured = {}

    def fake_call(client, action, query):
        captured["action"] = action
        captured["query"] = query
        return {"TaskId": "task-1", "RequestId": "r"}

    monkeypatch.setattr(eng, "_call", fake_call)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    tid = eng.submit_task(fs_id="bmcpfs-x", data_flow_id="df-1", action="Export",
                          directory="/out", region="cn-hangzhou")
    assert tid == "task-1"
    assert captured["action"] == "CreateDataFlowTask"
    q = captured["query"]
    assert q["TaskAction"] == "Export"
    assert q["Directory"] == "/out/"
    assert q["ConflictPolicy"]            # 智算版必带
    assert q["DataType"] == "MetaAndData"


def test_submit_task_general_no_conflict_policy(monkeypatch):
    captured = {}
    monkeypatch.setattr(eng, "_call", lambda c, a, q: captured.update(query=q) or {"TaskId": "t2"})
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    tid = eng.submit_task(fs_id="cpfs-x", data_flow_id="df-1", action="Import",
                          directory="/d", region="cn-hangzhou")
    assert tid == "t2"
    assert "ConflictPolicy" not in captured["query"]   # 通用版不下发


def test_submit_task_rejects_bad_action(monkeypatch):
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    with pytest.raises(eng.NasDataflowError):
        eng.submit_task(fs_id="cpfs-x", data_flow_id="df", action="Evict", directory="/d")


def test_query_task_parses_progress(monkeypatch):
    body = {"DataFlowTask": [{
        "TaskId": "task-1", "Status": "Executing",
        "FilesTotal": 100, "FilesDone": 40, "BytesTotal": 1000, "BytesDone": 400,
    }]}
    monkeypatch.setattr(eng, "_call", lambda *a, **k: body)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    st = eng.query_task("bmcpfs-x", "task-1", "cn-hangzhou")
    assert st["status"] == "Executing"
    assert st["files_done"] == 40
    assert st["bytes_total"] == 1000


def test_create_dataflow_computing(monkeypatch):
    captured = {}
    monkeypatch.setattr(eng, "_call",
                        lambda c, a, q: captured.update(action=a, query=q) or {"DataFlowId": "df-x"})
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    dfid = eng.create_dataflow("bmcpfs-a", "cn-hangzhou", oss_bucket="bk",
                               oss_path="/wuji_il/", fs_path="/cwr/")
    assert dfid == "df-x" and captured["action"] == "CreateDataFlow"
    q = captured["query"]
    assert q["SourceStorage"] == "oss://bk"
    assert q["FileSystemPath"] == "/cwr/" and q["SourceStoragePath"] == "/wuji_il/"


def test_create_dataflow_general_rejected(monkeypatch):
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    with pytest.raises(eng.NasDataflowError):
        eng.create_dataflow("cpfs-general", "cn-hangzhou", oss_bucket="bk")


def test_wait_dataflow_running_ok(monkeypatch):
    monkeypatch.setattr(eng, "_call", lambda *a, **k: {"DataFlows": {"DataFlow": [
        {"DataFlowId": "df-1", "Status": "Running"}]}})
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    eng.wait_dataflow_running("bmcpfs-a", "cn-hangzhou", "df-1", retries=1)   # 不抛即通过


def test_wait_dataflow_misconfigured_raises(monkeypatch):
    monkeypatch.setattr(eng, "_call", lambda *a, **k: {"DataFlows": {"DataFlow": [
        {"DataFlowId": "df-1", "Status": "Misconfigured", "ErrorMessage": "SourceStorageUnreachable"}]}})
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    with pytest.raises(eng.NasDataflowError):
        eng.wait_dataflow_running("bmcpfs-a", "cn-hangzhou", "df-1", retries=1)


def test_list_filesystems_filters_cpfs(monkeypatch):
    body = {"FileSystems": {"FileSystem": [
        {"FileSystemId": "bmcpfs-a"}, {"FileSystemId": "cpfs-b"},
        {"FileSystemId": "extreme-c"}, {"FileSystemId": "005494-nas"}]}}
    monkeypatch.setattr(eng, "_call", lambda *a, **k: body)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    ids = eng.list_filesystems("cn-hangzhou")
    assert ids == ["bmcpfs-a", "cpfs-b"]    # 仅 cpfs-/bmcpfs- 前缀


def test_poll_sink_ref_split(monkeypatch):
    body = {"DataFlowTask": [{"TaskId": "task-9", "Status": "Completed",
                              "BytesDone": 2048, "FilesDone": 3}]}
    monkeypatch.setattr(eng, "_call", lambda *a, **k: body)
    monkeypatch.setattr(eng, "_client", lambda *a, **k: object())
    st = eng.poll_sink("bmcpfs-x", "task-9@df-1", "cn-hangzhou")
    assert st["status"] == "Completed"
    assert st["bytes"] == 2048 and st["objects"] == 3
