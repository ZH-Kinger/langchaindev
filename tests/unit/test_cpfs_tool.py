"""manage_cpfs_dataflow 工具测试 + 注册校验。"""
from core.cpfs_dataflow import engine_nas, orchestrator
from tools.cpfs import cpfs_dataflow_tool


def test_tool_registered_in_all_tools():
    import tools
    names = {t.name for t in tools.ALL_TOOLS}
    assert "manage_cpfs_dataflow" in names
    assert tools.TOOL_GROUPS["cpfs"] == {"manage_cpfs_dataflow"}


def test_intent_key_matches_tool_group():
    from core.intent_router import INTENT_DESCRIPTIONS
    import tools
    # INTENT_DESCRIPTIONS 的 key 必须都在 TOOL_GROUPS 里
    assert "cpfs" in INTENT_DESCRIPTIONS
    assert "cpfs" in tools.TOOL_GROUPS


def test_tool_list_action(monkeypatch):
    monkeypatch.setattr(engine_nas, "list_dataflows",
                        lambda fs, **k: [{"data_flow_id": "df-1", "status": "Running",
                                          "source_storage": "oss://bk", "fs_path": "/a/"}])
    out = cpfs_dataflow_tool.func(action="list", fs_id="bmcpfs-x")
    assert "df-1" in out and "Running" in out


def test_tool_sink_submits(monkeypatch):
    monkeypatch.setattr(engine_nas, "resolve_dataflow", lambda *a, **k: {"data_flow_id": "df"})
    monkeypatch.setattr(engine_nas, "submit_task", lambda **k: "task-1")
    # 阻断后台真正轮询
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    out = cpfs_dataflow_tool.func(action="sink", cpfs_path="cpfs://bmcpfs-x/o/", oss="oss://bk/e/")
    assert "已提交" in out and "沉降" in out


def test_tool_bad_path():
    out = cpfs_dataflow_tool.func(action="sink", cpfs_path="")
    assert "❌" in out


def test_tool_status_not_found():
    out = cpfs_dataflow_tool.func(action="status", job_id="cpfs-nope")
    assert "未找到" in out
