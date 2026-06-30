"""CPFS 飞书接线：意图、卡片结构、action handler。"""
from core.feishu_bot import messages, actions
from core.cpfs_dataflow import cards, orchestrator


def test_sink_preheat_intent_keywords():
    assert messages._is_sink_preheat_entry_intent("数据预热")
    assert messages._is_sink_preheat_entry_intent("数据沉降")
    assert messages._is_sink_preheat_entry_intent("cpfs沉降")
    assert not messages._is_sink_preheat_entry_intent("查一下 GPU 利用率")


def test_cards_build(monkeypatch):
    from core.cpfs_dataflow import discovery
    monkeypatch.setattr(discovery, "get_options", lambda *a, **k: [])   # 测回退手填表单
    ec = cards.entry_card()
    assert ec["schema"] == "2.0"
    # 回退表单含 operation/cpfs_path/oss
    form = ec["body"]["elements"][1]
    names = {e.get("name") for e in form["elements"]}
    assert {"operation", "cpfs_path", "oss"} <= names

    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "cpfs://bmcpfs-x/o/", "oss://bk/e/"))
    cc = cards.confirm_card(job)
    # 确认卡按钮回调 action 正确
    assert _find_action(cc) == "confirm_cpfs_dataflow"
    job["stage"] = "DONE"
    assert cards.result_card(job)["header"]["template"] == "green"
    job["stage"] = "FAILED"
    assert cards.result_card(job)["header"]["template"] == "red"


def _find_action(obj):
    if isinstance(obj, dict):
        if obj.get("type") == "callback" and isinstance(obj.get("value"), dict):
            return obj["value"].get("action")
        for v in obj.values():
            r = _find_action(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_action(v)
            if r:
                return r
    return None


def test_submit_handler_requires_path():
    out = actions._h_submit_cpfs_dataflow({}, "ou_1", "chat", {"operation": "sink"})
    assert out["toast"]["type"] == "error"


def test_submit_handler_accepts(monkeypatch):
    import core.dsw_scheduler as sch
    monkeypatch.setattr(sch, "_send_card", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sch, "_send_text", lambda *a, **k: None, raising=False)
    out = actions._h_submit_cpfs_dataflow(
        {}, "ou_1", "chat",
        {"operation": "sink", "cpfs_path": "cpfs://bmcpfs-x/o/", "oss": "oss://bk/e/"})
    assert out["toast"]["type"] == "success"


def test_confirm_handler_missing_job():
    out = actions._h_confirm_cpfs_dataflow({"job_id": "cpfs-none"}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "error"


def test_entry_card_with_options(monkeypatch):
    from core.cpfs_dataflow import discovery
    monkeypatch.setattr(discovery, "get_options",
                        lambda *a, **k: [{"label": "cn-hangzhou·bmcpfs-a·oss://bk/wuji_il ↔ /cwr/",
                                          "value": '{"fs_id":"bmcpfs-a"}'}])
    ec = cards.entry_card()
    form = ec["body"]["elements"][1]
    names = {e.get("name") for e in form["elements"]}
    assert "target" in names and "subdir" in names    # 有绑定→下拉选择


def test_submit_handler_with_selection(monkeypatch):
    import json
    import core.dsw_scheduler as sch
    monkeypatch.setattr(sch, "_send_card", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sch, "_send_text", lambda *a, **k: None, raising=False)
    sel = {"fs_id": "bmcpfs-a", "region": "cn-hangzhou", "data_flow_id": "df-1",
           "oss_bucket": "bk", "oss_prefix": "wuji_il", "fs_path": "/cwr/"}
    out = actions._h_submit_cpfs_dataflow(
        {}, "ou_1", "chat",
        {"operation": "sink", "target": json.dumps(sel), "subdir": "third_party_data/label"})
    assert out["toast"]["type"] == "success"


def test_confirm_handler_launches(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "cpfs://bmcpfs-y/o2/", "oss://bk/e2/"))
    out = actions._h_confirm_cpfs_dataflow({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "success"
