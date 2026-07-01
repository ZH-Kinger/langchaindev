"""CPFS 飞书接线：意图、卡片结构、地址→建流 handler。"""
from core.feishu_bot import messages, actions
from core.cpfs_dataflow import cards, orchestrator, engine_nas


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


def test_sink_preheat_intent_keywords():
    assert messages._is_sink_preheat_entry_intent("数据预热")
    assert messages._is_sink_preheat_entry_intent("数据沉降")
    assert messages._is_sink_preheat_entry_intent("cpfs沉降")
    assert not messages._is_sink_preheat_entry_intent("查一下 GPU 利用率")


def test_entry_card_addresses(monkeypatch):
    from core.cpfs_dataflow import discovery
    monkeypatch.setattr(discovery, "regions", lambda *a, **k: ["cn-hangzhou", "cn-beijing"])
    ec = cards.entry_card()
    assert ec["schema"] == "2.0"
    form = ec["body"]["elements"][1]
    names = {e.get("name") for e in form["elements"]}
    assert {"region", "source", "dest"} <= names
    assert _find_action(ec) == "submit_cpfs_dataflow"


def test_entry_card_region_fallback_input(monkeypatch):
    from core.cpfs_dataflow import discovery
    monkeypatch.setattr(discovery, "regions", lambda *a, **k: [])   # 无发现→地区改输入框
    ec = cards.entry_card()
    form = ec["body"]["elements"][1]
    region_elem = next(e for e in form["elements"] if e.get("name") == "region")
    assert region_elem["tag"] == "input"


def test_confirm_result_cards():
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/o/", "oss://bk/e/", fs_id="bmcpfs-x", region="cn-hangzhou"))
    assert _find_action(cards.confirm_card(job)) == "confirm_cpfs_dataflow"
    job["stage"] = "DONE"
    assert cards.result_card(job)["header"]["template"] == "green"
    job["stage"] = "FAILED"
    assert cards.result_card(job)["header"]["template"] == "red"


def test_submit_handler_requires_fields():
    out = actions._h_submit_cpfs_dataflow({}, "ou_1", "chat", {"region": "cn-hangzhou"})
    assert out["toast"]["type"] == "error"


def test_submit_handler_accepts(monkeypatch):
    import core.dsw_scheduler as sch
    monkeypatch.setattr(sch, "_send_card", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sch, "_send_text", lambda *a, **k: None, raising=False)
    # 不真正建流：mock plan_from_addresses
    monkeypatch.setattr(orchestrator, "plan_from_addresses",
                        lambda region, s, d, **k: orchestrator.make_plan(
                            "sink", "/cwr/o/", "oss://bk/e/", fs_id="bmcpfs-x", region=region,
                            data_flow_id="df-new"))
    out = actions._h_submit_cpfs_dataflow(
        {}, "ou_1", "chat",
        {"region": "cn-hangzhou", "source": "/cpfs/cwr/o/", "dest": "oss://bk/e/"})
    assert out["toast"]["type"] == "success"


def test_confirm_handler_missing_job():
    out = actions._h_confirm_cpfs_dataflow({"job_id": "cpfs-none"}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "error"


def test_confirm_handler_launches(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/o2/", "oss://bk/e2/", fs_id="bmcpfs-y", region="cn-hangzhou"))
    out = actions._h_confirm_cpfs_dataflow({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "success"
