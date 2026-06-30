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


def test_entry_card_step1_with_regions(monkeypatch):
    from core.cpfs_dataflow import discovery
    monkeypatch.setattr(discovery, "regions", lambda *a, **k: ["cn-hangzhou", "cn-shanghai"])
    ec = cards.entry_card()
    form = ec["body"]["elements"][1]
    names = {e.get("name") for e in form["elements"]}
    assert "operation" in names and "region" in names    # 第1步：操作+地区
    assert "cpfs_path" not in names                       # 不再是扁平/手填


def test_wizard_step2_card_has_fs_and_bucket(monkeypatch):
    from core.cpfs_dataflow import discovery
    monkeypatch.setattr(discovery, "filesystems_in",
                        lambda r, **k: [{"fs_id": "bmcpfs-a", "edition": "computing"}])
    monkeypatch.setattr(discovery, "buckets_in", lambda r, **k: ["bk-1", "bk-2"])
    out = actions._h_cpfs_wizard({}, "ou_1", "chat", {"operation": "sink", "region": "cn-hangzhou"})
    card2 = out["card"]["data"]
    form = card2["body"]["elements"][1]
    names = {e.get("name") for e in form["elements"]}
    assert {"fs_id", "cpfs_dir", "oss_bucket", "oss_subdir"} <= names


def test_wizard_requires_region():
    out = actions._h_cpfs_wizard({}, "ou_1", "chat", {"operation": "sink"})
    assert out["toast"]["type"] == "error"


def test_resolve_wizard_locates_dataflow(monkeypatch):
    import core.dsw_scheduler as sch
    from core.cpfs_dataflow import engine_nas
    monkeypatch.setattr(sch, "_send_card", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sch, "_send_text", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(engine_nas, "resolve_dataflow",
                        lambda *a, **k: {"data_flow_id": "df-1", "oss_prefix": "wuji_il", "fs_path": "/cwr/"})
    out = actions._h_resolve_cpfs_wizard(
        {"operation": "sink", "region": "cn-hangzhou"}, "ou_1", "chat",
        {"fs_id": "bmcpfs-a", "cpfs_dir": "/cwr/third_party_data/raw/",
         "oss_bucket": "wuji-bucket-hangzhou", "oss_subdir": "label"})
    assert out["toast"]["type"] == "success"


def test_resolve_wizard_requires_fields():
    out = actions._h_resolve_cpfs_wizard(
        {"operation": "sink", "region": "cn-hangzhou"}, "ou_1", "chat", {"fs_id": "bmcpfs-a"})
    assert out["toast"]["type"] == "error"


def test_confirm_handler_launches(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "cpfs://bmcpfs-y/o2/", "oss://bk/e2/"))
    out = actions._h_confirm_cpfs_dataflow({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "success"
