"""CPFS 飞书接线：意图、卡片结构、地址→建流 handler。"""
from core.feishu_bot import messages, actions
from core.cpfs_dataflow import cards, orchestrator, engine_nas


def _find_action(obj):
    """兼容 schema2.0 behaviors.callback 与 btn() 的 value.action 两种结构。"""
    if isinstance(obj, dict):
        val = obj.get("value")
        if isinstance(val, dict) and val.get("action"):
            return val["action"]
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
    # 确认后原地换成带「查询进度」按钮的进度卡
    assert _find_action(out["card"]["data"]) == "query_cpfs_progress"


def test_query_progress_running(monkeypatch):
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/q/", "oss://bk/q/", fs_id="bmcpfs-z", region="cn-hangzhou"))
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)
    out = actions._h_query_cpfs_progress({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert out["card"]["data"]["header"]["template"] == "blue"     # 进度卡


def test_query_progress_done(monkeypatch):
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/q2/", "oss://bk/q2/", fs_id="bmcpfs-z", region="cn-hangzhou"))
    job["stage"] = orchestrator.STAGE_DONE
    orchestrator._save(job)
    out = actions._h_query_cpfs_progress({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert out["card"]["data"]["header"]["template"] == "green"    # 结果卡


def test_query_progress_missing():
    out = actions._h_query_cpfs_progress({"job_id": "cpfs-none"}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "error"


def test_progress_query_text_intent():
    assert messages._is_progress_query_text("查询进度")
    assert messages._is_progress_query_text("查询进度 cpfs-abc123ff")
    assert not messages._is_progress_query_text("帮我建个实例")


def test_progress_query_no_id_pops_input_card(monkeypatch):
    """不带 ID → 弹带输入框的查询卡（不再列出全部任务）。"""
    import json as _j
    from core.feishu_bot import messaging as mg
    sent = []
    monkeypatch.setattr(mg, "_feishu_reply_card", lambda mid, card: sent.append(card))
    monkeypatch.setattr(mg, "_feishu_reply", lambda mid, text: sent.append({"text": text}))
    messages._handle_progress_query("m1", "查询进度", "ou_me")
    assert sent
    dumped = _j.dumps(sent[0], ensure_ascii=False)
    # 是一张带 job_id 输入框 + query_progress_by_id 提交动作的表单卡
    assert "query_progress_by_id" in dumped and "job_id" in dumped


def test_query_progress_by_id_routes_to_cpfs(monkeypatch):
    """输入卡提交 cpfs- id → 推送该任务卡（不原地替换，避开 2.0↔1.0 schema 冲突）+ 回 toast。"""
    import json as _j
    from core.feishu_bot import actions
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/q/", "oss://bk/q/", fs_id="bmcpfs-x", region="cn-hangzhou"))
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)
    pushed = []
    monkeypatch.setattr("core.dsw_scheduler._send_card", lambda oid, cid, card: pushed.append(card))
    resp = actions._h_query_progress_by_id({}, "ou_me", "oc_x", {"job_id": job["job_id"]})
    # 推了一张带该任务的卡，且 toast 带阶段
    assert pushed and job["job_id"] in _j.dumps(pushed[0], ensure_ascii=False)
    assert "RUNNING" in _j.dumps(resp, ensure_ascii=False)


def test_query_progress_by_id_empty_is_noop(monkeypatch):
    """空 job_id（老式双投递）→ 静默 no-op，不误报。"""
    from core.feishu_bot import actions
    assert actions._h_query_progress_by_id({}, "ou_me", "oc_x", {}) == {}


def test_card_action_dedup_collapses_double_delivery():
    """飞书双投递（同一操作两条）→ 首条放行、后续判重，全局只处理一次。"""
    from core.feishu_bot import actions
    args = ("query_progress_by_id", "ou_me", "om_1", {"job_id": "cpfs-abc"}, {"action": "query_progress_by_id"})
    assert actions.card_action_is_duplicate(*args) is False   # 第一条：放行
    assert actions.card_action_is_duplicate(*args) is True    # 第二条（重复投递）：判重
    # 不同表单（另一个任务）不误伤
    other = ("query_progress_by_id", "ou_me", "om_2", {"job_id": "cpfs-xyz"}, {"action": "query_progress_by_id"})
    assert actions.card_action_is_duplicate(*other) is False


def test_progress_query_by_cpfs_id(monkeypatch):
    from core.feishu_bot import messaging as mg
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/p/", "oss://bk/p/", fs_id="bmcpfs-x", region="cn-hangzhou"))
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)
    cap = []
    monkeypatch.setattr(mg, "_feishu_reply", lambda mid, text: cap.append(text))
    messages._handle_progress_query("m1", f"查询进度 {job['job_id']}")
    assert cap and job["job_id"] in cap[0] and "RUNNING" in cap[0]
