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


def test_mfu_report_intent_keywords():
    assert messages._is_mfu_report_intent("算力日报")
    assert messages._is_mfu_report_intent("看看算力效率")
    assert messages._is_mfu_report_intent("MFU")
    assert messages._is_mfu_report_intent("集群算力")
    # 普通 GPU 申请不误命中（只含"算力"不含日报/效率等词）
    assert not messages._is_mfu_report_intent("申请一张 A100 算力")


def test_entry_card_is_cloud_picker():
    """入口卡=云平台选择（两个按钮 阿里/火山）。"""
    from core.dataflow_cards import entry_card
    ec = entry_card()
    assert ec["schema"] == "2.0"
    assert {"pick_cloud_aliyun", "pick_cloud_volcano"} <= _all_actions(ec)


def test_region_card_buttons():
    """选地区卡：每个地区一个按钮，带 pick_region_aliyun。"""
    from core.dataflow_cards import region_card
    rc = region_card("aliyun", ["cn-hangzhou", "cn-beijing"])
    assert "pick_region_aliyun" in _all_actions(rc)
    labels = [e["text"]["content"] for e in rc["body"]["elements"] if e.get("tag") == "button"]
    assert "cn-hangzhou" in labels and "cn-beijing" in labels


def test_form_card_fs_dropdown_and_addresses():
    """向导表单卡：fs 下拉 + 源/目的地址；无操作/桶选择器；提交 submit_cpfs_dataflow。"""
    from core.dataflow_cards import form_card
    fs = [{"value": "cpfs-a@cn-hangzhou", "text": "cpfs-a（cn-hangzhou）"}]
    ec = form_card("aliyun", "cn-hangzhou", fs)
    form = ec["body"]["elements"][1]["elements"]
    names = {e.get("name") for e in form}
    assert {"fs", "source", "dest", "same_name"} <= names
    assert "operation" not in names and "bucket" not in names   # 无操作/桶选择器
    fs_el = next(e for e in form if e.get("name") == "fs")
    assert fs_el["tag"] == "select_static"
    assert _find_action(ec) == "submit_cpfs_dataflow"


def test_form_card_empty_fs_fallback_to_input():
    """发现为空 → fs 回退成文本输入框。"""
    from core.dataflow_cards import form_card
    ec = form_card("aliyun", "", [])
    form = ec["body"]["elements"][1]["elements"]
    fs_el = next(e for e in form if e.get("name") == "fs")
    assert fs_el["tag"] == "input"


def _all_actions(obj):
    acts = set()
    def _w(o):
        if isinstance(o, dict):
            v = o.get("value")
            if isinstance(v, dict) and v.get("action"):
                acts.add(v["action"])
            for x in o.values():
                _w(x)
        elif isinstance(o, list):
            for x in o:
                _w(x)
    _w(obj)
    return acts


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
    """#46-A2：确认卡默认 reply_v2=True → 原地换成 schema 2.0 进度卡（同家族避 200830）。

    改前断言的 query_cpfs_progress 按钮已随 progress_card_v2 移除（无按钮，终态由后台线程推结果卡）。
    """
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/o2/", "oss://bk/e2/", fs_id="bmcpfs-y", region="cn-hangzhou"))
    out = actions._h_confirm_cpfs_dataflow({"job_id": job["job_id"]}, "ou_1", "chat", {})
    assert out["toast"]["type"] == "success"
    # 确认后原地换成 2.0 纯展示进度卡：schema 2.0、无按钮/表单（不能再连点）
    data = out["card"]["data"]
    assert data["schema"] == "2.0"
    assert _find_action(data) is None
    assert "button" not in str(data) and "form" not in str(data)


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


class _SyncThread:
    """同步跑后台线程体，便于断言 _bg 推送结果。"""
    def __init__(self, target=None, daemon=None, **k): self._t = target
    def start(self): self._t()


def test_query_progress_by_id_routes_to_cpfs(monkeypatch):
    """输入卡提交 cpfs- id → sync 秒回“正在查询”toast，后台线程推该任务卡（不原地替换，避开 2.0↔1.0）。

    HIGH#3 后：refresh（poll 云端）挪进后台线程，sync 只回 toast。用同步线程桩断言后台推送。
    """
    import json as _j
    from core.feishu_bot import actions
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    job = orchestrator.create_job_record(
        orchestrator.make_plan("sink", "/cwr/q/", "oss://bk/q/", fs_id="bmcpfs-x", region="cn-hangzhou"))
    job["stage"] = orchestrator.STAGE_RUNNING
    orchestrator._save(job)
    pushed = []
    monkeypatch.setattr("core.dsw_scheduler._send_card", lambda oid, cid, card: pushed.append(card))
    resp = actions._h_query_progress_by_id({}, "ou_me", "oc_x", {"job_id": job["job_id"]})
    # sync 只回 toast（秒返、不阻塞 3s），后台线程推该任务卡
    assert resp["toast"]["type"] == "success" and "正在查询" in resp["toast"]["content"]
    assert "card" not in resp                                     # sync 不原地替换
    assert pushed and job["job_id"] in _j.dumps(pushed[0], ensure_ascii=False)


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
