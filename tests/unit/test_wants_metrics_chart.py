"""#38-1 指标图按需：`core.feishu_bot.messages._wants_metrics_chart` 真值表
+ `_process_message` 在不命中时走纯文本、不触碰 Prometheus 云查询。

背景：原本每条 Agent 回复都无脑附一张实时趋势图（云查询 + matplotlib 渲染 +
图片上传 + 取 token），给「知识库/工单/迁移」等无关回复也塞 CPU/内存曲线。
优化后仅当问题确与监控/指标相关（intents 含 monitor/cluster，或文本命中指标词）
才附图，否则纯文本回复。

这些用例锁死门控行为——回归时若有人误删 gate 或改坏正则会立刻炸。
"""
import pytest

from core.feishu_bot import messages


# ── 真值表：命中（True）─────────────────────────────────────────────────────

@pytest.mark.parametrize("text, intents", [
    # intents 含 monitor / cluster → 直接放行（与文本无关）
    ("随便问个什么",            {"monitor"}),
    ("随便问个什么",            {"cluster"}),
    ("随便问个什么",            {"knowledge", "monitor"}),   # 多意图里有一个命中即可
    ("随便问个什么",            {"cluster", "jira"}),
    # 文本命中指标词（intents 为空/None 也应 True）
    ("现在 cpu 利用率多少",      None),
    ("看下内存占用",            set()),
    ("显存快满了吗",            None),
    ("gpu 使用率怎么样",         None),   # 命中「使用率」而非裸 gpu
    ("给我看看趋势",            None),
    ("集群监控情况",            None),
    ("当前负载高不高",          None),
    ("磁盘水位到多少了",         None),
    ("mfu 算力效率",            None),
    ("查一下 prometheus",       None),
    ("grafana 打开看看",        None),
    ("集群状态如何",            None),
    ("show me the metrics",     None),
    ("这些指标正常吗",          None),
    # 大小写不敏感
    ("CPU 高吗",               None),
    ("MFU report",             None),
])
def test_wants_chart_true(text, intents):
    assert messages._wants_metrics_chart(text, intents) is True


# ── 真值表：不命中（False）──────────────────────────────────────────────────

@pytest.mark.parametrize("text, intents", [
    # 无关问题 + intents 不含 monitor/cluster
    ("知识库里 K8s 怎么部署",    {"knowledge"}),
    ("查一下工单",              {"jira"}),
    ("帮我发起一个迁移任务",      {"transfer"}),
    ("介绍一下你自己",          None),
    ("你好",                   set()),
    ("帮我看下工单状态",         None),
    # 刻意不放裸 gpu：GPU 工单/裸词问句不该被塞监控图（源码注释明确此设计）
    ("查我的 gpu 工单",          {"jira"}),
    ("GPU 呢",                 None),
    # intents 与门控集不相交
    ("随便说点什么",            {"knowledge", "jira", "transfer"}),
    # 空/None 文本
    ("",                       None),
    (None,                     None),
])
def test_wants_chart_false(text, intents):
    assert messages._wants_metrics_chart(text, intents) is False


# ── intents 参数可选（默认 None）────────────────────────────────────────────

def test_intents_arg_optional():
    assert messages._wants_metrics_chart("cpu 利用率") is True
    assert messages._wants_metrics_chart("查工单") is False


# ── _process_message 集成：不命中 → 纯文本、绝不触碰云查询 ────────────────────

@pytest.fixture
def stub_message_pipeline(monkeypatch):
    """把 _process_message 里 Agent 之前的所有意图分支关掉、Agent 调用打桩，
    只留 `_wants_metrics_chart` 门控这一段被真实执行。返回一个 records 记录动作。"""
    records = {"replies": [], "charts": [], "fetch_called": 0}

    # 关闭首个后台线程（飞书↔RAM 自动映射，会打网络）
    monkeypatch.setattr(messages, "_auto_map_user", lambda *a, **k: None)

    # 关掉 Agent 之前的全部意图短路分支，确保一定走到 Agent + 门控段
    for name in [
        "_is_progress_query_text", "_is_mfu_report_intent", "_is_gpu_dist_intent",
        "_is_bucket_transfer_entry_intent", "_is_sink_preheat_entry_intent",
        "_is_volcano_account_query_entry_intent", "_is_ram_query_entry_intent",
        "_is_transfer_entry_intent", "_is_transfer_intent",
        "_is_transfer_confirm_text", "_is_gpu_intent", "_is_help_intent",
    ]:
        monkeypatch.setattr(messages, name, lambda *a, **k: False)

    # messaging 回复打桩
    from core.feishu_bot import messaging
    monkeypatch.setattr(messaging, "_feishu_reply",
                        lambda mid, text: records["replies"].append(text))
    monkeypatch.setattr(messaging, "_feishu_reply_with_chart",
                        lambda mid, text, key: records["charts"].append((text, key)))

    # core.agent：Agent 执行 + 历史打桩（函数内 from core.agent import ...）
    import core.agent as agent
    monkeypatch.setattr(agent, "_load_history", lambda sid: [])
    monkeypatch.setattr(agent, "_save_turn", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_names_to_tools", lambda names: [])

    class _FakeExec:
        def invoke(self, payload):
            return {"output": "这是答复"}
    monkeypatch.setattr(agent, "_build_executor", lambda tools=None: _FakeExec())

    # 云查询：一旦被调用就计数（不命中分支时应保持 0）
    from tools.aliyun import prometheus as prom

    def _boom():
        records["fetch_called"] += 1
        raise AssertionError("fetch_raw_series 不应在非指标问题时被调用")
    monkeypatch.setattr(prom, "fetch_raw_series", _boom)

    return records


def test_process_message_no_chart_for_unrelated(monkeypatch, stub_message_pipeline):
    """不含指标词 + intents 不含 monitor/cluster → 纯文本回复，不调 fetch_raw_series。"""
    rec = stub_message_pipeline
    import core.agent as agent
    monkeypatch.setattr(agent, "select_tools_scoped",
                        lambda text: ([], {"knowledge"}))

    messages._process_message("m1", "chat1", "介绍一下你自己", open_id="ou_x")

    assert rec["fetch_called"] == 0
    assert rec["charts"] == []
    # "思考中" 提示 + 最终纯文本答复都走 _feishu_reply
    assert "这是答复" in rec["replies"]


def test_process_message_attaches_chart_for_metrics(monkeypatch, stub_message_pipeline):
    """命中指标词 → 进入附图分支，调用 fetch_raw_series 并发带图卡片。"""
    rec = stub_message_pipeline
    import core.agent as agent
    monkeypatch.setattr(agent, "select_tools_scoped",
                        lambda text: ([], {"monitor"}))

    # 让附图链路走通（覆盖 fetch_raw_series 的抛错桩）
    from tools.aliyun import prometheus as prom
    monkeypatch.setattr(prom, "fetch_raw_series", lambda: {"cpu": [1, 2, 3]})
    import utils.chart_builder as cb
    monkeypatch.setattr(cb, "build_metrics_chart", lambda series: b"PNG")
    # _get_access_token 在 messages 模块顶部就地 import，须打在 messages 上
    monkeypatch.setattr(messages, "_get_access_token", lambda: "tok")
    from tools.feishu import notify
    monkeypatch.setattr(notify, "_upload_image", lambda tok, png: "img_key")

    messages._process_message("m2", "chat1", "现在 cpu 利用率多少", open_id="ou_x")

    assert rec["charts"], "命中指标问题应发带图卡片"
    assert rec["charts"][0] == ("这是答复", "img_key")
