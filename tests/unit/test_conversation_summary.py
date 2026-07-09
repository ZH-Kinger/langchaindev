"""#43 滚动摘要记忆（`core.agent` + `core.feishu_bot.messages`）——二轮，跟随 dev 据 auditor 3 Med 的改动。

dev 给对话 agent 的「逐字窗口 + 滚动摘要」两级记忆，二轮改后：
  - **MED-1 批量+异步**：`_save_turn` rpush 后，当 `llen >= FOLD_TRIGGER(=MAX_HISTORY+10=30)`
    才折叠一次（约每 5 轮 1 次），同步 `ltrim` 到最近 `MAX_HISTORY(20)` 条，压缩交
    `threading.Thread(daemon=True)` 后台跑（不挡回复）。
  - **MED-2 短超时**：`get_cloud_llm` 新增可选 `timeout`（传则一并 `max_retries=0`）；
    `_compress_summary` 用 `get_cloud_llm(temperature=0, timeout=SUMMARY_TIMEOUT=20)`。
  - **MED-3 摘要不再作第二条 SystemMessage**：`_load_history` 只返原文；新增
    `_load_summary(sid)->str`；由 `messages._process_message` 把摘要作前缀拼进 `agent_input`。
  - `_clear_history`：一并删 history 与 summary 两个 key。

压缩用 **get_cloud_llm / GLM**（非 get_edge_llm，edge 已废弃）。
全部用 conftest 的 fakeredis（autouse）跑；`get_cloud_llm` 用桩替换，`threading.Thread`
用假线程替换（避免真起线程 flaky，并能同步/不同步地控制压缩何时跑）。绝不真调 LLM。
"""
import json
import types

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import core.agent as agent

SID = "ou_summary_test"


def _hkey():
    return agent._session_key(SID)


def _skey():
    return agent._summary_key(SID)


# ── LLM 桩：记录被压缩进去的 prompt + 工厂调用 kwargs，返回带 .content 的假响应 ─────
class _Resp:
    def __init__(self, content):
        self.content = content


class _StubLLM:
    def __init__(self, ret):
        self.ret = ret
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return _Resp(self.ret)


def _patch_llm(monkeypatch, ret="[STUB摘要]压缩后的一段话", raises=False):
    """替换 core.agent.get_cloud_llm。返回 (stub, calls)；
    calls['n'] 计工厂调用次数，calls['kwargs'] 记录每次调用的 kwargs（验 timeout/temperature）。"""
    stub = _StubLLM(ret)
    calls = {"n": 0, "kwargs": []}

    def factory(*a, **k):
        calls["n"] += 1
        calls["kwargs"].append(k)
        if raises:
            raise RuntimeError("llm boom")
        return stub

    monkeypatch.setattr(agent, "get_cloud_llm", factory)
    return stub, calls


def _patch_thread(monkeypatch, run_sync=True):
    """把 core.agent.threading 换成假命名空间，Thread 记录调度参数。
    run_sync=True → start() 时同步执行 target（便于确定性验证压缩落库）；
    run_sync=False → start() 只记录不执行（模拟"仅调度、压缩没跑"，验证 ltrim 已同步完成）。
    返回 records 列表，每项含 target/args/daemon/started。"""
    records = []

    class _FakeThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            self.started = False
            records.append(self)

        def start(self):
            self.started = True
            if run_sync and self.target is not None:
                self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(agent, "threading", types.SimpleNamespace(Thread=_FakeThread))
    return records


def _seed_raw(fake, n):
    """直接往 history list 塞 n 条原文消息，content 形如 OLD0/OLD1…，便于验 off-by-one。"""
    key = _hkey()
    for i in range(n):
        role = "human" if i % 2 == 0 else "ai"
        fake.rpush(key, json.dumps({"role": role, "content": f"OLD{i}"}))


# ── 1. 不足 FOLD_TRIGGER：零压缩、无摘要 key、不调度线程、load 只返回原文 ──────────
def test_no_overflow_no_compression(fake_redis, monkeypatch):
    stub, calls = _patch_llm(monkeypatch)
    threads = _patch_thread(monkeypatch)

    for i in range(5):   # 5 轮 = 10 条 < FOLD_TRIGGER(30)
        agent._save_turn(SID, f"q{i}", f"a{i}")

    assert calls["n"] == 0                       # get_cloud_llm 零调用
    assert stub.prompts == []
    assert threads == []                         # 未调度任何后台压缩线程
    assert fake_redis.exists(_skey()) == 0       # 无摘要 key
    assert fake_redis.llen(_hkey()) == 10


def test_no_overflow_load_history_plain(fake_redis, monkeypatch):
    _patch_llm(monkeypatch)
    _patch_thread(monkeypatch)
    for i in range(5):
        agent._save_turn(SID, f"q{i}", f"a{i}")

    hist = agent._load_history(SID)
    assert len(hist) == 10
    assert not any(isinstance(m, SystemMessage) for m in hist)   # 从不注入 SystemMessage
    assert isinstance(hist[0], HumanMessage) and hist[0].content == "q0"
    assert isinstance(hist[1], AIMessage) and hist[1].content == "a0"


# ── 2. MED-3 _load_history 从不注入摘要：即便有摘要 key 也只返回原文 ─────────────
def test_load_history_never_injects_summary(fake_redis, monkeypatch):
    _patch_llm(monkeypatch)
    fake_redis.set(_skey(), "早前聊过：用户是王梓涵，在做迁移")   # 有摘要
    _seed_raw(fake_redis, 4)   # OLD0..OLD3

    hist = agent._load_history(SID)
    # 全是原文，无 SystemMessage、无摘要文本混入
    assert len(hist) == 4
    assert not any(isinstance(m, SystemMessage) for m in hist)
    assert all("王梓涵" not in m.content for m in hist)
    assert isinstance(hist[0], HumanMessage) and hist[0].content == "OLD0"
    assert isinstance(hist[1], AIMessage) and hist[1].content == "OLD1"


# ── 3. MED-3 _load_summary：取出摘要文本；无 key 返回空串 ────────────────────────
def test_load_summary_returns_text(fake_redis, monkeypatch):
    assert agent._load_summary(SID) == ""            # 无 key → ""
    fake_redis.set(_skey(), "摘要正文：用户在做跨云迁移")
    assert agent._load_summary(SID) == "摘要正文：用户在做跨云迁移"


def test_load_summary_redis_down_returns_empty(fake_redis, monkeypatch):
    fake_redis.set(_skey(), "有内容但读不到")
    monkeypatch.setattr(agent, "is_redis_available", lambda: False)
    assert agent._load_summary(SID) == ""


# ── 4. _fold_into_summary 直接测（同步）：压缩成功写摘要、prompt 含被移出对话 ──────
def test_fold_into_summary_success(fake_redis, monkeypatch):
    stub, calls = _patch_llm(monkeypatch, ret="ROLLING_SUMMARY_X")
    evicted = [
        json.dumps({"role": "human", "content": "旧问题Q0"}),
        json.dumps({"role": "ai", "content": "旧答复A0"}),
    ]

    agent._fold_into_summary(SID, evicted)

    assert calls["n"] == 1
    prompt = stub.prompts[0]
    assert "旧问题Q0" in prompt and "旧答复A0" in prompt
    assert "用户：旧问题Q0" in prompt and "助手：旧答复A0" in prompt
    assert fake_redis.get(_skey()) == "ROLLING_SUMMARY_X"
    # 摘要 key 落库后应有 ≈7 天 TTL
    ttl = fake_redis.ttl(_skey())
    assert agent.HISTORY_TTL_SECONDS - 10 <= ttl <= agent.HISTORY_TTL_SECONDS


def test_fold_into_summary_merges_old_summary(fake_redis, monkeypatch):
    """已有摘要应作为 prompt 的一部分参与合并（滚动，而非丢弃）。"""
    stub, calls = _patch_llm(monkeypatch, ret="NEW_MERGED")
    fake_redis.set(_skey(), "OLD_SUMMARY_TEXT")
    evicted = [json.dumps({"role": "human", "content": "增量对话X"})]

    agent._fold_into_summary(SID, evicted)

    assert "OLD_SUMMARY_TEXT" in stub.prompts[0]   # 旧摘要参与合并
    assert "增量对话X" in stub.prompts[0]
    assert fake_redis.get(_skey()) == "NEW_MERGED"


def test_fold_into_summary_failure_keeps_old(fake_redis, monkeypatch):
    """压缩抛异常 → 不覆盖旧摘要、不抛。"""
    stub, calls = _patch_llm(monkeypatch, raises=True)
    fake_redis.set(_skey(), "OLD_SUMMARY_KEEP")
    evicted = [json.dumps({"role": "human", "content": "x"})]

    agent._fold_into_summary(SID, evicted)   # 不应抛

    assert calls["n"] == 1
    assert fake_redis.get(_skey()) == "OLD_SUMMARY_KEEP"


def test_fold_into_summary_empty_returns_ai_yields_no_write(fake_redis, monkeypatch):
    """LLM 返回空串（视为失败）→ 不写摘要 key。"""
    stub, calls = _patch_llm(monkeypatch, ret="   ")   # strip 后为空
    evicted = [json.dumps({"role": "human", "content": "x"})]

    agent._fold_into_summary(SID, evicted)

    assert calls["n"] == 1
    assert fake_redis.exists(_skey()) == 0     # 空摘要不写


def test_fold_into_summary_no_valid_evicted_is_noop(fake_redis, monkeypatch):
    """被移出内容全是坏 JSON / 空 → 不调 LLM、不写摘要。"""
    stub, calls = _patch_llm(monkeypatch)

    agent._fold_into_summary(SID, ["not-json", "{bad"])
    assert calls["n"] == 0
    assert fake_redis.exists(_skey()) == 0

    agent._fold_into_summary(SID, [])
    assert calls["n"] == 0


def test_summary_truncated_to_max_chars(fake_redis, monkeypatch):
    """压缩返回超长 → 写入前截到 SUMMARY_MAX_CHARS。"""
    long = "字" * (agent.SUMMARY_MAX_CHARS + 500)
    _patch_llm(monkeypatch, ret=long)
    agent._fold_into_summary(SID, [json.dumps({"role": "human", "content": "x"})])
    assert len(fake_redis.get(_skey())) == agent.SUMMARY_MAX_CHARS


# ── 5. MED-2 _compress_summary 走带 timeout 的 get_cloud_llm ─────────────────────
def test_compress_summary_uses_timeout(fake_redis, monkeypatch):
    stub, calls = _patch_llm(monkeypatch, ret="OUT")
    out = agent._compress_summary("旧摘要", "新对话")

    assert out == "OUT"
    assert calls["n"] == 1
    kw = calls["kwargs"][0]
    assert kw.get("timeout") == agent.SUMMARY_TIMEOUT   # 短超时透传
    assert kw.get("temperature") == 0                   # 温度 0


def test_compress_summary_failure_returns_empty(fake_redis, monkeypatch):
    _patch_llm(monkeypatch, raises=True)
    assert agent._compress_summary("旧", "新") == ""     # 失败 → 空串（调用方保留旧摘要）


# ── 6. MED-1 _save_turn 达到 FOLD_TRIGGER：同步 ltrim + 后台线程调度 ─────────────
def test_save_turn_schedules_thread_and_ltrims(fake_redis, monkeypatch):
    """run_sync=False：验证「仅调度线程 + 同步 ltrim 已完成」，压缩此刻还没跑。"""
    stub, calls = _patch_llm(monkeypatch)
    threads = _patch_thread(monkeypatch, run_sync=False)
    _seed_raw(fake_redis, 28)                 # OLD0..OLD27
    agent._save_turn(SID, "newq", "newa")     # +2 = 30 = FOLD_TRIGGER → 触发

    # 后台压缩线程被调度且是 daemon
    assert len(threads) == 1
    t = threads[0]
    assert t.started is True and t.daemon is True
    assert t.target is agent._fold_into_summary
    # 被移出 = total - MAX_HISTORY = 30 - 20 = 10 条，且正是最旧 10 条 OLD0..OLD9
    ev_sid, ev_items = t.args
    assert ev_sid == SID
    assert len(ev_items) == 10
    contents = [json.loads(x)["content"] for x in ev_items]
    assert contents == [f"OLD{i}" for i in range(10)]
    # ltrim 同步完成：窗口有界 20，首条是 OLD10
    assert fake_redis.llen(_hkey()) == agent.MAX_HISTORY
    first = json.loads(fake_redis.lrange(_hkey(), 0, 0)[0])
    assert first == {"role": "human", "content": "OLD10"}
    # 线程没跑 → 压缩未发生、摘要未写
    assert calls["n"] == 0
    assert fake_redis.exists(_skey()) == 0


def test_save_turn_trigger_run_sync_writes_summary(fake_redis, monkeypatch):
    """run_sync=True：后台线程同步跑完 → 摘要落库、get_cloud_llm 恰 1 次、两 key 有 TTL。"""
    stub, calls = _patch_llm(monkeypatch, ret="ROLLING_SUMMARY_X")
    _patch_thread(monkeypatch, run_sync=True)
    _seed_raw(fake_redis, 28)
    agent._save_turn(SID, "q0", "a0")         # 30 → 折叠

    assert calls["n"] == 1
    prompt = stub.prompts[0]
    assert "OLD0" in prompt and "OLD9" in prompt   # 最旧 10 条进了压缩
    assert "OLD10" not in prompt                   # 第 11 旧仍在窗口
    assert fake_redis.get(_skey()) == "ROLLING_SUMMARY_X"
    assert fake_redis.llen(_hkey()) == agent.MAX_HISTORY
    for k in (_hkey(), _skey()):
        ttl = fake_redis.ttl(k)
        assert agent.HISTORY_TTL_SECONDS - 10 <= ttl <= agent.HISTORY_TTL_SECONDS


def test_save_turn_compression_failure_keeps_old_and_ltrims(fake_redis, monkeypatch):
    """折叠中压缩失败 → 旧摘要保留、_save_turn 不抛、list 仍 ltrim 到 20。"""
    stub, calls = _patch_llm(monkeypatch, raises=True)
    _patch_thread(monkeypatch, run_sync=True)
    fake_redis.set(_skey(), "OLD_SUMMARY_KEEP")
    _seed_raw(fake_redis, 28)

    agent._save_turn(SID, "new_q", "new_a")   # 30 → 折叠、压缩抛异常，不应传导

    assert calls["n"] == 1
    assert fake_redis.get(_skey()) == "OLD_SUMMARY_KEEP"   # 旧摘要原样保留
    assert fake_redis.llen(_hkey()) == agent.MAX_HISTORY
    last = [json.loads(x) for x in fake_redis.lrange(_hkey(), -2, -1)]
    assert last == [{"role": "human", "content": "new_q"},
                    {"role": "ai", "content": "new_a"}]


# ── 7. off-by-one：FOLD_TRIGGER 边界（29 不折叠 / 30 折叠移出 10 / 32 移出 12）─────
def test_below_trigger_29_no_fold(fake_redis, monkeypatch):
    stub, calls = _patch_llm(monkeypatch)
    threads = _patch_thread(monkeypatch)
    _seed_raw(fake_redis, 27)                 # OLD0..OLD26
    agent._save_turn(SID, "q", "a")           # +2 = 29 < FOLD_TRIGGER(30)

    assert calls["n"] == 0
    assert threads == []                      # 不折叠 → 不调度线程
    assert fake_redis.exists(_skey()) == 0
    assert fake_redis.llen(_hkey()) == 29     # 不 ltrim，逐字窗口暂超 20 直到下次折叠


def test_at_trigger_30_evicts_exactly_ten(fake_redis, monkeypatch):
    stub, calls = _patch_llm(monkeypatch)
    threads = _patch_thread(monkeypatch, run_sync=False)
    _seed_raw(fake_redis, 28)                 # OLD0..OLD27
    agent._save_turn(SID, "q", "a")           # 30 → 移出 total-MAX_HISTORY = 10 条

    ev_items = threads[0].args[1]
    contents = [json.loads(x)["content"] for x in ev_items]
    assert contents == [f"OLD{i}" for i in range(10)]   # 恰 OLD0..OLD9，无多移/少移
    assert fake_redis.llen(_hkey()) == agent.MAX_HISTORY
    first = json.loads(fake_redis.lrange(_hkey(), 0, 0)[0])
    assert first == {"role": "human", "content": "OLD10"}


def test_over_trigger_32_evicts_twelve(fake_redis, monkeypatch):
    """再多 2 条 → 移出 total-MAX_HISTORY = 12 条，验 total-MAX_HISTORY-1 无 off-by-one。"""
    stub, calls = _patch_llm(monkeypatch)
    threads = _patch_thread(monkeypatch, run_sync=False)
    _seed_raw(fake_redis, 30)                 # OLD0..OLD29
    agent._save_turn(SID, "q", "a")           # 32 → 移出 12 条

    ev_items = threads[0].args[1]
    contents = [json.loads(x)["content"] for x in ev_items]
    assert contents == [f"OLD{i}" for i in range(12)]   # OLD0..OLD11
    assert fake_redis.llen(_hkey()) == agent.MAX_HISTORY
    first = json.loads(fake_redis.lrange(_hkey(), 0, 0)[0])
    assert first == {"role": "human", "content": "OLD12"}


# ── 8. _clear_history：删 history + summary 两个 key ─────────────────────────────
def test_clear_history_deletes_both_keys(fake_redis, monkeypatch):
    fake_redis.rpush(_hkey(), json.dumps({"role": "human", "content": "x"}))
    fake_redis.set(_skey(), "some summary")
    assert fake_redis.exists(_hkey()) == 1
    assert fake_redis.exists(_skey()) == 1

    agent._clear_history(SID)

    assert fake_redis.exists(_hkey()) == 0
    assert fake_redis.exists(_skey()) == 0


# ── 9. Redis 不可用：load 返回 []、save/clear 静默不抛 ───────────────────────────
def test_redis_down_degrades_gracefully(fake_redis, monkeypatch):
    _patch_llm(monkeypatch)
    _patch_thread(monkeypatch)
    fake_redis.set(_skey(), "s")
    _seed_raw(fake_redis, 4)

    monkeypatch.setattr(agent, "is_redis_available", lambda: False)

    assert agent._load_history(SID) == []      # 不读 Redis，直接空
    assert agent._load_summary(SID) == ""      # 摘要同样降级
    agent._save_turn(SID, "q", "a")            # 静默 no-op，不抛
    agent._clear_history(SID)                  # 静默 no-op，不抛


def test_redis_down_save_writes_nothing(monkeypatch, fake_redis):
    _patch_llm(monkeypatch)
    _patch_thread(monkeypatch)
    monkeypatch.setattr(agent, "is_redis_available", lambda: False)
    agent._save_turn(SID, "q", "a")
    assert fake_redis.exists(_hkey()) == 0
    assert fake_redis.exists(_skey()) == 0


# ── 10. MED-3 messages._process_message：摘要作前缀拼进 agent_input ──────────────
def _stub_message_pipeline(monkeypatch):
    """关掉 _process_message 里 Agent 之前的意图分支、打桩 Agent，捕获传给 executor 的 input。
    返回 records：records['input'] = 最终 agent_input。"""
    from core.feishu_bot import messages, messaging
    import core.agent as ag

    records = {"input": None, "replies": []}

    monkeypatch.setattr(messages, "_auto_map_user", lambda *a, **k: None)
    for name in [
        "_is_progress_query_text", "_is_mfu_report_intent", "_is_gpu_dist_intent",
        "_is_bucket_transfer_entry_intent", "_is_sink_preheat_entry_intent",
        "_is_volcano_account_query_entry_intent", "_is_ram_query_entry_intent",
        "_is_transfer_entry_intent", "_is_transfer_intent",
        "_is_transfer_confirm_text", "_is_gpu_intent", "_is_help_intent",
    ]:
        monkeypatch.setattr(messages, name, lambda *a, **k: False)

    monkeypatch.setattr(messaging, "_feishu_reply",
                        lambda mid, text: records["replies"].append(text))

    monkeypatch.setattr(ag, "_load_history", lambda sid: [])
    monkeypatch.setattr(ag, "_save_turn", lambda *a, **k: None)
    monkeypatch.setattr(ag, "_names_to_tools", lambda names: [])
    # 有工具（非 None）→ agent_input = base_input，不套未知意图外壳；knowledge 意图不触发指标图
    monkeypatch.setattr(ag, "select_tools_scoped", lambda text: (["t"], {"knowledge"}))

    class _FakeExec:
        def invoke(self, payload):
            records["input"] = payload["input"]
            return {"output": "答复"}
    monkeypatch.setattr(ag, "_build_executor", lambda tools=None: _FakeExec())

    return records, messages, ag


def test_process_message_prepends_summary(monkeypatch, fake_redis):
    records, messages, ag = _stub_message_pipeline(monkeypatch)
    monkeypatch.setattr(ag, "_load_summary", lambda sid: "早前：用户在做 TOS→OSS 迁移")

    messages._process_message("m1", "chat1", "现在进度如何", open_id="ou_x")

    agent_input = records["input"]
    assert agent_input is not None
    assert "早前：用户在做 TOS→OSS 迁移" in agent_input          # 摘要正文
    assert "以下为你与该用户更早对话的摘要" in agent_input        # 前缀说明
    assert "现在进度如何" in agent_input                          # 原始用户消息仍在
    assert "[feishu_open_id=ou_x]" in agent_input                # base_input 元数据保留
    # 摘要在前、真正输入在后（前缀拼接顺序）
    assert agent_input.index("早前：用户在做") < agent_input.index("现在进度如何")


def test_process_message_no_summary_plain_input(monkeypatch, fake_redis):
    records, messages, ag = _stub_message_pipeline(monkeypatch)
    monkeypatch.setattr(ag, "_load_summary", lambda sid: "")   # 无摘要

    messages._process_message("m2", "chat1", "现在进度如何", open_id="ou_x")

    agent_input = records["input"]
    assert agent_input == "现在进度如何\n[feishu_open_id=ou_x]"   # 纯 base_input，无摘要前缀
    assert "摘要" not in agent_input
