"""JIRA_ENABLED / SCHEDULER_ENABLED 两开关（dev 新增，默认 Jira 停用 / 调度器启用）。

覆盖 4 个改动点：
  1. config.settings —— JIRA_ENABLED 默认 false、SCHEDULER_ENABLED 默认 true；解析 true/false；
     validate() 在 Jira 停用时**不**为 JIRA_URL/JIRA_PAT 缺失告警，启用时**照常**告警，其余必填项行为不变。
  2. dsw_scheduler.DSWScheduler.start() —— Jira 停用不创建/启动 jira-poll 线程（_ticket_thread 保持 None），
     dsw-check + morning-report 照常；启用则三个都起；关时 stop() 不 NPE。日志段 Jira 轮询 / (Jira 停用)。
  3. routes.health() —— Jira 停用 → jira=="disabled" 且**不**因 jira 置 degraded；启用+缺配置 → not_configured。
  4. actions._h_submit_gpu_request() —— Jira 停用 → 返回 info 停用 toast、不建工单、不起线程；启用走原逻辑。
"""
import importlib
import os
from unittest import mock

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# 1. config.settings —— 默认值 + 解析 + validate()
# ══════════════════════════════════════════════════════════════════════════════

def test_switch_defaults_when_env_unset():
    """env 未设时：JIRA_ENABLED 默认 False、SCHEDULER_ENABLED 默认 True。

    清空这两个环境变量并禁掉 .env 回填后 reload 模块，读类属性验证解析默认。
    """
    import config.settings as s
    orig_settings, orig_cls = s.settings, s.Config
    base = {k: v for k, v in os.environ.items()
            if k not in ("JIRA_ENABLED", "SCHEDULER_ENABLED")}
    try:
        with mock.patch("dotenv.load_dotenv", lambda *a, **k: False), \
             mock.patch.dict(os.environ, base, clear=True):
            importlib.reload(s)
            assert s.Config.JIRA_ENABLED is False
            assert s.Config.SCHEDULER_ENABLED is True
    finally:
        # reload 会把 config.settings.settings 换成新实例 → 已 import 该单例的其它模块
        # (routes/actions/dsw_scheduler...) 会与之错位。恢复原单例对象身份，避免全局污染。
        importlib.reload(s)
        s.settings, s.Config = orig_settings, orig_cls


def test_switches_parse_env_values():
    """JIRA_ENABLED=TRUE（大小写不敏感）→ True；SCHEDULER_ENABLED=false → False。"""
    import config.settings as s
    orig_settings, orig_cls = s.settings, s.Config
    try:
        with mock.patch("dotenv.load_dotenv", lambda *a, **k: False), \
             mock.patch.dict(os.environ,
                             {"JIRA_ENABLED": "TRUE", "SCHEDULER_ENABLED": "false"},
                             clear=False):
            importlib.reload(s)
            assert s.Config.JIRA_ENABLED is True
            assert s.Config.SCHEDULER_ENABLED is False
    finally:
        importlib.reload(s)
        s.settings, s.Config = orig_settings, orig_cls


def _fill_all_except(monkeypatch, settings, empties):
    """把所有必填项填成非空，再把 empties 里的置空 —— 隔离出待测字段。"""
    for f, _impact in settings._REQUIRED_FIELDS:
        monkeypatch.setattr(settings, f, "x", raising=False)
    for f in empties:
        monkeypatch.setattr(settings, f, "", raising=False)


def test_validate_skips_jira_fields_when_disabled(monkeypatch):
    """JIRA_ENABLED=false 且 JIRA_URL/PAT 空 → validate() 不含这两条。"""
    from config.settings import settings
    _fill_all_except(monkeypatch, settings, ["JIRA_URL", "JIRA_PAT"])
    monkeypatch.setattr(settings, "JIRA_ENABLED", False, raising=False)

    missing = {f for f, _ in settings.validate()}
    assert "JIRA_URL" not in missing
    assert "JIRA_PAT" not in missing
    assert missing == set()          # 其余全填 → 完全无缺失


def test_validate_reports_jira_fields_when_enabled(monkeypatch):
    """JIRA_ENABLED=true 且 JIRA_URL/PAT 空 → validate() 含这两条（附影响说明）。"""
    from config.settings import settings
    _fill_all_except(monkeypatch, settings, ["JIRA_URL", "JIRA_PAT"])
    monkeypatch.setattr(settings, "JIRA_ENABLED", True, raising=False)

    missing = dict(settings.validate())
    assert "JIRA_URL" in missing and missing["JIRA_URL"]
    assert "JIRA_PAT" in missing and missing["JIRA_PAT"]


@pytest.mark.parametrize("jira_on", [True, False])
def test_validate_other_field_reported_regardless_of_jira_flag(monkeypatch, jira_on):
    """非 Jira 必填项（如 API_KEY）无论 Jira 开关都照常告警；开关只影响 JIRA_URL/PAT。"""
    from config.settings import settings
    _fill_all_except(monkeypatch, settings, ["API_KEY"])
    monkeypatch.setattr(settings, "JIRA_ENABLED", jira_on, raising=False)

    missing = {f for f, _ in settings.validate()}
    assert "API_KEY" in missing
    # Jira 字段已填非空，故无论开关都不在缺失里（此处只验非 Jira 字段行为不变）
    assert "JIRA_URL" not in missing and "JIRA_PAT" not in missing


# ══════════════════════════════════════════════════════════════════════════════
# 2. dsw_scheduler.DSWScheduler.start() —— jira-poll 线程受 JIRA_ENABLED 控制
# ══════════════════════════════════════════════════════════════════════════════

class _FakeThread:
    """记录被创建/启动的线程 name，start() 不真正跑（避免起后台循环）。"""
    made: list = []

    def __init__(self, target=None, name=None, daemon=None, **k):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        _FakeThread.made.append(self)

    def start(self):
        self.started = True


@pytest.fixture
def sched_env(monkeypatch):
    """桩 threading.Thread + 关掉可选线程 + 捕获启动日志 + 复位 scheduler 状态。"""
    from core import dsw_scheduler as ds
    _FakeThread.made = []
    monkeypatch.setattr(ds.threading, "Thread", _FakeThread)
    # 关掉可选线程，隔离出三条核心线程（jira-poll / dsw-check / morning-report）
    for flag in ("CAPACITY_MONITOR_ENABLED", "OSS_PERM_PUSH_ENABLED",
                 "DATASET_DASHBOARD_ENABLED", "DATAFLOW_RECONCILE_ENABLED"):
        monkeypatch.setattr(ds.settings, flag, False, raising=False)
    log_calls: list = []
    monkeypatch.setattr(ds.logger, "info", lambda *a, **k: log_calls.append(a))
    ds.scheduler._running = False
    ds.scheduler._ticket_thread = None
    yield ds, log_calls
    ds.scheduler._running = False          # 复位：否则后续 start() 早返回
    ds.scheduler._ticket_thread = None
    _FakeThread.made = []


def _started_names():
    return {t.name for t in _FakeThread.made if t.started}


def test_start_skips_jira_poll_when_disabled(sched_env, monkeypatch):
    ds, log_calls = sched_env
    monkeypatch.setattr(ds.settings, "JIRA_ENABLED", False, raising=False)

    ds.scheduler.start()

    names = _started_names()
    assert "jira-poll" not in names          # 工单轮询线程不起
    assert "dsw-check" in names              # 实例检查照常
    assert "morning-report" in names         # 早报照常
    assert ds.scheduler._ticket_thread is None
    joined = " ".join(str(a) for a in log_calls)
    assert "(Jira 停用)" in joined


def test_start_launches_all_three_when_enabled(sched_env, monkeypatch):
    ds, log_calls = sched_env
    monkeypatch.setattr(ds.settings, "JIRA_ENABLED", True, raising=False)

    ds.scheduler.start()

    names = _started_names()
    assert {"jira-poll", "dsw-check", "morning-report"} <= names
    assert ds.scheduler._ticket_thread is not None
    assert ds.scheduler._ticket_thread.name == "jira-poll"
    assert ds.scheduler._ticket_thread.started is True
    joined = " ".join(str(a) for a in log_calls)
    assert "Jira 轮询" in joined


def test_stop_after_disabled_start_does_not_npe(sched_env, monkeypatch):
    """Jira 停用 → _ticket_thread 保持 None；stop() 只翻 _running，不触碰 None 线程。"""
    ds, _ = sched_env
    monkeypatch.setattr(ds.settings, "JIRA_ENABLED", False, raising=False)

    ds.scheduler.start()
    assert ds.scheduler._ticket_thread is None
    ds.scheduler.stop()                      # 不应 NPE
    assert ds.scheduler._running is False


def test_start_idempotent_when_already_running(sched_env, monkeypatch):
    """_running 已 True → start() 早返回，一个线程都不建（回归保护）。"""
    ds, _ = sched_env
    monkeypatch.setattr(ds.settings, "JIRA_ENABLED", True, raising=False)
    ds.scheduler._running = True

    ds.scheduler.start()
    assert _FakeThread.made == []


# ══════════════════════════════════════════════════════════════════════════════
# 3. routes.health() —— Jira 分支
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def health_ok_probes(monkeypatch):
    """把 Jira 以外的探测都桩成健康，让 result['status'] 只受 Jira 影响。"""
    from core.feishu_bot import routes
    import tools.aliyun.prometheus as prom
    import tools.aliyun.pai_dsw as dsw
    import utils.crypto as crypto

    monkeypatch.setattr(prom, "_query_instant", lambda *a, **k: [{"metric": {}, "value": [0, "1"]}])
    monkeypatch.setattr(dsw, "list_dsw_resources", lambda *a, **k: [])
    monkeypatch.setattr(routes, "_get_access_token", lambda: "tok")
    monkeypatch.setattr(crypto, "is_key_configured", lambda: True)
    return routes


def test_health_jira_disabled(health_ok_probes, monkeypatch):
    routes = health_ok_probes
    monkeypatch.setattr(routes.settings, "JIRA_ENABLED", False, raising=False)

    resp = routes.app.test_client().get("/health")
    body = resp.get_json()
    assert body["jira"] == "disabled"
    assert body["status"] == "ok"            # 停用不置 degraded
    assert resp.status_code == 200


def test_health_jira_not_configured_when_enabled_but_unset(health_ok_probes, monkeypatch):
    routes = health_ok_probes
    monkeypatch.setattr(routes.settings, "JIRA_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "JIRA_URL", "", raising=False)
    monkeypatch.setattr(routes.settings, "JIRA_PAT", "", raising=False)

    resp = routes.app.test_client().get("/health")
    body = resp.get_json()
    assert body["jira"] == "not_configured"  # 既有行为
    assert body["status"] == "ok"            # not_configured 也不置 degraded


def test_health_jira_probes_when_enabled_and_configured(health_ok_probes, monkeypatch):
    """启用 + 配齐 → 真探（此处桩 requests 返回 200）→ jira=='ok'。"""
    routes = health_ok_probes
    monkeypatch.setattr(routes.settings, "JIRA_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "JIRA_URL", "https://jira.example.com", raising=False)
    monkeypatch.setattr(routes.settings, "JIRA_PAT", "pat-xyz", raising=False)

    import requests

    class _Resp:
        status_code = 200

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    resp = routes.app.test_client().get("/health")
    body = resp.get_json()
    assert body["jira"] == "ok"
    assert body["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# 4. actions._h_submit_gpu_request() —— Jira 停用优雅停用
# ══════════════════════════════════════════════════════════════════════════════

def _gpu_fv():
    return {"instance_name": "train-01", "gpu_count": "1", "duration_hours": "8",
            "purpose": "微调", "priority": "中"}


class _RecordThread:
    made: list = []

    def __init__(self, target=None, **k):
        self.target = target
        _RecordThread.made.append(self)

    def start(self):
        pass


@pytest.fixture(autouse=True)
def _thread_stub(monkeypatch):
    from core.feishu_bot import actions
    _RecordThread.made = []
    monkeypatch.setattr(actions.threading, "Thread", _RecordThread)
    yield
    _RecordThread.made = []


def test_gpu_submit_disabled_returns_info_toast(monkeypatch):
    from core.feishu_bot import actions
    monkeypatch.setattr(actions.settings, "JIRA_ENABLED", False, raising=False)
    ticket_calls = []
    monkeypatch.setattr(actions, "create_gpu_ticket",
                        lambda **k: ticket_calls.append(k) or "GPU-X")

    out = actions._h_submit_gpu_request({}, "ou_1", "chat", _gpu_fv())

    assert out == {"toast": {"type": "info",
                             "content": "GPU 申请暂停（工单系统 Jira 停用中），请联系运维人员。"}}
    assert ticket_calls == []                 # 不建工单
    assert _RecordThread.made == []           # 不起后台线程


def test_gpu_submit_disabled_short_circuits_before_field_validation(monkeypatch):
    """停用分支排在字段校验前：缺实例名也回停用 toast（info），而非 error。"""
    from core.feishu_bot import actions
    monkeypatch.setattr(actions.settings, "JIRA_ENABLED", False, raising=False)

    out = actions._h_submit_gpu_request({}, "ou_1", "chat", {"gpu_count": "1"})
    assert out["toast"]["type"] == "info"
    assert "停用" in out["toast"]["content"]
    assert _RecordThread.made == []


def test_gpu_submit_empty_form_is_noop_regardless(monkeypatch):
    """空 form_value 早返回 {}（在 Jira 分支之前）。"""
    from core.feishu_bot import actions
    monkeypatch.setattr(actions.settings, "JIRA_ENABLED", False, raising=False)
    assert actions._h_submit_gpu_request({}, "ou_1", "chat", {}) == {}


def test_gpu_submit_enabled_runs_original_flow(monkeypatch):
    """启用 → 走原逻辑：同步回“处理中”卡并把建工单交给后台线程（回归保护）。"""
    from core.feishu_bot import actions
    monkeypatch.setattr(actions.settings, "JIRA_ENABLED", True, raising=False)
    monkeypatch.setattr(actions, "check_quota", lambda oid, g, d: (True, 0, 1000))
    monkeypatch.setattr(actions, "_cost_str", lambda g, d: "¥1")

    out = actions._h_submit_gpu_request({}, "ou_1", "chat", _gpu_fv())

    assert out["toast"]["type"] == "info" and "处理中" in out["toast"]["content"]
    assert "wuji-train-01" in str(out["card"])
    assert len(_RecordThread.made) == 1       # 起了后台线程建工单


# ══════════════════════════════════════════════════════════════════════════════
# 5. GPU 申请第二入口（文本路径）—— messages._process_message + gpu_flow._handle_gpu_request
#    （auditor 发现文本入口漏 gate，dev 已在两处补 JIRA_ENABLED 早返）
# ══════════════════════════════════════════════════════════════════════════════

_DISABLED_NOTICE = "GPU 申请暂停（工单系统 Jira 停用中），请联系运维人员。"


class _NoStartThread2:
    """messages._process_message 顶部会起 _auto_map_user 后台线程 —— 掐掉，避免真跑网络。"""
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


# ── ① messages 文本意图分支 ──────────────────────────────────────────────────

def test_messages_gpu_intent_disabled_returns_notice(monkeypatch, mock_feishu_send):
    """Jira 停用 → GPU 意图文本即时回停用提示，不弹 AK/GPU 卡、不置 gpu_state、不触达 _is_registered。"""
    from core.feishu_bot import messages
    monkeypatch.setattr(messages.settings, "JIRA_ENABLED", False, raising=False)
    monkeypatch.setattr(messages.threading, "Thread", _NoStartThread2)
    gpu_calls = []
    monkeypatch.setattr(messages.gpu_flow, "_send_gpu_card", lambda *a, **k: gpu_calls.append("gpu"))
    monkeypatch.setattr(messages.gpu_flow, "_send_ak_register_card", lambda *a, **k: gpu_calls.append("ak"))
    monkeypatch.setattr(messages.gpu_flow, "_set_gpu_state", lambda *a, **k: gpu_calls.append("state"))
    # 停用分支应在注册判定之前返回；置 True 以证明它未被触达。
    monkeypatch.setattr(messages, "_is_registered", lambda oid: gpu_calls.append("reg") or True)

    messages._process_message("msg_1", "chat", "申请GPU", "ou_1")

    assert any(m["text"] == _DISABLED_NOTICE for m in mock_feishu_send)
    assert gpu_calls == []          # 不弹卡、不置 state、没走到 _is_registered


def test_messages_gpu_intent_enabled_registered_shows_gpu_card(monkeypatch, mock_feishu_send):
    """启用 + 已注册 → 弹 GPU 卡 + 清空 state，不回停用提示（回归）。"""
    from core.feishu_bot import messages
    monkeypatch.setattr(messages.settings, "JIRA_ENABLED", True, raising=False)
    monkeypatch.setattr(messages.threading, "Thread", _NoStartThread2)
    monkeypatch.setattr(messages, "_is_registered", lambda oid: True)
    calls = []
    monkeypatch.setattr(messages.gpu_flow, "_send_gpu_card", lambda mid: calls.append(("gpu", mid)))
    monkeypatch.setattr(messages.gpu_flow, "_send_ak_register_card", lambda mid: calls.append(("ak", mid)))
    monkeypatch.setattr(messages.gpu_flow, "_set_gpu_state", lambda c, o, s: calls.append(("state", s)))

    messages._process_message("msg_2", "chat", "申请GPU", "ou_1")

    assert ("gpu", "msg_2") in calls
    assert ("state", {}) in calls
    assert not any(m["text"] == _DISABLED_NOTICE for m in mock_feishu_send)


def test_messages_gpu_intent_enabled_unregistered_shows_ak_card(monkeypatch, mock_feishu_send):
    """启用 + 未注册 → 弹 AK 注册卡 + 置 pending_gpu state，不回停用提示（回归）。"""
    from core.feishu_bot import messages
    monkeypatch.setattr(messages.settings, "JIRA_ENABLED", True, raising=False)
    monkeypatch.setattr(messages.threading, "Thread", _NoStartThread2)
    monkeypatch.setattr(messages, "_is_registered", lambda oid: False)
    calls = []
    monkeypatch.setattr(messages.gpu_flow, "_send_gpu_card", lambda mid: calls.append(("gpu", mid)))
    monkeypatch.setattr(messages.gpu_flow, "_send_ak_register_card", lambda mid: calls.append(("ak", mid)))
    monkeypatch.setattr(messages.gpu_flow, "_set_gpu_state", lambda c, o, s: calls.append(("state", s)))

    messages._process_message("msg_3", "chat", "申请GPU", "ou_9")

    assert ("ak", "msg_3") in calls
    assert any(c[0] == "state" and c[1].get("pending_gpu") for c in calls)
    assert ("gpu", "msg_3") not in calls
    assert not any(m["text"] == _DISABLED_NOTICE for m in mock_feishu_send)


# ── ② gpu_flow._handle_gpu_request 兜底 ─────────────────────────────────────

def _parsed():
    return {"instance_name": "train-01", "gpu_count": "2", "duration_hours": "8", "purpose": "微调"}


def test_handle_gpu_request_disabled(monkeypatch, mock_feishu_send):
    """Jira 停用 → 直接回停用提示、不调 create_gpu_ticket、不起后台线程。"""
    from core.feishu_bot import gpu_flow
    monkeypatch.setattr(gpu_flow.settings, "JIRA_ENABLED", False, raising=False)
    ticket_calls = []
    monkeypatch.setattr(gpu_flow, "create_gpu_ticket", lambda **k: ticket_calls.append(k) or "GPU-1")
    monkeypatch.setattr(gpu_flow.threading, "Thread", _RecordThread)

    gpu_flow._handle_gpu_request("msg_1", "chat", "ou_1", _parsed())

    assert any(m["text"] == _DISABLED_NOTICE for m in mock_feishu_send)
    assert ticket_calls == []            # 不建工单
    assert _RecordThread.made == []      # 不起后台线程


def test_handle_gpu_request_enabled_runs_original(monkeypatch, mock_feishu_send):
    """启用 → 回“正在提交申请”并把建工单交给后台线程；跑线程体确认调 create_gpu_ticket（回归）。"""
    from core.feishu_bot import gpu_flow
    monkeypatch.setattr(gpu_flow.settings, "JIRA_ENABLED", True, raising=False)
    ticket_calls = []
    monkeypatch.setattr(gpu_flow, "create_gpu_ticket", lambda **k: ticket_calls.append(k) or "GPU-77")
    monkeypatch.setattr(gpu_flow.threading, "Thread", _RecordThread)

    gpu_flow._handle_gpu_request("msg_2", "chat", "ou_1", _parsed())

    assert any("正在提交申请" in m["text"] for m in mock_feishu_send)
    assert not any(m["text"] == _DISABLED_NOTICE for m in mock_feishu_send)
    assert len(_RecordThread.made) == 1  # 建工单交给后台线程
    _RecordThread.made[0].target()       # 跑线程体
    assert len(ticket_calls) == 1        # 启用时确实建了工单
