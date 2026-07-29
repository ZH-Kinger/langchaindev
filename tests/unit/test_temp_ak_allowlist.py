"""#58 A —— 全局审批白名单硬门（core/feishu_bot/routes.py）。

`_approval_allowlist()` = {FEISHU_RAM_APPROVAL_CODE + 启用时 TEMP_AK_APPROVAL_CODE/TEMP_AK_EXTEND_APPROVAL_CODE}。
feishu_event：审批事件 code 在白名单→照常分发；不在白名单 或 无 code → 记日志丢弃返回 {"code":0}、
不进任何处理器；带 RAM code 仍进 ram_approval；非审批消息不受白名单影响。
"""
import pytest

from core.feishu_bot import routes
from core import ram_approval

RAM = "RAMCODE-1111"
TAK = "TAKCODE-2222"
EXT = "EXTCODE-3333"
CODE_1949 = "0133C4FC-8793-4FF3-A759-C4ECE8AC1FF9"   # 第二主账号「数据访问凭证申请（产线）」真机 code


@pytest.fixture(autouse=True)
def _no_extra_accounts(monkeypatch):
    """#63 后白名单会并入**所有已注册账号**的发放 code。本文件按「恰好三个 code」断言，
    因此必须把第二主账号档案清干净——否则跑在配了 1949 的环境（服务器 .env / 容器）上
    会多出一个真实 code 而红。多账号自身的白名单行为在
    `test_temp_ak_account_isolation.py` 里单独覆盖。"""
    for name in ("TEMP_AK_1949_APPROVAL_CODE",
                 "ALIYUN_1949_ACCESS_KEY_ID", "ALIYUN_1949_ACCESS_KEY_SECRET"):
        monkeypatch.setattr(routes.settings, name, "", raising=False)


@pytest.fixture
def wl_env(monkeypatch):
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", RAM)
    monkeypatch.setattr(routes.settings, "TEMP_AK_APPROVAL_CODE", TAK)
    monkeypatch.setattr(routes.settings, "TEMP_AK_EXTEND_APPROVAL_CODE", EXT)
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", True)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", "")


@pytest.fixture
def spawn_spy(monkeypatch):
    """捕获 feishu_event 里起的后台线程 target（不真跑）。"""
    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            self.target = target
            self.args = args

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(routes.threading, "Thread", FakeThread)
    # 事件去重永不命中
    monkeypatch.setattr(routes.messages, "_is_duplicate_event", lambda eid: False)
    return started


def _approval_event(code, eid="e1", status="APPROVED"):
    ev = {"header": {"event_type": "approval_instance", "event_id": eid, "token": ""},
          "event": {"instance_code": "i", "status": status}}
    if code:
        ev["event"]["approval_code"] = code
    return ev


def _post(client, payload):
    return client.post("/feishu/event", json=payload)


@pytest.fixture
def client():
    routes.app.config["TESTING"] = True
    return routes.app.test_client()


# ── _approval_allowlist() 集合内容 ────────────────────────────────────────────

def test_allowlist_includes_all_three_when_enabled(wl_env):
    al = routes._approval_allowlist()
    assert al == {RAM, TAK, EXT}


def test_allowlist_grows_to_four_with_second_aliyun_account(wl_env, monkeypatch):
    """#63 正向面：注册第二主账号（1949）后白名单**应当**变成 4 个并含它的发放 code。

    上面那条 autouse 把 1949 清空是为了让「恰好三个」可断言；光有它会把本次新功能测漏，
    故这里正向钉一次：清配置 ≠ 该功能不存在。（延长/撤销复用 EXT，不新增 code。）"""
    monkeypatch.setattr(routes.settings, "TEMP_AK_1949_APPROVAL_CODE", CODE_1949)
    monkeypatch.setattr(routes.settings, "ALIYUN_1949_ACCESS_KEY_ID", "LTAI_1949")
    monkeypatch.setattr(routes.settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "SK_1949")
    al = routes._approval_allowlist()
    assert al == {RAM, TAK, EXT, CODE_1949}
    assert len(al) == 4


def test_allowlist_second_account_code_routes_to_issue_handler(wl_env, monkeypatch,
                                                               spawn_spy, client):
    """并且这个 code 真能被放行、进发放处理器（不是只进集合、事件却被丢弃）。"""
    monkeypatch.setattr(routes.settings, "TEMP_AK_1949_APPROVAL_CODE", CODE_1949)
    monkeypatch.setattr(routes.settings, "ALIYUN_1949_ACCESS_KEY_ID", "LTAI_1949")
    monkeypatch.setattr(routes.settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "SK_1949")
    from core.temp_ak_issuance import approval as tak_approval
    resp = _post(client, _approval_event(CODE_1949, eid="e_1949"))
    assert resp.status_code == 200
    assert spawn_spy == [tak_approval.handle_temp_ak_event]


def test_allowlist_excludes_temp_ak_when_disabled(monkeypatch):
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", RAM)
    monkeypatch.setattr(routes.settings, "TEMP_AK_APPROVAL_CODE", TAK)
    monkeypatch.setattr(routes.settings, "TEMP_AK_EXTEND_APPROVAL_CODE", EXT)
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", False)
    al = routes._approval_allowlist()
    assert al == {RAM}
    assert TAK not in al and EXT not in al


def test_allowlist_empty_ram_code(monkeypatch):
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", "")
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", False)
    assert routes._approval_allowlist() == set()


# ── feishu_event 分发（白名单命中）───────────────────────────────────────────

def test_ram_code_dispatches_to_ram_handler(wl_env, spawn_spy, client):
    resp = _post(client, _approval_event(RAM))
    assert resp.get_json() == {"code": 0}
    assert ram_approval.handle_approval_event in spawn_spy


def test_temp_ak_code_dispatches_to_temp_ak_handler(wl_env, spawn_spy, client):
    from core.temp_ak_issuance import approval as tak
    resp = _post(client, _approval_event(TAK))
    assert resp.get_json() == {"code": 0}
    assert tak.handle_temp_ak_event in spawn_spy
    assert ram_approval.handle_approval_event not in spawn_spy


def test_extend_code_dispatches_to_extend_handler(wl_env, spawn_spy, client):
    from core.temp_ak_issuance import approval as tak
    resp = _post(client, _approval_event(EXT))
    assert resp.get_json() == {"code": 0}
    assert tak.handle_temp_ak_extend_event in spawn_spy
    assert ram_approval.handle_approval_event not in spawn_spy


# ── 非白名单 / 无 code 丢弃 ───────────────────────────────────────────────────

def test_non_allowlisted_code_dropped(wl_env, spawn_spy, client):
    resp = _post(client, _approval_event("SOME-OTHER-CODE"))
    assert resp.get_json() == {"code": 0}
    assert spawn_spy == []          # 不进任何处理器


def test_no_code_approval_dropped(wl_env, spawn_spy, client):
    """审批事件（event_type approval）但无 approval_code → 白名单丢弃、不进处理器。"""
    resp = _post(client, _approval_event(None))
    assert resp.get_json() == {"code": 0}
    assert spawn_spy == []


def test_temp_ak_code_dropped_when_disabled(monkeypatch, spawn_spy, client):
    """TEMP_AK_ENABLED=False → temp_ak code 不在白名单 → 丢弃。"""
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", RAM)
    monkeypatch.setattr(routes.settings, "TEMP_AK_APPROVAL_CODE", TAK)
    monkeypatch.setattr(routes.settings, "TEMP_AK_EXTEND_APPROVAL_CODE", EXT)
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", False)
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", "")
    resp = _post(client, _approval_event(TAK))
    assert resp.get_json() == {"code": 0}
    assert spawn_spy == []


# ── 非审批消息不受白名单影响 ─────────────────────────────────────────────────

def test_non_approval_message_not_dropped_by_allowlist(wl_env, spawn_spy, client):
    """普通消息事件（非审批）→ is_approval_like False → 跳过白名单，走消息路径（非 text 早退）。"""
    payload = {"header": {"event_type": "im.message.receive_v1", "event_id": "m1", "token": ""},
               "event": {"message": {"message_type": "image", "message_id": "mid"}}}
    resp = _post(client, payload)
    assert resp.get_json() == {"code": 0}
    # 未被当审批丢弃、也未误起审批处理器
    assert ram_approval.handle_approval_event not in spawn_spy


def test_is_approval_like_false_for_plain_message(wl_env):
    payload = {"header": {"event_type": "im.message.receive_v1"},
               "event": {"message": {"message_type": "text"}}}
    summary = ram_approval.event_log_summary(payload)
    is_approval_like = ("approval" in (summary.get("event_type") or "").lower()
                        or bool(summary.get("approval_code")))
    assert is_approval_like is False
