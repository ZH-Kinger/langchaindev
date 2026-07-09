"""HIGH#4（#37）：提交类回调不在 3s 死线内做同步网络调用。

- `_h_submit_gpu_request`：`_get_user_name`（飞书 HTTP）+ `create_gpu_ticket`（Jira）移进 `_do()` 线程。
- `_h_submit_ak_register`：RAM 映射（`list_ram_users_api` 阿里分页 + `_get_user_name` 飞书 HTTP）
  整块移进 `_link_ram` 线程；卡片不再引用已删的 `display` 变量。

手法：用「捕获但不启动」的线程桩，把线程体里会调的网络函数换成 recorder/raiser，
证明 sync 路径完全没碰它们、且仍成功回卡。
"""
import pytest

from core.feishu_bot import actions


class _NoStartThread:
    """捕获后台线程构造，但 start() 不运行 —— 证明 sync 路径不碰线程体里的网络调用。"""
    def __init__(self, target=None, args=(), daemon=None, **k):
        self.target = target
        _NoStartThread.made.append(self)

    def start(self):
        pass


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _NoStartThread.made = []
    monkeypatch.setattr(actions.threading, "Thread", _NoStartThread)
    yield
    _NoStartThread.made = []


# ── GPU 申请提交 ────────────────────────────────────────────────────────────────

def _gpu_fv():
    return {"instance_name": "train-01", "gpu_count": "1", "duration_hours": "8",
            "purpose": "微调", "priority": "中"}


def test_gpu_submit_no_network_in_sync(monkeypatch):
    """sync 立即回“处理中”卡；_get_user_name / create_gpu_ticket 均不在 sync 被调用。"""
    monkeypatch.setattr(actions, "check_quota", lambda oid, g, d: (True, 0, 1000))
    monkeypatch.setattr(actions, "_cost_str", lambda g, d: "¥1")

    from tools.feishu import notify
    from tools.jira import ticket
    name_calls, ticket_calls = [], []
    monkeypatch.setattr(notify, "_get_user_name",
                        lambda oid: name_calls.append(oid) or "张三")
    monkeypatch.setattr(actions, "create_gpu_ticket",
                        lambda **k: ticket_calls.append(k) or "GPU-1")

    out = actions._h_submit_gpu_request({}, "ou_1", "chat", _gpu_fv())
    assert out["toast"]["type"] == "info" and "处理中" in out["toast"]["content"]
    assert "wuji-train-01" in str(out["card"])
    # 关键：sync 路径没有任何网络调用
    assert name_calls == [] and ticket_calls == []
    # 但确实把活儿交给了后台线程
    assert len(_NoStartThread.made) == 1 and _NoStartThread.made[0].target is not None


def test_gpu_submit_survives_even_if_get_user_name_would_raise(monkeypatch):
    """把 _get_user_name 换成抛异常：sync 仍成功回卡（证明它不在 sync 路径）。"""
    monkeypatch.setattr(actions, "check_quota", lambda oid, g, d: (True, 0, 1000))
    monkeypatch.setattr(actions, "_cost_str", lambda g, d: "¥1")
    from tools.feishu import notify

    def _boom(oid):
        raise RuntimeError("飞书 HTTP 挂了")
    monkeypatch.setattr(notify, "_get_user_name", _boom)

    out = actions._h_submit_gpu_request({}, "ou_1", "chat", _gpu_fv())
    assert out["toast"]["type"] == "info"          # 未受 _get_user_name 异常影响


def test_gpu_submit_quota_block_is_sync_error(monkeypatch):
    """配额不足是本地判定 → 同步 error toast，不起线程。"""
    monkeypatch.setattr(actions, "check_quota", lambda oid, g, d: (False, 999, 100))
    out = actions._h_submit_gpu_request({}, "ou_1", "chat", _gpu_fv())
    assert out["toast"]["type"] == "error" and "配额" in out["toast"]["content"]
    assert _NoStartThread.made == []


def test_gpu_submit_missing_fields_sync(monkeypatch):
    monkeypatch.setattr(actions, "check_quota", lambda oid, g, d: (True, 0, 1000))
    out = actions._h_submit_gpu_request({}, "ou_1", "chat", {"gpu_count": "1"})
    assert out["toast"]["type"] == "error"         # 缺实例名
    assert _NoStartThread.made == []


# ── AK 绑定提交 ─────────────────────────────────────────────────────────────────

def _ak_fv(ram_user_name=""):
    fv = {"ak_id": "LTAItestAK1234567890", "ak_secret": "sk_secret_value"}
    if ram_user_name:
        fv["ram_user_name"] = ram_user_name
    return fv


def test_ak_submit_ram_mapping_not_in_sync(monkeypatch):
    """list_ram_users_api / _get_user_name 移进 _link_ram 线程 → sync 不调用。"""
    from tools.aliyun import ram
    from tools.feishu import notify
    ram_calls, name_calls = [], []
    monkeypatch.setattr(ram, "list_ram_users_api", lambda: ram_calls.append(1) or [])
    monkeypatch.setattr(notify, "_get_user_name", lambda oid: name_calls.append(oid) or "李四")

    out = actions._h_submit_ak_register({}, "ou_1", "chat", _ak_fv())
    assert out["toast"]["type"] == "success" and "已加密保存" in out["toast"]["content"]
    assert ram_calls == [] and name_calls == []     # sync 无 RAM/飞书网络
    # RAM 映射交给了后台线程
    assert any(t.target is not None for t in _NoStartThread.made)


def test_ak_submit_survives_ram_api_raising(monkeypatch):
    """list_ram_users_api 抛异常也不影响 sync 成功回卡（证明其在后台线程）。"""
    from tools.aliyun import ram
    monkeypatch.setattr(ram, "list_ram_users_api",
                        lambda: (_ for _ in ()).throw(RuntimeError("RAM 分页挂了")))
    out = actions._h_submit_ak_register({}, "ou_1", "chat", _ak_fv())
    assert out["toast"]["type"] == "success"


def test_ak_card_renders_without_removed_display_var(monkeypatch):
    """卡片不再引用已删的 `display` 变量：两个分支都能正常渲染、不抛 NameError。"""
    # 分支①：用户显式填了 ram_user_name → “指定 RAM 用户：…（后台关联中…）”
    out1 = actions._h_submit_ak_register({}, "ou_1", "chat", _ak_fv(ram_user_name="wangzh"))
    body1 = str(out1["card"])
    assert "指定 RAM 用户" in body1 and "wangzh" in body1 and "后台关联中" in body1
    assert "AccessKey 已绑定" in body1

    # 分支②：未填 → “RAM 用户：后台自动关联中…”
    out2 = actions._h_submit_ak_register({}, "ou_2", "chat", _ak_fv())
    body2 = str(out2["card"])
    assert "后台自动关联中" in body2


def test_ak_bad_id_is_sync_error(monkeypatch):
    out = actions._h_submit_ak_register({}, "ou_1", "chat",
                                        {"ak_id": "BADPREFIX123", "ak_secret": "s"})
    assert out["toast"]["type"] == "error" and "LTAI" in out["toast"]["content"]
    assert _NoStartThread.made == []
