"""跨云迁移确认幂等：连点“确认迁移”只真正下发一次，其余回“已下发”。

复现的线上问题：transfer 的 confirm handler 早先没有 launched 标记/锁（不像 cpfs/bucket），
用户连点不同确认卡（msg_id 不同、内容去重挡不住）→ 每次都起后台线程刷卡。
"""
import pytest


@pytest.fixture
def A(monkeypatch):
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    # run_to_completion 置空：避免线程里真的建 MGW 任务/推卡
    monkeypatch.setattr(orchestrator, "run_to_completion", lambda job, **k: job)
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b: False)
    monkeypatch.setattr(orchestrator, "set_same_name_policy", lambda job, p: job)
    return actions, orchestrator


def _new_job():
    return {"job_id": "tr-abc123", "stage": "NEW", "bytes_total": 1, "source": "tos://a/x/",
            "dest": "oss://b/y/", "same_name_policy": "skip"}


def test_confirm_launches_once_then_blocks(A, monkeypatch):
    actions, orch = A
    store = {"tr-abc123": _new_job()}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda job: store.__setitem__(job["job_id"], job))

    r1 = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r1["toast"]["type"] == "success"
    assert store["tr-abc123"].get("launched") is True
    # 确认后原地替换为“进行中”卡（确认按钮消失，无法再连点）
    assert r1.get("card", {}).get("type") == "raw"
    assert "进行中" in str(r1["card"]["data"]) or "迁移中" in str(r1["card"]["data"])

    # 第二次连点：launched 已置 → 直接挡回
    r2 = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r2["toast"]["type"] == "info"
    assert "已下发" in r2["toast"]["content"]


def test_retry_clears_launched_and_relaunches(A, monkeypatch):
    actions, orch = A
    failed = _new_job()
    failed["stage"] = "FAILED"
    failed["launched"] = True
    store = {"tr-abc123": failed}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda job: store.__setitem__(job["job_id"], job))
    # 先把锁占上，模拟上次下发残留
    from utils.redis_client import get_redis
    get_redis().set("transfer:launch:tr-abc123", 1)

    r = actions._h_retry_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success"          # 重试成功再次下发
    assert store["tr-abc123"]["stage"] != "FAILED"
    assert store["tr-abc123"]["launched"] is True    # 重新下发后再次置位


def test_reparse_running_path_shows_progress_not_confirm(monkeypatch):
    """对已在跑的同一源/目的再点“解析并预估”→ 回进度卡，不再弹确认卡、也不重复慢预估。"""
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    import core.transfer.cards as cards
    import core.dsw_scheduler as sched

    class _Sync:   # 让后台线程同步执行，便于断言
        def __init__(self, target=None, daemon=None, **k): self._t = target
        def start(self): self._t()
    monkeypatch.setattr(actions.threading, "Thread", _Sync)

    class _Plan:
        engine = "mgw"
        direction = "tos→oss"
    monkeypatch.setattr(orchestrator, "make_plan", lambda s, d="": _Plan())
    monkeypatch.setattr(orchestrator, "_job_id", lambda plan: "tr-run1")
    crossing = {"job_id": "tr-run1", "stage": orchestrator.STAGE_CROSSING}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: crossing)
    monkeypatch.setattr(orchestrator, "refresh", lambda jid: crossing)

    def _boom(*a, **k):
        raise AssertionError("已在跑不应再跑慢预估")
    monkeypatch.setattr(orchestrator, "estimate_source", _boom)
    monkeypatch.setattr(cards, "progress_card", lambda j: {"PROGRESS": j["job_id"]})

    def _no_confirm(*a, **k):
        raise AssertionError("已在跑不应再弹确认卡")
    monkeypatch.setattr(cards, "confirm_card", _no_confirm)
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append(card))

    actions._h_submit_transfer({"action": "submit_transfer"}, "ou_x", "chat",
                               {"source": "tos://a/x/", "dest": "oss://b/y/"})
    assert sent == [{"PROGRESS": "tr-run1"}]


# ── confirm 分支：审批门 / 已在途 / launch 锁独立 ────────────────────────────────

def test_confirm_stage_not_new_returns_info(A, monkeypatch):
    """job 已在 CROSSING（非 NEW/FAILED）→ 直接回“无需重复”，不下发。"""
    actions, orch = A
    job = _new_job(); job["stage"] = "CROSSING"
    monkeypatch.setattr(orch, "get_job", lambda jid: job)
    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "无需重复" in r["toast"]["content"]


def test_confirm_approval_blocks_non_admin_and_no_launch(A, monkeypatch):
    """超阈值 + 非管理员 → error toast；且 launched 不置、launch 锁不被占（管理员随后仍能下发）。"""
    actions, orch = A
    monkeypatch.setattr(orch, "needs_approval", lambda b: True)
    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    job = _new_job(); job["bytes_total"] = 10 ** 15
    store = {"tr-abc123": job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))

    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_not_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "审批" in r["toast"]["content"]
    assert store["tr-abc123"].get("launched") is not True     # 未下发
    from utils.redis_client import get_redis
    assert get_redis().get("transfer:launch:tr-abc123") is None   # 锁未被占，admin 仍可下发


def test_confirm_launch_lock_blocks_even_if_flag_unset(A, monkeypatch):
    """launched 标记还没落，但 launch NX 锁已被上一次点击占住 → 第二次回“已下发”，不重复起线程。"""
    actions, orch = A
    launched_calls = []
    monkeypatch.setattr(orch, "run_to_completion",
                        lambda job, **k: launched_calls.append(job["job_id"]) or job)
    job = _new_job()   # launched 未置、stage NEW
    store = {"tr-abc123": job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    from utils.redis_client import get_redis
    get_redis().set("transfer:launch:tr-abc123", 1)   # 上次点击残留的锁

    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "已下发" in r["toast"]["content"]
    assert launched_calls == []                        # 真正未下发


def test_confirm_return_card_sinking_stage(A, monkeypatch):
    """需沉降且未完成 → 原地进行中卡的展示阶段为 SINKING（否则 CROSSING）。

    confirm 默认 reply_v2=True → 走 progress_card_v2（2.0），故 patch/断言 v2 卡。
    """
    actions, orch = A
    import core.transfer.cards as cards
    job = _new_job(); job["needs_sink"] = True; job["sink_done"] = False
    store = {"tr-abc123": job}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(cards, "progress_card_v2", lambda j: {"P2": j["stage"]})
    # 同步线程避免后台噪声
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["card"]["data"] == {"P2": orch.STAGE_SINKING}


# ── submit：confirmcard 30s 去重锁 / 重解析 DONE → 结果卡 ─────────────────────────

class _SyncThread:
    def __init__(self, target=None, daemon=None, **k): self._t = target
    def start(self): self._t()


def test_reparse_confirmcard_lock_dedups(monkeypatch):
    """连点“解析并预估”：第二次被 transfer:confirmcard NX 锁挡 → 只预估一次、只推一张确认卡。"""
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    import core.transfer.cards as cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    class _Plan:
        engine = "mgw"; direction = "tos→oss"
    monkeypatch.setattr(orchestrator, "make_plan", lambda s, d="": _Plan())
    monkeypatch.setattr(orchestrator, "_job_id", lambda plan: "tr-lock1")
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: None)     # 任务未启动
    est = []
    monkeypatch.setattr(orchestrator, "estimate_source", lambda plan: (est.append(1) or (1, 1)))
    monkeypatch.setattr(orchestrator, "create_job_record",
                        lambda plan, **k: {"job_id": "tr-lock1", "stage": "NEW", "bytes_total": 1})
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b: False)
    monkeypatch.setattr(cards, "confirm_card", lambda job, need_approval=False: {"CONFIRM": job["job_id"]})
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append(card))

    fv = {"source": "tos://a/x/", "dest": "oss://b/y/"}
    actions._h_submit_transfer({"action": "submit_transfer"}, "ou_x", "chat", fv)
    actions._h_submit_transfer({"action": "submit_transfer"}, "ou_x", "chat", fv)
    assert len(est) == 1                       # 慢预估只跑一次
    assert sent == [{"CONFIRM": "tr-lock1"}]   # 确认卡只推一张


def test_reparse_done_shows_result_not_confirm(monkeypatch):
    """对当天已完成的同源/目的再点“解析并预估”→ 回结果卡，不再预估、不弹确认卡。"""
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    import core.transfer.cards as cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    class _Plan:
        engine = "mgw"; direction = "tos→oss"
    monkeypatch.setattr(orchestrator, "make_plan", lambda s, d="": _Plan())
    monkeypatch.setattr(orchestrator, "_job_id", lambda plan: "tr-done1")
    done = {"job_id": "tr-done1", "stage": orchestrator.STAGE_DONE}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: done)
    monkeypatch.setattr(orchestrator, "refresh", lambda jid: done)

    def _boom(*a, **k):
        raise AssertionError("已完成不应再跑慢预估")
    monkeypatch.setattr(orchestrator, "estimate_source", _boom)
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["job_id"]})

    def _no_confirm(*a, **k):
        raise AssertionError("已完成不应弹确认卡")
    monkeypatch.setattr(cards, "confirm_card", _no_confirm)
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append(card))

    actions._h_submit_transfer({"action": "submit_transfer"}, "ou_x", "chat",
                               {"source": "tos://a/x/", "dest": "oss://b/y/"})
    assert sent == [{"RESULT": "tr-done1"}]


# ── _on_update 终态：跨线程 NX 闸门保证只推一次 ─────────────────────────────────

def test_on_update_terminal_pushes_result_once(monkeypatch):
    """run_to_completion 对同一终态 job 回调两次（重复回调/竞态）→ 结果卡只推一次（_claim_dataflow_notify 吞第二次）。"""
    from core.feishu_bot import actions
    from core.transfer import orchestrator
    import core.transfer.cards as cards
    import core.dsw_scheduler as sched

    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b: False)
    monkeypatch.setattr(orchestrator, "set_same_name_policy", lambda job, p: job)

    job = _new_job()
    store = {"tr-once": {**job, "job_id": "tr-once"}}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orchestrator, "_save", lambda j: store.__setitem__(j["job_id"], j))

    done = {**store["tr-once"], "stage": orchestrator.STAGE_DONE}

    def _rtc(j, on_update=None, **k):
        on_update(done); on_update(done)    # 同 job_id 连回两次
        return done
    monkeypatch.setattr(orchestrator, "run_to_completion", _rtc)
    monkeypatch.setattr(cards, "result_card", lambda j: {"RESULT": j["job_id"]})
    monkeypatch.setattr(cards, "progress_card", lambda j: {"PROGRESS": j["job_id"]})
    sent = []
    monkeypatch.setattr(sched, "_send_card", lambda o, c, card: sent.append(card))

    r = actions._h_confirm_transfer({"job_id": "tr-once"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success"
    assert sent == [{"RESULT": "tr-once"}]   # 终态结果卡只推一次


# ── 200830 回归：确认路 2.0→2.0、retry 路 1.0→1.0（同家族原地替换） ───────────────

def test_confirm_default_returns_schema_2_0(A, monkeypatch):
    """确认成功下发 → 原地替换卡是 schema 2.0（与确认卡同家族，避开飞书 200830）。

    不 patch 卡函数，走真实 progress_card_v2，验证返回体形状。
    """
    actions, orch = A
    store = {"tr-abc123": _new_job()}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success"
    assert r["card"]["type"] == "raw"
    data = r["card"]["data"]
    assert data["schema"] == "2.0"          # 核心保证：同 schema 家族替换
    assert "form" not in str(data)          # 纯展示，无 form
    assert "button" not in str(data)        # 无按钮 → 不能再连点


def test_retry_returns_schema_1_0(A, monkeypatch):
    """retry 由 1.0 结果卡触发 → 原地替换回 1.0 progress_card（无 schema 键，1.0→1.0）。"""
    actions, orch = A
    failed = _new_job(); failed["stage"] = "FAILED"; failed["launched"] = True
    store = {"tr-abc123": failed}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)

    r = actions._h_retry_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "success"
    assert r["card"]["type"] == "raw"
    assert "schema" not in r["card"]["data"]   # 老式 1.0 卡（card() 不写 schema 键）


# ── progress_card_v2 纯渲染：各 stage + 缺省字段不抛 KeyError ─────────────────────

@pytest.mark.parametrize("stage,expect_cn", [
    ("SINKING", "沉降中"),
    ("CROSSING", "跨云迁移中"),
    ("WHATEVER", "WHATEVER"),   # 未知 stage 原样透传，不炸
])
def test_progress_card_v2_renders_each_stage(stage, expect_cn):
    from core.transfer import cards
    job = {"job_id": "tr-x", "stage": stage, "source": "tos://a/x/", "dest": "oss://b/y/"}
    d = cards.progress_card_v2(job)
    assert d["schema"] == "2.0"
    assert expect_cn in d["header"]["title"]["content"]
    body_md = d["body"]["elements"][0]["content"]
    assert expect_cn in body_md and "tr-x" in body_md
    assert "form" not in str(d) and "button" not in str(d)


def test_progress_card_v2_missing_optional_fields_no_keyerror():
    """job 只有必备字段、无 same_name_policy/overwrite_mode → _same_name_label 兜底不抛 KeyError。"""
    from core.transfer import cards
    job = {"job_id": "tr-min", "stage": "CROSSING", "source": "tos://a/x/", "dest": "oss://b/y/"}
    d = cards.progress_card_v2(job)          # 不应抛异常
    assert d["schema"] == "2.0"
    # 同名策略缺省回落到“跳过同名文件”
    assert "跳过同名文件" in d["body"]["elements"][0]["content"]


# ── confirm 各 early-return：只回 toast，不带 card 键 ─────────────────────────────

def test_confirm_early_returns_carry_no_card(A, monkeypatch):
    actions, orch = A
    from utils.redis_client import get_redis

    # 1) 缺 job_id → error toast
    r = actions._h_confirm_transfer({}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "card" not in r

    # 2) 任务不存在 → error toast
    monkeypatch.setattr(orch, "get_job", lambda jid: None)
    r = actions._h_confirm_transfer({"job_id": "nope"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "card" not in r

    # 3) stage 非 NEW/FAILED → info “无需重复”
    j = _new_job(); j["stage"] = "CROSSING"
    monkeypatch.setattr(orch, "get_job", lambda jid: j)
    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "无需重复" in r["toast"]["content"]
    assert "card" not in r

    # 4) 已 launched → info “已下发”
    j = _new_job(); j["launched"] = True
    monkeypatch.setattr(orch, "get_job", lambda jid: j)
    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "card" not in r

    # 5) 超阈值 + 非管理员 → error toast
    monkeypatch.setattr(orch, "needs_approval", lambda b: True)
    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    j = _new_job(); j["bytes_total"] = 10 ** 15
    monkeypatch.setattr(orch, "get_job", lambda jid: j)
    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_not_admin", "chat", {})
    assert r["toast"]["type"] == "error" and "审批" in r["toast"]["content"]
    assert "card" not in r

    # 6) launch NX 锁已被占（launched 未落）→ info “已下发”
    monkeypatch.setattr(orch, "needs_approval", lambda b: False)
    store = {"tr-abc123": _new_job()}
    monkeypatch.setattr(orch, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orch, "_save", lambda j: store.__setitem__(j["job_id"], j))
    get_redis().set("transfer:launch:tr-abc123", 1)
    r = actions._h_confirm_transfer({"job_id": "tr-abc123"}, "ou_admin", "chat", {})
    assert r["toast"]["type"] == "info" and "已下发" in r["toast"]["content"]
    assert "card" not in r
