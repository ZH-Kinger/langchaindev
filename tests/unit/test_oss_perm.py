"""core/oss_perm 测试：对账卡片 / 结果卡片 / apply_all 三分支 / audit_diff / 审批回调权限闸。

全部 mock 飞书与 RAM，不触网。覆盖：
  - cards.audit_card：有差异→橙色带按钮；一致→绿色无按钮；孤儿单独列
  - cards.result_card：有失败→红，全成功→绿
  - permsync.apply_all：用户存在 / 无 RAM 用户 / 失败 三种分支计数
  - permsync.audit_diff：missing / diff / ok 三态 + 孤儿过滤
  - feishu_bot._process_action("approve_oss_perm")：非管理员拒绝、管理员放行
"""
import pytest


# ── cards.audit_card ───────────────────────────────────────────��────────────

def test_audit_card_with_changes_has_approve_button():
    from core.oss_perm.cards import audit_card
    diff = {
        "rows": [
            {"name": "张三", "username": "zhangsan", "status": "missing"},
            {"name": "李四", "username": "lisi", "status": "ok"},
        ],
        "orphans": {},
        "n_diff": 1,
    }
    card = audit_card(diff)
    assert card["header"]["template"] == "orange"
    # 收集所有 action 按钮的 value.action
    actions = [a["value"]["action"]
               for el in card["elements"] if el.get("tag") == "action"
               for a in el["actions"]]
    assert "approve_oss_perm" in actions


def test_audit_card_consistent_no_button():
    from core.oss_perm.cards import audit_card
    diff = {
        "rows": [{"name": "李四", "username": "lisi", "status": "ok"}],
        "orphans": {},
        "n_diff": 0,
    }
    card = audit_card(diff)
    assert card["header"]["template"] == "green"
    assert not any(el.get("tag") == "action" for el in card["elements"])


def test_audit_card_orphans_only_still_has_button():
    """无人需同步但有孤儿策略 → 仍橙色带按钮。"""
    from core.oss_perm.cards import audit_card
    diff = {
        "rows": [{"name": "李四", "username": "lisi", "status": "ok"}],
        "orphans": {"ghost": ["GhostUser"]},
        "n_diff": 0,
    }
    card = audit_card(diff)
    assert card["header"]["template"] == "orange"
    body = "".join(el.get("text", {}).get("content", "") for el in card["elements"])
    assert "ghost" in body


def test_audit_card_diff_detail_renders():
    """status=diff 时逐桶多授/少授明细应出现在卡片文本里。"""
    from core.oss_perm.cards import audit_card
    diff = {
        "rows": [{
            "name": "王五", "username": "wangwu", "status": "diff",
            "diff": {"wuji-sing": {
                "over":  {"read": {"a/"}, "write": set()},
                "under": {"read": set(), "write": {"b/"}},
            }},
        }],
        "orphans": {},
        "n_diff": 1,
    }
    card = audit_card(diff)
    body = "".join(el.get("text", {}).get("content", "") for el in card["elements"])
    assert "王五" in body and "wuji-sing" in body
    assert "多授" in body and "少授" in body


# ── cards.result_card ─────────────────────────────────────────────────────────

def test_result_card_green_on_success():
    from core.oss_perm.cards import result_card
    card = result_card({"ok": 3, "fail": 0, "no_user": 1, "lines": ["• 张三 → zhangsan ✓"]})
    assert card["header"]["template"] == "green"


def test_result_card_red_on_failure():
    from core.oss_perm.cards import result_card
    card = result_card({"ok": 1, "fail": 2, "no_user": 0, "lines": ["• 张三 ✗ 附加失败"]})
    assert card["header"]["template"] == "red"


# ── permsync.apply_all 三分支 ─────────────────────────────────────────────────

def _plan(name, username, doc=None):
    return [{
        "member": {"name": name, "username": username},
        "policy_name": f"wuji-oss-auto-{username}",
        "resolved": {},
        "doc": doc or {"Version": "1", "Statement": []},
    }]


@pytest.fixture
def patch_apply(monkeypatch):
    """mock make_ram_client / ram_username_map / load_user_map / apply_user。"""
    from core.oss_perm import permsync
    monkeypatch.setattr(permsync, "make_ram_client", lambda: object())
    monkeypatch.setattr(permsync, "ram_username_map", lambda c: ({}, set()))
    monkeypatch.setattr(permsync, "load_user_map", lambda p: {})
    monkeypatch.setattr(permsync.time, "sleep", lambda *a: None)
    return permsync


def test_apply_all_user_exists_ok(patch_apply, monkeypatch):
    ps = patch_apply
    # 映射表/实时解析都命中 → real_user 非空 → 走 attach 分支
    monkeypatch.setattr(ps, "map_username", lambda *a: "zhangsan")
    monkeypatch.setattr(ps, "apply_user", lambda *a, **k: True)
    summary = ps.apply_all(_plan("张三", "zhangsan"))
    assert summary == {"ok": 1, "fail": 0, "no_user": 0, "lines": ["• 张三 → zhangsan ✓"]}


def test_apply_all_no_ram_user(patch_apply, monkeypatch):
    ps = patch_apply
    monkeypatch.setattr(ps, "map_username", lambda *a: None)   # 无 RAM 用户
    calls = []
    monkeypatch.setattr(ps, "apply_user",
                        lambda *a, **k: calls.append(k.get("attach")) or True)
    summary = ps.apply_all(_plan("徐悦", "xuyue"))
    assert summary["no_user"] == 1 and summary["ok"] == 0 and summary["fail"] == 0
    assert calls == [False]   # 无用户分支必须 attach=False
    assert "仅建策略未附加" in summary["lines"][0]


def test_apply_all_failure_counted(patch_apply, monkeypatch):
    ps = patch_apply
    monkeypatch.setattr(ps, "map_username", lambda *a: "lisi")
    monkeypatch.setattr(ps, "apply_user", lambda *a, **k: False)   # 下发失败
    summary = ps.apply_all(_plan("李四", "lisi"))
    assert summary["fail"] == 1 and summary["ok"] == 0
    assert summary["lines"][0].startswith("• 李四 ✗")


# ── permsync.audit_diff ───────────────────────────────────────────────────────

@pytest.fixture
def patch_audit(monkeypatch):
    from core.oss_perm import permsync
    monkeypatch.setattr(permsync, "make_ram_client", lambda: object())
    return permsync


def test_audit_diff_three_states_and_orphans(patch_audit, monkeypatch):
    ps = patch_audit
    plan = [
        {"member": {"name": "张三", "username": "zhangsan"}, "resolved": {"b": {"read": {"x/"}}}},
        {"member": {"name": "李四", "username": "lisi"}, "resolved": {"b": {"read": {"y/"}}}},
        {"member": {"name": "王五", "username": "wangwu"}, "resolved": {"b": {"read": {"z/"}}}},
    ]
    # zhangsan: RAM 无策略 → missing；lisi: 一致 → ok；wangwu: 有差异 → diff
    def fake_actual(client, username):
        return None if username == "zhangsan" else {"b": {"read": set(), "write": set()}}
    monkeypatch.setattr(ps, "ram_actual", fake_actual)

    def fake_diff(expected, actual):
        # 只有 wangwu 的 expected 含 z/ 我们让它产生差异，其余视为一致
        return {"b": {"over": {}, "under": {"read": {"z/"}}}} if expected.get("b", {}).get("read") == {"z/"} else {}
    monkeypatch.setattr(ps, "diff_resolved", fake_diff)
    # lisi 的 resolved 改成不触发 diff
    plan[1]["resolved"] = {"b": {"read": set()}}

    monkeypatch.setattr(ps, "list_auto_policies",
                        lambda c: {"zhangsan": ["Zhang"], "ghost": ["Ghost"]})

    result = ps.audit_diff(plan, active_usernames={"zhangsan", "lisi", "wangwu"})
    by_user = {r["username"]: r["status"] for r in result["rows"]}
    assert by_user == {"zhangsan": "missing", "lisi": "ok", "wangwu": "diff"}
    assert result["n_diff"] == 2                  # missing + diff
    assert result["orphans"] == {"ghost": ["Ghost"]}   # zhangsan 在职被过滤


# ── permsync.resolve_member：桶填了但读/写空 → 默认整桶读+写 ───────────────────

def _member(name, username, buckets, read, write):
    return {"name": name, "username": username, "account": f"{username}@x.com",
            "status": "在职", "buckets": buckets, "read": read, "write": write}


def test_resolve_member_bucket_only_defaults_full():
    """填了具体桶、读/写两列都空 → 该桶整桶(空前缀)读+写。"""
    from core.oss_perm import permsync
    mb = _member("韩朵", "handuo", ["新加坡-wuji-sing"], [], [])
    resolved = permsync.resolve_member(mb, {"新加坡-wuji-sing": {"egoscale/"}}, [])
    assert "wuji-sing" in resolved
    assert resolved["wuji-sing"]["read"] == {""}
    assert resolved["wuji-sing"]["write"] == {""}


def test_resolve_member_read_only_not_overridden():
    """读非空、写空 → 保持只读，不被整桶默认覆盖。"""
    from core.oss_perm import permsync
    mb = _member("张三", "zhangsan", ["新加坡-wuji-sing"], ["egoscale/"], [])
    resolved = permsync.resolve_member(mb, {"新加坡-wuji-sing": {"egoscale/"}}, [])
    assert resolved["wuji-sing"]["read"] == {"egoscale/"}
    assert resolved["wuji-sing"]["write"] == set()


def test_resolve_member_invalid_subdir_still_skipped():
    """填了非法子目录被过滤空（原始列非空）→ 不触发整桶默认，仍跳过。"""
    from core.oss_perm import permsync
    mb = _member("李四", "lisi", ["新加坡-wuji-sing"], ["nonexist/"], [])
    resolved = permsync.resolve_member(mb, {"新加坡-wuji-sing": {"egoscale/"}}, [])
    assert resolved == {}   # 非法读目录过滤后读写皆空 → 跳过


# ── feishu_bot._process_action 审批回调权限闸 ─────────────────────────────────

def test_approve_oss_perm_rejects_non_admin(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    from core.feishu_bot import _process_action
    resp = _process_action("approve_oss_perm", {}, open_id="ou_someone_else", chat_id="oc_x")
    assert resp["toast"]["type"] == "error"
    assert "管理员" in resp["toast"]["content"]


def test_approve_oss_perm_admin_starts_thread(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")

    started = []
    import core.feishu_bot as fb
    real_thread = fb.threading.Thread

    def fake_thread(*args, **kwargs):
        started.append(kwargs.get("target"))
        # 返回一个不真正运行的占位线程对象
        t = real_thread(target=lambda: None, daemon=True)
        return t
    monkeypatch.setattr(fb.threading, "Thread", fake_thread)

    resp = fb._process_action("approve_oss_perm", {}, open_id="ou_admin", chat_id="oc_x")
    assert resp["toast"]["type"] == "success"
    assert len(started) == 1   # 起了下发后台线程
