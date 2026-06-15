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


def test_audit_card_orphans_not_shown():
    """孤儿策略不再展示：仅孤儿（无人需同步）→ 绿色无按钮，body 不含孤儿名。"""
    from core.oss_perm.cards import audit_card
    diff = {
        "rows": [{"name": "李四", "username": "lisi", "status": "ok"}],
        "orphans": {"ghost": ["GhostUser"]},
        "n_diff": 0,
    }
    card = audit_card(diff)
    assert card["header"]["template"] == "green"
    assert not any(el.get("tag") == "action" for el in card["elements"])
    body = "".join(el.get("text", {}).get("content", "") for el in card["elements"])
    assert "ghost" not in body


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
    # summary 现含 level（缺省 dir），行尾拼了范围串（resolved={} → _scope_str 得 —）
    assert summary == {"ok": 1, "fail": 0, "no_user": 0,
                       "lines": ["• 张三 → zhangsan ✓ ｜ —"], "level": "dir"}


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
    from core.feishu_bot import actions
    real_thread = actions.threading.Thread

    def fake_thread(*args, **kwargs):
        started.append(kwargs.get("target"))
        # 返回一个不真正运行的占位线程对象
        t = real_thread(target=lambda: None, daemon=True)
        return t
    monkeypatch.setattr(actions.threading, "Thread", fake_thread)

    resp = actions._process_action("approve_oss_perm", {}, open_id="ou_admin", chat_id="oc_x")
    assert resp["toast"]["type"] == "success"
    assert len(started) == 1   # 起了下发后台线程


# ── permsync 桶级/目录级两档粒度 ──────────────────────────────────────────────

def test_coerce_level_bucket_collapses():
    """bucket 把非空读/写前缀塌缩成整桶(空串)，空集合保持空；dir 原样。"""
    from core.oss_perm import permsync
    resolved = {"b1": {"read": {"a/", "b/"}, "write": {"a/"}, "display": "桶1"},
                "b2": {"read": set(), "write": {"x/"}, "display": "桶2"}}
    assert permsync.coerce_level(resolved, "dir") is resolved          # dir 不动
    out = permsync.coerce_level(resolved, "bucket")
    assert out["b1"]["read"] == {""} and out["b1"]["write"] == {""}
    assert out["b2"]["read"] == set() and out["b2"]["write"] == {""}   # 空读保持空
    assert out["b1"]["display"] == "桶1"


def test_build_policy_bucket_vs_dir():
    """同一 resolved：桶级 → 整桶 ARN + List 无 Condition；目录级 → 前缀 ARN + List 带 Condition。"""
    from core.oss_perm import permsync
    resolved = {"wuji-ego-processed": {"read": {"batch_a/", "batch_b/"},
                                       "write": {"batch_a/"}, "display": "杭州"}}

    dir_doc = permsync.build_policy(permsync.coerce_level(resolved, "dir"))
    dir_arns = [r for st in dir_doc["Statement"] for r in st["Resource"]]
    assert any(a.endswith("/batch_a/*") for a in dir_arns)
    dir_list = next(st for st in dir_doc["Statement"] if "oss:ListObjects" in st["Action"])
    assert "Condition" in dir_list

    bk_doc = permsync.build_policy(permsync.coerce_level(resolved, "bucket"))
    bk_obj = [st for st in bk_doc["Statement"]
              if "oss:GetObject" in st["Action"] or "oss:PutObject" in st["Action"]]
    assert bk_obj and all(r.endswith("wuji-ego-processed/*")
                          for st in bk_obj for r in st["Resource"])
    bk_list = next(st for st in bk_doc["Statement"] if "oss:ListObjects" in st["Action"])
    assert "Condition" not in bk_list


def test_build_plan_threads_level():
    """build_plan 把 level 烤进 plan，并据此塌缩 resolved；默认 dir。"""
    from core.oss_perm import permsync
    mb = _member("张三", "zhangsan", ["新加坡-wuji-sing"], ["egoscale/"], ["egoscale/"])
    combos = {"新加坡-wuji-sing": {"egoscale/"}}

    plan = permsync.build_plan([mb], combos)               # 默认目录级
    assert plan[0]["level"] == "dir"
    assert plan[0]["resolved"]["wuji-sing"]["read"] == {"egoscale/"}

    planb = permsync.build_plan([mb], combos, level="bucket")
    assert planb[0]["level"] == "bucket"
    assert planb[0]["resolved"]["wuji-sing"]["read"] == {""}
    assert planb[0]["resolved"]["wuji-sing"]["write"] == {""}


def test_scope_str():
    """_scope_str：具体前缀照列、空串显示 <整桶>、空 resolved → —。"""
    from core.oss_perm.permsync import _scope_str
    resolved = {"wuji-sing": {"read": {"a/", "b/"}, "write": {""}, "display": "x"}}
    s = _scope_str(resolved)
    assert "wuji-sing" in s and "读[a/, b/]" in s and "写[<整桶>]" in s
    assert _scope_str({}) == "—"


def test_apply_all_includes_level_and_scope(patch_apply, monkeypatch):
    """apply_all 的 summary 带 level，每行拼上实际下发范围。"""
    ps = patch_apply
    monkeypatch.setattr(ps, "map_username", lambda *a: "zhangsan")
    monkeypatch.setattr(ps, "apply_user", lambda *a, **k: True)
    plan = [{"member": {"name": "张三", "username": "zhangsan"},
             "policy_name": "wuji-oss-auto-zhangsan",
             "resolved": {"wuji-sing": {"read": {""}, "write": {""}, "display": "x"}},
             "doc": {"Version": "1", "Statement": []}, "level": "bucket"}]
    summary = ps.apply_all(plan)
    assert summary["level"] == "bucket"
    assert "wuji-sing" in summary["lines"][0] and "<整桶>" in summary["lines"][0]


# ── cards.audit_form_card（选择性下发 2.0 表单卡）────────────────────────────────

def test_audit_form_card_structure():
    """有差异 → 2.0 橙卡：成员多选只含有差异者且默认全选、粒度默认桶级、提交按钮 action 正确。"""
    from core.oss_perm.cards import audit_form_card
    diff = {"rows": [
        {"name": "张三", "username": "zhangsan", "status": "missing"},
        {"name": "李四", "username": "lisi", "status": "diff",
         "diff": {"b": {"over": {"read": set(), "write": set()},
                        "under": {"read": {"x/"}, "write": set()}}}},
        {"name": "赵六", "username": "zhaoliu", "status": "ok"},
    ], "orphans": {}, "n_diff": 2}
    card = audit_form_card(diff)
    assert card["schema"] == "2.0"
    assert card["header"]["template"] == "orange"
    form = next(e for e in card["body"]["elements"] if e.get("tag") == "form")
    msel = next(e for e in form["elements"] if e.get("tag") == "multi_select_static")
    ssel = next(e for e in form["elements"] if e.get("tag") == "select_static")
    button = next(e for e in form["elements"] if e.get("tag") == "button")
    assert [o["value"] for o in msel["options"]] == ["zhangsan", "lisi"]   # ok 行排除
    assert msel["selected_values"] == ["zhangsan", "lisi"]                 # 默认全选
    assert ssel["initial_option"] == "bucket"
    assert button["behaviors"][0]["value"]["action"] == "approve_oss_perm_selected"


def test_audit_form_card_consistent_green():
    """无差异 → 绿色、无表单。"""
    from core.oss_perm.cards import audit_form_card
    diff = {"rows": [{"name": "李四", "username": "lisi", "status": "ok"}],
            "orphans": {}, "n_diff": 0}
    card = audit_form_card(diff)
    assert card["header"]["template"] == "green"
    assert not any(e.get("tag") == "form" for e in card["body"]["elements"])


# ── actions._h_approve_oss_perm_selected（选择性下发权限/输入闸）────────────────

def test_approve_selected_non_admin(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    from core.feishu_bot import actions
    resp = actions._process_action("approve_oss_perm_selected", {},
                                   open_id="ou_x", chat_id="oc_x",
                                   form_value={"level": "bucket", "selected": ["zhangsan"]})
    assert resp["toast"]["type"] == "error" and "管理员" in resp["toast"]["content"]


def test_approve_selected_empty_selection(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    from core.feishu_bot import actions
    resp = actions._process_action("approve_oss_perm_selected", {},
                                   open_id="ou_admin", chat_id="oc_x",
                                   form_value={"level": "bucket", "selected": []})
    assert resp["toast"]["type"] == "error" and "未选择" in resp["toast"]["content"]


def test_approve_selected_admin_starts_thread(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    started = []
    from core.feishu_bot import actions
    real_thread = actions.threading.Thread

    def fake_thread(*args, **kwargs):
        started.append(kwargs.get("target"))
        return real_thread(target=lambda: None, daemon=True)
    monkeypatch.setattr(actions.threading, "Thread", fake_thread)

    resp = actions._process_action("approve_oss_perm_selected", {},
                                   open_id="ou_admin", chat_id="oc_x",
                                   form_value={"level": "bucket", "selected": ["zhangsan", "lisi"]})
    assert resp["toast"]["type"] == "success"
    assert "2 人" in resp["toast"]["content"]
    assert len(started) == 1
