"""#57 临时 AK 发放 —— manage_temp_ak 工具（无 issue、revoke 管理员门、plan 只读、status 只读）。

安全核心：工具**没有 issue 动作**——杜绝绕审批发凭证（#55/#56 教训）。
"""
import time

import pytest

from tools.temp_ak_issuance.manage_temp_ak import manage_temp_ak, temp_ak_tool, TempAkSchema
from core.temp_ak_issuance import orchestrator as o, cleanup


# ── 无 issue 动作 ─────────────────────────────────────────────────────────────

def test_no_issue_in_schema_description():
    """schema action 描述只列 plan/status/revoke，无 issue。"""
    desc = TempAkSchema.model_fields["action"].description
    assert "issue" not in desc.lower()
    assert "plan" in desc and "status" in desc and "revoke" in desc


def test_issue_action_rejected():
    """dispatch 无 issue 分支 → 返回未知 action。"""
    out = manage_temp_ak("issue", bucket="b", expire="2099-01-01")
    assert "未知 action" in out
    assert "issue" not in out.lower() or "未知" in out


def test_tool_description_states_no_credential_issue():
    assert "审批" in temp_ak_tool.description
    # 工具描述明示不发凭证
    assert "不发凭证" in temp_ak_tool.description


# ── plan 只算 policy 不调云 ───────────────────────────────────────────────────

def test_plan_no_cloud_call(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    import utils.aliyun_sts as sts
    monkeypatch.setattr(sts, "assume_role_with_policy",
                        lambda *a, **k: pytest.fail("plan 不该调 STS"))
    import core.temp_ak_issuance.issuer as issuer_mod
    monkeypatch.setattr(issuer_mod, "permsync_client",
                        lambda *a, **k: pytest.fail("plan 不该建 RAM 客户端"))
    out = manage_temp_ak("plan", bucket="wuji-sing", read_prefixes="team/",
                         expire="2099-01-01 00:00:00")
    assert "dry-run" in out
    assert "policy" in out
    assert "本工具不发凭证" in out


def test_plan_requires_bucket_and_expire():
    assert "❌" in manage_temp_ak("plan", bucket="", expire="2099-01-01")
    assert "❌" in manage_temp_ak("plan", bucket="b", expire="")


def test_plan_bad_expire():
    out = manage_temp_ak("plan", bucket="b", expire="garbage")
    assert "❌" in out


# ── status 只读 ───────────────────────────────────────────────────────────────

def test_status_reads_grant(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    now = time.time()
    g = o.create_grant_record({
        "bucket": "wuji-sing", "read_prefixes": ["r/"], "write_prefixes": [],
        "not_before": now, "expire": now + 3600, "recipient_email": "e@x.com",
        "source_ips": [], "reason": "",
    }, instance_code="inst_tool")
    out = manage_temp_ak("status", grant_id=g["grant_id"])
    assert g["grant_id"] in out
    assert g["stage"] in out


def test_status_missing_grant_id():
    assert "❌" in manage_temp_ak("status", grant_id="")


def test_status_not_found():
    out = manage_temp_ak("status", grant_id="tak-doesnotexist")
    assert "未找到" in out


def test_status_does_not_call_cloud(monkeypatch):
    from core.oss_perm import permsync
    monkeypatch.setattr(permsync, "make_ram_client",
                        lambda: pytest.fail("status 只读、不调云"))
    manage_temp_ak("status", grant_id="tak-none")   # 不炸即证明未调云


# ── revoke 管理员门 ───────────────────────────────────────────────────────────

def test_revoke_non_admin_rejected(monkeypatch):
    monkeypatch.setattr("config.settings.settings.ADMIN_FEISHU_OPEN_ID", "ou_admin")
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda *a, **k: pytest.fail("非管理员不该调 cleanup"))
    out = manage_temp_ak("revoke", grant_id="tak-x", open_id="ou_evil")
    assert "管理员" in out


def test_revoke_admin_calls_cleanup(monkeypatch):
    monkeypatch.setattr("config.settings.settings.ADMIN_FEISHU_OPEN_ID", "ou_admin")
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    now = time.time()
    g = o.create_grant_record({
        "bucket": "b", "read_prefixes": ["r/"], "write_prefixes": [],
        "not_before": now, "expire": now + 3600, "recipient_email": "e@x.com",
        "source_ips": [], "reason": "",
    }, instance_code="inst_rev_tool")
    called = []
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda gr, *a, **k: (called.append(gr["grant_id"]) or True))
    out = manage_temp_ak("revoke", grant_id=g["grant_id"], open_id="ou_admin")
    assert called == [g["grant_id"]]
    assert "已吊销" in out


def test_revoke_admin_missing_grant_id(monkeypatch):
    monkeypatch.setattr("config.settings.settings.ADMIN_FEISHU_OPEN_ID", "ou_admin")
    out = manage_temp_ak("revoke", grant_id="", open_id="ou_admin")
    assert "❌" in out


def test_revoke_admin_grant_not_found(monkeypatch):
    monkeypatch.setattr("config.settings.settings.ADMIN_FEISHU_OPEN_ID", "ou_admin")
    out = manage_temp_ak("revoke", grant_id="tak-missing", open_id="ou_admin")
    assert "未找到" in out


# ── unknown action ────────────────────────────────────────────────────────────

def test_unknown_action():
    assert "未知 action" in manage_temp_ak("frobnicate")


# ── TOOL_GROUPS 注册 ──────────────────────────────────────────────────────────

def test_tool_registered_in_groups():
    import tools
    assert "temp_ak" in tools.TOOL_GROUPS
    assert "manage_temp_ak" in tools.TOOL_GROUPS["temp_ak"]
    assert "manage_temp_ak" in {t.name for t in tools.ALL_TOOLS}


def test_no_issue_verb_anywhere_in_tool_source():
    """静态断言：工具源码里没有 issue 动作分支（防未来误加）。"""
    import inspect
    from tools.temp_ak_issuance import manage_temp_ak as mod
    src = inspect.getsource(mod.manage_temp_ak)
    assert 'action == "issue"' not in src
    assert "issue_grant" not in src
