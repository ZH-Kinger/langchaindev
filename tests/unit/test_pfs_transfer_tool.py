"""PFS 跨云直传 —— Agent 工具 manage_pfs_transfer（plan/apply/status）。

plan 只读、apply 起链（mock run_to_completion + 同步线程）、status、需审批未 force 拦、路径错误回错。
"""
import json

import pytest

from tools.pfs_transfer import pfs_transfer as tp
from core.pfs_transfer import orchestrator as o
from config.settings import settings

VEPFS_FS = "vepfs-abc"
CPFS_FS = "cpfs-xyz"
_MAP = {
    f"vepfs://{VEPFS_FS}": {"region": "cn-beijing", "tos_bucket": "wuji-dc-bj", "tos_prefix": "pfs-staging/"},
    f"cpfs://{CPFS_FS}": {"region": "cn-hangzhou", "oss_bucket": "wuji-il-hz",
                          "oss_prefix": "pfs-staging/", "dataflow_id": "df-123"},
}
SRC = f"vepfs://{VEPFS_FS}/wzh/"
DST = f"cpfs://{CPFS_FS}/team/"


@pytest.fixture(autouse=True)
def _map(monkeypatch):
    monkeypatch.setattr(settings, "PFS_STAGING_MAP_RAW", json.dumps(_MAP))


class _SyncThread:
    def __init__(self, target=None, args=(), daemon=None, **k):
        self._t, self._a = target, args

    def start(self):
        if self._t:
            self._t(*self._a)


# ══════════════════════════════════════════════════════════════════════════════
# status
# ══════════════════════════════════════════════════════════════════════════════

def test_status_needs_job_id():
    r = tp.manage_pfs_transfer("status")
    assert "job_id" in r and "❌" in r


def test_status_not_found():
    r = tp.manage_pfs_transfer("status", job_id="xpfs-nope")
    assert "未找到" in r


def test_status_existing_shows_stage():
    from core.pfs_transfer import paths
    job = o.create_job_record(paths.build_plan(SRC, DST), open_id="ou1")
    r = tp.manage_pfs_transfer("status", job_id=job["job_id"])
    assert job["job_id"] in r and "NEW" in r
    assert "沉降" in r        # 段完成状态展示


# ══════════════════════════════════════════════════════════════════════════════
# plan（只读，不建 job、不起链）
# ══════════════════════════════════════════════════════════════════════════════

def test_plan_readonly_no_job_created(monkeypatch):
    created = []
    monkeypatch.setattr(o, "create_job_record",
                        lambda *a, **k: created.append(1) or {})
    monkeypatch.setattr(tp.threading, "Thread",
                        lambda *a, **k: pytest.fail("plan 不该起线程"))
    r = tp.manage_pfs_transfer("plan", source=SRC, dest=DST)
    assert "dry-run" in r
    assert "wuji-dc-bj" in r and "wuji-il-hz" in r     # 两 staging 桶入文
    assert created == []                                # 未建 job


def test_plan_shows_approval():
    r = tp.manage_pfs_transfer("plan", source=SRC, dest=DST)
    assert "审批" in r        # PFS 大小未知 → fail-safe 需审批


def test_plan_missing_args():
    r = tp.manage_pfs_transfer("plan", source=SRC)
    assert "❌" in r and "dest" in r


def test_plan_readonly_not_gated_by_admin(monkeypatch):
    """plan 是只读 dry-run，不经 admin 门（任何 open_id 都能预览）。"""
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    r = tp.manage_pfs_transfer("plan", source=SRC, dest=DST, open_id="ou_random_user")
    assert "dry-run" in r


# ══════════════════════════════════════════════════════════════════════════════
# apply
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "ou_admin")
    return "ou_admin"


def test_apply_non_admin_blocked_even_with_force(admin, monkeypatch):
    """MED-2 双路都堵：工具 apply 的 admin 门在 need/force 之前——非管理员 force=true 也拦，
    不建 job、不起线程。"""
    created, threaded = [], []
    monkeypatch.setattr(o, "create_job_record", lambda *a, **k: created.append(1) or {})
    monkeypatch.setattr(tp.threading, "Thread",
                        lambda *a, **k: threaded.append(1) or _SyncThread(*a, **k))
    r = tp.manage_pfs_transfer("apply", source=SRC, dest=DST, force=True, open_id="ou_not_admin")
    assert "需管理员" in r
    assert created == [] and threaded == []


def test_apply_admin_no_force_needs_confirm(admin):
    """管理员但未 force → 提示带 force 执行（不起链）。"""
    r = tp.manage_pfs_transfer("apply", source=SRC, dest=DST, force=False, open_id="ou_admin")
    assert "force" in r and "需管理员确认" in r


def test_apply_admin_force_starts_chain(admin, monkeypatch):
    ran = []
    monkeypatch.setattr(o, "run_to_completion",
                        lambda job, **k: ran.append(job["job_id"]) or job)
    monkeypatch.setattr(tp.threading, "Thread", _SyncThread)
    r = tp.manage_pfs_transfer("apply", source=SRC, dest=DST, force=True, open_id="ou_admin")
    assert "已提交" in r
    assert len(ran) == 1 and ran[0].startswith("xpfs-")


def test_apply_bad_path_returns_error(admin):
    r = tp.manage_pfs_transfer("apply", source="oss://x/y/", dest=DST, force=True, open_id="ou_admin")
    assert "❌" in r and ("路径" in r or "对象存储" in r)


# ══════════════════════════════════════════════════════════════════════════════
# 杂项
# ══════════════════════════════════════════════════════════════════════════════

def test_unknown_action():
    r = tp.manage_pfs_transfer("frobnicate")
    assert "未知 action" in r


def test_tool_registered_in_groups():
    from tools import TOOL_GROUPS, ALL_TOOLS
    assert "manage_pfs_transfer" in TOOL_GROUPS["pfs_transfer"]
    assert any(t.name == "manage_pfs_transfer" for t in ALL_TOOLS)
