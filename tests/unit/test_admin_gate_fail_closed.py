"""管理员门禁 fail-closed + 审批记录 TTL 回归（安全整改批次 1 的顺手改动）。

原写法散落各处：`if open_id != settings.ADMIN_FEISHU_OPEN_ID: 拒绝`。
`ADMIN_FEISHU_OPEN_ID` 未配置时它是空串，而 open_id 也可能是空串（老式卡片回调取不到操作人、
schema 2.0 路径的 `or ""` 兜底、Agent 工具的默认参数）→ `"" != ""` 为 False →
**所有管理员门禁全部放行**。隔离热备机、新部署、`.env` 漏填都会踩到这个组合。

覆盖三处：
  · `core.feishu_bot.actions._is_admin`（卡片回调侧的统一判定）
  · `tools/pfs_transfer/pfs_transfer.py` 的 `apply`
  · `tools/temp_ak_issuance/manage_temp_ak.py` 的 `revoke`
外加 `core.ram_approval._save_instance_record` 的 90 天 TTL（原来无 `ex=` → 建号失败记录里的
员工 PII 在 Redis 永不过期）。

每条拒绝用例都装哨兵证明危险动作（下发 / 起链 / 吊销凭证）真的没执行。
"""
import pytest

# 收集期导入（别挪进用例）：`core.ram_approval` 是 `from utils.redis_client import get_redis`，
# 导入即绑定。首次导入若发生在某个用例的 fakeredis fixture 之后，就会永久绑定到那一个用例的
# 假 Redis 上 —— 后续用例写进去的东西自己读不到，断言假空。
from core import ram_approval as _ra  # noqa: F401

ADMIN = "ou_admin_real"
OTHER = "ou_random_colleague"


# ── 1. actions._is_admin ────────────────────────────────────────────────────

@pytest.mark.parametrize("open_id,expected", [
    (ADMIN, True),
    (OTHER, False),
    ("", False),
    (None, False),
])
def test_is_admin_with_admin_configured(monkeypatch, open_id, expected):
    """ADMIN 有值时与旧的 `open_id == ADMIN` 语义等价（真管理员放行、其余拒绝）。"""
    from core.feishu_bot import actions

    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)
    assert actions._is_admin(open_id) is expected


@pytest.mark.parametrize("open_id", [ADMIN, OTHER, "", None])
def test_is_admin_rejects_everyone_when_admin_unconfigured(monkeypatch, open_id):
    """ADMIN 为空时**一律拒绝** —— 这正是旧代码 `"" != ""` 全放行的那个洞。"""
    from core.feishu_bot import actions

    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)
    assert actions._is_admin(open_id) is False


def test_is_admin_rejects_when_admin_setting_is_none(monkeypatch):
    from core.feishu_bot import actions

    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", None, raising=False)
    assert actions._is_admin("") is False
    assert actions._is_admin(ADMIN) is False


def test_is_admin_is_exact_match_not_prefix(monkeypatch):
    """精确相等，不是包含/前缀（`ou_admin_real_evil` 不能冒充）。"""
    from core.feishu_bot import actions

    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)
    assert actions._is_admin(ADMIN + "_evil") is False
    assert actions._is_admin(ADMIN[:-1]) is False
    assert actions._is_admin(" " + ADMIN) is False


def test_oss_perm_apply_handler_blocked_when_admin_unconfigured(monkeypatch):
    """端到端一条：ADMIN 未配置 → 权限下发 handler 拒绝，且下发线程一次没起。"""
    from core.feishu_bot import actions
    from core.oss_perm import permsync

    monkeypatch.setattr(actions.settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)

    hits = []
    monkeypatch.setattr(permsync, "load_members", lambda *a, **k: hits.append("load") or ([], []))
    monkeypatch.setattr(permsync, "apply_all", lambda *a, **k: hits.append("apply"))

    resp = actions._h_approve_oss_perm({"level": "bucket"}, "", "chat", {})

    assert resp["toast"]["type"] == "error"
    assert "管理员" in resp["toast"]["content"]
    assert hits == [], "拒绝之外还必须证明 OSS 权限下发没被执行"


# ── 2. manage_pfs_transfer apply ────────────────────────────────────────────

class _FakeStaging:
    def __init__(self, scheme, bucket, prefix):
        self.scheme, self.bucket, self.base_prefix = scheme, bucket, prefix


class _FakePlan:
    direction = "vepfs->cpfs"
    src_staging = _FakeStaging("tos", "data-tran", "xpfs/")
    dst_staging = _FakeStaging("oss", "wuji-data-tran", "xpfs/")

    def summary(self):
        return "vepfs://fs-a/d/ → cpfs://fs-b/d/"


@pytest.fixture
def pfs(monkeypatch):
    from core.pfs_transfer import paths, orchestrator as o

    hits = {"jobs": [], "runs": []}

    monkeypatch.setattr(paths, "build_plan", lambda src, dst: _FakePlan())
    monkeypatch.setattr(o, "estimate_source", lambda plan: (0, True))
    monkeypatch.setattr(o, "needs_approval", lambda size, known: False)
    monkeypatch.setattr(o, "create_job_record",
                        lambda plan, **k: hits["jobs"].append(k) or {"job_id": "xpfs-test"})
    monkeypatch.setattr(o, "run_to_completion", lambda job: hits["runs"].append(job))
    return hits


@pytest.mark.parametrize("open_id", ["", OTHER, ADMIN])
def test_pfs_apply_rejected_when_admin_unconfigured(monkeypatch, pfs, open_id):
    """ADMIN 未配置 → 谁都不能 apply，链子一段都不起。"""
    from tools.pfs_transfer.pfs_transfer import manage_pfs_transfer
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)

    out = manage_pfs_transfer(action="apply", source="vepfs://fs-a/d/",
                              dest="cpfs://fs-b/d/", open_id=open_id)

    assert "需管理员" in out
    assert pfs["jobs"] == [] and pfs["runs"] == [], "拒绝之外还必须证明没建 job、没起搬运链"


def test_pfs_apply_force_does_not_bypass_admin_gate(monkeypatch, pfs):
    """`force=true` 是越过体量阈值的逃生口，不是越过管理员门的。"""
    from tools.pfs_transfer.pfs_transfer import manage_pfs_transfer
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)

    out = manage_pfs_transfer(action="apply", source="vepfs://fs-a/d/",
                              dest="cpfs://fs-b/d/", force=True, open_id=OTHER)

    assert "需管理员" in out
    assert pfs["jobs"] == []


def test_pfs_apply_non_admin_rejected_with_admin_configured(monkeypatch, pfs):
    from tools.pfs_transfer.pfs_transfer import manage_pfs_transfer
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)

    out = manage_pfs_transfer(action="apply", source="vepfs://fs-a/d/",
                              dest="cpfs://fs-b/d/", open_id=OTHER)

    assert "需管理员" in out
    assert pfs["jobs"] == []


def test_pfs_apply_allows_real_admin(monkeypatch, pfs):
    """阳性对照：真管理员照常起链（证明门不是把功能焊死了）。"""
    from tools.pfs_transfer.pfs_transfer import manage_pfs_transfer
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)

    out = manage_pfs_transfer(action="apply", source="vepfs://fs-a/d/",
                              dest="cpfs://fs-b/d/", open_id=ADMIN)

    assert "已提交" in out
    assert len(pfs["jobs"]) == 1


def test_pfs_plan_stays_readable_without_admin(monkeypatch, pfs):
    """只读的 `plan` 不受管理员门影响（不该顺手把 dry-run 也堵死）。"""
    from tools.pfs_transfer.pfs_transfer import manage_pfs_transfer
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)

    out = manage_pfs_transfer(action="plan", source="vepfs://fs-a/d/", dest="cpfs://fs-b/d/")

    assert "计划（dry-run）" in out
    assert pfs["jobs"] == []


# ── 3. manage_temp_ak revoke ────────────────────────────────────────────────

@pytest.fixture
def tak(monkeypatch):
    from core.temp_ak_issuance import orchestrator, cleanup

    hits = {"revokes": []}
    grant = {"grant_id": "tak-abc123", "account": "", "bucket": "b", "status": "ISSUED"}

    monkeypatch.setattr(orchestrator, "get_grant", lambda gid: dict(grant) if gid == grant["grant_id"] else None)
    monkeypatch.setattr(cleanup, "revoke_grant", lambda g: hits["revokes"].append(g) or True)
    return hits


@pytest.mark.parametrize("open_id", ["", OTHER, ADMIN])
def test_temp_ak_revoke_rejected_when_admin_unconfigured(monkeypatch, tak, open_id):
    """ADMIN 未配置 → 「revoke tak-…」谁都用不了，外部方凭证不会被随手吊销。"""
    from tools.temp_ak_issuance.manage_temp_ak import manage_temp_ak
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", "", raising=False)

    out = manage_temp_ak(action="revoke", grant_id="tak-abc123", open_id=open_id)

    assert "需管理员" in out
    assert tak["revokes"] == [], "拒绝之外还必须证明没真去删 AK/policy/user"


def test_temp_ak_revoke_non_admin_rejected_with_admin_configured(monkeypatch, tak):
    from tools.temp_ak_issuance.manage_temp_ak import manage_temp_ak
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)

    out = manage_temp_ak(action="revoke", grant_id="tak-abc123", open_id=OTHER)

    assert "需管理员" in out
    assert tak["revokes"] == []


def test_temp_ak_revoke_allows_real_admin(monkeypatch, tak):
    """阳性对照。"""
    from tools.temp_ak_issuance.manage_temp_ak import manage_temp_ak
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)

    out = manage_temp_ak(action="revoke", grant_id="tak-abc123", open_id=ADMIN)

    assert "已吊销" in out
    assert len(tak["revokes"]) == 1


def test_temp_ak_admin_gate_runs_before_grant_lookup(monkeypatch, tak):
    """门在查 grant 之前：非管理员连「这个凭证 ID 存不存在」都探不出来。"""
    from tools.temp_ak_issuance.manage_temp_ak import manage_temp_ak
    from config.settings import settings

    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN, raising=False)

    missing = manage_temp_ak(action="revoke", grant_id="tak-does-not-exist", open_id=OTHER)
    existing = manage_temp_ak(action="revoke", grant_id="tak-abc123", open_id=OTHER)

    assert missing == existing, "拒绝文案不得因凭证是否存在而不同（信息泄漏）"
    assert tak["revokes"] == []


# ── 4. ram_approval 审批记录 TTL ────────────────────────────────────────────

def test_instance_record_written_with_ttl(fake_redis):
    """建号审批记录必须带 90 天 TTL（原来无 `ex=` → 员工 PII 永久沉淀在 Redis）。"""
    ra = _ra

    ra._save_instance_record("INST-TTL-1", {"result_status": "failed", "login_name": "zhangsan"})

    key = ra._instance_record_key("INST-TTL-1")
    ttl = fake_redis.ttl(key)
    assert ttl > 0, "记录没有 TTL —— 会永不过期"
    assert ttl <= ra.INSTANCE_RECORD_TTL_SECONDS
    assert ttl > ra.INSTANCE_RECORD_TTL_SECONDS - 60
    assert ra.INSTANCE_RECORD_TTL_SECONDS == 90 * 86400


def test_instance_record_ttl_refreshed_on_every_write(fake_redis):
    """每次写都刷新 TTL —— 而不是从第一次写开始倒数（幂等续做期间不能中途蒸发）。"""
    ra = _ra

    ra._save_instance_record("INST-TTL-2", {"result_status": "processing"})
    key = ra._instance_record_key("INST-TTL-2")

    fake_redis.expire(key, 100)          # 模拟「已经躺了 89 天」
    assert fake_redis.ttl(key) <= 100

    ra._save_instance_record("INST-TTL-2", {"result_status": "success"})

    assert fake_redis.ttl(key) > ra.INSTANCE_RECORD_TTL_SECONDS - 60


def test_instance_record_merge_semantics_unchanged_by_ttl(fake_redis):
    """加 TTL 不改合并语义：旧字段保留、None 被滤掉、created_at_ms 只写一次。"""
    import json
    ra = _ra

    ra._save_instance_record("INST-TTL-3", {"login_name": "zhangsan", "result_status": "processing"})
    first = json.loads(fake_redis.get(ra._instance_record_key("INST-TTL-3")))

    ra._save_instance_record("INST-TTL-3", {"result_status": "failed", "error": None})
    second = json.loads(fake_redis.get(ra._instance_record_key("INST-TTL-3")))

    assert second["login_name"] == "zhangsan"
    assert second["result_status"] == "failed"
    assert "error" not in second
    assert second["created_at_ms"] == first["created_at_ms"]


def test_instance_record_noop_without_code(fake_redis):
    ra = _ra

    ra._save_instance_record("", {"result_status": "failed"})
    assert fake_redis.keys("ram_approval:instance:*") == []
