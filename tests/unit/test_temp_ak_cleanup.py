"""#57 临时 AK 发放 —— cleanup 方案 B 到期硬删 + sweep。

覆盖：
  · revoke_grant 方案 B 删序：停用 AK(Inactive) → 删 AK → 解绑 policy → 删 policy(清非默认版本) → 删 user
    （mock RAM client 断言调用顺序）。
  · STS mode 只翻 REVOKED、不调云、幂等（删已删吞异常）。
  · sweep_expired 只挑 ISSUED 且 expire<now、跳过 REVOKED/未到期。
"""
import time

import pytest

from core.temp_ak_issuance import cleanup, orchestrator as o
from core.oss_perm import permsync


# ── 假 RAM client：记录方法调用顺序 ─────────────────────────────────────────

class _AK:
    def __init__(self, akid):
        self.access_key_id = akid


class _Ver:
    def __init__(self, vid, default):
        self.version_id = vid
        self.is_default_version = default


class FakeRamClient:
    def __init__(self, ak_ids=("ak1",), versions=None, raise_on=None):
        self.calls = []
        self._ak_ids = ak_ids
        self._versions = versions if versions is not None else [
            _Ver("v1", True), _Ver("v2", False)]
        self._raise_on = raise_on or set()

    def _rec(self, name):
        self.calls.append(name)
        if name in self._raise_on:
            raise RuntimeError(f"boom {name}")

    def list_access_keys(self, req):
        self._rec("list_access_keys")
        class B:
            class body:
                class access_keys:
                    access_key = [_AK(a) for a in self._ak_ids]
        # 需要实例化的 body 结构
        obj = type("R", (), {})()
        obj.body = type("Body", (), {})()
        obj.body.access_keys = type("AKs", (), {})()
        obj.body.access_keys.access_key = [_AK(a) for a in self._ak_ids]
        return obj

    def update_access_key(self, req):
        self._rec("update_access_key")
        assert req.status == "Inactive"

    def delete_access_key(self, req):
        self._rec("delete_access_key")

    def detach_policy_from_user(self, req):
        self._rec("detach_policy_from_user")

    def list_policy_versions(self, req):
        self._rec("list_policy_versions")
        obj = type("R", (), {})()
        obj.body = type("Body", (), {})()
        obj.body.policy_versions = type("PVs", (), {})()
        obj.body.policy_versions.policy_version = self._versions
        return obj

    def delete_policy_version(self, req):
        self._rec("delete_policy_version")

    def delete_policy(self, req):
        self._rec("delete_policy")

    def delete_user(self, req):
        self._rec("delete_user")


def _ram_grant(**over):
    now = time.time()
    g = {
        "grant_id": "tak-ram1", "stage": o.STAGE_ISSUED, "mode": "ram",
        "bucket": "b", "user_name": "tempak-ext-abc",
        "policy_name": "temp-ak-auto-tempak-ext-abc", "ak_id": "ak1",
        "expire": now - 100, "not_before": now - 200,
        "prefix": "r/", "caps": ["read"],
    }
    g.update(over)
    return g


def _patch_client(monkeypatch, fake):
    monkeypatch.setattr(permsync, "make_ram_client", lambda: fake)


# ── revoke_grant 方案 B 删序 ──────────────────────────────────────────────────

def test_revoke_ram_delete_order(monkeypatch):
    """当前实现的删序（不含 delete_policy_version——见下方 xfail：源码 bug 使清版本被吞）。
    停用 AK → 删 AK → 解绑 policy → 列版本 → 删 policy → 删 user。"""
    fake = FakeRamClient(ak_ids=("ak1",))
    _patch_client(monkeypatch, fake)
    g = _ram_grant()
    ok = cleanup.revoke_grant(g)
    assert ok is True
    assert g["stage"] == o.STAGE_REVOKED
    # 相对顺序（前置依赖）正确
    idx = {name: fake.calls.index(name) for name in
           ("update_access_key", "delete_access_key", "detach_policy_from_user",
            "delete_policy", "delete_user")}
    assert idx["update_access_key"] < idx["delete_access_key"]      # 先停用后删
    assert idx["delete_access_key"] < idx["detach_policy_from_user"]
    assert idx["detach_policy_from_user"] < idx["delete_policy"]
    assert idx["delete_policy"] < idx["delete_user"]               # user 最后


def test_revoke_deletes_nondefault_policy_versions(monkeypatch):
    """删 policy 前先清非默认版本（否则 RAM DeletePolicy 会因残留版本失败、policy 泄漏）。
    回归 dev 修复的 cleanup.py bug：DeletePolicyVersionRequest 去掉了误传的 policy_type。"""
    captured = {}
    fake = FakeRamClient(ak_ids=("ak1",),
                         versions=[_Ver("v1", True), _Ver("v2", False)])
    # 包一层捕获 delete_policy_version 收到的 request，断言用对参数（有 version_id、无 policy_type）
    orig_del_ver = fake.delete_policy_version

    def spy_del_ver(req):
        captured["req"] = req
        return orig_del_ver(req)

    fake.delete_policy_version = spy_del_ver
    _patch_client(monkeypatch, fake)

    cleanup.revoke_grant(_ram_grant())
    # 非默认版本 v2 被删、且在删 policy 之前
    assert "delete_policy_version" in fake.calls
    assert fake.calls.index("delete_policy_version") < fake.calls.index("delete_policy")
    # 用对参数：带正确 version_id、不含 policy_type（源码 bug 的根因）
    req = captured["req"]
    assert req.version_id == "v2"
    assert not hasattr(req, "policy_type") or getattr(req, "policy_type", None) is None


def test_revoke_ram_multiple_aks(monkeypatch):
    fake = FakeRamClient(ak_ids=("ak1", "ak2"))
    _patch_client(monkeypatch, fake)
    cleanup.revoke_grant(_ram_grant())
    # 两把 AK 各停用+删除
    assert fake.calls.count("update_access_key") == 2
    assert fake.calls.count("delete_access_key") == 2


def test_revoke_ram_persists_revoked(monkeypatch):
    fake = FakeRamClient()
    _patch_client(monkeypatch, fake)
    g = _ram_grant()
    cleanup.revoke_grant(g)
    # _save 落库
    stored = o.get_grant(g["grant_id"])
    assert stored is not None
    assert stored["stage"] == o.STAGE_REVOKED
    assert "revoked_ts" in stored


# ── STS 只翻状态不调云 ────────────────────────────────────────────────────────

def test_revoke_sts_no_cloud_call(monkeypatch):
    monkeypatch.setattr(permsync, "make_ram_client",
                        lambda: pytest.fail("STS 吊销不该建 RAM 客户端/调云"))
    g = _ram_grant(mode="sts", policy_name="", ak_id="")
    ok = cleanup.revoke_grant(g)
    assert ok is True
    assert g["stage"] == o.STAGE_REVOKED
    assert "revoked_ts" in g


# ── 幂等 ──────────────────────────────────────────────────────────────────────

def test_revoke_already_revoked_noop(monkeypatch):
    monkeypatch.setattr(permsync, "make_ram_client",
                        lambda: pytest.fail("已 REVOKED 不该再调云"))
    g = _ram_grant(stage=o.STAGE_REVOKED)
    assert cleanup.revoke_grant(g) is True


def test_revoke_swallows_delete_errors(monkeypatch):
    """删 AK 抛异常（已删）→ 吞掉、仍推进到删 user、翻 REVOKED。"""
    fake = FakeRamClient(raise_on={"delete_access_key"})
    _patch_client(monkeypatch, fake)
    g = _ram_grant()
    ok = cleanup.revoke_grant(g)
    assert ok is True
    assert g["stage"] == o.STAGE_REVOKED
    assert "delete_user" in fake.calls


# ── sweep_expired ─────────────────────────────────────────────────────────────

def test_sweep_only_expired_issued(monkeypatch):
    now = time.time()
    # 落 4 个 grant：到期ISSUED / 未到期ISSUED / 已REVOKED到期 / 到期但NEW
    o._save(_ram_grant(grant_id="tak-exp", stage=o.STAGE_ISSUED, expire=now - 10))
    o._save(_ram_grant(grant_id="tak-future", stage=o.STAGE_ISSUED, expire=now + 9999))
    o._save(_ram_grant(grant_id="tak-rev", stage=o.STAGE_REVOKED, expire=now - 10))
    o._save(_ram_grant(grant_id="tak-new", stage=o.STAGE_NEW, expire=now - 10))

    revoked_calls = []
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda g, **k: (revoked_calls.append(g["grant_id"]) or True))
    swept = cleanup.sweep_expired(now=now)
    assert swept == ["tak-exp"]
    assert revoked_calls == ["tak-exp"]


def test_sweep_empty_when_nothing_expired(monkeypatch):
    now = time.time()
    o._save(_ram_grant(grant_id="tak-future2", stage=o.STAGE_ISSUED, expire=now + 9999))
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda g, **k: pytest.fail("无到期 grant 不该 revoke"))
    assert cleanup.sweep_expired(now=now) == []


def test_sweep_boundary_expire_equals_now_not_swept(monkeypatch):
    """expire == now 不算过期（源码 >= now 跳过）。"""
    now = time.time()
    o._save(_ram_grant(grant_id="tak-eq", stage=o.STAGE_ISSUED, expire=now))
    monkeypatch.setattr(cleanup, "revoke_grant",
                        lambda g, **k: pytest.fail("expire==now 不该被扫"))
    assert cleanup.sweep_expired(now=now) == []
