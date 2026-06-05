"""core/capacity_bitable.write_scan 测试。

mock 飞书 API 调用层（_tenant_token / _create_one / _batch_create），
断言三级关联写入的 payload 正确：
  - 快照先建、拿到 record_id
  - 厂家总量带 关联巡检快照=[snap_id]，首次不写 较上次GB，有上次则写
  - 批次明细带 关联厂家总量=[对应厂家 record_id]
"""
import pytest


@pytest.fixture
def cfg(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_APP_TOKEN", "app1")
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_TABLE_SNAPSHOT", "tblSnap")
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_TABLE_VENDOR", "tblVendor")
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_TABLE_BATCH", "tblBatch")


@pytest.fixture
def capture(monkeypatch):
    """拦截写 API，记录每张表收到的 fields。"""
    calls = {"snap": [], "vendor": [], "batch": []}
    from core import capacity_bitable as cb

    monkeypatch.setattr(cb, "_tenant_token", lambda: "tok")

    def fake_create_one(headers, table_id, fields):
        calls["snap"].append(fields)
        return "snap1"

    def fake_batch_create(headers, table_id, records):
        if table_id == "tblVendor":
            calls["vendor"].extend(records)
            return [f"v{i}" for i in range(len(records))]
        calls["batch"].extend(records)
        return [f"b{i}" for i in range(len(records))]

    monkeypatch.setattr(cb, "_create_one", fake_create_one)
    monkeypatch.setattr(cb, "_batch_create", fake_batch_create)
    return calls


def _rows():
    return [
        {"云厂商": "OSS", "Bucket": "bk1", "父目录": "third-party-data", "厂家": "aether",
         "total_bytes": 2 * 1024 ** 3, "total_count": 5, "delta_bytes": None,
         "batches": [("batch-a", 1 * 1024 ** 3, 2), ("batch-b", 1 * 1024 ** 3, 3)]},
        {"云厂商": "TOS", "Bucket": "bk2", "父目录": "third-party-data", "厂家": "egodex",
         "total_bytes": 3 * 1024 ** 3, "total_count": 7, "delta_bytes": 1 * 1024 ** 3,
         "batches": [("b1", 3 * 1024 ** 3, 7)]},
    ]


def test_write_scan_three_level_links(cfg, capture):
    from core.capacity_bitable import write_scan
    ok = write_scan(_rows(), "2026-06-05 10:41 巡检", "合计 5 GB")
    assert ok is True

    # ① 快照一条，含可读标识
    assert len(capture["snap"]) == 1
    assert capture["snap"][0]["可读标识"] == "2026-06-05 10:41 巡检"

    # ② 厂家总量两条，都关联快照；首次不写较上次GB，有上次则写
    assert len(capture["vendor"]) == 2
    assert all(v["关联巡检快照"] == ["snap1"] for v in capture["vendor"])
    aether = next(v for v in capture["vendor"] if v["厂家"] == "aether")
    egodex = next(v for v in capture["vendor"] if v["厂家"] == "egodex")
    assert aether["大小GB"] == 2.0 and aether["对象数"] == 5
    assert "较上次GB" not in aether                 # 首次留空
    assert egodex["较上次GB"] == 1.0                # 有上次 → +1 GB

    # ③ 批次明细共 3 条，各自关联对应厂家行（aether=v0, egodex=v1）
    assert len(capture["batch"]) == 3
    a_batches = [b for b in capture["batch"] if b["关联厂家总量"] == ["v0"]]
    e_batches = [b for b in capture["batch"] if b["关联厂家总量"] == ["v1"]]
    assert len(a_batches) == 2 and len(e_batches) == 1
    assert {b["批次"] for b in a_batches} == {"batch-a", "batch-b"}


def test_write_scan_not_configured(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_APP_TOKEN", "")
    from core.capacity_bitable import write_scan
    assert write_scan(_rows(), "x", "y") is False


def test_write_scan_snapshot_fail_aborts(cfg, monkeypatch):
    from core import capacity_bitable as cb
    monkeypatch.setattr(cb, "_tenant_token", lambda: "tok")
    monkeypatch.setattr(cb, "_create_one", lambda *a, **k: "")   # 快照建失败
    called = {"batch": 0}
    monkeypatch.setattr(cb, "_batch_create",
                        lambda *a, **k: called.__setitem__("batch", called["batch"] + 1) or [])
    assert cb.write_scan(_rows(), "x", "y") is False
    assert called["batch"] == 0                  # 快照失败应直接返回，不写子表
