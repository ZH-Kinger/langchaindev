"""#46 B5：capacity_bitable 列举记录 code!=0 抛错，write_scan 放弃本次写入（不退化成 insert 造重复行）。

线上根因：`_list_records` 列举瞬时失败静默返 []→调用方当成"表里没记录"→upsert 退化成全 insert→重复行。
修：code!=0 抛 RuntimeError；write_scan 两处列举 try→失败 return False（不写）。
"""
import pytest


class _Resp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


# ── _list_records：code!=0 抛错 / code==0 正常 ─────────────────────────────────

def test_list_records_raises_on_nonzero_code(monkeypatch):
    from core import capacity_bitable as cb
    monkeypatch.setattr(cb.requests, "get",
                        lambda *a, **k: _Resp({"code": 1254043, "msg": "rate limit", "data": {}}))
    with pytest.raises(RuntimeError):
        cb._list_records({"Authorization": "Bearer x"}, "tblX")


def test_list_records_empty_when_code_zero(monkeypatch):
    from core import capacity_bitable as cb
    monkeypatch.setattr(cb.requests, "get",
                        lambda *a, **k: _Resp({"code": 0, "data": {"items": [], "has_more": False}}))
    assert cb._list_records({"Authorization": "Bearer x"}, "tblX") == []


def test_list_records_paginates_on_has_more(monkeypatch):
    from core import capacity_bitable as cb
    pages = [
        {"code": 0, "data": {"items": [{"record_id": "r1"}], "has_more": True, "page_token": "P2"}},
        {"code": 0, "data": {"items": [{"record_id": "r2"}], "has_more": False}},
    ]
    seq = iter(pages)
    monkeypatch.setattr(cb.requests, "get", lambda *a, **k: _Resp(next(seq)))
    out = cb._list_records({"Authorization": "Bearer x"}, "tblX")
    assert [r["record_id"] for r in out] == ["r1", "r2"]


# ── write_scan：列举失败 → 返 False + 零写 ─────────────────────────────────────

@pytest.fixture
def wired(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_APP_TOKEN", "app1")
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_TABLE_SNAPSHOT", "tblSnap")
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_TABLE_VENDOR", "tblVendor")
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_TABLE_BATCH", "tblBatch")

    from core import capacity_bitable as cb
    monkeypatch.setattr(cb, "_tenant_token", lambda: "tok")
    monkeypatch.setattr(cb, "_field_names", lambda h, t: {"数据类型"})
    calls = {"create": [], "update": [], "delete": []}
    monkeypatch.setattr(cb, "_create_one", lambda h, t, f: calls["create"].append((t, f)) or "rid")
    monkeypatch.setattr(cb, "_update_one", lambda h, t, r, f: calls["update"].append((t, r, f)) or True)
    monkeypatch.setattr(cb, "_delete_one", lambda h, t, r: calls["delete"].append((t, r)))
    return cb, calls


def _rows():
    return [{"云厂商": "OSS", "Bucket": "bk1", "父目录": "tp", "厂家": "lw",
             "total_bytes": 1, "total_count": 1, "struct": "itw", "dtype": "",
             "delta_bytes": None, "batches": [("6-2-504h", 1, 1, "itw", "")]}]


def test_write_scan_snapshot_list_error_aborts_no_write(wired, monkeypatch):
    """快照表列举抛错（第一处）→ write_scan 返 False，零 create/update。"""
    cb, calls = wired
    def _boom(h, t):
        raise RuntimeError("列举失败 code=1254043")
    monkeypatch.setattr(cb, "_list_records", _boom)

    assert cb.write_scan(_rows(), "id", "r") is False
    assert calls["create"] == [] and calls["update"] == []


def test_write_scan_vendor_list_error_aborts_no_write(wired, monkeypatch):
    """厂家/批次表列举抛错（第二处，快照阶段已过）→ 返 False，零 create/update。"""
    cb, calls = wired
    # 快照阶段列举返空（正常），厂家/批次阶段抛错
    def _list(h, t):
        if t == "tblSnap":
            return []
        raise RuntimeError("列举失败 code=1254043")
    monkeypatch.setattr(cb, "_list_records", _list)

    assert cb.write_scan(_rows(), "id", "r") is False
    # 快照可能新建了一行（snapshot 列举返空→create），但厂家/批次表零写
    assert all(t != "tblVendor" for t, _f in calls["create"])
    assert all(t != "tblBatch" for t, _f in calls["create"])
    assert calls["update"] == []


def test_write_scan_code_zero_empty_ok(wired, monkeypatch):
    """列举正常（空表）→ 走正常 upsert 路径，返 True 且有 create。"""
    cb, calls = wired
    monkeypatch.setattr(cb, "_list_records", lambda h, t: [])
    assert cb.write_scan(_rows(), "id", "r") is True
    assert any(t == "tblVendor" for t, _f in calls["create"])
