"""core/capacity_bitable.write_scan（upsert）测试。

用内存假表 mock _list_records/_create_one/_update_one/_delete_one，断言：
  - 空表 → 全部创建；已有行 → 更新而非新建
  - 本次未出现的旧行删除；历史重复行去重
  - 数据结构 / 数据时长（从名字解析）/ 关联 写入正确
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
def fake(monkeypatch, cfg):
    """内存假表 + 调用记录。tables[table_id] = [{record_id, fields}]。"""
    from core import capacity_bitable as cb
    tables = {"tblSnap": [], "tblVendor": [], "tblBatch": []}
    calls = {"create": [], "update": [], "delete": []}

    monkeypatch.setattr(cb, "_tenant_token", lambda: "tok")
    monkeypatch.setattr(cb, "_list_records", lambda h, t: [dict(r) for r in tables.get(t, [])])
    # 默认两表都已建「数据类型」列（gate 放行，便于断言写入）
    monkeypatch.setattr(cb, "_field_names", lambda h, t: {"数据类型"})

    def _create(h, t, fields):
        rid = f"{t}-{len(tables[t])}"
        tables[t].append({"record_id": rid, "fields": dict(fields)})
        calls["create"].append((t, fields))
        return rid

    def _update(h, t, rid, fields):
        calls["update"].append((t, rid, fields))
        return True

    def _delete(h, t, rid):
        calls["delete"].append((t, rid))

    monkeypatch.setattr(cb, "_create_one", _create)
    monkeypatch.setattr(cb, "_update_one", _update)
    monkeypatch.setattr(cb, "_delete_one", _delete)
    return tables, calls


def _rows():
    return [
        {"云厂商": "OSS", "Bucket": "bk1", "父目录": "tp", "厂家": "lightwheel",
         "total_bytes": 2 * 1024 ** 3, "total_count": 5, "struct": "itw",
         "dtype": "lerobot v3.0", "delta_bytes": None,
         "batches": [("6-2-504h", 2 * 1024 ** 3, 5, "itw", "lerobot v3.0")]},
        {"云厂商": "TOS", "Bucket": "bk2", "父目录": "tp", "厂家": "egodex",
         "total_bytes": 3 * 1024 ** 3, "total_count": 7, "struct": "ego",
         "dtype": "", "delta_bytes": 1 * 1024 ** 3,
         "batches": [("part1", 3 * 1024 ** 3, 7, "ego", "")]},
    ]


def _created(calls, table):
    return [f for t, f in calls["create"] if t == table]


def test_upsert_creates_when_empty(fake):
    from core.capacity_bitable import write_scan
    tables, calls = fake
    assert write_scan(_rows(), "2026-06-05 巡检", "remark") is True

    assert len(_created(calls, "tblSnap")) == 1
    vens = _created(calls, "tblVendor")
    bats = _created(calls, "tblBatch")
    assert len(vens) == 2 and len(bats) == 2

    lw = next(f for f in vens if f["厂家"] == "lightwheel")
    assert lw["数据结构"] == "itw" and lw["数据时长"] == 504        # 504h 解析
    assert lw["数据类型"] == "lerobot v3.0"                        # 新列写入
    assert lw["关联巡检快照"] == ["tblSnap-0"]
    eg = next(f for f in vens if f["厂家"] == "egodex")
    assert "数据时长" not in eg                                    # part1 无 h → 留空
    assert eg["数据类型"] == ""                                    # 未识别留空（列存在仍写空串）
    assert eg["较上次GB"] == 1.0

    b_lw = next(f for f in bats if f["批次"] == "6-2-504h")
    assert b_lw["数据结构"] == "itw" and b_lw["数据时长"] == 504
    assert b_lw["数据类型"] == "lerobot v3.0"
    assert b_lw["关联厂家总量"] == ["tblVendor-0"]


def test_dtype_skipped_when_column_absent(fake, monkeypatch):
    """表里没有「数据类型」列时，gate 应让该字段不写入（避免整条失败）。"""
    from core import capacity_bitable as cb
    monkeypatch.setattr(cb, "_field_names", lambda h, t: set())   # 两表都无该列
    from core.capacity_bitable import write_scan
    _tables, calls = fake
    write_scan(_rows(), "id", "r")
    for _t, f in calls["create"]:
        assert "数据类型" not in f


def test_upsert_updates_existing(fake):
    from core.capacity_bitable import write_scan
    tables, calls = fake
    tables["tblVendor"].append({"record_id": "V-OLD",
                                "fields": {"云厂商": "OSS", "Bucket": "bk1", "厂家": "lightwheel"}})
    tables["tblBatch"].append({"record_id": "B-OLD",
                               "fields": {"云厂商": "OSS", "厂家": "lightwheel", "批次": "6-2-504h"}})

    write_scan(_rows(), "id", "r")
    upd_tables = {t for t, _r, _f in calls["update"]}
    assert ("tblVendor", "V-OLD") in [(t, r) for t, r, _f in calls["update"]]
    assert ("tblBatch", "B-OLD") in [(t, r) for t, r, _f in calls["update"]]
    # lightwheel 走更新，不再创建；egodex 是新的 → 创建
    assert all(f["厂家"] != "lightwheel" for f in _created(calls, "tblVendor"))
    assert any(f["厂家"] == "egodex" for f in _created(calls, "tblVendor"))


def test_upsert_deletes_vanished(fake):
    from core.capacity_bitable import write_scan
    tables, calls = fake
    tables["tblVendor"].append({"record_id": "V-GONE",
                                "fields": {"云厂商": "OSS", "Bucket": "bk1", "厂家": "OLDVENDOR"}})
    tables["tblBatch"].append({"record_id": "B-GONE",
                               "fields": {"云厂商": "OSS", "厂家": "OLDVENDOR", "批次": "x"}})
    write_scan(_rows(), "id", "r")
    deleted = {rid for _t, rid in calls["delete"]}
    assert "V-GONE" in deleted and "B-GONE" in deleted


def test_upsert_dedupes_history(fake):
    """同键多行（旧版追加遗留）→ 留一行更新，其余删除。"""
    from core.capacity_bitable import write_scan
    tables, calls = fake
    for rid in ("V1", "V2", "V3"):
        tables["tblVendor"].append({"record_id": rid,
                                    "fields": {"云厂商": "OSS", "Bucket": "bk1", "厂家": "lightwheel"}})
    write_scan(_rows(), "id", "r")
    deleted = {rid for _t, rid in calls["delete"]}
    assert {"V2", "V3"}.issubset(deleted)        # 保留 V1 更新，删 V2/V3
    assert "V1" not in deleted


def test_hours_summed_at_vendor(fake):
    from core.capacity_bitable import write_scan
    _tables, calls = fake
    rows = [{"云厂商": "OSS", "Bucket": "b", "父目录": "tp", "厂家": "lingsheng",
             "total_bytes": 10, "total_count": 2, "struct": "itw", "dtype": "", "delta_bytes": None,
             "batches": [("wuji_itw_100h_a", 5, 1, "itw", ""), ("wuji_itw_100h_b", 5, 1, "itw", "")]}]
    write_scan(rows, "id", "r")
    ven = _created(calls, "tblVendor")[0]
    assert ven["数据时长"] == 200                 # 100 + 100


def test_struct_falls_back_to_dtype_when_no_modality(fake):
    """读不出模态(空 struct)时，数据结构列回退用数据类型(hdf5/mcap)。"""
    from core.capacity_bitable import write_scan
    _tables, calls = fake
    rows = [{"云厂商": "TOS", "Bucket": "b", "父目录": "tp", "厂家": "lightwheel",
             "total_bytes": 10, "total_count": 2, "struct": "", "dtype": "mcap",
             "delta_bytes": None,
             "batches": [("6-2-504h", 10, 2, "", "mcap")]}]
    write_scan(rows, "id", "r")
    ven = _created(calls, "tblVendor")[0]
    bat = _created(calls, "tblBatch")[0]
    assert ven["数据结构"] == "mcap"        # 模态空 → 回退数据类型
    assert bat["数据结构"] == "mcap"


def test_parse_hours_decimal_and_nested_path():
    """时长解析支持小数,且能从多层批次路径里取出。"""
    from core.capacity_bitable import _parse_hours
    assert _parse_hours("6-2-504h") == 504
    assert _parse_hours("wuji_itw_100h_202605271016") == 100
    assert _parse_hours("wuji/26_0430/wuji_home_scene_19.75h") == 19.75
    assert _parse_hours("wuji/26_0505/wuji_home_scene_85.11h") == 85.11
    assert _parse_hours("no_duration_here") is None


def test_not_configured(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CAPACITY_BITABLE_APP_TOKEN", "")
    from core.capacity_bitable import write_scan
    assert write_scan(_rows(), "x", "y") is False
