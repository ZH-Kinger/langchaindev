"""core/dataset_dashboard：uri 解析 / 存在性→状态 / 仅填空白 / 人工列绝不碰 / cpfs 跳过。"""
import pytest


@pytest.fixture
def D(monkeypatch):
    from core import dataset_dashboard as D
    # 默认探测桩：oss 存在且是 lerobot 布局；tos 存在
    monkeypatch.setattr(D, "_oss_probe", lambda b, p: (True, ["meta", "data"]))
    monkeypatch.setattr(D, "_tos_probe", lambda b, p: (True, []))
    return D


def test_parse_uri(D):
    assert D._parse_uri("oss://wuji-bangkok/vitra/guanqihe/x") == ("oss", "wuji-bangkok", "vitra/guanqihe/x")
    assert D._parse_uri("tos://b/") == ("tos", "b", "")
    assert D._parse_uri("oss://b") == ("oss", "b", "")
    assert D._parse_uri("不是uri") is None


def test_hours_from(D):
    assert D._hours_from("wuji/26_0430/home_504h") == "504h"
    assert D._hours_from("x/y_19.75h/z") == "19.75h"
    assert D._hours_from("no-duration-here") is None


def test_exists_marks_in_stock_and_fills_blanks(D):
    upd = D.compute_updates({"uri": "oss://wuji-bangkok/vitra/guanqihe/set_100h", "云": "", "厂商|来源": "",
                             "时长": "", "数据集类型": ""})
    assert upd["状态"] == "在库"
    assert upd["云"] == "OSS"
    assert upd["厂商|来源"] == "vitra"
    assert upd["时长"] == "100h"
    assert upd["数据集类型"] == "lerobot"   # subdirs 含 meta+data


def test_gone_marks_disappeared(D, monkeypatch):
    monkeypatch.setattr(D, "_oss_probe", lambda b, p: (False, []))
    upd = D.compute_updates({"uri": "oss://b/vendorx/ds", "云": "OSS", "厂商|来源": "vendorx"})
    assert upd["状态"] == "已消失"


def test_fill_only_blanks(D):
    # 云/厂商/时长/数据集类型 已有值 → 不在 updates 里；状态每次都刷新
    upd = D.compute_updates({"uri": "oss://b/vitra/ds_50h", "云": "OSS", "厂商|来源": "vitra",
                             "时长": "50h", "数据集类型": "data"})
    assert upd == {"状态": "在库"}


def test_manual_columns_never_written(D):
    fields = {"uri": "oss://b/vitra/ds", "云": "", "剖析结论|反馈": "人工写的",
              "坏帧率": "0.1", "负责人": "张三", "交付状态": "已验收", "备注": "别动我"}
    upd = D.compute_updates(fields)
    for manual in ("剖析结论|反馈", "坏帧率", "负责人", "交付状态", "备注", "mono偏移",
                   "跳变率", "一句话描述", "训练任务|路径", "数据源类型"):
        assert manual not in upd


def test_cpfs_skips_status(D):
    upd = D.compute_updates({"uri": "cpfs://fs-x/label/data", "云": "", "厂商|来源": ""})
    assert "状态" not in upd            # cpfs 不探测存在性
    assert upd["云"] == "CPFS" and upd["厂商|来源"] == "label"   # 仍从 uri 拆空白列


def test_creds_unavailable_skips_status(D, monkeypatch):
    monkeypatch.setattr(D, "_oss_probe", lambda b, p: (None, []))   # creds 不可用
    upd = D.compute_updates({"uri": "oss://b/vendory/ds", "云": "", "厂商|来源": ""})
    assert "状态" not in upd
    assert upd["云"] == "OSS"


def test_bad_uri_no_updates(D):
    assert D.compute_updates({"uri": "garbage"}) == {}
    assert D.compute_updates({"uri": ""}) == {}


def test_run_once_writes_only_script_cols(D, monkeypatch):
    rows = [
        {"record_id": "rec1", "fields": {"uri": "oss://b/vitra/ds_10h", "云": "", "厂商|来源": "",
                                          "时长": "", "数据集类型": "", "负责人": "李四"}},
        {"record_id": "rec2", "fields": {"uri": "cpfs://fs/x", "云": "CPFS", "厂商|来源": "x"}},  # 无更新→跳过
    ]
    monkeypatch.setattr(D, "_tenant_token", lambda: "tok")
    monkeypatch.setattr(D, "_list_records", lambda h: rows)
    written = {}
    monkeypatch.setattr(D, "_update_one", lambda h, rid, fields: written.__setitem__(rid, fields) or True)
    monkeypatch.setattr(D, "_is_configured", lambda: True)

    stats = D.run_once()
    assert stats["ok"] and stats["total"] == 2 and stats["written"] == 1
    assert "rec1" in written and "rec2" not in written
    # 写入 rec1 的字段只含脚本列，负责人（人工列）绝不在内
    assert "负责人" not in written["rec1"]
    assert written["rec1"]["状态"] == "在库" and written["rec1"]["云"] == "OSS"


def test_run_once_dry_run_no_write(D, monkeypatch):
    rows = [{"record_id": "r", "fields": {"uri": "oss://b/v/ds", "云": ""}}]
    monkeypatch.setattr(D, "_tenant_token", lambda: "tok")
    monkeypatch.setattr(D, "_list_records", lambda h: rows)
    monkeypatch.setattr(D, "_is_configured", lambda: True)
    boom = lambda *a, **k: pytest.fail("dry_run 不应写入")
    monkeypatch.setattr(D, "_update_one", boom)
    stats = D.run_once(dry_run=True)
    assert stats["ok"] and stats["written"] == 0 and stats["preview"][0]["uri"] == "oss://b/v/ds"
