"""discovery 单测：fs 清单解析 / 枚举 / cri 屏蔽 / 选项构造 / 解码。"""
import json

from core.cpfs_dataflow import discovery, engine_nas


def test_parse_fs_list(monkeypatch):
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_IDS",
                        "bmcpfs-a@cn-hangzhou, cpfs-b , bmcpfs-c@cn-shanghai")
    monkeypatch.setattr(discovery.settings, "CPFS_REGION", "cn-hangzhou")
    assert discovery.parse_fs_list() == [
        ("bmcpfs-a", "cn-hangzhou"), ("cpfs-b", "cn-hangzhou"), ("bmcpfs-c", "cn-shanghai")]


def test_parse_fs_list_falls_back_to_single(monkeypatch):
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_IDS", "")
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_ID", "bmcpfs-only")
    monkeypatch.setattr(discovery.settings, "CPFS_REGION", "cn-beijing")
    assert discovery.parse_fs_list() == [("bmcpfs-only", "cn-beijing")]


def test_split_source_storage():
    assert discovery._split_source_storage("oss://bk/prefix/x") == ("bk", "prefix/x")
    assert discovery._split_source_storage("oss://12345:bk/p") == ("bk", "p")
    assert discovery._split_source_storage("oss://bk") == ("bk", "")


def test_discover_excludes_cri_buckets(monkeypatch):
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_IDS", "bmcpfs-a@cn-hangzhou")
    flows = [
        {"data_flow_id": "df-1", "source_storage": "oss://wuji-bucket-hangzhou", "fs_path": "/cwr/"},
        {"data_flow_id": "df-2", "source_storage": "oss://cri-image-repo", "fs_path": "/img/"},
        {"data_flow_id": "df-3", "source_storage": "oss://CRI-Mixed", "fs_path": "/x/"},
    ]
    monkeypatch.setattr(engine_nas, "list_dataflows", lambda fs, region, **k: flows)
    opts = discovery.discover()
    buckets = {o["oss_bucket"] for o in opts}
    assert buckets == {"wuji-bucket-hangzhou"}        # cri / CRI 均被屏蔽
    assert opts[0]["data_flow_id"] == "df-1"
    assert "cn-hangzhou" in opts[0]["label"] and "bmcpfs-a" in opts[0]["label"]


def test_option_value_roundtrip(monkeypatch):
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_IDS", "bmcpfs-a@cn-hangzhou")
    monkeypatch.setattr(engine_nas, "list_dataflows",
                        lambda fs, region, **k: [{"data_flow_id": "df-1",
                                                  "source_storage": "oss://bk/wuji_il",
                                                  "fs_path": "/cwr/"}])
    opt = discovery.discover()[0]
    sel = discovery.decode_selection(opt["value"])
    assert sel["fs_id"] == "bmcpfs-a" and sel["region"] == "cn-hangzhou"
    assert sel["data_flow_id"] == "df-1" and sel["oss_bucket"] == "bk"
    assert sel["fs_path"] == "/cwr/"


def test_discover_region_scan(monkeypatch):
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_IDS", "")
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_ID", "")
    monkeypatch.setattr(discovery.settings, "CPFS_REGIONS", "cn-hangzhou,cn-shanghai")
    monkeypatch.setattr(engine_nas, "list_filesystems",
                        lambda region, **k: {"cn-hangzhou": ["bmcpfs-a"],
                                             "cn-shanghai": ["cpfs-b"]}.get(region, []))
    monkeypatch.setattr(engine_nas, "list_dataflows",
                        lambda fs, region, **k: [{"data_flow_id": f"df-{fs}",
                                                  "source_storage": f"oss://bk-{fs}",
                                                  "fs_path": "/p/"}])
    opts = discovery.discover()
    assert {o["fs_id"] for o in opts} == {"bmcpfs-a", "cpfs-b"}
    assert {o["region"] for o in opts} == {"cn-hangzhou", "cn-shanghai"}


def test_region_fs_bucket_helpers(monkeypatch):
    opts = [
        {"region": "cn-hangzhou", "fs_id": "bmcpfs-a", "edition": "computing", "oss_bucket": "bk1"},
        {"region": "cn-hangzhou", "fs_id": "bmcpfs-a", "edition": "computing", "oss_bucket": "bk2"},
    ]
    # 文件系统清单含北京(无绑定)——地区/文件系统下拉来自它
    fss = [
        {"region": "cn-hangzhou", "fs_id": "bmcpfs-a", "edition": "computing"},
        {"region": "cn-beijing", "fs_id": "bmcpfs-b", "edition": "computing"},
    ]
    monkeypatch.setattr(discovery, "get_options", lambda *a, **k: opts)
    monkeypatch.setattr(discovery, "get_filesystems", lambda *a, **k: fss)
    assert discovery.regions() == ["cn-beijing", "cn-hangzhou"]           # 北京也在(无绑定)
    assert discovery.filesystems_in("cn-beijing") == [{"fs_id": "bmcpfs-b", "edition": "computing"}]
    assert discovery.buckets_in("cn-hangzhou") == ["bk1", "bk2"]
    assert discovery.buckets_in("cn-beijing") == []                       # 无绑定→空桶


def test_get_options_uses_cache(monkeypatch):
    calls = {"n": 0}
    def fake_list(fs, region, **k):
        calls["n"] += 1
        return [{"data_flow_id": "df-1", "source_storage": "oss://bk", "fs_path": "/a/"}]
    monkeypatch.setattr(discovery.settings, "CPFS_FILE_SYSTEM_IDS", "bmcpfs-a@cn-hangzhou")
    monkeypatch.setattr(engine_nas, "list_dataflows", fake_list)
    discovery.refresh()                 # 填缓存
    n_after_refresh = calls["n"]
    discovery.get_options()             # 命中缓存，不再调用
    assert calls["n"] == n_after_refresh
