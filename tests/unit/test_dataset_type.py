"""tools/aliyun/oss.py 数据集类型识别（按目录结构，list-only）测试。

覆盖 _dataset_type_bits / _resolve_dataset_type / agg_dtype：
  LeRobot v3.0 / v2.0 / v2.1 / 仅 info.json；RLDS / rosbag / mcap / hdf5；普通文件留空。
并用 FakeBucket 跑通 compute_nested_sizes：批次的第 5 元素是识别出的数据类型。
"""
import pytest

from tools.aliyun.oss import _dataset_type_bits, _resolve_dataset_type, resolve_dtype, agg_dtype


def _dt(within: str) -> str:
    """单文件路径 → 数据类型串（便于断言）。"""
    return _resolve_dataset_type(_dataset_type_bits(within))


# ── LeRobot 各版本 ────────────────────────────────────────────────────────────

def test_lerobot_v3_tasks_parquet():
    assert _dt("meta/tasks.parquet") == "lerobot v3.0"


def test_lerobot_v3_episodes_dir():
    assert _dt("meta/episodes/chunk-000/file-000.parquet") == "lerobot v3.0"


def test_lerobot_v2_episodes_jsonl():
    assert _dt("meta/episodes.jsonl") == "lerobot v2.0"


def test_lerobot_v21_episodes_stats():
    assert _dt("meta/episodes_stats.jsonl") == "lerobot v2.1"


def test_lerobot_info_only_unknown_version():
    assert _dt("meta/info.json") == "lerobot"


def test_lerobot_nested_dataset_root():
    """数据集在批次更深一层也能识别（后缀匹配）。"""
    assert _dt("sub_dataset/meta/tasks.parquet") == "lerobot v3.0"


def test_lerobot_meta_plus_data_still_lerobot():
    """带 meta/ 标记的 LeRobot 数据集即使含裸 parquet 数据，仍判 lerobot（结构优先）。"""
    bits = (_dataset_type_bits("meta/tasks.parquet")
            | _dataset_type_bits("data/chunk-000/file-000.parquet"))
    assert _resolve_dataset_type(bits) == "lerobot v3.0"


# ── 其余常见格式 ──────────────────────────────────────────────────────────────

def test_rlds_tfrecord():
    assert _dt("1.0.0/dataset-train.tfrecord-00000-of-00010") == "rlds"


def test_rlds_dataset_info():
    assert _dt("my_ds/1.0.0/dataset_info.json") == "rlds"


def test_rosbag_bag():
    assert _dt("recordings/run1.bag") == "rosbag"


def test_rosbag_db3():
    assert _dt("ros2bag/run1_0.db3") == "rosbag"


def test_mcap():
    assert _dt("logs/session.mcap") == "mcap"


def test_hdf5():
    assert _dt("episodes/episode_0.hdf5") == "hdf5"
    assert _dt("episodes/episode_0.h5") == "hdf5"


def test_zarr_v3_json():
    assert _dt("2025-09-20-17-42-51-000000/images.front_1/zarr.json") == "zarr"


def test_zarr_v2_markers():
    assert _dt("arr/.zarray") == "zarr"
    assert _dt("grp/.zgroup") == "zarr"
    assert _dt("arr/.zattrs") == "zarr"


def test_zarr_zip_and_dir():
    assert _dt("replay_buffer.zarr.zip") == "zarr"
    assert _dt("pusht_replay.zarr/data/action/0.0") == "zarr"


def test_rlds_features_json():
    assert _dt("my_ds/1.0.0/features.json") == "rlds"


def test_raw_capture_camera_params():
    assert _dt("uuid1/camera_params/head_param.json") == "raw-capture"


def test_raw_capture_kalibr():
    assert _dt("uuid1/kalibr_parameters.yaml") == "raw-capture"


def test_raw_capture_plain_media_empty():
    """裸视频/音频本身不算（避免误判普通视频目录）。"""
    assert _dt("uuid1/rgb_head.mp4") == ""
    assert _dt("uuid1/mic.wav") == ""


def test_bare_parquet():
    assert _dt("part-00000.parquet") == "parquet"


def test_numpy():
    assert _dt("arrays/states.npy") == "numpy"
    assert _dt("arrays/batch.npz") == "numpy"


def test_pointcloud():
    assert _dt("scans/0001.pcd") == "pointcloud"
    assert _dt("mesh/scene.ply") == "pointcloud"


def test_plain_files_empty_at_file_level():
    """单文件级解析：无标记仍返回空（'other' 只在批次/聚合级兜底）。"""
    assert _dt("readme.txt") == ""
    assert _dt("images/0001.jpg") == ""
    assert _dt("notes.md") == ""


def test_resolve_dtype_other_fallback():
    """批次级 resolve_dtype：识别到→具体类型；没识别到→other。"""
    assert resolve_dtype(_dataset_type_bits("readme.txt")) == "other"
    assert resolve_dtype(_dataset_type_bits("images/0001.jpg")) == "other"
    assert resolve_dtype(_dataset_type_bits("meta/tasks.parquet")) == "lerobot v3.0"


def test_parse_lerobot_info():
    import json
    from tools.aliyun.oss import parse_lerobot_info
    hours, ver = parse_lerobot_info(json.dumps(
        {"codebase_version": "v2.1", "fps": 30, "total_frames": 1080000}))
    assert ver == "lerobot v2.1"
    assert hours == 10.0                         # 1080000/30/3600
    assert parse_lerobot_info("not json") == (None, None)
    assert parse_lerobot_info(json.dumps({})) == (None, None)         # 无字段
    assert parse_lerobot_info(json.dumps({"codebase_version": "v3.0"})) == (None, "lerobot v3.0")  # 有版本无帧数


def test_agg_dtype_other_semantics():
    assert agg_dtype(["other", "other"]) == "other"
    assert agg_dtype(["lerobot v3.0", "other"]) == "lerobot v3.0"   # 有具体类型则丢掉 other
    assert agg_dtype(["", ""]) == ""


# ── 优先级 + 聚合 ─────────────────────────────────────────────────────────────

def test_lerobot_wins_over_extension():
    """同时见到 lerobot meta 与 parquet 数据 → 仍判 lerobot（结构优先）。"""
    bits = _dataset_type_bits("meta/info.json") | _dataset_type_bits("meta/tasks.parquet")
    assert _resolve_dataset_type(bits) == "lerobot v3.0"


def test_agg_dtype_dedup_join():
    assert agg_dtype(["lerobot v3.0", "lerobot v3.0", "rlds", ""]) == "lerobot v3.0/rlds"
    assert agg_dtype(["", ""]) == ""


# ── 端到端：compute_nested_sizes 批次带出 dtype ───────────────────────────────

class _FakeObj:
    def __init__(self, key, size):
        self.key, self.size = key, size


class _FakeResult:
    def __init__(self, prefix_list=None, object_list=None):
        self.prefix_list = prefix_list or []
        self.object_list = object_list or []
        self.is_truncated = False
        self.next_continuation_token = ""


class _FakeBucket:
    endpoint = "https://oss-cn-hangzhou.aliyuncs.com"

    def __init__(self, objects):
        self.objects = objects

    def list_objects_v2(self, prefix="", delimiter="", continuation_token="", max_keys=1000):
        if delimiter == "/":
            subs, objs = set(), []
            for key, size in self.objects:
                if not key.startswith(prefix):
                    continue
                rest = key[len(prefix):]
                if "/" in rest:
                    subs.add(prefix + rest.split("/", 1)[0] + "/")
                else:
                    objs.append(_FakeObj(key, size))
            return _FakeResult(prefix_list=sorted(subs), object_list=objs)
        objs = [_FakeObj(k, s) for k, s in self.objects if k.startswith(prefix)]
        return _FakeResult(object_list=objs)


def test_compute_nested_sizes_emits_dtype(monkeypatch):
    objs = [
        # lightwheel/批次A：完整 LeRobot v3.0
        ("tp/lightwheel/ds_a/meta/info.json", 1),
        ("tp/lightwheel/ds_a/meta/tasks.parquet", 2),
        ("tp/lightwheel/ds_a/data/chunk-000/file-000.parquet", 100),
        # egodex/批次B：纯 hdf5
        ("tp/egodex/p1/episode_0.hdf5", 50),
    ]
    from tools.aliyun import oss as oss_mod
    monkeypatch.setattr(oss_mod, "_resolve_bucket", lambda *a, **k: _FakeBucket(objs))
    entries, _ = oss_mod.compute_nested_sizes("", "bkt", "tp/")
    by = {e["厂家"]: e for e in entries}
    assert by["lightwheel"]["dtype"] == "lerobot v3.0"
    assert {b[0]: b[4] for b in by["lightwheel"]["batches"]}["ds_a"] == "lerobot v3.0"
    assert by["egodex"]["dtype"] == "hdf5"
