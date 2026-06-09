"""tools/aliyun/oss.py 的 dir_sizes / tree 能力测试。

用 FakeBucket 模拟 oss2.Bucket（list_objects_v2），不调真实 OSS。
覆盖：子目录大小求和、占位对象跳过、目录树降级阈值、噪音目录折叠。
"""
import pytest


# ── 假对象存储 ────────────────────────────────────────────────────────────────

class _FakeObj:
    def __init__(self, key, size):
        self.key = key
        self.size = size


class _FakeResult:
    def __init__(self, prefix_list=None, object_list=None):
        self.prefix_list = prefix_list or []
        self.object_list = object_list or []
        self.is_truncated = False
        self.next_continuation_token = ""


class _FakeGetResult:
    def __init__(self, content):
        self._content = content

    def read(self):
        return self._content


class _FakeBucket:
    """按 (key, size) 列表模拟对象存储；支持 delimiter='/' 列子目录、get_object 取内容。"""
    endpoint = "https://oss-cn-hangzhou.aliyuncs.com"

    def __init__(self, objects, contents=None):
        self.objects = objects
        self.contents = contents or {}     # {key: bytes/str}，供 get_object

    def get_object(self, key):
        if key not in self.contents:
            raise KeyError(key)
        return _FakeGetResult(self.contents[key])

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


@pytest.fixture
def patch_bucket(monkeypatch):
    """把 _resolve_bucket 替换为返回指定 FakeBucket。"""
    def _install(objects, contents=None):
        from tools.aliyun import oss
        monkeypatch.setattr(oss, "_resolve_bucket", lambda *a, **k: _FakeBucket(objects, contents))
    return _install


# ── compute_dir_sizes ────────────────────────────────────────────────────────

def test_compute_dir_sizes_sums_and_skips_placeholder(patch_bucket):
    patch_bucket([
        ("third-party/aether/a.bin", 100),
        ("third-party/aether/b.bin", 50),
        ("third-party/lightwheel/", 0),          # 占位对象，应跳过
        ("third-party/lightwheel/c.bin", 200),
        ("third-party/lightwheel/sub/d.bin", 25),
    ])
    from tools.aliyun.oss import compute_dir_sizes
    rows, endpoint = compute_dir_sizes("", "bkt", "third-party/")
    by_name = {name: (size, cnt) for name, size, cnt in rows}

    assert by_name["aether"] == (150, 2)
    assert by_name["lightwheel"] == (225, 2)   # 占位 0 字节不计入
    assert endpoint.startswith("https://oss-")


def test_compute_dir_sizes_empty(patch_bucket):
    patch_bucket([])
    from tools.aliyun.oss import compute_dir_sizes
    rows, _ = compute_dir_sizes("", "bkt", "nothing/")
    assert rows == []


def test_manage_oss_dir_sizes_renders_table(patch_bucket):
    patch_bucket([
        ("d/x/a", 1024),
        ("d/y/b", 2048),
    ])
    from tools.aliyun.oss import manage_oss
    out = manage_oss(action="dir_sizes", bucket="bkt", prefix="d/")
    assert "合计" in out
    assert "| `x` |" in out and "| `y` |" in out


def test_manage_oss_dir_sizes_requires_bucket():
    from tools.aliyun.oss import manage_oss
    assert "需要提供 bucket" in manage_oss(action="dir_sizes")


# ── build_tree ───────────────────────────────────────────────────────────────

def test_build_tree_basic_nesting(patch_bucket):
    patch_bucket([
        ("a/1/x", 1),
        ("a/2/y", 1),
        ("b/z", 1),
    ])
    from tools.aliyun.oss import build_tree
    out = build_tree("", "bkt", "", max_depth=2)
    assert "|-- a/" in out and "|-- b/" in out
    assert "共" in out and "目录" in out


def test_build_tree_degrades_when_too_many_siblings(patch_bucket):
    # 60 个同级目录 > 阈值 50，应触发降级
    patch_bucket([(f"d{i}/x", 1) for i in range(60)])
    from tools.aliyun.oss import build_tree
    out = build_tree("", "bkt", "", max_depth=3)
    assert "降级" in out and "省略其余" in out


def test_compute_nested_sizes_two_levels(patch_bucket):
    patch_bucket([
        ("tp/lightwheel/2026-01/a.bin", 100),
        ("tp/lightwheel/2026-01/b.bin", 50),
        ("tp/lightwheel/2026-02/c.bin", 200),
        ("tp/lightwheel/readme.txt", 10),       # 厂家直属文件 → /
        ("tp/aether/x/d.bin", 5),
    ])
    from tools.aliyun.oss import compute_nested_sizes
    entries, _ep = compute_nested_sizes("", "bkt", "tp/")
    by = {e["厂家"]: e for e in entries}

    lw = by["lightwheel"]
    assert lw["total_bytes"] == 360 and lw["total_count"] == 4
    batches = {name: (b, c) for name, b, c, st, dt, hrs in lw["batches"]}
    assert batches["2026-01"] == (150, 2)
    assert batches["2026-02"] == (200, 1)
    assert batches["/"] == (10, 1)            # 直属文件归入 /

    assert by["aether"]["total_bytes"] == 5


def test_batch_key_descends_to_dataset_root():
    """_batch_key：含时长目录段 → 下钻到该层；否则退回第一段；家直属文件 → '/'。"""
    from tools.aliyun.oss import _batch_key
    # 深层嵌套（shutu 风格）：批次 = 真实数据集根
    assert _batch_key("wuji/26_0430/wuji_home_scene_19.75h/meta/tasks.parquet") \
        == "wuji/26_0430/wuji_home_scene_19.75h"
    # 时长在第一层（lingsheng/lightwheel 风格）：保持第一段
    assert _batch_key("wuji_itw_100h_202605271016/data/x.parquet") == "wuji_itw_100h_202605271016"
    # 无时长目录（egodex/egoverse 风格）：退回第一段
    assert _batch_key("part1/0.hdf5") == "part1"
    # 家直属文件
    assert _batch_key("readme.txt") == "/"


def test_compute_nested_sizes_deep_dataset_roots(patch_bucket):
    """深层嵌套的家（容器/容器/数据集_NNh）应拆成各真实数据集为批次,而非停在容器层。"""
    patch_bucket([
        ("tp/shutu/wuji/26_0430/wuji_home_scene_19.75h/meta/tasks.parquet", 10),
        ("tp/shutu/wuji/26_0430/wuji_home_scene_19.75h/data/chunk-000/file-000.parquet", 500),
        ("tp/shutu/wuji/26_0505/wuji_home_scene_85.11h/meta/tasks.parquet", 10),
        ("tp/shutu/wuji/26_0505/wuji_home_scene_85.11h/data/chunk-000/file-000.parquet", 800),
    ])
    from tools.aliyun.oss import compute_nested_sizes
    by = {e["厂家"]: e for e in compute_nested_sizes("", "bkt", "tp/")[0]}
    batches = {n: dt for n, b, c, st, dt, hrs in by["shutu"]["batches"]}
    # 不再是一个容器批次 "wuji",而是两个真实数据集根
    assert "wuji/26_0430/wuji_home_scene_19.75h" in batches
    assert "wuji/26_0505/wuji_home_scene_85.11h" in batches
    assert "wuji" not in batches
    assert batches["wuji/26_0430/wuji_home_scene_19.75h"] == "lerobot v3.0"


def test_dot_and_test_dirs_ignored(patch_bucket):
    """. 开头 与 test 目录（批次级 / 批次内）完全忽略：不算批次、不计入大小/对象数。"""
    patch_bucket([
        ("tp/lw/2026-01/a.bin", 100),
        ("tp/lw/.cache/x", 999),                  # 厂家级点目录 → 忽略
        ("tp/lw/2026-01/.deliver/y.txt", 40),     # 批次内点目录 → 忽略
        ("tp/lw/test/z.bin", 500),                # 厂家级 test 目录 → 忽略
        ("tp/lw/2026-01/test/w.bin", 30),         # 批次内 test 目录 → 忽略
    ])
    from tools.aliyun.oss import compute_nested_sizes
    lw = compute_nested_sizes("", "bkt", "tp/")[0][0]
    names = {b[0] for b in lw["batches"]}
    assert ".cache" not in names and "test" not in names
    assert lw["total_bytes"] == 100 and lw["total_count"] == 1


def test_struct_detection(patch_bucket):
    """按扩展名判数据结构：hdf5→ego，parquet/mcap→itw。"""
    patch_bucket([
        ("tp/egodex/p1/0.hdf5", 10), ("tp/egodex/p1/0.mp4", 5),
        ("tp/lingsheng/itw1/data/chunk-000/f.parquet", 20),
        ("tp/lightwheel/6-2-504h/u/x.mcap", 30),
    ])
    from tools.aliyun.oss import compute_nested_sizes
    by = {e["厂家"]: e for e in compute_nested_sizes("", "bkt", "tp/")[0]}
    assert by["egodex"]["struct"] == "ego"
    assert by["lingsheng"]["struct"] == "itw"
    assert by["lightwheel"]["struct"] == "itw"
    assert {b[0]: b[3] for b in by["egodex"]["batches"]}["p1"] == "ego"


def test_struct_from_name_fallback_unit():
    """目录名兜底：含 itw→itw、含 ego→ego、其余→空。"""
    from tools.aliyun.oss import _struct_from_name
    assert _struct_from_name("wuji-itw_500h_200items-202605181031") == "itw"
    assert _struct_from_name("egodex") == "ego"
    assert _struct_from_name("nuoyiteng") == ""
    assert _struct_from_name("larybench") == ""


def test_struct_name_fallback_fills_raw_capture(patch_bucket):
    """原始采集（mkv/csv/json，无 parquet/hdf5）→ 扩展名判不出，靠批次名 itw 兜底。"""
    patch_bucket([
        ("tp/nuoyiteng/wuji-itw_500h_200items/uuid1/config.json", 10),
        ("tp/nuoyiteng/wuji-itw_500h_200items/uuid1/depth_head.mkv", 9999),
        ("tp/nuoyiteng/wuji-itw_500h_200items/uuid1/depth_head.csv", 20),
    ])
    from tools.aliyun.oss import compute_nested_sizes
    by = {e["厂家"]: e for e in compute_nested_sizes("", "bkt", "tp/")[0]}
    assert by["nuoyiteng"]["struct"] == "itw"      # 名字含 itw → 兜底判 itw
    assert {b[0]: b[3] for b in by["nuoyiteng"]["batches"]}["wuji-itw_500h_200items"] == "itw"


def test_struct_extension_not_overridden_by_name(patch_bucket):
    """已按文件扩展名判出的不被目录名兜底覆盖（egoverse 文件是 parquet→itw，名字含 ego 不改成 ego）。"""
    patch_bucket([
        ("tp/egoverse/batch1/data/chunk-000/f.parquet", 100),   # 文件→itw
    ])
    from tools.aliyun.oss import compute_nested_sizes
    by = {e["厂家"]: e for e in compute_nested_sizes("", "bkt", "tp/")[0]}
    assert by["egoverse"]["struct"] == "itw"       # 不因厂家名含 ego 变成 ego/itw


def test_scan_reads_info_json_for_lerobot(patch_bucket):
    """LeRobot 批次读 meta/info.json：精确时长(total_frames/fps) + 权威版本(codebase_version)。"""
    import json
    info = json.dumps({"codebase_version": "v3.0", "fps": 30, "total_frames": 2160000})  # 20h
    patch_bucket(
        [("tp/lr/ds_a/meta/info.json", 5), ("tp/lr/ds_a/data/chunk-000/file-000.parquet", 100)],
        contents={"tp/lr/ds_a/meta/info.json": info},
    )
    from tools.aliyun.oss import compute_nested_sizes
    lr = compute_nested_sizes("", "bkt", "tp/")[0][0]
    bat = {b[0]: b for b in lr["batches"]}["ds_a"]
    assert bat[4] == "lerobot v3.0"        # 来自 codebase_version（即便结构只见 info.json）
    assert bat[5] == 20.0                  # 2160000/30/3600


def test_scan_info_json_read_failure_degrades(patch_bucket):
    """info.json 读取失败（无内容）→ 时长 None、版本退回结构启发式，不报错。"""
    patch_bucket([
        ("tp/lr/ds_b/meta/info.json", 5),
        ("tp/lr/ds_b/meta/tasks.parquet", 5),
    ])  # 没给 contents → get_object 抛错
    from tools.aliyun.oss import compute_nested_sizes
    lr = compute_nested_sizes("", "bkt", "tp/")[0][0]
    bat = {b[0]: b for b in lr["batches"]}["ds_b"]
    assert bat[4] == "lerobot v3.0"        # 结构兜底（tasks.parquet）
    assert bat[5] is None                  # 读不到 → 时长空


def test_flat_family_collapses_to_total(patch_bucket):
    """批次数超过 _MAX_BATCHES 的「平铺」厂家应折叠成单行 'ALL'，不细分。"""
    from tools.aliyun import oss as oss_mod
    objs = [(f"tp/egoverse/b{i}/f.bin", 1) for i in range(oss_mod._MAX_BATCHES + 30)]
    objs.append(("tp/egodex/2026-01/a.bin", 100))   # 正常厂家不应被折叠
    patch_bucket(objs)

    entries, _ = oss_mod.compute_nested_sizes("", "bkt", "tp/")
    by = {e["厂家"]: e for e in entries}
    ego = by["egoverse"]
    assert len(ego["batches"]) == 1 and ego["batches"][0][0] == "ALL"
    assert ego["total_count"] == oss_mod._MAX_BATCHES + 30
    # 正常厂家保留批次明细
    assert by["egodex"]["batches"][0][0] == "2026-01"


def test_build_tree_collapses_noise_dirs(patch_bucket):
    patch_bucket([
        (".git/objects/aa/bb", 1),
        ("src/main.py", 1),
    ])
    from tools.aliyun.oss import build_tree
    out = build_tree("", "bkt", "", max_depth=3)
    assert "已折叠" in out          # .git 命中噪音目录
