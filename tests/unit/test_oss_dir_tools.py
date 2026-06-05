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


class _FakeBucket:
    """按 (key, size) 列表模拟对象存储；支持 delimiter='/' 列子目录。"""
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


@pytest.fixture
def patch_bucket(monkeypatch):
    """把 _resolve_bucket 替换为返回指定 FakeBucket。"""
    def _install(objects):
        from tools.aliyun import oss
        monkeypatch.setattr(oss, "_resolve_bucket", lambda *a, **k: _FakeBucket(objects))
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
        ("tp/lightwheel/readme.txt", 10),       # 厂家直属文件 → (根)
        ("tp/aether/x/d.bin", 5),
    ])
    from tools.aliyun.oss import compute_nested_sizes
    entries, _ep = compute_nested_sizes("", "bkt", "tp/")
    by = {e["厂家"]: e for e in entries}

    lw = by["lightwheel"]
    assert lw["total_bytes"] == 360 and lw["total_count"] == 4
    batches = {name: (b, c) for name, b, c in lw["batches"]}
    assert batches["2026-01"] == (150, 2)
    assert batches["2026-02"] == (200, 1)
    assert batches["(根)"] == (10, 1)            # 直属文件归入 (根)

    assert by["aether"]["total_bytes"] == 5


def test_compute_nested_sizes_uses_cache(monkeypatch, patch_bucket):
    """已缓存的批次应复用缓存、不再调用 _prefix_size 实扫；新批次仍实扫。"""
    patch_bucket([
        ("tp/lw/2026-01/a.bin", 100),     # 已交付（缓存）
        ("tp/lw/2026-02/b.bin", 200),     # 新批次（须实扫）
    ])
    from tools.aliyun import oss as oss_mod
    calls = []
    real_prefix_size = oss_mod._prefix_size
    monkeypatch.setattr(oss_mod, "_prefix_size",
                        lambda b, p: calls.append(p) or real_prefix_size(b, p))

    cached = {"lw/2026-01": (100, 1)}     # 2026-01 已缓存
    entries, _ = oss_mod.compute_nested_sizes("", "bkt", "tp/", cached=cached)
    lw = entries[0]
    assert lw["total_bytes"] == 300       # 100(缓存) + 200(实扫)
    # 只对新批次 2026-02 实扫，已缓存的 2026-01 未触发 _prefix_size
    assert any("2026-02" in p for p in calls)
    assert not any("2026-01" in p for p in calls)


def test_flat_family_collapses_to_total(patch_bucket):
    """批次数超过 _MAX_BATCHES 的「平铺」厂家应折叠成单行 '(整体)'，不细分。"""
    from tools.aliyun import oss as oss_mod
    objs = [(f"tp/egoverse/b{i}/f.bin", 1) for i in range(oss_mod._MAX_BATCHES + 30)]
    objs.append(("tp/egodex/2026-01/a.bin", 100))   # 正常厂家不应被折叠
    patch_bucket(objs)

    entries, _ = oss_mod.compute_nested_sizes("", "bkt", "tp/")
    by = {e["厂家"]: e for e in entries}
    ego = by["egoverse"]
    assert len(ego["batches"]) == 1 and ego["batches"][0][0] == "(整体)"
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
