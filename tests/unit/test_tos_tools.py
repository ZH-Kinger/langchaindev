"""tools/volcano/tos.py 测试。

用 FakeTosClient 模拟 tos.TosClientV2.list_objects_type2，不调真实 TOS。
另含路由冒烟：火山/tos 关键词应命中 manage_tos，不串到 oss。
"""
import pytest


class _FakeObj:
    def __init__(self, key, size):
        self.key = key
        self.size = size
        self.last_modified = "2026-06-04T00:00:00Z"


class _FakePrefix:
    def __init__(self, prefix):
        self.prefix = prefix


class _FakeResult:
    def __init__(self, common_prefixes=None, contents=None):
        self.common_prefixes = common_prefixes or []
        self.contents = contents or []
        self.is_truncated = False
        self.next_continuation_token = ""


class _FakeTosClient:
    def __init__(self, objects):
        self.objects = objects
        self.closed = False

    def list_objects_type2(self, bucket, prefix="", delimiter="",
                           continuation_token="", max_keys=1000):
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
            return _FakeResult(common_prefixes=[_FakePrefix(p) for p in sorted(subs)],
                               contents=objs)
        objs = [_FakeObj(k, s) for k, s in self.objects if k.startswith(prefix)]
        return _FakeResult(contents=objs)

    def close(self):
        self.closed = True


@pytest.fixture
def patch_tos(monkeypatch):
    def _install(objects):
        from tools.volcano import tos
        client = _FakeTosClient(objects)
        monkeypatch.setattr(tos, "get_tos_client", lambda: client)
        return client
    return _install


def test_tos_compute_dir_sizes(patch_tos):
    client = patch_tos([
        ("tp/egodex/a", 100),
        ("tp/egodex/b", 200),
        ("tp/egoverse/", 0),         # 占位对象，跳过
        ("tp/egoverse/c", 50),
    ])
    from tools.volcano.tos import compute_dir_sizes
    rows = compute_dir_sizes("bkt", "tp/")
    by_name = {n: (s, c) for n, s, c in rows}
    assert by_name["egodex"] == (300, 2)
    assert by_name["egoverse"] == (50, 1)
    assert client.closed is True     # 用完应关闭


def test_tos_compute_dir_sizes_no_creds(monkeypatch):
    from tools.volcano import tos
    monkeypatch.setattr(tos, "get_tos_client", lambda: None)
    assert tos.compute_dir_sizes("bkt", "tp/") is None


def test_tos_compute_nested_sizes(patch_tos):
    patch_tos([
        ("tp/egodex/2026-01/a", 100),
        ("tp/egodex/2026-01/b", 50),
        ("tp/egodex/2026-02/c", 200),
        ("tp/egodex/root.txt", 10),             # 厂家直属文件 → (根)
        ("tp/egoverse/x/d", 5),
    ])
    from tools.volcano.tos import compute_nested_sizes
    entries = compute_nested_sizes("bkt", "tp/")
    by = {e["厂家"]: e for e in entries}
    eg = by["egodex"]
    assert eg["total_bytes"] == 360 and eg["total_count"] == 4
    batches = {name: (b, c) for name, b, c in eg["batches"]}
    assert batches["2026-01"] == (150, 2) and batches["(根)"] == (10, 1)
    assert by["egoverse"]["total_bytes"] == 5


def test_manage_tos_dir_sizes_table(patch_tos):
    patch_tos([("tp/x/a", 1024), ("tp/y/b", 2048)])
    from tools.volcano.tos import manage_tos
    out = manage_tos(action="dir_sizes", bucket="bkt", prefix="tp/")
    assert "合计" in out and "| `x` |" in out


def test_manage_tos_no_creds_message(monkeypatch):
    from tools.volcano import tos
    monkeypatch.setattr(tos, "get_tos_client", lambda: None)
    out = tos.manage_tos(action="dir_sizes", bucket="bkt", prefix="tp/")
    assert "凭证不可用" in out


# ── 路由冒烟：火山/tos 不应被 oss 的"对象存储"抢走 ───────────────────────────

def test_legacy_route_tos_not_leak_to_oss():
    from core.agent import _legacy_keyword_route
    names = {t.name for t in _legacy_keyword_route("火山 tos 目录大小")}
    assert "manage_tos" in names
    assert "manage_oss" not in names


def test_legacy_route_oss_dir_size_keyword():
    from core.agent import _legacy_keyword_route
    names = {t.name for t in _legacy_keyword_route("oss 目录大小统计")}
    assert "manage_oss" in names
