"""HIGH#2（#37）：manage_oss('list_objects') 用 itertools.islice 在第一页截断，
不再 list(ObjectIterator(...))[:max_keys] 枚举整个 prefix 下全部对象。

oss2 2.19.1 的 ObjectIterator.max_keys 只是「每页大小」，list(...) 会翻页枚举整桶
（百万级桶→挂线程/OOM）。本测注入一个「超大」ObjectIterator（可 yield 远多于 max_keys），
断言：① 只返回 max_keys 条；② 迭代器只被 next 了 max_keys 次（未被耗尽）。
"""
import sys
import types

import pytest

from tools.aliyun import oss as oss_tool


class _Obj:
    def __init__(self, i):
        self.key = f"data/file{i}.bin"
        self.size = 1
        self.last_modified = "2026-07-09T00:00:00"


class _CountingIterator:
    """无限（超大）对象迭代器：记录被 next 的次数。若 next 超过安全上限则直接失败
    （模拟“整桶枚举”会挂/OOM 的场景，让退化实现无处遁形）。"""
    SAFETY_CAP = 100_000

    def __init__(self):
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed >= self.SAFETY_CAP:
            raise AssertionError("ObjectIterator 被整桶枚举（未在第一页截断）")
        self.consumed += 1
        return _Obj(self.consumed)


@pytest.fixture
def fake_oss2(monkeypatch):
    """注入假 oss2 模块（manage_oss 内 `import oss2`），捕获创建的迭代器实例与构造参数。"""
    cap = {}

    def _object_iterator(bucket, prefix="", max_keys=100, **kw):
        it = _CountingIterator()
        cap["iter"] = it
        cap["prefix"] = prefix
        cap["max_keys_arg"] = max_keys
        return it

    fake = types.SimpleNamespace(ObjectIterator=_object_iterator)
    monkeypatch.setitem(sys.modules, "oss2", fake)
    monkeypatch.setattr(oss_tool, "get_oss_bucket", lambda open_id, bucket, region="": object())
    return cap


def test_list_objects_truncates_at_max_keys(fake_oss2):
    out = oss_tool.manage_oss(action="list_objects", bucket="huge-bucket", max_keys=50)
    # 只被 next 了 max_keys 次 → 第一页就 break，绝不耗尽整个迭代器
    assert fake_oss2["iter"].consumed == 50
    # 返回内容也只含 50 条
    assert "前 50 条" in out
    assert out.count("data/file") == 50


def test_list_objects_small_max_keys(fake_oss2):
    """极小 max_keys 也只取该数量。"""
    out = oss_tool.manage_oss(action="list_objects", bucket="b", prefix="logs/", max_keys=3)
    assert fake_oss2["iter"].consumed == 3
    assert fake_oss2["prefix"] == "logs/"
    assert out.count("data/file") == 3


def test_list_objects_does_not_exhaust_iterator(fake_oss2):
    """强断言“未耗尽”：取完 max_keys 后迭代器仍能继续 next（还有剩余对象）。"""
    oss_tool.manage_oss(action="list_objects", bucket="b", max_keys=10)
    it = fake_oss2["iter"]
    assert it.consumed == 10
    # 迭代器并未走到 StopIteration —— 还能取到第 11 个
    nxt = next(it)
    assert nxt.key == "data/file11.bin"
