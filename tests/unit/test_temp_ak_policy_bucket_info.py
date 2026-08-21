"""修 1：只勾「上传」的凭证也要拿到桶信息语句。

线上「元客」那单：caps=['write'] → 策略里只有一条 PutObject 语句，连 GetBucketInfo 都没有。
ossutil / SDK / 控制台在上传前普遍会先探一次桶 → 403 → 表现成「凭证发了但什么都干不了、
策略里看不到任何路径」。
"""
import itertools
import json
import time

import pytest

from core.temp_ak_issuance.policy import (BUCKET_INFO_ACTIONS, LIST_ACTIONS,
                                          build_policy_with_window)

NOW = 1787000000


def _p(caps, prefix="a/b/"):
    return build_policy_with_window("bkt", prefix=prefix, caps=caps,
                                    not_before=NOW, expire=NOW + 9999)


def _actions(doc):
    return [tuple(s["Action"]) for s in doc["Statement"]]


def test_write_only_now_gets_bucket_info():
    doc = _p(["write"])
    assert tuple(BUCKET_INFO_ACTIONS) in _actions(doc), "只勾上传的凭证拿不到桶信息 = 客户端探桶即 403"


def test_write_only_still_cannot_list_objects():
    """给桶信息不等于给列举 —— 权限模型的正交性不能被这次修复破坏。

    外部方只勾了上传，就不该看得到桶里有什么。"""
    doc = _p(["write"])
    flat = json.dumps(doc)
    assert "oss:ListObjects" not in flat
    assert "oss:GetObject" not in flat        # 也不该能下载


def test_bucket_info_statement_has_no_prefix_condition():
    """桶级操作不带 prefix 参数，叠 oss:Prefix 会被服务端拒 —— 线上踩过。"""
    doc = _p(["write"])
    info = [s for s in doc["Statement"] if tuple(s["Action"]) == tuple(BUCKET_INFO_ACTIONS)][0]
    assert "Condition" not in info
    assert info["Resource"] == ["acs:oss:*:*:bkt"]


@pytest.mark.parametrize("caps", [c for r in (1, 2, 3)
                                  for c in itertools.combinations(["read", "download", "write"], r)])
def test_every_combination_has_bucket_info(caps):
    """任何非空 caps 都要能定位桶。"""
    assert tuple(BUCKET_INFO_ACTIONS) in _actions(_p(list(caps)))


def test_empty_caps_still_produces_nothing():
    """caps 为空仍是空策略（审批层有守卫会先拦，这里守住兜底行为不被这次修改带偏）。"""
    assert _p([])["Statement"] == []
