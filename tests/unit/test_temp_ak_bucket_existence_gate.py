"""桶存在性校验接进发放流程的**拒绝路径**与 profile 透传。

auditor 指出的 T1：此前没有任何测试锁住这段 —— 把 `approval._validate_spec` 里的
桶存在性校验整段删掉、或去掉它的 profile 参数，全套测试照样绿。那意味着下一次
重构可以无声地把它删掉，而线上再次出现「桶名填成目录 → 发出废凭证」。
（线上单：主体 maxinsights，填 `third-party-data/xxx/`，而那是桶里的一个目录。）
"""
import pytest

from core.temp_ak_issuance import accounts, approval, orchestrator


@pytest.fixture
def spec_ok():
    """一份除桶之外都合法的 spec —— 让被测的只有桶存在性这一条。"""
    import time as _t
    now = _t.time()
    return {"platform": "aliyun", "bucket": "some-bucket", "prefix": "p/",
            "caps": ["write"], "not_before": now, "expire": now + 86400,
            "enterprise": "测试方"}


def test_missing_bucket_blocks_issuance(monkeypatch, spec_ok):
    """`bucket_missing_reason` 报「不存在」时**必须拒发**，且原因原样透出给用户。"""
    monkeypatch.setattr(orchestrator, "bucket_missing_reason",
                        lambda b, p=None: "桶 `x` 不存在。常见原因：只填了路径没填桶名")
    with pytest.raises(orchestrator.TempAkError, match="只填了路径没填桶名"):
        approval._validate_spec(spec_ok)


def test_existing_bucket_does_not_block(monkeypatch, spec_ok):
    """返回空 = 没结论 = 放行。这条链拦的是笔误，不是越权，不能宁枉勿纵。"""
    monkeypatch.setattr(orchestrator, "bucket_missing_reason", lambda b, p=None: "")
    approval._validate_spec(spec_ok)          # 不抛即通过


def test_profile_is_passed_through(monkeypatch, spec_ok):
    """**必须把该单所属账号档传下去** —— 读桶地域的权限是按账号授的，
    拿默认档的 AK 去探 1949 档的桶只会 403、探不到。"""
    seen = {}
    monkeypatch.setattr(orchestrator, "bucket_missing_reason",
                        lambda b, p=None: seen.update(bucket=b, profile=p) or "")
    d = accounts.default()
    other = d.__class__(**{**d.__dict__, "slug": "1949", "ak_id": "ak-1949"})
    approval._validate_spec(spec_ok, other)
    assert seen["bucket"] == "some-bucket"
    assert seen["profile"] is other, "传的不是该单的账号档"
    assert seen["profile"].slug == "1949"


def test_default_profile_when_omitted(monkeypatch, spec_ok):
    seen = {}
    monkeypatch.setattr(orchestrator, "bucket_missing_reason",
                        lambda b, p=None: seen.update(profile=p) or "")
    approval._validate_spec(spec_ok)
    assert seen["profile"] is None      # 由 bucket_missing_reason 内部回退默认档


def test_gate_is_actually_wired(spec_ok):
    """防「整段被删还全绿」：直接断言 _validate_spec 的源码里调了这个校验。

    比 mock 断言更钝，但正是它能挡住"重构时顺手删掉"这种改动 —— mock 版在函数被
    删掉后会因为 monkeypatch 无副作用而静默通过。
    """
    import inspect
    src = inspect.getsource(approval._validate_spec)
    assert "bucket_missing_reason" in src, "桶存在性校验被从发放流程里摘掉了"
    assert "profile" in inspect.signature(approval._validate_spec).parameters, \
        "_validate_spec 丢了 profile 参数 —— 会拿错账号的凭证探桶"


def test_caps_and_expire_checks_still_fire(monkeypatch, spec_ok):
    """本次改动不能把既有的门禁挤掉。"""
    monkeypatch.setattr(orchestrator, "bucket_missing_reason", lambda b, p=None: "")
    with pytest.raises(orchestrator.TempAkError, match="未勾选任何项"):
        approval._validate_spec({**spec_ok, "caps": []})
    with pytest.raises(orchestrator.TempAkError, match="到期"):
        approval._validate_spec({**spec_ok, "expire": 1})
