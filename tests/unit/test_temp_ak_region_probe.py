"""桶地域实时探测。

线上背景：`wuji-rl-dataset` 不在任何桶映射表里 → `resolve_bucket` 返回空 region →
凭证正文的「地域/Endpoint/桶域名」三行退化成「未知」→ 使用方随手用了默认 endpoint →
OSS 回 403 `must be addressed using the specified endpoint`，而这个报错和「没权限」
长得一模一样，于是被当成「权限策略没建好」。实际策略是好的（ARN 里 region 是 `*`）。
"""
import pytest

from core.temp_ak_issuance import accounts, orchestrator

# 全局 conftest 会 autouse 把 _probe_region_once 桩掉（防止单测真出网）。要测**它本身**，
# 就得在模块导入期先抓一份真函数 —— 那时 fixture 还没跑。
_REAL_PROBE_ONCE = orchestrator._probe_region_once


@pytest.fixture(autouse=True)
def _clear_cache():
    orchestrator._REGION_PROBE_CACHE.clear()
    yield
    orchestrator._REGION_PROBE_CACHE.clear()


def _mk(slug="", ak="ak", sk="sk"):
    d = accounts.default()
    return d.__class__(**{**d.__dict__, "slug": slug, "ak_id": ak, "ak_secret": sk})


def test_map_hit_does_not_probe(monkeypatch):
    """映射表命中时**不该**打 API —— 别为已知的桶白白多一次往返。"""
    called = []
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda *a, **k: called.append(1) or ("cn-beijing", ""))
    region, bucket = orchestrator.resolve_bucket("杭州-wuji-bucket-hangzhou")
    assert bucket == "wuji-bucket-hangzhou"
    assert region == "oss-cn-hangzhou"
    assert called == []


def test_map_miss_falls_back_to_probe(monkeypatch):
    """映射表查不到 → 实时探测。这是本次修复的主路径。"""
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("cn-hangzhou", ""))
    region, bucket = orchestrator.resolve_bucket("wuji-rl-dataset")
    assert (region, bucket) == ("cn-hangzhou", "wuji-rl-dataset")


def test_probe_failure_returns_empty_never_guesses(monkeypatch):
    """探不到必须返回 ""，**绝不回退默认地域**。

    一个自信的错地域比「未知」更坏：使用方会照着连、拿到 403，而 403 和「没权限」
    同形，排查方向会整个跑偏。返回空则 delivery 会如实写「未知，请按控制台自查」。
    """
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("网络炸了")))
    region, bucket = orchestrator.resolve_bucket("some-new-bucket")
    assert region == ""
    assert bucket == "some-new-bucket"


def test_probe_exception_never_breaks_issuance(monkeypatch):
    """探测异常不能把发放搞挂 —— 地域只影响正文展示，探不到照发。"""
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert orchestrator.resolve_bucket("b1") == ("", "b1")


def test_probe_result_is_cached_including_empty(monkeypatch):
    """空结果也要缓存：审批回调是同步路径，不能对同一个探不到的桶反复打 API。"""
    calls = []

    def _once(*a, **k):
        calls.append(1)
        return "", ""

    monkeypatch.setattr(orchestrator, "_probe_region_once", _once)
    for _ in range(3):
        orchestrator.probe_bucket_region("ghost-bucket")
    assert len(calls) == 1


def test_cache_is_keyed_by_account(monkeypatch):
    """**同名桶在两个主账号下不是同一个桶** —— 缓存必须按账号分开，
    否则会把 A 账号探到的地域套到 B 账号的同名桶上。"""
    seen = []

    def _once(bucket, ak, sk, **k):
        seen.append(ak)
        return ("cn-beijing" if ak == "ak-default" else "cn-shenzhen"), ""

    monkeypatch.setattr(orchestrator, "_probe_region_once", _once)
    pa = _mk(slug="", ak="ak-default")
    pb = _mk(slug="1949", ak="ak-1949")
    assert orchestrator.probe_bucket_region("same-name", pa) == "cn-beijing"
    assert orchestrator.probe_bucket_region("same-name", pb) == "cn-shenzhen"
    assert seen == ["ak-default", "ak-1949"]


def test_no_credentials_returns_empty_not_crash(monkeypatch):
    """账号没配 AK 时探不了，但不能抛 —— 发放流程不该因为一个展示字段而中断。"""
    assert orchestrator.probe_bucket_region("b", _mk(ak="", sk="")) == ""


def test_empty_bucket_is_rejected_before_probe():
    assert orchestrator.probe_bucket_region("") == ""
    assert orchestrator.probe_bucket_region("   ") == ""


# ── 探测本身（这段是从 tools/aliyun/oss.py 借鉴的，之前 8 个用例全 mock 掉了它，零覆盖）──

class _FakeOssError(Exception):
    """仿 oss2.exceptions.OssError：带 status / code / headers / body / request_id。"""
    def __init__(self, *, headers=None, body="", status=403, code="AccessDenied"):
        super().__init__(code)
        self.headers, self.body, self.status, self.code = headers or {}, body, status, code
        self.request_id = "req-1"


def _install_fake_oss2(monkeypatch, *, location=None, exc=None):
    """注入一个假 oss2，让 _probe_region_once 走真实解析逻辑但不出网。"""
    import sys, types

    class _Loc:
        def __init__(self, v): self.location = v

    class _Bucket:
        def __init__(self, *a, **k): pass
        def get_bucket_location(self):
            if exc is not None:
                raise exc
            return _Loc(location)

    fake = types.ModuleType("oss2")
    fake.Auth = lambda *a, **k: object()
    fake.Bucket = _Bucket
    fake.exceptions = types.SimpleNamespace(OssError=_FakeOssError)
    monkeypatch.setitem(sys.modules, "oss2", fake)


def test_probe_parses_normal_location(monkeypatch):
    """桶就在探测用的那个地域 → 直接拿到 location（带 oss- 前缀，要归一成裸 region）。"""
    _install_fake_oss2(monkeypatch, location="oss-cn-hangzhou")
    assert _REAL_PROBE_ONCE("b", "ak", "sk") == ("cn-hangzhou", "")


def test_probe_parses_x_oss_region_header(monkeypatch):
    """异地桶：OSS 拒绝但响应头带正确地域。"""
    _install_fake_oss2(monkeypatch, exc=_FakeOssError(headers={"x-oss-region": "oss-cn-shenzhen"}))
    assert _REAL_PROBE_ONCE("b", "ak", "sk") == ("cn-shenzhen", "")


@pytest.mark.parametrize("body", [
    "<Error><Endpoint>oss-ap-southeast-1.aliyuncs.com</Endpoint></Error>",
    b"<Error><Endpoint>oss-ap-southeast-1.aliyuncs.com</Endpoint></Error>",   # oss2 真实类型
])
def test_probe_parses_endpoint_in_body(monkeypatch, body):
    """异地桶的另一种形态：正确 endpoint 在 body 的 <Endpoint> 里。

    **必须同时测 bytes** —— oss2 的 `e.body` 是 bytes，拿 str 正则 search 会 TypeError，
    而那个异常会被上层吞掉、静默返回空 → 地域探测与桶存在性校验双双永不生效。
    最初这条只测了 str，于是真机一跑就崩、测试却全绿。
    """
    _install_fake_oss2(monkeypatch, exc=_FakeOssError(body=body))
    assert _REAL_PROBE_ONCE("b", "ak", "sk") == ("ap-southeast-1", "")


def test_probe_returns_empty_when_nothing_parseable(monkeypatch):
    """桶不存在 / 无权限：headers 和 body 都没线索 → 返回 ""，**不猜**。"""
    _install_fake_oss2(monkeypatch, exc=_FakeOssError(
        body=b"<Error><Code>NoSuchBucket</Code></Error>", code="NoSuchBucket"))
    assert _REAL_PROBE_ONCE("b", "ak", "sk") == ("", "NoSuchBucket")


def test_probe_never_logs_response_body(monkeypatch, caplog):
    """失败日志**绝不能带 e.body** —— SignatureDoesNotMatch 的 body 含 AccessKeyId 与 StringToSign。"""
    secret_ish = "<StringToSign>GET\n\nLTAI5tSECRETLOOKING</StringToSign>"
    _install_fake_oss2(monkeypatch, exc=_FakeOssError(body=secret_ish, code="SignatureDoesNotMatch"))
    with caplog.at_level("WARNING"):
        assert _REAL_PROBE_ONCE("b", "ak", "sk") == ("", "SignatureDoesNotMatch")
    blob = caplog.text
    assert "StringToSign" not in blob and "LTAI5tSECRETLOOKING" not in blob
    assert "SignatureDoesNotMatch" in blob      # 但错误码要留痕，否则排障没线索


# ── 输入/输出校验 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["杭州-wuji-sing", "UPPER", "a", "has_underscore", "-lead"])
def test_illegal_bucket_name_short_circuits(monkeypatch, bad):
    """中文展示名等非法桶名直接短路，不进 oss2（它的构造器会在 try 之外抛 ClientError）。"""
    called = []
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: called.append(1) or ("cn-x", ""))
    assert orchestrator.probe_bucket_region(bad) == ""
    assert called == []


def test_garbage_region_is_rejected(monkeypatch):
    """探测吐出不像地域的串（意外域名会让 region_from_endpoint 返回 `data` 这种）→ 当未知。

    一个写着「地域：data」的凭证比「未知」更误导，正好违背"绝不猜"的前提。"""
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("data", ""))
    assert orchestrator.probe_bucket_region("some-bucket") == ""


def test_failure_uses_short_ttl_so_it_can_self_heal(monkeypatch):
    """瞬时失败只短缓存：grant 一旦落库 region 就写死了，不能让一次抖动把这个桶永久钉成「未知」。"""
    import time as _t
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("", ""))
    orchestrator.probe_bucket_region("bkt")
    _, exp_fail = orchestrator._REGION_PROBE_CACHE[("", "bkt")]
    assert exp_fail - _t.time() <= orchestrator._PROBE_TTL_FAIL + 1

    orchestrator._REGION_PROBE_CACHE.clear()
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("cn-hangzhou", ""))
    orchestrator.probe_bucket_region("bkt")
    _, exp_ok = orchestrator._REGION_PROBE_CACHE[("", "bkt")]
    assert exp_ok - _t.time() > orchestrator._PROBE_TTL_FAIL      # 成功缓存明显更长


def test_expired_cache_is_reprobed(monkeypatch):
    """过期后要重探，否则短 TTL 形同虚设。"""
    calls = []
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda *a, **k: calls.append(1) or ("", ""))
    orchestrator.probe_bucket_region("bkt")
    orchestrator._REGION_PROBE_CACHE[("", "bkt")] = ("", 0)      # 手动置为已过期
    orchestrator.probe_bucket_region("bkt")
    assert len(calls) == 2


# ── 修 3：桶确定不存在才拦（bucket_missing_reason）──────────────────────────

def test_missing_bucket_blocks_only_on_nosuchbucket(monkeypatch):
    """确定性答案才拦：NoSuchBucket → 拦并说清成因。"""
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("", "NoSuchBucket"))
    why = orchestrator.bucket_missing_reason("third-party-data")
    assert "不存在" in why and "只填了路径没填桶名" in why


@pytest.mark.parametrize("code", ["AccessDenied", "RequestTimeout", "", "InvalidAccessKeyId"])
def test_non_definitive_errors_never_block(monkeypatch, code):
    """探不到 ≠ 不存在。权限不足/网络抖动一律放行 —— 拦错了会挡住正常发放。

    这条链的默认方向与 caps/审批那些门禁**相反**：那些拦越权，这条只拦笔误。
    """
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("", code))
    assert orchestrator.bucket_missing_reason("some-bucket") == ""


def test_existing_bucket_never_blocks(monkeypatch):
    monkeypatch.setattr(orchestrator, "_probe_region_once", lambda *a, **k: ("cn-hangzhou", ""))
    assert orchestrator.bucket_missing_reason("wuji-rl-dataset") == ""


def test_probe_crash_never_blocks(monkeypatch):
    """探测本身炸了 —— 判断不了就别拦。"""
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert orchestrator.bucket_missing_reason("valid-bucket") == ""


def test_missing_bucket_check_skips_without_credentials(monkeypatch):
    assert orchestrator.bucket_missing_reason("valid-bucket", _mk(ak="", sk="")) == ""


def test_missing_bucket_uses_own_account_credentials(monkeypatch):
    """必须用该档自己的凭证 —— 读桶地域的权限是按账号授的。"""
    seen = []
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda b, ak, sk, **k: seen.append(ak) or ("", "NoSuchBucket"))
    orchestrator.bucket_missing_reason("valid-bucket", _mk(slug="1949", ak="ak-1949"))
    assert seen == ["ak-1949"]


def test_single_network_entrypoint(monkeypatch):
    """地域探测与桶存在性检查**必须共用同一个出网点** —— 多一个入口就多一个会被漏桩的坑，
    而漏桩的后果是单测拿生产 AK 打真 API（本次已经踩过一次）。"""
    calls = []
    monkeypatch.setattr(orchestrator, "_probe_region_once",
                        lambda *a, **k: calls.append(1) or ("", ""))
    orchestrator._REGION_PROBE_CACHE.clear()
    orchestrator.probe_bucket_region("bkt-a")
    orchestrator.bucket_missing_reason("bkt-b")
    assert len(calls) == 2      # 两条路径都只经过这一个函数


def test_illegal_bucket_name_is_not_reported_as_missing():
    """形状就不合法的桶名交给别的校验报，这里不重复报 —— 也别因此去打 API。

    （这条是写测试时自己踩的：用 "b" 当桶名，1 个字符过不了正则，
    于是断言其实什么都没验到。）"""
    assert orchestrator.bucket_missing_reason("b") == ""          # 太短
    assert orchestrator.bucket_missing_reason("杭州-某桶") == ""    # 中文展示名
