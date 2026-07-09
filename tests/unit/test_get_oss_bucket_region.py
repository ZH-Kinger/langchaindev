"""#46 C5：get_oss_bucket region 处理。

  - 传 region → endpoint = https://oss-<region>.aliyuncs.com（不探测）
  - 不传 region → 调 tools.aliyun.oss._detect_region_endpoint 探测桶真实地域（跨地域桶不打错 endpoint）
  - 探测抛错 → 回退默认地域（settings.PAI_DSW_REGION_ID or cn-hangzhou）
"""
import pytest


@pytest.fixture
def cap(monkeypatch):
    """mock get_oss_auth + oss2.Bucket，捕获传给 Bucket 的 endpoint。"""
    from utils import aliyun_client_factory as factory
    monkeypatch.setattr(factory, "get_oss_auth", lambda oid: ("AUTH", "ep-ignored"))

    captured = {}

    class _FakeBucket:
        def __init__(self, auth, endpoint, bucket_name):
            captured["auth"] = auth
            captured["endpoint"] = endpoint
            captured["bucket"] = bucket_name

    import oss2
    monkeypatch.setattr(oss2, "Bucket", _FakeBucket)
    return factory, captured


def test_explicit_region_no_detection(cap, monkeypatch):
    factory, captured = cap
    import tools.aliyun.oss as oss_mod
    called = {"n": 0}
    monkeypatch.setattr(oss_mod, "_detect_region_endpoint",
                        lambda a, b: called.__setitem__("n", called["n"] + 1) or "SHOULD_NOT_USE")

    factory.get_oss_bucket("ou_1", "mybk", region="cn-beijing")
    assert captured["endpoint"] == "https://oss-cn-beijing.aliyuncs.com"
    assert called["n"] == 0, "传 region 时不应探测"


def test_no_region_calls_detect(cap, monkeypatch):
    factory, captured = cap
    import tools.aliyun.oss as oss_mod
    seen = {}
    def _detect(auth, bucket):
        seen["auth"], seen["bucket"] = auth, bucket
        return "https://oss-cn-shenzhen.aliyuncs.com"
    monkeypatch.setattr(oss_mod, "_detect_region_endpoint", _detect)

    factory.get_oss_bucket("ou_1", "mybk")     # region 缺省
    assert captured["endpoint"] == "https://oss-cn-shenzhen.aliyuncs.com"
    assert seen == {"auth": "AUTH", "bucket": "mybk"}


def test_detect_error_falls_back_to_default(cap, monkeypatch):
    factory, captured = cap
    import tools.aliyun.oss as oss_mod
    monkeypatch.setattr(oss_mod, "_detect_region_endpoint",
                        lambda a, b: (_ for _ in ()).throw(RuntimeError("probe boom")))
    monkeypatch.setattr(factory.settings, "PAI_DSW_REGION_ID", "cn-hangzhou")

    factory.get_oss_bucket("ou_1", "mybk")
    assert captured["endpoint"] == "https://oss-cn-hangzhou.aliyuncs.com"


def test_no_auth_returns_none(monkeypatch):
    from utils import aliyun_client_factory as factory
    monkeypatch.setattr(factory, "get_oss_auth", lambda oid: (None, None))
    assert factory.get_oss_bucket("ou_1", "mybk") is None
