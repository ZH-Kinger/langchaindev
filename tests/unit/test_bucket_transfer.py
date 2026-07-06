"""桶间迁移单测：paths 解析 / 引擎源端分支 / orchestrator 状态机（mock 引擎）。"""
import pytest

from core.bucket_transfer import paths, orchestrator as orch
from core.transfer import engine_mgw, engine_tos
from tools.aliyun import oss as oss_tool


# ── paths ─────────────────────────────────────────────────────────────────────

def test_plan_oss_to_oss():
    p = paths.build_plan("oss://a/data/", "oss://b/data/")
    assert p.cloud == "aliyun" and p.engine == "mgw" and p.direction == "oss→oss"
    assert p.src.bucket == "a" and p.dest.bucket == "b"


def test_plan_tos_to_tos():
    p = paths.build_plan("tos://a/x/", "tos://b/x/")
    assert p.cloud == "volcano" and p.engine == "dms" and p.direction == "tos→tos"


def test_plan_mixed_scheme_rejected():
    with pytest.raises(paths.PathError):
        paths.build_plan("oss://a/x/", "tos://b/x/")


def test_plan_requires_both():
    with pytest.raises(paths.PathError):
        paths.build_plan("oss://a/x/", "")


def test_plan_same_src_dest_rejected():
    with pytest.raises(paths.PathError):
        paths.build_plan("oss://a/x/", "oss://a/x/")


def test_bucket_transfer_intent_and_card():
    from core.feishu_bot import messages
    from core.bucket_transfer import cards
    assert messages._is_bucket_transfer_entry_intent("桶间迁移")
    assert messages._is_bucket_transfer_entry_intent("oss到oss")
    assert not messages._is_bucket_transfer_entry_intent("查一下 GPU")
    ec = cards.entry_card()
    names = {e.get("name") for e in ec["body"]["elements"][1]["elements"]}
    assert {"source", "dest", "same_name"} <= names
    # 提交动作正确
    btn = [e for e in ec["body"]["elements"][1]["elements"] if e.get("tag") == "button"][0]
    assert btn["behaviors"][0]["value"]["action"] == "submit_bucket_transfer"


# ── 引擎源端分支 ────────────────────────────────────────────────────────────────

def test_mgw_oss_source_address_fields(monkeypatch):
    cap = {}
    monkeypatch.setattr(engine_mgw, "_models", lambda: _FakeModels())
    monkeypatch.setattr(engine_mgw, "_put_address", lambda c, n, d: cap.update(name=n, detail=d))
    engine_mgw.create_oss_source_address(object(), "job-src", bucket="a", prefix="p/",
                                         region="cn-beijing", role="r", internal=False)
    d = cap["detail"]
    assert d.address_type == "oss" and d.region_id == "oss-cn-beijing"
    assert d.role == "r" and d.domain == "oss-cn-beijing.aliyuncs.com"   # 公网


def test_tos_source_branch_uses_tos_vendor(monkeypatch):
    cap = {}
    fake = _FakeDms(cap)
    monkeypatch.setattr(engine_tos, "_api", lambda region: (fake, fake))
    tid = engine_tos.create_migrate_task(
        task_name="job", src_bucket="a", src_prefix="p/", src_region="cn-shanghai",
        src_access_id="ak", src_access_secret="sk", dest_bucket="b", dest_region="cn-shanghai",
        src_is_tos=True)
    assert tid == 123
    bac = cap["src_bac"]
    assert bac["vendor"] == engine_tos.SOURCE_VENDOR_TOS
    assert bac["endpoint"] == "https://tos-cn-shanghai.volces.com"
    assert bac["region"] == "cn-shanghai"          # 不加 oss- 前缀


def test_mgw_submit_idempotent_on_repeated_job(monkeypatch):
    """同源+目的已有任务（ImportJobRepeated 409）→ 幂等返回 job_name，不抛错。"""
    monkeypatch.setattr(engine_mgw, "get_mgw_client", lambda oid="": object())
    monkeypatch.setattr(engine_mgw.settings, "MGW_USER_ID", "uid")
    monkeypatch.setattr(engine_mgw.settings, "TRANSFER_OSS_ROLE", "role")
    monkeypatch.setattr(engine_mgw, "create_oss_source_address", lambda *a, **k: None)
    monkeypatch.setattr(engine_mgw, "verify_address", lambda *a, **k: True)
    monkeypatch.setattr(engine_mgw, "create_oss_dest_address", lambda *a, **k: None)
    def boom(*a, **k):
        raise RuntimeError("ImportJobRepeatedOnSameAddress Already has job on the same srcAddress")
    monkeypatch.setattr(engine_mgw, "create_job", boom)
    ref = engine_mgw.submit_cross_job(
        job_name="j1", src_bucket="a", src_prefix="p/", src_scheme="oss", src_region="cn-beijing",
        dest_bucket="b", dest_prefix="p/", dest_region="cn-beijing")
    assert ref == "j1"


def test_tos_oss_source_branch_default(monkeypatch):
    cap = {}
    fake = _FakeDms(cap)
    monkeypatch.setattr(engine_tos, "_api", lambda region: (fake, fake))
    engine_tos.create_migrate_task(
        task_name="job", src_bucket="a", src_prefix="p/", src_region="cn-hangzhou",
        src_access_id="ak", src_access_secret="sk", dest_bucket="b", dest_region="cn-shanghai")
    bac = cap["src_bac"]
    assert bac["vendor"] == engine_tos.SOURCE_VENDOR         # 默认阿里 OSS 源
    assert bac["endpoint"] == "https://oss-cn-hangzhou.aliyuncs.com"
    assert bac["region"] == "oss-cn-hangzhou"


# ── region 探测 helper ──────────────────────────────────────────────────────────

def test_region_from_endpoint():
    assert oss_tool.region_from_endpoint("https://oss-cn-beijing.aliyuncs.com") == "cn-beijing"
    assert oss_tool.region_from_endpoint("https://oss-cn-hangzhou-internal.aliyuncs.com") == "cn-hangzhou"


def test_tos_detect_bucket_region(monkeypatch):
    engine_tos._tos_region_cache.clear()
    class _B:
        def __init__(self, n, loc): self.name, self.location = n, loc
    class _R:
        buckets = [_B("data-tran", "cn-shanghai"), _B("umi-x", "cn-guangzhou")]
    class _C:
        def list_buckets(self): return _R()
    monkeypatch.setattr("utils.volcano_client_factory.get_tos_client", lambda: _C())
    assert engine_tos.detect_tos_bucket_region("data-tran", "cn-beijing") == "cn-shanghai"
    assert engine_tos.detect_tos_bucket_region("umi-x", "cn-beijing") == "cn-guangzhou"
    assert engine_tos.detect_tos_bucket_region("不存在的桶", "cn-beijing") == "cn-beijing"   # 回退默认


def test_tos_query_task_extracts_error(monkeypatch):
    class _Prog:
        transferred_bytes = 0; transferred_objects = 0
        failed_objects = 3; not_exist_object_count = 1
    class _R:
        task_status = "Failure"; task_progress = _Prog(); task_report = None
    class _Api:
        def query_data_migrate_task(self, req): return _R()
    monkeypatch.setattr(engine_tos, "_api", lambda region: (_Api(), _FakeDms({})))
    st = engine_tos.query_task(1, "cn-shanghai")
    assert st["status"] == "Failure"
    assert "3 个对象迁移失败" in st["error"] and "源端不存在" in st["error"]


# ── orchestrator 状态机 ────────────────────────────────────────────────────────

def test_orch_oss_run_to_completion(monkeypatch):
    plan = paths.build_plan("oss://a/data/", "oss://b/data/")
    job = orch.create_job_record(plan, same_name="skip", open_id="ou_1")
    assert job["stage"] == orch.STAGE_NEW and job["engine"] == "mgw"
    monkeypatch.setattr("tools.aliyun.oss.detect_bucket_region", lambda oid, b: "cn-beijing")
    monkeypatch.setattr(engine_mgw, "submit_cross_job", lambda **k: k["job_name"])
    monkeypatch.setattr(engine_mgw, "poll_job",
                        lambda name, **k: {"status": engine_mgw.STATUS_FINISHED, "bytes": 10, "objects": 2})
    job = orch.run_to_completion(job, poll_interval=0, max_polls=3)
    assert job["stage"] == orch.STAGE_DONE and job["objects"] == 2


def test_orch_tos_failure(monkeypatch):
    plan = paths.build_plan("tos://a/x/", "tos://b/x/")
    job = orch.create_job_record(plan, open_id="ou_1")
    monkeypatch.setattr(engine_tos, "submit_cross_job", lambda **k: "555")
    monkeypatch.setattr(engine_tos, "poll_job",
                        lambda ref, **k: {"status": engine_tos.STATUS_FAILURE, "error": "boom"})
    job = orch.run_to_completion(job, poll_interval=0, max_polls=3)
    assert job["stage"] == orch.STAGE_FAILED and "boom" in job["error"]


# ── fakes ──────────────────────────────────────────────────────────────────────

class _FakeDetail:
    pass


class _FakeModels:
    def AddressDetail(self):
        return _FakeDetail()


class _FakeDms:
    """同时充当 dms1 模块（构造 model）和 api（create/query）。"""
    def __init__(self, cap):
        self.cap = cap

    # model 构造器
    def BucketAccessConfigForCreateDataMigrateTaskInput(self, **kw):
        self.cap["src_bac"] = kw
        return kw

    def ObjectSourceConfigForCreateDataMigrateTaskInput(self, **kw):
        return kw

    def SourceForCreateDataMigrateTaskInput(self, **kw):
        return kw

    def TargetForCreateDataMigrateTaskInput(self, **kw):
        return kw

    def BasicConfigForCreateDataMigrateTaskInput(self, **kw):
        return kw

    def CreateDataMigrateTaskRequest(self, **kw):
        return kw

    def QueryDataMigrateTaskRequest(self, **kw):
        return kw

    # api
    def create_data_migrate_task(self, body):
        return type("R", (), {"task_id": 123})()
