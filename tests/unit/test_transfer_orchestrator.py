"""core/transfer/orchestrator.py + cards.py 测试。

orchestrator：用 fakeredis（conftest 已注入）+ monkeypatch engine_mgw，验证状态机推进、
幂等、审批阈值判断；cards：锁住卡片形状关键字段。
"""
import json
import pytest


@pytest.fixture
def bmap(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "TRANSFER_BUCKET_MAP_RAW", json.dumps({
        "tos://src-bucket": "dst-oss-bucket",
    }))
    monkeypatch.setattr(settings, "TRANSFER_APPROVAL_TB", 1.0)
    monkeypatch.setattr(settings, "TRANSFER_OVERWRITE_DEFAULT", "no")
    return settings


# ── 计划 / 阈值 ───────────────────────────────────────────────────────────────

def test_make_plan_uses_settings_map(bmap):
    from core.transfer import orchestrator
    p = orchestrator.make_plan("tos://src-bucket/a/")
    assert p.dest.bucket == "dst-oss-bucket"


def test_needs_approval_threshold(bmap):
    from core.transfer import orchestrator
    assert orchestrator.needs_approval(2 * 1024 ** 4)         # 2TB > 1TB
    assert not orchestrator.needs_approval(500 * 1024 ** 3)   # 0.5TB


# ── 任务记录 / 幂等 ───────────────────────────────────────────────────────────

def test_create_job_record_idempotent(bmap):
    from core.transfer import orchestrator
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    j1 = orchestrator.create_job_record(plan, open_id="ou_1", bytes_total=100)
    j2 = orchestrator.create_job_record(plan, open_id="ou_2", bytes_total=999)
    assert j1["job_id"] == j2["job_id"]      # 同源+目的+当天 → 同 job
    assert j2["created_by"] == "ou_1"        # 命中旧记录，不覆盖


def test_get_job_roundtrip(bmap):
    from core.transfer import orchestrator
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    job = orchestrator.create_job_record(plan)
    got = orchestrator.get_job(job["job_id"])
    assert got["source"] == "tos://src-bucket/a/"
    assert got["stage"] == orchestrator.STAGE_NEW


# ── 状态机：start_cross + poll_once ──────────────────────────────────────────

@pytest.fixture
def fake_engine(monkeypatch):
    from core.transfer import engine_mgw
    calls = {"submit": 0, "status": "IMPORT_JOB_FINISHED", "bytes": 4096, "objects": 7}
    monkeypatch.setattr(engine_mgw, "submit_cross_job",
                        lambda **k: calls.__setitem__("submit", calls["submit"] + 1) or k["job_name"])
    monkeypatch.setattr(engine_mgw, "poll_job",
                        lambda name, open_id="": {"status": calls["status"],
                                                  "bytes": calls["bytes"],
                                                  "objects": calls["objects"], "error": ""})
    return calls


def test_start_cross_success(bmap, fake_engine):
    from core.transfer import orchestrator
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    job = orchestrator.create_job_record(plan)
    job = orchestrator.start_cross(job)
    assert job["stage"] == orchestrator.STAGE_CROSSING
    assert fake_engine["submit"] == 1
    assert job["cross_job_name"] == job["job_id"]


def test_poll_once_to_done(bmap, fake_engine):
    from core.transfer import orchestrator
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    job = orchestrator.start_cross(orchestrator.create_job_record(plan))
    job = orchestrator.poll_once(job)
    assert job["stage"] == orchestrator.STAGE_DONE
    assert job["bytes_total"] == 4096
    assert job["objects_total"] == 7


def test_poll_once_interrupted_fails(bmap, fake_engine):
    from core.transfer import orchestrator
    fake_engine["status"] = "IMPORT_JOB_INTERRUPTED"
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    job = orchestrator.start_cross(orchestrator.create_job_record(plan))
    job = orchestrator.poll_once(job)
    assert job["stage"] == orchestrator.STAGE_FAILED


def test_start_cross_tos_mig_stores_task_id(bmap, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "TRANSFER_BUCKET_MAP_RAW",
                        json.dumps({"oss://b": "t"}))
    from core.transfer import orchestrator, engine_tos

    seen = {}

    def fake_submit(**kwargs):
        seen.update(kwargs)
        return "552548"

    monkeypatch.setattr(engine_tos, "submit_cross_job", fake_submit)

    plan = orchestrator.make_plan("oss://b/a/")
    job = orchestrator.start_cross(orchestrator.create_job_record(plan))

    assert job["stage"] == orchestrator.STAGE_CROSSING
    assert job["cross_job_name"] == "552548"
    assert seen["src_bucket"] == "b"
    assert seen["dest_bucket"] == "t"
    assert seen["overwrite_mode"] == "skip"


# ── 卡片形状 ──────────────────────────────────────────────────────────────────

def _sample_job():
    return {"job_id": "tr-abc", "source": "tos://s/a/", "dest": "oss://d/a/",
            "direction": "tos→oss", "engine": "mgw", "overwrite_mode": "no",
            "stage": "NEW", "bytes_total": 1024, "objects_total": 3, "error": ""}


def test_confirm_card_is_form_v2():
    from core.transfer.cards import confirm_card
    c = confirm_card(_sample_job(), need_approval=False)
    assert c["schema"] == "2.0"
    form = c["body"]["elements"][-1]
    assert form["tag"] == "form"
    submit = form["elements"][-1]
    assert submit["behaviors"][0]["value"]["action"] == "confirm_transfer"
    assert submit["behaviors"][0]["value"]["job_id"] == "tr-abc"


def test_confirm_card_approval_orange():
    from core.transfer.cards import confirm_card
    c = confirm_card(_sample_job(), need_approval=True)
    assert c["header"]["template"] == "orange"


def test_confirm_card_has_same_name_policy_options():
    """????????????????????????"""
    from core.transfer.cards import confirm_card
    c = confirm_card(_sample_job(), need_approval=False)
    form = c["body"]["elements"][-1]
    selector = next(e for e in form["elements"] if e.get("name") == "same_name_policy")
    assert selector["tag"] == "select_static"
    assert selector["initial_option"] == "skip"
    assert [o["value"] for o in selector["options"]] == ["skip", "overwrite"]
    blob = json.dumps(c, ensure_ascii=False)
    assert "????" not in blob and "????" not in blob

def test_entry_card_has_source_dest_inputs():
    from core.transfer.cards import entry_card
    form = entry_card()["body"]["elements"][1]
    names = {e["name"] for e in form["elements"] if e.get("tag") == "input"}
    assert names == {"source", "dest"}
    submit = form["elements"][-1]
    assert submit["behaviors"][0]["value"]["action"] == "submit_transfer"


def test_same_name_policy_maps_to_engine_modes(bmap):
    """????????????????????? overwrite ????"""
    from core.transfer import orchestrator
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    job = orchestrator.create_job_record(plan, same_name_policy="skip")
    assert job["same_name_policy"] == "skip"
    assert job["transfer_mode"] == "lastmodified"
    assert job["overwrite_mode"] == "always"

    job = orchestrator.set_same_name_policy(job, "overwrite")
    assert job["same_name_policy"] == "overwrite"
    assert job["transfer_mode"] == "all"
    assert job["overwrite_mode"] == "always"

def test_result_card_success_green():
    from core.transfer.cards import result_card
    j = _sample_job(); j["stage"] = "DONE"
    c = result_card(j)
    assert c["header"]["template"] == "green"


def test_result_card_shows_time_and_path():
    """终态卡须含发起/完成时间、耗时、路径、迁移量、发起人。"""
    from core.transfer.cards import result_card
    j = _sample_job()
    j.update(stage="DONE", created_ts=1_700_000_000, finished_ts=1_700_000_133,
             created_by="ou_user", bytes_total=4532542, objects_total=2)
    blob = json.dumps(result_card(j), ensure_ascii=False)
    for kw in ("发起时间", "完成时间", "耗时", "迁移量", "对象数", "ou_user",
               j["source"], j["dest"]):
        assert kw in blob, f"终态卡缺 {kw}"
    assert "2分13秒" in blob          # 133 秒 → 2分13秒


def test_fmt_duration_and_ts():
    from core.transfer.orchestrator import fmt_duration, fmt_ts
    assert fmt_duration(0, 0) == "-"
    assert fmt_duration(100, 100 + 45) == "45秒"
    assert fmt_duration(100, 100 + 133) == "2分13秒"
    assert fmt_duration(100, 100 + 3725) == "1小时2分5秒"
    assert fmt_ts(0) == "-"
    assert len(fmt_ts(1_700_000_000)) == 19   # YYYY-MM-DD HH:MM:SS


def test_result_card_failure_has_retry():
    from core.transfer.cards import result_card
    j = _sample_job(); j["stage"] = "FAILED"; j["error"] = "boom"
    c = result_card(j)
    assert c["header"]["template"] == "red"
    blob = json.dumps(c, ensure_ascii=False)
    assert "retry_transfer" in blob and "boom" in blob


# ── engine_mgw 建址幂等（重试不撞名）─────────────────────────────────────────

class _FakeModels:
    """最小 SDK models 替身，构造对象只存属性。"""
    class AddressDetail:
        pass

    class CreateAddressInfo:
        def __init__(self, name=None, address_detail=None):
            self.name, self.address_detail = name, address_detail

    class CreateAddressRequest:
        def __init__(self, import_address=None):
            self.import_address = import_address


def test_put_address_idempotent_on_exists(monkeypatch):
    from core.transfer import engine_mgw
    monkeypatch.setattr(engine_mgw, "_models", lambda: _FakeModels)

    class _Client:
        def __init__(self):
            self.calls = 0
        def create_address(self, uid, req):
            self.calls += 1
            raise RuntimeError("BackendAddressAlreadyExist: dup")

    c = _Client()
    # 不应抛错（已存在被吞掉）
    engine_mgw._put_address(c, "tr-x-src", _FakeModels.AddressDetail())
    assert c.calls == 1


def test_put_address_reraises_other_errors(monkeypatch):
    from core.transfer import engine_mgw
    monkeypatch.setattr(engine_mgw, "_models", lambda: _FakeModels)

    class _Client:
        def create_address(self, uid, req):
            raise RuntimeError("InvalidAccessKey: bad creds")

    with pytest.raises(RuntimeError, match="InvalidAccessKey"):
        engine_mgw._put_address(_Client(), "tr-x-src", _FakeModels.AddressDetail())


# ── wuji_il 校准：内网域名 / 源域名格式 ───────────────────────────────────────

def test_oss_domain_internal_vs_public():
    from core.transfer.engine_mgw import _oss_domain
    assert _oss_domain("cn-hangzhou", True) == "oss-cn-hangzhou-internal.aliyuncs.com"
    assert _oss_domain("cn-hangzhou", False) == "oss-cn-hangzhou.aliyuncs.com"


def test_tos_domain_shanghai():
    from core.transfer.engine_mgw import _tos_domain
    assert _tos_domain("cn-shanghai") == "tos-s3-cn-shanghai.volces.com"


def test_oss_dest_address_carries_internal_domain(monkeypatch):
    from core.transfer import engine_mgw
    monkeypatch.setattr(engine_mgw, "_models", lambda: _FakeModels)
    captured = {}

    class _Client:
        def create_address(self, uid, req):
            captured["detail"] = req.import_address.address_detail

    engine_mgw.create_oss_dest_address(
        _Client(), "tr-x-dst", bucket="wuji-bucket-hangzhou",
        prefix="p/", region="cn-hangzhou", role="oss-import-x", internal=True)
    d = captured["detail"]
    assert d.region_id == "oss-cn-hangzhou"
    assert d.domain == "oss-cn-hangzhou-internal.aliyuncs.com"
    assert d.role == "oss-import-x"


def test_start_cross_uses_transfer_tos_creds_fallback(bmap, fake_engine, monkeypatch):
    """TRANSFER_TOS_* 留空时回退 TOS_*；提供时优先。"""
    from config.settings import settings
    captured = {}
    from core.transfer import engine_mgw
    monkeypatch.setattr(engine_mgw, "submit_cross_job",
                        lambda **k: captured.update(k) or k["job_name"])
    monkeypatch.setattr(settings, "TOS_ACCESS_KEY", "fallback-ak")
    monkeypatch.setattr(settings, "TRANSFER_TOS_ACCESS_KEY", "")
    monkeypatch.setattr(settings, "TRANSFER_TOS_REGION", "cn-shanghai")

    from core.transfer import orchestrator
    plan = orchestrator.make_plan("tos://src-bucket/a/")
    orchestrator.start_cross(orchestrator.create_job_record(plan))
    assert captured["src_access_id"] == "fallback-ak"     # 回退
    assert captured["src_region"] == "cn-shanghai"         # 专用区域优先


# ── 真实 SDK 契约：构造请求模型的 kwargs 必须存在（防字段名漂移）──────────────
# 这组用例会在真实 SDK 装了时跑；没装则跳过（CI 不强制装迁移 SDK）。

mgw_models = pytest.importorskip("alibabacloud_hcs_mgw20240626.models")


def test_sdk_create_address_request_kwarg():
    d = mgw_models.AddressDetail(address_type="tos", access_id="x", access_secret="y",
                                 bucket="b", prefix="p/", domain="tos-s3-cn-shanghai.volces.com")
    info = mgw_models.CreateAddressInfo(name="n", address_detail=d)
    req = mgw_models.CreateAddressRequest(import_address=info)   # 我代码用的 kwarg
    assert req.import_address is info


def test_sdk_job_request_kwargs():
    ji = mgw_models.CreateJobInfo(name="n", transfer_mode="lastmodified", overwrite_mode="always",
                                  src_address="s", dest_address="d")
    assert mgw_models.CreateJobRequest(import_job=ji).import_job is ji
    uj = mgw_models.UpdateJobInfo(status="IMPORT_JOB_LAUNCHING")
    assert mgw_models.UpdateJobRequest(import_job=uj).import_job is uj


def test_sdk_verify_response_nested_field():
    """verify 返回体真实路径 body.verify_address_response.{status,error_message}。"""
    import inspect
    body = [p for p in inspect.signature(
        mgw_models.VerifyAddressResponseBody.__init__).parameters if p != "self"]
    assert "verify_address_response" in body
    inner = [p for p in inspect.signature(
        mgw_models.VerifyAddressResp.__init__).parameters if p != "self"]
    assert "status" in inner and "error_message" in inner


def test_verify_address_polls_until_available(monkeypatch):
    """status 异步：前两次空，第三次 available → verify_address 返回 True。"""
    from core.transfer import engine_mgw
    seq = iter([("", ""), ("", ""), ("available", "")])
    monkeypatch.setattr(engine_mgw, "_verify_once", lambda c, n: next(seq))
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    assert engine_mgw.verify_address(object(), "n", retries=5, interval=0)


def test_verify_address_raises_on_failed(monkeypatch):
    from core.transfer import engine_mgw
    monkeypatch.setattr(engine_mgw, "_verify_once", lambda c, n: ("failed", "bad creds"))
    import pytest as _p
    with _p.raises(engine_mgw.MgwError, match="bad creds"):
        engine_mgw.verify_address(object(), "n", retries=2, interval=0)


def test_sdk_address_detail_has_our_fields():
    fields = set(mgw_models.AddressDetail().__dict__) | set(
        p for p in __import__("inspect").signature(
            mgw_models.AddressDetail.__init__).parameters if p != "self")
    for f in ("access_id", "access_secret", "address_type", "bucket",
              "prefix", "region_id", "role", "domain"):
        assert f in fields, f"AddressDetail 缺字段 {f}"
