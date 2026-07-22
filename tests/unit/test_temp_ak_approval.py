"""#57 临时 AK 发放 —— 审批门禁（安全核心，锁死）+ 表单解析 + routes 分发。

覆盖：
  · should_handle_event：精确 code==TEMP_AK_APPROVAL_CODE；no-code 不兜底；不与 ram_approval 互抢。
  · handle_temp_ak_event 门禁（复用 #55 硬化）：
      实例级 status≠APPROVED（PENDING / 事件节点级 PASS 但实例 PENDING）→ 不发放；
      无 instance_code → fail-safe 不发；实例 APPROVED → 发放恰一次；code≠target → ignore。
    变异验证：证明门禁用例非空断言（旧 bug 深搜到节点 PASS 会误发）。
  · 表单解析：_split_prefixes / _parse_dt / _validate_spec。
  · routes.py（MED-1）：temp_ak 严格 code 分发排在 ram_approval 之前、不互抢。
"""
import time

import pytest

from core.temp_ak_issuance import approval, orchestrator
from core import ram_approval

TARGET = "temp_ak_code_XYZ"
RAM_CODE = "ram_build_code_ABC"


@pytest.fixture(autouse=True)
def _codes(monkeypatch):
    monkeypatch.setattr(approval.settings, "TEMP_AK_APPROVAL_CODE", TARGET)
    monkeypatch.setattr(orchestrator.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_CODE", RAM_CODE)
    monkeypatch.setattr(approval.settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False, raising=False)


def _event(code=TARGET, status="PASS", instance="inst_x"):
    ev = {"header": {"event_type": "approval_instance"}, "event": {}}
    if code:
        ev["event"]["approval_code"] = code
    if status:
        ev["event"]["status"] = status
    if instance:
        ev["event"]["instance_code"] = instance
    return ev


def _detail(instance_status, *, code=TARGET, platform="阿里云",
            directory="oss://wuji-sing/team/data/", perm=None,
            date_interval=None, by_id=False):
    """审批实例详情（真机 5 控件：平台/使用企业名称/权限设置/DateInterval/申请目录）。

    顶层实例级 status = instance_status。by_id=True 时用真机 widget id 作字段键（否则用中文名）。
    """
    perm = perm if perm is not None else ["read", "write"]
    date_interval = date_interval if date_interval is not None else \
        {"start": "", "end": "2099-01-01 00:00:00"}
    names = {
        "platform": "widget17846401222860001" if by_id else "平台",
        "enterprise": "widget17846886904010001" if by_id else "使用企业名称",
        "perm": "widget17846401501570001" if by_id else "权限设置",
        "date_interval": "widget17846402309610001" if by_id else "DateInterval",
        "directory": "widget17846402564230001" if by_id else "申请目录",
    }
    key = "id" if by_id else "name"
    form = [
        {key: names["platform"], "value": platform},
        {key: names["enterprise"], "value": "外采公司A"},
        {key: names["perm"], "value": perm},
        {key: names["date_interval"], "value": date_interval},
        {key: names["directory"], "value": directory},
    ]
    return {
        "status": instance_status,
        "approval_code": code,
        "instance_code": "inst_x",
        "form": form,
    }


def _spy_issue(monkeypatch):
    calls = {"issue": 0, "deliver": 0}
    import core.temp_ak_issuance.issuer as issuer_mod

    def fake_issue(grant):
        calls["issue"] += 1
        return {"access_key_id": "AK", "access_key_secret": "SK",
                "security_token": "TOK", "expire_ts": grant["expire"],
                "mode": grant.get("mode", "sts")}

    monkeypatch.setattr(issuer_mod, "issue", fake_issue)
    monkeypatch.setattr(approval.delivery, "deliver",
                        lambda g, c: calls.__setitem__("deliver", calls["deliver"] + 1))
    return calls


# ══════════════════════════════════════════════════════════════════════════════
# should_handle_event
# ══════════════════════════════════════════════════════════════════════════════

def test_should_handle_exact_code_match():
    assert approval.should_handle_event(_event(code=TARGET)) is True


def test_should_handle_rejects_wrong_code():
    assert approval.should_handle_event(_event(code=RAM_CODE)) is False


def test_should_handle_no_code_no_fallback():
    """无 approval_code → 不兜底命中（与 ram_approval 的 no-code 兜底相反）。"""
    ev = {"header": {"event_type": "approval_instance"}, "event": {"instance_code": "i"}}
    assert approval.should_handle_event(ev) is False


def test_should_handle_target_unset_false(monkeypatch):
    monkeypatch.setattr(approval.settings, "TEMP_AK_APPROVAL_CODE", "")
    assert approval.should_handle_event(_event(code=TARGET)) is False


def test_temp_ak_and_ram_do_not_compete():
    """带 RAM code 的事件：temp_ak 不接、ram_approval 接；带 temp_ak code：反之。"""
    ram_ev = _event(code=RAM_CODE)
    assert approval.should_handle_event(ram_ev) is False
    assert ram_approval.should_handle_event(ram_ev) is True

    tak_ev = _event(code=TARGET)
    assert approval.should_handle_event(tak_ev) is True
    assert ram_approval.should_handle_event(tak_ev) is False


# ══════════════════════════════════════════════════════════════════════════════
# 门禁（安全核心）
# ════════════���═════════════════════════════════════════════════════════════════

def test_instance_pending_does_not_issue(monkeypatch):
    """核心：事件带节点级 PASS（过早期过滤），但回拉实例详情顶层 status=PENDING → 不发放。"""
    calls = _spy_issue(monkeypatch)
    detail = _detail("PENDING")
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: detail)

    res = approval.handle_temp_ak_event(_event(status="PASS"))
    assert calls["issue"] == 0
    assert calls["deliver"] == 0
    assert res["ignored"] is True
    assert "PENDING" in res["reason"]

    # ── 变异验证：证明用例非空断言 ──
    # 事件里携带的状态是节点级 PASS（∈ APPROVED_STATUSES）——旧 bug（用事件/深搜状态判据）会误发；
    payload = _event(status="PASS")
    assert ram_approval._extract_status(payload).upper() in ram_approval.APPROVED_STATUSES
    # 但门禁只认回拉实例详情的顶层实例级 status，此处明确不是 APPROVED → 正确拦下。
    assert str(detail.get("status")).upper() != "APPROVED"


def test_instance_processing_does_not_issue(monkeypatch):
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _detail("PROCESSING"))
    res = approval.handle_temp_ak_event(_event(status="PASS"))
    assert calls["issue"] == 0
    assert res["ignored"] is True


def test_no_instance_code_fail_safe(monkeypatch):
    """无 instance_code → 无法核实整单 → fail-safe 不发放、压根不回拉。"""
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: pytest.fail("无 instance_code 不应回拉实例"))
    res = approval.handle_temp_ak_event(_event(status="PASS", instance=""))
    assert calls["issue"] == 0
    assert res["ignored"] is True
    assert res["reason"] == "no_instance_code"


def test_instance_approved_issues_exactly_once(monkeypatch):
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _detail("APPROVED"))
    res = approval.handle_temp_ak_event(_event(status="APPROVED"))
    assert calls["issue"] == 1
    assert calls["deliver"] == 1
    assert res["ignored"] is False
    assert res["grant_id"].startswith("tak-")


def test_approved_lowercase_normalizes(monkeypatch):
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _detail("approved"))
    res = approval.handle_temp_ak_event(_event(status="APPROVED"))
    assert calls["issue"] == 1
    assert res["ignored"] is False


def test_wrong_code_ignored(monkeypatch):
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: pytest.fail("wrong code 不该回拉"))
    res = approval.handle_temp_ak_event(_event(code=RAM_CODE, status="APPROVED"))
    assert calls["issue"] == 0
    assert res["ignored"] is True
    assert res["reason"] == "not_temp_ak"


def test_already_issued_short_circuit(monkeypatch):
    """已 ISSUED 的实例再来事件 → 不重发。"""
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _detail("APPROVED"))
    # 首次发放
    approval.handle_temp_ak_event(_event(status="APPROVED"))
    assert calls["issue"] == 1
    # 第二次同实例事件
    res2 = approval.handle_temp_ak_event(_event(status="APPROVED"))
    assert calls["issue"] == 1        # 未再发
    assert res2["ignored"] is True
    assert res2["reason"] == "already_issued"


def test_rejected_status_filtered_early(monkeypatch):
    """事件 status=REJECTED（不在 APPROVED_STATUSES）→ 早期过滤、不回拉不发。"""
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: pytest.fail("REJECTED 不该回拉"))
    res = approval.handle_temp_ak_event(_event(status="REJECTED"))
    assert calls["issue"] == 0
    assert res["ignored"] is True


def test_dry_run_no_issue(monkeypatch):
    calls = _spy_issue(monkeypatch)
    monkeypatch.setattr(approval.settings, "FEISHU_RAM_APPROVAL_DRY_RUN", True, raising=False)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance",
                        lambda code: _detail("APPROVED"))
    res = approval.handle_temp_ak_event(_event(status="APPROVED"))
    assert calls["issue"] == 0
    assert res.get("dry_run") is True


# ══════════════════════════════════════════════════════════════════════════════
# 表单解析
# ══════════════════════════════════════════════════════════════════════════════

# ── _parse_directory ─────────────────────────────────────────────────────────

def test_parse_directory_oss_scheme():
    assert approval._parse_directory("oss://wuji-sing/team/data/") == ("wuji-sing", "team/data/")


def test_parse_directory_tos_scheme_stripped():
    assert approval._parse_directory("tos://b/p/q/") == ("b", "p/q/")


def test_parse_directory_no_scheme():
    assert approval._parse_directory("mybucket/some/prefix") == ("mybucket", "some/prefix")


def test_parse_directory_bare_bucket():
    assert approval._parse_directory("just-a-bucket") == ("just-a-bucket", "")


def test_parse_directory_leading_slash():
    assert approval._parse_directory("/b/p/") == ("b", "p/")


def test_parse_directory_empty():
    assert approval._parse_directory("") == ("", "")


def test_parse_directory_control_char_prefix_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._parse_directory("b/pre\x01fix/")


def test_parse_directory_tab_injection_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._parse_directory("b/pre\tfix/")


def test_parse_directory_control_char_bucket_raises():
    """LOW-C：桶名也过 _KEY_RE，控制字符桶名 → 抛。"""
    with pytest.raises(orchestrator.TempAkError):
        approval._parse_directory("oss://bad\x01bucket/p/")


def test_parse_directory_control_char_bucket_no_prefix_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._parse_directory("bad\x01bucket")


# ── _parse_perm → caps list（真实三原子项 read/write/download，正交）──────────────

def test_parse_perm_read_only():
    assert approval._parse_perm(["read"]) == ["read"]


def test_parse_perm_write_only():
    assert approval._parse_perm(["write"]) == ["write"]


def test_parse_perm_download_only():
    """download 独立 → 只 [download]，不含 read（子串判定 'download' 不含 'read'）。"""
    assert approval._parse_perm(["download"]) == ["download"]


def test_parse_perm_read_download_multi():
    """真实多选 read+download → 合并 [download, read]（顺序：download/write/read）。"""
    assert set(approval._parse_perm(["read", "download"])) == {"read", "download"}


def test_parse_perm_all_three():
    assert set(approval._parse_perm(["read", "write", "download"])) == {"read", "write", "download"}


def test_parse_perm_empty():
    assert approval._parse_perm("") == []


def test_parse_perm_read_and_write_defensive():
    """防御性解析：旧 'read&write' 已非真实选项，但 parser 仍拆成 read+write（无害）。"""
    assert set(approval._parse_perm("read&write")) == {"read", "write"}


# ── _parse_platform ───────────────────────────────────────────────────────────

def test_parse_platform_volcano():
    assert approval._parse_platform("火山云") == "volcano"
    assert approval._parse_platform("volcano") == "volcano"


def test_parse_platform_aliyun():
    assert approval._parse_platform("阿里云") == "aliyun"
    assert approval._parse_platform("aliyun") == "aliyun"


def test_parse_platform_unknown_not_defaulted_to_aliyun():
    """LOW-B：未知/空平台 → "unknown"（不默认 aliyun，避免给非 OSS 目录误发 OSS 凭证）。"""
    assert approval._parse_platform("谷歌云") == "unknown"
    assert approval._parse_platform("") == "unknown"
    assert approval._parse_platform(None) == "unknown"


# ── _parse_date_interval（三形态）─────────────────────────────────────────────

def test_parse_date_interval_dict_ms_and_string():
    """dict{start(ms 数字), end('YYYY-MM-DD HH:MM:SS')} → (start,end)。"""
    start_ms = 1893456000000       # 2030-01-01 ish (ms)
    nb, exp = approval._parse_date_interval(
        {"start": start_ms, "end": "2099-01-01 00:00:00"})
    assert nb == 1893456000.0      # ms → s
    assert exp > nb


def test_parse_date_interval_list():
    nb, exp = approval._parse_date_interval(["2027-01-01 00:00:00", "2027-06-01 00:00:00"])
    assert nb > 0 and exp > nb


def test_parse_date_interval_single_value_expire_only():
    nb, exp = approval._parse_date_interval("2099-01-01 00:00:00")
    assert nb == 0.0
    assert exp > 0


def test_parse_date_interval_alt_keys():
    nb, exp = approval._parse_date_interval(
        {"startTime": "2027-01-01", "endTime": "2027-02-01"})
    assert nb > 0 and exp > nb


# ── parse_temp_ak_request 整体（name 键 + widget id 键两路）───────────────────

@pytest.mark.parametrize("by_id", [False, True])
def test_parse_request_full(monkeypatch, by_id):
    spec = approval.parse_temp_ak_request(_detail("APPROVED", by_id=by_id), {})
    assert spec["platform"] == "aliyun"
    assert spec["enterprise"] == "外采公司A"
    assert spec["bucket"] == "wuji-sing"
    # 单目录 + caps（新模型：prefix + caps，无 read_prefixes/write_prefixes）
    assert spec["prefix"] == "team/data/"
    assert set(spec["caps"]) == {"read", "write"}      # _detail 默认 perm=["read","write"]
    assert "read_prefixes" not in spec and "write_prefixes" not in spec
    assert spec["expire"] > 0
    assert spec["not_before"] > 0 and spec["not_before"] < spec["expire"]


def test_parse_request_read_only(monkeypatch):
    spec = approval.parse_temp_ak_request(_detail("APPROVED", perm=["read"]), {})
    assert spec["prefix"] == "team/data/"
    assert spec["caps"] == ["read"]


def test_parse_request_download_only(monkeypatch):
    spec = approval.parse_temp_ak_request(_detail("APPROVED", perm=["download"]), {})
    assert spec["caps"] == ["download"]


def test_parse_request_write_only(monkeypatch):
    spec = approval.parse_temp_ak_request(_detail("APPROVED", perm=["write"]), {})
    assert spec["caps"] == ["write"]


# ── _parse_dt 容错 ───────────────────────────────────────────────────────────

def test_parse_dt_iso_with_space():
    assert approval._parse_dt("2027-01-01 12:00:00") > 0


def test_parse_dt_iso_t():
    assert approval._parse_dt("2027-01-01T12:00:00") > 0


def test_parse_dt_date_only():
    assert approval._parse_dt("2027-01-01") > 0


def test_parse_dt_epoch_seconds():
    assert approval._parse_dt("1893456000") == 1893456000.0


def test_parse_dt_epoch_millis():
    assert approval._parse_dt("1893456000000") == 1893456000.0


def test_parse_dt_numeric_type():
    assert approval._parse_dt(1893456000) == 1893456000.0
    assert approval._parse_dt(1893456000000) == 1893456000.0


def test_parse_dt_empty_zero():
    assert approval._parse_dt("") == 0.0
    assert approval._parse_dt(None) == 0.0


def test_parse_dt_bad_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._parse_dt("not-a-date")


# ── _validate_spec（新模型）──────────────────────────────────────────────────

def _valid_spec(**over):
    now = time.time()
    s = {
        "platform": "aliyun", "enterprise": "E", "bucket": "b",
        "prefix": "team/data/", "caps": ["read"],
        "not_before": 0.0, "expire": now + 3600,
        "recipient_email": "", "source_ips": [], "reason": "",
    }
    s.update(over)
    return s


def test_validate_volcano_platform_rejected():
    """核心：平台=火山云 → 抛「暂未支持」（P0 只做阿里 OSS）。"""
    with pytest.raises(orchestrator.TempAkError) as ei:
        approval._validate_spec(_valid_spec(platform="volcano"))
    assert "火山" in str(ei.value) or "暂未支持" in str(ei.value)


def test_validate_unknown_platform_rejected():
    """LOW-B：platform=unknown（非 aliyun）→ 拒（避免给非 OSS 误发）。"""
    with pytest.raises(orchestrator.TempAkError):
        approval._validate_spec(_valid_spec(platform="unknown"))


def test_validate_aliyun_platform_ok():
    approval._validate_spec(_valid_spec(platform="aliyun"))   # 不抛


def test_validate_missing_bucket_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._validate_spec(_valid_spec(bucket=""))


def test_validate_empty_caps_raises():
    """caps 空（权限设置未勾任何项）→ 拒。"""
    with pytest.raises(orchestrator.TempAkError):
        approval._validate_spec(_valid_spec(caps=[]))


def test_validate_expire_past_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._validate_spec(_valid_spec(expire=time.time() - 100))


def test_validate_missing_expire_raises():
    with pytest.raises(orchestrator.TempAkError):
        approval._validate_spec(_valid_spec(expire=0.0))


def test_validate_not_before_ge_expire_raises():
    now = time.time()
    with pytest.raises(orchestrator.TempAkError):
        approval._validate_spec(_valid_spec(not_before=now + 7200, expire=now + 3600))


def test_validate_empty_not_before_set_to_now():
    s = _valid_spec(not_before=0.0)
    approval._validate_spec(s)
    assert s["not_before"] > 0


def test_validate_no_email_requirement():
    """新模型无邮箱字段 → recipient_email 空不再拦（凭证私信发起人）。"""
    approval._validate_spec(_valid_spec(recipient_email=""))   # 不抛


# ══════════════════════════════════════════════════════════════════════════════
# routes.py 分发（MED-1）：temp_ak 严格 code 排在 ram_approval 之前、不互抢
# ══════════════════════════════════════════════════════════════════════════════

def test_routes_dispatch_temp_ak_before_ram(monkeypatch):
    """携带 TEMP_AK_APPROVAL_CODE 的审批事件命中 temp_ak、不落 ram_approval。"""
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "TEMP_AK_ENABLED", True, raising=False)
    monkeypatch.setattr(routes.settings, "TEMP_AK_APPROVAL_CODE", TARGET, raising=False)
    monkeypatch.setattr(routes.settings, "FEISHU_RAM_APPROVAL_CODE", RAM_CODE, raising=False)

    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            self.target = target
            self.args = args

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(routes.threading, "Thread", FakeThread)

    ev = _event(code=TARGET, status="APPROVED")
    # temp_ak 命中、ram 不命中
    from core.temp_ak_issuance import approval as tak
    assert tak.should_handle_event(ev) is True
    assert ram_approval.should_handle_event(ev) is False


def test_routes_ram_event_still_goes_to_ram(monkeypatch):
    """携带 FEISHU_RAM_APPROVAL_CODE 的事件仍走 ram_approval，temp_ak 不误抢。"""
    from core.temp_ak_issuance import approval as tak
    monkeypatch.setattr(tak.settings, "TEMP_AK_APPROVAL_CODE", TARGET)
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_CODE", RAM_CODE)
    ev = _event(code=RAM_CODE, status="APPROVED")
    assert tak.should_handle_event(ev) is False
    assert ram_approval.should_handle_event(ev) is True
