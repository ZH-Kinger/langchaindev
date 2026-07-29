"""#64 RAM 建号失败重复评论修复 —— 终态标记 + "同一错误只播报一次"闸门。

真机 BC9CE2EB：飞书为**一个审批实例**推 7 个事件，3 个被"正在处理中"锁拦掉、3 个真跑完整建号
流程（每次都打云 API、每次都贴一条一模一样的密码策略错误评论）→ 申请人看到 3 条重复评论。

两道机制，各管一层，本文件都钉死：
  · **终态标记** `error_terminal`（`_is_terminal_error` + `_is_instance_done`）——确定性失败
    （密码不合策略等）后，**后续事件在入口就短路，连云 API 都不再打**。这是省钱省副作用那层。
  · **播报闸门** `_claim_failure_notice`——同实例 + 同错误文本只播报一次。这是兜底那层，覆盖
    **所有**失败（含瞬时失败重试三次、含将来新增的重入路径），与是否落库无关。

刻意收窄过的地方（回归钉子，别手滑加回去）：`_TERMINAL_ERROR_MARKERS` 曾含
`invalidparameter` / `parameterinvalid` / `malformed`——宽泛子串会把网关 5xx、限流这类
**瞬时**错误永久标成终态、彻底失去自愈。本文件用真实形态的瞬时错误文本反向断言它们不再命中。
"""
import json

import pytest

from core import ram_approval


# ── 真机形态的异常 ────────────────────────────────────────────────────────────

class FakeApiException(Exception):
    """火山 IAM SDK 的 ApiException 形态：str(exc) 是很长的 HTTP dump，body 里才是结构化错误。

    `_is_terminal_error` 对 `str(exc).lower()` 做子串匹配，所以关键词得出现在 dump 里
    （真机就是这样：`InvalidPassword` 在 `HTTP response body` 那段）。
    """
    def __init__(self, body: str, status: int = 400, reason: str = "Bad Request"):
        self.body = body
        self.status = status
        super().__init__(
            f"({status})\n"
            f"Reason: {reason}\n"
            "HTTP response headers: HTTPHeaderDict({'Content-Type': 'application/json', "
            "'X-Top-Request-Id': 'abc123', 'Server': 'nginx'})\n"
            f"HTTP response body: {body}\n"
        )


_PWD_BODY = json.dumps({"Error": {
    "Code": "InvalidPassword",
    "Message": "Given password does not satisfy required password policy:[******]"}})


def _pwd_exc():
    """真机 BC9CE2EB 那单的确定性失败（CreateLoginProfile 密码不合策略）。"""
    return FakeApiException(_PWD_BODY)


def _timeout_exc():
    """瞬时失败：重试有意义，绝不能被标成终态。"""
    return TimeoutError("HTTPSConnectionPool(host='iam.volcengineapi.com', port=443): "
                        "Read timed out. (read timeout=15)")


# ══════════════════════════════════════════════════════════════════════════════
# 1. _is_terminal_error
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "InvalidPassword",                                   # 阿里/火山错误码
    "invalidpassword",                                   # 已是小写
    "InVaLiDpAsSwOrD: whatever",                         # 大小写混合（匹配前 .lower()）
    "Given password does not satisfy required Password Policy:[***]",
    "登录密码不符合密码策略要求",                          # 中文那条
    "InvalidLoginName: login name contains illegal char",
    "invalidloginname",
])
def test_terminal_markers_hit(text):
    assert ram_approval._is_terminal_error(Exception(text)) is True


def test_terminal_hits_real_machine_api_exception():
    """真机形态：关键词埋在长 HTTP dump 里也要认出来。"""
    exc = _pwd_exc()
    assert "InvalidPassword" in str(exc)
    assert ram_approval._is_terminal_error(exc) is True


@pytest.mark.parametrize("text", [
    # ↓↓↓ 本轮**收窄**掉的三个宽泛词。它们出现在真实的瞬时错误文本里，
    #     若还当终态 → 网关抖动一次就永久放弃自愈、申请人只能重开审批。
    "(500) gateway timeout, malformed upstream response",
    "InvalidParameter.Throttling",
    "ParameterInvalid: transient backend error, please retry",
    "(502) Bad Gateway: malformed response from upstream",
])
def test_removed_broad_markers_are_retryable(text):
    """**回归钉子**：这些必须判「可重试」。谁把 invalidparameter/malformed 加回 markers，这里就红。"""
    assert ram_approval._is_terminal_error(Exception(text)) is False


def test_markers_list_stays_narrow():
    """直接钉住 marker 清单本身：只允许这四个窄而明确的。"""
    assert set(ram_approval._TERMINAL_ERROR_MARKERS) == {
        "invalidpassword", "password policy", "密码不符合", "invalidloginname"}
    for banned in ("invalidparameter", "parameterinvalid", "malformed"):
        assert banned not in ram_approval._TERMINAL_ERROR_MARKERS


@pytest.mark.parametrize("exc", [
    TimeoutError("Read timed out"),
    ConnectionError("Connection reset by peer"),
    Exception("Throttling.User: Request was denied due to request throttling"),
    Exception("(503) Service Unavailable"),
    Exception(""),                       # 空消息 → 判不准 → 按可重试
])
def test_transient_errors_are_retryable(exc):
    assert ram_approval._is_terminal_error(exc) is False


def test_timeout_exc_fixture_is_retryable():
    assert ram_approval._is_terminal_error(_timeout_exc()) is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. _is_instance_done
# ══════════════════════════════════════════════════════════════════════════════

def _put_record(fake_redis, instance_code: str, record: dict) -> None:
    fake_redis.set(ram_approval._instance_record_key(instance_code),
                   json.dumps(record, ensure_ascii=False))


@pytest.mark.parametrize("status", ["success", "dry_run"])
def test_instance_done_for_success_and_dry_run(fake_redis, status):
    """原行为不变。"""
    _put_record(fake_redis, "i1", {"result_status": status})
    assert ram_approval._is_instance_done("i1") is True


def test_instance_done_for_terminal_failure(fake_redis):
    _put_record(fake_redis, "i2", {"result_status": "failed", "error_terminal": True})
    assert ram_approval._is_instance_done("i2") is True


def test_instance_not_done_for_transient_failure(fake_redis):
    _put_record(fake_redis, "i3", {"result_status": "failed", "error_terminal": False})
    assert ram_approval._is_instance_done("i3") is False


def test_instance_not_done_for_legacy_failure_record_without_flag(fake_redis):
    """**历史记录**：线上 Redis 里已有一堆 #64 之前写的 failed 记录（没有 error_terminal 字段）。
    它们必须仍可重试自愈 —— 不能因为新增字段就把老失败一律当终态、再也不重试。"""
    _put_record(fake_redis, "i4", {"result_status": "failed", "error_message": "boom"})
    assert ram_approval._is_instance_done("i4") is False


@pytest.mark.parametrize("record", [
    {"result_status": "processing"},
    {"result_status": "failed", "error_terminal": None},
    {"result_status": "failed", "error_terminal": ""},
    {"result_status": "failed", "error_terminal": 0},
    {},
])
def test_instance_not_done_for_other_shapes(fake_redis, record):
    _put_record(fake_redis, "i5", record)
    assert ram_approval._is_instance_done("i5") is False


def test_instance_not_done_when_no_record(fake_redis):
    assert ram_approval._is_instance_done("never-seen") is False


def test_instance_not_done_for_empty_code(fake_redis):
    assert ram_approval._is_instance_done("") is False


def test_terminal_flag_alone_without_failed_status_is_not_done(fake_redis):
    """只有 error_terminal 但状态不是 failed（脏数据）→ 不算终态，别误短路掉一个没跑过的实例。"""
    _put_record(fake_redis, "i6", {"result_status": "processing", "error_terminal": True})
    assert ram_approval._is_instance_done("i6") is False


# ══════════════════════════════════════════════════════════════════════════════
# 3. save_approval_failure 写 error_terminal
# ══════════════════════════════════════════════════════════════════════════════

def test_save_failure_marks_terminal_for_password_policy(fake_redis):
    ram_approval.save_approval_failure("inst_pwd", _pwd_exc())
    rec = ram_approval.load_approval_record("inst_pwd")
    assert rec["result_status"] == "failed"
    assert rec["error_terminal"] is True
    assert ram_approval._is_instance_done("inst_pwd") is True      # 串起来


def test_save_failure_marks_non_terminal_for_timeout(fake_redis):
    ram_approval.save_approval_failure("inst_to", _timeout_exc())
    rec = ram_approval.load_approval_record("inst_to")
    assert rec["result_status"] == "failed"
    assert rec["error_terminal"] is False
    assert ram_approval._is_instance_done("inst_to") is False      # 仍可自愈


def test_save_failure_records_error_message_and_instance(fake_redis):
    ram_approval.save_approval_failure("inst_msg", Exception("InvalidLoginName: bad"))
    rec = ram_approval.load_approval_record("inst_msg")
    assert rec["instance_code"] == "inst_msg"
    assert "InvalidLoginName" in rec["error_message"]
    assert rec["error_terminal"] is True


def test_save_failure_false_flag_survives_persistence(fake_redis):
    """`_save_instance_record` 会过滤 `None` 值——`False` 必须活着落进记录里
    （若被当空值丢掉，读回来就是「缺字段」，语义上刚好也是可重试，但这里显式钉住写入行为）。"""
    ram_approval.save_approval_failure("inst_false", _timeout_exc())
    raw = fake_redis.get(ram_approval._instance_record_key("inst_false"))
    assert '"error_terminal": false' in raw


# ══════════════════════════════════════════════════════════════════════════════
# 4. _claim_failure_notice 闸门
# ══════════════════════════════════════════════════════════════════════════════

def test_claim_notice_first_wins_second_blocked(fake_redis):
    assert ram_approval._claim_failure_notice("inst_a", "同一条错误") is True
    assert ram_approval._claim_failure_notice("inst_a", "同一条错误") is False
    assert ram_approval._claim_failure_notice("inst_a", "同一条错误") is False


def test_claim_notice_different_error_still_reported(fake_redis):
    """换了错误内容仍放行 —— 有信息量的新失败不该被吞（键里含错误签名就是为这个）。"""
    assert ram_approval._claim_failure_notice("inst_b", "密码不符合策略") is True
    assert ram_approval._claim_failure_notice("inst_b", "登录名非法") is True
    assert ram_approval._claim_failure_notice("inst_b", "密码不符合策略") is False   # 但重复的仍挡


def test_claim_notice_scoped_per_instance(fake_redis):
    """同一条错误文本、不同实例 → 各报一次（别把 A 实例的报过当成 B 实例报过）。"""
    assert ram_approval._claim_failure_notice("inst_c1", "same error") is True
    assert ram_approval._claim_failure_notice("inst_c2", "same error") is True


def test_claim_notice_key_shape_and_ttl(fake_redis):
    ram_approval._claim_failure_notice("inst_d", "err")
    keys = [k for k in fake_redis.keys("*") if "failnotice" in k]
    assert len(keys) == 1
    assert keys[0].startswith(ram_approval.REDIS_INSTANCE_PREFIX + "failnotice:inst_d:")
    ttl = fake_redis.ttl(keys[0])
    assert 0 < ttl <= 7 * 86400                       # 7 天，不永久占用


def test_claim_notice_fails_open_when_redis_down(monkeypatch):
    """Redis 不可用 → **放行**。宁可重复评论，也绝不能漏报失败（漏报会让人以为审批成功了）。"""
    def boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr(ram_approval, "get_redis", boom)
    assert ram_approval._claim_failure_notice("inst_e", "err") is True
    assert ram_approval._claim_failure_notice("inst_e", "err") is True    # 每次都放行


# ══════════════════════════════════════════════════════════════════════════════
# notify_failure 层：不落库也不重复评论（验证 dev 对我 (b) 的推理）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def comment_spy(monkeypatch):
    """抓 `_send_approval_comment`（失败播报的实际出口）。"""
    sent = []
    monkeypatch.setattr(ram_approval, "_send_approval_comment",
                        lambda instance, text, user_id, comment_id="": sent.append(
                            {"instance": instance, "text": text, "user_id": user_id}))
    monkeypatch.setattr(ram_approval, "_approval_comment_user_id", lambda req=None: "ou_admin")
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DELIVERY", "approval_comment")
    return sent


def test_notify_failure_dedupes_without_any_record(fake_redis, comment_spy):
    """dev 的推理成立：`notify_failure` **无条件先过闸门**，所以"在 save_approval_failure 之外
    抛出的失败"（没落库、没有 error_terminal）连续三次也只发一条评论。"""
    exc = _pwd_exc()
    for _ in range(3):
        ram_approval.notify_failure("inst_norec", exc)
    assert len(comment_spy) == 1
    assert ram_approval.load_approval_record("inst_norec") == {}      # 确实没落库
    assert "RAM 子账号审批执行失败" in comment_spy[0]["text"]
    assert "inst_norec" in comment_spy[0]["text"]


def test_notify_failure_dedupes_transient_retries(fake_redis, comment_spy):
    """瞬时失败重试三次都超时 → 也只该看到一条（终态标记不管这层，闸门管）。"""
    for _ in range(3):
        ram_approval.notify_failure("inst_to3", _timeout_exc())
    assert len(comment_spy) == 1


def test_notify_failure_reports_new_error_kind(fake_redis, comment_spy):
    ram_approval.notify_failure("inst_two_kinds", _timeout_exc())
    ram_approval.notify_failure("inst_two_kinds", _pwd_exc())        # 换了错误 → 再报
    ram_approval.notify_failure("inst_two_kinds", _pwd_exc())        # 重复的仍挡
    assert len(comment_spy) == 2


def test_notify_failure_without_instance_code_is_not_gated(fake_redis, comment_spy):
    """无 instance_code 时闸门不介入（键无从构造）；此时走群消息路径、不发评论。"""
    ram_approval.notify_failure("", _pwd_exc())
    assert comment_spy == []          # 评论路径要求 instance_code
    assert not [k for k in fake_redis.keys("*") if "failnotice" in k]   # 也没占闸门键


def test_notify_failure_comment_failure_does_not_raise(fake_redis, monkeypatch, comment_spy):
    """评论发送抛异常也不外泄（best-effort），不影响主流程返回。"""
    monkeypatch.setattr(ram_approval, "_send_approval_comment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("feishu 500")))
    ram_approval.notify_failure("inst_raise", _pwd_exc())            # 不抛


# ══════════════════════════════════════════════════════════════════════════════
# 5. 端到端：真机 BC9CE2EB 形态 —— 同实例 3 个 APPROVED 事件 + 每次都密码策略错
# ══════════════════════════════════════════════════════════════════════════════

APPROVAL_CODE = "ram_code_64"


def _valid_form():
    return [
        {"id": "f1", "name": "登录名称", "value": "hurunze"},
        {"id": "f2", "name": "安全邮箱", "value": "hurunze@example.com"},
        {"id": "f3", "name": "安全手机", "value": "13800138000"},
    ]


def _detail(instance_code="BC9CE2EB", status="APPROVED"):
    return {
        "approval_code": APPROVAL_CODE,
        "instance_code": instance_code,
        "user_id": "ou_requester",
        "status": status,
        "form": json.dumps(_valid_form(), ensure_ascii=False),
    }


def _event(instance_code="BC9CE2EB"):
    """飞书为同一实例反复推的事件（节点级 status=PASS，实例级由回拉详情定）。"""
    return {"header": {"event_type": "approval_instance.status_changed_v4"},
            "event": {"approval_code": APPROVAL_CODE, "status": "PASS",
                      "instance_code": instance_code}}


@pytest.fixture
def e2e(monkeypatch, fake_redis, comment_spy):
    """真实跑 `_is_instance_done` / `save_approval_failure` / `notify_failure` / 闸门，
    只桩掉云建号（哨兵计数）、回拉详情、以及与本主题无关的通知/校验副作用。"""
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_CODE", APPROVAL_CODE)
    monkeypatch.setattr(ram_approval.settings, "FEISHU_RAM_APPROVAL_DRY_RUN", False)
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _detail(code))
    monkeypatch.setattr(ram_approval, "_assert_account_delivery_ready", lambda *a, **k: None)
    monkeypatch.setattr(ram_approval, "notify_processing", lambda *a, **k: "")
    monkeypatch.setattr(ram_approval, "_safe_notify_result", lambda *a, **k: None)

    state = {"create_calls": 0, "raise": None}

    def fake_create(req):
        state["create_calls"] += 1
        if state["raise"] is not None:
            raise state["raise"]
        return ram_approval.RamAccountResult(user_name=req.login_name, created_user=True)

    monkeypatch.setattr(ram_approval, "create_accounts_for_platforms", fake_create)
    state["comments"] = comment_spy
    return state


def test_e2e_three_events_password_policy_one_comment_one_cloud_call(e2e):
    """**本文件最值钱的一条**，直接对着真机现象：

    同一实例连来 3 个 APPROVED 事件、每次建号都抛密码策略错 →
      ① 评论只发**一条**（原来 3 条）
      ② 云 API 只打**一次**（原来真打了 3 次）—— 第 2、3 个事件在 `_is_instance_done` 就短路
    """
    e2e["raise"] = _pwd_exc()
    results = [ram_approval.handle_approval_event(_event()) for _ in range(3)]

    assert e2e["create_calls"] == 1, "第 2、3 个事件不该再打云 API"
    assert len(e2e["comments"]) == 1, "同一条错误只该播报一次"

    assert "error" in results[0]                       # 第一个事件真跑了、失败了
    for res in results[1:]:
        assert res == {"ignored": True, "reason": "already_processed",
                       "instance_code": "BC9CE2EB"}
    rec = ram_approval.load_approval_record("BC9CE2EB")
    assert rec["result_status"] == "failed" and rec["error_terminal"] is True
    assert "密码策略" in e2e["comments"][0]["text"]      # 人话提示，不是原始 HTTP dump


def test_e2e_seven_events_like_real_machine(e2e):
    """真机是 7 个事件（3 个被处理中锁拦、3 个真跑）。这里连推 7 次 → 仍只 1 次云调用 + 1 条评论。"""
    e2e["raise"] = _pwd_exc()
    for _ in range(7):
        ram_approval.handle_approval_event(_event())
    assert e2e["create_calls"] == 1
    assert len(e2e["comments"]) == 1


def test_e2e_new_instance_after_user_fixes_password_still_processed(e2e):
    """终态是**按实例**的：用户改完密码会新开一个审批实例 → 新实例照常建号，没被误挡。"""
    e2e["raise"] = _pwd_exc()
    ram_approval.handle_approval_event(_event("BC9CE2EB"))
    assert e2e["create_calls"] == 1

    e2e["raise"] = None                                  # 新单密码合规
    res = ram_approval.handle_approval_event(_event("NEWINST01"))
    assert res["ignored"] is False and res["user_name"] == "hurunze"
    assert e2e["create_calls"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# 6. 自愈没被堵死
# ══════════════════════════════════════════════════════════════════════════════

def test_e2e_transient_failure_still_retries(e2e):
    """瞬时失败（超时）→ 不置终态 → 下一个事件**仍会重试建号**（自愈没被堵死）。
    但评论仍只有一条（闸门那层管刷屏）。"""
    e2e["raise"] = _timeout_exc()
    ram_approval.handle_approval_event(_event())
    assert e2e["create_calls"] == 1
    assert ram_approval.load_approval_record("BC9CE2EB")["error_terminal"] is False

    ram_approval.handle_approval_event(_event())
    assert e2e["create_calls"] == 2, "瞬时失败后必须还能重试"
    assert len(e2e["comments"]) == 1, "同一条错误不重复播报"


def test_e2e_transient_then_success_self_heals(e2e):
    """超时失败 → 重试成功 → 落 success → 后续事件短路，不再重复建号。"""
    e2e["raise"] = _timeout_exc()
    ram_approval.handle_approval_event(_event())
    e2e["raise"] = None
    res = ram_approval.handle_approval_event(_event())
    assert res["ignored"] is False and e2e["create_calls"] == 2

    ram_approval.handle_approval_event(_event())
    assert e2e["create_calls"] == 2, "成功后不该再建号"


def test_e2e_success_first_time_short_circuits_later_events(e2e, fake_redis):
    """成功路径原行为不变：后续事件 already_processed 短路。"""
    res = ram_approval.handle_approval_event(_event())
    assert res["ignored"] is False and e2e["create_calls"] == 1
    res2 = ram_approval.handle_approval_event(_event())
    assert res2 == {"ignored": True, "reason": "already_processed",
                    "instance_code": "BC9CE2EB"}
    assert e2e["create_calls"] == 1
    assert e2e["comments"] == []


def test_e2e_transient_then_terminal_error_reports_twice(e2e):
    """先超时（可重试）→ 重试时撞上密码策略错 → 第二条评论该发（不同错误，有信息量），
    此后置终态、第三个事件短路。"""
    e2e["raise"] = _timeout_exc()
    ram_approval.handle_approval_event(_event())
    e2e["raise"] = _pwd_exc()
    ram_approval.handle_approval_event(_event())
    assert e2e["create_calls"] == 2
    assert len(e2e["comments"]) == 2

    ram_approval.handle_approval_event(_event())
    assert e2e["create_calls"] == 2, "终态后不再打云 API"
    assert len(e2e["comments"]) == 2
