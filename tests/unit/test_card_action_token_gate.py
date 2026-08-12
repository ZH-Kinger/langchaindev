"""P0 安全回归：/feishu/card_action 的验证 token 门禁。

守的是什么（漏洞原型）
--------------------
`/feishu/card_action` 此前**完全没有**验证 token 校验，而操作人身份 `open_id` 是直接从
请求体里取的（`operator.operator_id.open_id` / `data["open_id"]` / `data["user_id"]`），
随后原样传给 `actions._process_action`。多个 handler 用 `open_id != settings.ADMIN_FEISHU_OPEN_ID`
做管理员门禁（actions.py:304/335/537/1247/1375 —— OSS 权限下发、跨云/SSH/PFS 迁移的确认下发）。
open_id 在组织内并不是秘密 → 任何人 POST 一个把 open_id 填成管理员的请求，就能过掉全部管理员门禁。

修法：模块级 `_extract_request_token` + `_token_verified`（hmac.compare_digest；
`FEISHU_VERIFICATION_TOKEN` 未配置时返回 True 保持降级），两条入站路由共用。

本文件的断言纪律：**只断言状态码不算数** —— 每条拒绝用例都装了哨兵，必须证明
`_process_action` / `_handle_card_trigger_sync` 真的一次都没被调到（危险动作没执行）。
"""
import pytest


TOKEN = "verif-token-abc123"
WRONG = "verif-token-abc124"          # 与 TOKEN 等长、仅一字之差（常数时间比较也必须判假）
ADMIN = "ou_admin_open_id_for_test"


def _client():
    from core.feishu_bot import app
    return app.test_client()


@pytest.fixture
def token_configured(monkeypatch):
    """线上姿态：验证 token 已配置 + 一个已知的管理员 open_id。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN)
    return settings


@pytest.fixture
def token_absent(monkeypatch):
    """未配置 token 的降级姿态（隔离热备机等）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "FEISHU_VERIFICATION_TOKEN", "")
    monkeypatch.setattr(settings, "ADMIN_FEISHU_OPEN_ID", ADMIN)
    return settings


@pytest.fixture
def sentinel(monkeypatch):
    """把两条动作出口全换成哨兵：任何一次调用都会被记下来（并且不会真执行危险动作）。"""
    from core.feishu_bot import actions
    rec = {"process": [], "trigger": []}

    def _fake_process(action_name, action_val, open_id, chat_id, form_value=None):
        rec["process"].append({
            "action": action_name, "value": action_val, "open_id": open_id,
            "chat_id": chat_id, "form_value": form_value,
        })
        return {"stub": True}

    def _fake_trigger(data):
        rec["trigger"].append(data)
        return {"stub": True}

    monkeypatch.setattr(actions, "_process_action", _fake_process)
    monkeypatch.setattr(actions, "_handle_card_trigger_sync", _fake_trigger)
    return rec


def _legacy_body(*, token=None, header_token=None, open_id=ADMIN,
                 action="approve_oss_perm", id_field="operator"):
    """老式卡片回调请求体（无 schema / 无 event → 走 routes.py 的 legacy 分支）。

    `id_field` 覆盖三条身份取值路径：operator.operator_id.open_id / open_id / user_id。
    """
    body = {"action": {"value": {"action": action}}, "open_chat_id": "oc_test"}
    if id_field == "operator":
        body["operator"] = {"operator_id": {"open_id": open_id}}
    else:
        body[id_field] = open_id
    if token is not None:
        body["token"] = token
    if header_token is not None:
        body["header"] = {"token": header_token}
    return body


def _v2_body(*, token=None, header_token=None, open_id=ADMIN, action="confirm_transfer"):
    """schema 2.0 卡片回调请求体（含 event → 走 _handle_card_trigger_sync 分支）。"""
    body = {
        "schema": "2.0",
        "event": {
            "operator": {"operator_id": {"open_id": open_id}},
            "action": {"value": {"action": action}},
            "context": {"open_chat_id": "oc_test", "open_message_id": "om_test"},
        },
    }
    if token is not None:
        body["token"] = token
    if header_token is not None:
        body["header"] = {"token": header_token, "event_type": "card.action.trigger"}
    return body


def _assert_untouched(rec):
    assert rec["process"] == [], f"危险动作被执行了：{rec['process']}"
    assert rec["trigger"] == [], f"危险动作被执行了：{rec['trigger']}"


# ── 1. 错误 token → 403 且 handler 未被触达 ─────────────────────────────────────

def test_card_action_wrong_token_rejected_and_handler_never_called(token_configured, sentinel):
    resp = _client().post("/feishu/card_action", json=_legacy_body(token=WRONG))
    assert resp.status_code == 403
    assert resp.get_json() == {"code": 1, "msg": "invalid token"}
    _assert_untouched(sentinel)


def test_card_action_wrong_token_rejected_on_schema2_path(token_configured, sentinel):
    """2.0 分支（真实卡片点击走这条）同样过门，不能只护住 legacy 分支。"""
    resp = _client().post("/feishu/card_action", json=_v2_body(header_token=WRONG))
    assert resp.status_code == 403
    _assert_untouched(sentinel)


def test_card_action_wrong_token_in_header_rejected(token_configured, sentinel):
    resp = _client().post("/feishu/card_action", json=_legacy_body(header_token=WRONG))
    assert resp.status_code == 403
    _assert_untouched(sentinel)


# ── 2. 完全不带 token → 403 ────────────────────────────────────────────────────

def test_card_action_missing_token_rejected(token_configured, sentinel):
    resp = _client().post("/feishu/card_action", json=_legacy_body())
    assert resp.status_code == 403
    _assert_untouched(sentinel)


def test_card_action_empty_body_rejected(token_configured, sentinel):
    """空请求体（既不是 challenge 也没 token）也必须被挡，不能因为"没东西可解析"就放行。"""
    resp = _client().post("/feishu/card_action", json={})
    assert resp.status_code == 403
    _assert_untouched(sentinel)


def test_card_action_empty_string_token_rejected(token_configured, sentinel):
    """token="" 走的是 `header.token or data.token or ""` 的假值分支，别被当成"未配置"放行。"""
    resp = _client().post("/feishu/card_action", json=_legacy_body(token=""))
    assert resp.status_code == 403
    _assert_untouched(sentinel)


# ── 3. 正确 token → 正常放行（两种 token 位置都要认） ──────────────────────────

def test_card_action_correct_top_level_token_passes(token_configured, sentinel):
    resp = _client().post("/feishu/card_action", json=_legacy_body(token=TOKEN, open_id="ou_normal"))
    assert resp.status_code == 200
    assert len(sentinel["process"]) == 1
    assert sentinel["process"][0]["action"] == "approve_oss_perm"
    assert sentinel["process"][0]["open_id"] == "ou_normal"


def test_card_action_correct_header_token_passes(token_configured, sentinel):
    """旧式回调没有 header，新式有 —— header.token 位置同样得认，否则线上正常点击被 403 误杀。"""
    resp = _client().post("/feishu/card_action",
                          json=_legacy_body(header_token=TOKEN, open_id="ou_normal"))
    assert resp.status_code == 200
    assert len(sentinel["process"]) == 1


def test_card_action_correct_token_passes_on_schema2_path(token_configured, sentinel):
    resp = _client().post("/feishu/card_action", json=_v2_body(header_token=TOKEN))
    assert resp.status_code == 200
    assert len(sentinel["trigger"]) == 1
    assert sentinel["process"] == []          # 2.0 分支由 _handle_card_trigger_sync 内部再分派


# ── 4. 核心回归：伪造管理员 open_id + 错误 token → 必须 403 ──────────────────────

@pytest.mark.parametrize("id_field", ["operator", "open_id", "user_id"])
@pytest.mark.parametrize("action", [
    "approve_oss_perm",            # actions.py:304  OSS 权限整批下发
    "approve_oss_perm_selected",   # actions.py:335  OSS 权限按人下发
    "confirm_transfer",            # actions.py:537  跨云迁移超阈值确认
    "confirm_ssh_transfer",        # actions.py:1247 SSH 三跳链确认
    "confirm_pfs_transfer",        # actions.py:1375 PFS 直传确认
])
def test_forged_admin_open_id_with_wrong_token_is_rejected(token_configured, sentinel,
                                                           id_field, action):
    """**本次漏洞的直接复现**：请求体把操作人 open_id 填成管理员，妄图过掉 handler 里的
    `open_id != ADMIN_FEISHU_OPEN_ID` 判定。三条身份取值路径 × 五个管理员门禁动作全覆盖。

    修复前：token 无人校验 → 请求直达 _process_action → 管理员动作被陌生人执行。
    修复后：token 不对 → 403，且 handler 一次都不该被调到（哨兵为空）。
    """
    body = _legacy_body(token=WRONG, open_id=ADMIN, action=action, id_field=id_field)
    resp = _client().post("/feishu/card_action", json=body)
    assert resp.status_code == 403
    _assert_untouched(sentinel)


@pytest.mark.parametrize("id_field", ["operator", "open_id", "user_id"])
def test_forged_admin_open_id_would_be_effective_without_the_gate(token_configured, sentinel,
                                                                  id_field):
    """上一条的**阳性对照**：同一份伪造请求体，只把 token 换对，就一路带着管理员 open_id
    走进 _process_action。说明拦下它的确实是 token 这道门（而不是请求体碰巧解析不出身份），
    也说明这道门一旦被拆，伪造身份立刻生效。
    """
    body = _legacy_body(token=TOKEN, open_id=ADMIN, action="approve_oss_perm", id_field=id_field)
    resp = _client().post("/feishu/card_action", json=body)
    assert resp.status_code == 200
    assert len(sentinel["process"]) == 1
    assert sentinel["process"][0]["open_id"] == ADMIN, "身份取值路径变了？该断言同时钉住三条 fallback"


def test_forged_admin_open_id_on_schema2_path_rejected(token_configured, sentinel):
    resp = _client().post("/feishu/card_action", json=_v2_body(token=WRONG, open_id=ADMIN))
    assert resp.status_code == 403
    _assert_untouched(sentinel)


# ── 5. 未配置 token 时的降级（记录当前行为，防无意改成 fail-closed 而不自知） ─────

def test_card_action_passthrough_when_token_not_configured(token_absent, sentinel):
    """`FEISHU_VERIFICATION_TOKEN` 为空 → 门是敞的、请求照常处理（run() 会打 ERROR 告警）。

    这是**刻意保留的降级**：未配置 token 的环境要能起来。此用例只是把这个事实钉死 ——
    哪天有人改成 fail-closed（未配置就全 403），线上未配 token 的实例会整体瘫掉，
    这条会立刻变红提醒；反过来若有人以为已 fail-closed，这条也说明它并没有。
    """
    resp = _client().post("/feishu/card_action", json=_legacy_body(open_id=ADMIN))
    assert resp.status_code == 200
    assert len(sentinel["process"]) == 1


def test_event_passthrough_when_token_not_configured(token_absent, sentinel):
    resp = _client().post("/feishu/event", json={"event": {}})
    assert resp.status_code == 200


# ── 6. /feishu/event 既有行为不回归 ────────────────────────────────────────────

def test_event_challenge_echo_precedes_token_gate(token_configured):
    """url_verification 回显必须排在 token 校验**之前** —— 飞书后台配置地址时那一发
    challenge 不带正确 token，若被 403 挡掉就永远配不上事件订阅地址。"""
    resp = _client().post("/feishu/event",
                          json={"type": "url_verification", "challenge": "chal-event"})
    assert resp.status_code == 200
    assert resp.get_json() == {"challenge": "chal-event"}


def test_event_correct_header_token_passes(token_configured, sentinel):
    resp = _client().post("/feishu/event",
                          json={"header": {"token": TOKEN, "event_id": "ev-1"}, "event": {}})
    assert resp.status_code == 200


def test_event_correct_top_level_token_passes(token_configured):
    """旧版 v1 事件（审批回调等）token 在顶层，不能因为没有 header 就被 403 误杀。"""
    resp = _client().post("/feishu/event", json={"token": TOKEN, "event": {}})
    assert resp.status_code == 200


def test_event_wrong_token_rejected(token_configured, sentinel):
    resp = _client().post("/feishu/event",
                          json={"header": {"token": WRONG, "event_type": "card.action.trigger"},
                                "event": {}})
    assert resp.status_code == 403
    _assert_untouched(sentinel)      # card.action.trigger 也没被同步执行


def test_event_missing_token_rejected(token_configured):
    resp = _client().post("/feishu/event", json={"event": {}})
    assert resp.status_code == 403


# ── 7. card_action 的 challenge 回显（token 已配置时也必须先回显） ───────────────

def test_card_action_get_challenge_echo_with_token_configured(token_configured):
    resp = _client().get("/feishu/card_action?challenge=chal-get")
    assert resp.status_code == 200
    assert resp.get_json() == {"challenge": "chal-get"}


def test_card_action_post_url_verification_echo_with_token_configured(token_configured):
    """填卡片回调地址时飞书发的 url_verification 不带业务 token，必须在门之前回显。"""
    resp = _client().post("/feishu/card_action",
                          json={"type": "url_verification", "challenge": "chal-post"})
    assert resp.status_code == 200
    assert resp.get_json() == {"challenge": "chal-post"}


# ── 8. _extract_request_token 单元级 ───────────────────────────────────────────

def test_extract_token_prefers_header_over_top_level():
    from core.feishu_bot import routes
    assert routes._extract_request_token({"header": {"token": "h"}, "token": "t"}) == "h"


def test_extract_token_falls_back_to_top_level():
    from core.feishu_bot import routes
    assert routes._extract_request_token({"token": "t"}) == "t"


def test_extract_token_returns_empty_when_absent():
    from core.feishu_bot import routes
    assert routes._extract_request_token({}) == ""
    assert routes._extract_request_token({"header": {}}) == ""


def test_extract_token_tolerates_null_header():
    """`{"header": null}` 是外部可构造的：`data.get("header") or {}` 必须兜住，
    否则取 token 这一步就 AttributeError（500 而非 403）。"""
    from core.feishu_bot import routes
    assert routes._extract_request_token({"header": None, "token": "t"}) == "t"
    assert routes._extract_request_token({"header": None}) == ""


def test_extract_token_empty_header_token_falls_through_to_top_level():
    """header.token 为空串时应继续看顶层，别把空串当"取到了"。"""
    from core.feishu_bot import routes
    assert routes._extract_request_token({"header": {"token": ""}, "token": "t"}) == "t"


def test_token_verified_true_when_not_configured(monkeypatch):
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", "")
    assert routes._token_verified({}) is True


def test_token_verified_exact_match_only(monkeypatch):
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", TOKEN)
    assert routes._token_verified({"token": TOKEN}) is True
    assert routes._token_verified({"token": TOKEN + "x"}) is False   # 前缀不算数
    assert routes._token_verified({"token": TOKEN[:-1]}) is False    # 短一位不算数
    assert routes._token_verified({"token": TOKEN.upper()}) is False # 大小写敏感


def test_token_verified_survives_non_string_token(monkeypatch):
    """请求体里 token 可以是任意 JSON 类型（外部可控）。非字符串不该把校验打崩成 500，
    也不该因为 str() 化而意外相等。"""
    from core.feishu_bot import routes
    monkeypatch.setattr(routes.settings, "FEISHU_VERIFICATION_TOKEN", TOKEN)
    for bogus in (123, True, {"a": 1}, ["x"]):
        assert routes._token_verified({"token": bogus}) is False


# ── 9. 畸形请求体不得把门打崩成 500（token / header / body 全是外部可控数据） ────
#
# 这一组曾抓到一个真缺口（发现时 xfail，dev 当轮修完即转常绿）：`hmac.compare_digest` 对含
# 非 ASCII 的 str 抛 `TypeError: comparing strings with non-ASCII characters is not supported`，
# 未捕获则匿名请求可把这两条路由打成 500 —— 而 500 会经 utils/logger 的 ERROR 回调给管理员
# 刷飞书私信，等于把一道鉴权门变成**免鉴权的告警放大器**。修法见 routes._token_verified：
# 比较前两侧都 encode("utf-8")，并对 header/body/token 做 isinstance 兜底。
# 这些用例守的是"畸形输入走正常的 403 拒绝路径"，而不是"碰巧也没执行动作"。

@pytest.mark.parametrize("route", ["/feishu/card_action", "/feishu/event"])
def test_non_ascii_token_is_403_not_500(token_configured, sentinel, route):
    resp = _client().post(route, json=_legacy_body(token="令牌不对"))
    assert resp.status_code == 403
    _assert_untouched(sentinel)


@pytest.mark.parametrize("bogus_token", [123, True, {"a": 1}, ["x"], None])
def test_non_string_token_in_body_is_403_not_500(token_configured, sentinel, bogus_token):
    """请求体是攻击者构造的 JSON，token 字段可以是任意类型。非字符串既不能 500，
    也不能因为 str() 化而意外匹配上。"""
    body = _legacy_body()
    body["token"] = bogus_token
    resp = _client().post("/feishu/card_action", json=body)
    assert resp.status_code == 403
    _assert_untouched(sentinel)


@pytest.mark.parametrize("route", ["/feishu/card_action", "/feishu/event"])
@pytest.mark.parametrize("bogus_header", [None, "x", 1, ["h"]])
def test_non_dict_header_is_403_not_500(token_configured, sentinel, route, bogus_header):
    """`data.get("header", {})` 对显式 `"header": null` / 非 dict 拿到的不是字典，
    随后 `.get(...)` 就是 AttributeError → 500。取 token 与取 event_id 两处都得兜。"""
    body = _legacy_body()
    body["header"] = bogus_header
    resp = _client().post(route, json=body)
    assert resp.status_code == 403
    _assert_untouched(sentinel)


@pytest.mark.parametrize("route", ["/feishu/card_action", "/feishu/event"])
@pytest.mark.parametrize("body", [[1, 2, 3], "just-a-string", 42])
def test_non_object_json_body_is_403_not_500(token_configured, sentinel, route, body):
    """body 是 JSON 数组/标量时 `data.get` 会 AttributeError → 500。必须当成"没 token"→ 403。"""
    resp = _client().post(route, json=body)
    assert resp.status_code == 403
    _assert_untouched(sentinel)


def test_extract_token_tolerates_non_dict_data():
    from core.feishu_bot import routes
    for bogus in ([1, 2], "s", 42, None):
        assert routes._extract_request_token(bogus) == ""
