"""延期绝不推迟生效时间（`orchestrator.extend_grant` 的 not_before 单调回归）。

线上事故：延期审批表单的 DateInterval 被申请人填成「原到期日 → 新到期日」，
`approval._parse_date_interval` 忠实解析成 (not_before, expire)，原样传给 `extend_grant`
→ grant["not_before"] 被推到未来 → policy 里 `DateGreaterThan(acs:CurrentTime, not_before)`
当场不成立 → **使用方在延期通过的那一刻反而用不了了**。

修复语义（`core/temp_ak_issuance/orchestrator.py::extend_grant`）：
    延期只允许推后到期时间，**永不推迟生效时间**；往前挪（更早的生效时间）无害，保留。

本文件覆盖：
  · 回归本体：「原到期日 → 更晚到期日」→ not_before 原值不动、expire 换新（修复前这条必红）。
  · 更早的 not_before → 取更早那个（min 语义，不是"一律忽略传入值"）。
  · not_before 空/None/0 → 保持原值（既有行为不能被改坏）；原 grant 无 not_before → 回落 now。
  · 校验顺序：expire 已过 / 传入 not_before ≥ expire / stage≠ISSUED 一律抛，**且校验用的是
    传入值、发生在 min() 之前**（畸形表单要报错，不能被 min 悄悄"救"回来）。
  · 真实 policy 文档（走 issuer.rewrite_ram_window，只桩 RAM 客户端）的时间窗没被推后，
    且延期完成的那一刻窗口是「当下有效」的。
  · STS 模式同样不推迟生效时间（重签发 / 转方案 B 两条路都测）。
  · extends 审计记录 old_expire/new_expire + old_not_before/new_not_before。

【2026-09-23 语义补全】钳制只对**已经生效**的凭证成立（`effective_nb <= now`）：
  · 已生效 + 表单填未来 start → 仍钳制回原值（上面那条事故的修复，§1 §10 对照组）。
  · **尚未生效**（窗口整段在未来）+ 表单挪档期 → **允许**整体往后挪（§10）；一律钳制会把
    「10-01→10-31 改批成 11-01→11-30」压回 10-01 生效，比最新审批批准的起点早一个月 = 扩权。
  · 新增校验「新到期时间必须晚于原到期时间」，且**跑在钳制之前**（§12）——
    否则「新生效日 → 原到期日」的表单会被钳制压成一次空操作，却记审计、还发「有效期已延长」回执。

既有 extend 用例在 tests/unit/test_temp_ak_extend.py（审批门禁/撤销/幂等），本文件只管时间窗。
"""
import json
import time

import pytest

from core.temp_ak_issuance import issuer, orchestrator as o, policy

DAY = 86400


@pytest.fixture(autouse=True)
def _no_bucket_map(monkeypatch):
    monkeypatch.setattr(o.settings, "TEMP_AK_BUCKET_MAP_RAW", "{}", raising=False)


def _issued_grant(**over):
    """已发放的方案 B（RAM）grant：生效时间在过去、到期在未来（正常在用的凭证）。"""
    now = time.time()
    g = {
        "grant_id": "tak-nbtest", "account": "", "stage": o.STAGE_ISSUED, "mode": "ram",
        "platform": "aliyun", "enterprise": "外采公司A", "bucket": "wuji-sing",
        "prefix": "team/data/", "caps": ["read", "download"],
        "not_before": now - 7 * DAY,          # 7 天前生效
        "expire": now + 1 * DAY,              # 明天到期
        "user_name": "tempak-ext-nb", "policy_name": "temp-ak-auto-tempak-ext-nb",
        "ak_id": "LTAI_ORIG", "requester": "ou_alice", "source_ips": [],
    }
    g.update(over)
    return g


def _stub_ram_rewrite(monkeypatch):
    """桩掉云调用，只观察 grant 字典（policy 文档层面另有专测）。"""
    seen = []
    monkeypatch.setattr(issuer, "rewrite_ram_window", lambda g: seen.append(dict(g)))
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("方案 B 延期不该重签发凭证"))
    return seen


def _fmt(ts) -> str:
    return f"{ts!r}({o.fmt_ts(ts) if ts else '-'})"


# ══════════════════════════════════════════════════════════════════════════════
# 1. 回归本体：「原到期日 → 新到期日」的 DateInterval 不得推迟生效时间
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_with_old_expire_as_start_keeps_not_before(monkeypatch):
    """线上 bug 的精确复现：start 填成原到期日 → not_before 必须**保持原值**。

    修复前：not_before 被写成 now+1d（未来）→ policy DateGreaterThan 当场不成立 →
    凭证在延期通过的那一刻失效。
    """
    seen = _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    o._save(g)
    orig_nb, orig_exp = g["not_before"], g["expire"]
    new_exp = orig_exp + 30 * DAY

    g2, creds = o.extend_grant(g, orig_exp, new_exp, extend_instance="ext_inst_nb1")

    assert creds is None, "方案 B 延期只改写 policy 时间窗，不该重发凭证"
    assert g2["not_before"] == orig_nb, (
        "延期把生效时间推迟了（回归 bug！）："
        f"期望保持原值 {_fmt(orig_nb)}，实际 {_fmt(g2['not_before'])}；"
        f"表单传入的 start 是原到期日 {_fmt(orig_exp)}"
    )
    assert g2["expire"] == float(new_exp), (
        f"到期时间应换成新值：期望 {_fmt(float(new_exp))}，实际 {_fmt(g2['expire'])}")
    assert seen and seen[-1]["not_before"] == orig_nb, (
        "传给 issuer.rewrite_ram_window 的 grant 里 not_before 也必须是原值，"
        f"实际 {_fmt(seen[-1]['not_before'] if seen else None)}")


def test_extended_window_is_valid_right_now(monkeypatch):
    """语义断言：延期完成的那一刻，凭证必须**立刻可用**（not_before ≤ now < expire）。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_exp = g["expire"]
    g2, _ = o.extend_grant(g, orig_exp, orig_exp + 30 * DAY)
    now = time.time()
    assert g2["not_before"] <= now, (
        f"延期后生效时间落在未来 {_fmt(g2['not_before'])} > now {_fmt(now)}，"
        "使用方在审批通过的瞬间反而用不了")
    assert g2["expire"] > now, "延期后到期时间应在未来"


def test_repeated_extends_never_push_not_before(monkeypatch):
    """连续多次「原到期日 → 新到期日」式延期，not_before 始终钉在最初那个值。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb = g["not_before"]
    for i in range(3):
        cur_exp = g["expire"]
        g, _ = o.extend_grant(g, cur_exp, cur_exp + 10 * DAY,
                              extend_instance=f"ext_inst_{i}")
        assert g["not_before"] == orig_nb, (
            f"第 {i + 1} 次延期后生效时间漂移：期望 {_fmt(orig_nb)}，实际 {_fmt(g['not_before'])}")
    assert len(g["extends"]) == 3, f"三次延期应留三条审计记录，实际 {len(g.get('extends') or [])}"


def test_persisted_record_also_keeps_not_before(monkeypatch):
    """落回 Redis 的记录（后续延期/撤销/清理读的就是它）同样不能被推迟。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    o.extend_grant(g, orig_exp, orig_exp + 30 * DAY)
    stored = o.get_grant("tak-nbtest")
    assert stored is not None, "延期后 grant 应已落盘"
    assert stored["not_before"] == orig_nb, (
        f"落盘记录里的生效时间被推迟：期望 {_fmt(orig_nb)}，实际 {_fmt(stored['not_before'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 传更早的生效时间 → 允许（取更早的那个）
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_with_earlier_not_before_moves_backward(monkeypatch):
    """往前挪无害：min(传入, 原值) → 取传入的更早值。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    earlier = orig_nb - 3 * DAY
    g2, _ = o.extend_grant(g, earlier, orig_exp + DAY)
    assert g2["not_before"] == float(earlier), (
        f"更早的生效时间应被采纳：期望 {_fmt(float(earlier))}，实际 {_fmt(g2['not_before'])}")


def test_extend_with_same_not_before_is_noop(monkeypatch):
    """传入等于原值 → 不变（min 的边界，不该出现浮点漂移或类型退化）。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb = g["not_before"]
    g2, _ = o.extend_grant(g, orig_nb, g["expire"] + DAY)
    assert g2["not_before"] == orig_nb
    assert isinstance(g2["not_before"], float), "not_before 必须落成 float"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 不传生效时间 → 保持原值（既有行为，别改坏）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("falsy", [0, 0.0, None, "", False])
def test_extend_without_not_before_keeps_original(monkeypatch, falsy):
    """DateInterval 的 start 留空（parse 返 0.0）→ 沿用原 not_before。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb = g["not_before"]
    g2, _ = o.extend_grant(g, falsy, g["expire"] + 30 * DAY)
    assert g2["not_before"] == orig_nb, (
        f"未传生效时间时应沿用原值：期望 {_fmt(orig_nb)}，实际 {_fmt(g2['not_before'])}（传入 {falsy!r}）")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 老数据：原 grant 没有 not_before
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("original", ["missing", 0, 0.0, None])
def test_extend_missing_original_not_before_no_form_start_falls_back_to_now(
        monkeypatch, original):
    """老记录无 not_before（或存成 0/None）且表单也没填 → 回落 now，不抛异常。

    【2026-09-23 重命名】本用例原名与下面那条**逐字相同**，Python 里后定义的直接覆盖前者
    → 这条从未被执行过（详见下方注释）。顺带把 0/0.0/None 三种落库形态也参数化进来 ——
    `float(original_nb or now)` 对它们走的是同一条兜底分支。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    if original == "missing":
        g.pop("not_before")
    else:
        g["not_before"] = original
    before = time.time()
    g2, _ = o.extend_grant(g, 0, g["expire"] + 30 * DAY)
    after = time.time()
    assert before <= g2["not_before"] <= after, (
        f"应回落到当前时间：期望 [{before}, {after}]，实际 {_fmt(g2['not_before'])}")


def test_extend_missing_original_not_before_with_form_start_falls_back_to_now(monkeypatch):
    """回归锁（2026-09-23 补齐）：原 grant 没有 not_before 时按 now 兜底，照样钳制。

    修之前 `elif original_nb:` 只在原 grant **有** not_before 时才钳制，老数据
    （无该键，或存成 0）+ 表单 start 填了未来时间 → 生效时间仍被推到未来，即原 bug。
    线上不可达（`create_grant_record` 必写 not_before），属纵深兜底。

    【命名】本条与上一条测的是**两件事**（表单填没填 start），名字必须不同 ——
    2026-09-23 这两条一度重名，上一条被静默覆盖、整条不执行。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    g.pop("not_before")
    before = time.time()
    future = before + 5 * DAY
    g2, _ = o.extend_grant(g, future, future + 10 * DAY)
    after = time.time()
    assert before <= g2["not_before"] <= after, (
        f"老记录应按 now 兜底、不许把生效时间推到未来：实际 {_fmt(g2['not_before'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 既有校验不受 min() 影响（校验用传入值、跑在 min 之前）
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_expire_in_past_rejected(monkeypatch):
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    with pytest.raises(o.TempAkError, match="到期"):
        o.extend_grant(g, 0, time.time() - 100)


def test_extend_not_before_after_expire_rejected_not_rescued_by_min(monkeypatch):
    """畸形表单（start 晚于 end）必须**报错**，不能被 min() 悄悄救回来。

    校验顺序是刻意的：min 是给「start 被填成原到期日」这种**方向正确但取值过大**的表单兜底，
    而 start > end 是表单本身填反/填错，静默修正会让人以为审批单填对了、下次继续错。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    now = time.time()
    with pytest.raises(o.TempAkError, match="生效时间"):
        o.extend_grant(g, now + 7200, now + 3600)
    assert g["not_before"] == orig_nb and g["expire"] == orig_exp, (
        f"抛错路径不得改写 grant 时间窗：now_before={_fmt(g['not_before'])} "
        f"expire={_fmt(g['expire'])}")
    assert "extends" not in g, "被拒的延期不该留审计记录"


def test_extend_not_before_equal_expire_rejected(monkeypatch):
    """边界：start == end 也拒（校验是 >=）。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    t = time.time() + 3 * DAY
    with pytest.raises(o.TempAkError, match="生效时间"):
        o.extend_grant(g, t, t)


@pytest.mark.parametrize("stage", [o.STAGE_NEW, o.STAGE_REVOKED, o.STAGE_FAILED])
def test_extend_non_issued_rejected(monkeypatch, stage):
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant(stage=stage)
    with pytest.raises(o.TempAkError, match="ISSUED"):
        o.extend_grant(g, g["expire"], g["expire"] + DAY)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 真实 policy 文档的时间窗（只桩 RAM 客户端，policy 生成走真代码）
# ══════════════════════════════════════════════════════════════════════════════

class _FakeRamClient:
    """只接 create_policy_version（rewrite_ram_window 唯一用到的调用）。"""

    def __init__(self):
        self.docs = []

    def create_policy_version(self, req):
        self.docs.append(json.loads(req.policy_document))


def _patch_ram_client(monkeypatch):
    from core.oss_perm import permsync
    fake = _FakeRamClient()
    monkeypatch.setattr(permsync, "make_ram_client", lambda: fake)
    return fake


def test_policy_window_not_pushed_forward(monkeypatch):
    """端到端到 policy 文档：DateGreaterThan 必须还是**原**生效时间。

    这是用户实际受影响的那一层 —— grant 字典只是中间态，真正决定「还能不能用」的是
    下发到 RAM 的 policy 里那个 `acs:CurrentTime` 条件。
    """
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("方案 B 延期不该重签发凭证"))
    fake = _patch_ram_client(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    new_exp = orig_exp + 30 * DAY

    o.extend_grant(g, orig_exp, new_exp)          # start 填成原到期日（bug 触发形状）

    assert len(fake.docs) == 1, f"应下发一份新 policy 版本，实际 {len(fake.docs)} 份"
    stmts = fake.docs[0]["Statement"]
    assert stmts, "policy 不该为空（caps=read/download）"
    want_nb = policy.iso8601_bj(orig_nb)
    want_exp = policy.iso8601_bj(new_exp)
    for st in stmts:
        cond = st.get("Condition") or {}
        got_nb = cond.get("DateGreaterThan", {}).get("acs:CurrentTime")
        got_exp = cond.get("DateLessThan", {}).get("acs:CurrentTime")
        assert got_nb == want_nb, (
            f"policy 生效时间被推后：期望 {want_nb}，实际 {got_nb}（语句 Action={st.get('Action')}）")
        assert got_exp == want_exp, (
            f"policy 到期时间未延长：期望 {want_exp}，实际 {got_exp}（语句 Action={st.get('Action')}）")


def test_policy_window_covers_now_after_extend(monkeypatch):
    """同上，但断言的是「现在是否落在窗口内」—— 与使用方的实际体验同构。"""
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("方案 B 延期不该重签发凭证"))
    fake = _patch_ram_client(monkeypatch)
    g = _issued_grant()
    orig_exp = g["expire"]
    o.extend_grant(g, orig_exp, orig_exp + 30 * DAY)
    now_iso = policy.iso8601_bj(time.time())
    for st in fake.docs[0]["Statement"]:
        cond = st["Condition"]
        nb = cond["DateGreaterThan"]["acs:CurrentTime"]
        exp = cond["DateLessThan"]["acs:CurrentTime"]
        assert nb <= now_iso < exp, (
            f"延期后的 policy 当下不生效：窗口 {nb} → {exp}，now={now_iso}"
            "（ISO8601 同时区可直接字典序比较）")


# ══════════════════════════════════════════════════════════════════════════════
# 7. STS 模式的延期
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_sts_short_window_keeps_not_before(monkeypatch):
    """STS 新窗 ≤12h → 重签发；传进 issuer.issue 的 grant 不得带未来的 not_before。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    monkeypatch.setattr(issuer, "rewrite_ram_window",
                        lambda g: pytest.fail("STS 路径不该改写 RAM policy"))
    seen = []

    def fake_issue(grant):
        seen.append(dict(grant))
        return {"access_key_id": "STS.NEW", "access_key_secret": "SK",
                "security_token": "TOK", "expire_ts": grant["expire"], "mode": "sts"}

    monkeypatch.setattr(issuer, "issue", fake_issue)
    g = _issued_grant(mode="sts", policy_name="", ak_id="",
                      expire=time.time() + 3600)      # STS 原窗很短
    orig_nb, orig_exp = g["not_before"], g["expire"]

    g2, creds = o.extend_grant(g, orig_exp, time.time() + 6 * 3600)

    assert g2["mode"] == "sts", f"新窗 ≤12h 应仍走 STS，实际 {g2['mode']}"
    assert creds["security_token"] == "TOK"
    assert g2["not_before"] == orig_nb, (
        f"STS 延期推迟了生效时间：期望 {_fmt(orig_nb)}，实际 {_fmt(g2['not_before'])}")
    assert seen[-1]["not_before"] == orig_nb, (
        "重签发用的 session policy 会拿这个 not_before 去写时间窗，"
        f"实际 {_fmt(seen[-1]['not_before'])}")


def test_extend_sts_long_window_switches_to_ram_keeps_not_before(monkeypatch):
    """STS 新窗 >12h → 转方案 B 发长期 AK，同样不得推迟生效时间。"""
    monkeypatch.setattr(issuer.settings, "TEMP_AK_STS_MAX_SECONDS", 43200)
    seen = []

    def fake_issue(grant):
        seen.append(dict(grant))
        return {"access_key_id": "LTAI_NEW", "access_key_secret": "SK",
                "security_token": "", "expire_ts": grant["expire"], "mode": "ram"}

    monkeypatch.setattr(issuer, "issue", fake_issue)
    g = _issued_grant(mode="sts", policy_name="", ak_id="",
                      expire=time.time() + 3600)
    orig_nb, orig_exp = g["not_before"], g["expire"]

    g2, creds = o.extend_grant(g, orig_exp, time.time() + 5 * DAY)

    assert g2["mode"] == "ram", f"新窗 >12h 应转方案 B，实际 {g2['mode']}"
    assert g2["ak_id"] == "LTAI_NEW"
    assert g2["policy_name"], "转方案 B 应补出 policy_name"
    assert g2["not_before"] == orig_nb, (
        f"STS→RAM 转换时推迟了生效时间：期望 {_fmt(orig_nb)}，实际 {_fmt(g2['not_before'])}")
    assert seen[-1]["not_before"] == orig_nb


# ══════════════════════════════════════════════════════════════════════════════
# 8. extends 审计记录
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_appends_audit_entry(monkeypatch):
    """延期留痕：old_expire / new_expire / at，且 old≠new、不覆盖历史条目。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant(extends=[{"at": 1.0, "old_expire": 1.0, "new_expire": 2.0}])
    orig_exp = g["expire"]
    new_exp = orig_exp + 30 * DAY
    before = time.time()
    g2, _ = o.extend_grant(g, orig_exp, new_exp, extend_instance="ext_inst_audit")
    after = time.time()

    assert len(g2["extends"]) == 2, "应在既有审计记录后追加一条，而不是覆盖"
    entry = g2["extends"][-1]
    assert entry["old_expire"] == orig_exp, (
        f"old_expire 应为延期前的到期时间：期望 {_fmt(orig_exp)}，实际 {_fmt(entry['old_expire'])}")
    assert entry["new_expire"] == float(new_exp), (
        f"new_expire 应为新到期时间：期望 {_fmt(float(new_exp))}，实际 {_fmt(entry['new_expire'])}")
    assert before <= entry["at"] <= after, f"at 应是本次延期时刻，实际 {_fmt(entry['at'])}"
    assert "ext_inst_audit" in g2["extend_instances"], "延期实例号应落库（幂等判重的依据）"


# ══════════════════════════════════════════════════════════════════════════════
# 9. 整链复现：审批事件 → parse → extend_grant → policy 文档
#    （只桩「拉审批详情」「RAM 客户端」「凭证评论下发」三个出网点，中间全走真代码）
# ══════════════════════════════════════════════════════════════════════════════

_EXT_CODE = "EXTCODE-NB-9999"


def _ext_event(instance="ext_inst_chain"):
    return {"header": {"event_type": "approval_instance"},
            "event": {"approval_code": _EXT_CODE, "status": "APPROVED",
                      "instance_code": instance}}


def _ext_detail(date_interval, *, grant_id="tak-nbtest", enterprise="外采公司A"):
    return {
        "status": "APPROVED", "approval_code": _EXT_CODE, "instance_code": "ext_inst_chain",
        "form": [
            {"name": "凭证ID", "value": grant_id},
            {"name": "撤销/延长", "value": "延长"},
            {"name": "使用企业信息", "value": enterprise},
            {"name": "DateInterval", "value": date_interval},
        ],
    }


def test_approval_chain_old_expire_as_start_does_not_break_credential(monkeypatch):
    """线上事故的整链复现：审批单 DateInterval = 「原到期日 → 新到期日」。

    走真实 `handle_temp_ak_extend_event` → `parse_temp_ak_extend_request`
    （`_parse_date_interval` 把它解析成 not_before/expire）→ `extend_grant` →
    `issuer.rewrite_ram_window` → 真实 policy 文档。断言下发到 RAM 的时间窗**当下有效**。
    """
    from core import ram_approval
    from core.temp_ak_issuance import approval, delivery

    monkeypatch.setattr(approval.settings, "TEMP_AK_EXTEND_APPROVAL_CODE", _EXT_CODE)
    fake = _patch_ram_client(monkeypatch)
    delivered = []
    monkeypatch.setattr(delivery, "deliver_extend",
                        lambda g, c: delivered.append((dict(g), c)))

    g = _issued_grant()
    o._save(g)
    orig_nb, orig_exp = g["not_before"], g["expire"]
    new_exp = orig_exp + 30 * DAY
    # 审批表单里两个日期都按北京时间字符串填 —— 与飞书 DateInterval 控件一致
    monkeypatch.setattr(ram_approval, "fetch_approval_instance", lambda code: _ext_detail(
        {"start": o.fmt_ts(orig_exp), "end": o.fmt_ts(new_exp)}))

    res = approval.handle_temp_ak_extend_event(_ext_event())

    assert res.get("error") is None and res["ignored"] is False, f"延期链路应成功，实际 {res}"
    assert res["action"] == "extend"
    stored = o.get_grant("tak-nbtest")
    assert abs(stored["not_before"] - orig_nb) < 1.0, (
        "整链跑下来生效时间被推到了新窗起点（回归 bug！）："
        f"期望 ≈{_fmt(orig_nb)}，实际 {_fmt(stored['not_before'])}")
    assert abs(stored["expire"] - new_exp) < 1.0, "到期时间应延长到新值"

    now_iso = policy.iso8601_bj(time.time())
    assert fake.docs, "应有一份新 policy 版本下发到 RAM"
    for st in fake.docs[0]["Statement"]:
        nb = st["Condition"]["DateGreaterThan"]["acs:CurrentTime"]
        exp = st["Condition"]["DateLessThan"]["acs:CurrentTime"]
        assert nb <= now_iso < exp, (
            f"延期通过的这一刻凭证反而不可用：policy 窗口 {nb} → {exp}，now={now_iso}")

    # 回执正文里的有效期也该是「原生效时间 → 新到期时间」，不能写成未来才生效
    text = delivery._extended_text(stored)
    assert o.fmt_ts(orig_nb) in text, f"延期回执应显示原生效时间 {o.fmt_ts(orig_nb)}，实际正文：\n{text}"
    assert o.fmt_ts(new_exp) in text, "延期回执应显示新到期时间"


# ══════════════════════════════════════════════════════════════════════════════
# 10. 尚未生效的凭证：允许整体往后挪档期（2026-09-23 刻意放开的那一半语义）
#
# 钳制的目的是「已经在用的凭证不能在延期通过那一刻失效」。对**还没生效**的凭证，
# 这个理由不成立：一律钳制会把「10-01→10-31 改批成 11-01→11-30」压回 10-01 生效，
# 比最新审批批准的起点**早一个月**，等于凭证提前一个月可用 —— 是扩权，不是保守。
# 源码里对应 `elif effective_nb <= now:` 那个条件，本节锁它的 else 分支。
# ══════════════════════════════════════════════════════════════════════════════

def _pending_grant(**over):
    """已发放但**尚未生效**的方案 B grant：窗口整段在未来（例：今天 09-23，窗口 10-01→10-31）。"""
    now = time.time()
    g = _issued_grant(grant_id="tak-pending",
                      user_name="tempak-pending", policy_name="temp-ak-auto-tempak-pending",
                      not_before=now + 8 * DAY,      # 8 天后才生效
                      expire=now + 38 * DAY)         # 生效后一个月到期
    g.update(over)
    return g


def test_extend_not_yet_effective_allows_moving_start_forward(monkeypatch):
    """未生效 + 表单填了更晚的生效时间 → **采纳表单值**，不压回原值。"""
    seen = _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    new_nb, new_exp = orig_exp + DAY, orig_exp + 30 * DAY    # 整段挪到下一个月

    g2, creds = o.extend_grant(g, new_nb, new_exp, extend_instance="ext_pending_1")

    assert creds is None, "方案 B 延期不重发凭证"
    assert g2["not_before"] == float(new_nb), (
        "未生效的凭证被错误钳制：期望采纳表单的新生效时间 "
        f"{_fmt(float(new_nb))}，实际 {_fmt(g2['not_before'])}（原值 {_fmt(orig_nb)}）—— "
        "压回原值 = 凭证比最新审批批准的起点提前生效")
    assert g2["expire"] == float(new_exp)
    assert seen and seen[-1]["not_before"] == float(new_nb), (
        "下发 policy 用的 grant 里 not_before 也必须是新值，"
        f"实际 {_fmt(seen[-1]['not_before'] if seen else None)}")


def test_extend_not_yet_effective_window_stays_in_future(monkeypatch):
    """语义断言：挪档期后窗口整段仍在未来（凭证不会因延期而提前可用）。"""
    _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    orig_exp = g["expire"]
    g2, _ = o.extend_grant(g, orig_exp + DAY, orig_exp + 30 * DAY)
    assert g2["not_before"] > time.time(), (
        f"挪档期后生效时间落到了当下之前：{_fmt(g2['not_before'])}")


def test_extend_not_yet_effective_policy_window_uses_new_start(monkeypatch):
    """到 policy 文档这一层：DateGreaterThan 必须是**新**生效时间。

    grant 字典只是中间态，真正决定「什么时候开始能用」的是下发到 RAM 的那个 acs:CurrentTime 条件。
    """
    monkeypatch.setattr(issuer, "issue", lambda g: pytest.fail("方案 B 延期不该重签发凭证"))
    fake = _patch_ram_client(monkeypatch)
    g = _pending_grant()
    orig_exp = g["expire"]
    new_nb, new_exp = orig_exp + DAY, orig_exp + 30 * DAY

    o.extend_grant(g, new_nb, new_exp)

    assert len(fake.docs) == 1
    want_nb, want_exp = policy.iso8601_bj(new_nb), policy.iso8601_bj(new_exp)
    for st in fake.docs[0]["Statement"]:
        cond = st["Condition"]
        assert cond["DateGreaterThan"]["acs:CurrentTime"] == want_nb, (
            f"policy 生效时间不是新档期起点：期望 {want_nb}，"
            f"实际 {cond['DateGreaterThan']['acs:CurrentTime']}（Action={st.get('Action')}）")
        assert cond["DateLessThan"]["acs:CurrentTime"] == want_exp


def test_extend_not_yet_effective_without_form_start_keeps_future_start(monkeypatch):
    """未生效 + 表单没填 start → 沿用原（未来的）生效时间，**不**被拉到 now。

    `if not not_before: not_before = effective_nb`，而 effective_nb 用的是原值，不是 now。
    拉到 now 会让凭证立刻可用，同样是扩权。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    orig_nb = g["not_before"]
    g2, _ = o.extend_grant(g, 0, g["expire"] + 30 * DAY)
    assert g2["not_before"] == orig_nb, (
        f"未填 start 时应沿用原生效时间 {_fmt(orig_nb)}，实际 {_fmt(g2['not_before'])}")
    assert g2["not_before"] > time.time(), "沿用后仍应在未来"


def test_extend_not_yet_effective_earlier_start_is_accepted(monkeypatch):
    """未生效 + 表单填更早的生效时间（仍在未来）→ 采纳（else 分支不做任何方向限制）。

    注意这条方向上是「更早 = 更宽」，但它来自新审批单、是审批人批准的档期，与已生效凭证
    走 min() 得到的结果一致，不引入新的扩权面。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    earlier = g["not_before"] - 3 * DAY          # 仍 > now（原窗口 8 天后才开始）
    assert earlier > time.time()
    g2, _ = o.extend_grant(g, earlier, g["expire"] + DAY)
    assert g2["not_before"] == float(earlier)


def test_extend_already_effective_still_clamped(monkeypatch):
    """对照组（回归确认没被放开语义带偏）：**已生效**的凭证 + 表单填未来生效时间 → 仍钳制回原值。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()                           # not_before = 7 天前（已生效）
    orig_nb = g["not_before"]
    future_start = time.time() + 5 * DAY
    g2, _ = o.extend_grant(g, future_start, g["expire"] + 30 * DAY)
    assert g2["not_before"] == orig_nb, (
        "已生效的凭证必须钳制：期望保持 "
        f"{_fmt(orig_nb)}，实际 {_fmt(g2['not_before'])}（表单填的是 {_fmt(future_start)}）")
    assert g2["not_before"] <= time.time(), "已生效凭证延期后仍须当下有效"


# ══════════════════════════════════════════════════════════════════════════════
# 11. extends 审计项记 old_not_before / new_not_before
#     钳制真发生时，事后要能回溯「审批单写的是 11-01、实际窗口是 10-01」
# ══════════════════════════════════════════════════════════════════════════════

def test_extends_audit_records_not_before_pair_when_clamped(monkeypatch):
    """已生效 + 表单填未来 start（被钳制）→ old/new_not_before 都落审计，且都等于原值。

    两个值相等本身就是「本次钳制生效了」的信号：审批单写的起点在 extends 里查不到，
    只能靠 approval 侧的实例记录对，所以这两个键缺一不可。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    g2, _ = o.extend_grant(g, time.time() + 5 * DAY, orig_exp + 30 * DAY)
    entry = g2["extends"][-1]
    assert "old_not_before" in entry and "new_not_before" in entry, (
        f"extends 审计项缺生效时间字段：{entry}")
    assert entry["old_not_before"] == orig_nb
    assert entry["new_not_before"] == orig_nb, (
        f"钳制后 new_not_before 应等于原值，实际 {_fmt(entry['new_not_before'])}")
    assert entry["old_expire"] == orig_exp and entry["new_expire"] == float(orig_exp + 30 * DAY)


def test_extends_audit_records_moved_window(monkeypatch):
    """未生效 + 整体挪档期 → old/new_not_before 记的是**不同**的两个值（挪动留痕）。"""
    _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    new_nb, new_exp = orig_exp + DAY, orig_exp + 30 * DAY
    g2, _ = o.extend_grant(g, new_nb, new_exp)
    entry = g2["extends"][-1]
    assert entry["old_not_before"] == orig_nb
    assert entry["new_not_before"] == float(new_nb), (
        f"挪档期后 new_not_before 应为新起点 {_fmt(float(new_nb))}，实际 {_fmt(entry['new_not_before'])}")
    assert entry["old_not_before"] != entry["new_not_before"]


def test_extends_audit_not_before_pair_on_legacy_record(monkeypatch):
    """老记录（无 not_before）：old_not_before 如实记 None，new_not_before 记兜底后的 now。

    不要把 old 也写成兜底值 —— 那会抹掉「这条记录本来就没有生效时间」这个事实。
    """
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    g.pop("not_before")
    g2, _ = o.extend_grant(g, 0, g["expire"] + 30 * DAY)
    entry = g2["extends"][-1]
    assert entry["old_not_before"] is None, f"应如实记 None，实际 {entry['old_not_before']!r}"
    assert entry["new_not_before"] == g2["not_before"]


def test_extends_audit_persisted(monkeypatch):
    """审计项要落 Redis —— 事后回溯读的是落盘记录，不是内存里的那份。"""
    _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    orig_nb = g["not_before"]
    new_nb = g["expire"] + DAY
    o.extend_grant(g, new_nb, g["expire"] + 30 * DAY)
    stored = o.get_grant("tak-pending")
    entry = stored["extends"][-1]
    assert entry["old_not_before"] == orig_nb and entry["new_not_before"] == float(new_nb)


# ══════════════════════════════════════════════════════════════════════════════
# 12. 新到期时间必须晚于原到期时间（2026-09-23 新增校验）
#
# 没有这条校验时，「新生效日 → 原到期日」的表单会走完全程：钳制把两端都压回原值 →
# 一次**完完全全的空操作**，却记一条 extends、还给使用方发「有效期已延长」。
# 安全侧无害（只会更窄），但使用方会以为延期成功、到点直接断。
# 关键：该校验必须**跑在钳制之前**，否则被压成空操作后就没法区分了。
# ══════════════════════════════════════════════════════════════════════════════

def test_extend_expire_equal_to_old_rejected(monkeypatch):
    """边界：新到期 == 原到期 → 拒（校验是 <=，不是 <）。"""
    seen = _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    with pytest.raises(o.TempAkError, match="不晚于原到期时间"):
        o.extend_grant(g, 0, orig_exp)
    assert g["not_before"] == orig_nb and g["expire"] == orig_exp, "抛错路径不得改写时间窗"
    assert "extends" not in g, "被拒的延期不该留审计记录"
    assert not seen, "被拒的延期不该下发 policy"


def test_extend_expire_earlier_than_old_rejected(monkeypatch):
    """新到期早于原到期（但仍在未来）→ 拒。缩短有效期不是「延长」，要走撤销重发。"""
    seen = _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()                                   # 原到期 = now + 1d
    orig_exp = g["expire"]
    shorter = time.time() + 0.5 * DAY
    assert time.time() < shorter < orig_exp
    with pytest.raises(o.TempAkError, match="不晚于原到期时间"):
        o.extend_grant(g, 0, shorter)
    assert g["expire"] == orig_exp
    assert not seen


def test_extend_form_new_start_to_old_expire_rejected_before_clamping(monkeypatch):
    """**校验顺序**锁：表单填「新生效日 → 原到期日」必须报错，不能被钳制悄悄变成空操作。

    这形状能同时通过前两道校验（expire 在未来、start < expire），全靠新校验拦下：
      · 钳制若先跑 → not_before 压回原值、expire 原样 → 空操作 + 一条 extends +
        「有效期已延长」回执 → 使用方以为延期了，到原到期日直接断。
    """
    seen = _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_nb, orig_exp = g["not_before"], g["expire"]
    form_start = time.time() + 0.25 * DAY                 # 落在 now 与原到期之间
    assert time.time() < form_start < orig_exp

    with pytest.raises(o.TempAkError, match="不晚于原到期时间"):
        o.extend_grant(g, form_start, orig_exp)

    assert g["not_before"] == orig_nb and g["expire"] == orig_exp
    assert "extends" not in g and "extend_instances" not in g
    assert not seen, "空操作延期绝不能下发 policy / 触发回执"
    assert o.get_grant("tak-nbtest") is None, "被拒的延期不该落盘"


def test_extend_expire_one_second_later_accepted(monkeypatch):
    """边界另一侧：原到期 + 1 秒 → 通过（校验是 `<=`，多一秒就算延长）。"""
    seen = _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    orig_exp = g["expire"]
    g2, _ = o.extend_grant(g, 0, orig_exp + 1)
    assert g2["expire"] == float(orig_exp + 1), (
        f"期望 {_fmt(float(orig_exp + 1))}，实际 {_fmt(g2['expire'])}")
    assert len(g2["extends"]) == 1 and seen, "通过的延期应留审计并下发 policy"


def test_extend_pending_grant_expire_not_later_rejected(monkeypatch):
    """未生效的凭证同样受这条校验约束（放开钳制没有把它一起放开）。"""
    seen = _stub_ram_rewrite(monkeypatch)
    g = _pending_grant()
    orig_exp = g["expire"]
    with pytest.raises(o.TempAkError, match="不晚于原到期时间"):
        o.extend_grant(g, g["not_before"] + DAY, orig_exp)
    assert g["expire"] == orig_exp
    assert not seen


def test_extend_missing_old_expire_not_blocked(monkeypatch):
    """老记录没有 expire 键 → `float(old_expire or 0)` 兜底 0，任何未来到期都算延长、不误拒。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    target = g.pop("expire") + 10 * DAY
    g2, _ = o.extend_grant(g, 0, target)
    assert g2["expire"] == float(target)
    assert g2["extends"][-1]["old_expire"] is None


def test_extend_expire_in_past_checked_before_old_expire_rule(monkeypatch):
    """校验顺序（另一对）：已过的到期时间报「已过」，不报「不晚于原到期」—— 错的提示会让人改错表单。"""
    _stub_ram_rewrite(monkeypatch)
    g = _issued_grant()
    with pytest.raises(o.TempAkError, match="新到期时间已过"):
        o.extend_grant(g, 0, time.time() - 100)
