"""#51 续 —— cards 四件套结构：entry/confirm/progress_card_v2(2.0) + result_card(1.0)。

progress_card_v2 无 form/button 且含 progress_line；result_card DONE 无 schema 键、
FAILED 含 retry_ssh_transfer 按钮；缺字段不 KeyError。
"""
import pytest

from config.settings import settings
from core.ssh_transfer import cards


def _form_of(card_dict):
    """从 entry_card body 里取出 tag==form 的元素（不假设固定下标）。"""
    for el in card_dict["body"]["elements"]:
        if el.get("tag") == "form":
            return el
    raise AssertionError("entry_card 里没有 form 元素")


def _input_names(form):
    return {el.get("name") for el in form["elements"] if el.get("tag") == "input"}


def _submit_button(form):
    for el in form["elements"]:
        if el.get("tag") == "button":
            return el
    raise AssertionError("form 里没有 button")


def _full_job(stage="STAGE1", **over):
    j = {
        "job_id": "sgp-x", "stage": stage,
        "source_uri": "oss://wuji-data-tran/ossutil_output/",
        "source_prefix": "ossutil_output/",
        "dest_uri": "wuji@host:/root/data/test/",
        "bytes_total": 123, "objects_total": 4, "estimate_ok": True,
        "bytes_done": 60, "speed_bps": 30, "created_ts": 1000, "finished_ts": 1120,
    }
    j.update(over)
    return j


# ── entry_card ─────────────────────────────────────────────────────────────────

def test_entry_card_schema_2_0():
    c = cards.entry_card()
    assert c["schema"] == "2.0"
    s = str(c)
    assert "source" in s and "dest" in s              # 源 + 目标子目录输入
    assert "submit_ssh_transfer" in s                 # 提交回调


def test_entry_card_structure_regression():
    """回归：form 的 input 名仍是 source/dest，提交按钮回调仍 submit_ssh_transfer。"""
    c = cards.entry_card()
    form = _form_of(c)
    assert _input_names(form) == {"source", "dest"}
    btn_el = _submit_button(form)
    behaviors = btn_el["behaviors"]
    assert behaviors[0]["value"]["action"] == "submit_ssh_transfer"


def test_entry_card_shows_thai_dest_root(monkeypatch):
    """正文含「泰国目标根」+ user@host:dest_root/ 拼接；placeholder 含 dest_root。"""
    monkeypatch.setattr(settings, "THAI_USER", "wuji")
    monkeypatch.setattr(settings, "THAI_HOST", "1.2.3.4")
    monkeypatch.setattr(settings, "THAI_DEST_ROOT", "/mnt/data/team/")  # 带尾斜杠，验证 rstrip
    c = cards.entry_card()
    md = c["body"]["elements"][0]["content"]
    assert "泰国目标根" in md
    assert "/mnt/data/team" in md                      # dest_root 值出现（尾斜杠已 rstrip）
    assert "wuji@1.2.3.4:/mnt/data/team/" in md        # user@host:dest_root/ 拼接（与确认卡 dest_uri 同形）
    # placeholder 含 dest_root
    form = _form_of(c)
    dest_input = next(el for el in form["elements"]
                      if el.get("tag") == "input" and el.get("name") == "dest")
    placeholder = dest_input["placeholder"]["content"]
    assert "/mnt/data/team" in placeholder
    assert "/mnt/data/team/test/" in placeholder       # 「如 test → 落到 <root>/test/」


def test_entry_card_empty_dest_root_no_crash(monkeypatch):
    """THAI_DEST_ROOT 为空 → rstrip 空串安全、不炸，仍 schema 2.0。"""
    monkeypatch.setattr(settings, "THAI_DEST_ROOT", "")
    monkeypatch.setattr(settings, "THAI_HOST", "")
    c = cards.entry_card()
    assert c["schema"] == "2.0"
    assert "泰国目标根" in c["body"]["elements"][0]["content"]
    # 结构未坏：input 名仍在
    assert _input_names(_form_of(c)) == {"source", "dest"}


def test_entry_card_dest_root_none_no_crash(monkeypatch):
    """THAI_DEST_ROOT / THAI_HOST 为 None → `(x or "")` 兜底、不炸。"""
    monkeypatch.setattr(settings, "THAI_DEST_ROOT", None)
    monkeypatch.setattr(settings, "THAI_HOST", None)
    c = cards.entry_card()
    assert c["schema"] == "2.0"


# ── confirm_card ───────────────────────────────────────────────────────────────

def test_confirm_card_schema_2_0_orange():
    c = cards.confirm_card(_full_job(), need_approval=False)
    assert c["schema"] == "2.0"
    assert c["header"]["template"] == "orange"
    assert "confirm_ssh_transfer" in str(c)


def test_confirm_card_approval_red():
    c = cards.confirm_card(_full_job(), need_approval=True)
    assert c["header"]["template"] == "red"
    assert "管理员" in str(c)


def test_confirm_card_estimate_failed_label():
    c = cards.confirm_card(_full_job(estimate_ok=False), need_approval=True)
    assert "估算失败" in str(c)


def test_confirm_card_missing_optional_no_keyerror():
    """无 bytes_total/objects_total/estimate_ok → 不抛 KeyError。"""
    job = {"job_id": "sgp-x", "source_uri": "oss://b/p/",
           "source_prefix": "p/", "dest_uri": "u@h:/x/"}
    c = cards.confirm_card(job)
    assert c["schema"] == "2.0"


# ── progress_card_v2 ───────────────────────────────────────────────────────────

def test_progress_card_v2_schema_no_form_button_has_progress_line():
    c = cards.progress_card_v2(_full_job(stage="STAGE1"))
    assert c["schema"] == "2.0"
    s = str(c)
    assert "form" not in s and "button" not in s      # 纯展示（避 200830）
    md = c["body"]["elements"][0]["content"]
    assert "sgp-x" in md
    assert "已传" in md                                # progress_line 内容


def test_progress_card_v2_minimal_no_keyerror():
    """无进度字段 → progress_line 回落「进度采集中…」、不抛。"""
    job = {"job_id": "sgp-x", "stage": "NEW",
           "source_uri": "oss://b/p/", "dest_uri": "u@h:/x/"}
    c = cards.progress_card_v2(job)
    assert c["schema"] == "2.0"
    assert "进度采集中" in str(c)


@pytest.mark.parametrize("stage", ["NEW", "STAGE1", "STAGE2", "DONE", "FAILED", "WHATEVER"])
def test_progress_card_v2_any_stage_no_crash(stage):
    c = cards.progress_card_v2(_full_job(stage=stage))
    assert c["schema"] == "2.0"


# ── result_card ────────────────────────────────────────────────────────────────

def test_result_card_done_is_1_0_no_retry():
    c = cards.result_card(_full_job(stage="DONE"))
    assert "schema" not in c                           # 1.0 卡不写 schema 键
    assert "retry_ssh_transfer" not in str(c)          # 完成卡无重试按钮


def test_result_card_failed_1_0_has_retry_button():
    c = cards.result_card(_full_job(stage="FAILED", error="boom"))
    assert "schema" not in c
    assert "retry_ssh_transfer" in str(c)              # 失败卡带重试按钮


def test_result_card_missing_optional_no_keyerror():
    """DONE 只带必备字段 → fmt_size/fmt_duration 兜底不抛。"""
    c = cards.result_card({"job_id": "sgp-x", "stage": "DONE"})
    assert "schema" not in c


def test_result_card_failed_missing_error_shows_unknown():
    c = cards.result_card({"job_id": "sgp-x", "stage": "FAILED"})
    assert "未知" in str(c)                             # error 缺 → 「未知」
    assert "retry_ssh_transfer" in str(c)
