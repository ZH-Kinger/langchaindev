"""#51 续 —— cards 四件套结构：entry/confirm/progress_card_v2(2.0) + result_card(1.0)。

progress_card_v2 无 form/button 且含 progress_line；result_card DONE 无 schema 键、
FAILED 含 retry_ssh_transfer 按钮；缺字段不 KeyError。
"""
import pytest

from core.ssh_transfer import cards


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
