"""PFS 跨云直传 —— cards：entry/confirm(2.0 审批门)/progress_card_v2(2.0 无按钮)/result_card(1.0 retry)。

200830 规避：confirm/progress 都是 schema 2.0（同家族原地替换）；result 是 1.0（新推，不原地替换）。
"""
import json

from core.pfs_transfer import cards


def _job(stage="SINKING", **over):
    j = {
        "job_id": "xpfs-abc123def",
        "chain": "pfs",
        "direction": "vepfs2cpfs",
        "stage": stage,
        "created_by": "ou_x",
        "created_ts": 1000.0,
        "finished_ts": 1600.0,
        "error": "",
        "bytes_done": 50,
        "bytes_total": 100,
        "src_pfs": {"scheme": "vepfs", "fs_id": "vepfs-a", "sub_path": "wzh/", "region": "cn-beijing"},
        "dst_pfs": {"scheme": "cpfs", "fs_id": "cpfs-b", "sub_path": "team/", "region": "cn-hangzhou"},
        "src_staging": {"scheme": "tos", "bucket": "b1", "prefix": "p/xpfs-abc123def/",
                        "region": "cn-beijing", "dataflow_id": ""},
        "dst_staging": {"scheme": "oss", "bucket": "b2", "prefix": "p/xpfs-abc123def/",
                        "region": "cn-hangzhou", "dataflow_id": "df-1"},
        "sink_done": False, "cross_done": False, "preheat_done": False,
    }
    j.update(over)
    return j


# ══════════════════════════════════════════════════════════════════════════════
# entry_card
# ══════════════════════════════════════════════════════════════════════════════

def test_entry_card_schema_2_and_form():
    c = cards.entry_card()
    assert c["schema"] == "2.0"
    blob = json.dumps(c, ensure_ascii=False)
    assert "submit_pfs_transfer" in blob
    # 源/目的地址两个输入
    assert '"name": "source"' in blob and '"name": "dest"' in blob


def test_entry_card_title_mentions_pfs():
    c = cards.entry_card()
    assert "PFS" in json.dumps(c, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# confirm_card（2.0 + 审批门）
# ══════════════════════════════════════════════════════════════════════════════

def test_confirm_card_schema_2_normal():
    c = cards.confirm_card(_job(), need_approval=False)
    assert c["schema"] == "2.0"
    assert c["header"]["template"] == "orange"
    blob = json.dumps(c, ensure_ascii=False)
    assert "confirm_pfs_transfer" in blob
    assert "xpfs-abc123def" in blob


def test_confirm_card_approval_red_and_admin_hint():
    c = cards.confirm_card(_job(), need_approval=True)
    assert c["header"]["template"] == "red"
    assert "管理员" in json.dumps(c, ensure_ascii=False)


def test_confirm_card_no_missing_field_crash():
    """缺 dataflow_id 等可选字段不 KeyError。"""
    j = _job()
    j["src_staging"].pop("dataflow_id", None)
    cards.confirm_card(j, need_approval=False)   # 不抛即通过


# ══════════════════════════════════════════════════════════════════════════════
# progress_card_v2（2.0 纯展示，无按钮/表单）
# ══════════════════════════════════════════════════════════════════════════════

def test_progress_card_v2_schema_2_no_button():
    c = cards.progress_card_v2(_job(stage="CROSSING"))
    assert c["schema"] == "2.0"
    blob = json.dumps(c, ensure_ascii=False)
    assert '"tag": "button"' not in blob
    assert '"tag": "form"' not in blob


def test_progress_card_v2_contains_job_id_and_query_hint():
    c = cards.progress_card_v2(_job(stage="CROSSING"))
    blob = json.dumps(c, ensure_ascii=False)
    assert "xpfs-abc123def" in blob
    assert "查询进度" in blob         # 无按钮 2.0 卡的文本查法提示


def test_progress_card_v2_all_stages_no_crash():
    for st in ("SINKING", "CROSSING", "PREHEATING"):
        cards.progress_card_v2(_job(stage=st))


# ══════════════════════════════════════════════════════════════════════════════
# result_card（1.0 + FAILED 带 retry）
# ══════════════════════════════════════════════════════════════════════════════

def test_result_card_done_is_1_0_no_retry():
    c = cards.result_card(_job(stage="DONE"))
    assert "schema" not in c            # 1.0（tools.feishu.cards.card 不带 schema）
    assert c["header"]["template"] == "green"
    assert "retry_pfs_transfer" not in json.dumps(c, ensure_ascii=False)


def test_result_card_failed_has_retry_button():
    c = cards.result_card(_job(stage="FAILED", error="跨云失败"))
    assert "schema" not in c
    assert c["header"]["template"] == "red"
    blob = json.dumps(c, ensure_ascii=False)
    assert "retry_pfs_transfer" in blob
    assert "xpfs-abc123def" in blob
    assert "跨云失败" in blob


def test_result_card_failed_missing_error_shows_unknown():
    j = _job(stage="FAILED")
    j.pop("error", None)
    c = cards.result_card(j)
    assert "未知" in json.dumps(c, ensure_ascii=False)


def test_result_card_missing_optional_no_crash():
    j = _job(stage="DONE")
    j.pop("created_ts", None)
    j.pop("finished_ts", None)
    cards.result_card(j)     # 不抛即通过
