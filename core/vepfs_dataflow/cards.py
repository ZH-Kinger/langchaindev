"""Feishu cards：火山 vePFS 数据预热/沉降（录入 / 确认 / 进度 / 结果）。"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


# 入口云平台选择在统一卡（core/cpfs_dataflow/cards.py::entry_card 两个云平台按钮）；点「火山」→ guided_card。


def _sel_or_input(name, options, label, placeholder):
    """有 options → 下拉；否则回退成文本输入框（发现为空/失败时不至于卡死）。"""
    if options:
        return {"tag": "select_static", "name": name, "required": False,
                "placeholder": _pt(placeholder),
                "options": [{"text": _pt(o["text"]), "value": o["value"]} for o in options]}
    return {"tag": "input", "name": name, "label": _pt(label), "required": True,
            "placeholder": _pt(placeholder)}


def _op_options():
    return [
        {"text": _pt("沉降（vePFS→TOS 刷回）"), "value": "sink"},
        {"text": _pt("预热（TOS→vePFS 加载）"), "value": "preheat"},
    ]


def _same_name_options():
    return [
        {"text": _pt("跳过同名（默认，永不覆盖）"), "value": "skip"},
        {"text": _pt("保留最新（比最后修改时间）"), "value": "keeplatest"},
        {"text": _pt("覆盖同名"), "value": "overwrite"},
    ]


def guided_card(fs_options=None, bucket_options=None):
    """火山 vePFS↔TOS 向导卡：下拉选 文件系统 + TOS 桶（地区随文件系统带出），填子目录/前缀。

    发现为空时对应项回退文本框。提交动作 submit_vepfs_dataflow（handler 识别 fs/bucket 走向导分支）。
    """
    fs_options = fs_options or []
    bucket_options = bucket_options or []
    intro = ("**火山 vePFS ↔ TOS**：选操作、文件系统、TOS 桶（都在同一地区），填子目录/前缀 → 解析预览。\n"
             "> 文件系统的地区自动带出；只列同地区的 TOS 桶。首次加载稍慢，下拉为空可点「🔄 刷新资源」。")
    form_elems = [
        {"tag": "select_static", "name": "operation", "required": False,
         "placeholder": _pt("操作（默认沉降）"), "options": _op_options()},
        _sel_or_input("fs", fs_options, "vePFS 文件系统（vepfs-...@region）", "选择 vePFS 文件系统"),
        {"tag": "input", "name": "sub_path", "label": _pt("vePFS 子目录"), "required": True,
         "placeholder": _pt("如 /label/")},
        _sel_or_input("bucket", bucket_options, "TOS 桶（bucket@region）", "选择 TOS 桶"),
        {"tag": "input", "name": "tos_prefix", "label": _pt("TOS 前缀"), "required": False,
         "placeholder": _pt("如 archive/（可空）")},
        {"tag": "select_static", "name": "same_name", "required": False,
         "placeholder": _pt("同名策略（默认跳过）"), "options": _same_name_options()},
        {"tag": "button", "text": _pt("➡️ 解析预览"), "type": "primary",
         "form_action_type": "submit", "name": "submit",
         "behaviors": [{"type": "callback", "value": {"action": "submit_vepfs_dataflow"}}]},
    ]
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f30b 火山 vePFS↔TOS 向导"), "template": "orange"},
        "body": {"elements": [
            {"tag": "markdown", "content": intro},
            {"tag": "form", "name": "vepfs_guided", "elements": form_elems},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("🔄 刷新资源"), "type": "default",
             "behaviors": [{"type": "callback", "value": {"action": "refresh_vepfs_options"}}]},
        ]},
    }


def confirm_card(job: dict):
    from core.vepfs_dataflow.orchestrator import _same_name_label
    info_md = (
        f"**任务ID**：`{job['job_id']}`\n"
        f"**操作**：{job['operation_label']}\n"
        f"**文件系统**：`{job['fs_id']}`　**区域**：{job['region']}\n"
        f"**vePFS 子目录**：`{job['sub_path']}`\n"
        f"**TOS**：`{job.get('tos_bucket') or '-'}/{job.get('tos_prefix','')}`\n"
        f"**同名策略**：{_same_name_label(job.get('same_name',''))}"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f30b 预热/沉降确认"), "template": "orange"},
        "body": {"elements": [
            {"tag": "markdown", "content": info_md},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
             "behaviors": [{"type": "callback",
                            "value": {"action": "confirm_vepfs_dataflow", "job_id": job["job_id"]}}]},
        ]},
    }


def progress_card(job: dict):
    return card(f"⏳ {job['operation_label']}进行中", [
        fields(("任务ID", f"`{job['job_id']}`"), ("阶段", job["stage"]),
               ("TaskId", f"`{job.get('task_id') or '-'}`"),
               ("文件", f"{job.get('files_done',0)}/{job.get('files_total',0)}")),
        div(f"**fs** `{job['fs_id']}`\n**vePFS** `{job['sub_path']}`　**TOS** `{job.get('tos_bucket')}/{job.get('tos_prefix','')}`"),
        hr(),
        buttons(btn("\U0001f504 查询进度", {"action": "query_vepfs_progress", "job_id": job["job_id"]})),
    ], color="orange")


def result_card(job: dict):
    from core.vepfs_dataflow.orchestrator import fmt_size, fmt_ts, fmt_duration
    ok = job["stage"] == "DONE"
    created = job.get("created_ts", 0)
    finished = job.get("finished_ts", 0)
    if ok:
        return card(f"✅ {job['operation_label']}完成", [
            fields(("任务ID", f"`{job['job_id']}`"), ("TaskId", f"`{job.get('task_id') or '-'}`"),
                   ("文件数", str(job.get("files_done", 0))),
                   ("数据量", fmt_size(job.get("bytes_done", 0))),
                   ("耗时", fmt_duration(created, finished))),
            div(f"**fs** `{job['fs_id']}`\n**vePFS** `{job['sub_path']}`　**TOS** `{job.get('tos_bucket')}/{job.get('tos_prefix','')}`"),
        ], color="green")
    return card(f"❌ {job['operation_label']}失败", [
        fields(("任务ID", f"`{job['job_id']}`"), ("TaskId", f"`{job.get('task_id') or '-'}`"),
               ("阶段", job["stage"]), ("结束", fmt_ts(finished))),
        div(f"**fs** `{job['fs_id']}`\n**vePFS** `{job['sub_path']}`\n"
            f"**失败原因**：{job.get('error') or '未知'}"),
        hr(),
        buttons(btn("\U0001f501 重试", {"action": "retry_vepfs_dataflow", "job_id": job["job_id"]}, "danger")),
    ], color="red")
