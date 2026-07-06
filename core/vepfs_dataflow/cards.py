"""Feishu cards：火山 vePFS 数据预热/沉降（录入 / 确认 / 进度 / 结果）。"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


def _same_name_options():
    return [
        {"text": _pt("跳过同名文件（默认，永不覆盖）"), "value": "Skip"},
        {"text": _pt("保留最新（比最后修改时间）"), "value": "KeepLatest"},
        {"text": _pt("覆盖同名文件"), "value": "OverWrite"},
    ]


def entry_card():
    """填 源地址 + 目的地址 + 选地区 + 同名策略 → 后端建数据流动任务。方向由地址类型自动判断。"""
    intro = ("填**源地址**、**目的地址**并选地区 → 解析预览 → 确认下发。\n"
             "• 源 vePFS → 目的 TOS = **沉降**；源 TOS → 目的 vePFS = **预热**\n"
             "• vePFS：`vepfs://<fs-id>/<dir>/`（或裸子目录 `/<dir>/`，用默认文件系统）；TOS：`tos://<bucket>/<prefix>/`\n"
             "> vePFS 与 TOS 须**同地域**；账号需已开通 vePFS→TOS 服务授权 + 数据流动带宽>0。")
    form_elems = [
        {"tag": "input", "name": "region", "label": _pt("地区"), "required": True,
         "placeholder": _pt("如 cn-beijing（vePFS 文件系统所在区域）")},
        {"tag": "input", "name": "source", "label": _pt("源地址"), "required": True,
         "placeholder": _pt("如 vepfs://vepfs-xxx/label/ 或 tos://bk/prefix/")},
        {"tag": "input", "name": "dest", "label": _pt("目的地址"), "required": True,
         "placeholder": _pt("如 tos://bk/prefix/ 或 vepfs://vepfs-xxx/label/")},
        {"tag": "select_static", "name": "same_name", "required": False,
         "placeholder": _pt("同名策略（默认跳过）"), "options": _same_name_options()},
        {"tag": "button", "text": _pt("➡️ 解析预览"), "type": "primary",
         "form_action_type": "submit", "name": "submit",
         "behaviors": [{"type": "callback", "value": {"action": "submit_vepfs_dataflow"}}]},
    ]
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f30b vePFS 数据预热 / 沉降"), "template": "orange"},
        "body": {"elements": [
            {"tag": "markdown", "content": intro},
            {"tag": "form", "name": "vepfs_entry", "elements": form_elems},
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
