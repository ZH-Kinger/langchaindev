"""Feishu cards：火山 vePFS 数据预热/沉降（录入 / 确认 / 进度 / 结果）。"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


# 录入入口（选云→选地区→选文件系统+填地址）已统一到 core/dataflow_cards.py；
# 本模块只保留 vePFS 侧的 确认/进度/结果 卡。


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


def progress_card_v2(job: dict):
    """schema 2.0 纯展示"进行中"卡：确认卡是 2.0，回调里原地替换须同 schema 家族（2.0→2.0）。
    用 1.0 的 progress_card 替换 2.0 确认卡会被飞书拒（错误 200830，静默失败，确认到完成间无反馈）。
    无按钮——按钮随确认消失，终态由后台线程推结果卡。"""
    info_md = (
        f"**任务**：`{job['job_id']}`\n"
        f"**操作**：{job.get('operation_label','')}\n"
        f"**阶段**：{job['stage']}\n"
        f"**vePFS** `{job.get('sub_path','')}`\n"
        f"**TOS** `{job.get('tos_bucket','')}/{job.get('tos_prefix','')}`\n"
        f"> 已在后台运行，完成后推送结果。"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("⏳ 数据流动进行中"), "template": "blue"},
        "body": {"elements": [{"tag": "markdown", "content": info_md}]},
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
