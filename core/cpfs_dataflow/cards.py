"""Feishu cards：CPFS 数据预热/沉降（录入 / 确认 / 进度 / 结果）。"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


def _op_options():
    return [
        {"text": _pt("预热（OSS→CPFS 加载）"), "value": "preheat"},
        {"text": _pt("沉降（CPFS→OSS 刷回）"), "value": "sink"},
    ]


def _region_options(open_id: str = ""):
    try:
        from core.cpfs_dataflow import discovery
        return [{"text": _pt(r), "value": r} for r in discovery.regions(open_id=open_id)]
    except Exception:
        return []


def entry_card(region_options=None):
    """填 源地址 + 目的地址 + 选地区 → 后端建 DataFlow + 迁移任务。方向由地址类型自动判断。"""
    if region_options is None:
        region_options = _region_options()
    region_elem = (
        {"tag": "select_static", "name": "region", "placeholder": _pt("选择地区"),
         "options": region_options}
        if region_options else
        {"tag": "input", "name": "region", "label": _pt("地区"), "required": True,
         "placeholder": _pt("如 cn-hangzhou")}
    )
    intro = ("填**源地址**、**目的地址**并选地区 → 解析预览 → 确认下发时才建 DataFlow + 任务（用完自动删）。\n"
             "• 源 CPFS → 目的 OSS = **沉降**；源 OSS → 目的 CPFS = **预热**\n"
             "• CPFS：`/cpfs/<dir>/` 或 `cpfs://<fs-id>/<dir>/`；OSS：`oss://<bucket>/<prefix>/`\n"
             "> CPFS 与 OSS 须同地区。镜像仓库（cri 开头）请勿填。")
    form_elems = [
        region_elem,
        {"tag": "input", "name": "source", "label": _pt("源地址"), "required": True,
         "placeholder": _pt("如 /cpfs/cwr/label/ 或 oss://bk/prefix/")},
        {"tag": "input", "name": "dest", "label": _pt("目的地址"), "required": True,
         "placeholder": _pt("如 oss://wuji-bucket-hangzhou/wuji_il/ 或 /cpfs/cwr/label/")},
        {"tag": "button", "text": _pt("➡️ 解析预览"), "type": "primary",
         "form_action_type": "submit", "name": "submit",
         "behaviors": [{"type": "callback", "value": {"action": "submit_cpfs_dataflow"}}]},
    ]
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f504 CPFS 数据预热 / 沉降"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": intro},
            {"tag": "form", "name": "cpfs_entry", "elements": form_elems},
        ]},
    }


def query_input_card():
    """查询进度：弹一张带输入框的表单卡，让用户填任务 ID（cpfs-/tr-）再查。"""
    form_elems = [
        {"tag": "input", "name": "job_id", "label": _pt("任务 ID"), "required": True,
         "placeholder": _pt("如 cpfs-xxxxxx 或 tr-xxxxxx")},
        {"tag": "button", "text": _pt("\U0001f504 查询"), "type": "primary",
         "form_action_type": "submit", "name": "submit",
         "behaviors": [{"type": "callback", "value": {"action": "query_progress_by_id"}}]},
    ]
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f504 查询任务进度"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown",
             "content": "填入**任务 ID** 查询进度（发起任务时的确认卡 / 进度卡上都带了 ID）。"},
            {"tag": "form", "name": "query_progress", "elements": form_elems},
        ]},
    }


def confirm_card(job: dict):
    info_md = (
        f"**任务ID**：`{job['job_id']}`\n"
        f"**操作**：{job['operation_label']}\n"
        f"**文件系统**：`{job['fs_id']}`（{job.get('edition','')}）　**区域**：{job['region']}\n"
        f"**CPFS 目录**：`{job['cpfs_dir']}`\n"
        f"**OSS**：`{job.get('oss_bucket') or '-'}/{job.get('oss_prefix','')}`\n"
        f"**Directory**：`{job['directory']}`　**DstDirectory**：`{job.get('dst_directory') or '(default)'}`"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f504 预热/沉降确认"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": info_md},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
             "behaviors": [{"type": "callback",
                            "value": {"action": "confirm_cpfs_dataflow", "job_id": job["job_id"]}}]},
        ]},
    }


def progress_card(job: dict):
    return card(f"⏳ {job['operation_label']}进行中", [
        fields(("任务ID", f"`{job['job_id']}`"), ("阶段", job["stage"]),
               ("TaskId", f"`{job.get('task_id') or '-'}`"),
               ("文件", f"{job.get('files_done',0)}/{job.get('files_total',0)}")),
        div(f"**fs** `{job['fs_id']}`\n**目录** `{job['directory']}`"),
        hr(),
        buttons(btn("\U0001f504 查询进度", {"action": "query_cpfs_progress", "job_id": job["job_id"]})),
    ], color="blue")


def result_card(job: dict):
    from core.cpfs_dataflow.orchestrator import fmt_size, fmt_ts, fmt_duration
    ok = job["stage"] == "DONE"
    created = job.get("created_ts", 0)
    finished = job.get("finished_ts", 0)
    if ok:
        return card(f"✅ {job['operation_label']}完成", [
            fields(("任务ID", f"`{job['job_id']}`"), ("TaskId", f"`{job.get('task_id') or '-'}`"),
                   ("文件数", str(job.get("files_done", 0))),
                   ("数据量", fmt_size(job.get("bytes_done", 0))),
                   ("耗时", fmt_duration(created, finished))),
            div(f"**fs** `{job['fs_id']}`\n**目录** `{job['directory']}`"),
        ], color="green")
    return card(f"❌ {job['operation_label']}失败", [
        fields(("任务ID", f"`{job['job_id']}`"), ("TaskId", f"`{job.get('task_id') or '-'}`"),
               ("阶段", job["stage"]), ("结束", fmt_ts(finished))),
        div(f"**fs** `{job['fs_id']}`\n**目录** `{job['directory']}`\n"
            f"**失败原因**：{job.get('error') or '未知'}"),
        hr(),
        buttons(btn("\U0001f501 重试", {"action": "retry_cpfs_dataflow", "job_id": job["job_id"]}, "danger")),
    ], color="red")
