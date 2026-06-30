"""Feishu cards：CPFS 数据预热/沉降（录入 / 确认 / 进度 / 结果）。"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


def _op_options():
    return [
        {"text": _pt("预热（OSS→CPFS 加载）"), "value": "preheat"},
        {"text": _pt("沉降（CPFS→OSS 刷回）"), "value": "sink"},
    ]


def _dataflow_options():
    """从发现的映射表构造 select 选项；失败/为空返回 []（回退手填）。"""
    try:
        from core.cpfs_dataflow import discovery
        opts = discovery.get_options()
    except Exception:
        opts = []
    return [{"text": _pt(o["label"]), "value": o["value"]} for o in opts if o.get("value")]


def entry_card(options=None):
    if options is None:
        options = _dataflow_options()
    submit_btn = {"tag": "button", "text": _pt("➡️ 解析"), "type": "primary",
                  "form_action_type": "submit", "name": "submit",
                  "behaviors": [{"type": "callback", "value": {"action": "submit_cpfs_dataflow"}}]}
    op_select = {"tag": "select_static", "name": "operation", "initial_option": "sink",
                 "placeholder": _pt("选择操作"), "options": _op_options()}
    if options:
        intro = ("选择操作 + CPFS↔OSS 绑定，填相对子目录（基于所选绑定，可留空=整目录）。\n"
                 "> 预热=OSS→CPFS 加载；沉降=CPFS→OSS 刷回。绑定标签含地域，镜像仓库（cri 开头）已屏蔽。")
        form_elems = [
            op_select,
            {"tag": "select_static", "name": "target", "placeholder": _pt("选择 CPFS↔OSS 绑定"),
             "options": options},
            {"tag": "input", "name": "subdir", "label": _pt("相对子目录（可留空）"),
             "placeholder": _pt("如 third_party_data/label/")},
            submit_btn,
        ]
    else:
        intro = ("未发现绑定，请**按规范**手填：\n"
                 "• CPFS：完整路径 `/cpfs/<dir>/`（去挂载前缀）或 `cpfs://<fs-id>/<dir>/`\n"
                 "• OSS：`oss://<bucket>/<prefix>/`（可留空，由绑定决定）\n"
                 "• 地域：与 CPFS/OSS 同区，示例 cn-hangzhou / cn-shanghai / cn-wulanchabu\n"
                 "> 预热=OSS→CPFS；沉降=CPFS→OSS。镜像仓库（cri 开头）请勿填。")
        form_elems = [
            op_select,
            {"tag": "input", "name": "cpfs_path", "label": _pt("CPFS 目录（含 fs 或完整路径）"), "required": True,
             "placeholder": _pt("如 cpfs://bmcpfs-xxx/cwr/label/ 或 /cpfs/cwr/label/")},
            {"tag": "input", "name": "oss", "label": _pt("OSS 路径（可留空）"),
             "placeholder": _pt("如 oss://wuji-bucket-hangzhou/wuji_il/")},
            {"tag": "input", "name": "region", "label": _pt("地域（可留空，默认 CPFS_REGION）"),
             "placeholder": _pt("如 cn-hangzhou")},
            submit_btn,
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


def confirm_card(job: dict):
    info_md = (
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
        fields(("任务", f"`{job['job_id']}`"), ("阶段", job["stage"]),
               ("文件", f"{job.get('files_done',0)}/{job.get('files_total',0)}")),
        div(f"**fs** `{job['fs_id']}`\n**目录** `{job['directory']}`"),
    ], color="blue")


def result_card(job: dict):
    from core.cpfs_dataflow.orchestrator import fmt_size, fmt_ts, fmt_duration
    ok = job["stage"] == "DONE"
    created = job.get("created_ts", 0)
    finished = job.get("finished_ts", 0)
    if ok:
        return card(f"✅ {job['operation_label']}完成", [
            fields(("任务", f"`{job['job_id']}`"),
                   ("文件数", str(job.get("files_done", 0))),
                   ("数据量", fmt_size(job.get("bytes_done", 0))),
                   ("耗时", fmt_duration(created, finished))),
            div(f"**fs** `{job['fs_id']}`\n**目录** `{job['directory']}`"),
        ], color="green")
    return card(f"❌ {job['operation_label']}失败", [
        fields(("任务", f"`{job['job_id']}`"), ("阶段", job["stage"]),
               ("结束", fmt_ts(finished))),
        div(f"**fs** `{job['fs_id']}`\n**目录** `{job['directory']}`\n"
            f"**失败原因**：{job.get('error') or '未知'}"),
        hr(),
        buttons(btn("\U0001f501 重试", {"action": "retry_cpfs_dataflow", "job_id": job["job_id"]}, "danger")),
    ], color="red")
