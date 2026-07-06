"""飞书卡片：桶间迁移（录入 / 确认 / 进度 / 结果）。"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(s):
    return {"tag": "plain_text", "content": s}


def _same_name_options():
    return [
        {"text": _pt("跳过同名文件（默认，永不覆盖）"), "value": "skip"},
        {"text": _pt("覆盖同名文件"), "value": "overwrite"},
    ]


def entry_card():
    """发「桶间迁移」→ 填源/目的（同云同 scheme）+ 同名策略 �� 解析并预估。"""
    intro = ("**同云桶间迁移**（一次性搬运现有数据）。填源、目的地址（都必填、目录以 `/` 结尾）。\n"
             "• 阿里 `oss://桶/前缀/` → `oss://桶/前缀/`；火山 `tos://桶/前缀/` → `tos://桶/前缀/`\n"
             "> 只做**同云**（oss→oss / tos→tos）；跨云 oss↔tos 请用「跨云迁移」。\n"
             "> ⚠️ 火山目的只到**桶级**，对象保持源 key 原样落入目的桶（不改子目录结构）。")
    form_elems = [
        {"tag": "input", "name": "source", "label": _pt("源地址"), "required": True,
         "placeholder": _pt("如 oss://src-bucket/data/ 或 tos://src-bucket/data/")},
        {"tag": "input", "name": "dest", "label": _pt("目的地址"), "required": True,
         "placeholder": _pt("如 oss://dst-bucket/data/ 或 tos://dst-bucket/data/")},
        {"tag": "select_static", "name": "same_name", "required": False,
         "placeholder": _pt("同名策略（默认跳过）"), "options": _same_name_options()},
        {"tag": "button", "text": _pt("➡️ 解析并预估"), "type": "primary",
         "form_action_type": "submit", "name": "submit",
         "behaviors": [{"type": "callback", "value": {"action": "submit_bucket_transfer"}}]},
    ]
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f69a 桶间迁移（同云搬运）"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": intro},
            {"tag": "form", "name": "bucket_transfer_entry", "elements": form_elems},
        ]},
    }


def _cloud_label(job):
    return "阿里 OSS" if job.get("engine") == "mgw" else "火山 TOS"


def confirm_card(job):
    from core.bucket_transfer.orchestrator import same_name_policy_label
    warn = ""
    if job.get("engine") == "dms":
        dp = job.get("dest_prefix") or ""
        if dp:
            # 用户填了目的子目录，但火山会忽略 → 明确警告，避免误以为能落到该子目录
            warn = (f"\n\n⚠️ **火山不支持指定目的子目录**：你填的 `{dp}` 会被**忽略**，对象保持源 key "
                    f"原样落到桶 `{job['dest_bucket']}`（如 `{job['dest_bucket']}/{job.get('src_prefix','')}…`）。"
                    f"如需固定目录，请迁完自行 rename。")
        else:
            warn = "\n\n> 火山目的只到桶级，对象保持源 key 结构落入目的桶。"
    info = (
        f"**任务ID**：`{job['job_id']}`\n"
        f"**云**：{_cloud_label(job)}（{job['direction']}）\n"
        f"**源**：`{job['source']}`\n"
        f"**目的**：`{job['dest']}`\n"
        f"**同名策略**：{same_name_policy_label(job.get('same_name',''))}"
        + warn)
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f69a 桶间迁移确认"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown", "content": info},
            {"tag": "hr"},
            {"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
             "behaviors": [{"type": "callback",
                            "value": {"action": "confirm_bucket_transfer", "job_id": job["job_id"]}}]},
        ]},
    }


def progress_card(job):
    from core.bucket_transfer.orchestrator import fmt_size
    return card(f"⏳ 桶间迁移进行中（{job['direction']}）", [
        fields(("任务ID", f"`{job['job_id']}`"), ("阶段", job["stage"]),
               ("已传对象", str(job.get("objects", 0))),
               ("已传数据", fmt_size(job.get("bytes", 0)))),
        div(f"**源** `{job['source']}`\n**目的** `{job['dest']}`"),
        hr(),
        buttons(btn("\U0001f504 查询进度", {"action": "query_bucket_transfer", "job_id": job["job_id"]})),
    ], color="blue")


def result_card(job):
    from core.bucket_transfer.orchestrator import fmt_size, fmt_ts
    if job["stage"] == "DONE":
        return card(f"✅ 桶间迁移完成（{job['direction']}）", [
            fields(("任务ID", f"`{job['job_id']}`"),
                   ("对象数", str(job.get("objects", 0))),
                   ("数据量", fmt_size(job.get("bytes", 0))),
                   ("结束", fmt_ts(job.get("finished_ts", 0)))),
            div(f"**源** `{job['source']}`\n**目的** `{job['dest']}`"),
        ], color="green")
    return card(f"❌ 桶间迁移失败（{job['direction']}）", [
        fields(("任务ID", f"`{job['job_id']}`"), ("阶段", job["stage"]),
               ("结束", fmt_ts(job.get("finished_ts", 0)))),
        div(f"**源** `{job['source']}`\n**目的** `{job['dest']}`\n"
            f"**失败原因**：{job.get('error') or '未知'}"),
        hr(),
        buttons(btn("\U0001f501 重试", {"action": "retry_bucket_transfer", "job_id": job["job_id"]}, "danger")),
    ], color="red")
