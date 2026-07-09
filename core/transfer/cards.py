"""Feishu cards for cross-cloud transfer."""
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


def _same_name_options():
    return [
        {"text": _pt("\u8df3\u8fc7\u540c\u540d\u6587\u4ef6"), "value": "skip"},
        {"text": _pt("\u8986\u76d6\u540c\u540d\u6587\u4ef6"), "value": "overwrite"},
    ]


def _same_name_policy(job: dict | None = None) -> str:
    if not job:
        return "skip"
    from core.transfer.orchestrator import normalize_same_name_policy
    return normalize_same_name_policy(
        job.get("same_name_policy") or job.get("overwrite_policy") or job.get("overwrite_mode")
    )


def _same_name_label(job: dict | None = None) -> str:
    from core.transfer.orchestrator import same_name_policy_label
    return same_name_policy_label(_same_name_policy(job))


def _fmt_size(num_bytes: int) -> str:
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def entry_card():
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f69a \u65b0\u5efa\u8de8\u4e91\u8fc1\u79fb"), "template": "blue"},
        "body": {"elements": [
            {"tag": "markdown",
             "content": "\u586b\u5199\u6e90 / \u76ee\u7684\u5730\u5740\uff08\u76ee\u5f55\u4ee5 `/` \u7ed3\u5c3e\uff09\u3002\u76ee\u7684\u53ef\u7559\u7a7a\uff0c\u6309\u6620\u5c04\u81ea\u52a8\u63a8\u5bfc\u3002\n"
                        "\u683c\u5f0f\uff1a`tos://bucket/prefix/`\u3001`oss://bucket/prefix/`\n"
                        "> \u540c\u540d\u7b56\u7565\u53ef\u9009\u62e9\uff0c\u9ed8\u8ba4\u8df3\u8fc7\u540c\u540d\u6587\u4ef6\u3002"},
            {"tag": "form", "name": "transfer_entry", "elements": [
                {"tag": "input", "name": "source", "label": _pt("\u6e90\u5730\u5740"), "required": True,
                 "placeholder": _pt("\u5982 tos://lightwheel/third-party-data/x/")},
                {"tag": "input", "name": "dest", "label": _pt("\u76ee\u7684\u5730\u5740\uff08\u53ef\u7559\u7a7a\uff09"),
                 "placeholder": _pt("\u5982 oss://wuji-bucket-hangzhou/x/\uff0c\u7559\u7a7a\u81ea\u52a8\u63a8\u5bfc")},
                {"tag": "select_static", "name": "same_name_policy", "initial_option": "skip",
                 "placeholder": _pt("\u9009\u62e9\u540c\u540d\u7b56\u7565"), "options": _same_name_options()},
                {"tag": "button", "text": _pt("\u27a1\ufe0f \u89e3\u6790\u5e76\u9884\u4f30"), "type": "primary",
                 "form_action_type": "submit", "name": "submit",
                 "behaviors": [{"type": "callback", "value": {"action": "submit_transfer"}}]},
            ]},
        ]},
    }


def confirm_card(job: dict, *, need_approval: bool, is_admin_target: bool = False):
    size = _fmt_size(job.get("bytes_total", 0))
    objs = job.get("objects_total", 0)
    same_name_policy = _same_name_policy(job)
    same_name_label = _same_name_label(job)
    head = "\U0001f69a \u8de8\u4e91\u8fc1\u79fb\u786e\u8ba4" + ("\uff08\u9700\u7ba1\u7406\u5458\u5ba1\u6279\uff09" if need_approval else "")
    title_tpl = "orange" if need_approval else "blue"
    info_md = (
        f"**\u6e90**\uff1a`{job['source']}`\n"
        f"**\u76ee\u7684**\uff1a`{job['dest']}`\n"
        f"**\u65b9\u5411**\uff1a{job['direction']}\u3000**\u9884\u4f30**\uff1a{size} / {objs} \u5bf9\u8c61\n"
        f"**\u540c\u540d\u7b56\u7565**\uff1a{same_name_label}"
    )
    if need_approval:
        info_md += "\n\n> \u26a0\ufe0f \u8d85\u8fc7\u5ba1\u6279\u9608\u503c\uff0c\u9700\u7ba1\u7406\u5458\u5728\u4e0b\u65b9\u786e\u8ba4\u4e0b\u53d1\u3002"
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt(head), "template": title_tpl},
        "body": {"elements": [
            {"tag": "markdown", "content": info_md},
            {"tag": "hr"},
            {"tag": "form", "name": "transfer_form", "elements": [
                {"tag": "select_static", "name": "same_name_policy", "initial_option": same_name_policy,
                 "placeholder": _pt("\u9009\u62e9\u540c\u540d\u7b56\u7565"), "options": _same_name_options()},
                {"tag": "button", "text": _pt("\u2705 \u786e\u8ba4\u8fc1\u79fb"), "type": "primary",
                 "form_action_type": "submit", "name": "submit",
                 "behaviors": [{"type": "callback", "value": {"action": "confirm_transfer", "job_id": job["job_id"]}}]},
            ]},
        ]},
    }


def accepted_card(job: dict):
    return card("\U0001f4e8 \u8fc1\u79fb\u4efb\u52a1\u5df2\u53d7\u7406", [
        fields(("\u4efb\u52a1", f"`{job['job_id']}`"), ("\u65b9\u5411", job["direction"]),
               ("\u540c\u540d\u7b56\u7565", _same_name_label(job))),
        div(f"**\u6e90** `{job['source']}`\n**\u76ee\u7684** `{job['dest']}`"),
        note("\u5df2\u5728\u540e\u53f0\u8fd0\u884c\uff0c\u5b8c\u6210\u540e\u63a8\u9001\u7ed3\u679c\u3002"),
    ], color="blue")


def progress_card(job: dict):
    stage_cn = {"SINKING": "\u6c89\u964d\u4e2d", "CROSSING": "\u8de8\u4e91\u8fc1\u79fb\u4e2d"}.get(job["stage"], job["stage"])
    return card(f"\u23f3 {stage_cn}", [
        fields(("\u4efb\u52a1", f"`{job['job_id']}`"), ("\u9636\u6bb5", stage_cn),
               ("\u540c\u540d\u7b56\u7565", _same_name_label(job))),
        div(f"**\u6e90** `{job['source']}`\n**\u76ee\u7684** `{job['dest']}`"),
    ], color="blue")


def progress_card_v2(job: dict):
    """schema 2.0 纯展示"进行中"卡。

    确认卡是 schema 2.0，回调里原地替换必须同 schema 家族（2.0→2.0），
    用 1.0 的 progress_card 原地替换 2.0 卡会被飞书拒（错误 200830，静默失败，
    用户在确认到完成之间看不到"进行中"反馈）。故确认后的原地替换专用这张 2.0 卡。
    无 form/按钮——按钮随确认消失，中间态不再另推，终态由后台线程推结果卡。
    """
    stage_cn = {"SINKING": "沉降中", "CROSSING": "跨云迁移中"}.get(job["stage"], job["stage"])
    info_md = (
        f"**任务**：`{job['job_id']}`\n"
        f"**阶段**：{stage_cn}\n"
        f"**同名策略**：{_same_name_label(job)}\n"
        f"**源** `{job['source']}`\n"
        f"**目的** `{job['dest']}`\n"
        f"> 已在后台运行，完成后自动推送结果。\n"
        f"> 想随时查看进度：发送 `查询进度 {job['job_id']}`"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt(f"⏳ {stage_cn}"), "template": "blue"},
        "body": {"elements": [{"tag": "markdown", "content": info_md}]},
    }


def result_card(job: dict):
    from core.transfer.orchestrator import fmt_ts, fmt_duration
    ok = job["stage"] == "DONE"
    size = _fmt_size(job.get("bytes_total", 0))
    objs = job.get("objects_total", 0)
    created = job.get("created_ts", 0)
    finished = job.get("finished_ts", 0)
    started_by = job.get("created_by", "")
    if ok:
        elements = [
            fields(("\u4efb\u52a1", f"`{job['job_id']}`"), ("\u65b9\u5411", job["direction"]),
                   ("\u8fc1\u79fb\u91cf", size), ("\u5bf9\u8c61\u6570", str(objs)),
                   ("\u53d1\u8d77\u65f6\u95f4", fmt_ts(created)), ("\u5b8c\u6210\u65f6\u95f4", fmt_ts(finished)),
                   ("\u8017\u65f6", fmt_duration(created, finished)), ("\u540c\u540d\u7b56\u7565", _same_name_label(job))),
            div(f"**\u6e90** `{job['source']}`\n**\u76ee\u7684** `{job['dest']}`"),
        ]
        if started_by:
            elements.append(note(f"\u53d1\u8d77\u4eba {started_by}"))
        return card("\u2705 \u8de8\u4e91\u8fc1\u79fb\u5b8c\u6210", elements, color="green")
    unknown = "\u672a\u77e5"
    elements = [
        fields(("\u4efb\u52a1", f"`{job['job_id']}`"), ("\u65b9\u5411", job["direction"]),
               ("\u540c\u540d\u7b56\u7565", _same_name_label(job)),
               ("\u53d1\u8d77\u65f6\u95f4", fmt_ts(created)), ("\u7ed3\u675f\u65f6\u95f4", fmt_ts(finished)),
               ("\u8017\u65f6", fmt_duration(created, finished))),
        div(f"**\u6e90** `{job['source']}`\n**\u76ee\u7684** `{job['dest']}`\n"
            f"**\u5931\u8d25\u539f\u56e0**\uff1a{job.get('error') or unknown}"),
        hr(),
        buttons(btn("\U0001f501 \u91cd\u8bd5", {"action": "retry_transfer", "job_id": job["job_id"]}, "danger")),
    ]
    if started_by:
        elements.append(note(f"\u53d1\u8d77\u4eba {started_by}"))
    return card("\u274c \u8de8\u4e91\u8fc1\u79fb\u5931\u8d25", elements, color="red")
