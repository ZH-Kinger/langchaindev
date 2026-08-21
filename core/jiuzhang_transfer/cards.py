"""九章迁移的飞书卡片。录入/确认用 schema 2.0；结果卡 1.0（同泰国链的约定，避 200830）。"""
from config.settings import settings

from . import orchestrator as o


def _pt(t):
    return {"tag": "plain_text", "content": t}


def _md(t):
    return {"tag": "div", "text": {"tag": "lark_md", "content": t}}


def entry_card() -> dict:
    """录入卡：填源路径 + 可选目标子目录。"""
    root = getattr(settings, "JIUZHANG_DEST_ROOT", "") or "/root/nas"
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"template": "blue", "title": _pt("🖥️ 数据迁移（九章 B200）")},
        "body": {"elements": [
            _md(f"杭州 OSS → 九章本地盘 `{root}`（单跳直连，不经中转）"),
            {"tag": "form", "name": "jz_form", "elements": [
                {"tag": "input", "name": "source", "label": _pt("源路径"),
                 "placeholder": _pt("oss://wuji-bucket-hangzhou/目录/子目录/")},
                {"tag": "input", "name": "dest", "label": _pt("目标子目录（可选，留空则镜像源前缀）"),
                 "placeholder": _pt("如 datasets/foo/")},
                {"tag": "button", "text": _pt("下一步：预估"), "type": "primary",
                 "action_type": "form_submit",
                 "value": {"action": "submit_jiuzhang_transfer"}},
            ]},
        ]},
    }


def confirm_card(job: dict, need_approval: bool) -> dict:
    lines = [
        f"**任务ID**：`{job['job_id']}`",
        f"**源**：`oss://{job['source_bucket']}/{job['source_prefix']}`",
        f"**目的**：九章 `{job.get('dest_dir', '')}`",
    ]
    if job.get("estimate_ok"):
        lines.append(f"**估算**：{o._fmt(job.get('bytes_total', 0))} / "
                     f"{job.get('objects_total', 0)} 对象")
    else:
        lines.append("**估算**：失败（大小未知）")
    if need_approval:
        lines.append("\n⚠️ 超过审批阈值（或大小未知），**仅管理员**可确认下发。")
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"template": "orange", "title": _pt("🖥️ 九章迁移确认")},
        "body": {"elements": [
            _md("\n".join(lines)),
            {"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
             "behaviors": [{"type": "callback",
                            "value": {"action": "confirm_jiuzhang_transfer",
                                      "job_id": job["job_id"]}}]},
        ]},
    }


def progress_card_v2(job: dict) -> dict:
    """纯展示进度卡（2.0，无按钮）——与确认卡同家族，原地替换避 200830。"""
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"template": "blue", "title": _pt("🖥️ 九章迁移进行中")},
        "body": {"elements": [_md(
            f"**任务ID**：`{job['job_id']}`\n"
            f"**阶段**：{job.get('stage')}\n"
            f"{o.progress_line(job)}")]},
    }


def result_card(job: dict) -> dict:
    ok = job.get("stage") == o.STAGE_DONE
    lines = [
        f"**任务ID**：`{job['job_id']}`",
        f"**源**：`oss://{job['source_bucket']}/{job['source_prefix']}`",
        f"**目的**：`{job.get('dest_dir', '')}`",
    ]
    if job.get("bytes_done"):
        lines.append(f"**已传**：{o._fmt(job['bytes_done'])}")
    # 校验结论**必须显示**：不显示的话，关掉校验开关后卡片与以前一模一样，
    # 那个开关就成了隐形的 fail-open 后门。
    if job.get("verify_summary"):
        lines.append(f"\n**端到端校验**\n```\n{job['verify_summary']}\n```")
    if job.get("error"):
        lines.append(f"\n❌ {job['error']}")
    if job.get("error_detail"):
        lines.append(f"```\n{str(job['error_detail'])[:800]}\n```")
    elements = [{"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}}]
    if not ok:
        elements.append({"tag": "action", "actions": [
            {"tag": "button", "text": _pt("🔄 重试"), "type": "danger",
             "value": {"action": "retry_jiuzhang_transfer", "job_id": job["job_id"]}}]})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"template": "green" if ok else "red",
                   "title": _pt("✅ 九章迁移完成" if ok else "❌ 九章迁移失败")},
        "elements": elements,
    }
