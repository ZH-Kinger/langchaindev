"""Feishu cards：SSH 迁移链（杭州OSS→新加坡→泰国）录入 / 确认 / 进度 / 结果。

确认卡/录入卡是 schema 2.0；进度用 2.0 纯展示卡（progress_card_v2，避 200830）；结果卡用 1.0
（含重试按钮）。进度卡显示已传字节 + 速率 + 预计剩余（progress_line）。
"""
from config.settings import settings
from tools.feishu.cards import btn, buttons, card, div, fields, hr, note


def _pt(content):
    return {"tag": "plain_text", "content": content}


def entry_card():
    """录入卡：填源 OSS 路径 + 泰国目标子目录（可空=同源目录名）。"""
    dest_root = (settings.THAI_DEST_ROOT or "").rstrip("/")
    thai_host = settings.THAI_HOST or ""
    thai_user = settings.THAI_USER or "wuji"
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f69a 数据迁移（泰国H200）"), "template": "orange"},
        "body": {"elements": [
            {"tag": "markdown",
             "content": "把杭州 OSS 数据迁到泰国 H200 服务器（经新加坡中转，两段：ossutil→rsync）。\n"
                        f"**泰国目标根**：`{thai_user}@{thai_host}:{dest_root}/`\n"
                        "你填的「目标子目录」会接在这根后面（留空则用源目录名）。"},
            {"tag": "form", "name": "ssh_transfer", "elements": [
                {"tag": "input", "name": "source", "required": True,
                 "label": _pt("源 OSS 路径"),
                 "placeholder": _pt("oss://wuji-data-tran/子目录/")},
                {"tag": "input", "name": "dest", "required": False,
                 "label": _pt("泰国目标子目录（留空=用源目录名）"),
                 "placeholder": _pt(f"如 test → 落到 {dest_root}/test/")},
                {"tag": "button", "text": _pt("下一步：确认"), "type": "primary",
                 "form_action_type": "submit", "name": "submit",
                 "behaviors": [{"type": "callback", "value": {"action": "submit_ssh_transfer"}}]},
            ]},
        ]},
    }


def confirm_card(job: dict, *, need_approval: bool = False):
    from core.ssh_transfer.orchestrator import fmt_size
    mount = (settings.SGP_OSS_MOUNT or "/mnt/sgp_oss").rstrip("/")
    size = fmt_size(job.get("bytes_total", 0))
    if not job.get("estimate_ok", True):
        size += "（估算失败）"
    info_md = (
        f"**任务ID**：`{job['job_id']}`\n"
        f"**源**：`{job['source_uri']}`\n"
        f"**段1**：→ 新加坡挂载盘 `{mount}/{job['source_prefix']}`\n"
        f"**段2**：→ 泰国 `{job['dest_uri']}`\n"
        f"**估算**：{size} / {job.get('objects_total', 0)} 对象"
    )
    elements = [{"tag": "markdown", "content": info_md}, {"tag": "hr"}]
    if need_approval:
        elements.append({"tag": "markdown",
                         "content": "⚠️ 超过审批阈值（或大小未知），**仅管理员**可确认下发。"})
    elements.append({"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
                     "behaviors": [{"type": "callback",
                                    "value": {"action": "confirm_ssh_transfer", "job_id": job["job_id"]}}]})
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f69a 泰国迁移确认"),
                   "template": "red" if need_approval else "orange"},
        "body": {"elements": elements},
    }


def progress_card_v2(job: dict):
    """schema 2.0 纯展示进度卡（含进度+速率）。确认卡是 2.0，原地替换须同家族避 200830；无按钮。"""
    from core.ssh_transfer.orchestrator import stage_label, progress_line
    info_md = (
        f"**任务**：`{job['job_id']}`\n"
        f"**阶段**：{stage_label(job['stage'])}\n"
        f"**进度**：{progress_line(job)}\n"
        f"**源** `{job['source_uri']}`\n"
        f"**目的** `{job['dest_uri']}`\n"
        f"> 已在后台运行，完成后自动推送结果。\n"
        f"> 想随时看进度：发送 `查询进度 {job['job_id']}`"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt(f"⏳ {stage_label(job['stage'])}"), "template": "blue"},
        "body": {"elements": [{"tag": "markdown", "content": info_md}]},
    }


def result_card(job: dict):
    from core.ssh_transfer.orchestrator import fmt_size, fmt_ts, fmt_duration
    ok = job["stage"] == "DONE"
    created, finished = job.get("created_ts", 0), job.get("finished_ts", 0)
    if ok:
        els = [
            fields(("任务ID", f"`{job['job_id']}`"),
                   ("数据量", fmt_size(job.get("bytes_total", 0))),
                   ("对象数", str(job.get("objects_total", 0))),
                   ("耗时", fmt_duration(created, finished))),
            div(f"**源** `{job.get('source_uri')}`\n**目的** `{job.get('dest_uri')}`"),
        ]
        # 端到端校验结论必须在**成功卡**上可见。不显示的话，把 SSH_STAGE2_VERIFY 关掉后
        # 这张卡与以前一模一样 —— 那个开关就成了隐形的 fail-open ���门（「成功」到底核过没核过，
        # 看卡片看不出来）。所以：核过就写明核了什么，没核就明确警示。
        v = job.get("verify") or {}
        if v.get("passed") is True:
            els.append(div("**端到端校验** ✓ 已核对（对象覆盖 / 字节总量 / 逐文件字节 / 抽样内容）\n"
                           + "\n".join(f"· {ln}" for ln in (v.get("summary") or "").splitlines()
                                       if ln.strip())))
        elif v.get("passed") is None:
            els.append(div("**⚠ 本次未做端到端校验**（`SSH_STAGE2_VERIFY=false`）——"
                           "「完成」仅代表传输器自己报了成功，未核对数据是否完整。"))
        else:
            els.append(div("**⚠ 无校验记录**——本任务未经端到端核对。"))
        return card(f"✅ 泰国迁移完成", els, color="green")
    elements = [
        fields(("任务ID", f"`{job['job_id']}`"), ("阶段", job.get("stage", "")),
               ("结束", fmt_ts(finished))),
        div(f"**源** `{job.get('source_uri')}`\n**目的** `{job.get('dest_uri')}`\n"
            f"**失败原因**：{job.get('error') or '未知'}"),
    ]
    # 明细来自 engine_ssh.failure_detail（ossutil/rsync 汇总行 + 报告路径 + 首条根因）。
    # 只给「退出码 N」的话排障得手工翻十几 MB 日志，故在卡上直接摊开。
    detail = (job.get("error_detail") or "").strip()
    if detail:
        elements.append(div("**明细**\n" + "\n".join(f"· {ln}" for ln in detail.splitlines() if ln.strip())))
    elements += [
        hr(),
        buttons(btn("\U0001f501 重试", {"action": "retry_ssh_transfer", "job_id": job["job_id"]}, "danger")),
    ]
    return card(f"❌ 泰国迁移失败", elements, color="red")
