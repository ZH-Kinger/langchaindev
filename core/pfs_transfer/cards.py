"""Feishu cards：PFS↔PFS 跨云直传 录入 / 确认 / 进度 / 结果。

录入/确认卡 schema 2.0；进度用 2.0 纯展示卡（progress_card_v2，避 200830）；结果卡 1.0（含重试）。
"""
from tools.feishu.cards import btn, buttons, card, div, fields, hr


def _pt(content):
    return {"tag": "plain_text", "content": content}


def _pfs_str(ep: dict) -> str:
    return f"{ep.get('scheme')}://{ep.get('fs_id')}/{ep.get('sub_path', '')}"


def entry_card():
    """录入卡：源 PFS 地址 + 目的 PFS 地址（+ 可选预估量/同名策略）。方向由地址自动判断。"""
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f501 PFS 跨云直传（vePFS↔CPFS）"), "template": "orange"},
        "body": {"elements": [
            {"tag": "markdown",
             "content": "把一个 PFS 目录直传到另一朵云的 PFS（自动经 3 段：源沉降→跨云→目的预热）。\n"
                        "方向按地址自动判断：源 `vepfs://` → CPFS；源 `cpfs://` → vePFS。"},
            {"tag": "form", "name": "pfs_transfer", "elements": [
                {"tag": "input", "name": "source", "required": True,
                 "label": _pt("源 PFS 地址"),
                 "placeholder": _pt("vepfs://<fs-id>/子目录/  或  cpfs://<fs-id>/子目录/")},
                {"tag": "input", "name": "dest", "required": True,
                 "label": _pt("目的 PFS 地址"),
                 "placeholder": _pt("cpfs://<fs-id>/子目录/  或  vepfs://<fs-id>/子目录/")},
                {"tag": "input", "name": "size_tb", "required": False,
                 "label": _pt("预估数据量（TB，可留空）"),
                 "placeholder": _pt("留空=大小未知→按需审批")},
                {"tag": "button", "text": _pt("下一步：确认"), "type": "primary",
                 "form_action_type": "submit", "name": "submit",
                 "behaviors": [{"type": "callback", "value": {"action": "submit_pfs_transfer"}}]},
            ]},
        ]},
    }


def confirm_card(job: dict, *, need_approval: bool = False):
    from . import orchestrator as o
    ss, ds = job["src_staging"], job["dst_staging"]
    est = (o.fmt_size(job["approx_bytes"]) if job.get("size_known") and job.get("approx_bytes")
           else "未填（大小未知）")
    info_md = (
        f"**任务ID**：`{job['job_id']}`\n"
        f"**方向**：{job.get('direction')}\n"
        f"**源 PFS**：`{_pfs_str(job['src_pfs'])}`\n"
        f"**目的 PFS**：`{_pfs_str(job['dst_pfs'])}`\n"
        f"**① 沉降落点**：`{ss['scheme']}://{ss['bucket']}/{ss['prefix']}`\n"
        f"**② 跨云落点**：`{ds['scheme']}://{ds['bucket']}/{ds['prefix']}`\n"
        f"**③ 预热** → 目的 PFS\n"
        f"**预估量（自报）**：{est}"
    )
    elements = [{"tag": "markdown", "content": info_md}, {"tag": "hr"}]
    if need_approval:
        elements.append({"tag": "markdown",
                         "content": "⚠️ PFS 跨云直传**一律仅管理员**可确认下发（自报量仅供参考）。"})
    elements.append({"tag": "button", "text": _pt("✅ 确认下发"), "type": "primary",
                     "behaviors": [{"type": "callback",
                                    "value": {"action": "confirm_pfs_transfer", "job_id": job["job_id"]}}]})
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt("\U0001f501 PFS 直传确认"),
                   "template": "red" if need_approval else "orange"},
        "body": {"elements": elements},
    }


def progress_card_v2(job: dict):
    """schema 2.0 纯展示进度卡（含段进度）。确认卡 2.0，原地替换须同家族避 200830；无按钮。"""
    from . import orchestrator as o
    info_md = (
        f"**任务**：`{job['job_id']}`\n"
        f"**阶段**：{o._STAGE_LABEL.get(job['stage'], job['stage'])}\n"
        f"**进度**：{o.progress_line(job)}\n"
        f"**源** `{_pfs_str(job['src_pfs'])}`\n"
        f"**目的** `{_pfs_str(job['dst_pfs'])}`\n"
        f"> 已在后台运行 3 段链，完成后自动推送结果。\n"
        f"> 想随时看进度：发送 `查询进度 {job['job_id']}`"
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": _pt(f"⏳ {o._STAGE_LABEL.get(job['stage'], job['stage'])}"), "template": "blue"},
        "body": {"elements": [{"tag": "markdown", "content": info_md}]},
    }


def result_card(job: dict):
    from . import orchestrator as o
    ok = job["stage"] == "DONE"
    created, finished = job.get("created_ts", 0), job.get("finished_ts", 0)
    src, dst = _pfs_str(job["src_pfs"]), _pfs_str(job["dst_pfs"])
    if ok:
        return card("✅ PFS 跨云直传完成", [
            fields(("任务ID", f"`{job['job_id']}`"),
                   ("方向", job.get("direction", "")),
                   ("耗时", o.fmt_duration(created, finished))),
            div(f"**源** `{src}`\n**目的** `{dst}`\n三段（沉降→跨云→预热）全部完成。"),
        ], color="green")
    return card("❌ PFS 跨云直传失败", [
        fields(("任务ID", f"`{job['job_id']}`"), ("阶段", job.get("stage", "")),
               ("结束", o.fmt_ts(finished))),
        div(f"**源** `{src}`\n**目的** `{dst}`\n**失败原因**：{job.get('error') or '未知'}"),
        hr(),
        buttons(btn("\U0001f501 重试", {"action": "retry_pfs_transfer", "job_id": job["job_id"]}, "danger")),
    ], color="red")
