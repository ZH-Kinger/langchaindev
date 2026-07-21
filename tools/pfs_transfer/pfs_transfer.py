"""manage_pfs_transfer tool：vePFS↔CPFS 跨云直传（3 段链：沉降→跨云→预热）。"""
import logging
import threading

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

logger = logging.getLogger(__name__)


class PfsTransferSchema(BaseModel):
    action: str = Field(description="plan / apply / status")
    source: str = Field(default="", description="源 PFS 地址，vepfs://<fs>/dir/ 或 cpfs://<fs>/dir/")
    dest: str = Field(default="", description="目的 PFS 地址（另一朵云的 PFS）")
    same_name: str = Field(default="", description="同名策略 skip/keeplatest/overwrite（默认 skip）")
    force: bool = Field(default=False, description="apply 时越过审批阈值")
    job_id: str = Field(default="", description="status 用：任务 ID（xpfs-…）")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


def manage_pfs_transfer(action: str, source: str = "", dest: str = "", same_name: str = "",
                        force: bool = False, job_id: str = "", open_id: str = "") -> str:
    action = (action or "").strip().lower()
    from core.pfs_transfer import paths, orchestrator as o

    try:
        if action == "status":
            if not job_id:
                return "❌ status 需要 job_id（xpfs-…）。"
            job = o.get_job(job_id)
            if not job:
                return f"未找到任务 `{job_id}`（可能已过期）。"
            return (f"任务 `{job_id}`：阶段 **{job['stage']}** 方向 {job.get('direction')}\n"
                    f"- 段完成：沉降={job.get('sink_done')} 跨云={job.get('cross_done')} 预热={job.get('preheat_done')}\n"
                    f"- 进度：{o.progress_line(job)}"
                    + (f"\n- 错误：{job['error']}" if job.get("error") else ""))

        if action in ("plan", "apply"):
            if not (source and dest):
                return "❌ 需要 source 与 dest，如 `vepfs://fs-x/data/` `cpfs://fs-y/data/`。"
            plan = paths.build_plan(source, dest)
            _, size_known = o.estimate_source(plan)
            need = o.needs_approval(0, size_known)
            if action == "plan":
                return (f"计划（dry-run）：{plan.summary()}\n"
                        f"- ① 沉降落点 {plan.src_staging.scheme}://{plan.src_staging.bucket}/{plan.src_staging.base_prefix}\n"
                        f"- ② 跨云落点 {plan.dst_staging.scheme}://{plan.dst_staging.bucket}/{plan.dst_staging.base_prefix}\n"
                        f"- 审批：一律需管理员确认下发")
            # apply 一律需管理员（MED-2 收紧，用户拍板）：本工具由 Agent 执行器调用、open_id=聊天者本人，
            # force 逃生口不能绕过 admin 门（否则非管理员「apply force=true」即绕过卡片路的 admin 门）。
            # CLI 走 orchestrator 直连、不经本工具，其 --force 属服务器特权运维、可接受。
            from config.settings import settings
            if open_id != settings.ADMIN_FEISHU_OPEN_ID:
                return "❌ PFS 跨云直传需管理员确认下发，请联系管理员（或走飞书确认卡由管理员点确认）。"
            if need and not force:
                return f"⚠️ 需管理员确认。管理员带 force=true 执行。计划：{plan.summary()}"
            job = o.create_job_record(plan, open_id=open_id, same_name=same_name)
            threading.Thread(target=o.run_to_completion, args=(job,), daemon=True).start()
            return (f"✅ 已提交 PFS 直传 `{job['job_id']}`（{plan.summary()}），"
                    f"后台跑 3 段链，用 status 查进度。")

        return f"❌ 未知 action：{action}，可选 plan / apply / status。"

    except paths.PfsPathError as e:
        return f"❌ 路径/配置错误：{e}"
    except Exception as e:
        logger.error("[PFS] manage_pfs_transfer failed action=%s", action, exc_info=True)
        return f"❌ 操作失败：{e}"


pfs_transfer_tool = StructuredTool.from_function(
    func=manage_pfs_transfer,
    name="manage_pfs_transfer",
    description=("vePFS↔CPFS 跨云直传（自动经 3 段：源 PFS 沉降→跨云迁移→目的 PFS 预热）。"
                 "plan 预览计划(dry-run)；apply 提交；status 查任务。"
                 "source/dest 形如 vepfs://<fs>/dir/、cpfs://<fs>/dir/，方向自动判断。"),
    args_schema=PfsTransferSchema,
)
