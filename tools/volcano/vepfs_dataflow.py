"""manage_vepfs_dataflow tool：火山 vePFS 数据预热(Import)/沉降(Export)。"""
import logging
import threading

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

logger = logging.getLogger(__name__)


class VepfsDataflowSchema(BaseModel):
    action: str = Field(description="preheat / sink / status")
    vepfs_path: str = Field(default="", description="vePFS 目录，vepfs://<fs-id>/<dir>/ 或裸目录")
    tos: str = Field(default="", description="tos://bucket/prefix/")
    same_name: str = Field(default="", description="同名策略 Skip/KeepLatest/OverWrite（默认 Skip）")
    job_id: str = Field(default="", description="status 用：任务 ID")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


def manage_vepfs_dataflow(action: str, vepfs_path: str = "", tos: str = "",
                          same_name: str = "", job_id: str = "", open_id: str = "") -> str:
    action = (action or "").strip().lower()
    from core.vepfs_dataflow import engine_vepfs, orchestrator
    from core.vepfs_dataflow.orchestrator import DataflowPathError

    try:
        if action == "status":
            if not job_id:
                return "❌ status 需要 job_id。"
            job = orchestrator.get_job(job_id)
            if not job:
                return f"未找到任务 `{job_id}`（可能已过期）。"
            return (f"任务 `{job_id}`：阶段 **{job['stage']}**  {job['operation_label']}\n"
                    f"- fs:`{job['fs_id']}` task:`{job.get('task_id','-')}` region:`{job['region']}`\n"
                    f"- vePFS:`{job['sub_path']}`  TOS:`{job.get('tos_bucket')}/{job.get('tos_prefix','')}`\n"
                    f"- 进度：{job.get('files_done',0)}/{job.get('files_total',0)} 文件，"
                    f"{orchestrator.fmt_size(job.get('bytes_done',0))}/{orchestrator.fmt_size(job.get('bytes_total',0))}"
                    + (f"\n- 错误：{job['error']}" if job.get("error") else ""))

        if action in ("preheat", "sink", "预热", "沉降", "import", "export"):
            if not (vepfs_path and tos):
                return "❌ 需要 vepfs_path 与 tos，如 preheat `vepfs://vepfs-x/data/` `tos://bk/data/`。"
            plan = orchestrator.make_plan(action, vepfs_path, tos, same_name=same_name)
            job = orchestrator.create_job_record(plan, open_id=open_id)
            threading.Thread(target=orchestrator.run_to_completion, args=(job,), daemon=True).start()
            return (f"✅ 已提交{orchestrator.operation_label(plan.operation)}任务 `{job['job_id']}`"
                    f"（fs:{plan.fs_id} vePFS:{plan.sub_path} TOS:{plan.tos_bucket}/{plan.tos_prefix}），"
                    f"后台运行中，用 status 查进度。")

        return f"❌ 未知 action：{action}，可选 preheat / sink / status。"

    except DataflowPathError as e:
        return f"❌ 路径错误：{e}"
    except engine_vepfs.VepfsDataflowError as e:
        return f"❌ vePFS 数据流动调用失败：{e}"
    except Exception as e:
        logger.error("[VEPFS] manage_vepfs_dataflow failed action=%s", action, exc_info=True)
        return f"❌ 操作失败：{e}"


vepfs_dataflow_tool = StructuredTool.from_function(
    func=manage_vepfs_dataflow,
    name="manage_vepfs_dataflow",
    description=("火山 vePFS 数据预热与沉降（vePFS↔TOS 数据流动）。"
                 "preheat 预热(TOS→vePFS 加载)；sink 沉降(vePFS→TOS 刷回)；status 查任务。"
                 "vepfs_path 形如 vepfs://<fs-id>/<dir>/，tos 形如 tos://bucket/prefix/。"),
    args_schema=VepfsDataflowSchema,
)
