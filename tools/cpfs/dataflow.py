"""manage_cpfs_dataflow tool：CPFS 数据预热(Import)/沉降(Export)。"""
import logging
import threading

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

logger = logging.getLogger(__name__)


class CpfsDataflowSchema(BaseModel):
    action: str = Field(description="discover / list / preheat / sink / status")
    cpfs_path: str = Field(default="", description="CPFS 目录，cpfs://<fs-id>/<dir>/ 或裸目录")
    oss: str = Field(default="", description="可选 oss://bucket/prefix/，定位 DataFlow 与源/目标目录")
    fs_id: str = Field(default="", description="list 用：文件系统 ID（缺省取 CPFS_FILE_SYSTEM_ID）")
    job_id: str = Field(default="", description="status 用：任务 ID")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


def manage_cpfs_dataflow(action: str, cpfs_path: str = "", oss: str = "",
                         fs_id: str = "", job_id: str = "", open_id: str = "") -> str:
    action = (action or "").strip().lower()
    from config.settings import settings
    from core.cpfs_dataflow import engine_nas, orchestrator, discovery
    from core.cpfs_dataflow.orchestrator import DataflowPathError

    try:
        if action in ("discover", "map", "refresh"):
            opts = discovery.refresh(open_id=open_id) if action == "refresh" \
                else discovery.get_options(open_id=open_id)
            if not opts:
                return "未发现 CPFS↔OSS 绑定（检查 CPFS_FILE_SYSTEM_IDS 与 nas 读权限）。"
            lines = [f"- {o['label']}  `df:{o['data_flow_id']}`" for o in opts]
            return f"## CPFS↔OSS 绑定映射（{len(opts)}，cri 镜像仓库已屏蔽）\n" + "\n".join(lines)

        if action == "list":
            fs = fs_id or settings.CPFS_FILE_SYSTEM_ID
            if not fs:
                return "❌ list 需要 fs_id（或配置 CPFS_FILE_SYSTEM_ID）。"
            flows = engine_nas.list_dataflows(fs, open_id=open_id)
            if not flows:
                return f"文件系统 `{fs}` 下没有 DataFlow。"
            lines = [f"- `{f['data_flow_id']}` {f.get('status','')}  "
                     f"{f.get('source_storage','')}  fs:{f.get('fs_path','')}" for f in flows]
            return f"## `{fs}` 的 DataFlow（{len(flows)}）\n" + "\n".join(lines)

        if action == "status":
            if not job_id:
                return "❌ status 需要 job_id。"
            job = orchestrator.get_job(job_id)
            if not job:
                return f"未找到任务 `{job_id}`（可能已过期）。"
            return (f"任务 `{job_id}`：阶段 **{job['stage']}**  {job['operation_label']}\n"
                    f"- fs:`{job['fs_id']}` df:`{job.get('data_flow_id','-')}` task:`{job.get('task_id','-')}`\n"
                    f"- 进度：{job.get('files_done',0)}/{job.get('files_total',0)} 文件，"
                    f"{orchestrator.fmt_size(job.get('bytes_done',0))}/{orchestrator.fmt_size(job.get('bytes_total',0))}"
                    + (f"\n- 错误：{job['error']}" if job.get("error") else ""))

        if action in ("preheat", "sink", "预热", "沉降", "import", "export"):
            if not cpfs_path:
                return "❌ 需要 cpfs_path，如 `cpfs://bmcpfs-x/dataset/`。"
            plan = orchestrator.make_plan(action, cpfs_path, oss)
            job = orchestrator.create_job_record(plan, open_id=open_id)
            threading.Thread(target=orchestrator.run_to_completion, args=(job,), daemon=True).start()
            return (f"✅ 已提交{orchestrator.operation_label(plan.operation)}任务 `{job['job_id']}`"
                    f"（fs:{plan.fs_id} dir:{plan.directory}），后台运行中，用 status 查进度。")

        return f"❌ 未知 action：{action}，可选 discover / list / preheat / sink / status。"

    except DataflowPathError as e:
        return f"❌ 路径错误：{e}"
    except engine_nas.NasDataflowError as e:
        return f"❌ DataFlow 调用失败：{e}"
    except Exception as e:
        logger.error("[CPFS] manage_cpfs_dataflow failed action=%s", action, exc_info=True)
        return f"❌ 操作失败：{e}"


cpfs_dataflow_tool = StructuredTool.from_function(
    func=manage_cpfs_dataflow,
    name="manage_cpfs_dataflow",
    description=("CPFS/NAS 数据预热与沉降。action=discover 枚举所有 CPFS↔OSS 绑定映射表；"
                 "list 列单个 fs 的 DataFlow；preheat 预热(OSS→CPFS 加载)；"
                 "sink 沉降(CPFS→OSS 刷回)；status 查任务。"
                 "cpfs_path 形如 cpfs://<fs-id>/<dir>/，oss 可选 oss://bucket/prefix/。"),
    args_schema=CpfsDataflowSchema,
)
