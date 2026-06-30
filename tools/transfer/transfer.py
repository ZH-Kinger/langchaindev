"""manage_transfer tool: cross-cloud data transfer."""
import logging

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

logger = logging.getLogger(__name__)


class TransferSchema(BaseModel):
    action: str = Field(description="plan / apply / status")
    source: str = Field(default="", description="source path, such as tos://bucket/prefix/")
    dest: str = Field(default="", description="optional destination path")
    job_id: str = Field(default="", description="job id for status")
    same_name_policy: str = Field(default="skip", description="skip or overwrite")
    open_id: str = Field(default="", description="injected Feishu open_id")


def _plan_summary(plan, bytes_total, objects_total, same_name_policy="") -> str:
    from core.transfer.orchestrator import fmt_size, needs_approval, same_name_policy_label
    appr = "\uff08\u8d85\u9608\u503c\uff0c\u9700\u7ba1\u7406\u5458\u5ba1\u6279\uff09" if needs_approval(bytes_total) else ""
    return (
        f"## \u8fc1\u79fb\u8ba1\u5212{appr}\n"
        f"- **\u6e90**\uff1a`{plan.source.uri()}`\n"
        f"- **\u76ee\u7684**\uff1a`{plan.dest.uri()}`\n"
        f"- **\u65b9\u5411**\uff1a{plan.direction}  **\u5f15\u64ce**\uff1a{plan.engine}\n"
        f"- **\u540c\u540d\u7b56\u7565**\uff1a{same_name_policy_label(same_name_policy)}\n"
        f"- **\u9884\u4f30**\uff1a{fmt_size(bytes_total)} / {objects_total} \u5bf9\u8c61\n"
        + ("- **\u94fe\u8def**\uff1a\u542b\u6c89\u964d\u6bb5\uff08\u5168\u95ea->\u5bf9\u8c61\u5b58\u50a8->\u8de8\u4e91\uff09\n" if plan.needs_sink else "")
    )


def _push_admin_confirm(job: dict) -> bool:
    try:
        from config.settings import settings
        from core.dsw_scheduler import _send_card
        from core.transfer.cards import confirm_card
        admin = settings.ADMIN_FEISHU_OPEN_ID
        chat = settings.TRANSFER_CHAT_ID or settings.FEISHU_CHAT_ID
        if not admin and not chat:
            return False
        _send_card(admin, chat, confirm_card(job, need_approval=True))
        return True
    except Exception:
        logger.warning("[Transfer] failed to push admin confirm card", exc_info=True)
        return False


def manage_transfer(action: str, source: str = "", dest: str = "",
                    job_id: str = "", same_name_policy: str = "skip",
                    open_id: str = "") -> str:
    action = (action or "").strip().lower()
    from core.transfer import orchestrator
    from core.transfer.paths import PathError

    try:
        if action == "status":
            if not job_id:
                return "\u274c status \u9700\u8981 job_id\u3002"
            job = orchestrator.get_job(job_id)
            if not job:
                return f"\u672a\u627e\u5230\u4efb\u52a1 `{job_id}`\uff08\u53ef\u80fd\u5df2\u8fc7\u671f\uff09\u3002"
            return (f"\u4efb\u52a1 `{job_id}`\uff1a\u9636\u6bb5 **{job['stage']}**  "
                    f"{job['source']} -> {job['dest']}  "
                    f"\u540c\u540d\u7b56\u7565\uff1a{orchestrator.same_name_policy_label(job.get('same_name_policy') or job.get('overwrite_mode'))}  "
                    f"{orchestrator.fmt_size(job.get('bytes_total', 0))}"
                    + (f"\n\u9519\u8bef\uff1a{job['error']}" if job.get("error") else ""))

        if action in ("plan", "apply"):
            if not source:
                return "\u274c \u9700\u8981 source \u8def\u5f84\uff0c\u5982 `tos://bucket/prefix/`\u3002"
            plan = orchestrator.make_plan(source, dest)
            if plan.engine not in ("mgw", "tos_mig"):
                return f"\u26a0\ufe0f \u65b9\u5411 {plan.direction} \u7684\u5f15\u64ce `{plan.engine}` \u5c1a\u672a\u5b9e\u73b0\u3002"
            bytes_total, objects_total = orchestrator.estimate_source(plan)

            if action == "plan":
                return _plan_summary(plan, bytes_total, objects_total, same_name_policy)

            job = orchestrator.create_job_record(
                plan, open_id=open_id, same_name_policy=same_name_policy,
                bytes_total=bytes_total, objects_total=objects_total)
            if orchestrator.needs_approval(bytes_total):
                pushed = _push_admin_confirm(job)
                tip = "\u5df2\u63a8\u9001\u786e\u8ba4\u5361\u7ed9\u7ba1\u7406\u5458" if pushed else "\u8bf7\u7ba1\u7406\u5458\u5728\u98de\u4e66\u786e\u8ba4\u4e0b\u53d1"
                return (f"\u26a0\ufe0f \u9884\u4f30 {orchestrator.fmt_size(bytes_total)} \u8d85\u8fc7\u5ba1\u6279\u9608\u503c\uff0c"
                        f"\u5df2\u751f\u6210\u4efb\u52a1 `{job['job_id']}`\uff08{tip}\uff09\uff0c\u672a\u81ea\u52a8\u542f\u52a8\u3002")
            import threading
            threading.Thread(target=orchestrator.run_to_completion, args=(job,), daemon=True).start()
            return (f"\u2705 \u5df2\u63d0\u4ea4\u8fc1\u79fb\u4efb\u52a1 `{job['job_id']}`\uff08{plan.direction}\uff0c"
                    f"\u540c\u540d\u7b56\u7565\uff1a{orchestrator.same_name_policy_label(job.get('same_name_policy'))}\uff09\uff0c"
                    f"\u540e\u53f0\u8fd0\u884c\u4e2d\uff0c\u7528 status \u67e5\u8be2\u8fdb\u5ea6\u3002")

        return f"\u274c \u672a\u77e5 action\uff1a{action}\uff0c\u53ef\u9009 plan / apply / status\u3002"

    except PathError as e:
        return f"\u274c \u8def\u5f84\u9519\u8bef\uff1a{e}"
    except Exception as e:
        logger.error("[Transfer] manage_transfer failed action=%s", action, exc_info=True)
        return f"\u274c \u8fc1\u79fb\u64cd\u4f5c\u5931\u8d25\uff1a{e}"


transfer_tool = StructuredTool.from_function(
    func=manage_transfer,
    name="manage_transfer",
    description="Cross-cloud transfer. Supports plan/apply/status and same_name_policy=skip|overwrite.",
    args_schema=TransferSchema,
)
