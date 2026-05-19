"""
阿里云 ECS 管理工具。

凭证通过 STS AssumeRole 获取（utils.aliyun_client_factory.get_ecs_client），
按飞书 open_id 隔离权限。

写操作（create / start / stop / reboot / delete）必须 confirm=true。
"""
import logging

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings
from utils.aliyun_client_factory import get_ecs_client

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class ECSSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'list'    - 列出实例（默认显示前 20 台）\n"
            "  'get'     - 查看实例详情，需 instance_id\n"
            "  'create'  - 创建实例，需 instance_type/image_id/vswitch_id/security_group_id；confirm=true\n"
            "  'start'   - 启动实例（Stopped → Running），需 instance_id；confirm=true\n"
            "  'stop'    - 停止实例（Running → Stopped），需 instance_id；confirm=true\n"
            "  'reboot'  - 重启实例，需 instance_id；confirm=true\n"
            "  'delete'  - 释放实例（不可恢复），需 instance_id；confirm=true"
        )
    )
    instance_id: str = Field(default="", description="ECS 实例 ID，如 i-bp1xxxxxx")
    region: str = Field(default="", description="区域 ID，留空用 cn-hangzhou")

    # 创建参数
    instance_name: str = Field(default="", description="新实例名称（create 用）")
    instance_type: str = Field(default="", description="规格 ID，如 ecs.g7.large（create 必填）")
    image_id: str = Field(default="", description="镜像 ID（create 必填）")
    vswitch_id: str = Field(default="", description="交换机 ID（create 必填，决定 VPC 和可用区）")
    security_group_id: str = Field(default="", description="安全组 ID（create 必填）")
    internet_max_bandwidth_out: int = Field(default=0, description="出公网带宽 Mbps，0 表示不分配公网")
    password: str = Field(default="", description="实例登录密码（create 可选，最少 8 位）")

    confirm: bool = Field(default=False, description="所有写操作必须显式 true")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def manage_ecs(
    action: str,
    instance_id: str = "",
    region: str = "",
    instance_name: str = "",
    instance_type: str = "",
    image_id: str = "",
    vswitch_id: str = "",
    security_group_id: str = "",
    internet_max_bandwidth_out: int = 0,
    password: str = "",
    confirm: bool = False,
    open_id: str = "",
) -> str:
    region = region or settings.PAI_DSW_REGION_ID or "cn-hangzhou"
    client = get_ecs_client(open_id=open_id, region=region)
    if client is None:
        return "❌ ECS 凭证不可用：请确认 open_id 已建立 RAM 映射，或检查 STS 配置。"

    try:
        from alibabacloud_ecs20140526 import models as ecs_models
    except ImportError:
        return "❌ alibabacloud_ecs20140526 未安装，请执行：pip install alibabacloud_ecs20140526"

    action = action.strip().lower()

    try:
        # ── 只读：list ────────────────────────────────────────────────────────
        if action == "list":
            req = ecs_models.DescribeInstancesRequest(region_id=region, page_size=20, page_number=1)
            resp = client.describe_instances(req)
            insts = resp.body.instances.instance if resp.body.instances else []
            if not insts:
                return f"当前 region={region} 下无 ECS 实例（或当前角色无查询权限）。"
            lines = [f"## ECS 实例（region={region}，共 {resp.body.total_count or len(insts)} 台，展示前 {len(insts)}）"]
            lines.append("| 实例 ID | 名称 | 规格 | 状态 | 公网 IP |")
            lines.append("|---------|------|------|------|---------|")
            for i in insts:
                public_ip = ""
                if i.public_ip_address and i.public_ip_address.ip_address:
                    public_ip = i.public_ip_address.ip_address[0]
                elif i.eip_address and i.eip_address.ip_address:
                    public_ip = i.eip_address.ip_address
                lines.append(
                    f"| `{i.instance_id}` | {i.instance_name or '-'} | "
                    f"{i.instance_type or '-'} | {i.status} | {public_ip or '-'} |"
                )
            return "\n".join(lines)

        # ── 只读：get ─────────────────────────────────────────────────────────
        if action == "get":
            if not instance_id:
                return "❌ get 操作需要提供 instance_id。"
            req = ecs_models.DescribeInstancesRequest(
                region_id=region, instance_ids=f'["{instance_id}"]'
            )
            resp = client.describe_instances(req)
            insts = resp.body.instances.instance if resp.body.instances else []
            if not insts:
                return f"未找到实例 {instance_id}。"
            i = insts[0]
            return (
                f"## ECS 实例 `{instance_id}`\n"
                f"- 名称: {i.instance_name or '-'}\n"
                f"- 规格: {i.instance_type}\n"
                f"- 状态: {i.status}\n"
                f"- 镜像: {i.image_id}\n"
                f"- 区域: {i.region_id} / {i.zone_id}\n"
                f"- VPC: {i.vpc_attributes.vpc_id if i.vpc_attributes else '-'}\n"
                f"- 创建时间: {i.creation_time}\n"
                f"- 到期时间: {i.expired_time or '-'}\n"
                f"- CPU: {i.cpu} 核  内存: {i.memory} MB"
            )

        # ── 写操作：统一拦截 confirm ────────────────────────────────────────
        if action in ("create", "start", "stop", "reboot", "delete") and not confirm:
            return (
                f"⚠️ ECS `{action}` 是写操作（涉及计费/中断业务），"
                "请明确传入 `confirm=true` 后再调用。"
            )

        if action == "create":
            missing = [f for f, v in [
                ("instance_type", instance_type),
                ("image_id", image_id),
                ("vswitch_id", vswitch_id),
                ("security_group_id", security_group_id),
            ] if not v]
            if missing:
                return f"❌ create 缺少必填参数：{', '.join(missing)}"
            req = ecs_models.RunInstancesRequest(
                region_id=region,
                instance_type=instance_type,
                image_id=image_id,
                v_switch_id=vswitch_id,
                security_group_id=security_group_id,
                instance_name=instance_name or None,
                password=password or None,
                internet_max_bandwidth_out=internet_max_bandwidth_out or None,
                amount=1,
            )
            resp = client.run_instances(req)
            new_ids = resp.body.instance_id_sets.instance_id_set or []
            logger.info("[ECS] 创建实例成功 ids=%s 操作者=%s", new_ids, open_id)
            return (
                f"✅ ECS 实例创建指令已下发：\n"
                f"- 实例 ID：{', '.join(f'`{x}`' for x in new_ids) or '-'}\n"
                f"- 规格：{instance_type}　镜像：{image_id}\n"
                f"- 启动通常需要 1–2 分钟，可用 `get` 查看状态。"
            )

        if action == "start":
            if not instance_id:
                return "❌ start 操作需要 instance_id。"
            req = ecs_models.StartInstanceRequest(instance_id=instance_id)
            client.start_instance(req)
            logger.info("[ECS] 启动实例 %s 操作者=%s", instance_id, open_id)
            return f"✅ 实例 `{instance_id}` 启动指令已下发。"

        if action == "stop":
            if not instance_id:
                return "❌ stop 操作需要 instance_id。"
            req = ecs_models.StopInstanceRequest(instance_id=instance_id)
            client.stop_instance(req)
            logger.info("[ECS] 停止实例 %s 操作者=%s", instance_id, open_id)
            return f"✅ 实例 `{instance_id}` 停止指令已下发。"

        if action == "reboot":
            if not instance_id:
                return "❌ reboot 操作需要 instance_id。"
            req = ecs_models.RebootInstanceRequest(instance_id=instance_id)
            client.reboot_instance(req)
            logger.info("[ECS] 重启实例 %s 操作者=%s", instance_id, open_id)
            return f"✅ 实例 `{instance_id}` 重启指令已下发，预计 1–2 分钟恢复。"

        if action == "delete":
            if not instance_id:
                return "❌ delete 操作需要 instance_id。"
            req = ecs_models.DeleteInstanceRequest(instance_id=instance_id, force=True)
            client.delete_instance(req)
            logger.warning("[ECS] 释放实例 %s 操作者=%s", instance_id, open_id)
            return f"⚠️ 实例 `{instance_id}` 释放指令已下发（不可恢复）。"

        return f"❓ 未知 action：{action}，可选 list / get / create / start / stop / reboot / delete。"

    except Exception as e:
        logger.error("[ECS] 调用失败 action=%s err=%s", action, e, exc_info=True)
        return f"❌ ECS API 调用失败：{e}"


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

ecs_tool = StructuredTool.from_function(
    func=manage_ecs,
    name="manage_ecs",
    description=(
        "管理阿里云 ECS 实例。"
        "只读：list 列出实例、get 查看详情。"
        "写操作（create 创建 / start 启动 / stop 停止 / reboot 重启 / delete 释放）"
        "必须显式传入 confirm=true，否则 Bot 拒绝执行。"
        "调用时系统会自动注入飞书 open_id 用于 STS 凭证派发。"
    ),
    args_schema=ECSSchema,
)
