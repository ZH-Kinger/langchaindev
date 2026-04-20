"""
阿里云 PAI DSW 实例管理工具。
使用 alibabacloud-pai-dsw20220101 Python SDK 直接调用 API，无 Node.js 依赖。
"""
import json
from functools import lru_cache

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from alibabacloud_pai_dsw20220101.client import Client
from alibabacloud_pai_dsw20220101 import models as dsw_models
from alibabacloud_tea_openapi import models as open_api_models

from config.settings import settings


# ── Schema ────────────────────────────────────────────────────────────────────

class PAIDSWSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型，支持：\n"
            "  'list'        - 列出 DSW 实例（可按工作空间/名称/用户/资源过滤）\n"
            "  'get'         - 查询单个实例详情（需提供 instance_id）\n"
            "  'start'       - 启动实例（需提供 instance_id）\n"
            "  'stop'        - 停止实例（需提供 instance_id，可选 save_image）\n"
            "  'delete'      - 删除实例（需提供 instance_id，高风险操作）\n"
            "  'discover'    - 从现有实例提取可复用的资源配置（镜像、VPC、数据集等）\n"
            "  'specs'       - 列出可用机器规格（可选 accelerator_type: CPU 或 GPU）\n"
            "  'create_json' - 从 JSON 字符串创建实例（需提供 create_config_json）"
        )
    )
    instance_id: str = Field(
        default="",
        description="DSW 实例 ID（格式如 dsw-xxxxxxxx），get/start/stop/delete 时必填。"
    )
    workspace_id: str = Field(
        default="",
        description="工作空间 ID，用于 list/discover 过滤，留空则用 .env 默认值。"
    )
    instance_name: str = Field(
        default="",
        description="实例名称关键词，用于 list 过滤，留空则查全部。"
    )
    user_id: str = Field(
        default="",
        description="创建者用户 ID，用于 list/discover 过滤，留空则查全部。"
    )
    resource_id: str = Field(
        default="",
        description="资源组 ID，用于 list/discover 过滤，留空则用 .env 默认值。"
    )
    save_image: bool = Field(
        default=False,
        description="stop 操作时是否保存镜像，默认 False。"
    )
    accelerator_type: str = Field(
        default="GPU",
        description="specs 操作时查询的规格类型：'CPU' 或 'GPU'，默认 GPU。"
    )
    create_config_json: str = Field(
        default="",
        description=(
            "create_json 操作时使用的实例配置 JSON 字符串。"
            "字段：instanceName, workspaceId, resourceId, imageUrl, "
            "requestedResource(cpu/gpu/gputype/memory), datasets, userVpc, "
            "priority, accessibility 等。"
        )
    )


# ── SDK 客户端（缓存，相同凭证复用同一实例）──────────────────────────────────────

@lru_cache(maxsize=4)
def _get_client(region: str, ak: str, sk: str) -> Client:
    config = open_api_models.Config(
        access_key_id=ak,
        access_key_secret=sk,
        region_id=region,
        endpoint=f"pai-dsw.{region}.aliyuncs.com",
    )
    return Client(config)


def _client() -> Client | None:
    ak = settings.PAI_DSW_ACCESS_KEY_ID
    sk = settings.PAI_DSW_ACCESS_KEY_SECRET
    region = settings.PAI_DSW_REGION_ID or "cn-hangzhou"
    if not ak or not sk:
        return None
    return _get_client(region, ak, sk)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def manage_pai_dsw(
    action: str,
    instance_id: str = "",
    workspace_id: str = "",
    instance_name: str = "",
    user_id: str = "",
    resource_id: str = "",
    save_image: bool = False,
    accelerator_type: str = "GPU",
    create_config_json: str = "",
) -> str:
    """管理阿里云 PAI DSW 实例：查询、启动、停止、删除、创建、资源发现。"""
    client = _client()
    if client is None:
        return "❌ PAI_DSW_ACCESS_KEY_ID / PAI_DSW_ACCESS_KEY_SECRET 未配置，请在 .env 中设置。"

    action = action.strip().lower()
    ws = workspace_id or settings.PAI_DSW_WORKSPACE_ID
    rs = resource_id or settings.PAI_DSW_RESOURCE_ID

    try:
        # ── list ──────────────────────────────────────────────────────────────
        if action == "list":
            req = dsw_models.ListInstancesRequest(
                page_size=100,
                page_number=1,
                workspace_id=ws or None,
                instance_name=instance_name or None,
                create_user_id=user_id or None,
                resource_id=rs or None,
            )
            resp = client.list_instances(req)
            instances = resp.body.instances or []
            if not instances:
                return "当前没有 DSW 实例。"
            lines = ["## PAI DSW 实例列表\n"]
            for i, inst in enumerate(instances, 1):
                rr = inst.requested_resource
                cpu = rr.cpu if rr else "-"
                gpu = rr.gpu if rr else "-"
                mem = rr.memory if rr else "-"
                lines.append(
                    f"{i}. **{inst.instance_name}** | {inst.status} | "
                    f"{inst.accelerator_type} | "
                    f"CPU {cpu} GPU {gpu} MEM {mem} | "
                    f"`{inst.instance_id}`"
                )
            lines.append(f"\n共 {resp.body.total_count or len(instances)} 个实例。")
            return "\n".join(lines)

        # ── get ───────────────────────────────────────────────────────────────
        if action == "get":
            if not instance_id:
                return "❌ get 操作需要提供 instance_id。"
            req = dsw_models.GetInstanceRequest()
            resp = client.get_instance(instance_id, req)
            inst = resp.body
            rr = inst.requested_resource
            lines = [
                f"## 实例详情：{inst.instance_name}",
                f"- **状态**：{inst.status}",
                f"- **ID**：{inst.instance_id}",
                f"- **加速类型**：{inst.accelerator_type}",
                f"- **规格**：CPU {rr.cpu if rr else '-'} | GPU {rr.gpu if rr else '-'} | 内存 {rr.memory if rr else '-'}",
                f"- **镜像**：{inst.image_name or inst.image_url or '-'}",
                f"- **工作空间**：{inst.workspace_name} ({inst.workspace_id})",
                f"- **资源组**：{inst.resource_name} ({inst.resource_id})",
                f"- **创建时间**：{inst.gmt_create_time or '-'}",
            ]
            return "\n".join(lines)

        # ── start ─────────────────────────────────────────────────────────────
        if action == "start":
            if not instance_id:
                return "❌ start 操作需要提供 instance_id。"
            resp = client.start_instance(instance_id)
            return f"✅ 实例 {instance_id} 启动请求已提交（requestId: {resp.body.request_id or '-'}）。"

        # ── stop ──────────────────────────────────────────────────────────────
        if action == "stop":
            if not instance_id:
                return "❌ stop 操作需要提供 instance_id。"
            req = dsw_models.StopInstanceRequest(save_image=save_image)
            resp = client.stop_instance(instance_id, req)
            img_note = "（已保存镜像）" if save_image else ""
            return f"✅ 实例 {instance_id} 停止请求已提交{img_note}（requestId: {resp.body.request_id or '-'}）。"

        # ── delete ────────────────────────────────────────────────────────────
        if action == "delete":
            if not instance_id:
                return "❌ delete 操作需要提供 instance_id。"
            # 只允许删除 Stopped 状态的实例，防止误删运行中的实例
            check_req = dsw_models.GetInstanceRequest()
            check_resp = client.get_instance(instance_id, check_req)
            current_status = check_resp.body.status or ""
            if current_status.upper() not in ("STOPPED", "FAILED", "DELETED"):
                return (
                    f"⚠️ 拒绝删除：实例 {instance_id} 当前状态为 {current_status}，"
                    "只允许删除 Stopped/Failed 状态的实例，请先停止后再删除。"
                )
            resp = client.delete_instance(instance_id)
            return f"✅ 实例 {instance_id} 已删除（requestId: {resp.body.request_id or '-'}）。"

        # ── discover ──────────────────────────────────────────────────────────
        if action == "discover":
            req = dsw_models.ListInstancesRequest(
                page_size=100,
                page_number=1,
                workspace_id=ws or None,
                create_user_id=user_id or None,
                resource_id=rs or None,
            )
            resp = client.list_instances(req)
            instances = resp.body.instances or []

            def uniq(lst):
                seen, out = set(), []
                for x in lst:
                    if x and x not in seen:
                        seen.add(x); out.append(x)
                return out

            workspace_map = {i.workspace_id: i.workspace_name for i in instances if i.workspace_id}
            resource_map  = {i.resource_id:  i.resource_name  for i in instances if i.resource_id}
            image_catalog = [
                {"instanceName": i.instance_name, "imageName": i.image_name,
                 "acceleratorType": i.accelerator_type,
                 "cpu": i.requested_resource.cpu if i.requested_resource else "-",
                 "gpu": i.requested_resource.gpu if i.requested_resource else "-"}
                for i in instances
            ]
            dataset_candidates, seen_ds = [], set()
            for inst in instances:
                for ds in inst.datasets or []:
                    key = ds.uri or ""
                    if key and key not in seen_ds:
                        seen_ds.add(key)
                        dataset_candidates.append({
                            "uri": ds.uri, "mountPath": ds.mount_path,
                            "mountAccess": ds.mount_access, "from": inst.instance_name,
                        })

            lines = [
                f"## PAI DSW 资源配置发现（共 {resp.body.total_count or len(instances)} 个实例）\n",
                "### 工作空间",
            ]
            for wid, wname in workspace_map.items():
                lines.append(f"- {wname} (`{wid}`)")
            lines.append("\n### 资源组")
            for rid, rname in resource_map.items():
                lines.append(f"- {rname} (`{rid}`)")
            lines.append("\n### 镜像（最近使用前 5 个）")
            for img in image_catalog[:5]:
                lines.append(
                    f"- [{img['acceleratorType']}] {img['imageName']} "
                    f"CPU {img['cpu']} GPU {img['gpu']} — {img['instanceName']}"
                )
            if dataset_candidates:
                lines.append("\n### 数据集挂载")
                for ds in dataset_candidates:
                    lines.append(f"- `{ds['uri']}` → `{ds['mountPath']}` ({ds['mountAccess']})")
            return "\n".join(lines)

        # ── specs ─────────────────────────────────────────────────────────────
        if action == "specs":
            req = dsw_models.ListEcsSpecsRequest(
                page_size=20, page_number=1,
                accelerator_type=accelerator_type.upper(),
            )
            resp = client.list_ecs_specs(req)
            specs = resp.body.ecs_specs or []
            if not specs:
                return f"未找到 {accelerator_type} 类型规格。"
            lines = [f"## PAI DSW 可用规格（{accelerator_type}）\n"]
            for s in specs[:20]:
                lines.append(
                    f"- **{s.instance_type or '-'}**："
                    f"CPU {s.cpu or '-'} | GPU {s.gpu or '-'} ({s.gputype or '-'}) | 内存 {s.memory or '-'}"
                )
            return "\n".join(lines)

        # ── create_json ───────────────────────────────────────────────────────
        if action == "create_json":
            if not create_config_json.strip():
                return "❌ create_json 操作需要提供 create_config_json（JSON 字符串）。"
            try:
                cfg = json.loads(create_config_json)
            except json.JSONDecodeError as e:
                return f"❌ create_config_json 不是合法 JSON：{e}"

            rr_cfg = cfg.get("requestedResource") or cfg.get("requested_resource") or {}
            requested_resource = dsw_models.CreateInstanceRequestRequestedResource(
                cpu=str(rr_cfg["cpu"]) if rr_cfg.get("cpu") is not None else None,
                gpu=str(rr_cfg["gpu"]) if rr_cfg.get("gpu") is not None else None,
                gputype=rr_cfg.get("gputype") or rr_cfg.get("GPUType"),
                memory=str(rr_cfg["memory"]) if rr_cfg.get("memory") else None,
                shared_memory=str(rr_cfg.get("sharedMemory") or rr_cfg.get("shared_memory") or rr_cfg.get("memory") or ""),
            ) if rr_cfg else None

            datasets = []
            for ds in cfg.get("datasets", []):
                datasets.append(dsw_models.CreateInstanceRequestDatasets(
                    uri=ds.get("uri"),
                    mount_path=ds.get("mountPath") or ds.get("mount_path"),
                    mount_access=ds.get("mountAccess", "RW"),
                    dynamic=ds.get("dynamic", False),
                ))

            vpc_cfg = cfg.get("userVpc") or cfg.get("user_vpc")
            user_vpc = None
            if vpc_cfg:
                user_vpc = dsw_models.CreateInstanceRequestUserVpc(
                    vpc_id=vpc_cfg.get("vpcId") or vpc_cfg.get("vpc_id"),
                    v_switch_id=vpc_cfg.get("vSwitchId") or vpc_cfg.get("v_switch_id"),
                    security_group_id=vpc_cfg.get("securityGroupId") or vpc_cfg.get("security_group_id"),
                    default_route=vpc_cfg.get("defaultRoute", "eth0"),
                )

            req = dsw_models.CreateInstanceRequest(
                accessibility=cfg.get("accessibility", "PRIVATE"),
                instance_name=cfg.get("instanceName") or cfg.get("instance_name"),
                workspace_id=str(cfg["workspaceId"]) if cfg.get("workspaceId") else str(cfg.get("workspace_id", "")),
                resource_id=cfg.get("resourceId") or cfg.get("resource_id"),
                image_url=cfg.get("imageUrl") or cfg.get("image_url"),
                image_id=cfg.get("imageId") or cfg.get("image_id"),
                priority=cfg.get("priority"),
                requested_resource=requested_resource,
                datasets=datasets or None,
                user_vpc=user_vpc,
                environment_variables=cfg.get("environmentVariables") or cfg.get("environment_variables"),
            )
            resp = client.create_instance(req)
            body = resp.body
            if body.instance_id:
                return (
                    f"✅ 实例创建成功！\n"
                    f"- **实例 ID**：{body.instance_id}\n"
                    f"- **requestId**：{body.request_id or '-'}"
                )
            return f"⚠️ 创建响应：{body.to_map()}"

    except Exception as e:
        err_msg = str(e)
        if hasattr(e, "message"):
            err_msg = e.message
        elif hasattr(e, "data") and e.data:
            err_msg = f"{e} | {e.data}"
        return f"❌ API 调用失败：{err_msg}"

    return (
        f"❓ 未知操作：{action}\n"
        "支持的操作：list / get / start / stop / delete / discover / specs / create_json"
    )


# ── LangChain 工具封装 ─────────────────────────────────────────────────────────

pai_dsw_tool = StructuredTool.from_function(
    func=manage_pai_dsw,
    name="manage_pai_dsw",
    description=(
        "管理阿里云 PAI DSW（数据科学工作站）实例。"
        "支持列出实例、查询详情、启动/停止/删除实例、发现可复用资源配置、查看可用规格及创建实例。"
        "当用户询问'DSW 实例'、'PAI 工作站'、'启动/停止实例'、'查看 GPU 规格'等问题时调用。"
    ),
    args_schema=PAIDSWSchema,
)
