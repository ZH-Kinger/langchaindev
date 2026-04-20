"""
阿里云 PAI DSW 实例管理工具。
通过调用 pai-dsw-cli.js（Node.js CLI）实现 DSW 实例的查询、启停与资源发现。
CLI 路径由环境变量 PAI_DSW_CLI_PATH 指定。
"""
import json
import os
import subprocess
import tempfile

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

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
        description="工作空间 ID，用于 list/discover 过滤，留空则查全部。"
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
        description="资源组 ID，用于 list/discover 过滤，留空则查全部。"
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
            "字段参考 pai-dsw-create.example.json：instanceName, workspaceId, "
            "resourceId, imageUrl, requestedResource(CPU/GPU/GPUType/memory), "
            "datasets, userVpc, priority, accessibility 等。"
        )
    )


# ── 内部工具函数 ───────────────────────────────────────────────────────────────

def _cli_path() -> str:
    path = settings.PAI_DSW_CLI_PATH
    if not path:
        return ""
    return path


def _run_cli(args: list[str]) -> str:
    cli = _cli_path()
    if not cli:
        return "❌ PAI_DSW_CLI_PATH 未配置，请在 .env 中设置 CLI 脚本路径。"

    env = os.environ.copy()
    # 将 settings 中的阿里云密钥透传给 Node.js 进程
    if settings.PAI_DSW_ACCESS_KEY_ID:
        env["ALIBABA_CLOUD_ACCESS_KEY_ID"] = settings.PAI_DSW_ACCESS_KEY_ID
    if settings.PAI_DSW_ACCESS_KEY_SECRET:
        env["ALIBABA_CLOUD_ACCESS_KEY_SECRET"] = settings.PAI_DSW_ACCESS_KEY_SECRET
    if settings.PAI_DSW_REGION_ID:
        env["ALIBABA_CLOUD_REGION_ID"] = settings.PAI_DSW_REGION_ID

    cmd = ["node", cli] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            env=env,
            timeout=30,
        )
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        if result.returncode != 0:
            err = stderr.strip() or stdout.strip()
            return f"❌ CLI 执行失败（exit {result.returncode}）：{err}"
        return stdout.strip()
    except FileNotFoundError:
        return "❌ 未找到 node 命令，请确认 Node.js 已安装并在 PATH 中。"
    except subprocess.TimeoutExpired:
        return "❌ CLI 执行超时（>30s），请检查网络或阿里云 API 连通性。"
    except Exception as e:
        return f"❌ 执行异常：{e}"


def _arg(val: str) -> str:
    return val if val else "-"


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
    action = action.strip().lower()

    if action == "list":
        ws = workspace_id or settings.PAI_DSW_WORKSPACE_ID
        rs = resource_id or settings.PAI_DSW_RESOURCE_ID
        raw = _run_cli([
            "picklist",
            _arg(ws),
            _arg(user_id),
            _arg(rs),
        ])
        try:
            data = json.loads(raw)
            instances = data.get("items") or data.get("instances") or []
            if not instances:
                return f"当前没有 DSW 实例。（API 原始响应：{raw[:300]}）"
            lines = ["## PAI DSW 实例列表\n"]
            for i, inst in enumerate(instances, 1):
                rr = inst.get("requestedResource") or {}
                lines.append(
                    f"{i}. **{inst.get('instanceName')}** | {inst.get('status')} | "
                    f"{inst.get('acceleratorType')} | "
                    f"CPU {rr.get('CPU', '-')} GPU {rr.get('GPU', '-')} MEM {rr.get('memory', '-')} | "
                    f"`{inst.get('instanceId')}`"
                )
            lines.append(f"\n共 {data.get('totalCount', len(instances))} 个实例。")
            return "\n".join(lines)
        except Exception:
            return raw

    if action == "get":
        if not instance_id:
            return "❌ get 操作需要提供 instance_id。"
        raw = _run_cli(["instance", instance_id])
        try:
            data = json.loads(raw)
            inst = data
            rr = inst.get("requestedResource") or {}
            lines = [
                f"## 实例详情：{inst.get('instanceName')}",
                f"- **状态**：{inst.get('status')}",
                f"- **ID**：{inst.get('instanceId')}",
                f"- **加速类型**：{inst.get('acceleratorType')}",
                f"- **规格**：CPU {rr.get('CPU', '-')} | GPU {rr.get('GPU', '-')} | 内存 {rr.get('memory', '-')}",
                f"- **镜像**：{inst.get('imageName', inst.get('imageUrl', '-'))}",
                f"- **工作空间**：{inst.get('workspaceName')} ({inst.get('workspaceId')})",
                f"- **资源组**：{inst.get('resourceName')} ({inst.get('resourceId')})",
                f"- **创建时间**：{inst.get('gmtCreateTime', '-')}",
            ]
            return "\n".join(lines)
        except Exception:
            return raw

    if action == "start":
        if not instance_id:
            return "❌ start 操作需要提供 instance_id。"
        raw = _run_cli(["start", instance_id])
        try:
            data = json.loads(raw)
            if data.get("success") or data.get("instanceId"):
                return f"✅ 实例 {instance_id} 启动请求已提交（requestId: {data.get('requestId', '-')}）。"
            return f"⚠️ 响应：{raw}"
        except Exception:
            return raw

    if action == "stop":
        if not instance_id:
            return "❌ stop 操作需要提供 instance_id。"
        raw = _run_cli(["stop", instance_id, "true" if save_image else "false"])
        try:
            data = json.loads(raw)
            img_note = "（已保存镜像）" if save_image else ""
            if data.get("success") or data.get("instanceId"):
                return f"✅ 实例 {instance_id} 停止请求已提交{img_note}（requestId: {data.get('requestId', '-')}）。"
            return f"⚠️ 响应：{raw}"
        except Exception:
            return raw

    if action == "delete":
        if not instance_id:
            return "❌ delete 操作需要提供 instance_id。"
        raw = _run_cli(["delete", instance_id])
        try:
            data = json.loads(raw)
            if data.get("success"):
                return f"✅ 实例 {instance_id} 已删除（requestId: {data.get('requestId', '-')}）。"
            return f"⚠️ 响应：{raw}"
        except Exception:
            return raw

    if action == "discover":
        ws = workspace_id or settings.PAI_DSW_WORKSPACE_ID
        rs = resource_id or settings.PAI_DSW_RESOURCE_ID
        raw = _run_cli([
            "discover",
            _arg(ws),
            _arg(user_id),
            _arg(rs),
        ])
        try:
            data = json.loads(raw)
            lines = [
                f"## PAI DSW 资源配置发现（共 {data.get('totalCount', 0)} 个实例）\n",
                "### 工作空间",
            ]
            for w in data.get("workspaceCandidates", []):
                lines.append(f"- {w.get('workspaceName')} (`{w.get('workspaceId')}`)")
            lines.append("\n### 资源组")
            for r in data.get("resourceCandidates", []):
                lines.append(f"- {r.get('resourceName')} (`{r.get('resourceId')}`)")
            lines.append("\n### 镜像（最近使用）")
            for img in data.get("imageCandidates", [])[:5]:
                lines.append(f"- [{img.get('acceleratorType')}] {img.get('imageName')} — {img.get('instanceName')}")
            if data.get("datasetCandidates"):
                lines.append("\n### 数据集挂载")
                seen = set()
                for ds in data["datasetCandidates"]:
                    key = ds.get("uri", "")
                    if key not in seen:
                        seen.add(key)
                        lines.append(f"- `{ds.get('uri')}` → `{ds.get('mountPath')}` ({ds.get('mountAccess', 'RW')})")
            return "\n".join(lines)
        except Exception:
            return raw

    if action == "specs":
        raw = _run_cli(["specs", accelerator_type.upper()])
        try:
            data = json.loads(raw)
            specs = data.get("specs", data.get("ecsSpecs", []))
            if not specs:
                return f"未找到 {accelerator_type} 类型规格，原始响应：{raw[:500]}"
            lines = [f"## PAI DSW 可用规格（{accelerator_type}）\n"]
            for s in specs[:20]:
                lines.append(
                    f"- **{s.get('resourceType', s.get('instanceType', '-'))}**："
                    f"CPU {s.get('cpu', '-')} | GPU {s.get('gpu', '-')} | 内存 {s.get('memory', '-')}"
                )
            return "\n".join(lines)
        except Exception:
            return raw

    if action == "create_json":
        if not create_config_json.strip():
            return "❌ create_json 操作需要提供 create_config_json（JSON 字符串）。"
        try:
            json.loads(create_config_json)
        except json.JSONDecodeError as e:
            return f"❌ create_config_json 不是合法 JSON：{e}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(create_config_json)
            tmp_path = f.name
        try:
            raw = _run_cli(["create", "--file", tmp_path])
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        try:
            data = json.loads(raw)
            if data.get("success") or data.get("instanceId"):
                return (
                    f"✅ 实例创建成功！\n"
                    f"- **实例 ID**：{data.get('instanceId', '-')}\n"
                    f"- **requestId**：{data.get('requestId', '-')}"
                )
            return f"⚠️ 响应：{raw}"
        except Exception:
            return raw

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