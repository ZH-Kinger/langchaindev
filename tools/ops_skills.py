from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

# 受保护命名空间白名单：精确匹配，禁止 Agent 自动执行重启
# 扩展时在此处添加，不要修改 restart_service_logic 的判断逻辑
PROTECTED_NAMESPACES: frozenset[str] = frozenset({
    "prod", "production", "prd",
    "monitoring", "infra", "kube-system",
})


# --- 部件 1: 参数模型 ---
class K8sRestartSchema(BaseModel):
    pod_name: str = Field(description="需要重启的 Pod 名称，例如 'order-service-86fbc'")
    namespace: str = Field(
        default="default",
        description="Pod 所属的命名空间。如果不清楚，请务必使用 'default'。"
    )
    reason: str = Field(description="重启的具体原因，用于 9527 运维审计系统记录")


# --- 部件 2: 核心逻辑 ---
def restart_service_logic(pod_name: str, reason: str, namespace: str = "default") -> str:
    """K8s 重启逻辑执行器"""
    # 白名单检查：精确匹配，不依赖 reason 内容（防止提示注入绕过）
    if namespace.lower() in PROTECTED_NAMESPACES:
        return (
            f"❌ 拒绝操作：命名空间 [{namespace}] 为受保护环境，"
            f"禁止自动执行重启。请通过变更管理系统（ITSM）提交工单后由人工操作。"
        )

    # reason 仅用于审计记录，不参与授权判断
    return f"✅ [指令下达] 已在 {namespace} 成功重启 {pod_name}。原因：{reason}。"

# --- 部件 3: 工具封装 ---
k8s_restart_tool = StructuredTool.from_function(
    func=restart_service_logic,
    name="restart_k8s_service",
    description="当服务出现 OOM (内存溢出) 或死锁且无法自愈时，由云端决策 Agent 执行的重启操作。",
    args_schema=K8sRestartSchema
)