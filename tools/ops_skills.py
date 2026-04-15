from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

# --- 部件 1: 参数模型 (强化描述) ---
class K8sRestartSchema(BaseModel):
    pod_name: str = Field(description="需要重启的 Pod 名称，例如 'order-service-86fbc'")
    # 在 Schema 层级设置默认值
    namespace: str = Field(
        default="default",
        description="Pod 所属的命名空间。如果不清楚，请务必使用 'default'。"
    )
    reason: str = Field(description="重启的具体原因，用于 9527 运维审计系统记录")

# --- 部件 2: 核心逻辑 (修复函数签名) ---
# 关键改动：在函数定义处增加 namespace="default"
def restart_service_logic(pod_name: str, reason: str, namespace: str = "default") -> str:
    """
    K8s 重启逻辑执行器
    注意：参数顺序建议将有默认值的放在最后，符合 Python 惯例
    """
    # 安全审计逻辑
    if "prod" in namespace.lower() and "admin" not in reason.lower():
        return f"❌ 拒绝操作：生产环境 {namespace} 的重启必须包含详细的管理员(admin)审计原因。"

    # 模拟执行
    return f"✅ [指令下达] 已在 {namespace} 成功重启 {pod_name}。原因：{reason}。"

# --- 部件 3: 工具封装 ---
k8s_restart_tool = StructuredTool.from_function(
    func=restart_service_logic,
    name="restart_k8s_service",
    description="当服务出现 OOM (内存溢出) 或死锁且无法自愈时，由云端决策 Agent 执行的重启操作。",
    args_schema=K8sRestartSchema
)