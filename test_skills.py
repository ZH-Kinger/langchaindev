from tools.analysis_skills import alarm_reduction_tool
from tools.ops_skills import k8s_restart_tool

# 测试 1: 告警降噪（测试 Pandas 逻辑）
print("--- 测试告警降噪 ---")
res1 = alarm_reduction_tool.run({"file_name": "raw_alarms_200.csv"})
print(res1)

# 测试 2: 权限检查（测试逻辑分支）
print("\n--- 测试 K8s 重启（触发安全拦截） ---")
res2 = k8s_restart_tool.run({
    "pod_name": "order-service-v1",
    "namespace": "prod-cluster",
    "reason": "just try" # 原因不够详细
})
print(res2)