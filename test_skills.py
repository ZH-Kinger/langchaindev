"""
核心工具路径验证。
运行：python test_skills.py
全部断言通过后打印 OK，任意失败则抛出 AssertionError。
"""
from tools.analysis_skills import alarm_reduction_tool
from tools.ops_skills import k8s_restart_tool

# ── 测试 1：告警降噪（Pandas 逻辑）────────────────────────────────────────────
print("--- 测试告警降噪 ---")
res1 = alarm_reduction_tool.run({"file_name": "raw_alarms_200.csv"})
print(res1)
assert isinstance(res1, str) and len(res1) > 0, \
    "告警降噪工具应返回非空字符串"
assert "Traceback" not in res1, \
    "告警降噪工具不应抛出未捕获异常"

# ── 测试 2：受保护命名空间拦截（安全边界）────────────────────────────────────
print("\n--- 测试 K8s 重启（受保护命名空间应被拦截）---")
res2 = k8s_restart_tool.run({
    "pod_name":  "order-service-v1",
    "namespace": "prod",          # 精确匹配 PROTECTED_NAMESPACES
    "reason":    "测试安全拦截",
})
print(res2)
assert "拒绝" in res2 or "禁止" in res2 or "受保护" in res2, \
    f"prod 命名空间应被安全拦截，实际返回：{res2!r}"

# ── 测试 3：正常命名空间允许重启 ──────────────────────────────────────────────
print("\n--- 测试 K8s 重启（非保护命名空间应允许）---")
res3 = k8s_restart_tool.run({
    "pod_name":  "order-service-v1",
    "namespace": "staging",
    "reason":    "OOM 内存溢出，需重启恢复",
})
print(res3)
assert "成功" in res3 or "重启" in res3, \
    f"staging 命名空间应允许重启，实际返回：{res3!r}"

print("\n✅ 全部断言通过")