"""真实 LLM 路由烟测（非自动化测试，手动跑用）。

跑：PYTHONIOENCODING=utf-8 python tests/_smoke_llm_route.py
需要 .env 配置 OPENAI_API_KEY。
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.intent_router import route, clear_cache
from core.agent import _select_tools

# ── 8 个真实场景：覆盖快通道、LLM、降级、多意图、歧义 ──────────────────────
CASES = [
    # (输入, 期望命中的意图, 是否快通道)
    ("查看我的 dsw 实例",          ["pai_dsw"],         True),
    ("整个集群效率不太行",         ["advisor", "cluster"], False),
    ("跑训练慢得不行",            ["training"],        False),
    ("成本怎么降低",              ["advisor"],         False),
    ("我的服务挂了快帮我看看",     ["ops"],             False),
    ("帮我看一下昨天我做了哪些事", ["workflow"],        False),
    ("发个飞书通知运维",          ["notify"],          False),
    ("看下知识库怎么部署 K8s",     ["ops", "knowledge"], False),  # 含 k8s → 快通道
]


def main():
    print("=" * 70)
    print("真实 LLM 意图路由烟测")
    print("=" * 70)

    total_time = 0.0
    llm_calls = 0

    for i, (text, expected_hint, expected_fast) in enumerate(CASES, 1):
        clear_cache()  # 每次都强制重新调 LLM

        # 第一次：测路由准确性 + 延迟
        t0 = time.time()
        intents = route(text)
        elapsed = time.time() - t0

        # 走完整 _select_tools，看最终工具集
        tools = _select_tools(text)
        tool_names = sorted(t.name for t in tools)

        # 判断是否走的快通道（知识库覆盖会让"含 fast token"也不走快通道）
        from core.agent import _FAST_PATH_TOKENS, _KNOWLEDGE_OVERRIDE_TOKENS
        t = text.lower()
        knowledge_intent = any(tok in t for tok in _KNOWLEDGE_OVERRIDE_TOKENS)
        is_fast = (not knowledge_intent) and any(tok in t for tok in _FAST_PATH_TOKENS)

        if not is_fast:
            llm_calls += 1
            total_time += elapsed

        print(f"\n[{i}] 输入：{text!r}")
        print(f"    快通道：{'✓' if is_fast else '✗'}（期望 {'✓' if expected_fast else '✗'}）")
        if not is_fast:
            print(f"    LLM 路由耗时：{elapsed*1000:.0f}ms")
            print(f"    LLM 识别意图：{intents}")
        print(f"    最终工具数：{len(tools)}（共 {len(tool_names)}）")
        print(f"    工具列表：{tool_names}")
        print(f"    期望命中意图（含其一即可）：{expected_hint}")

    print("\n" + "=" * 70)
    print(f"总结：LLM 调用 {llm_calls} 次，平均耗时 {total_time/max(llm_calls,1)*1000:.0f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
