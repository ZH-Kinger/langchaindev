"""
快速测试云端 + 边缘两个 LLM API 的连通性与响应延迟。
运行：python test_llm_apis.py
"""
import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import settings

PROMPT = "用一句话介绍你自己。"


def test_api(name: str, llm_getter, extra_info: dict | None = None) -> None:
    print(f"\n{'='*50}")
    print(f"测试：{name}")
    if extra_info:
        for k, v in extra_info.items():
            masked = (str(v)[:8] + "****") if v and len(str(v)) > 8 else (v or "❌ 未配置")
            print(f"  {k}: {masked}")
    print(f"  Prompt: {PROMPT}")
    print("-" * 50)

    try:
        llm = llm_getter()
        t0 = time.time()
        resp = llm.invoke(PROMPT)
        elapsed = time.time() - t0
        content = resp.content if hasattr(resp, "content") else str(resp)
        print(f"  ✅ 成功 | 耗时 {elapsed:.2f}s")
        print(f"  回复: {content[:200]}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")


if __name__ == "__main__":
    from core.llm_factory import get_cloud_llm, get_edge_llm

    test_api(
        "☁️  云端 Qwen-Max (DashScope)",
        get_cloud_llm,
        {
            "model": settings.MODEL_NAME,
            "base_url": settings.BASE_URL,
            "api_key": settings.API_KEY,
        },
    )

    if settings.EDGE_API_KEY and settings.EDGE_BASE_URL:
        test_api(
            "🖥️  边缘 Qwen3-4B (ModelScope/本地)",
            get_edge_llm,
            {
                "model": settings.EDGE_MODEL_NAME,
                "base_url": settings.EDGE_BASE_URL,
                "api_key": settings.EDGE_API_KEY,
            },
        )
    else:
        print("\n" + "="*50)
        print("🖥️  边缘 Qwen3-4B")
        print("  ⚠️  跳过：EDGE_API_KEY 或 EDGE_BASE_URL 未在 .env 中配置")

    print("\n" + "="*50)
    print("测试完成")