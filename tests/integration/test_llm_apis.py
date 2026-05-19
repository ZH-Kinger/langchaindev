"""
LLM API 连通性测试（集成测试 — 默认 skip）。
跑：pytest -m integration tests/integration/test_llm_apis.py
"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

pytestmark = pytest.mark.integration

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import settings

PROMPT = "用一句话介绍你自己。"


def _invoke_llm(llm_getter) -> tuple[float, str]:
    """返回 (耗时秒, 回复前 200 字)。"""
    llm = llm_getter()
    t0 = time.time()
    resp = llm.invoke(PROMPT)
    elapsed = time.time() - t0
    content = resp.content if hasattr(resp, "content") else str(resp)
    return elapsed, content[:200]


def test_cloud_llm():
    """云端 Qwen-Max（DashScope）应能正常响应。"""
    if not settings.API_KEY:
        pytest.skip("OPENAI_API_KEY 未配置")
    from core.llm_factory import get_cloud_llm
    elapsed, content = _invoke_llm(get_cloud_llm)
    assert content, "回复不应为空"
    assert elapsed < 30, f"响应超过 30s（实际 {elapsed:.1f}s）"
    print(f"\n☁️  云端 LLM 响应 {elapsed:.2f}s：{content}")


def test_edge_llm():
    """边缘 Qwen3-4B 应能正常响应（如已配置且凭证有效）。
    边缘 LLM 是 hybrid 模式的可选依赖，凭证失效 / 服务不可达时 skip。
    """
    if not (settings.EDGE_API_KEY and settings.EDGE_BASE_URL):
        pytest.skip("EDGE_API_KEY 或 EDGE_BASE_URL 未配置")
    from core.llm_factory import get_edge_llm
    try:
        elapsed, content = _invoke_llm(get_edge_llm)
    except Exception as e:
        # 401 / 网络断 / 服务下线等：作为 skip 而不是 fail（边缘 LLM 是可选）
        pytest.skip(f"边缘 LLM 不可达：{type(e).__name__}: {str(e)[:120]}")
    assert content, "回复不应为空"
    print(f"\n🖥️  边缘 LLM 响应 {elapsed:.2f}s：{content}")
