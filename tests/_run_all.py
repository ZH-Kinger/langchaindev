"""全流程测试 orchestrator（手动跑）。

按顺序执行：
  1. pytest 全量（单元 + 工具测试）
  2. LLM 路由烟测（_smoke_llm_route.py）
  3. 端到端业务烟测（_smoke_e2e.py）

跑：PYTHONIOENCODING=utf-8 python tests/_run_all.py
不创建真实云资源。
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def section(title: str):
    print("\n" + "█" * 70)
    print(f"█  {title}")
    print("█" * 70)


def run_step(name: str, args: list, cwd: Path = ROOT) -> tuple[bool, float, str]:
    """跑一个子进程，返回 (success, elapsed_seconds, last_500_chars)。"""
    t0 = time.time()
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - t0
    output = (proc.stdout or "") + (proc.stderr or "")
    print(output)
    return proc.returncode == 0, elapsed, output[-500:]


def main():
    section("第 1 步：pytest 全量（单元 + 工具测试，不含集成测试）")
    ok1, t1, out1 = run_step("pytest", [PYTHON, "-m", "pytest", "-v", "--tb=short"])

    section("第 2 步：LLM 路由真实烟测（约 1 分钟）")
    ok2, t2, out2 = run_step("llm_route", [PYTHON, "tests/_smoke_llm_route.py"])

    section("第 3 步：端到端业务烟测（不创建真实资源）")
    ok3, t3, out3 = run_step("e2e", [PYTHON, "tests/_smoke_e2e.py"])

    section("全流程汇总")
    icons = {True: "✅", False: "❌"}
    print(f"  {icons[ok1]} pytest 全量          {t1:6.1f}s")
    print(f"  {icons[ok2]} LLM 路由烟测         {t2:6.1f}s")
    print(f"  {icons[ok3]} 端到端业务烟测       {t3:6.1f}s")
    total = t1 + t2 + t3
    print(f"  {'─' * 50}")
    print(f"  总耗时：{total:.1f}s")
    print(f"  整体结果：{'✅ 全部通过' if all([ok1, ok2, ok3]) else '❌ 有失败项'}")

    return {
        "pytest":     {"ok": ok1, "elapsed": t1, "tail": out1},
        "llm_route":  {"ok": ok2, "elapsed": t2, "tail": out2},
        "e2e":        {"ok": ok3, "elapsed": t3, "tail": out3},
    }


if __name__ == "__main__":
    main()
