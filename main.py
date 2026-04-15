import sys

# Windows GBK 终端下强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
from config.settings import settings

settings.setup_env()

MODES = {
    "rag":    "RAG 对话模式   (知识库检索 + 对话记忆)",
    "agent":  "单 Agent 模式 (全工具 + 对话记忆)",
    "hybrid": "混合双模型模式 (边缘感知 → 云端决策)",
    "collab": "协作多 Agent  (诊断专家 → 执行官)",
    "bot":    "飞书 Bot 模式  (Webhook 服务，接收飞书消息并回复)",
}


def run_rag():
    from app import chat
    chat()


def run_agent():
    from core.agent import run
    run()


def run_hybrid():
    from core.hybrid_agents import run
    run()


def run_collab():
    from core.multi_agent_system import run
    run()


def run_bot():
    from core.feishu_bot import run
    run()


RUNNERS = {
    "rag":    run_rag,
    "agent":  run_agent,
    "hybrid": run_hybrid,
    "collab": run_collab,
    "bot":    run_bot,
}


def select_mode_interactive() -> str:
    mode_list = list(MODES.keys())
    print("\n" + "=" * 50)
    print("AIOps 智能运维助手 — 请选择运行模式")
    print("=" * 50)
    for i, key in enumerate(mode_list, 1):
        print(f"  {i}. [{key}] {MODES[key]}")
    print("=" * 50)
    while True:
        choice = input("请输入序号或模式名称 (默认 2-agent): ").strip()
        if choice == "":
            return "agent"
        if choice in MODES:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(mode_list):
            return mode_list[int(choice) - 1]
        print(f"  无效输入，请输入 1-{len(mode_list)} 或模式名称。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps 智能运维助手")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        help="运行模式（不指定则交互选择）",
    )
    args = parser.parse_args()

    mode = args.mode if args.mode else select_mode_interactive()

    print("\n" + "=" * 50)
    print(f"AIOps 助手 | {MODES[mode]}")
    print("输入 exit 退出")
    print("=" * 50)

    RUNNERS[mode]()
