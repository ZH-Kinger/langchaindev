"""
统一日志工厂。

日志目录：项目根目录 logs/
  - logs/app.log    INFO+，每天轮转，保留 14 天（主日志）
  - logs/error.log  ERROR+，每天轮转，保留 30 天（错误专项）

用法：
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("实例创建成功: %s", instance_name)
    logger.error("飞书回复失败: %s", e, exc_info=True)
"""
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# ── 日志目录（项目根 / logs）────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# ── 格式 ────────────────────────────────────────────────────────────────────────
_FMT     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)


def _make_file_handler(filename: str, level: int) -> TimedRotatingFileHandler:
    keep_days = 30 if level >= logging.ERROR else 14
    h = TimedRotatingFileHandler(
        _LOG_DIR / filename,
        when="midnight",
        backupCount=keep_days,
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(_formatter)
    return h


def _make_console_handler() -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)          # 终端只显示 INFO+，不刷 DEBUG
    h.setFormatter(_formatter)
    return h


# ── 根 logger（首次调用时初始化一次）────────────────────────────────────────────
_root_configured = False

def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_make_file_handler("app.log",   logging.INFO))
    root.addHandler(_make_file_handler("error.log", logging.ERROR))
    root.addHandler(_make_console_handler())
    # 抑制三方库噪音
    for noisy in ("urllib3", "requests", "httpx", "openai", "langchain",
                  "chromadb", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """返回以模块名为标识的 logger，首次调用时初始化根 logger。"""
    _configure_root()
    return logging.getLogger(name)
