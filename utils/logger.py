"""
统一日志工厂。

日志目录：项目根目录 logs/
  - logs/app.log    INFO+，每天轮转，保留 14 天
  - logs/error.log  ERROR+，每天轮转，保留 30 天

功能：
  - trace_id：每条飞书消息生成一个短 ID，串联同一次请求的全部日志
  - 错误飞书推送：ERROR 级别自动通知管理��（需调用 register_error_callback 注册）

用法：
    from utils.logger import get_logger, set_trace_id, clear_trace_id
    logger = get_logger(__name__)

    set_trace_id("a1b2c3d4")          # 请求开始时设置
    logger.info("处理消息: %s", text)  # 日志自动带上 trace_id
    clear_trace_id()                   # 请求结束时清除（可选）
"""
import logging
import sys
import contextvars
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# ── 日志目录 ─────────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# ── trace_id（每条飞书消息一个短 ID，用 contextvars 做线程隔离）──────────────────
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def set_trace_id(tid: str) -> None:
    _trace_id_var.set(tid)


def clear_trace_id() -> None:
    _trace_id_var.set("-")


class _TraceFilter(logging.Filter):
    """把当前 trace_id 注入每条日志记录。"""
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()
        return True


# ── 格式 ─────────────────────────────────────────────────────────────────────────
_FMT     = "%(asctime)s | %(levelname)-8s | %(trace_id)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)


def _make_file_handler(filename: str, level: int) -> TimedRotatingFileHandler:
    keep = 30 if level >= logging.ERROR else 14
    h = TimedRotatingFileHandler(
        _LOG_DIR / filename, when="midnight",
        backupCount=keep, encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(_formatter)
    h.addFilter(_TraceFilter())
    return h


def _make_console_handler() -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(_formatter)
    h.addFilter(_TraceFilter())
    return h


# ── 飞书错误推送（回调注册，避免循环导入）────────────────────────────────────────
_error_callback = None


def register_error_callback(fn) -> None:
    """注册一个回调：fn(message: str)，ERROR 日志触发时调用。"""
    global _error_callback
    _error_callback = fn


class _FeishuAlertHandler(logging.Handler):
    """ERROR+ 级别日志触发飞书推送（通过已注册的回调）。"""
    def emit(self, record: logging.LogRecord) -> None:
        if _error_callback is None:
            return
        try:
            tid   = getattr(record, "trace_id", "-")
            brief = self.format(record).splitlines()[0]   # 只取第一行，不发堆栈
            msg   = f"🚨 [AIOps ERROR]\ntrace={tid}\n{brief}"
            _error_callback(msg)
        except Exception:
            pass


# ── 根 logger 初始化（只初始化一次）─────────────────────────────────────────────
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
    root.addHandler(_FeishuAlertHandler())   # ERROR+ 触发飞书推送

    # 抑制三方库噪音
    for lib in ("urllib3", "requests", "httpx", "openai", "langchain",
                "chromadb", "apscheduler", "werkzeug"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
