import logging
from contextvars import ContextVar

import colorlog

# trace_id 上下文变量：每个请求/任务在入口处 set，日志自动带上
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str) -> None:
    """设置当前上下文的 trace_id（在请求/任务入口处调用）。"""
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """获取当前上下文的 trace_id，无则返回 '-'。"""
    return _trace_id_var.get()


class _TraceFilter(logging.Filter):
    """把 contextvar 中的 trace_id 注入到每一条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


logger = logging.getLogger("knowledge_base")
logger.setLevel(logging.INFO)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - [%(trace_id)s] - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
))

logger.handlers.clear()
logger.addHandler(handler)
logger.addFilter(_TraceFilter())

# 避免日志向上传播到 root logger 造成重复打印
logger.propagate = False
