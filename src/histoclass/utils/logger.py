"""Logging utilities for histoclass."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Final


DEFAULT_LOG_LEVEL: Final[int] = logging.INFO
ALLOWED_LEVELS: Final[frozenset[int]] = frozenset(
    {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR}
)
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S.%f"


class LevelAllowlistFilter(logging.Filter):
    """@brief 日志等级白名单过滤器；Allowlist filter for log levels.

    @param record 日志记录对象；Log record object.
    @return 若日志等级在白名单内则返回 True；Returns True if level is allowed.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in ALLOWED_LEVELS


class MicrosecondFormatter(logging.Formatter):
    """@brief 微秒级时间戳格式化器；Formatter with microsecond precision timestamps.

    @param record 日志记录对象；Log record object.
    @param datefmt 日期格式字符串；Date format string.
    @return 格式化后的时间字符串；Formatted datetime string.
    """

    def formatTime(  # noqa: N802
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime(datefmt or DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """@brief 获取模块级 logger；Get a module-scoped logger instance.

    @param name logger 名称，通常使用模块名；Logger name, usually module name.
    @return 配置完成且可直接使用的 logger；A configured, ready-to-use logger.
    @note 只会为同名 logger 安装一次 handler，避免重复输出。
    """
    logger = logging.getLogger(name)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.addFilter(LevelAllowlistFilter())
    handler.setFormatter(MicrosecondFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    return logger


__all__ = ["get_logger"]
