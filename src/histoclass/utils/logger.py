"""Logging utilities for histoclass."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from threading import RLock
from typing import Final, Mapping, TextIO
from weakref import WeakSet


DEFAULT_LOG_LEVEL: Final[int] = logging.INFO
ALLOWED_LEVELS: Final[frozenset[int]] = frozenset(
    {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
)
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S.%f"
_SUPPORTED_TARGETS: Final[frozenset[str]] = frozenset({"stdout", "stderr", "file"})

_LOGGER_LOCK = RLock()
_MANAGED_LOGGERS: WeakSet[logging.Logger] = WeakSet()
_GLOBAL_LEVEL: int = DEFAULT_LOG_LEVEL
_DEFAULT_LEVEL_TARGETS: Final[dict[int, str]] = {
    logging.DEBUG: "stdout",
    logging.INFO: "stdout",
    logging.WARNING: "stderr",
    logging.ERROR: "stderr",
    logging.CRITICAL: "stderr",
}
_LEVEL_TARGETS: dict[int, str] = dict(_DEFAULT_LEVEL_TARGETS)
_FILE_PATH: Path | None = None


class LevelAllowlistFilter(logging.Filter):
    """@brief 日志等级白名单过滤器；Allowlist filter for log levels.

    @param record 日志记录对象；Log record object.
    @return 若日志等级在白名单内则返回 True；Returns True if level is allowed.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in ALLOWED_LEVELS


class LevelSetFilter(logging.Filter):
    """@brief 级别集合过滤器；Level-set filter.

    @param record 日志记录对象；Log record object.
    @return 当日志级别在集合内时返回 True；Returns True if level is in set.
    """

    def __init__(self, levels: set[int]) -> None:
        """@brief 初始化集合过滤器；Initialize set filter.

        @param levels 允许通过的日志级别集合；Allowed log levels.
        """
        super().__init__()
        self._levels = frozenset(levels)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self._levels


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


class _LazyFileHandler(logging.Handler):
    """@brief 按需创建文件的处理器；Lazy file-creating handler.

    @note 首次写入时才创建目录与文件，线程安全；Creates file lazily on first emit, thread-safe.
    """

    def __init__(self, path: Path) -> None:
        """@brief 初始化懒文件处理器；Initialize lazy file handler.

        @param path 日志文件路径；Log file path.
        """
        super().__init__(level=logging.DEBUG)
        self._path = path
        self._stream: TextIO | None = None
        self._open_lock = RLock()

    def emit(self, record: logging.LogRecord) -> None:
        """@brief 写入日志记录；Emit log record.

        @param record 日志记录对象；Log record object.
        """
        try:
            message = self.format(record)
            stream = self._ensure_stream()
            stream.write(message + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """@brief 关闭文件流；Close file stream."""
        with self._open_lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            stream.close()
        super().close()

    def _ensure_stream(self) -> TextIO:
        with self._open_lock:
            if self._stream is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self._path.open("a", encoding="utf-8")
            return self._stream


def _build_formatter() -> MicrosecondFormatter:
    return MicrosecondFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)


def _build_stream_handler(target: str) -> logging.Handler:
    stream = sys.stdout if target == "stdout" else sys.stderr
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(LevelAllowlistFilter())
    handler.setFormatter(_build_formatter())
    return handler


def _build_file_handler(path: Path) -> logging.Handler:
    handler = _LazyFileHandler(path)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(LevelAllowlistFilter())
    handler.setFormatter(_build_formatter())
    return handler


def _target_to_handler(target: str, *, file_path: Path | None) -> logging.Handler:
    if target == "stdout":
        return _build_stream_handler("stdout")
    if target == "stderr":
        return _build_stream_handler("stderr")
    if target == "file":
        if file_path is None:
            raise ValueError("File target is selected but no log file path is configured.")
        return _build_file_handler(file_path)
    raise ValueError(f"Unsupported log target: {target}")


def _normalize_level(level: int | str | None) -> int:
    if level is None:
        return DEFAULT_LOG_LEVEL
    if isinstance(level, int):
        normalized = level
    elif isinstance(level, str):
        normalized = logging.getLevelName(level.upper())
        if isinstance(normalized, str):
            raise ValueError(f"Unsupported log level: {level}")
    else:
        raise TypeError(f"log level must be int/str/None, got {type(level).__name__}.")

    if normalized not in ALLOWED_LEVELS:
        raise ValueError(f"Log level {normalized} is not supported.")
    return normalized


def _normalize_target_map(level_targets: Mapping[str, str] | None) -> dict[int, str]:
    merged = dict(_DEFAULT_LEVEL_TARGETS)
    if level_targets is None:
        return merged

    for raw_level_name, raw_target in level_targets.items():
        level_name = str(raw_level_name).upper()
        level_value = logging.getLevelName(level_name)
        if isinstance(level_value, str) or level_value not in ALLOWED_LEVELS:
            raise ValueError(f"Unsupported level in logging.streams: {raw_level_name}")

        target = str(raw_target).lower()
        if target not in _SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported stream target for {raw_level_name}: {raw_target}"
            )
        merged[level_value] = target

    return merged


def _configure_logger_handlers(logger: logging.Logger) -> None:
    logger.handlers.clear()
    logger.setLevel(_GLOBAL_LEVEL)
    logger.propagate = False

    target_levels: dict[str, set[int]] = {}
    for level in ALLOWED_LEVELS:
        target = _LEVEL_TARGETS[level]
        target_levels.setdefault(target, set()).add(level)

    for target, levels in target_levels.items():
        handler = _target_to_handler(target, file_path=_FILE_PATH)
        handler.addFilter(LevelSetFilter(levels))
        logger.addHandler(handler)


def configure_logging(
    *,
    level: int | str | None = None,
    level_targets: Mapping[str, str] | None = None,
    file_path: str | Path | None = None,
) -> None:
    """@brief 配置全局日志路由；Configure global log routing.

    @param level logger 最低级别；Global logger level threshold.
    @param level_targets 各级别输出目标映射；Per-level target mapping.
    @param file_path 文件输出路径（仅 target=file 时需要）；File path for file target.
    @note 会重配已创建 logger，确保配置即时生效；Reconfigures existing loggers immediately.
    """
    global _GLOBAL_LEVEL, _LEVEL_TARGETS, _FILE_PATH

    with _LOGGER_LOCK:
        normalized_level = _normalize_level(level)
        normalized_targets = _normalize_target_map(level_targets)
        resolved_file_path = (
            None if file_path is None else Path(file_path).expanduser().resolve()
        )

        if any(target == "file" for target in normalized_targets.values()):
            if resolved_file_path is None:
                raise ValueError(
                    "logging.file_path is required when any level target is 'file'."
                )

        _GLOBAL_LEVEL = normalized_level
        _LEVEL_TARGETS = normalized_targets
        _FILE_PATH = resolved_file_path

        for managed_logger in list(_MANAGED_LOGGERS):
            _configure_logger_handlers(managed_logger)


def get_logger(name: str) -> logging.Logger:
    """@brief 获取模块级 logger；Get a module-scoped logger instance.

    @param name logger 名称，通常使用模块名；Logger name, usually module name.
    @return 配置完成且可直接使用的 logger；A configured, ready-to-use logger.
    @note 只会为同名 logger 安装一次 handler，避免重复输出。
    """
    with _LOGGER_LOCK:
        logger = logging.getLogger(name)
        _MANAGED_LOGGERS.add(logger)
        _configure_logger_handlers(logger)
        return logger


__all__ = ["configure_logging", "get_logger"]
