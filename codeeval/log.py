"""Structured logging for the harness.

:func:`configure_logging` installs one handler on the root logger that writes
either JSON lines (one object per record with ``timestamp``, ``level``,
``logger`` and ``message``, followed by any extras such as ``run_id`` and
``task_id``) or a plain text fallback. Format and level come from the settings
(``VERIFYBENCH_LOG_FORMAT``, ``VERIFYBENCH_LOG_LEVEL``). Calling it again
replaces the handler instead of stacking a second one.

Extras travel two ways: per record, through ``logger.info("...", extra={...})``,
or for a whole block of code through :func:`bind_context`, which stamps
``run_id``, ``task_id`` or any other field on every record emitted inside the
block, including records from libraries that know nothing about the harness.
The context is a :mod:`contextvars` variable, so it follows async tasks but is
not inherited by threads started inside the block.
"""

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, TextIO

from codeeval.settings import LogFormat, Settings, get_settings

HANDLER_NAME = "verifybench"

# Attributes every LogRecord carries; anything else set on a record is an extra.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "asctime",
    "message",
}

_context: ContextVar[Mapping[str, Any]] = ContextVar(
    "verifybench_log_context", default=MappingProxyType({})
)


def current_context() -> Mapping[str, Any]:
    """The fields :func:`bind_context` currently stamps on every record."""
    return _context.get()


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """Stamp ``fields`` (``run_id``, ``task_id``, ...) on every record emitted in the block.

    Blocks nest; inner fields shadow outer ones and the outer context returns
    when the block ends. A field passed as a record's own ``extra`` wins over
    the bound context.
    """
    token = _context.set(MappingProxyType({**_context.get(), **fields}))
    try:
        yield
    finally:
        _context.reset(token)


class ContextFilter(logging.Filter):
    """Copies the bound context onto each record that passes through a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _context.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _timestamp(record: logging.LogRecord) -> str:
    stamp = datetime.fromtimestamp(record.created, tz=UTC)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def record_extras(record: logging.LogRecord) -> dict[str, Any]:
    """The fields a record carries beyond the standard LogRecord attributes, sorted by name."""
    return {
        key: value
        for key, value in sorted(vars(record).items())
        if key not in _RESERVED and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per record: timestamp, level, logger, message, extras, exception."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(record_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """``<timestamp> <level> <logger>: <message> key=value ...`` with tracebacks below."""

    def format(self, record: logging.LogRecord) -> str:
        line = f"{_timestamp(record)} {record.levelname:<8} {record.name}: {record.getMessage()}"
        for key, value in record_extras(record).items():
            line += f" {key}={value}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


def make_formatter(log_format: LogFormat) -> logging.Formatter:
    return JsonFormatter() if log_format == "json" else TextFormatter()


def configure_logging(
    settings: Settings | None = None, *, stream: TextIO | None = None
) -> logging.Handler:
    """Install the harness handler on the root logger, replacing an earlier one.

    ``settings`` defaults to :func:`get_settings`; ``stream`` defaults to
    ``sys.stderr`` as it is at call time. Returns the handler so callers (tests)
    can inspect or remove it.
    """
    if settings is None:
        settings = get_settings()
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == HANDLER_NAME:
            root.removeHandler(existing)
            existing.close()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(make_formatter(settings.log_format))
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    return handler


__all__ = [
    "HANDLER_NAME",
    "ContextFilter",
    "JsonFormatter",
    "TextFormatter",
    "bind_context",
    "configure_logging",
    "current_context",
    "make_formatter",
    "record_extras",
]
