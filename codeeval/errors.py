"""Exceptions raised by the harness.

Everything codeeval raises on purpose derives from :class:`CodeEvalError`, so
the CLI and the API can catch one type at their boundary and turn it into an
exit status or an HTTP response. Each subclass names the pipeline stage that
failed: configuration, data, provider, execution, grading, storage.

Failures of the *program under evaluation* are results, not errors. A
completion that fails its tests or runs past its timeout produces an
``ExecutionResult`` with that status; :class:`ExecutionError` is for the
harness itself being unable to run it.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class CodeEvalError(Exception):
    """Base class for all errors raised by codeeval.

    ``str(error)`` is the human-readable message alone. ``details`` carries
    structured context (file names, record numbers, env var names) meant for
    logs and API responses. Subclasses keep this constructor signature so that
    every error pickles and unpickles unchanged.
    """

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: Mapping[str, Any] = MappingProxyType(dict(details or {}))

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        if not self.details:
            return f"{type(self).__name__}({self.message!r})"
        return f"{type(self).__name__}({self.message!r}, details={dict(self.details)!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        # MappingProxyType does not pickle; rebuild the proxy from a plain dict.
        return type(self), (self.message,), {"details": dict(self.details)}

    def __setstate__(self, state: dict[str, Any] | None) -> None:
        self.details = MappingProxyType(dict((state or {}).get("details", {})))


class ConfigError(CodeEvalError):
    """The settings are missing, malformed or inconsistent."""


class DataError(CodeEvalError):
    """A task, sample or result record, or the file holding it, is invalid."""


class ProviderError(CodeEvalError):
    """A model provider could not produce a completion."""


class ExecutionError(CodeEvalError):
    """The harness could not run a program (no worker, no sandbox, no image)."""


class GradingError(CodeEvalError):
    """A grader or judge could not produce a verdict."""


class StorageError(CodeEvalError):
    """Results could not be read from or written to the results store."""


__all__ = [
    "CodeEvalError",
    "ConfigError",
    "DataError",
    "ExecutionError",
    "GradingError",
    "ProviderError",
    "StorageError",
]
