"""codeeval: an evaluation harness for LLM-generated code.

This package builds on top of the upstream ``human_eval`` package (kept
importable and unchanged in spirit) and hosts the harness-specific modules:
providers, graders, sandboxed execution, reports and the CLI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("codeeval")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
