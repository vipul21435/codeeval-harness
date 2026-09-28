"""Failure categories for evaluator result files.

The evaluator records one of ``"passed"``, ``"timed out"`` or ``"failed:
<reason>"`` per sample. :func:`categorize` folds those strings into five
categories, :func:`breakdown` counts them per task and for the whole run from
the records of a ``*_results.jsonl`` file, and :func:`read_results` loads that
file, naming the file and line of any malformed record in a :class:`DataError`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from codeeval.errors import DataError

_SANDBOX_PREFIX = "failed: worker exited with"
# A compile-time SyntaxError (or IndentationError) raised by exec() of the check
# program prints as "<message> (<string>, line N)"; no runtime exception does.
_SYNTAX_SUFFIX = re.compile(r"\(<string>, line \d+\)$")


class Category(StrEnum):
    """Where a sample ended up."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    SANDBOX_ERROR = "sandbox_error"


def categorize(result: str) -> Category:
    """Map an evaluator ``result`` string to its :class:`Category`."""
    if result == "passed":
        return Category.PASS
    if result == "timed out":
        return Category.TIMEOUT
    if result.startswith(_SANDBOX_PREFIX):
        return Category.SANDBOX_ERROR
    if result.startswith("failed: "):
        return Category.SYNTAX_ERROR if _SYNTAX_SUFFIX.search(result) else Category.FAIL
    raise DataError("unknown result string", details={"result": result})


@dataclass(frozen=True, slots=True)
class Breakdown:
    """Category counts for a run and for each of its tasks."""

    run: Counter[Category] = field(default_factory=Counter)
    tasks: dict[str, Counter[Category]] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        return sum(self.run.values())

    def counts(self) -> dict[str, int]:
        """The run totals with every category present, in declaration order."""
        return {category.value: self.run[category] for category in Category}


def _fields(record: Mapping[str, Any]) -> tuple[str, str]:
    """The ``task_id`` and ``result`` strings of one record, or a :class:`DataError`."""
    task_id, result = record.get("task_id"), record.get("result")
    if not isinstance(task_id, str) or not isinstance(result, str):
        raise DataError(
            "record needs a task_id string and a result string",
            details={"task_id": task_id, "result": result},
        )
    return task_id, result


def breakdown(records: Iterable[Mapping[str, Any]]) -> Breakdown:
    """Count categories per task and overall from ``task_id``/``result`` records."""
    result = Breakdown()
    for record in records:
        task_id, verdict = _fields(record)
        category = categorize(verdict)
        result.run[category] += 1
        result.tasks.setdefault(task_id, Counter())[category] += 1
    return result


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Load a ``*_results.jsonl`` file.

    Every record needs a ``task_id`` string and a ``result`` string that
    :func:`categorize` accepts; the :class:`DataError` for a bad record names
    the file and line.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{path}: no such results file", details={"path": str(path)})
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise DataError(f"{path}:{number}: record needs task_id and result")
            try:
                categorize(_fields(record)[1])
            except DataError as exc:
                raise DataError(f"{path}:{number}: {exc}", details=exc.details) from exc
            records.append(record)
    if not records:
        raise DataError(f"{path}: no result records", details={"path": str(path)})
    return records


__all__ = ["Breakdown", "Category", "breakdown", "categorize", "read_results"]
