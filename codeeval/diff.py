"""Run-to-run regression diff between two evaluator result files.

:func:`diff_results` compares the per-task pass *rate* (passes / samples) of
a *before* and an *after* run (records as :func:`codeeval.breakdown.read_results`
returns them) and sorts every task into ``regressed`` (lower rate), ``improved``
(higher rate), ``unchanged``, ``added`` (only in *after*) or ``removed`` (only
in *before*). Rates are exact :class:`fractions.Fraction` values, so runs with
different sample counts per task compare correctly (5/10 equals 1/2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from codeeval.breakdown import Category, breakdown


@dataclass(frozen=True, slots=True)
class TaskChange:
    """Passes out of samples for one task in each run."""

    task_id: str
    before: tuple[int, int]
    after: tuple[int, int]

    @property
    def before_rate(self) -> Fraction:
        """Exact pass rate in the *before* run."""
        return Fraction(self.before[1], self.before[0])

    @property
    def after_rate(self) -> Fraction:
        """Exact pass rate in the *after* run."""
        return Fraction(self.after[1], self.after[0])

    @property
    def delta(self) -> Fraction:
        """Change in pass rate (not in pass count) from *before* to *after*."""
        return self.after_rate - self.before_rate


@dataclass(frozen=True, slots=True)
class RunDiff:
    """Tasks grouped by how their pass rate moved between two runs."""

    regressed: list[TaskChange] = field(default_factory=list)
    improved: list[TaskChange] = field(default_factory=list)
    unchanged: list[TaskChange] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no task regressed and no task disappeared."""
        return not self.regressed and not self.removed


def _passes(records: Iterable[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    tasks = breakdown(records).tasks
    return {
        task_id: (sum(counts.values()), counts[Category.PASS]) for task_id, counts in tasks.items()
    }


def diff_results(
    before: Iterable[Mapping[str, Any]], after: Iterable[Mapping[str, Any]]
) -> RunDiff:
    """Group tasks by the change in pass rate from ``before`` to ``after``."""
    old, new = _passes(before), _passes(after)
    diff = RunDiff(added=sorted(new.keys() - old.keys()), removed=sorted(old.keys() - new.keys()))
    for task_id in sorted(old.keys() & new.keys()):
        change = TaskChange(task_id, old[task_id], new[task_id])
        if change.delta < 0:
            diff.regressed.append(change)
        elif change.delta > 0:
            diff.improved.append(change)
        else:
            diff.unchanged.append(change)
    return diff


__all__ = ["RunDiff", "TaskChange", "diff_results"]
