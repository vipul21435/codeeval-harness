"""pass@k with bootstrap confidence intervals.

:func:`pass_at_k` is the unbiased estimator from the Codex paper
(``1 - C(n - c, k) / C(n, k)``) for one task with ``n`` samples of which ``c``
passed. :func:`bootstrap_pass_at_k` averages it over a suite of tasks and
resamples the *tasks* with replacement (seeded, stdlib ``random``) to give the
mean a percentile confidence interval. Bad inputs raise :class:`DataError`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from codeeval.errors import DataError

Count = tuple[int, int]
"""``(n, c)``: samples drawn for a task and how many of them passed."""


@dataclass(frozen=True, slots=True)
class PassAtK:
    """A pass@k estimate for a suite with its bootstrap interval."""

    k: int
    estimate: float
    low: float
    high: float
    tasks: int
    resamples: int
    confidence: float
    seed: int


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task: ``1 - C(n - c, k) / C(n, k)``."""
    if k < 1:
        raise DataError("k must be at least 1", details={"k": k})
    if n < 1 or c < 0 or c > n:
        raise DataError("need 0 <= c <= n with n >= 1", details={"n": n, "c": c})
    if n < k:
        raise DataError("k cannot exceed the number of samples", details={"n": n, "k": k})
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(counts: Sequence[Count], k: int) -> float:
    """Average :func:`pass_at_k` over the tasks in ``counts``."""
    if not counts:
        raise DataError("no tasks to score")
    return sum(pass_at_k(n, c, k) for n, c in counts) / len(counts)


def bootstrap_pass_at_k(
    counts: Sequence[Count],
    k: int,
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PassAtK:
    """Mean pass@k over tasks with a seeded percentile bootstrap interval.

    Each resample draws ``len(counts)`` tasks with replacement and averages
    their pass@k; ``low`` and ``high`` are the ``(1 - confidence) / 2`` and
    ``(1 + confidence) / 2`` quantiles of those means (nearest rank). The
    same ``seed`` always gives the same interval.
    """
    if resamples < 1:
        raise DataError("resamples must be at least 1", details={"resamples": resamples})
    if not 0.0 < confidence < 1.0:
        raise DataError("confidence must be in (0, 1)", details={"confidence": confidence})
    estimate = mean_pass_at_k(counts, k)
    scores = [pass_at_k(n, c, k) for n, c in counts]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0
    low = means[math.floor(alpha * (resamples - 1))]
    high = means[math.ceil((1.0 - alpha) * (resamples - 1))]
    return PassAtK(
        k=k,
        estimate=estimate,
        low=low,
        high=high,
        tasks=len(counts),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


__all__ = ["Count", "PassAtK", "bootstrap_pass_at_k", "mean_pass_at_k", "pass_at_k"]
