"""Hand-checked numbers for codeeval.metrics."""

from math import comb

import pytest

from codeeval.errors import DataError
from codeeval.metrics import PassAtK, bootstrap_pass_at_k, mean_pass_at_k, pass_at_k


def test_readme_example_numbers() -> None:
    """Six samples, three correct: 0.5, 0.8 and 1.0, as in the README."""
    assert pass_at_k(6, 3, 1) == pytest.approx(0.5)
    assert pass_at_k(6, 3, 2) == pytest.approx(0.8)
    assert pass_at_k(6, 3, 4) == pytest.approx(1.0)


def test_hand_computed_five_samples_two_correct() -> None:
    """n=5, c=2, k=2: 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7."""
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_at_k(5, 2, 2) == pytest.approx(1 - comb(3, 2) / comb(5, 2))


def test_extremes() -> None:
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(1, 1, 1) == 1.0


@pytest.mark.parametrize(
    ("n", "c", "k", "message"),
    [
        (6, 3, 0, "k must be at least 1"),
        (0, 0, 1, "need 0 <= c <= n"),
        (3, 4, 1, "need 0 <= c <= n"),
        (3, -1, 1, "need 0 <= c <= n"),
        (3, 1, 4, "k cannot exceed"),
    ],
)
def test_rejects_bad_counts(n: int, c: int, k: int, message: str) -> None:
    with pytest.raises(DataError, match=message):
        pass_at_k(n, c, k)


def test_mean_over_tasks() -> None:
    """(0.5 + 0.0 + 1.0) / 3 = 0.5 for k=1."""
    assert mean_pass_at_k([(6, 3), (4, 0), (2, 2)], 1) == pytest.approx(0.5)


def test_mean_rejects_empty_suite() -> None:
    with pytest.raises(DataError, match="no tasks"):
        mean_pass_at_k([], 1)


def test_identical_tasks_give_a_degenerate_interval() -> None:
    result = bootstrap_pass_at_k([(6, 3)] * 5, 2, resamples=50, seed=7)
    assert result == PassAtK(
        k=2, estimate=0.8, low=0.8, high=0.8, tasks=5, resamples=50, confidence=0.95, seed=7
    )


def test_two_extreme_tasks_span_zero_to_one() -> None:
    """Resampling {0, 1} twice gives means 0, 0.5, 1; the 95% interval is [0, 1]."""
    result = bootstrap_pass_at_k([(4, 0), (4, 4)], 1, resamples=1000, seed=0)
    assert result.estimate == pytest.approx(0.5)
    assert (result.low, result.high) == (0.0, 1.0)


def test_interval_contains_the_estimate_and_is_seed_deterministic() -> None:
    counts = [(4, c) for c in (0, 1, 2, 3, 4, 4, 1, 0)]
    first = bootstrap_pass_at_k(counts, 2, seed=123)
    again = bootstrap_pass_at_k(counts, 2, seed=123)
    other = bootstrap_pass_at_k(counts, 2, seed=124)
    assert first == again
    assert first.low <= first.estimate <= first.high
    assert 0.0 <= first.low < first.high <= 1.0
    assert (other.low, other.high) != (first.low, first.high) or other.estimate == first.estimate


def test_single_resample_collapses_to_that_resample() -> None:
    result = bootstrap_pass_at_k([(4, 0), (4, 4)], 1, resamples=1, seed=0)
    assert result.low == result.high
    assert result.low in (0.0, 0.5, 1.0)


def test_narrower_confidence_never_widens() -> None:
    counts = [(4, c) for c in (0, 1, 2, 3, 4, 4, 1, 0)]
    wide = bootstrap_pass_at_k(counts, 1, confidence=0.99, seed=1)
    narrow = bootstrap_pass_at_k(counts, 1, confidence=0.5, seed=1)
    assert wide.low <= narrow.low <= narrow.high <= wide.high


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resamples": 0}, "resamples must be at least 1"),
        ({"confidence": 0.0}, "confidence must be in"),
        ({"confidence": 1.0}, "confidence must be in"),
    ],
)
def test_bootstrap_rejects_bad_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(DataError, match=message):
        bootstrap_pass_at_k([(4, 2)], 1, **kwargs)  # type: ignore[arg-type]
