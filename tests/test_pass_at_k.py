"""Unit tests for the unbiased pass@k estimator from the Codex paper."""

from math import comb

import numpy as np
import pytest

from human_eval.evaluation import estimate_pass_at_k


def exact_pass_at_k(n: int, c: int, k: int) -> float:
    """Closed-form reference: 1 - C(n - c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def test_readme_example_numbers() -> None:
    """Six samples with three correct: the numbers quoted in the README."""
    assert estimate_pass_at_k(6, [3], 1) == pytest.approx([0.5])
    assert estimate_pass_at_k(6, [3], 2) == pytest.approx([0.8])
    assert estimate_pass_at_k(6, [3], 4) == pytest.approx([1.0])


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(n, c, k) for n in (1, 2, 5, 10, 20) for c in range(n + 1) for k in (1, 2, 5, 10)],
)
def test_matches_closed_form(n: int, c: int, k: int) -> None:
    (estimate,) = estimate_pass_at_k(n, [c], k)
    assert estimate == pytest.approx(exact_pass_at_k(n, c, k))


def test_no_correct_samples_is_zero() -> None:
    assert estimate_pass_at_k(10, [0, 0], 1).tolist() == [0.0, 0.0]


def test_all_correct_samples_is_one() -> None:
    assert estimate_pass_at_k(10, [10], 5).tolist() == [1.0]


def test_k_larger_than_failures_is_one() -> None:
    # With n - c < k every draw of k samples must contain a correct one.
    assert estimate_pass_at_k(5, [3], 3).tolist() == [1.0]
    assert estimate_pass_at_k(5, [3], 2) == pytest.approx([1 - comb(2, 2) / comb(5, 2)])


def test_monotone_in_k() -> None:
    values = [estimate_pass_at_k(100, [10], k)[0] for k in range(1, 101)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == 1.0


def test_broadcasts_scalar_num_samples() -> None:
    scalar = estimate_pass_at_k(8, [1, 4, 8], 2)
    per_problem = estimate_pass_at_k([8, 8, 8], [1, 4, 8], 2)
    np.testing.assert_allclose(scalar, per_problem)
    assert scalar.dtype == np.float64
    assert scalar.shape == (3,)


def test_accepts_numpy_arrays() -> None:
    total = np.array([4, 6])
    correct = np.array([2, 3])
    np.testing.assert_allclose(estimate_pass_at_k(total, correct, 1), [0.5, 0.5])


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_pass_at_k([4, 6], [1], 1)


def test_empty_input_gives_empty_array() -> None:
    assert estimate_pass_at_k(5, [], 1).shape == (0,)


def test_values_are_probabilities() -> None:
    grid = [(n, c) for n in range(1, 8) for c in range(n + 1)]
    for k in range(1, 8):
        values = estimate_pass_at_k([n for n, _ in grid], [c for _, c in grid], k)
        assert ((values >= 0.0) & (values <= 1.0)).all()
