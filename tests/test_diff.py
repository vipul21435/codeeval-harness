"""Hand-checked regression diffs for codeeval.diff."""

from fractions import Fraction

from codeeval.diff import RunDiff, TaskChange, diff_results


def records(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"task_id": task_id, "result": result} for task_id, result in pairs]


BEFORE = records(
    ("T/0", "passed"),
    ("T/0", "passed"),
    ("T/1", "passed"),
    ("T/1", "failed: "),
    ("T/2", "failed: "),
    ("T/3", "passed"),
)
AFTER = records(
    ("T/0", "passed"),
    ("T/0", "failed: "),
    ("T/1", "passed"),
    ("T/1", "timed out"),
    ("T/2", "passed"),
    ("T/4", "failed: "),
)


def test_diff_groups_every_task() -> None:
    """T/0 lost a pass, T/1 kept one, T/2 gained one, T/3 vanished, T/4 is new."""
    diff = diff_results(BEFORE, AFTER)
    assert diff.regressed == [TaskChange("T/0", (2, 2), (2, 1))]
    assert diff.unchanged == [TaskChange("T/1", (2, 1), (2, 1))]
    assert diff.improved == [TaskChange("T/2", (1, 0), (1, 1))]
    assert diff.added == ["T/4"]
    assert diff.removed == ["T/3"]
    assert diff.regressed[0].delta == Fraction(-1, 2)
    assert diff.improved[0].delta == 1
    assert not diff.ok


def test_diff_compares_rates_not_counts() -> None:
    """5/10 and 1/2 are the same rate; 4/10 is lower and 3/4 is higher."""
    before = records(*[("T/0", "passed")] * 5, *[("T/0", "failed: x")] * 5)
    same = records(("T/0", "passed"), ("T/0", "failed: x"))
    assert diff_results(before, same) == RunDiff(unchanged=[TaskChange("T/0", (10, 5), (2, 1))])
    assert diff_results(before, same).ok
    worse = records(*[("T/0", "passed")] * 4, *[("T/0", "failed: x")] * 6)
    assert diff_results(before, worse).regressed == [TaskChange("T/0", (10, 5), (10, 4))]
    assert diff_results(before, worse).regressed[0].delta == Fraction(-1, 10)
    better = records(*[("T/0", "passed")] * 3, ("T/0", "timed out"))
    change = diff_results(before, better).improved[0]
    assert change == TaskChange("T/0", (10, 5), (4, 3))
    assert (change.before_rate, change.after_rate, change.delta) == (
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(1, 4),
    )


def test_identical_runs_are_ok() -> None:
    diff = diff_results(BEFORE, BEFORE)
    assert diff == RunDiff(
        unchanged=[
            TaskChange("T/0", (2, 2), (2, 2)),
            TaskChange("T/1", (2, 1), (2, 1)),
            TaskChange("T/2", (1, 0), (1, 0)),
            TaskChange("T/3", (1, 1), (1, 1)),
        ]
    )
    assert diff.ok


def test_added_tasks_do_not_break_ok_but_removed_ones_do() -> None:
    assert diff_results([], AFTER).ok
    assert diff_results(BEFORE, []) == RunDiff(removed=["T/0", "T/1", "T/2", "T/3"])
    assert not diff_results(BEFORE, []).ok
