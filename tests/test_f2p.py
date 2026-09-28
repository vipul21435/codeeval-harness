"""Tests for codeeval.f2p: the fail-to-pass validator."""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from codeeval.errors import GradingError
from codeeval.f2p import (
    VERDICTS,
    GraderRun,
    Outcome,
    Solution,
    SuiteVerdict,
    TaskVerdict,
    decide,
    grader_environment,
    outcome_of_exit,
    require_pytest,
    run_grader,
    validate_suite,
    validate_task,
)
from codeeval.settings import override_settings
from codeeval.tasks import Task, TaskSuite

# --- verdict logic, no subprocesses --------------------------------------------


def run(
    solution: Solution, attempt: int, outcome: Outcome, returncode: int | None = 1
) -> GraderRun:
    if outcome == "passed":
        returncode = 0
    return GraderRun(
        solution=solution, attempt=attempt, outcome=outcome, returncode=returncode, duration=0.1
    )


def runs(baseline: str, reference: str) -> list[GraderRun]:
    """Runs from outcome initials, e.g. ``runs("ffp", "ppp")``."""
    names: dict[str, Outcome] = {"p": "passed", "f": "failed", "e": "error"}
    return [
        *(run("baseline", i, names[c], 5 if c == "e" else 1) for i, c in enumerate(baseline, 1)),
        *(run("reference", i, names[c], 5 if c == "e" else 1) for i, c in enumerate(reference, 1)),
    ]


@pytest.mark.parametrize(
    ("returncode", "outcome"),
    [
        (0, "passed"),
        (1, "failed"),
        (2, "failed"),
        (-9, "failed"),
        (-11, "failed"),
        (3, "error"),
        (4, "error"),
        (5, "error"),
        (6, "error"),
        (127, "error"),
    ],
)
def test_outcome_of_exit(returncode: int, outcome: Outcome) -> None:
    assert outcome_of_exit(returncode) == outcome


@pytest.mark.parametrize(
    ("baseline", "reference", "verdict"),
    [
        ("fff", "ppp", "ok"),
        ("f", "p", "ok"),
        ("ppp", "ppp", "baseline_passes"),
        ("fff", "fff", "reference_fails"),
        ("ppp", "fff", "baseline_passes"),  # both wrong: the baseline is reported first
        ("fpf", "ppp", "flaky"),
        ("fff", "pfp", "flaky"),
        ("ppf", "fff", "flaky"),  # flaky beats baseline_passes and reference_fails
        ("fef", "ppp", "error"),  # error beats flaky
        ("fff", "eee", "error"),
    ],
)
def test_decide_precedence(baseline: str, reference: str, verdict: str) -> None:
    assert decide(runs(baseline, reference))[0] == verdict


def test_decide_names_the_first_error_run() -> None:
    verdict, error = decide(runs("ff", "fe"))
    assert (verdict, error) == (
        "error",
        "reference attempt 2: pytest exited with status 5 (no tests collected)",
    )
    _, error = decide([*runs("ff", "pp"), run("baseline", 3, "error", None)])
    assert error == "baseline attempt 3: the interpreter did not start"


def test_decide_needs_both_solutions() -> None:
    with pytest.raises(ValueError, match="both the baseline and the reference"):
        decide(runs("fff", ""))
    with pytest.raises(ValueError, match="both the baseline and the reference"):
        decide([])


def test_task_verdict_from_runs() -> None:
    verdict = TaskVerdict.from_runs("synthetic/add", runs("fff", "ppp"))
    assert verdict.ok
    assert verdict.error is None
    assert verdict.outcomes("baseline") == ["failed"] * 3
    assert verdict.outcomes("reference") == ["passed"] * 3
    assert not TaskVerdict.from_runs("synthetic/add", runs("ppp", "ppp")).ok


def test_grader_run_describes_problems() -> None:
    assert run("reference", 1, "error", 4).describe_problem() == (
        "pytest exited with status 4 (pytest usage error)"
    )
    assert run("reference", 1, "error", 42).describe_problem() == (
        "pytest exited with status 42 (unexpected exit status)"
    )
    started = GraderRun(
        solution="baseline",
        attempt=1,
        outcome="error",
        returncode=None,
        duration=0.0,
        output="could not start /nope: boom\n",
    )
    assert started.describe_problem() == "could not start /nope: boom"


def test_suite_verdict_counts_and_failures() -> None:
    ok = TaskVerdict.from_runs("s/ok", runs("f", "p"))
    bad = TaskVerdict.from_runs("s/bad", runs("p", "p"))
    flaky = TaskVerdict.from_runs("s/flaky", runs("fp", "pp"))
    suite = SuiteVerdict(verdicts=[ok, bad, flaky])
    assert suite.counts == {
        "ok": 1,
        "baseline_passes": 1,
        "reference_fails": 0,
        "flaky": 1,
        "error": 0,
    }
    assert list(suite.counts) == list(VERDICTS)
    assert not suite.ok
    assert [v.task_id for v in suite.failures] == ["s/bad", "s/flaky"]
    assert SuiteVerdict(verdicts=[ok]).ok
    assert SuiteVerdict(verdicts=[]).ok


def test_validate_task_rejects_bad_parameters(make_task: Callable[..., Task]) -> None:
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        validate_task(make_task(), repeats=0)
    with pytest.raises(ValueError, match="timeout must be positive"):
        validate_task(make_task(), timeout=0)
    with pytest.raises(ValueError, match="workers must be at least 1"):
        validate_suite([make_task()], workers=0)


def test_grader_environment_is_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "9")
    env = grader_environment(tmp_path)
    assert env["HOME"] == env["TMPDIR"] == str(tmp_path)
    assert env["PATH"] == os.environ["PATH"]
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTHONHASHSEED" not in env
    assert not any(key.startswith("VERIFYBENCH_") for key in env)


# --- real pytest runs in subprocesses ---------------------------------------------

pytestmark_slow = pytest.mark.slow


@pytestmark_slow
def test_good_task_is_ok(make_task: Callable[..., Task]) -> None:
    verdict = validate_task(make_task(), repeats=2)
    assert verdict.ok
    assert verdict.verdict == "ok"
    assert [(r.solution, r.attempt) for r in verdict.runs] == [
        ("baseline", 1),
        ("baseline", 2),
        ("reference", 1),
        ("reference", 2),
    ]
    baseline, reference = verdict.runs[0], verdict.runs[-1]
    assert (baseline.returncode, baseline.outcome, baseline.timed_out) == (1, "failed", False)
    assert (reference.returncode, reference.outcome, reference.timed_out) == (0, "passed", False)
    assert "NotImplementedError" in baseline.output
    assert "1 passed" in reference.output
    assert all(r.duration > 0 for r in verdict.runs)


@pytestmark_slow
def test_baseline_that_passes_is_caught(make_task: Callable[..., Task]) -> None:
    task = make_task(baseline_solution="def add(a, b):\n    return b + a\n")
    verdict = validate_task(task, repeats=1)
    assert verdict.verdict == "baseline_passes"
    assert verdict.outcomes("baseline") == ["passed"]


@pytestmark_slow
def test_reference_that_fails_is_caught(make_task: Callable[..., Task]) -> None:
    task = make_task(reference_solution="def add(a, b):\n    return a - b\n")
    verdict = validate_task(task, repeats=1)
    assert verdict.verdict == "reference_fails"
    assert "assert" in verdict.runs[-1].output


@pytestmark_slow
def test_flaky_grader_is_caught(tmp_path: Path, make_task: Callable[..., Task]) -> None:
    counter = tmp_path / "counter"
    tests = (
        "from pathlib import Path\n"
        "from solution import add\n\n\n"
        "def test_add() -> None:\n"
        f"    counter = Path({str(counter)!r})\n"
        "    n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "    counter.write_text(str(n))\n"
        "    assert n % 2 == 0\n"
    )
    verdict = validate_task(make_task(tests=tests), repeats=3)
    assert verdict.verdict == "flaky"
    assert verdict.outcomes("baseline") == ["failed", "passed", "failed"]
    assert counter.read_text() == "6"


@pytestmark_slow
def test_hanging_reference_times_out_and_leaves_no_children(
    tmp_path: Path, make_task: Callable[..., Task]
) -> None:
    pid_file = tmp_path / "child.pid"
    reference = (
        "import pathlib, subprocess, time\n"
        'child = subprocess.Popen(["sleep", "60"])\n'
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n\n\n"
        "def add(a, b):\n    return a + b\n"
    )
    started = time.monotonic()
    verdict = validate_task(make_task(reference_solution=reference), repeats=1, timeout=2.0)
    assert time.monotonic() - started < 20
    assert verdict.verdict == "reference_fails"
    reference_run = verdict.runs[-1]
    assert (reference_run.timed_out, reference_run.returncode, reference_run.outcome) == (
        True,
        None,
        "failed",
    )
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"child {child_pid} of the timed-out run is still alive")


@pytestmark_slow
def test_nothing_collected_is_an_error(make_task: Callable[..., Task]) -> None:
    # The schema sees a test function; pytest never collects it.
    tests = (
        "from solution import add\n\n"
        "if False:\n\n    def test_add():\n        assert add(1, 1) == 2\n"
    )
    verdict = validate_task(make_task(tests=tests), repeats=1)
    assert verdict.verdict == "error"
    assert verdict.error == "baseline attempt 1: pytest exited with status 5 (no tests collected)"
    assert verdict.runs[0].returncode == 5


@pytestmark_slow
def test_interpreter_that_does_not_start(make_task: Callable[..., Task]) -> None:
    grader_run = run_grader(make_task(), "reference", python="/nonexistent/python")
    assert grader_run.outcome == "error"
    assert grader_run.returncode is None
    assert grader_run.output.startswith("could not start /nonexistent/python:")
    with pytest.raises(
        GradingError, match="pytest is not importable by /nonexistent/python"
    ) as info:
        validate_task(make_task(), python="/nonexistent/python")
    assert info.value.details == {"python": "/nonexistent/python"}
    require_pytest()  # the current interpreter runs this suite, so it has pytest


@pytestmark_slow
def test_validate_suite_keeps_task_order(
    make_task: Callable[..., Task], caplog: pytest.LogCaptureFixture
) -> None:
    good = make_task(task_id="synthetic/good")
    bad = make_task(task_id="synthetic/bad", reference_solution="def add(a, b):\n    return 0\n")
    suite = TaskSuite(tasks=[bad, good, make_task(task_id="synthetic/also-good")])
    with caplog.at_level(logging.INFO, logger="codeeval.f2p"), override_settings(workers=2):
        result = validate_suite(suite, repeats=1)
    assert [v.task_id for v in result.verdicts] == suite.ids
    assert [v.verdict for v in result.verdicts] == ["reference_fails", "ok", "ok"]
    assert result.counts["ok"] == 2
    assert not result.ok

    # Extras are plain attributes on the records; vars() keeps mypy happy.
    task_records = [vars(r) for r in caplog.records if r.getMessage() == "validated task"]
    assert sorted(r["task_id"] for r in task_records) == sorted(suite.ids)
    bad_record = next(r for r in task_records if r["task_id"] == "synthetic/bad")
    assert (bad_record["levelno"], bad_record["verdict"]) == (logging.WARNING, "reference_fails")
    suite_record = next(vars(r) for r in caplog.records if r.getMessage() == "validated suite")
    assert (suite_record["n_tasks"], suite_record["ok"]) == (3, False)
    assert suite_record["counts"]["reference_fails"] == 1
