"""Fail-to-pass validation of task graders.

A task's tests are worth trusting only once they have been shown to reject a
stub and accept a correct solution. :func:`validate_task` runs a task's pytest
module against its ``baseline_solution`` (which must fail) and its
``reference_solution`` (which must pass). Every run happens in a fresh
temporary directory and a fresh interpreter under a timeout, and is repeated
``repeats`` times (default 3): a solution whose outcome differs between
identical runs marks the grader as flaky.

Per-task verdicts, from worst to best; the first that applies wins:

- ``error``: a run could not be graded (pytest usage or internal error,
  nothing collected, the interpreter did not start);
- ``flaky``: the same solution passed on one attempt and failed on another;
- ``baseline_passes``: the stub satisfies the tests, so they test nothing;
- ``reference_fails``: the reference solution does not satisfy the tests;
- ``ok``: the baseline failed and the reference passed on every attempt.

A timed-out run counts as failed (``GraderRun.timed_out`` records it), so a
hanging reference is a ``reference_fails`` and a hanging stub still fails.
``PYTHONHASHSEED`` is left unset on purpose: outcomes that depend on hash
randomisation are exactly the flakiness the repeats exist to surface.

This validates suites you author; it is not a sandbox. Solutions and tests
run with the privileges of the current user.
"""

import contextlib
import functools
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codeeval.errors import GradingError
from codeeval.log import bind_context
from codeeval.settings import get_settings
from codeeval.tasks import SOLUTION_MODULE, Task, TaskSuite

log = logging.getLogger(__name__)

DEFAULT_REPEATS = 3
# Seconds one pytest run may take, interpreter start-up included.
DEFAULT_TIMEOUT = 30.0
# Bytes of pytest output kept per run (the tail).
OUTPUT_TAIL = 4096
TEST_FILE = "test_task.py"

Solution = Literal["baseline", "reference"]
Outcome = Literal["passed", "failed", "error"]
Verdict = Literal["ok", "baseline_passes", "reference_fails", "flaky", "error"]

SOLUTIONS: tuple[Solution, ...] = ("baseline", "reference")
VERDICTS: tuple[Verdict, ...] = ("ok", "baseline_passes", "reference_fails", "flaky", "error")

# pytest exit statuses that mean the grader itself did not work.
_GRADER_PROBLEMS = {
    3: "pytest internal error",
    4: "pytest usage error",
    5: "no tests collected",
}


def outcome_of_exit(returncode: int) -> Outcome:
    """The outcome a pytest exit status stands for.

    0 is a pass; 1 (tests failed), 2 (interrupted, which covers collection
    errors) and a death by signal are failures of the solution; 3 (internal
    error), 4 (usage error), 5 (nothing collected) and anything unknown mean
    the grader could not do its job.
    """
    if returncode == 0:
        return "passed"
    if returncode in (1, 2) or returncode < 0:
        return "failed"
    return "error"


class GraderRun(BaseModel):
    """One pytest run of a task's tests against one of its solutions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    solution: Solution
    attempt: int = Field(ge=1)
    outcome: Outcome
    returncode: int | None = Field(
        description="pytest exit status; None when the run timed out or never started."
    )
    timed_out: bool = False
    duration: float = Field(ge=0, description="Wall-clock seconds, start-up included.")
    output: str = Field(default="", description="Tail of pytest's stdout and stderr.")

    def describe_problem(self) -> str:
        """Why an ``error`` run could not be graded."""
        if self.returncode is None:
            return self.output.strip() or "the interpreter did not start"
        meaning = _GRADER_PROBLEMS.get(self.returncode, "unexpected exit status")
        return f"pytest exited with status {self.returncode} ({meaning})"


class TaskVerdict(BaseModel):
    """The fail-to-pass verdict for one task, with every run behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    verdict: Verdict
    runs: list[GraderRun]
    error: str | None = Field(default=None, description="What went wrong on an error verdict.")

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"

    def outcomes(self, solution: Solution) -> list[Outcome]:
        """The outcomes of ``solution``'s attempts, in order."""
        return [run.outcome for run in self.runs if run.solution == solution]

    @classmethod
    def from_runs(cls, task_id: str, runs: Sequence[GraderRun]) -> "TaskVerdict":
        verdict, error = decide(runs)
        return cls(task_id=task_id, verdict=verdict, runs=list(runs), error=error)


def decide(runs: Sequence[GraderRun]) -> tuple[Verdict, str | None]:
    """The verdict for a task's runs, with a message on ``error``.

    See the module docstring for the precedence. ``runs`` must contain at
    least one run of each solution.
    """
    outcomes: dict[Solution, set[Outcome]] = {solution: set() for solution in SOLUTIONS}
    for run in runs:
        outcomes[run.solution].add(run.outcome)
    if not all(outcomes.values()):
        raise ValueError("runs must cover both the baseline and the reference solution")
    for run in runs:
        if run.outcome == "error":
            return "error", f"{run.solution} attempt {run.attempt}: {run.describe_problem()}"
    if any(len(seen) > 1 for seen in outcomes.values()):
        return "flaky", None
    if outcomes["baseline"] == {"passed"}:
        return "baseline_passes", None
    if outcomes["reference"] == {"failed"}:
        return "reference_fails", None
    return "ok", None


class SuiteVerdict(BaseModel):
    """The verdicts of a whole suite, in task order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdicts: list[TaskVerdict]

    @property
    def counts(self) -> dict[Verdict, int]:
        """How many tasks got each verdict; every verdict is a key."""
        counts = dict.fromkeys(VERDICTS, 0)
        for verdict in self.verdicts:
            counts[verdict.verdict] += 1
        return counts

    @property
    def ok(self) -> bool:
        return all(verdict.ok for verdict in self.verdicts)

    @property
    def failures(self) -> list[TaskVerdict]:
        return [verdict for verdict in self.verdicts if not verdict.ok]


def grader_environment(root: Path) -> dict[str, str]:
    """The environment of a grader run.

    Only ``PATH`` is inherited. ``HOME`` and ``TMPDIR`` point into the run
    directory so user configuration stays out and temporary files vanish
    with it, and pytest's plugin autoloading is off so the run does not depend
    on what happens to be installed next to pytest. ``PYTHONHASHSEED`` is
    deliberately not set.
    """
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(root),
        "TMPDIR": str(root),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }


def run_grader(
    task: Task,
    solution: Solution,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    attempt: int = 1,
    python: str = sys.executable,
) -> GraderRun:
    """Run ``task.tests`` against one solution in a fresh directory and interpreter.

    The solution is written as ``solution.py`` next to the tests and a
    ``pytest.ini`` pins pytest's rootdir to the directory, so no configuration
    from the working tree leaks in. The process runs in its own session and
    the whole session is killed on timeout, children included.
    """
    source = task.reference_solution if solution == "reference" else task.baseline_solution
    with tempfile.TemporaryDirectory(prefix="verifybench-f2p-") as tmp:
        root = Path(tmp)
        (root / f"{SOLUTION_MODULE}.py").write_text(source, encoding="utf-8")
        (root / TEST_FILE).write_text(task.tests, encoding="utf-8")
        (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        command = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", TEST_FILE]
        started = time.monotonic()
        returncode, timed_out, output = _run(command, root, timeout)
        duration = time.monotonic() - started
    if timed_out:
        outcome: Outcome = "failed"
    elif returncode is None:
        outcome = "error"
    else:
        outcome = outcome_of_exit(returncode)
    return GraderRun(
        solution=solution,
        attempt=attempt,
        outcome=outcome,
        returncode=returncode,
        timed_out=timed_out,
        duration=duration,
        output=output,
    )


def _run(command: list[str], root: Path, timeout: float) -> tuple[int | None, bool, str]:
    """Run ``command`` in ``root``; returns (exit status, timed out, output tail)."""
    log_path = root / "pytest.log"
    with open(log_path, "wb") as sink:
        try:
            proc = subprocess.Popen(
                command,
                cwd=root,
                env=grader_environment(root),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return None, False, f"could not start {command[0]}: {exc}"
        try:
            returncode: int | None = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_session(proc)
            returncode, timed_out = None, True
    return returncode, timed_out, _tail(log_path)


def _kill_session(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _tail(path: Path) -> str:
    with open(path, "rb") as fp:
        fp.seek(0, os.SEEK_END)
        fp.seek(max(0, fp.tell() - OUTPUT_TAIL))
        return fp.read().decode("utf-8", errors="replace")


@functools.cache
def _pytest_problem(python: str) -> str | None:
    """None when ``python`` can import pytest, else what went wrong; cached per interpreter."""
    try:
        proc = subprocess.run(
            [python, "-c", "import pytest"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return None if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1]


def require_pytest(python: str = sys.executable) -> None:
    """Raise :class:`~codeeval.errors.GradingError` unless ``python`` can import pytest."""
    problem = _pytest_problem(python)
    if problem is not None:
        raise GradingError(
            f"pytest is not importable by {python}: {problem}", details={"python": python}
        )


def validate_task(
    task: Task,
    *,
    repeats: int = DEFAULT_REPEATS,
    timeout: float = DEFAULT_TIMEOUT,
    python: str = sys.executable,
) -> TaskVerdict:
    """Run the baseline and the reference ``repeats`` times each and judge the grader.

    Raises :class:`~codeeval.errors.GradingError` when ``python`` has no
    pytest; every other problem ends up in the verdict.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    require_pytest(python)
    started = time.monotonic()
    runs: list[GraderRun] = []
    with bind_context(task_id=task.task_id):
        for solution in SOLUTIONS:
            for attempt in range(1, repeats + 1):
                runs.append(
                    run_grader(task, solution, timeout=timeout, attempt=attempt, python=python)
                )
        verdict = TaskVerdict.from_runs(task.task_id, runs)
        log.log(
            logging.INFO if verdict.ok else logging.WARNING,
            "validated task",
            extra={
                "task_id": task.task_id,
                "verdict": verdict.verdict,
                "error": verdict.error,
                "repeats": repeats,
                "runs": len(runs),
                "duration": round(time.monotonic() - started, 3),
            },
        )
    return verdict


def validate_suite(
    suite: TaskSuite | Iterable[Task],
    *,
    repeats: int = DEFAULT_REPEATS,
    timeout: float = DEFAULT_TIMEOUT,
    workers: int | None = None,
    python: str = sys.executable,
) -> SuiteVerdict:
    """Validate every task, ``workers`` tasks at a time (default: the ``workers`` setting).

    A task's own runs stay sequential; verdicts come back in task order.
    """
    tasks = list(suite.tasks if isinstance(suite, TaskSuite) else suite)
    if workers is None:
        workers = get_settings().workers
    if workers < 1:
        raise ValueError("workers must be at least 1")
    require_pytest(python)
    started = time.monotonic()

    def one(task: Task) -> TaskVerdict:
        return validate_task(task, repeats=repeats, timeout=timeout, python=python)

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
        result = SuiteVerdict(verdicts=list(pool.map(one, tasks)))
    log.info(
        "validated suite",
        extra={
            "n_tasks": len(tasks),
            "ok": result.ok,
            "counts": result.counts,
            "repeats": repeats,
            "duration": round(time.monotonic() - started, 3),
        },
    )
    return result


__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_TIMEOUT",
    "OUTPUT_TAIL",
    "SOLUTIONS",
    "VERDICTS",
    "GraderRun",
    "Outcome",
    "Solution",
    "SuiteVerdict",
    "TaskVerdict",
    "Verdict",
    "decide",
    "grader_environment",
    "outcome_of_exit",
    "require_pytest",
    "run_grader",
    "validate_suite",
    "validate_task",
]
