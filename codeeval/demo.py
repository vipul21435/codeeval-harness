"""End-to-end demo: validate a task suite, then score completions on it with pass@k.

:func:`run_demo` is what ``make demo`` and ``python -m codeeval demo`` run.
It needs no network and no model: for every task of a suite ported from
HumanEval it validates the pytest grader with :mod:`codeeval.f2p` (the
baseline stub must fail, the reference must pass), then hands two completions
per task to the upstream pass@k evaluator, the canonical solution and the
same ``NotImplementedError`` stub, so the expected result is known up front:
every canonical completion passes, every stub fails, pass@1 is 0.5 and
pass@2 is 1.0. Anything else is a bug in the harness and the demo exits with
status 1.

Samples, the problem subset and the evaluator's results file are written
under ``<results_dir>/demo`` (``VERIFYBENCH_RESULTS_DIR``, ``results`` by
default), next to a ``summary.json`` with everything the table shows.
"""

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from codeeval.convert import STUB_STATEMENT, body_indent
from codeeval.errors import DataError
from codeeval.f2p import DEFAULT_TIMEOUT, SuiteVerdict, Verdict, validate_suite
from codeeval.settings import get_settings
from codeeval.tasks import Task, TaskSuite, read_tasks
from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "tasks" / "humaneval_mini.jsonl"
DEFAULT_REPEATS = 3
DEMO_KS = (1, 2)

Row = tuple[str, ...]


class TaskSummary(BaseModel):
    """One row of the demo table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    entry_point: str
    difficulty: str
    verdict: Verdict
    samples: int = Field(ge=0)
    passed: int = Field(ge=0)
    canonical: str = Field(description="Evaluator result of the canonical completion.")
    stub: str = Field(description="Evaluator result of the stub completion.")

    @property
    def pass_at_1(self) -> float:
        return self.passed / self.samples if self.samples else 0.0

    @property
    def as_expected(self) -> bool:
        return self.verdict == "ok" and self.canonical == "passed" and self.stub != "passed"


class DemoReport(BaseModel):
    """Everything the demo measured; ``ok`` when every task behaved as expected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks_file: str
    n_tasks: int
    repeats: int
    workers: int
    sample_timeout: float
    grader_timeout: float
    verdict_counts: dict[Verdict, int]
    validation_seconds: float
    grader_runs: int
    n_samples: int
    pass_at_k: dict[str, float]
    evaluation_seconds: float
    rows: list[TaskSummary]
    results_file: str

    @property
    def ok(self) -> bool:
        return all(row.as_expected for row in self.rows)


def upstream_problem(task: Task, problems: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """The HumanEval problem a task was ported from; ``DataError`` when there is none."""
    upstream_id = task.metadata.get("upstream_task_id", task.task_id)
    if not isinstance(upstream_id, str) or upstream_id not in problems:
        raise DataError(
            f"task {task.task_id} was not ported from a packaged HumanEval problem",
            details={"task_id": task.task_id, "upstream_task_id": upstream_id},
        )
    return problems[upstream_id]


def demo_samples(
    suite: TaskSuite, problems: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The problem subset and the samples (canonical, then stub, per task) the demo scores."""
    subset: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for task in suite.tasks:
        problem = dict(upstream_problem(task, problems))
        canonical = problem["canonical_solution"]
        subset.append(problem)
        samples.append({"task_id": problem["task_id"], "completion": canonical})
        samples.append(
            {
                "task_id": problem["task_id"],
                "completion": f"{body_indent(canonical)}{STUB_STATEMENT}\n",
            }
        )
    return subset, samples


def summarise(
    suite: TaskSuite, verdicts: SuiteVerdict, results: Sequence[Mapping[str, Any]]
) -> list[TaskSummary]:
    """Join the validator's verdicts with the evaluator's results, in suite order."""
    by_task: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[str(result["task_id"])].append(result)
    verdict_of = {verdict.task_id: verdict.verdict for verdict in verdicts.verdicts}
    rows: list[TaskSummary] = []
    for task in suite.tasks:
        upstream_id = str(task.metadata.get("upstream_task_id", task.task_id))
        outcomes = [str(result["result"]) for result in by_task[upstream_id]]
        rows.append(
            TaskSummary(
                task_id=task.task_id,
                entry_point=task.entry_point,
                difficulty=task.difficulty,
                verdict=verdict_of[task.task_id],
                samples=len(outcomes),
                passed=sum(1 for result in by_task[upstream_id] if result["passed"]),
                canonical=outcomes[0] if outcomes else "missing",
                stub=outcomes[1] if len(outcomes) > 1 else "missing",
            )
        )
    return rows


def render_table(headers: Sequence[str], rows: Sequence[Row]) -> str:
    """A fixed-width ASCII table; numeric-looking cells are right-aligned."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        out: list[str] = []
        for cell, width in zip(cells, widths, strict=True):
            out.append(cell.rjust(width) if _numeric(cell) else cell.ljust(width))
        return "  ".join(out).rstrip()

    lines = [fmt(headers), "  ".join("-" * width for width in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _numeric(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return True


def table_rows(rows: Sequence[TaskSummary]) -> list[Row]:
    return [
        (
            row.task_id,
            row.entry_point,
            row.difficulty,
            row.verdict,
            str(row.samples),
            str(row.passed),
            f"{row.pass_at_1:.2f}",
            short_result(row.canonical),
            short_result(row.stub),
        )
        for row in rows
    ]


def short_result(result: str, width: int = 24) -> str:
    """An evaluator result trimmed for the table: no empty reason, at most ``width`` chars."""
    result = result.rstrip(": ")
    return result if len(result) <= width else result[: width - 3] + "..."


TABLE_HEADERS = (
    "task_id",
    "entry_point",
    "difficulty",
    "f2p",
    "samples",
    "passed",
    "pass@1",
    "canonical",
    "stub",
)


def render_report(report: DemoReport) -> str:
    """The demo's stdout: two phase summaries, the per-task table and the totals."""
    counts = " ".join(f"{name}={count}" for name, count in report.verdict_counts.items())
    pass_at_k = " ".join(f"{name}={value:.3f}" for name, value in report.pass_at_k.items())
    lines = [
        f"VerifyBench demo: {report.tasks_file} ({report.n_tasks} tasks)",
        "",
        (
            f"[1/2] fail-to-pass validation: {report.n_tasks} tasks x 2 solutions x "
            f"{report.repeats} {'repeat' if report.repeats == 1 else 'repeats'} = "
            f"{report.grader_runs} grader runs, "
            f"{report.workers} workers, {report.grader_timeout:g}s timeout each"
        ),
        f"      {counts} in {report.validation_seconds:.1f}s",
        "",
        (
            f"[2/2] pass@k evaluation: {report.n_samples} completions "
            "(canonical solution + NotImplementedError stub per task), "
            f"{report.workers} workers, {report.sample_timeout:g}s timeout each"
        ),
        f"      {pass_at_k} in {report.evaluation_seconds:.1f}s",
        "",
        render_table(TABLE_HEADERS, table_rows(report.rows)),
        "",
        f"results: {report.results_file}",
        "verdict: " + ("OK, every task behaved as expected" if report.ok else "MISMATCH"),
    ]
    if not report.ok:
        for row in report.rows:
            if not row.as_expected:
                lines.append(
                    f"  {row.task_id}: f2p={row.verdict} canonical={row.canonical} stub={row.stub}"
                )
    return "\n".join(lines)


def display_path(path: os.PathLike[str] | str) -> str:
    """``path`` relative to the working directory when it lies inside it, else as given."""
    try:
        return os.fspath(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return os.fspath(path)


def run_demo(
    tasks_file: os.PathLike[str] | str = DEFAULT_TASKS,
    *,
    limit: int | None = None,
    repeats: int = DEFAULT_REPEATS,
    workers: int | None = None,
    sample_timeout: float | None = None,
    grader_timeout: float = DEFAULT_TIMEOUT,
    results_dir: os.PathLike[str] | str | None = None,
) -> DemoReport:
    """Validate ``tasks_file`` (at most ``limit`` tasks), score it, and report.

    ``workers``, ``sample_timeout`` and ``results_dir`` default to the
    settings. Raises :class:`~codeeval.errors.DataError` when the suite
    cannot be read or a task has no packaged HumanEval problem behind it.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    settings = get_settings()
    workers = settings.workers if workers is None else workers
    sample_timeout = settings.sample_timeout if sample_timeout is None else sample_timeout
    out_dir = Path(results_dir if results_dir is not None else settings.results_dir) / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = read_tasks(tasks_file)
    if limit is not None:
        suite = TaskSuite(tasks=suite.tasks[:limit])
    if not suite.tasks:
        raise DataError(f"{os.fspath(tasks_file)} holds no tasks", details={"path": tasks_file})
    subset, samples = demo_samples(suite, read_problems())

    started = time.monotonic()
    verdicts = validate_suite(suite, repeats=repeats, timeout=grader_timeout, workers=workers)
    validation_seconds = time.monotonic() - started

    problem_file = out_dir / "problems.jsonl"
    sample_file = out_dir / "samples.jsonl"
    write_jsonl(problem_file, subset)
    write_jsonl(sample_file, samples)
    started = time.monotonic()
    pass_at_k = evaluate_functional_correctness(
        sample_file, DEMO_KS, workers, sample_timeout, problem_file
    )
    evaluation_seconds = time.monotonic() - started
    results_file = f"{os.fspath(sample_file)}_results.jsonl"
    rows = summarise(suite, verdicts, list(stream_jsonl(results_file)))

    report = DemoReport(
        tasks_file=display_path(tasks_file),
        n_tasks=len(suite),
        repeats=repeats,
        workers=workers,
        sample_timeout=sample_timeout,
        grader_timeout=grader_timeout,
        verdict_counts=verdicts.counts,
        validation_seconds=validation_seconds,
        grader_runs=sum(len(verdict.runs) for verdict in verdicts.verdicts),
        n_samples=len(samples),
        pass_at_k=pass_at_k,
        evaluation_seconds=evaluation_seconds,
        rows=rows,
        results_file=results_file,
    )
    (out_dir / "summary.json").write_text(
        json.dumps({**report.model_dump(mode="json"), "ok": report.ok}, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info(
        "demo finished",
        extra={
            "n_tasks": report.n_tasks,
            "ok": report.ok,
            "pass_at_k": pass_at_k,
            "validation_seconds": round(validation_seconds, 3),
            "evaluation_seconds": round(evaluation_seconds, 3),
        },
    )
    return report


__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_TASKS",
    "DEMO_KS",
    "TABLE_HEADERS",
    "DemoReport",
    "TaskSummary",
    "demo_samples",
    "display_path",
    "render_report",
    "render_table",
    "run_demo",
    "short_result",
    "summarise",
    "table_rows",
    "upstream_problem",
]
