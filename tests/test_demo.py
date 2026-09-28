"""Tests for codeeval.demo: the offline validate-then-score demo."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from codeeval.backends import MockBackend, generate_samples
from codeeval.demo import (
    DEFAULT_TASKS,
    DEMO_N,
    DemoReport,
    TaskSummary,
    demo_canned,
    display_path,
    render_report,
    render_table,
    run_demo,
    short_result,
    summarise,
    table_rows,
    upstream_problem,
)
from codeeval.errors import DataError
from codeeval.f2p import VERDICTS, GraderRun, SuiteVerdict, TaskVerdict
from codeeval.settings import override_settings
from codeeval.tasks import Task, TaskSuite, read_tasks
from human_eval.data import read_problems


def test_render_table_aligns_columns() -> None:
    text = render_table(("name", "n"), [("a", "1"), ("longer", "10")])
    assert text.splitlines() == [
        "name    n",
        "------  --",
        "a        1",
        "longer  10",
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("passed", "passed"),
        ("failed: ", "failed"),
        ("failed: AssertionError", "failed: AssertionError"),
        ("failed: " + "x" * 40, "failed: " + "x" * 13 + "..."),
    ],
)
def test_short_result(raw: str, expected: str) -> None:
    assert short_result(raw) == expected


def test_display_path_is_relative_inside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert display_path(tmp_path / "a" / "b.jsonl") == "a/b.jsonl"
    assert display_path("/definitely/elsewhere.jsonl") == "/definitely/elsewhere.jsonl"


def test_upstream_problem_requires_a_packaged_problem(make_task: Callable[..., Task]) -> None:
    problems = {"HumanEval/0": {"task_id": "HumanEval/0"}}
    ported = make_task(task_id="mine/x", metadata={"upstream_task_id": "HumanEval/0"})
    assert upstream_problem(ported, problems)["task_id"] == "HumanEval/0"
    with pytest.raises(DataError, match="synthetic/add"):
        upstream_problem(make_task(), problems)


def test_demo_canned_pairs_canonical_with_stub() -> None:
    suite = TaskSuite(tasks=read_tasks(DEFAULT_TASKS).tasks[:2])
    subset, canned = demo_canned(suite, read_problems())
    assert [problem["task_id"] for problem in subset] == ["HumanEval/0", "HumanEval/1"]
    assert [task_id for task_id, _ in canned] == ["HumanEval/0"] * 2 + ["HumanEval/1"] * 2
    assert canned[0][1] == subset[0]["canonical_solution"]
    assert canned[1][1] == "    raise NotImplementedError\n"


def test_demo_mock_generates_canonical_then_stub() -> None:
    suite = TaskSuite(tasks=read_tasks(DEFAULT_TASKS).tasks[:1])
    subset, canned = demo_canned(suite, read_problems())
    backend = MockBackend.from_pairs(canned)
    samples = generate_samples(backend, [(subset[0]["task_id"], subset[0]["prompt"])], n=DEMO_N)
    assert [sample.completion for sample in samples] == [
        subset[0]["canonical_solution"],
        "    raise NotImplementedError\n",
    ]


def _verdicts(*ids: str) -> SuiteVerdict:
    run_ok = GraderRun(solution="baseline", attempt=1, outcome="failed", returncode=1, duration=0)
    run_ref = GraderRun(solution="reference", attempt=1, outcome="passed", returncode=0, duration=0)
    return SuiteVerdict(
        verdicts=[TaskVerdict.from_runs(task_id, [run_ok, run_ref]) for task_id in ids]
    )


def test_summarise_joins_verdicts_and_results(make_task: Callable[..., Task]) -> None:
    task = make_task(task_id="mine/x", metadata={"upstream_task_id": "HumanEval/0"})
    results: list[dict[str, Any]] = [
        {"task_id": "HumanEval/0", "passed": True, "result": "passed"},
        {"task_id": "HumanEval/0", "passed": False, "result": "failed: "},
    ]
    [row] = summarise(TaskSuite(tasks=[task]), _verdicts("mine/x"), results)
    assert row.samples == 2
    assert row.passed == 1
    assert row.pass_at_1 == 0.5
    assert row.as_expected
    [empty] = summarise(TaskSuite(tasks=[task]), _verdicts("mine/x"), [])
    assert (empty.samples, empty.canonical, empty.stub) == (0, "missing", "missing")
    assert empty.pass_at_1 == 0
    assert not empty.as_expected


def _report(rows: list[TaskSummary]) -> DemoReport:
    return DemoReport(
        tasks_file="tasks.jsonl",
        n_tasks=len(rows),
        backend="mock",
        generation_seconds=0.01,
        samples_file="results/demo/samples.jsonl",
        repeats=1,
        workers=2,
        sample_timeout=3.0,
        grader_timeout=30.0,
        verdict_counts=dict.fromkeys(VERDICTS, 0) | {"ok": 1},
        validation_seconds=1.25,
        grader_runs=2 * len(rows),
        n_samples=2 * len(rows),
        pass_at_k={"pass@1": 0.5, "pass@2": 1.0},
        evaluation_seconds=0.5,
        rows=rows,
        results_file="results/demo/samples.jsonl_results.jsonl",
    )


def _row(**overrides: Any) -> TaskSummary:
    fields: dict[str, Any] = {
        "task_id": "HumanEval/0",
        "entry_point": "f",
        "difficulty": "easy",
        "verdict": "ok",
        "samples": 2,
        "passed": 1,
        "canonical": "passed",
        "stub": "failed: ",
    }
    return TaskSummary(**{**fields, **overrides})


def test_render_report_ok() -> None:
    text = render_report(_report([_row()]))
    assert "[1/3] generation: 2 completions from the mock backend" in text
    assert "results/demo/samples.jsonl in 0.0s" in text
    assert "[2/3] fail-to-pass validation: 1 tasks x 2 solutions x 1 repeat = 2 grader runs" in text
    assert "[3/3] pass@k evaluation: 2 completions" in text
    assert "pass@1=0.500 pass@2=1.000" in text
    assert "verdict: OK" in text
    assert table_rows([_row()])[0][-1] == "failed"


def test_render_report_lists_mismatches() -> None:
    bad = _row(task_id="HumanEval/1", verdict="baseline_passes", stub="passed", passed=2)
    text = render_report(_report([_row(), bad]))
    assert "verdict: MISMATCH" in text
    assert "HumanEval/1: f2p=baseline_passes canonical=passed stub=passed" in text
    assert "HumanEval/0:" not in text


def test_run_demo_rejects_bad_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        run_demo(limit=0)


def test_run_demo_needs_tasks(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(DataError, match="no tasks"):
        run_demo(empty, results_dir=tmp_path)


@pytest.mark.slow
def test_run_demo_end_to_end(tmp_path: Path) -> None:
    with override_settings(workers=2):
        report = run_demo(DEFAULT_TASKS, limit=2, repeats=1, results_dir=tmp_path)
    assert report.ok
    assert report.n_tasks == 2
    assert report.grader_runs == 4
    assert report.n_samples == 4
    assert report.backend == "mock"
    assert report.generation_seconds >= 0
    assert report.pass_at_k == pytest.approx({"pass@1": 0.5, "pass@2": 1.0})
    samples = [
        json.loads(line) for line in (tmp_path / "demo" / "samples.jsonl").read_text().splitlines()
    ]
    assert [(sample["task_id"], sample["backend"], sample["index"]) for sample in samples] == [
        ("HumanEval/0", "mock", 0),
        ("HumanEval/0", "mock", 1),
        ("HumanEval/1", "mock", 0),
        ("HumanEval/1", "mock", 1),
    ]
    assert report.verdict_counts["ok"] == 2
    assert [row.task_id for row in report.rows] == ["HumanEval/0", "HumanEval/1"]
    assert Path(report.results_file).is_file()
    summary = json.loads((tmp_path / "demo" / "summary.json").read_text())
    assert summary["ok"] is True
    assert summary["pass_at_k"]["pass@2"] == 1.0
