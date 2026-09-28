"""Tests for codeeval.convert: porting HumanEval problems to pytest-graded tasks."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from codeeval.convert import (
    body_indent,
    difficulty_for,
    humaneval_to_task,
    humaneval_to_tasks,
    join_module,
    main,
)
from codeeval.errors import DataError
from codeeval.f2p import validate_suite, validate_task
from codeeval.settings import override_settings
from codeeval.tasks import read_tasks
from human_eval.data import read_problems

SHIPPED_SUITE = Path(__file__).resolve().parent.parent / "data" / "tasks" / "humaneval_mini.jsonl"


def test_converts_the_example_problem(example_problem: dict[str, Any]) -> None:
    task = humaneval_to_task(example_problem)
    assert task.task_id == "test/0"
    assert task.prompt == "def return1():\n"
    assert task.entry_point == "return1"
    assert task.reference_solution == "def return1():\n    return 1\n"
    assert task.baseline_solution == "def return1():\n    raise NotImplementedError\n"
    assert task.tests == (
        "# Ported from HumanEval task test/0; check() is the upstream grader, unchanged.\n"
        "from solution import *  # noqa: F403 - check() may use prompt helpers\n"
        "from solution import return1\n"
        "\n"
        "def check(candidate):\n"
        "    assert candidate() == 1\n"
        "\n\n"
        "def test_return1() -> None:\n"
        "    check(return1)\n"
    )
    assert (task.language, task.difficulty) == ("python", "easy")
    assert task.metadata == {"source": "openai/human-eval", "upstream_task_id": "test/0"}


def test_stub_follows_the_canonical_indentation(example_problem: dict[str, Any]) -> None:
    tabbed = {**example_problem, "canonical_solution": "\n\treturn 1\n"}
    assert (
        humaneval_to_task(tabbed).baseline_solution
        == "def return1():\n\traise NotImplementedError\n"
    )
    assert body_indent("") == "    "
    assert body_indent("  x = 1\n    return x") == "  "


def test_join_module_normalises_newlines() -> None:
    assert join_module("def f():", "    return 1") == "def f():\n    return 1\n"
    assert join_module("def f():\n", "    return 1\n") == "def f():\n    return 1\n"


@pytest.mark.parametrize(
    ("lines", "difficulty"),
    [(0, "easy"), (1, "easy"), (4, "easy"), (5, "medium"), (12, "medium"), (13, "hard")],
)
def test_difficulty_is_a_size_proxy(lines: int, difficulty: str) -> None:
    solution = "".join(f"    x{i} = {i}\n\n" for i in range(lines))
    assert difficulty_for(solution) == difficulty


def test_missing_keys_are_a_data_error(example_problem: dict[str, Any]) -> None:
    del example_problem["canonical_solution"]
    del example_problem["test"]
    with pytest.raises(DataError, match="HumanEval problem test/0 lacks canonical_solution, test"):
        humaneval_to_task(example_problem)
    with pytest.raises(DataError, match="<unknown> lacks task_id"):
        humaneval_to_task({})


def test_test_without_check_is_a_data_error(example_problem: dict[str, Any]) -> None:
    example_problem["test"] = "assert True\n"
    with pytest.raises(DataError, match="test/0 has no check\\(candidate\\)") as info:
        humaneval_to_task(example_problem)
    assert info.value.details == {"task_id": "test/0"}


def test_unconvertible_problem_names_the_field(example_problem: dict[str, Any]) -> None:
    example_problem["canonical_solution"] = "    return (\n"
    with pytest.raises(DataError, match="test/0 does not convert to a valid task") as info:
        humaneval_to_task(example_problem)
    assert [error["field"] for error in info.value.details["errors"]] == ["reference_solution"]


def test_humaneval_to_tasks_limits_in_dataset_order() -> None:
    suite = humaneval_to_tasks(limit=3)
    assert suite.ids == ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    assert len(humaneval_to_tasks(limit=0)) == 0
    with pytest.raises(ValueError, match="limit must not be negative"):
        humaneval_to_tasks(limit=-1)


def test_humaneval_to_tasks_rejects_duplicate_ids(example_problem: dict[str, Any]) -> None:
    with pytest.raises(DataError, match="duplicate task_id 'test/0'"):
        humaneval_to_tasks([example_problem, example_problem])


def test_converted_tasks_round_trip_through_jsonl(tmp_path: Path) -> None:
    suite = humaneval_to_tasks(limit=3)
    assert main([str(tmp_path / "three.jsonl"), "--limit", "3"]) == 0
    assert read_tasks(tmp_path / "three.jsonl") == suite


def test_main_reads_another_problem_file(
    tmp_path: Path, example_problem_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "tasks.jsonl"
    assert main([str(out), "--problem-file", str(example_problem_file), "--limit", "0"]) == 0
    assert read_tasks(out).ids == ["test/0"]
    assert capsys.readouterr().out == f"wrote 1 tasks to {out}\n"


def test_shipped_suite_is_current_and_deterministic(tmp_path: Path) -> None:
    suite = read_tasks(SHIPPED_SUITE)
    assert len(suite) == 20
    assert suite.ids == [f"HumanEval/{i}" for i in range(20)]
    assert {task.language for task in suite.tasks} == {"python"}
    problems = read_problems()
    for task in suite.tasks:
        assert task.entry_point == problems[task.task_id]["entry_point"]
        assert task.reference_solution.startswith(task.prompt)
        assert "raise NotImplementedError" in task.baseline_solution
    main([str(tmp_path / "regenerated.jsonl")])
    assert (tmp_path / "regenerated.jsonl").read_bytes() == SHIPPED_SUITE.read_bytes()


@pytest.mark.slow
def test_converted_example_problem_is_fail_to_pass(example_problem: dict[str, Any]) -> None:
    verdict = validate_task(humaneval_to_task(example_problem), repeats=1)
    assert verdict.verdict == "ok", verdict


@pytest.mark.slow
def test_shipped_suite_validates_clean() -> None:
    with override_settings(workers=4):
        report = validate_suite(read_tasks(SHIPPED_SUITE))
    assert report.counts == {
        "ok": 20,
        "baseline_passes": 0,
        "reference_fails": 0,
        "flaky": 0,
        "error": 0,
    }, [(v.task_id, v.verdict, v.error) for v in report.failures]
    assert report.ok


@pytest.mark.slow
def test_module_entry_point(tmp_path: Path) -> None:
    out = tmp_path / "two.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "codeeval.convert", str(out), "--limit", "2"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"wrote 2 tasks to {out}\n"
    assert read_tasks(out).ids == ["HumanEval/0", "HumanEval/1"]
