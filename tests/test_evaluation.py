"""End-to-end tests for evaluate_functional_correctness on the bundled datasets."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

pytestmark = pytest.mark.slow


def test_readme_example_reproduces_documented_pass_at_k(
    example_problem_file: Path, example_samples_file: Path
) -> None:
    """The six example samples give pass@1 = 0.5, pass@2 = 0.8 and pass@4 = 1.0."""
    pass_at_k = evaluate_functional_correctness(
        example_samples_file,
        k=[1, 2, 4],
        n_workers=2,
        timeout=1.0,
        problem_file=example_problem_file,
    )

    assert pass_at_k == pytest.approx({"pass@1": 0.5, "pass@2": 0.8, "pass@4": 1.0})
    assert all(type(value) is float for value in pass_at_k.values())


def test_results_file_is_written_next_to_samples(
    example_problem_file: Path, example_samples_file: Path
) -> None:
    evaluate_functional_correctness(
        example_samples_file, k=[1], n_workers=2, timeout=1.0, problem_file=example_problem_file
    )

    results_file = example_samples_file.with_name(example_samples_file.name + "_results.jsonl")
    rows = list(stream_jsonl(results_file))
    samples = list(stream_jsonl(example_samples_file))

    assert len(rows) == len(samples) == 6
    # Original sample fields are preserved and results are attached in input order.
    assert [row["completion"] for row in rows] == [s["completion"] for s in samples]
    assert [row["passed"] for row in rows] == [False, False, False, True, True, True]
    assert rows[1]["result"] == "timed out"
    assert rows[3]["result"] == "passed"
    assert all(row["result"].startswith("failed") for row in (rows[0], rows[2]))


def test_k_larger_than_sample_count_is_skipped(
    example_problem_file: Path, example_samples_file: Path
) -> None:
    pass_at_k = evaluate_functional_correctness(
        example_samples_file,
        k=[1, 6, 7, 100],
        n_workers=2,
        timeout=1.0,
        problem_file=example_problem_file,
    )
    assert set(pass_at_k) == {"pass@1", "pass@6"}


def test_unattempted_problem_raises(tmp_path: Path, example_problem_file: Path) -> None:
    problems = list(stream_jsonl(example_problem_file))
    extra = {**problems[0], "task_id": "test/1"}
    problem_file = tmp_path / "two_problems.jsonl"
    write_jsonl(problem_file, [*problems, extra])
    sample_file = tmp_path / "samples.jsonl"
    write_jsonl(sample_file, [{"task_id": "test/0", "completion": "    return 1"}])

    with pytest.raises(ValueError, match=r"not attempted: \['test/1'\]"):
        evaluate_functional_correctness(sample_file, k=[1], problem_file=problem_file)


def test_canonical_solutions_pass_on_real_humaneval_tasks(tmp_path: Path) -> None:
    """Sanity-check the execution path against real HumanEval tasks and tests."""
    problems = read_problems()
    subset = ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    problem_file = tmp_path / "subset.jsonl"
    write_jsonl(problem_file, [problems[task_id] for task_id in subset])
    sample_file = tmp_path / "samples.jsonl"
    write_jsonl(
        sample_file,
        [
            {"task_id": task_id, "completion": problems[task_id]["canonical_solution"]}
            for task_id in subset
        ],
    )

    pass_at_k = evaluate_functional_correctness(
        sample_file, k=[1], n_workers=3, timeout=3.0, problem_file=problem_file
    )
    assert pass_at_k == {"pass@1": 1.0}


def test_all_canonical_solutions_pass_with_more_workers_than_cpus(tmp_path: Path) -> None:
    """Oversubscribing the CPUs slows interpreter start-up, which must not be
    charged to the sample: every canonical solution has to come back passed."""
    problems = read_problems()
    sample_file = tmp_path / "canonical.jsonl"
    write_jsonl(
        sample_file,
        [
            {"task_id": task_id, "completion": problem["canonical_solution"]}
            for task_id, problem in problems.items()
        ],
    )
    n_workers = max(16, 2 * (os.cpu_count() or 1))

    pass_at_k = evaluate_functional_correctness(sample_file, k=[1], n_workers=n_workers)

    assert pass_at_k == {"pass@1": 1.0}
    results_file = sample_file.with_name(sample_file.name + "_results.jsonl")
    not_passed = {row["task_id"]: row["result"] for row in stream_jsonl(results_file)}
    assert {k: v for k, v in not_passed.items() if v != "passed"} == {}


def test_unguarded_script_can_evaluate(
    tmp_path: Path, example_problem_file: Path, example_samples_file: Path
) -> None:
    """A script that evaluates at module level, with no __main__ guard, works:
    workers are plain subprocesses and never re-import the caller's script."""
    script = tmp_path / "unguarded.py"
    script.write_text(
        "from human_eval.evaluation import evaluate_functional_correctness\n"
        "print(evaluate_functional_correctness("
        f"{str(example_samples_file)!r}, k=[1], timeout=1.0, "
        f"problem_file={str(example_problem_file)!r}))\n"
    )

    proc = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False
    )

    assert proc.returncode == 0, proc.stderr
    printed = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
    assert printed == pytest.approx({"pass@1": 0.5})
