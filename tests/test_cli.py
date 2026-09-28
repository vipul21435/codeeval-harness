"""Tests for the evaluate_functional_correctness console script.

The subprocess tests are the regression tests for the macOS spawn bug: with the
spawn start method every worker re-imports the parent's __main__ module, so the
CLI must not run at import time.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from human_eval.evaluate_functional_correctness import parse_k

EXPECTED = {"pass@1": 0.5, "pass@2": 0.8, "pass@4": 1.0}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,10,100", [1, 10, 100]),
        ("1", [1]),
        ("1, 2 ,4", [1, 2, 4]),
        ("1,,2", [1, 2]),
        (5, [5]),
        ((1, 2, 4), [1, 2, 4]),
        ([3], [3]),
    ],
)
def test_parse_k(raw: str | int | tuple[int, ...] | list[int], expected: list[int]) -> None:
    assert parse_k(raw) == expected


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _printed_pass_at_k(stdout: str) -> dict[str, float]:
    last_line = stdout.strip().splitlines()[-1]
    parsed = ast.literal_eval(last_line)
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.slow
def test_module_entry_point_runs_end_to_end(
    example_problem_file: Path, example_samples_file: Path
) -> None:
    proc = _run(
        [
            sys.executable,
            "-m",
            "human_eval.evaluate_functional_correctness",
            example_samples_file.name,
            f"--problem_file={example_problem_file.name}",
            "--k=1,2,4",
            "--timeout=1",
            "--n_workers=2",
        ],
        cwd=example_samples_file.parent,
    )

    assert proc.returncode == 0, proc.stderr
    assert _printed_pass_at_k(proc.stdout) == pytest.approx(EXPECTED)
    assert (example_samples_file.parent / "example_samples.jsonl_results.jsonl").is_file()


@pytest.mark.slow
def test_console_script_runs_end_to_end(
    example_problem_file: Path, example_samples_file: Path
) -> None:
    script = Path(sys.executable).with_name("evaluate_functional_correctness")
    if not script.exists():
        found = shutil.which("evaluate_functional_correctness")
        if found is None:
            pytest.skip("console script not installed in this environment")
        script = Path(found)

    proc = _run(
        [
            str(script),
            str(example_samples_file),
            "--problem_file",
            str(example_problem_file),
            "--k",
            "1,2,4",
            "--timeout",
            "1",
        ],
        cwd=example_samples_file.parent,
    )

    assert proc.returncode == 0, proc.stderr
    assert _printed_pass_at_k(proc.stdout) == pytest.approx(EXPECTED)


@pytest.mark.slow
def test_single_k_flag(example_problem_file: Path, example_samples_file: Path) -> None:
    proc = _run(
        [
            sys.executable,
            "-m",
            "human_eval.evaluate_functional_correctness",
            str(example_samples_file),
            f"--problem_file={example_problem_file}",
            "--k=1",
            "--timeout=1",
        ],
        cwd=example_samples_file.parent,
    )
    assert proc.returncode == 0, proc.stderr
    assert _printed_pass_at_k(proc.stdout) == pytest.approx({"pass@1": 0.5})


def test_help_flag_does_not_run_an_evaluation() -> None:
    proc = _run(
        [sys.executable, "-m", "human_eval.evaluate_functional_correctness", "--help"],
        cwd=Path.cwd(),
    )
    assert proc.returncode == 0, proc.stderr
    assert "SAMPLE_FILE" in proc.stdout + proc.stderr
