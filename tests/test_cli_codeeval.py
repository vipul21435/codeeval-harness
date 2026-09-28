"""Tests for the verifybench command line (python -m codeeval)."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from codeeval import __version__
from codeeval.cli import EXIT_ERROR, EXIT_MISMATCH, EXIT_OK, build_parser, main
from codeeval.tasks import Task, read_tasks, write_tasks


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "demo" in out
    assert "validate" in out
    assert "convert" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_demo_defaults() -> None:
    args = build_parser().parse_args(["demo"])
    assert args.tasks is None
    assert args.repeats == 3
    assert args.workers is None


def test_module_help_runs_without_side_effects() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "codeeval", "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: verifybench" in proc.stdout


@pytest.mark.slow
def test_validate_reports_each_task(
    tmp_path: Path, make_task: Callable[..., Task], capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tasks.jsonl"
    write_tasks(path, [make_task(), make_task(task_id="synthetic/other")])
    assert main(["validate", str(path), "--repeats", "1", "--workers", "2"]) == EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ["synthetic/add\tok", "synthetic/other\tok"]
    assert out[2].startswith("2 tasks: ok=2 ")


@pytest.mark.slow
def test_validate_fails_on_a_bad_grader(
    tmp_path: Path, make_task: Callable[..., Task], capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tasks.jsonl"
    write_tasks(path, [make_task(baseline_solution="def add(a, b):\n    return a + b\n")])
    assert main(["validate", str(path), "--repeats", "1"]) == EXIT_MISMATCH
    assert "synthetic/add\tbaseline_passes" in capsys.readouterr().out


def test_data_errors_become_exit_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate", str(tmp_path / "missing.jsonl")]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("verifybench: DataError: cannot read task file")


def test_convert_passes_arguments_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "one.jsonl"
    assert main(["convert", str(out), "--limit", "1"]) == EXIT_OK
    assert "wrote 1 tasks" in capsys.readouterr().out
    assert read_tasks(out).ids == ["HumanEval/0"]


@pytest.mark.slow
def test_demo_command_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = main(
        [
            "demo",
            "--limit",
            "1",
            "--repeats",
            "1",
            "--workers",
            "2",
            "--results-dir",
            str(tmp_path),
        ]
    )
    assert status == EXIT_OK
    out = capsys.readouterr().out
    assert "verdict: OK" in out
    assert "HumanEval/0" in out
    assert (tmp_path / "demo" / "summary.json").is_file()
