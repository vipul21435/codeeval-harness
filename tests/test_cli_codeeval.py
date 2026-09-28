"""Tests for the verifybench command line (python -m codeeval)."""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeServer

from codeeval import __version__
from codeeval.backends import STUB_COMPLETION
from codeeval.cli import EXIT_ERROR, EXIT_MISMATCH, EXIT_OK, build_parser, main, positive_int
from codeeval.tasks import Task, read_tasks, write_tasks


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "demo" in out
    assert "validate" in out
    assert "convert" in out
    assert "generate" in out


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


# --- generate ------------------------------------------------------------------


def _samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generate_mock_replays_reference_bodies_by_default(
    tmp_path: Path, make_task: Callable[..., Task], capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task(), make_task(task_id="synthetic/other")])
    out = tmp_path / "out" / "samples.jsonl"

    assert main(["generate", "--tasks", str(tasks), "--n", "2", "--out", str(out)]) == EXIT_OK

    records = _samples(out)
    assert [record["task_id"] for record in records] == ["synthetic/add"] * 2 + [
        "synthetic/other"
    ] * 2
    # The synthetic reference does not start with its prompt, so it is replayed whole.
    reference = "def add(a: int, b: int) -> int:\n    return a + b\n"
    assert all(record["completion"] == reference for record in records)
    assert all(record["backend"] == "mock" for record in records)
    assert capsys.readouterr().out.strip().startswith("4 samples from mock for 2 tasks -> ")


def test_generate_mock_from_canned_file_with_failure_rate(
    tmp_path: Path, make_task: Callable[..., Task]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task()])
    canned = tmp_path / "canned.jsonl"
    canned.write_text(
        json.dumps({"task_id": "synthetic/add", "completion": "    return 42\n"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "samples.jsonl"
    argv = ["generate", "--backend", "mock", "--tasks", str(tasks), "--out", str(out)]

    assert main([*argv, "--canned", str(canned), "--n", "3", "--failure-rate", "1"]) == EXIT_OK
    assert [record["completion"] for record in _samples(out)] == [STUB_COMPLETION] * 3

    assert main([*argv, "--canned", str(canned), "--n", "1", "--failure-rate", "0"]) == EXIT_OK
    assert [record["completion"] for record in _samples(out)] == ["    return 42\n"]


def test_generate_honours_limit_and_settings_backend(
    tmp_path: Path, make_task: Callable[..., Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task(), make_task(task_id="synthetic/other")])
    out = tmp_path / "samples.jsonl"
    monkeypatch.setenv("VERIFYBENCH_MODEL_BACKEND", "mock")
    monkeypatch.setenv("VERIFYBENCH_MOCK_FAILURE_RATE", "1")
    assert main(["generate", "--tasks", str(tasks), "--out", str(out), "--limit", "1"]) == EXIT_OK
    records = _samples(out)
    assert [record["task_id"] for record in records] == ["synthetic/add"]
    assert records[0]["completion"] == STUB_COMPLETION


def test_generate_openai_against_local_server(
    tmp_path: Path,
    make_task: Callable[..., Task],
    monkeypatch: pytest.MonkeyPatch,
    fake_server: FakeServer,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task()])
    out = tmp_path / "samples.jsonl"
    monkeypatch.setenv("VERIFYBENCH_OPENAI_BASE_URL", fake_server.base_url)
    monkeypatch.setenv("VERIFYBENCH_OPENAI_API_KEY", "sk-local")
    monkeypatch.setenv("VERIFYBENCH_OPENAI_MODEL", "local")
    fake_server.script((200, {"choices": [{"message": {"content": "    return a + b\n"}}]}))

    argv = ["generate", "--backend", "openai", "--tasks", str(tasks), "--out", str(out)]
    assert main(argv) == EXIT_OK

    assert _samples(out) == [
        {
            "task_id": "synthetic/add",
            "completion": "    return a + b\n",
            "backend": "openai",
            "index": 0,
        }
    ]
    assert fake_server.requests[0]["model"] == "local"
    assert fake_server.headers[0]["Authorization"] == "Bearer sk-local"


def test_generate_openai_without_key_is_a_config_error(
    tmp_path: Path, make_task: Callable[..., Task], capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task()])
    argv = ["generate", "--backend", "openai", "--tasks", str(tasks), "--out", str(tmp_path / "s")]
    assert main(argv) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "ConfigError" in err
    assert "VERIFYBENCH_OPENAI_API_KEY" in err
    assert not (tmp_path / "s").exists()


def test_generate_unknown_canned_task_is_a_provider_error(
    tmp_path: Path, make_task: Callable[..., Task], capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    write_tasks(tasks, [make_task()])
    canned = tmp_path / "canned.jsonl"
    canned.write_text(json.dumps({"task_id": "x/y", "completion": "z"}) + "\n", encoding="utf-8")
    argv = [
        "generate",
        "--tasks",
        str(tasks),
        "--out",
        str(tmp_path / "s"),
        "--canned",
        str(canned),
    ]
    assert main(argv) == EXIT_ERROR
    assert "ProviderError" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3"])
def test_generate_rejects_non_positive_n(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(["generate", "--tasks", "t", "--out", "o", "--n", value])
    assert info.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_positive_int_parses_valid_values() -> None:
    assert positive_int("7") == 7
