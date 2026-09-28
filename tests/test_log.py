"""Tests for codeeval.log: JSON and text formatting, configuration, bound context."""

import ast
import io
import json
import logging
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from codeeval.log import (
    HANDLER_NAME,
    JsonFormatter,
    TextFormatter,
    bind_context,
    configure_logging,
    current_context,
)
from codeeval.settings import override_settings
from human_eval import evaluation

TIMESTAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


def _logger_writing_to(stream: io.StringIO, formatter: logging.Formatter) -> logging.Logger:
    logger = logging.getLogger("codeeval.test")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


# --- JSON formatter ---------------------------------------------------------


def test_json_record_has_timestamp_level_logger_and_message(stream: io.StringIO) -> None:
    logger = _logger_writing_to(stream, JsonFormatter())
    logger.warning("disk %s%% full", 93)

    (record,) = _records(stream)
    assert list(record) == ["timestamp", "level", "logger", "message"]
    assert re.fullmatch(TIMESTAMP, record["timestamp"])
    assert record["level"] == "WARNING"
    assert record["logger"] == "codeeval.test"
    assert record["message"] == "disk 93% full"


def test_json_record_carries_extras_sorted_after_the_core_fields(stream: io.StringIO) -> None:
    logger = _logger_writing_to(stream, JsonFormatter())
    logger.info("graded", extra={"task_id": "HumanEval/0", "run_id": "r1", "passed": True})

    (record,) = _records(stream)
    assert list(record) == [
        "timestamp",
        "level",
        "logger",
        "message",
        "passed",
        "run_id",
        "task_id",
    ]
    assert record["run_id"] == "r1"
    assert record["task_id"] == "HumanEval/0"
    assert record["passed"] is True


def test_json_record_stringifies_values_json_cannot_encode(
    stream: io.StringIO, tmp_path: Path
) -> None:
    logger = _logger_writing_to(stream, JsonFormatter())
    logger.info("wrote", extra={"path": tmp_path / "out.jsonl"})
    (record,) = _records(stream)
    assert record["path"] == str(tmp_path / "out.jsonl")


def test_json_record_is_one_line_and_includes_the_traceback(stream: io.StringIO) -> None:
    logger = _logger_writing_to(stream, JsonFormatter())
    try:
        raise ValueError("boom\nsecond line")
    except ValueError:
        logger.exception("grading failed", extra={"task_id": "t/0"})

    assert len(stream.getvalue().splitlines()) == 1
    (record,) = _records(stream)
    assert record["level"] == "ERROR"
    assert record["task_id"] == "t/0"
    assert record["exception"].startswith("Traceback (most recent call last):")
    assert record["exception"].rstrip().endswith("ValueError: boom\nsecond line")


def test_json_record_includes_stack_info_on_request(stream: io.StringIO) -> None:
    logger = _logger_writing_to(stream, JsonFormatter())
    logger.info("where am I", stack_info=True)
    (record,) = _records(stream)
    assert record["stack"].startswith("Stack (most recent call last):")


# --- text formatter ---------------------------------------------------------


def test_text_record_layout(stream: io.StringIO) -> None:
    logger = _logger_writing_to(stream, TextFormatter())
    logger.info("graded", extra={"task_id": "HumanEval/0", "run_id": "r1"})

    (line,) = stream.getvalue().splitlines()
    assert re.fullmatch(
        TIMESTAMP + r" INFO     codeeval\.test: graded run_id=r1 task_id=HumanEval/0", line
    )


def test_text_record_appends_the_traceback(stream: io.StringIO) -> None:
    logger = _logger_writing_to(stream, TextFormatter())
    try:
        raise KeyError("missing")
    except KeyError:
        logger.exception("lookup failed")

    lines = stream.getvalue().splitlines()
    assert re.fullmatch(TIMESTAMP + r" ERROR    codeeval\.test: lookup failed", lines[0])
    assert lines[1] == "Traceback (most recent call last):"
    assert lines[-1] == "KeyError: 'missing'"


# --- configuration from settings -------------------------------------------


def test_configure_logging_uses_the_json_format_and_level_from_settings(
    stream: io.StringIO,
) -> None:
    with override_settings(log_format="json", log_level="DEBUG"):
        configure_logging(stream=stream)
    logging.getLogger("codeeval.anything").debug("hello", extra={"seed": 7})

    (record,) = _records(stream)
    assert record["level"] == "DEBUG"
    assert record["logger"] == "codeeval.anything"
    assert record["seed"] == 7


def test_configure_logging_uses_the_text_fallback(stream: io.StringIO) -> None:
    with override_settings(log_format="text"):
        configure_logging(stream=stream)
    logging.getLogger("codeeval.anything").warning("careful")
    assert re.fullmatch(TIMESTAMP + r" WARNING  codeeval\.anything: careful\n", stream.getvalue())


def test_configure_logging_honours_the_level(stream: io.StringIO) -> None:
    with override_settings(log_level="WARNING"):
        configure_logging(stream=stream)
    logger = logging.getLogger("codeeval.anything")
    logger.info("dropped")
    logger.error("kept")
    assert [record["message"] for record in _records(stream)] == ["kept"]


def test_configure_logging_accepts_explicit_settings(stream: io.StringIO) -> None:
    with override_settings(log_format="text") as text_settings:
        pass
    configure_logging(text_settings, stream=stream)
    logging.getLogger("codeeval.anything").info("plain")
    assert stream.getvalue().endswith(" INFO     codeeval.anything: plain\n")


def test_configure_logging_replaces_its_own_handler() -> None:
    root = logging.getLogger()
    first = configure_logging(stream=io.StringIO())
    second = configure_logging(stream=io.StringIO())

    ours = [handler for handler in root.handlers if handler.get_name() == HANDLER_NAME]
    assert ours == [second]
    assert first not in root.handlers


# --- bound context ----------------------------------------------------------


def test_bind_context_stamps_every_record_and_nests(stream: io.StringIO) -> None:
    configure_logging(stream=stream)
    logger = logging.getLogger("codeeval.anything")

    with bind_context(run_id="run-1"):
        logger.info("start")
        with bind_context(task_id="HumanEval/0"):
            logger.info("task")
            assert dict(current_context()) == {"run_id": "run-1", "task_id": "HumanEval/0"}
        logger.info("end")
    logger.info("after")
    assert dict(current_context()) == {}

    start, task, end, after = _records(stream)
    assert (start["run_id"], "task_id" in start) == ("run-1", False)
    assert (task["run_id"], task["task_id"]) == ("run-1", "HumanEval/0")
    assert (end["run_id"], "task_id" in end) == ("run-1", False)
    assert "run_id" not in after


def test_record_extras_win_over_the_bound_context(stream: io.StringIO) -> None:
    configure_logging(stream=stream)
    with bind_context(task_id="outer"):
        logging.getLogger("codeeval.anything").info("x", extra={"task_id": "inner"})
    (record,) = _records(stream)
    assert record["task_id"] == "inner"


def test_bind_context_is_restored_after_an_exception() -> None:
    with pytest.raises(RuntimeError), bind_context(run_id="r"):
        raise RuntimeError("boom")
    assert dict(current_context()) == {}


# --- the upstream evaluator logs its run ------------------------------------


def test_evaluate_logs_the_run_with_extras(
    stubbed_evaluation: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    example_problem_file: Path,
    example_samples_file: Path,
) -> None:
    caplog.set_level(logging.INFO, logger="human_eval.evaluation")
    with override_settings(workers=3, sample_timeout=0.5):
        evaluation.evaluate_functional_correctness(
            example_samples_file, k=[1], problem_file=example_problem_file
        )

    started, finished = caplog.records
    assert started.message == "evaluating samples"
    assert (started.n_samples, started.n_problems) == (6, 1)  # type: ignore[attr-defined]
    assert (started.n_workers, started.timeout) == (3, 0.5)  # type: ignore[attr-defined]
    assert finished.message == "wrote results"
    assert finished.pass_at_k == {"pass@1": 1.0}  # type: ignore[attr-defined]
    assert finished.results_file == f"{example_samples_file}_results.jsonl"  # type: ignore[attr-defined]


@pytest.mark.slow
def test_cli_writes_json_logs_to_stderr_and_text_on_request(
    example_problem_file: Path, example_samples_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = [
        sys.executable,
        "-m",
        "human_eval.evaluate_functional_correctness",
        str(example_samples_file),
        f"--problem_file={example_problem_file}",
        "--k=1",
        "--timeout=1",
    ]

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    records = [json.loads(line) for line in proc.stderr.splitlines() if line.startswith("{")]
    assert [record["message"] for record in records] == ["evaluating samples", "wrote results"]
    assert len({record["run_id"] for record in records}) == 1
    assert records[0]["logger"] == "human_eval.evaluation"
    printed = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
    assert printed == pytest.approx({"pass@1": 0.5})

    monkeypatch.setenv("VERIFYBENCH_LOG_FORMAT", "text")
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    text_lines = [line for line in proc.stderr.splitlines() if "human_eval.evaluation:" in line]
    assert len(text_lines) == 2
    assert re.match(
        TIMESTAMP + r" INFO     human_eval\.evaluation: evaluating samples ", text_lines[0]
    )
    assert "run_id=" in text_lines[0]
