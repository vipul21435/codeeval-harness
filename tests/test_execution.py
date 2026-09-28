"""Tests for human_eval.execution: the per-sample subprocess runner and its guards."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from human_eval.execution import (
    TimeoutException,
    check_correctness,
    create_tempdir,
    swallow_io,
    time_limit,
)

pytestmark = pytest.mark.slow


def test_correct_completion_passes(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    return 1", timeout=2.0, completion_id=7)
    assert result == {
        "task_id": "test/0",
        "passed": True,
        "result": "passed",
        "completion_id": 7,
    }


def test_completion_id_defaults_to_none(example_problem: dict[str, Any]) -> None:
    assert check_correctness(example_problem, "    return 1", timeout=2.0)["completion_id"] is None


def test_wrong_answer_fails(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    return 2", timeout=2.0)
    assert result["passed"] is False
    assert result["result"].startswith("failed")


def test_syntax_error_fails(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    return (", timeout=2.0)
    assert result["passed"] is False
    assert "failed" in result["result"]


def test_sleeping_completion_times_out(example_problem: dict[str, Any]) -> None:
    started = time.monotonic()
    result = check_correctness(
        example_problem, "    import time\n    time.sleep(30)\n    return 1", timeout=0.5
    )
    assert result == {
        "task_id": "test/0",
        "passed": False,
        "result": "timed out",
        "completion_id": None,
    }
    # The parent waits at most timeout + 1 second for the worker.
    assert time.monotonic() - started < 10


def test_busy_loop_times_out(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    while True:\n        pass", timeout=0.5)
    assert result["result"] == "timed out"


def test_stdin_is_blocked(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    return input('n?')", timeout=2.0)
    assert result["passed"] is False
    assert result["result"].startswith("failed")


def test_subprocess_is_blocked(example_problem: dict[str, Any]) -> None:
    completion = "    import subprocess\n    subprocess.check_output('rm -rf tmp')"
    result = check_correctness(example_problem, completion, timeout=2.0)
    assert result["passed"] is False
    assert "NoneType" in result["result"]


def test_completion_may_write_to_its_scratch_cwd(example_problem: dict[str, Any]) -> None:
    # Each run executes inside a throwaway temporary directory; writing there is
    # allowed and cleanup afterwards must not break the result.
    completion = "    open('scratch.txt', 'w').write('x')\n    return 1"
    result = check_correctness(example_problem, completion, timeout=2.0)
    assert result["result"] == "passed"


def test_runs_independently_per_call(example_problem: dict[str, Any]) -> None:
    # State leaked by one completion (a module-level global) must not reach the next.
    leaky = "    global LEAK\n    LEAK = 1\n    return 1"
    probe = "    return 1 if 'LEAK' not in globals() else 0"
    assert check_correctness(example_problem, leaky, timeout=2.0)["passed"]
    assert check_correctness(example_problem, probe, timeout=2.0)["passed"]


def test_time_limit_raises_after_deadline() -> None:
    with pytest.raises(TimeoutException), time_limit(0.1):
        time.sleep(5)


def test_time_limit_is_cleared_on_exit() -> None:
    with time_limit(0.2):
        pass
    # Without cancelling the timer the SIGALRM would fire during this sleep.
    time.sleep(0.3)


def test_swallow_io_hides_output_and_blocks_input(capsys: pytest.CaptureFixture[str]) -> None:
    with swallow_io():
        print("hidden")
        print("hidden too", file=sys.stderr)
        with pytest.raises(OSError, match="stdin is not readable"):
            input()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_create_tempdir_changes_and_restores_cwd() -> None:
    before = Path.cwd()
    with create_tempdir() as dirname:
        inside = Path.cwd()
        assert inside == Path(dirname).resolve()
        assert inside != before
    assert Path.cwd() == before
    assert not Path(dirname).exists()


def test_reliability_guard_disables_destructive_functions() -> None:
    # The guard mutates process-wide state, so exercise it in a fresh interpreter.
    probe = """
import builtins, os, shutil, subprocess, sys
from human_eval.execution import reliability_guard
reliability_guard()
assert os.system is None and os.kill is None and os.remove is None
assert shutil.rmtree is None and subprocess.Popen is None
assert builtins.exit is None and builtins.quit is None
assert os.environ["OMP_NUM_THREADS"] == "1"
try:
    import tkinter
except ImportError:
    pass
else:
    raise AssertionError("tkinter import should be blocked")
print("guard-ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "guard-ok"
