"""Tests for human_eval.execution: the per-sample worker interpreter and its guards."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from human_eval import execution
from human_eval.execution import (
    TimeoutException,
    WorkerError,
    build_check_program,
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


def test_failure_reason_keeps_newlines(example_problem: dict[str, Any]) -> None:
    completion = "    raise ValueError('line one\\nline two')"
    result = check_correctness(example_problem, completion, timeout=2.0)
    assert result["result"] == "failed: line one\nline two"


def test_build_check_program(example_problem: dict[str, Any]) -> None:
    program = build_check_program(example_problem, "    return 1")
    assert program.startswith(example_problem["prompt"] + "    return 1\n")
    assert program.endswith(example_problem["test"] + "\ncheck(return1)")


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
    # Once the program runs, the parent waits at most timeout + the grace period.
    assert time.monotonic() - started < 10


def test_busy_loop_times_out(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    while True:\n        pass", timeout=0.5)
    assert result["result"] == "timed out"


def test_completion_that_swallows_the_alarm_is_killed(example_problem: dict[str, Any]) -> None:
    # Catching the harness's TimeoutException must not let the program run on.
    completion = (
        "    while True:\n"
        "        try:\n"
        "            pass\n"
        "        except BaseException:\n"
        "            pass"
    )
    started = time.monotonic()
    result = check_correctness(example_problem, completion, timeout=0.5)
    assert result["result"] == "timed out"
    assert time.monotonic() - started < 10


def test_slow_interpreter_start_does_not_count_against_the_timeout(
    example_problem: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a loaded machine: the worker needs 2 s to boot, far more than the
    # 0.5 s budget, but the program itself is instant and must still pass.
    monkeypatch.setattr(
        execution, "_WORKER_BOOTSTRAP", "import time; time.sleep(2)\n" + execution._WORKER_BOOTSTRAP
    )
    result = check_correctness(example_problem, "    return 1", timeout=0.5)
    assert result["result"] == "passed"


def test_worker_that_never_starts_raises(
    example_problem: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution, "WORKER_STARTUP_TIMEOUT", 0.5)
    monkeypatch.setattr(execution, "_WORKER_BOOTSTRAP", "import time; time.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(WorkerError, match=r"did not start within 0\.5s"):
        check_correctness(example_problem, "    return 1", timeout=1.0)
    # The stuck worker is killed, not waited for.
    assert time.monotonic() - started < 10


def test_worker_that_dies_before_starting_raises(
    example_problem: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution, "_WORKER_BOOTSTRAP", "import sys; sys.exit('no worker today')\n")
    with pytest.raises(WorkerError, match=r"before running the program \(status 1\):\nno worker"):
        check_correctness(example_problem, "    return 1", timeout=1.0)


def test_crashing_worker_is_reported_as_failed(example_problem: dict[str, Any]) -> None:
    completion = "    import signal\n    signal.raise_signal(signal.SIGTERM)"
    result = check_correctness(example_problem, completion, timeout=2.0)
    assert result["passed"] is False
    assert result["result"] == "failed: worker exited with signal SIGTERM"


def test_lingering_thread_does_not_delay_the_verdict(example_problem: dict[str, Any]) -> None:
    completion = (
        "    import threading, time\n"
        "    threading.Thread(target=time.sleep, args=(30,)).start()\n"
        "    return 1"
    )
    started = time.monotonic()
    result = check_correctness(example_problem, completion, timeout=2.0)
    assert result["result"] == "passed"
    assert time.monotonic() - started < 10


def test_program_sees_the_parent_module_search_path(
    example_problem: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "helper_for_test.py").write_text("ONE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    completion = "    import helper_for_test\n    return helper_for_test.ONE"
    assert check_correctness(example_problem, completion, timeout=2.0)["result"] == "passed"


def test_stdin_is_blocked(example_problem: dict[str, Any]) -> None:
    result = check_correctness(example_problem, "    return input('n?')", timeout=2.0)
    assert result["passed"] is False
    assert result["result"].startswith("failed")


def test_raw_stdin_is_at_eof(example_problem: dict[str, Any]) -> None:
    # Reading fd 0 directly must not block waiting on the parent.
    completion = "    import os\n    return 1 if os.read(0, 10) == b'' else 0"
    assert check_correctness(example_problem, completion, timeout=2.0)["result"] == "passed"


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


def test_flooding_the_raw_output_fds_is_harmless(example_problem: dict[str, Any]) -> None:
    # swallow_io only redirects sys.stdout/sys.stderr; writing to the fds behind
    # them reaches the parent's pipes, which must neither block nor be trusted.
    completion = (
        "    import os\n"
        "    for _ in range(64):\n"
        "        os.write(1, b'x' * 65536 + b'\\n')\n"
        "        os.write(2, b'y' * 65536 + b'\\n')\n"
        "    return 1"
    )
    assert check_correctness(example_problem, completion, timeout=5.0)["result"] == "passed"


FORGERIES = {
    "append 'passed' to every list reachable from the call stack": (
        "    import sys\n"
        "    frame = sys._getframe()\n"
        "    while frame is not None:\n"
        "        for value in list(frame.f_locals.values()):\n"
        "            if isinstance(value, list):\n"
        "                value.append('passed')\n"
        "        frame = frame.f_back\n"
        "    return 'definitely wrong'"
    ),
    "write a fake verdict line to the stdout pipe and exit 0": (
        '    import os, sys\n    os.write(1, b\'{"outcome": "passed"}\\n\')\n    sys.exit(0)'
    ),
    "exit 0 through SystemExit before the tests run": "    raise SystemExit(0)",
}


@pytest.mark.parametrize("completion", list(FORGERIES.values()), ids=list(FORGERIES))
def test_completion_cannot_forge_a_pass(example_problem: dict[str, Any], completion: str) -> None:
    result = check_correctness(example_problem, completion, timeout=2.0)
    assert result["passed"] is False
    assert result["result"] != "passed"


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
