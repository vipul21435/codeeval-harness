"""Runs a completion's check program in a throwaway interpreter and grades it.

Each call to :func:`check_correctness` starts a fresh Python process with
:mod:`subprocess` (never :mod:`multiprocessing`), so:

- the caller needs no ``if __name__ == "__main__"`` guard and no start method
  is forced: the worker never re-imports the parent's ``__main__``;
- the worker boots with stdlib imports only, then reports ``started`` on its
  stdout; the ``timeout`` clock starts at that point, so interpreter start-up
  under load is never mistaken for a slow completion;
- the verdict is derived from the worker's exit status. The program under test
  shares the worker interpreter with :func:`unsafe_execute`, so nothing it can
  reach (frames, module globals, the stdout pipe) holds a "passed" flag it could
  flip; the parent grades ``passed`` only for exit status 0, which the worker
  produces solely after ``exec`` returned normally.

This is still not a security sandbox: the completion runs with the worker's
privileges and :func:`reliability_guard` only disables the obvious escape
hatches. Run evaluations inside a container or VM you are prepared to lose.
"""

import builtins
import contextlib
import faulthandler
import io
import json
import os
import platform
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from typing import Any, TextIO

# Seconds the parent waits for the worker interpreter to boot and report that
# it is about to run the program. Only start-up counts against it, so it can be
# generous without letting a hung program linger.
WORKER_STARTUP_TIMEOUT = 60.0

# Seconds added to ``timeout`` once the program is running: room for the
# worker's own alarm to fire, the verdict to be written and the process to exit.
WORKER_GRACE_PERIOD = 1.0

# Bytes of worker stdout and stderr kept in memory (the tail of each stream).
_CAPTURE_LIMIT = 1 << 16

_STARTED = b"started"

# Runs as the worker's ``__main__``: gives the program the parent's module
# search path, then hands over to _worker_main. Reads exactly one line so the
# request that follows it can be read by _worker_main.
_WORKER_BOOTSTRAP = (
    "import json, sys\n"
    "sys.path[:] = json.loads(sys.stdin.buffer.readline())\n"
    "from human_eval.execution import _worker_main\n"
    "_worker_main()\n"
)


class WorkerError(RuntimeError):
    """The worker interpreter could not be started or died before running the program."""


def build_check_program(problem: dict[str, Any], completion: str) -> str:
    """The program that is executed: prompt, completion, tests and the check call."""
    prompt: str = problem["prompt"]
    test: str = problem["test"]
    return prompt + completion + "\n" + test + "\n" + f"check({problem['entry_point']})"


def check_correctness(
    problem: dict[str, Any], completion: str, timeout: float, completion_id: int | None = None
) -> dict[str, Any]:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    outcome = run_check_program(build_check_program(problem, completion), timeout)
    return {
        "task_id": problem["task_id"],
        "passed": outcome == "passed",
        "result": outcome,
        "completion_id": completion_id,
    }


def run_check_program(program: str, timeout: float) -> str:
    """Runs ``program`` in a fresh interpreter under :func:`reliability_guard`.

    Returns ``"passed"``, ``"timed out"`` or ``"failed: <reason>"``. ``timeout``
    bounds the program's own run time; interpreter start-up is not counted.
    Raises :class:`WorkerError` when no worker could be brought up at all.
    """
    request = json.dumps({"program": program, "timeout": timeout})
    payload = f"{json.dumps(sys.path)}\n{request}\n".encode()
    with subprocess.Popen(
        [sys.executable, "-c", _WORKER_BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    ) as proc:
        try:
            return _grade(proc, payload, timeout)
        finally:
            proc.kill()  # a no-op once the worker has exited


def _grade(proc: subprocess.Popen[bytes], payload: bytes, timeout: float) -> str:
    stdin, stdout, stderr = proc.stdin, proc.stdout, proc.stderr
    assert stdin is not None
    assert stdout is not None
    assert stderr is not None
    with contextlib.suppress(BrokenPipeError):
        stdin.write(payload)
        stdin.close()

    out = bytearray()
    err = bytearray()
    streams = {stdout.fileno(): out, stderr.fileno(): err}

    # Phase 1: only the worker's own start-up runs; wait for its first line.
    started = _pump(streams, time.monotonic() + WORKER_STARTUP_TIMEOUT, lambda: b"\n" in out)
    if not started:
        raise WorkerError(f"worker interpreter did not start within {WORKER_STARTUP_TIMEOUT}s")
    if out.split(b"\n", 1)[0] != _STARTED:
        proc.kill()
        proc.wait()
        raise WorkerError(
            "worker interpreter exited before running the program "
            f"({_describe_exit(proc.returncode)}):\n{err.decode(errors='replace').strip()}"
        )

    # Phase 2: the program is running; it gets ``timeout`` plus a grace period.
    deadline = time.monotonic() + timeout + WORKER_GRACE_PERIOD
    if not _pump(streams, deadline, lambda: False):
        return "timed out"
    try:
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return "timed out"
    return _verdict(proc.returncode, bytes(out))


def _pump(streams: dict[int, bytearray], deadline: float, done: Callable[[], bool]) -> bool:
    """Reads the fds in ``streams`` into their buffers until they all hit EOF.

    Stops early once ``done()`` is true. Returns False if ``deadline`` (a
    ``time.monotonic()`` value) passes first. Only the tail of each stream is
    kept so a program writing to the raw fds cannot exhaust the parent's memory.
    """
    while streams and not done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        readable, _, _ = select.select(list(streams), [], [], remaining)
        if not readable:
            return False
        for fd in readable:
            chunk = os.read(fd, 65536)
            if not chunk:
                del streams[fd]
                continue
            buffer = streams[fd]
            buffer += chunk
            if len(buffer) > 2 * _CAPTURE_LIMIT:
                del buffer[: len(buffer) - _CAPTURE_LIMIT]
    return True


def _verdict(returncode: int, stdout: bytes) -> str:
    """Combines the worker's exit status with the verdict line it wrote last.

    Exit status 0 is the only evidence of a pass; the verdict line carries the
    reason for a failure and cannot upgrade a non-zero exit into a pass.
    """
    lines = stdout.splitlines()
    outcome: str | None = None
    if lines:
        with contextlib.suppress(ValueError):
            data = json.loads(lines[-1])
            if isinstance(data, dict) and isinstance(data.get("outcome"), str):
                outcome = data["outcome"]
    if returncode == 0 and outcome == "passed":
        return "passed"
    if outcome is not None and outcome != "passed":
        return outcome
    return f"failed: worker exited with {_describe_exit(returncode)}"


def _describe_exit(returncode: int | None) -> str:
    if returncode is not None and returncode < 0:
        with contextlib.suppress(ValueError):
            return f"signal {signal.Signals(-returncode).name}"
    return f"status {returncode}"


def _worker_main() -> None:
    """Entry point of the worker interpreter; see ``_WORKER_BOOTSTRAP``."""
    request = json.loads(sys.stdin.buffer.readline())
    # Taken before reliability_guard runs so the program cannot reach them
    # through the os module; they are how the verdict leaves the process.
    exit_ = os._exit
    write = os.write

    _write_all(write, _STARTED + b"\n")
    outcome = unsafe_execute(request["program"], request["timeout"])
    _write_all(write, json.dumps({"outcome": outcome}).encode() + b"\n")
    exit_(0 if outcome == "passed" else 1)


def _write_all(write: Callable[[int, bytes], int], data: bytes) -> None:
    while data:
        data = data[write(1, data) :]


def unsafe_execute(check_program: str, timeout: float) -> str:
    """Runs a check program in the current process and returns the outcome.

    Installs :func:`reliability_guard`, which disables functions process-wide,
    so this belongs in a throwaway interpreter: :func:`run_check_program` is
    the entry point for callers.
    """
    with create_tempdir():
        # These functions are needed when cleaning up the tempdir after the
        # reliability guard has disabled them; keep references to restore.
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink
        getcwd = os.getcwd

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        try:
            exec_globals: dict[str, Any] = {}
            with swallow_io(), time_limit(timeout):
                # WARNING
                # This program exists to execute untrusted model-generated code. Although
                # it is highly unlikely that model-generated code will do something overtly
                # malicious in response to this test suite, model-generated code may act
                # destructively due to a lack of model capability or alignment.
                # Users are strongly encouraged to sandbox this evaluation suite so that it
                # does not perform destructive actions on their host or network. For more
                # information on how OpenAI sandboxes its code, see the accompanying paper.
                exec(check_program, exec_globals)
            outcome = "passed"
        except TimeoutException:
            outcome = "timed out"
        except BaseException as e:
            outcome = f"failed: {_safe_str(e)}"

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink
        os.getcwd = getcwd

    return outcome


def _safe_str(exc: BaseException) -> str:
    try:
        return str(exc)
    except BaseException:  # a __str__ written by the program may itself be broken
        return type(exc).__name__


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    def signal_handler(_signum: int, _frame: object) -> None:
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io() -> Iterator[None]:
    stream = WriteOnlyStringIO()
    with (
        contextlib.redirect_stdout(stream),
        contextlib.redirect_stderr(stream),
        redirect_stdin(stream),
    ):
        yield


@contextlib.contextmanager
def create_tempdir() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as dirname, chdir(dirname):
        yield dirname


class TimeoutException(Exception):  # noqa: N818 - upstream public name
    pass


_STDIN_BLOCKED = "stdin is not readable while a completion is being evaluated"


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    # typeshed declares the str-returning text overrides of readline/readlines
    # with the same ignore because they clash with the bytes-based _IOBase.
    def read(self, size: int | None = -1, /) -> str:
        raise OSError(_STDIN_BLOCKED)

    def readline(self, size: int | None = -1, /) -> str:  # type: ignore[override]
        raise OSError(_STDIN_BLOCKED)

    def readlines(self, hint: int = -1, /) -> list[str]:  # type: ignore[override]
        raise OSError(_STDIN_BLOCKED)

    def readable(self) -> bool:
        """Returns True if the IO object can be read."""
        return False


@contextlib.contextmanager
def redirect_stdin(new_target: TextIO) -> Iterator[TextIO]:
    old_target = sys.stdin
    sys.stdin = new_target
    try:
        yield new_target
    finally:
        sys.stdin = old_target


@contextlib.contextmanager
def chdir(root: str) -> Iterator[None]:
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


# Names disabled by reliability_guard, grouped by the module they live on. The
# os names are disabled on the posix module as well, since os re-exports it and
# ``__import__("posix")`` would otherwise hand back every function untouched.
_DISABLED_BUILTINS = ("exit", "quit", "help")
_DISABLED_OS_FUNCTIONS = (
    "kill",
    "system",
    "putenv",
    "remove",
    "removedirs",
    "rmdir",
    "fchdir",
    "setuid",
    "fork",
    "forkpty",
    "killpg",
    "rename",
    "renames",
    "truncate",
    "replace",
    "unlink",
    "fchmod",
    "fchown",
    "chmod",
    "chown",
    "chroot",
    "lchflags",
    "lchmod",
    "lchown",
    "getcwd",
    "chdir",
    # Ways to end or replace the worker process with an exit status of the
    # program's choosing, which is what the parent grades a pass from.
    "_exit",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "posix_spawn",
    "posix_spawnp",
)
_DISABLED_SHUTIL_FUNCTIONS = ("rmtree", "move", "chown")
# ctypes would give back every disabled function through libc.
_BLOCKED_MODULES = ("ipdb", "joblib", "resource", "psutil", "tkinter", "ctypes", "_ctypes")


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    for name in _DISABLED_BUILTINS:
        setattr(builtins, name, None)

    os.environ["OMP_NUM_THREADS"] = "1"

    # importlib._bootstrap_external calls posix.getcwd, posix.replace and
    # posix.unlink directly: getcwd to resolve relative sys.path entries and the
    # other two to write bytecode caches. Make both unnecessary so that imports
    # keep working once the posix names below are gone.
    sys.path[:] = [os.path.abspath(entry) for entry in sys.path]
    sys.dont_write_bytecode = True

    posix = sys.modules.get("posix")
    for name in _DISABLED_OS_FUNCTIONS:
        setattr(os, name, None)
        if posix is not None and hasattr(posix, name):
            setattr(posix, name, None)

    for name in _DISABLED_SHUTIL_FUNCTIONS:
        setattr(shutil, name, None)

    subprocess.Popen = None  # type: ignore[misc, assignment]

    # Setting a sys.modules entry to None makes 'import <name>' raise ImportError.
    for name in _BLOCKED_MODULES:
        sys.modules[name] = None  # type: ignore[assignment]
