import builtins
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator, MutableSequence
from typing import Any, TextIO

# "spawn" is the default start method on macOS and Windows and the only one that
# is safe to use from a multi-threaded parent (evaluation.py drives this module
# from a ThreadPoolExecutor). Selecting it explicitly makes Linux behave the same
# way instead of forking a threaded process, which CPython 3.12 warns about.
_MP_CONTEXT = multiprocessing.get_context("spawn")


def unsafe_execute(
    problem: dict[str, Any], completion: str, timeout: float, result: MutableSequence[str]
) -> None:
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

        # Construct the check program and run it.
        check_program = (
            problem["prompt"]
            + completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )

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
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink
        os.getcwd = getcwd


def check_correctness(
    problem: dict[str, Any], completion: str, timeout: float, completion_id: int | None = None
) -> dict[str, Any]:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    with _MP_CONTEXT.Manager() as manager:
        result = manager.list()

        p = _MP_CONTEXT.Process(target=unsafe_execute, args=(problem, completion, timeout, result))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
            p.join()

        outcome: str = result[0] if len(result) else "timed out"

    return {
        "task_id": problem["task_id"],
        "passed": outcome == "passed",
        "result": outcome,
        "completion_id": completion_id,
    }


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


# Names disabled by reliability_guard, grouped by the module they live on.
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
)
_DISABLED_SHUTIL_FUNCTIONS = ("rmtree", "move", "chown")
_BLOCKED_MODULES = ("ipdb", "joblib", "resource", "psutil", "tkinter")


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

    for name in _DISABLED_OS_FUNCTIONS:
        setattr(os, name, None)

    for name in _DISABLED_SHUTIL_FUNCTIONS:
        setattr(shutil, name, None)

    subprocess.Popen = None  # type: ignore[misc, assignment]

    # Setting a sys.modules entry to None makes 'import <name>' raise ImportError.
    for name in _BLOCKED_MODULES:
        sys.modules[name] = None  # type: ignore[assignment]
