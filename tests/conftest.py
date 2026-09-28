import copy
import json
import os
import shutil
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from codeeval.settings import ENV_PREFIX, clear_settings_cache
from codeeval.tasks import Task
from human_eval import evaluation
from human_eval.data import read_problems

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"

# A minimal valid task: every field a test may want to override is spelled out.
SYNTHETIC_TASK: dict[str, Any] = {
    "task_id": "synthetic/add",
    "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
    "entry_point": "add",
    "reference_solution": "def add(a: int, b: int) -> int:\n    return a + b\n",
    "baseline_solution": "def add(a: int, b: int) -> int:\n    raise NotImplementedError\n",
    "tests": (
        "from solution import add\n\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n"
        "    assert add(-1, 1) == 0\n"
    ),
    "language": "python",
    "difficulty": "easy",
    "metadata": {"source": "synthetic"},
}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's VERIFYBENCH_* variables and cached settings out of every test."""
    for name in list(os.environ):
        if name.upper().startswith(ENV_PREFIX):
            monkeypatch.delenv(name)
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def example_problem_file(tmp_path: Path) -> Path:
    """A private copy of data/example_problem.jsonl (one trivial task, id test/0)."""
    return Path(shutil.copy(DATA_DIR / "example_problem.jsonl", tmp_path))


@pytest.fixture
def example_samples_file(tmp_path: Path) -> Path:
    """A private copy of data/example_samples.jsonl.

    Copied so that the results file the evaluator writes next to it lands in
    tmp_path instead of the repository's data directory.
    """
    return Path(shutil.copy(DATA_DIR / "example_samples.jsonl", tmp_path))


@pytest.fixture
def example_problem(example_problem_file: Path) -> dict[str, Any]:
    return read_problems(example_problem_file)["test/0"]


@pytest.fixture
def stubbed_evaluation(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the evaluator's worker pool and sample runner with stand-ins.

    Nothing is executed: every sample passes. The returned dict records the
    ``workers`` and ``timeout`` the evaluator handed to them.
    """
    seen: dict[str, Any] = {}

    def fake_check(
        problem: dict[str, Any], completion: str, timeout: float, completion_id: int | None = None
    ) -> dict[str, Any]:
        seen["timeout"] = timeout
        return {
            "task_id": problem["task_id"],
            "completion_id": completion_id,
            "passed": True,
            "result": "passed",
        }

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
            seen["workers"] = max_workers
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(evaluation, "check_correctness", fake_check)
    monkeypatch.setattr(evaluation, "ThreadPoolExecutor", RecordingExecutor)
    return seen


@pytest.fixture
def task_record() -> dict[str, Any]:
    """A fresh copy of the minimal valid task record."""
    return copy.deepcopy(SYNTHETIC_TASK)


@pytest.fixture
def make_task() -> Callable[..., Task]:
    """Factory for valid tasks: ``make_task(task_id="x/y", tests=...)`` overrides fields."""

    def factory(**overrides: Any) -> Task:
        return Task.model_validate({**SYNTHETIC_TASK, **overrides})

    return factory


class FakeServer:
    """An OpenAI-compatible chat completions endpoint with scripted responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.responses: list[tuple[int, bytes]] = []
        self.server = HTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = self.server.server_address[1]
        return f"http://127.0.0.1:{port}/v1"

    def script(self, *responses: tuple[int, object]) -> None:
        self.responses = [
            (status, body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
            for status, body in responses
        ]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                fake.requests.append(json.loads(self.rfile.read(length)))
                fake.headers.append(dict(self.headers.items()))
                status, body = fake.responses.pop(0) if fake.responses else (500, b"{}")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    """A scripted OpenAI-compatible server on 127.0.0.1; nothing leaves the machine."""
    server = FakeServer()
    server.start()
    yield server
    server.stop()
