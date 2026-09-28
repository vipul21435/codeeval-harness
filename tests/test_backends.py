"""Tests for the model backends: the deterministic mock and the OpenAI-compatible client.

The HTTP backend is exercised against a local ``http.server`` running in a
thread; nothing leaves the machine.
"""

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeServer

from codeeval.backends import (
    STUB_COMPLETION,
    MockBackend,
    ModelBackend,
    OpenAIBackend,
    Sample,
    generate_samples,
    make_backend,
    reference_completions,
    write_samples,
)
from codeeval.errors import ConfigError, DataError, ProviderError
from codeeval.settings import load_settings
from codeeval.tasks import Task, TaskSuite

CANNED = [
    {"task_id": "t/0", "completion": "    return a + b\n"},
    {"task_id": "t/0", "completion": "    return b + a\n"},
    {"task_id": "t/1", "completion": "    return 1\n"},
]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.fixture
def canned_file(tmp_path: Path) -> Path:
    return write_jsonl(tmp_path / "canned.jsonl", CANNED)


# --- MockBackend ---------------------------------------------------------------


def test_mock_replays_in_order_and_cycles(canned_file: Path) -> None:
    backend = MockBackend.from_jsonl(canned_file)
    assert isinstance(backend, ModelBackend)
    assert backend.name == "mock"
    assert backend.complete("t/0", "prompt", n=3) == [
        "    return a + b\n",
        "    return b + a\n",
        "    return a + b\n",
    ]
    assert backend.complete("t/1", "prompt") == ["    return 1\n"]


def test_mock_is_deterministic_across_instances_and_call_order(canned_file: Path) -> None:
    first = MockBackend.from_jsonl(canned_file, seed=7, failure_rate=0.5)
    second = MockBackend.from_jsonl(canned_file, seed=7, failure_rate=0.5)
    a = first.complete("t/0", "p", n=20)
    second.complete("t/1", "p", n=5)  # a different call order must not change t/0
    assert second.complete("t/0", "p", n=20) == a
    assert STUB_COMPLETION in a
    assert any(completion != STUB_COMPLETION for completion in a)


def test_mock_seed_changes_which_completions_fail(canned_file: Path) -> None:
    a = MockBackend.from_jsonl(canned_file, seed=1, failure_rate=0.5).complete("t/0", "p", n=40)
    b = MockBackend.from_jsonl(canned_file, seed=2, failure_rate=0.5).complete("t/0", "p", n=40)
    assert a != b


@pytest.mark.parametrize(("rate", "expected"), [(0.0, 0), (1.0, 10)])
def test_mock_failure_rate_extremes(canned_file: Path, rate: float, expected: int) -> None:
    backend = MockBackend.from_jsonl(canned_file, failure_rate=rate)
    stubs = backend.complete("t/1", "p", n=10).count(STUB_COMPLETION)
    assert stubs == expected


def test_mock_rejects_bad_failure_rate() -> None:
    with pytest.raises(ConfigError, match="failure_rate"):
        MockBackend({"t/0": ["x"]}, failure_rate=1.5)


def test_mock_rejects_task_without_completions() -> None:
    with pytest.raises(ConfigError, match="no canned completion") as info:
        MockBackend({"t/0": []})
    assert info.value.details["task_ids"] == ["t/0"]


def test_mock_unknown_task_is_a_provider_error(canned_file: Path) -> None:
    backend = MockBackend.from_jsonl(canned_file)
    with pytest.raises(ProviderError, match="t/9") as info:
        backend.complete("t/9", "p")
    assert info.value.details == {"task_id": "t/9", "known": 2}


def test_mock_rejects_n_below_one(canned_file: Path) -> None:
    with pytest.raises(ValueError, match="n must be"):
        MockBackend.from_jsonl(canned_file).complete("t/0", "p", n=0)


def test_mock_from_pairs() -> None:
    backend = MockBackend.from_pairs([("t/0", "a"), ("t/0", "b")])
    assert backend.complete("t/0", "p", n=2) == ["a", "b"]


def test_mock_rejects_an_empty_canned_mapping() -> None:
    with pytest.raises(ConfigError, match="at least one canned completion"):
        MockBackend.from_pairs([])
    with pytest.raises(ConfigError, match="at least one canned completion"):
        MockBackend({})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "holds no completions"),
        ('{"task_id": 1, "completion": "x"}\n', ":1: expected string"),
        ('{"task_id": "t/0"}\n', ":1: expected string"),
        ("not json\n", ":1: invalid JSON"),
        ("[1, 2]\n", ":1: expected a JSON object"),
    ],
)
def test_mock_from_jsonl_reports_bad_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DataError, match=message):
        MockBackend.from_jsonl(path)


def test_mock_from_jsonl_skips_blank_lines_and_reports_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "canned.jsonl"
    path.write_text('\n{"task_id": "t/0", "completion": "x"}\n\n', encoding="utf-8")
    assert MockBackend.from_jsonl(path).complete("t/0", "p") == ["x"]
    with pytest.raises(DataError, match="cannot read"):
        MockBackend.from_jsonl(tmp_path / "missing.jsonl")


# --- OpenAIBackend -------------------------------------------------------------


def chat_response(*contents: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}} for text in contents]}


@pytest.fixture
def make_client(fake_server: FakeServer) -> Callable[..., OpenAIBackend]:
    def factory(**overrides: Any) -> OpenAIBackend:
        options: dict[str, Any] = {
            "base_url": fake_server.base_url + "/",
            "model": "test-model",
            "api_key": "sk-test",
            "timeout": 5.0,
            "retries": 2,
            "backoff": 0.0,
        }
        options.update(overrides)
        return OpenAIBackend(**options)

    return factory


def test_openai_posts_chat_request_and_returns_choices(
    fake_server: FakeServer, make_client: Callable[..., OpenAIBackend]
) -> None:
    fake_server.script((200, chat_response("    return 1\n", "    return 2\n")))
    client = make_client()
    assert isinstance(client, ModelBackend)
    assert client.name == "openai"
    assert client.url == f"{fake_server.base_url}/chat/completions"

    assert client.complete("t/0", "def f():", n=2) == ["    return 1\n", "    return 2\n"]

    request = fake_server.requests[0]
    assert request["model"] == "test-model"
    assert request["n"] == 2
    assert request["messages"][-1] == {"role": "user", "content": "def f():"}
    assert fake_server.headers[0]["Authorization"] == "Bearer sk-test"
    assert fake_server.headers[0]["Content-Type"] == "application/json"


def test_openai_retries_on_429_and_5xx_then_succeeds(
    fake_server: FakeServer, make_client: Callable[..., OpenAIBackend]
) -> None:
    fake_server.script((429, {}), (503, {}), (200, chat_response("ok")))
    slept: list[float] = []
    client = make_client(backoff=0.5, sleep=slept.append)
    assert client.complete("t/0", "p") == ["ok"]
    assert len(fake_server.requests) == 3
    assert slept == [0.5, 1.0]


def test_openai_gives_up_after_retries(
    fake_server: FakeServer,
    make_client: Callable[..., OpenAIBackend],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_server.script((500, {}), (500, {}))
    client = make_client(retries=1, sleep=lambda _: None)
    with (
        caplog.at_level("WARNING", logger="codeeval.backends"),
        pytest.raises(ProviderError, match="failed after 2 attempt") as info,
    ):
        client.complete("t/0", "p")
    assert info.value.details["attempts"] == 2
    assert info.value.details["reason"] == "HTTP 500"
    # One retry followed the first failure; the last attempt is not announced as a retry.
    retrying = [record for record in caplog.records if record.msg == "openai backend retrying"]
    assert [record.attempt for record in retrying] == [1]  # type: ignore[attr-defined]


def test_openai_without_retries_does_not_log_a_retry(
    fake_server: FakeServer,
    make_client: Callable[..., OpenAIBackend],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_server.script((503, {}))
    slept: list[float] = []
    client = make_client(retries=0, sleep=slept.append)
    with (
        caplog.at_level("WARNING", logger="codeeval.backends"),
        pytest.raises(ProviderError, match="failed after 1 attempt"),
    ):
        client.complete("t/0", "p")
    assert slept == []
    assert "retrying" not in caplog.text


def test_openai_does_not_retry_client_errors(
    fake_server: FakeServer, make_client: Callable[..., OpenAIBackend]
) -> None:
    fake_server.script((401, {"error": "bad key"}))
    with pytest.raises(ProviderError, match="HTTP 401") as info:
        make_client().complete("t/0", "p")
    assert len(fake_server.requests) == 1
    assert info.value.details["status"] == 401
    assert "bad key" in info.value.details["body"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not json", "non-JSON body"),
        (b"[]", "non-object body"),
        (b"{}", "no choices"),
        (b'{"choices": []}', "no choices"),
        (b'{"choices": [{"message": {}}]}', "without message.content"),
        (b'{"choices": ["x"]}', "without message.content"),
    ],
)
def test_openai_rejects_malformed_responses(
    fake_server: FakeServer, make_client: Callable[..., OpenAIBackend], body: bytes, message: str
) -> None:
    fake_server.script((200, body))
    with pytest.raises(ProviderError, match=message):
        make_client().complete("t/0", "p")


def test_openai_truncates_to_n_and_logs_short_answers(
    fake_server: FakeServer,
    make_client: Callable[..., OpenAIBackend],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_server.script((200, chat_response("a", "b", "c")), (200, chat_response("a")))
    client = make_client()
    assert client.complete("t/0", "p", n=2) == ["a", "b"]
    with caplog.at_level("WARNING", logger="codeeval.backends"):
        assert client.complete("t/0", "p", n=2) == ["a"]
    assert "fewer completions" in caplog.text


def test_openai_retries_transport_errors(make_client: Callable[..., OpenAIBackend]) -> None:
    # Nothing listens on a closed port: every attempt is a connection error.
    probe = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = probe.server_address[1]
    probe.server_close()
    client = make_client(base_url=f"http://127.0.0.1:{port}/v1", retries=1, sleep=lambda _: None)
    with pytest.raises(ProviderError, match="failed after 2 attempt") as info:
        client.complete("t/0", "p")
    assert "URLError" in str(info.value.details["reason"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": ""}, "needs an API key"),
        ({"base_url": "ftp://x"}, "http:// or https://"),
        ({"retries": -1}, "retries must be"),
        ({"timeout": 0}, "retries must be"),
    ],
)
def test_openai_validates_construction(
    make_client: Callable[..., OpenAIBackend], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        make_client(**overrides)


def test_openai_rejects_n_below_one(make_client: Callable[..., OpenAIBackend]) -> None:
    with pytest.raises(ValueError, match="n must be"):
        make_client().complete("t/0", "p", n=0)


# --- make_backend, generate_samples, write_samples -----------------------------


def test_make_backend_mock_reads_settings(canned_file: Path) -> None:
    settings = load_settings(seed=3, mock_failure_rate=1.0)
    backend = make_backend("mock", settings=settings, canned=canned_file)
    assert isinstance(backend, MockBackend)
    assert (backend.seed, backend.failure_rate) == (3, 1.0)
    overridden = make_backend("mock", settings=settings, canned=canned_file, seed=9, failure_rate=0)
    assert isinstance(overridden, MockBackend)
    assert (overridden.seed, overridden.failure_rate) == (9, 0.0)


def test_make_backend_mock_needs_a_canned_file_or_pairs() -> None:
    with pytest.raises(ConfigError, match="canned"):
        make_backend("mock", settings=load_settings())
    backend = make_backend("mock", settings=load_settings(), pairs=[("t/0", "x")])
    assert backend.complete("t/0", "p") == ["x"]


def test_reference_completions_strip_the_prompt_when_present(
    make_task: Callable[..., Task],
) -> None:
    prompt = 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n'
    ported = make_task(
        task_id="h/0", prompt=prompt, reference_solution=prompt + "    return a + b\n"
    )
    other = make_task(task_id="h/1")  # reference without the docstring: replayed whole
    pairs = reference_completions(TaskSuite(tasks=[ported, other]))
    assert pairs == [
        ("h/0", "    return a + b\n"),
        ("h/1", "def add(a: int, b: int) -> int:\n    return a + b\n"),
    ]
    backend = MockBackend.from_pairs(pairs)
    assert backend.complete("h/0", prompt) == ["    return a + b\n"]


def test_make_backend_openai_reads_settings_and_needs_a_key() -> None:
    settings = load_settings(
        openai_api_key="sk-1",
        openai_base_url="http://localhost:1/v1/",
        openai_model="m",
        openai_timeout=2.5,
        openai_retries=0,
    )
    backend = make_backend("openai", settings=settings)
    assert isinstance(backend, OpenAIBackend)
    assert backend.url == "http://localhost:1/v1/chat/completions"
    assert (backend.model, backend.api_key) == ("m", "sk-1")
    assert (backend.timeout, backend.retries) == (2.5, 0)
    with pytest.raises(ConfigError, match="VERIFYBENCH_OPENAI_API_KEY"):
        make_backend("openai", settings=load_settings())


def test_make_backend_uses_process_settings_by_default(canned_file: Path) -> None:
    backend = make_backend("mock", canned=canned_file)
    assert isinstance(backend, MockBackend)
    assert backend.seed == 0


def test_make_backend_rejects_unknown_name() -> None:
    with pytest.raises(ConfigError, match="unknown model backend 'nope'"):
        make_backend("nope", settings=load_settings())


def test_settings_hide_the_api_key() -> None:
    settings = load_settings(openai_api_key="sk-secret")
    assert "sk-secret" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-secret"


def test_generate_and_write_samples(canned_file: Path, tmp_path: Path) -> None:
    backend = MockBackend.from_jsonl(canned_file)
    samples = generate_samples(backend, [("t/0", "p0"), ("t/1", "p1")], n=2)
    assert samples == [
        Sample("t/0", "    return a + b\n", "mock", 0),
        Sample("t/0", "    return b + a\n", "mock", 1),
        Sample("t/1", "    return 1\n", "mock", 0),
        Sample("t/1", "    return 1\n", "mock", 1),
    ]
    out = tmp_path / "nested" / "samples.jsonl"
    assert write_samples(out, samples) == 4
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert records[0] == {
        "task_id": "t/0",
        "completion": "    return a + b\n",
        "backend": "mock",
        "index": 0,
    }
    assert len(records) == 4


def test_generate_samples_rejects_n_below_one(canned_file: Path) -> None:
    with pytest.raises(ValueError, match="n must be"):
        generate_samples(MockBackend.from_jsonl(canned_file), [], n=0)
