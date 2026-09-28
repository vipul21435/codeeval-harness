"""Model backends: where completions come from, behind one small protocol.

A :class:`ModelBackend` turns a task id and a prompt into ``n`` completions.
Two implementations ship:

- :class:`MockBackend` replays canned completions keyed by task id, in order,
  cycling when more are asked for than exist. It is deterministic: with the
  same canned file, seed and failure rate it produces the same samples on
  every run. ``failure_rate`` swaps that fraction of completions for a stub
  that raises ``NotImplementedError``, drawn from a generator seeded per task
  from ``seed`` so the outcome does not depend on the order of calls.
  :meth:`MockBackend.from_jsonl` reads the ``{"task_id", "completion"}``
  lines the upstream evaluator also consumes.
- :class:`OpenAIBackend` posts to any OpenAI-compatible chat completions
  endpoint with the standard library only (no SDK, no extra dependency).
  Base URL, model, key, timeout and retry count come from the settings
  (``VERIFYBENCH_OPENAI_*``); it is never selected unless asked for and
  refuses to start without a key. Transport failures, HTTP 429 and 5xx are
  retried with an exponential backoff; anything else is a
  :class:`~codeeval.errors.ProviderError` naming the status and the URL.

:func:`generate_samples` drives any backend over a list of prompts and
returns sample records; :func:`write_samples` stores them as JSONL for
``evaluate_functional_correctness``. :func:`make_backend` builds a backend
by name from the settings, which is what the CLI and the demo use.
"""

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from codeeval.errors import ConfigError, DataError, ProviderError
from codeeval.settings import Settings, get_settings
from codeeval.tasks import TaskSuite

log = logging.getLogger(__name__)

PathLike = str | os.PathLike[str]
BackendName = Literal["mock", "openai"]
BACKEND_NAMES: tuple[BackendName, ...] = ("mock", "openai")

#: What the mock returns for a completion it decides to fail.
STUB_COMPLETION = "    raise NotImplementedError\n"

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
DEFAULT_SYSTEM_PROMPT = (
    "You complete Python functions. Reply with the function body only, "
    "indented to continue the given code, with no explanation and no code fences."
)


@runtime_checkable
class ModelBackend(Protocol):
    """Anything that produces completions for a prompt."""

    @property
    def name(self) -> str:
        """Short backend name recorded on every sample (``mock``, ``openai``)."""

    def complete(self, task_id: str, prompt: str, *, n: int = 1) -> list[str]:
        """``n`` completions for ``prompt``; raises ``ProviderError`` when it cannot."""


@dataclass(frozen=True)
class Sample:
    """One generated completion, as written to ``samples.jsonl``."""

    task_id: str
    completion: str
    backend: str
    index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "completion": self.completion,
            "backend": self.backend,
            "index": self.index,
        }


@dataclass
class MockBackend:
    """Replays canned completions per task id; see the module docstring."""

    canned: Mapping[str, Sequence[str]]
    seed: int = 0
    failure_rate: float = 0.0
    name: str = field(default="mock", init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_rate <= 1.0:
            raise ConfigError(
                f"failure_rate must be between 0 and 1, got {self.failure_rate}",
                details={"failure_rate": self.failure_rate},
            )
        self.canned = {task_id: tuple(items) for task_id, items in self.canned.items()}
        empty = sorted(task_id for task_id, items in self.canned.items() if not items)
        if empty:
            raise ConfigError(
                f"no canned completion for {len(empty)} task(s): {', '.join(empty[:5])}",
                details={"task_ids": empty},
            )

    @classmethod
    def from_jsonl(
        cls, path: PathLike, *, seed: int = 0, failure_rate: float = 0.0
    ) -> "MockBackend":
        """A mock over the ``{"task_id", "completion"}`` lines of ``path``, in file order."""
        canned: dict[str, list[str]] = {}
        for number, record in enumerate(_read_records(path), start=1):
            task_id, completion = record.get("task_id"), record.get("completion")
            if not isinstance(task_id, str) or not isinstance(completion, str):
                raise DataError(
                    f"{os.fspath(path)}:{number}: expected string task_id and completion",
                    details={"path": os.fspath(path), "line": number},
                )
            canned.setdefault(task_id, []).append(completion)
        if not canned:
            raise DataError(f"{os.fspath(path)} holds no completions", details={"path": path})
        return cls(canned, seed=seed, failure_rate=failure_rate)

    @classmethod
    def from_pairs(
        cls, pairs: Iterable[tuple[str, str]], *, seed: int = 0, failure_rate: float = 0.0
    ) -> "MockBackend":
        """A mock over ``(task_id, completion)`` pairs, in order."""
        canned: dict[str, list[str]] = {}
        for task_id, completion in pairs:
            canned.setdefault(task_id, []).append(completion)
        return cls(canned, seed=seed, failure_rate=failure_rate)

    def complete(self, task_id: str, prompt: str, *, n: int = 1) -> list[str]:
        if n < 1:
            raise ValueError("n must be at least 1")
        try:
            canned = self.canned[task_id]
        except KeyError:
            raise ProviderError(
                f"mock backend has no canned completion for {task_id}",
                details={"task_id": task_id, "known": len(self.canned)},
            ) from None
        rng = random.Random(f"{self.seed}:{task_id}")
        out: list[str] = []
        for index in range(n):
            completion = canned[index % len(canned)]
            if self.failure_rate > 0 and rng.random() < self.failure_rate:
                completion = STUB_COMPLETION
            out.append(completion)
        return out


def _read_records(path: PathLike) -> Iterable[dict[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(
            f"cannot read {os.fspath(path)}: {exc.strerror}", details={"path": os.fspath(path)}
        ) from exc
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(
                f"{os.fspath(path)}:{number}: invalid JSON: {exc.msg}",
                details={"path": os.fspath(path), "line": number},
            ) from None
        if not isinstance(record, dict):
            raise DataError(
                f"{os.fspath(path)}:{number}: expected a JSON object",
                details={"path": os.fspath(path), "line": number},
            )
        yield record


Sleep = Callable[[float], None]


@dataclass
class OpenAIBackend:
    """Chat completions over HTTP against an OpenAI-compatible server."""

    base_url: str
    model: str
    api_key: str
    timeout: float = 60.0
    retries: int = 2
    backoff: float = 0.5
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    sleep: Sleep = time.sleep
    name: str = field(default="openai", init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigError(
                "the openai backend needs an API key (VERIFYBENCH_OPENAI_API_KEY)",
                details={"env": Settings.env_name("openai_api_key")},
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"openai base URL must start with http:// or https://, got {self.base_url!r}",
                details={"env": Settings.env_name("openai_base_url")},
            )
        if self.retries < 0 or self.timeout <= 0:
            raise ConfigError("retries must be >= 0 and timeout > 0", details={})
        self.base_url = self.base_url.rstrip("/")

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(self, task_id: str, prompt: str, *, n: int = 1) -> list[str]:
        if n < 1:
            raise ValueError("n must be at least 1")
        payload = {
            "model": self.model,
            "n": n,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        body = self._post(payload, task_id=task_id)
        return self._completions(body, task_id=task_id, n=n)

    def _post(self, payload: dict[str, Any], *, task_id: str) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last: str = "no attempt made"
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                status, raw = exc.code, exc.read()
                if status not in RETRY_STATUSES:
                    raise ProviderError(
                        f"{self.url} answered HTTP {status} for {task_id}",
                        details={
                            "status": status,
                            "url": self.url,
                            "body": raw[:500].decode("utf-8", "replace"),
                        },
                    ) from None
                last = f"HTTP {status}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
            else:
                return self._decode(raw, task_id=task_id, status=status)
            log.warning(
                "openai backend retrying",
                extra={"task_id": task_id, "attempt": attempt + 1, "reason": last},
            )
            if attempt < self.retries:
                self.sleep(self.backoff * 2**attempt)
        raise ProviderError(
            f"{self.url} failed after {self.retries + 1} attempt(s) for {task_id}: {last}",
            details={"url": self.url, "attempts": self.retries + 1, "reason": last},
        )

    def _decode(self, raw: bytes, *, task_id: str, status: int) -> dict[str, Any]:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.url} returned a non-JSON body for {task_id}",
                details={"status": status, "url": self.url, "error": exc.msg},
            ) from None
        if not isinstance(body, dict):
            raise ProviderError(
                f"{self.url} returned a non-object body for {task_id}",
                details={"status": status, "url": self.url},
            )
        return body

    def _completions(self, body: dict[str, Any], *, task_id: str, n: int) -> list[str]:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                f"{self.url} returned no choices for {task_id}",
                details={"url": self.url, "keys": sorted(body)},
            )
        out: list[str] = []
        for choice in choices:
            content = choice.get("message", {}).get("content") if isinstance(choice, dict) else None
            if not isinstance(content, str):
                raise ProviderError(
                    f"{self.url} returned a choice without message.content for {task_id}",
                    details={"url": self.url},
                )
            out.append(content)
        if len(out) < n:
            log.warning(
                "openai backend returned fewer completions than asked",
                extra={"task_id": task_id, "asked": n, "got": len(out)},
            )
        return out[:n]


def reference_completions(suite: TaskSuite) -> list[tuple[str, str]]:
    """``(task_id, completion)`` per task: the reference solution minus the prompt.

    A reference module that starts with the prompt (every converted HumanEval
    task does) yields the body that follows it, which is exactly what the
    upstream evaluator appends to the prompt again; any other reference is
    used whole. Feeding these to the mock gives a "perfect model".
    """
    pairs: list[tuple[str, str]] = []
    for task in suite.tasks:
        reference = task.reference_solution
        body = reference[len(task.prompt) :] if reference.startswith(task.prompt) else reference
        pairs.append((task.task_id, body))
    return pairs


def make_backend(
    name: str,
    *,
    settings: Settings | None = None,
    canned: PathLike | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    failure_rate: float | None = None,
    seed: int | None = None,
) -> ModelBackend:
    """A backend by name from the settings.

    ``mock`` replays ``canned`` (a samples JSONL file) or, failing that,
    ``pairs`` of ``(task_id, completion)``; ``openai`` needs
    ``VERIFYBENCH_OPENAI_API_KEY``. ``failure_rate`` and ``seed`` override the
    settings for the mock.
    """
    settings = get_settings() if settings is None else settings
    if name == "mock":
        mock_seed = settings.seed if seed is None else seed
        rate = settings.mock_failure_rate if failure_rate is None else failure_rate
        if canned is not None:
            return MockBackend.from_jsonl(canned, seed=mock_seed, failure_rate=rate)
        if pairs is not None:
            return MockBackend.from_pairs(pairs, seed=mock_seed, failure_rate=rate)
        raise ConfigError(
            "the mock backend needs a canned completions file or reference completions",
            details={"backend": name},
        )
    if name == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        return OpenAIBackend(
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=key,
            timeout=settings.openai_timeout,
            retries=settings.openai_retries,
        )
    raise ConfigError(
        f"unknown model backend {name!r}; choose one of {', '.join(BACKEND_NAMES)}",
        details={"backend": name, "choices": list(BACKEND_NAMES)},
    )


def generate_samples(
    backend: ModelBackend, prompts: Iterable[tuple[str, str]], *, n: int = 1
) -> list[Sample]:
    """``n`` samples per ``(task_id, prompt)`` from ``backend``, in prompt order."""
    if n < 1:
        raise ValueError("n must be at least 1")
    samples: list[Sample] = []
    started = time.monotonic()
    for task_id, prompt in prompts:
        completions = backend.complete(task_id, prompt, n=n)
        samples.extend(
            Sample(task_id=task_id, completion=completion, backend=backend.name, index=index)
            for index, completion in enumerate(completions)
        )
    log.info(
        "samples generated",
        extra={
            "backend": backend.name,
            "n": n,
            "n_samples": len(samples),
            "seconds": round(time.monotonic() - started, 3),
        },
    )
    return samples


def write_samples(path: PathLike, samples: Iterable[Sample]) -> int:
    """Write ``samples`` as JSONL to ``path`` (parent created); returns the count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_SYSTEM_PROMPT",
    "RETRY_STATUSES",
    "STUB_COMPLETION",
    "BackendName",
    "MockBackend",
    "ModelBackend",
    "OpenAIBackend",
    "Sample",
    "generate_samples",
    "make_backend",
    "reference_completions",
    "write_samples",
]
