"""Typed configuration for the harness, read from the environment and ``.env``.

Every option is a field of :class:`Settings`. Each one can be set with an
environment variable named ``VERIFYBENCH_<FIELD>`` (case-insensitive) or with
the same key in a ``.env`` file in the current working directory. Precedence,
highest first:

1. keyword arguments to :func:`load_settings` (or :func:`override_settings`)
2. environment variables
3. the ``.env`` file
4. the defaults declared on the fields

The defaults reproduce upstream ``evaluate_functional_correctness`` behaviour
(4 workers, 3 seconds per sample), so an unconfigured checkout behaves exactly
like upstream.

:func:`get_settings` loads the settings once per process and caches them.
:func:`override_settings` makes :func:`get_settings` return a different
instance inside a ``with`` block, which is how tests pin a configuration
without touching the process environment.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from codeeval.errors import ConfigError

ENV_PREFIX = "VERIFYBENCH_"
DEFAULT_ENV_FILE = ".env"

SandboxBackend = Literal["subprocess", "docker"]
LogFormat = Literal["json", "text"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ModelBackendName = Literal["mock", "openai"]


class Settings(BaseSettings):
    """Harness configuration; see the module docstring for where values come from."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        # An empty VERIFYBENCH_X= counts as unset instead of failing validation.
        env_ignore_empty=True,
        # A .env also holds keys that are not settings (provider API keys).
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    sandbox_backend: SandboxBackend = Field(
        default="subprocess",
        description="Where generated code runs: a fresh local interpreter or a Docker container.",
    )
    sample_timeout: float = Field(
        default=3.0,
        gt=0,
        description="Seconds one sample may run before it is graded as timed out.",
    )
    workers: int = Field(default=4, ge=1, description="Samples executed concurrently.")
    docker_image: str = Field(
        default="python:3.12-slim",
        description="Image reference (tag or digest) the docker backend runs samples in.",
    )
    results_dir: Path = Field(
        default=Path("results"), description="Directory that receives run outputs."
    )
    db_path: Path = Field(
        default=Path("results/verifybench.sqlite"), description="SQLite results store."
    )
    log_format: LogFormat = Field(default="json", description="Log record format: json or text.")
    log_level: LogLevel = Field(default="INFO", description="Least severe log level emitted.")
    seed: int = Field(default=0, description="Seed for everything randomised (sampling, stubs).")
    model_backend: ModelBackendName = Field(
        default="mock", description="Where completions come from: the offline mock or openai."
    )
    mock_failure_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of mock completions replaced by a NotImplementedError stub.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of an OpenAI-compatible chat completions API.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini", description="Model name sent to the OpenAI-compatible API."
    )
    openai_api_key: SecretStr | None = Field(
        default=None, description="API key; the openai backend refuses to start without one."
    )
    openai_timeout: float = Field(
        default=60.0, gt=0, description="Seconds one HTTP request to the model API may take."
    )
    openai_retries: int = Field(
        default=2, ge=0, description="Retries after a transport error, HTTP 429 or 5xx."
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("docker_image")
    @classmethod
    def single_image_reference(cls, value: str) -> str:
        value = value.strip()
        if not value or any(char.isspace() for char in value):
            raise ValueError(
                "must be one image reference such as python:3.12-slim or name@sha256:<digest>"
            )
        return value

    @staticmethod
    def env_name(field: str) -> str:
        """The environment variable that sets ``field``, e.g. ``VERIFYBENCH_WORKERS``."""
        return f"{ENV_PREFIX}{field.upper()}"


def _describe(error: dict[str, Any]) -> dict[str, Any]:
    field = ".".join(str(part) for part in error["loc"]) or "<root>"
    return {
        "field": field,
        "env": Settings.env_name(field),
        "message": error["msg"],
        "value": error.get("input"),
    }


def load_settings(**overrides: Any) -> Settings:
    """Build a fresh :class:`Settings` from ``overrides``, the environment and ``.env``.

    ``overrides`` take precedence over every other source. Bad values raise
    :class:`ConfigError`; ``details["errors"]`` lists one entry per offending
    field with the environment variable that sets it.
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        errors = [_describe(dict(error)) for error in exc.errors(include_url=False)]
        summary = "; ".join(f"{error['env']}: {error['message']}" for error in errors)
        raise ConfigError(f"invalid settings: {summary}", details={"errors": errors}) from exc


_override: Settings | None = None


@cache
def _load_cached() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    """The process-wide settings: loaded on first use, then cached.

    An active :func:`override_settings` block wins over the cache. A failed
    load is not cached, so the next call retries once the environment is fixed.
    """
    if _override is not None:
        return _override
    return _load_cached()


def clear_settings_cache() -> None:
    """Drop the cached settings so the next :func:`get_settings` reloads them."""
    _load_cached.cache_clear()


@contextmanager
def override_settings(settings: Settings | None = None, /, **changes: Any) -> Iterator[Settings]:
    """Make :func:`get_settings` return other settings inside the ``with`` block.

    Pass a ready :class:`Settings`, or keyword changes that are applied on top
    of the current settings and validated like any other source (a bad value
    raises :class:`ConfigError` before the block runs). Blocks nest; leaving
    one restores whatever was active before it.
    """
    global _override
    if settings is None:
        settings = load_settings(**{**get_settings().model_dump(), **changes})
    elif changes:
        raise TypeError("pass either a Settings instance or keyword changes, not both")
    previous = _override
    _override = settings
    try:
        yield settings
    finally:
        _override = previous


__all__ = [
    "DEFAULT_ENV_FILE",
    "ENV_PREFIX",
    "LogFormat",
    "LogLevel",
    "ModelBackendName",
    "SandboxBackend",
    "Settings",
    "clear_settings_cache",
    "get_settings",
    "load_settings",
    "override_settings",
]
