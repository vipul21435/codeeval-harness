"""Tests for codeeval.settings: sources and precedence, validation, caching, overrides."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from codeeval.errors import ConfigError
from codeeval.settings import (
    ENV_PREFIX,
    Settings,
    clear_settings_cache,
    get_settings,
    load_settings,
    override_settings,
)
from human_eval import evaluation

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_an_empty_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The .env file is looked up in the working directory: start without one."""
    monkeypatch.chdir(tmp_path)


def write_dotenv(directory: Path, **values: str) -> Path:
    path = directory / ".env"
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return path


# --- sources and precedence -------------------------------------------------


def test_defaults_reproduce_the_upstream_evaluator() -> None:
    settings = load_settings()
    assert settings.model_dump() == {
        "sandbox_backend": "subprocess",
        "sample_timeout": 3.0,
        "workers": 4,
        "docker_image": "python:3.12-slim",
        "results_dir": Path("results"),
        "db_path": Path("results/verifybench.sqlite"),
        "log_format": "json",
        "log_level": "INFO",
        "seed": 0,
    }


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "python@sha256:" + "0" * 64
    monkeypatch.setenv("VERIFYBENCH_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("VERIFYBENCH_SAMPLE_TIMEOUT", "0.5")
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "16")
    monkeypatch.setenv("VERIFYBENCH_DOCKER_IMAGE", digest)
    monkeypatch.setenv("VERIFYBENCH_RESULTS_DIR", "/srv/out")
    monkeypatch.setenv("VERIFYBENCH_DB_PATH", "/srv/out/runs.sqlite")
    monkeypatch.setenv("VERIFYBENCH_LOG_FORMAT", "text")
    monkeypatch.setenv("VERIFYBENCH_LOG_LEVEL", "debug")
    monkeypatch.setenv("VERIFYBENCH_SEED", "42")

    settings = load_settings()

    assert settings.model_dump() == {
        "sandbox_backend": "docker",
        "sample_timeout": 0.5,
        "workers": 16,
        "docker_image": digest,
        "results_dir": Path("/srv/out"),
        "db_path": Path("/srv/out/runs.sqlite"),
        "log_format": "text",
        "log_level": "DEBUG",
        "seed": 42,
    }


def test_dotenv_overrides_defaults(tmp_path: Path) -> None:
    write_dotenv(tmp_path, VERIFYBENCH_WORKERS="2", VERIFYBENCH_LOG_FORMAT="text")
    settings = load_settings()
    assert settings.workers == 2
    assert settings.log_format == "text"
    assert settings.seed == 0


def test_environment_beats_dotenv_beats_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_dotenv(tmp_path, VERIFYBENCH_WORKERS="2", VERIFYBENCH_SAMPLE_TIMEOUT="7.5")
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "9")

    settings = load_settings()

    assert settings.workers == 9  # environment beats .env
    assert settings.sample_timeout == 7.5  # .env beats the default
    assert settings.seed == 0  # default


def test_explicit_overrides_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "9")
    assert load_settings(workers=1).workers == 1


def test_dotenv_is_read_from_the_working_directory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_dotenv(tmp_path, VERIFYBENCH_WORKERS="2")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert load_settings().workers == 4


def test_dotenv_may_hold_comments_and_unrelated_keys(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# provider credentials live next to the settings\n"
        "OPENAI_API_KEY=sk-not-a-setting\n"
        "\n"
        "VERIFYBENCH_SEED=7\n"
    )
    assert load_settings().seed == 7


def test_empty_environment_variable_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "")
    assert load_settings().workers == 4


def test_environment_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("verifybench_seed", "3")
    assert load_settings().seed == 3


def test_env_name_derives_from_the_field_name() -> None:
    assert ENV_PREFIX == "VERIFYBENCH_"
    assert Settings.env_name("sample_timeout") == "VERIFYBENCH_SAMPLE_TIMEOUT"


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VERIFYBENCH_SANDBOX_BACKEND", "firecracker"),
        ("VERIFYBENCH_SAMPLE_TIMEOUT", "0"),
        ("VERIFYBENCH_SAMPLE_TIMEOUT", "fast"),
        ("VERIFYBENCH_WORKERS", "0"),
        ("VERIFYBENCH_WORKERS", "2.5"),
        ("VERIFYBENCH_LOG_FORMAT", "yaml"),
        ("VERIFYBENCH_LOG_LEVEL", "LOUD"),
        ("VERIFYBENCH_SEED", "random"),
        ("VERIFYBENCH_DOCKER_IMAGE", "python:3.12 slim"),
        ("VERIFYBENCH_DOCKER_IMAGE", "   "),
    ],
)
def test_bad_values_raise_config_error_naming_the_variable(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError) as info:
        load_settings()

    error = info.value
    assert str(error).startswith("invalid settings: ")
    assert name in str(error)
    (problem,) = error.details["errors"]
    assert problem["env"] == name
    assert problem["field"] == name.removeprefix(ENV_PREFIX).lower()
    assert problem["value"] == value
    assert isinstance(error.__cause__, ValidationError)


def test_config_error_lists_every_bad_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "0")
    monkeypatch.setenv("VERIFYBENCH_SEED", "x")

    with pytest.raises(ConfigError) as info:
        load_settings()

    assert {problem["env"] for problem in info.value.details["errors"]} == {
        "VERIFYBENCH_WORKERS",
        "VERIFYBENCH_SEED",
    }


def test_log_level_is_normalised_to_upper_case() -> None:
    assert load_settings(log_level="warning").log_level == "WARNING"


def test_docker_image_is_stripped() -> None:
    assert load_settings(docker_image=" python:3.12-slim\n").docker_image == "python:3.12-slim"


def test_settings_are_frozen() -> None:
    settings = load_settings()
    with pytest.raises(ValidationError, match="frozen"):
        settings.workers = 1


# --- caching ----------------------------------------------------------------


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "9")
    assert get_settings() is first
    assert get_settings().workers == 4


def test_clear_settings_cache_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_settings().workers == 4
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "9")
    clear_settings_cache()
    assert get_settings().workers == 9


def test_failed_load_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "0")
    with pytest.raises(ConfigError):
        get_settings()
    monkeypatch.setenv("VERIFYBENCH_WORKERS", "3")
    assert get_settings().workers == 3


# --- overrides --------------------------------------------------------------


def test_override_settings_applies_changes_and_restores() -> None:
    original = get_settings()

    with override_settings(workers=1, sample_timeout=0.25) as active:
        assert get_settings() is active
        assert active.workers == 1
        assert active.sample_timeout == 0.25
        assert active.seed == original.seed

    assert get_settings() is original


def test_override_settings_nests() -> None:
    with override_settings(workers=2):
        with override_settings(seed=5) as inner:
            assert get_settings() is inner
            assert (inner.workers, inner.seed) == (2, 5)
        assert get_settings().workers == 2
        assert get_settings().seed == 0
    assert get_settings().workers == 4


def test_override_settings_validates_changes() -> None:
    with pytest.raises(ConfigError, match="VERIFYBENCH_WORKERS"), override_settings(workers=0):
        pytest.fail("the block must not run")
    assert get_settings().workers == 4


def test_override_settings_accepts_an_instance() -> None:
    pinned = load_settings(seed=99)
    with override_settings(pinned):
        assert get_settings() is pinned


def test_override_settings_rejects_instance_plus_changes() -> None:
    with pytest.raises(TypeError), override_settings(load_settings(), seed=1):
        pytest.fail("the block must not run")


def test_override_is_visible_from_worker_threads() -> None:
    with override_settings(workers=1) as active, ThreadPoolExecutor(max_workers=2) as pool:
        seen = list(pool.map(lambda _: get_settings(), range(4)))
    assert all(settings is active for settings in seen)


def test_override_settings_restores_after_an_exception() -> None:
    original = get_settings()
    with pytest.raises(RuntimeError), override_settings(seed=1):
        raise RuntimeError("boom")
    assert get_settings() is original


# --- .env.example stays in sync with the code -------------------------------


def _documented_settings() -> dict[str, str]:
    documented: dict[str, str] = {}
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        if line.startswith(f"# {ENV_PREFIX}"):
            key, _, value = line.removeprefix("# ").partition("=")
            documented[key.removeprefix(ENV_PREFIX).lower()] = value
    return documented


def test_env_example_documents_every_setting() -> None:
    assert set(_documented_settings()) == set(Settings.model_fields)


def test_env_example_values_are_the_defaults(tmp_path: Path) -> None:
    write_dotenv(
        tmp_path, **{Settings.env_name(key): value for key, value in _documented_settings().items()}
    )
    from_example = load_settings().model_dump()
    from_code = load_settings(_env_file=None).model_dump()
    assert from_example == from_code


# --- the upstream evaluator reads its defaults from the settings ------------


def _stub_evaluation(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the worker pool and the sample runner; record what they were given."""
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


def test_evaluate_reads_workers_and_timeout_from_settings(
    monkeypatch: pytest.MonkeyPatch, example_problem_file: Path, example_samples_file: Path
) -> None:
    seen = _stub_evaluation(monkeypatch)

    with override_settings(workers=7, sample_timeout=0.25):
        pass_at_k = evaluation.evaluate_functional_correctness(
            example_samples_file, k=[1], problem_file=example_problem_file
        )

    assert seen == {"workers": 7, "timeout": 0.25}
    assert pass_at_k == {"pass@1": 1.0}


def test_explicit_arguments_beat_settings(
    monkeypatch: pytest.MonkeyPatch, example_problem_file: Path, example_samples_file: Path
) -> None:
    seen = _stub_evaluation(monkeypatch)

    with override_settings(workers=7, sample_timeout=0.25):
        evaluation.evaluate_functional_correctness(
            example_samples_file, k=[1], n_workers=2, timeout=1.5, problem_file=example_problem_file
        )

    assert seen == {"workers": 2, "timeout": 1.5}
