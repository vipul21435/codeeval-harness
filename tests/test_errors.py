"""Tests for the codeeval error hierarchy."""

import pickle

import pytest

from codeeval.errors import (
    CodeEvalError,
    ConfigError,
    DataError,
    ExecutionError,
    GradingError,
    ProviderError,
    StorageError,
)

STAGE_ERRORS = [ConfigError, DataError, ProviderError, ExecutionError, GradingError, StorageError]


@pytest.mark.parametrize("error_type", STAGE_ERRORS)
def test_every_stage_error_is_a_codeeval_error(error_type: type[CodeEvalError]) -> None:
    assert issubclass(error_type, CodeEvalError)
    assert issubclass(error_type, Exception)
    assert not issubclass(error_type, KeyError | ValueError | OSError)


def test_stage_errors_are_siblings_not_a_chain() -> None:
    # Catching one stage must never swallow another.
    for outer in STAGE_ERRORS:
        for inner in STAGE_ERRORS:
            assert issubclass(inner, outer) == (inner is outer)


def test_one_except_clause_catches_everything() -> None:
    with pytest.raises(CodeEvalError):
        raise StorageError("disk full")


def test_str_is_the_message_and_details_default_to_empty() -> None:
    error = CodeEvalError("boom")
    assert str(error) == "boom"
    assert error.message == "boom"
    assert error.args == ("boom",)
    assert dict(error.details) == {}
    assert repr(error) == "CodeEvalError('boom')"


def test_details_are_read_only_and_copied() -> None:
    details = {"path": "tasks.jsonl", "record": 3}
    error = DataError("bad record", details=details)
    details["record"] = 99  # the caller mutating its dict must not leak in

    assert error.details["record"] == 3
    with pytest.raises(TypeError):
        error.details["record"] = 4  # type: ignore[index]
    assert repr(error) == "DataError('bad record', details={'path': 'tasks.jsonl', 'record': 3})"


@pytest.mark.parametrize("error_type", [CodeEvalError, *STAGE_ERRORS])
def test_pickle_round_trip_keeps_type_message_and_details(
    error_type: type[CodeEvalError],
) -> None:
    error = error_type("nope", details={"env": "CODEEVAL_API__PORT", "value": 0})
    clone = pickle.loads(pickle.dumps(error))
    assert type(clone) is error_type
    assert str(clone) == "nope"
    assert dict(clone.details) == {"env": "CODEEVAL_API__PORT", "value": 0}
