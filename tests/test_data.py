"""Tests for the jsonl helpers and the bundled HumanEval dataset."""

import gzip
from pathlib import Path
from typing import Any

import pytest

from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl

PROBLEM_KEYS = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}


def test_bundled_humaneval_dataset_loads() -> None:
    problems = read_problems()
    assert len(problems) == 164
    assert set(problems) == {f"HumanEval/{i}" for i in range(164)}
    for task_id, problem in problems.items():
        assert problem.keys() >= PROBLEM_KEYS
        assert problem["task_id"] == task_id
        assert f"def {problem['entry_point']}(" in problem["prompt"]


def test_default_dataset_path_exists() -> None:
    assert Path(HUMAN_EVAL).is_file()


def test_read_problems_accepts_path_objects(example_problem_file: Path) -> None:
    assert read_problems(example_problem_file) == read_problems(str(example_problem_file))


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_write_then_stream_round_trip(tmp_path: Path, suffix: str) -> None:
    rows: list[dict[str, Any]] = [
        {"task_id": "a", "n": 1},
        {"task_id": "b", "text": "tab\tand\nnewline"},
    ]
    path = tmp_path / f"rows{suffix}"

    write_jsonl(path, rows)

    assert list(stream_jsonl(path)) == rows
    assert list(stream_jsonl(str(path))) == rows


def test_gzip_output_is_really_compressed(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"
    write_jsonl(path, [{"x": 1}])
    with gzip.open(path, "rt") as fp:
        assert fp.read() == '{"x": 1}\n'


def test_append_mode_keeps_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"x": 1}])
    write_jsonl(path, [{"x": 2}], append=True)
    assert list(stream_jsonl(path)) == [{"x": 1}, {"x": 2}]


def test_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"x": 1}, {"x": 2}])
    write_jsonl(path, [{"x": 3}])
    assert list(stream_jsonl(path)) == [{"x": 3}]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"x": 1}\n\n   \n\t\n{"x": 2}\n')
    assert list(stream_jsonl(path)) == [{"x": 1}, {"x": 2}]


def test_write_expands_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write_jsonl("~/rows.jsonl", [{"x": 1}])
    assert list(stream_jsonl(tmp_path / "rows.jsonl")) == [{"x": 1}]
