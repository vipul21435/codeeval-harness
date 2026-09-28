"""Tests for codeeval.tasks: the task schema and JSONL suite files."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from codeeval.errors import DataError
from codeeval.tasks import (
    TASK_ID_PATTERN,
    Task,
    TaskSuite,
    parse_task_line,
    read_tasks,
    write_tasks,
)


def errors_of(exc: ValidationError) -> list[tuple[str, str]]:
    return [(".".join(map(str, e["loc"])), e["msg"]) for e in exc.errors(include_url=False)]


def test_valid_task_round_trips_through_a_dict(
    make_task: Callable[..., Task], task_record: dict[str, Any]
) -> None:
    task = make_task()
    assert task.task_id == "synthetic/add"
    assert task.language == "python"
    assert task.difficulty == "easy"
    assert task.model_dump() == task_record
    assert Task.model_validate(task.model_dump()) == task


def test_language_and_difficulty_have_defaults(task_record: dict[str, Any]) -> None:
    for field in ("language", "difficulty", "metadata"):
        del task_record[field]
    task = Task.model_validate(task_record)
    assert (task.language, task.difficulty, task.metadata) == ("python", "medium", {})


def test_task_is_frozen(make_task: Callable[..., Task]) -> None:
    task = make_task()
    with pytest.raises(ValidationError):
        task.prompt = "changed"
    assert task.prompt.startswith("def add")


def test_unknown_fields_are_rejected(make_task: Callable[..., Task]) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(entry_points="add")
    assert ("entry_points", "Extra inputs are not permitted") in errors_of(info.value)


def test_strict_types_reject_coercion(make_task: Callable[..., Task]) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(prompt=42)
    assert [loc for loc, _ in errors_of(info.value)] == ["prompt"]


@pytest.mark.parametrize(
    "task_id",
    ["HumanEval/0", "test/0", "my-suite/fizz_buzz.v2", "a/b", "A1/B2"],
)
def test_task_id_pattern_accepts_namespaced_ids(
    make_task: Callable[..., Task], task_id: str
) -> None:
    assert make_task(task_id=task_id).task_id == task_id


@pytest.mark.parametrize(
    "task_id",
    ["", "HumanEval", "/0", "HumanEval/", "a/b/c", "a b/c", "a/b c", "-a/b", "a/.b", "a\n/b"],
)
def test_task_id_pattern_rejects_malformed_ids(
    make_task: Callable[..., Task], task_id: str
) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(task_id=task_id)
    (loc, msg), *rest = errors_of(info.value)
    assert (loc, rest) == ("task_id", [])
    assert TASK_ID_PATTERN in msg


@pytest.mark.parametrize("entry_point", ["", "add two", "2add", "class", "add()", "a.b"])
def test_entry_point_must_be_an_identifier(
    make_task: Callable[..., Task], entry_point: str
) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(entry_point=entry_point)
    assert errors_of(info.value) == [
        ("entry_point", f"Value error, {entry_point!r} is not a valid Python identifier")
    ]


@pytest.mark.parametrize("field", ["reference_solution", "baseline_solution"])
def test_solutions_must_parse(make_task: Callable[..., Task], field: str) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(**{field: "def add(a, b):\n    return (\n"})
    [(loc, msg)] = errors_of(info.value)
    assert loc == field
    assert "solution is not valid Python" in msg
    assert "(line 2)" in msg or "(line 3)" in msg


@pytest.mark.parametrize("field", ["prompt", "reference_solution", "baseline_solution", "tests"])
def test_text_fields_must_not_be_empty(make_task: Callable[..., Task], field: str) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(**{field: ""})
    assert [loc for loc, _ in errors_of(info.value)] == [field]


@pytest.mark.parametrize("field", ["reference_solution", "baseline_solution"])
def test_solutions_must_define_the_entry_point(make_task: Callable[..., Task], field: str) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(**{field: "def plus(a, b):\n    return a + b\n"})
    assert errors_of(info.value) == [
        ("", f"Value error, {field} does not define the entry point 'add' at module level")
    ]


@pytest.mark.parametrize(
    "source",
    [
        "add = lambda a, b: a + b\n",
        "add: object = None\n",
        "from operator import add\n",
        "from operator import mul as add\n",
        "class add:\n    pass\n",
        "async def add(a, b):\n    return a + b\n",
    ],
)
def test_entry_point_may_be_bound_by_any_module_level_statement(
    make_task: Callable[..., Task], source: str
) -> None:
    assert make_task(reference_solution=source).reference_solution == source


def test_baseline_must_differ_from_reference(
    make_task: Callable[..., Task], task_record: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(baseline_solution=task_record["reference_solution"])
    assert errors_of(info.value) == [
        ("", "Value error, baseline_solution is identical to reference_solution")
    ]


@pytest.mark.parametrize(
    ("tests", "message"),
    [
        ("from solution import add\n\ndef check():\n    assert add(1, 1) == 2\n", "no function"),
        ("def test_add():\n    assert True\n", "never import the 'solution' module"),
        ("from solution import add\n\ndef test_add():\n    assert add(1, 1) == 2\n  x\n", "valid"),
        ("import solutions\n\ndef test_add():\n    assert True\n", "never import"),
    ],
)
def test_tests_must_be_a_pytest_module_using_the_solution(
    make_task: Callable[..., Task], tests: str, message: str
) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(tests=tests)
    [(loc, msg)] = errors_of(info.value)
    assert loc == "tests"
    assert message in msg


@pytest.mark.parametrize(
    "tests",
    [
        "import solution\n\ndef test_add():\n    assert solution.add(1, 1) == 2\n",
        "import solution.helpers\n\ndef testadd():\n    assert True\n",
        (
            "from solution import add\n\n"
            "class TestAdd:\n    def test_two(self):\n        assert add(1, 1) == 2\n"
        ),
        "from solution import add\n\nasync def test_add():\n    assert add(1, 1) == 2\n",
    ],
)
def test_tests_accept_pytest_collection_styles(make_task: Callable[..., Task], tests: str) -> None:
    assert make_task(tests=tests).tests == tests


def test_language_and_difficulty_are_closed_sets(make_task: Callable[..., Task]) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(language="rust", difficulty="brutal")
    assert [loc for loc, _ in errors_of(info.value)] == ["language", "difficulty"]


def test_metadata_must_be_json(make_task: Callable[..., Task]) -> None:
    with pytest.raises(ValidationError) as info:
        make_task(metadata={"tags": {"a", "b"}})
    assert all(loc.startswith("metadata.tags") for loc, _ in errors_of(info.value))
    assert make_task(metadata={"n": 1, "nested": {"list": [1.5, None, True]}}).metadata == {
        "n": 1,
        "nested": {"list": [1.5, None, True]},
    }


def test_to_json_line_is_one_stable_line(
    make_task: Callable[..., Task], task_record: dict[str, Any]
) -> None:
    line = make_task(metadata={"unicode": "caf\u00e9"}).to_json_line()
    assert "\n" not in line
    assert line.isascii()
    assert json.loads(line)["metadata"] == {"unicode": "caf\u00e9"}
    assert list(json.loads(line)) == list(task_record)


# --- TaskSuite ---------------------------------------------------------------


def test_suite_rejects_duplicate_ids(make_task: Callable[..., Task]) -> None:
    other = make_task(prompt="def add(a, b):\n")
    with pytest.raises(ValidationError) as info:
        TaskSuite(tasks=[make_task(), make_task(task_id="synthetic/sub"), other])
    assert errors_of(info.value) == [
        ("tasks", "Value error, duplicate task_id 'synthetic/add' at positions 0 and 2")
    ]


def test_suite_lookup(make_task: Callable[..., Task]) -> None:
    suite = TaskSuite(tasks=[make_task(), make_task(task_id="synthetic/sub")])
    assert len(suite) == 2
    assert suite.ids == ["synthetic/add", "synthetic/sub"]
    assert suite.get("synthetic/sub").task_id == "synthetic/sub"
    with pytest.raises(KeyError):
        suite.get("synthetic/missing")
    assert len(TaskSuite()) == 0


# --- JSONL read and write -----------------------------------------------------


def test_write_then_read_round_trips(tmp_path: Path, make_task: Callable[..., Task]) -> None:
    tasks = [make_task(), make_task(task_id="synthetic/sub", difficulty="hard")]
    path = tmp_path / "suites" / "tasks.jsonl"
    assert write_tasks(path, tasks) == 2
    suite = read_tasks(path)
    assert suite.tasks == tasks
    assert path.read_text(encoding="utf-8").count("\n") == 2
    # Writing the suite object again produces the very same bytes.
    write_tasks(tmp_path / "again.jsonl", suite)
    assert (tmp_path / "again.jsonl").read_bytes() == path.read_bytes()


def test_read_skips_blank_lines(tmp_path: Path, make_task: Callable[..., Task]) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n" + make_task().to_json_line() + "\n\n   \n", encoding="utf-8")
    assert read_tasks(path).ids == ["synthetic/add"]


def test_read_names_the_line_with_invalid_json(
    tmp_path: Path, make_task: Callable[..., Task]
) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(make_task().to_json_line() + "\n{not json\n", encoding="utf-8")
    with pytest.raises(DataError) as info:
        read_tasks(path)
    assert str(info.value).startswith(f"{path}:2: invalid JSON:")
    assert info.value.details == {"path": str(path), "line": 2}


def test_read_rejects_a_line_that_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DataError, match=rf"{path}:1: expected a JSON object, got list"):
        read_tasks(path)


def test_read_names_the_line_and_field_of_a_schema_error(
    tmp_path: Path, task_record: dict[str, Any]
) -> None:
    record = {**task_record, "task_id": "bad id", "difficulty": "brutal"}
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(DataError) as info:
        read_tasks(path)
    message = str(info.value)
    assert message.startswith(f"{path}:1: invalid task: task_id: ")
    assert "; difficulty: " in message
    details = info.value.details
    assert (details["path"], details["line"], details["task_id"]) == (str(path), 1, "bad id")
    assert [error["field"] for error in details["errors"]] == ["task_id", "difficulty"]


def test_read_names_both_lines_of_a_duplicate_id(
    tmp_path: Path, make_task: Callable[..., Task]
) -> None:
    lines = [make_task().to_json_line(), make_task(task_id="synthetic/sub").to_json_line()]
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(DataError) as info:
        read_tasks(path)
    assert str(info.value) == f"{path}:3: duplicate task_id 'synthetic/add' (first seen on line 1)"
    assert info.value.details == {
        "path": str(path),
        "line": 3,
        "task_id": "synthetic/add",
        "first_line": 1,
    }


def test_read_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    with pytest.raises(DataError, match=rf"cannot read task file {path}: No such file") as info:
        read_tasks(path)
    assert info.value.details == {"path": str(path)}


def test_read_rejects_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(b'{"task_id": "\xff"}\n')
    with pytest.raises(DataError, match="not valid UTF-8"):
        read_tasks(path)


def test_write_rejects_duplicates_before_writing(
    tmp_path: Path, make_task: Callable[..., Task]
) -> None:
    path = tmp_path / "tasks.jsonl"
    with pytest.raises(DataError, match="duplicate task_id 'synthetic/add'") as info:
        write_tasks(path, [make_task(), make_task()])
    assert not path.exists()
    assert info.value.details["errors"][0]["field"] == "tasks"


def test_parse_task_line_defaults_name_the_string_source(task_record: dict[str, Any]) -> None:
    with pytest.raises(DataError, match=r"<string>:1: invalid JSON"):
        parse_task_line("nope")
    assert parse_task_line(json.dumps(task_record)).task_id == "synthetic/add"
