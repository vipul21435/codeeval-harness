"""Task schema for pytest-graded, fail-to-pass coding tasks, and JSONL suites of them.

A :class:`Task` is one coding problem graded by pytest. It carries the
``prompt`` a model sees, the ``entry_point`` a solution must define, ``tests``
(a pytest module that imports the solution as the module ``solution``) and two
complete solution modules that exist to validate the grader itself:
``reference_solution`` must pass the tests and ``baseline_solution``, a stub,
must fail them. :mod:`codeeval.f2p` runs both to prove it.

The schema rejects what can be checked without running anything: malformed
ids, an entry point that is not an identifier, solutions that do not parse or
do not define the entry point, a tests module without a test function or
without an import of ``solution``, and a baseline identical to the reference.

A :class:`TaskSuite` is an ordered list of tasks with unique ids. Suites are
stored as JSON Lines, one task per line. :func:`read_tasks` reports a bad line
by file name and line number in a :class:`~codeeval.errors.DataError`;
:func:`write_tasks` writes lines that :func:`read_tasks` turns back into equal
tasks, byte for byte the same on every run.
"""

import ast
import json
import keyword
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from codeeval.errors import DataError

PathLike = str | os.PathLike[str]

# <namespace>/<name>; each part starts with a letter or digit and continues
# with letters, digits, '_', '-' or '.', e.g. HumanEval/0 or mysuite/fizz-buzz.
TASK_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"

# The module name a task's tests import the solution under.
SOLUTION_MODULE = "solution"

Language = Literal["python"]
Difficulty = Literal["easy", "medium", "hard"]


def parse_python(source: str, what: str) -> ast.Module:
    """``ast.parse(source)``, raising ``ValueError`` that names ``what`` on a syntax error."""
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"{what} is not valid Python: {exc.msg} (line {exc.lineno})") from None


def module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at the top level of a module: def, class, assignment or from-import."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def test_function_names(tree: ast.Module) -> list[str]:
    """Functions pytest would collect by name (``test`` prefix), anywhere in the module."""
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")
    ]


def imports_module(tree: ast.Module, module: str) -> bool:
    """Whether the module contains ``import <module>`` or ``from <module> import ...``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == module or alias.name.startswith(module + ".") for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and (node.module == module or node.module.startswith(module + "."))
        ):
            return True
    return False


class Task(BaseModel):
    """One pytest-graded coding task; see the module docstring for the contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    task_id: str = Field(
        pattern=TASK_ID_PATTERN,
        description="Unique id of the form <namespace>/<name>, e.g. HumanEval/0.",
    )
    prompt: str = Field(min_length=1, description="What the model is shown.")
    entry_point: str = Field(description="Name the solution module must define; the tests call it.")
    reference_solution: str = Field(
        min_length=1, description="Complete solution module that must pass the tests."
    )
    baseline_solution: str = Field(
        min_length=1, description="Complete stub module that must fail the tests."
    )
    tests: str = Field(
        min_length=1,
        description="pytest module; imports the solution as the module 'solution'.",
    )
    language: Language = Field(default="python", description="Language of the solutions.")
    difficulty: Difficulty = Field(default="medium", description="Difficulty tag.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict, description="Free-form, JSON-serialisable provenance."
    )

    @field_validator("entry_point")
    @classmethod
    def entry_point_is_identifier(cls, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError(f"{value!r} is not a valid Python identifier")
        return value

    @field_validator("reference_solution", "baseline_solution")
    @classmethod
    def solution_parses(cls, value: str) -> str:
        parse_python(value, "solution")
        return value

    @field_validator("tests")
    @classmethod
    def tests_are_a_pytest_module(cls, value: str) -> str:
        tree = parse_python(value, "tests")
        if not test_function_names(tree):
            raise ValueError("tests define no function named test*")
        if not imports_module(tree, SOLUTION_MODULE):
            raise ValueError(f"tests never import the {SOLUTION_MODULE!r} module")
        return value

    @model_validator(mode="after")
    def solutions_define_entry_point_and_differ(self) -> "Task":
        for field in ("reference_solution", "baseline_solution"):
            tree = ast.parse(getattr(self, field))
            if self.entry_point not in module_level_names(tree):
                raise ValueError(
                    f"{field} does not define the entry point {self.entry_point!r} at module level"
                )
        if self.reference_solution == self.baseline_solution:
            raise ValueError("baseline_solution is identical to reference_solution")
        return self

    def to_json_line(self) -> str:
        """The task as one line of JSON, stable across runs."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=True)


class TaskSuite(BaseModel):
    """An ordered list of tasks with unique ids."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tasks: list[Task] = Field(default_factory=list)

    @field_validator("tasks")
    @classmethod
    def ids_are_unique(cls, tasks: list[Task]) -> list[Task]:
        seen: dict[str, int] = {}
        for index, task in enumerate(tasks):
            first = seen.setdefault(task.task_id, index)
            if first != index:
                raise ValueError(
                    f"duplicate task_id {task.task_id!r} at positions {first} and {index}"
                )
        return tasks

    @property
    def ids(self) -> list[str]:
        return [task.task_id for task in self.tasks]

    def get(self, task_id: str) -> Task:
        """The task with ``task_id``; ``KeyError`` when there is none."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def __len__(self) -> int:
        return len(self.tasks)


def describe_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """One ``{"field", "message"}`` entry per error in ``exc``."""
    return [
        {
            "field": ".".join(str(part) for part in error["loc"]) or "<root>",
            "message": error["msg"],
        }
        for error in exc.errors(include_url=False)
    ]


def _line_error(path: str, line: int, message: str, **details: Any) -> DataError:
    return DataError(f"{path}:{line}: {message}", details={"path": path, "line": line, **details})


def parse_task_line(line: str, *, path: str = "<string>", number: int = 1) -> Task:
    """Parse one JSONL line into a :class:`Task`; errors name ``path`` and ``number``."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise _line_error(path, number, f"invalid JSON: {exc.msg} (column {exc.colno})") from exc
    if not isinstance(record, dict):
        raise _line_error(path, number, f"expected a JSON object, got {type(record).__name__}")
    try:
        return Task.model_validate(record)
    except ValidationError as exc:
        errors = describe_validation_error(exc)
        summary = "; ".join(f"{error['field']}: {error['message']}" for error in errors)
        raise _line_error(
            path,
            number,
            f"invalid task: {summary}",
            task_id=record.get("task_id"),
            errors=errors,
        ) from exc


def read_tasks(path: PathLike) -> TaskSuite:
    """Read a JSONL task file into a :class:`TaskSuite`.

    Blank lines are skipped. Any other problem raises
    :class:`~codeeval.errors.DataError` whose message starts with
    ``<path>:<line>:`` and whose ``details`` carry ``path``, ``line`` and,
    for schema errors, ``task_id`` and the offending ``errors``.
    """
    path_str = os.fspath(path)
    tasks: list[Task] = []
    first_line: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fp:
            for number, line in enumerate(fp, start=1):
                if not line.strip():
                    continue
                task = parse_task_line(line, path=path_str, number=number)
                if task.task_id in first_line:
                    raise _line_error(
                        path_str,
                        number,
                        f"duplicate task_id {task.task_id!r} (first seen on line "
                        f"{first_line[task.task_id]})",
                        task_id=task.task_id,
                        first_line=first_line[task.task_id],
                    )
                first_line[task.task_id] = number
                tasks.append(task)
    except OSError as exc:
        raise DataError(
            f"cannot read task file {path_str}: {exc.strerror or exc}", details={"path": path_str}
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataError(
            f"cannot read task file {path_str}: not valid UTF-8 ({exc.reason})",
            details={"path": path_str},
        ) from exc
    return TaskSuite(tasks=tasks)


def write_tasks(path: PathLike, tasks: TaskSuite | Iterable[Task]) -> int:
    """Write tasks as JSON Lines to ``path`` (parent directories are created).

    Duplicate ids raise :class:`~codeeval.errors.DataError` before anything
    is written. Returns the number of tasks written.
    """
    suite = tasks if isinstance(tasks, TaskSuite) else _suite_from(tasks)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as fp:
        for task in suite.tasks:
            fp.write(task.to_json_line() + "\n")
    return len(suite)


def _suite_from(tasks: Iterable[Task]) -> TaskSuite:
    try:
        return TaskSuite(tasks=list(tasks))
    except ValidationError as exc:
        errors = describe_validation_error(exc)
        raise DataError(
            "invalid task suite: " + "; ".join(error["message"] for error in errors),
            details={"errors": errors},
        ) from exc


__all__ = [
    "SOLUTION_MODULE",
    "TASK_ID_PATTERN",
    "Difficulty",
    "Language",
    "PathLike",
    "Task",
    "TaskSuite",
    "describe_validation_error",
    "imports_module",
    "module_level_names",
    "parse_python",
    "parse_task_line",
    "read_tasks",
    "test_function_names",
    "write_tasks",
]
