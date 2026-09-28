"""Ports HumanEval problems to pytest-graded fail-to-pass tasks.

A HumanEval problem is a ``prompt`` (imports, signature and docstring of one
function), a ``canonical_solution`` (the function body), a ``test`` module
that defines ``check(candidate)`` and the ``entry_point`` name. Upstream
grades a completion by running ``prompt + completion + test`` followed by
``check(entry_point)`` in one namespace.

:func:`humaneval_to_task` keeps ``check`` as the grader and turns it into a
pytest module: the upstream test source stays verbatim, preceded by imports
of the solution module and followed by one test function that calls
``check`` on the entry point. The reference solution is the prompt plus the
canonical solution; the baseline is the prompt plus a body that raises
``NotImplementedError``, indented like the canonical body. Both are complete
modules, so a task is graded on its own, without the prompt.

``python -m codeeval.convert data/tasks/humaneval_mini.jsonl`` regenerates
the suite shipped in the repository: the first 20 problems in dataset order,
written with :func:`~codeeval.tasks.write_tasks`, byte for byte the same on
every run.
"""

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from codeeval.errors import DataError
from codeeval.tasks import (
    SOLUTION_MODULE,
    Difficulty,
    Task,
    TaskSuite,
    describe_validation_error,
    write_tasks,
)
from human_eval.data import read_problems

log = logging.getLogger(__name__)

SOURCE = "openai/human-eval"
REQUIRED_KEYS = ("task_id", "prompt", "entry_point", "canonical_solution", "test")
DEFAULT_INDENT = "    "
DEFAULT_LIMIT = 20
STUB_STATEMENT = "raise NotImplementedError"


def body_indent(canonical_solution: str) -> str:
    """The indentation of the first statement of a function body (default four spaces)."""
    for line in canonical_solution.splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return DEFAULT_INDENT


def join_module(prompt: str, body: str) -> str:
    """``prompt`` followed by ``body`` on its own line, ending with exactly one newline."""
    module = prompt if prompt.endswith("\n") else prompt + "\n"
    module += body
    return module if module.endswith("\n") else module + "\n"


def difficulty_for(canonical_solution: str) -> Difficulty:
    """A proxy tag from the size of the canonical solution.

    Up to 4 non-blank lines is ``easy``, up to 12 ``medium``, more is ``hard``.
    HumanEval carries no difficulty labels of its own; this keeps the tag
    deterministic and roughly ordered by how much code a solution needs.
    """
    lines = sum(1 for line in canonical_solution.splitlines() if line.strip())
    if lines <= 4:
        return "easy"
    if lines <= 12:
        return "medium"
    return "hard"


def humaneval_tests(task_id: str, entry_point: str, test: str) -> str:
    """The pytest module for a problem: the upstream test source plus one test that calls check."""
    return (
        f"# Ported from HumanEval task {task_id}; check() is the upstream grader, unchanged.\n"
        f"from {SOLUTION_MODULE} import *  # noqa: F403 - check() may use prompt helpers\n"
        f"from {SOLUTION_MODULE} import {entry_point}\n"
        "\n"
        f"{test.strip()}\n"
        "\n\n"
        f"def test_{entry_point}() -> None:\n"
        f"    check({entry_point})\n"
    )


def humaneval_to_task(problem: Mapping[str, Any]) -> Task:
    """Convert one HumanEval problem record; :class:`~codeeval.errors.DataError` names bad ones."""
    task_id = str(problem.get("task_id", "<unknown>"))
    missing = [key for key in REQUIRED_KEYS if key not in problem]
    if missing:
        raise DataError(
            f"HumanEval problem {task_id} lacks {', '.join(missing)}",
            details={"task_id": task_id, "missing": missing},
        )
    prompt, canonical, test = problem["prompt"], problem["canonical_solution"], problem["test"]
    entry_point = problem["entry_point"]
    if "def check(" not in test:
        raise DataError(
            f"HumanEval problem {task_id} has no check(candidate) in its test",
            details={"task_id": task_id},
        )
    try:
        return Task(
            task_id=task_id,
            prompt=prompt,
            entry_point=entry_point,
            reference_solution=join_module(prompt, canonical),
            baseline_solution=join_module(prompt, f"{body_indent(canonical)}{STUB_STATEMENT}\n"),
            tests=humaneval_tests(task_id, entry_point, test),
            language="python",
            difficulty=difficulty_for(canonical),
            metadata={"source": SOURCE, "upstream_task_id": task_id},
        )
    except ValidationError as exc:
        errors = describe_validation_error(exc)
        summary = "; ".join(f"{error['field']}: {error['message']}" for error in errors)
        raise DataError(
            f"HumanEval problem {task_id} does not convert to a valid task: {summary}",
            details={"task_id": task_id, "errors": errors},
        ) from exc


def humaneval_to_tasks(
    problems: Iterable[Mapping[str, Any]] | None = None, *, limit: int | None = None
) -> TaskSuite:
    """Convert ``problems`` (default: the packaged dataset, in file order), at most ``limit``."""
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")
    if problems is None:
        problems = read_problems().values()
    tasks: list[Task] = []
    for problem in problems:
        if limit is not None and len(tasks) >= limit:
            break
        tasks.append(humaneval_to_task(problem))
    try:
        suite = TaskSuite(tasks=tasks)
    except ValidationError as exc:
        errors = describe_validation_error(exc)
        raise DataError(
            "converted problems do not form a suite: "
            + "; ".join(error["message"] for error in errors),
            details={"errors": errors},
        ) from exc
    log.info("converted humaneval problems", extra={"n_tasks": len(suite), "limit": limit})
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m codeeval.convert",
        description="Port HumanEval problems to pytest-graded tasks, one JSON object per line.",
    )
    parser.add_argument("output", type=Path, help="JSONL file to write")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="how many problems to port, in dataset order; 0 means all (default: %(default)s)",
    )
    parser.add_argument(
        "--problem-file",
        type=Path,
        default=None,
        help="HumanEval JSONL (optionally gzipped) to read instead of the packaged dataset",
    )
    args = parser.parse_args(argv)
    problems = read_problems(args.problem_file) if args.problem_file else read_problems()
    suite = humaneval_to_tasks(problems.values(), limit=args.limit or None)
    count = write_tasks(args.output, suite)
    print(f"wrote {count} tasks to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    sys.exit(main())


__all__ = [
    "DEFAULT_INDENT",
    "DEFAULT_LIMIT",
    "REQUIRED_KEYS",
    "SOURCE",
    "STUB_STATEMENT",
    "body_indent",
    "difficulty_for",
    "humaneval_tests",
    "humaneval_to_task",
    "humaneval_to_tasks",
    "join_module",
    "main",
]
