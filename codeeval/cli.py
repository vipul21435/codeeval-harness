"""The ``verifybench`` command line: ``python -m codeeval <command>``.

Commands:

- ``demo``: validate the bundled HumanEval mini suite and score it (see
  :mod:`codeeval.demo`);
- ``validate``: run the fail-to-pass validator on a task file and print one
  verdict per task, exit status 1 when any task is not ``ok``;
- ``convert``: port HumanEval problems to a task file (:mod:`codeeval.convert`);
- ``generate``: sample completions for a task file from a model backend
  (:mod:`codeeval.backends`) into a ``samples.jsonl`` the evaluator scores.

Every command configures the structured logger from the settings and binds a
``run_id`` to its records. A :class:`~codeeval.errors.CodeEvalError` is
printed as one line on stderr and becomes exit status 2.
"""

import argparse
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from codeeval import __version__
from codeeval.backends import BACKEND_NAMES
from codeeval.errors import CodeEvalError, DataError
from codeeval.log import bind_context, configure_logging

PROG = "verifybench"

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Evaluation harness for LLM-generated code: pytest tasks, "
        "fail-to-pass validation and pass@k.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    demo = commands.add_parser(
        "demo",
        help="validate the bundled HumanEval mini suite and score it (no network, no model)",
        description="Generate two completions per task with the mock backend (the canonical "
        "solution, then a NotImplementedError stub), validate the suite with the "
        "fail-to-pass validator, then score the samples with pass@k. Offline: no network, "
        "no model key.",
    )
    demo.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="task file (default: data/tasks/humaneval_mini.jsonl)",
    )
    demo.add_argument("--limit", type=int, default=None, help="use only the first N tasks")
    demo.add_argument(
        "--repeats", type=int, default=3, help="grader runs per solution (default: %(default)s)"
    )
    demo.add_argument("--workers", type=int, default=None, help="tasks and samples in parallel")
    demo.add_argument(
        "--timeout", type=float, default=None, help="seconds per sample (default: settings)"
    )
    demo.add_argument(
        "--results-dir", type=Path, default=None, help="where outputs go (default: settings)"
    )

    validate = commands.add_parser(
        "validate",
        help="run the fail-to-pass validator on a task file",
        description="Run each task's tests against its baseline (must fail) and its reference "
        "(must pass), several times, and print one verdict per task.",
    )
    validate.add_argument("tasks", type=Path, help="JSONL task file")
    validate.add_argument(
        "--repeats", type=int, default=3, help="grader runs per solution (default: %(default)s)"
    )
    validate.add_argument("--workers", type=int, default=None, help="tasks validated in parallel")
    validate.add_argument(
        "--timeout", type=float, default=30.0, help="seconds per grader run (default: %(default)s)"
    )

    convert = commands.add_parser(
        "convert",
        help="port HumanEval problems to a task file",
        add_help=False,
        description="Arguments are passed to python -m codeeval.convert.",
    )
    convert.add_argument("args", nargs=argparse.REMAINDER)

    generate = commands.add_parser(
        "generate",
        help="sample completions for a task file from a model backend",
        description="Ask a model backend for N completions per task and write them as a "
        "samples file for evaluate_functional_correctness. The mock backend replays a "
        "canned file or, without one, each task's reference solution (a perfect model) "
        "and is fully offline; the openai backend needs VERIFYBENCH_OPENAI_API_KEY.",
    )
    generate.add_argument(
        "--backend",
        choices=BACKEND_NAMES,
        default=None,
        help="model backend (default: VERIFYBENCH_MODEL_BACKEND, mock)",
    )
    generate.add_argument("--tasks", type=Path, required=True, help="JSONL task file")
    generate.add_argument(
        "--n", type=positive_int, default=1, help="completions per task (default: %(default)s)"
    )
    generate.add_argument("--out", type=Path, required=True, help="samples file to write")
    generate.add_argument("--limit", type=positive_int, default=None, help="first N tasks only")
    generate.add_argument(
        "--canned", type=Path, default=None, help="mock only: samples file to replay per task id"
    )
    generate.add_argument(
        "--failure-rate",
        type=float,
        default=None,
        help="mock only: fraction of completions replaced by a stub (default: settings)",
    )
    generate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="mock only: seed for the stub draw (default: settings)",
    )
    return parser


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def cmd_demo(args: argparse.Namespace) -> int:
    from codeeval.demo import DEFAULT_TASKS, render_report, run_demo

    report = run_demo(
        args.tasks if args.tasks is not None else DEFAULT_TASKS,
        limit=args.limit,
        repeats=args.repeats,
        workers=args.workers,
        sample_timeout=args.timeout,
        results_dir=args.results_dir,
    )
    print(render_report(report))
    return EXIT_OK if report.ok else EXIT_MISMATCH


def cmd_validate(args: argparse.Namespace) -> int:
    from codeeval.f2p import validate_suite
    from codeeval.tasks import read_tasks

    suite = read_tasks(args.tasks)
    result = validate_suite(suite, repeats=args.repeats, timeout=args.timeout, workers=args.workers)
    for verdict in result.verdicts:
        line = f"{verdict.task_id}\t{verdict.verdict}"
        if verdict.error:
            line += f"\t{verdict.error}"
        print(line)
    counts = " ".join(f"{name}={count}" for name, count in result.counts.items())
    print(f"{len(result.verdicts)} tasks: {counts}")
    return EXIT_OK if result.ok else EXIT_MISMATCH


def cmd_convert(args: argparse.Namespace) -> int:
    from codeeval.convert import main as convert_main

    return convert_main(args.args)


def cmd_generate(args: argparse.Namespace) -> int:
    from codeeval.backends import (
        generate_samples,
        make_backend,
        reference_completions,
        write_samples,
    )
    from codeeval.demo import display_path
    from codeeval.settings import get_settings
    from codeeval.tasks import TaskSuite, read_tasks

    suite = read_tasks(args.tasks)
    if not suite.tasks:
        raise DataError(f"{args.tasks} holds no tasks", details={"path": args.tasks})
    if args.limit is not None:
        suite = TaskSuite(tasks=suite.tasks[: args.limit])
    name = args.backend if args.backend is not None else get_settings().model_backend
    backend = make_backend(
        name,
        canned=args.canned,
        pairs=reference_completions(suite),
        failure_rate=args.failure_rate,
        seed=args.seed,
    )
    samples = generate_samples(
        backend, [(task.task_id, task.prompt) for task in suite.tasks], n=args.n
    )
    count = write_samples(args.out, samples)
    print(f"{count} samples from {backend.name} for {len(suite)} tasks -> {display_path(args.out)}")
    return EXIT_OK


COMMANDS = {
    "demo": cmd_demo,
    "validate": cmd_validate,
    "convert": cmd_convert,
    "generate": cmd_generate,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        with bind_context(run_id=uuid.uuid4().hex[:12], command=args.command):
            return COMMANDS[args.command](args)
    except CodeEvalError as exc:
        print(f"{PROG}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


__all__ = [
    "EXIT_ERROR",
    "EXIT_MISMATCH",
    "EXIT_OK",
    "PROG",
    "build_parser",
    "main",
    "positive_int",
]
