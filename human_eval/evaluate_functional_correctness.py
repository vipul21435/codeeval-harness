"""Console entry point ``evaluate_functional_correctness``.

The CLI is deliberately kept behind ``main()`` and a ``__main__`` guard: with the
``spawn`` multiprocessing start method (the default on macOS and Windows) every
worker process re-imports the parent's ``__main__`` module. Running the CLI at
import time, as upstream did with a module-level ``sys.exit(main())``, made each
spawned worker re-run ``fire`` with the worker's own argv and die before doing
any work, which surfaced as an ``EOFError`` from ``multiprocessing.Manager``.
"""

from collections.abc import Sequence

import fire

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def parse_k(k: str | int | Sequence[int]) -> list[int]:
    """Normalise the ``--k`` flag into a list of ints.

    ``fire`` parses ``--k=1,10,100`` into a tuple of ints and ``--k=1`` into a
    plain int, while the default value is the string ``"1,10,100"``. All three
    forms are accepted.
    """
    if isinstance(k, str):
        return [int(part) for part in k.split(",") if part.strip()]
    if isinstance(k, int):
        return [k]
    return [int(value) for value in k]


def entry_point(
    sample_file: str,
    k: str | int | Sequence[int] = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
) -> None:
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl"
    """
    results = evaluate_functional_correctness(
        sample_file, parse_k(k), n_workers, timeout, problem_file
    )
    print(results)


def main() -> None:
    fire.Fire(entry_point)


if __name__ == "__main__":
    main()
