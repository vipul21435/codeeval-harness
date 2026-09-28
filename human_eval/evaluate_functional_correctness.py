"""Console entry point ``evaluate_functional_correctness``.

The CLI lives behind ``main()`` and a ``__main__`` guard so that importing this
module never runs an evaluation. Correctness does not depend on the guard:
sample workers are plain subprocesses (see ``human_eval.execution``) and never
re-import the caller's ``__main__``, so scripts that call the evaluator at
module level work too.
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
    n_workers: int | None = None,
    timeout: float | None = None,
    problem_file: str = HUMAN_EVAL,
) -> None:
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl"

    ``--n_workers`` and ``--timeout`` default to the ``VERIFYBENCH_WORKERS``
    and ``VERIFYBENCH_SAMPLE_TIMEOUT`` settings (4 and 3.0 seconds).
    """
    results = evaluate_functional_correctness(
        sample_file, parse_k(k), n_workers, timeout, problem_file
    )
    print(results)


def main() -> None:
    fire.Fire(entry_point)


if __name__ == "__main__":
    main()
