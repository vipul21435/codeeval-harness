import itertools
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import itemgetter
from typing import Any

import numpy as np
import numpy.typing as npt
import tqdm

from codeeval.settings import get_settings
from human_eval.data import HUMAN_EVAL, PathLike, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness

log = logging.getLogger(__name__)


def estimate_pass_at_k(
    num_samples: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    num_correct: Sequence[int] | npt.NDArray[np.integer[Any]],
    k: int,
) -> npt.NDArray[np.float64]:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    num_samples_it: Iterator[int]
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        if len(num_samples) != len(num_correct):
            raise ValueError(
                "num_samples and num_correct must have the same length, "
                f"got {len(num_samples)} and {len(num_correct)}"
            )
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct, strict=True)],
        dtype=np.float64,
    )


def evaluate_functional_correctness(
    sample_file: PathLike,
    k: Sequence[int] = (1, 10, 100),
    n_workers: int | None = None,
    timeout: float | None = None,
    problem_file: PathLike = HUMAN_EVAL,
) -> dict[str, float]:
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl"

    ``n_workers`` and ``timeout`` default to the ``VERIFYBENCH_WORKERS`` and
    ``VERIFYBENCH_SAMPLE_TIMEOUT`` settings (4 and 3.0 seconds unless
    configured; see ``codeeval.settings``).
    """
    settings = get_settings()
    if n_workers is None:
        n_workers = settings.workers
    if timeout is None:
        timeout = settings.sample_timeout

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id: Counter[str] = Counter()
        n_samples = 0
        results: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        missing = sorted(problems.keys() - completion_id.keys())
        if missing:
            raise ValueError(f"Some problems are not attempted: {missing}")

        log.info(
            "evaluating samples",
            extra={
                "sample_file": os.fspath(sample_file),
                "problem_file": os.fspath(problem_file),
                "n_samples": n_samples,
                "n_problems": len(problems),
                "n_workers": n_workers,
                "timeout": timeout,
            },
        )
        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total_list: list[int] = []
    correct_list: list[int] = []
    for result_list in results.values():
        result_list.sort(key=itemgetter(0))
        passed = [r[1]["passed"] for r in result_list]
        total_list.append(len(passed))
        correct_list.append(sum(passed))
    total = np.array(total_list)
    correct = np.array(correct_list)

    pass_at_k = {
        f"pass@{k_}": float(estimate_pass_at_k(total, correct, k_).mean())
        for k_ in k
        if (total >= k_).all()
    }

    # Finally, save the results in one file:
    def combine_results() -> Iterator[dict[str, Any]]:
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = f"{os.fspath(sample_file)}_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))
    log.info(
        "wrote results",
        extra={"results_file": out_file, "n_samples": n_samples, "pass_at_k": pass_at_k},
    )

    return pass_at_k
