# codeeval-harness

An evaluation harness for LLM-generated code, built on OpenAI's
[human-eval](https://github.com/openai/human-eval): the HumanEval dataset and
the pass@k evaluator from the paper "[Evaluating Large Language Models Trained
on Code](https://arxiv.org/abs/2107.03374)".

The upstream `human_eval` package stays importable and keeps its evaluation
semantics. This fork modernizes the packaging and tooling around it and grows a
`codeeval` package on top. Planned, in order:

- sandboxed, dockerized execution of model-generated code (digest-pinned images)
- pytest graders with fail-to-pass verification
- pass@k and per-task reports, results stored in SQLite
- LLM-as-judge grading against a rubric
- a Typer CLI and a FastAPI service
- a deterministic stub model provider so the whole pipeline runs offline

## Changes from upstream so far

- `pyproject.toml` with a [hatchling](https://hatch.pypa.io/) build and
  dependencies locked with [uv](https://docs.astral.sh/uv/) on Python 3.12.
  The `pkg_resources`-based `setup.py`, which broke editable installs on
  current setuptools, is gone.
- Each sample runs in a fresh interpreter started with `subprocess` instead of
  a `multiprocessing` worker plus a `Manager` process per sample. That fixes
  the upstream `EOFError` on macOS (where the `spawn` start method re-imported
  the CLI's `__main__` in every worker) without forcing a start method: no
  `if __name__ == "__main__"` guard is needed on any platform, the worker boots
  with stdlib imports only, and one process is started per sample.
- The per-sample `timeout` starts when the worker reports that it is running
  the program, not when the process is launched, so interpreter start-up on a
  loaded machine no longer turns correct solutions into `timed out`.
- A pass is derived from the worker's exit status. Upstream let the completion
  share the interpreter with a `Manager` list and trusted whatever was in it, so
  a completion could append `"passed"` itself.
- `--k=1,2,4` no longer crashes (`fire` passes it as a tuple), and pass@k values
  are plain floats rather than `np.float64`.
- Strictly typed (`mypy --strict`), linted and formatted with ruff, and covered
  by a pytest suite that reproduces the documented example numbers.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/); it
installs Python 3.12 itself if needed.

```
$ git clone https://github.com/vipul21435/codeeval-harness
$ cd codeeval-harness
$ make install    # uv sync + pre-commit install
```

## Usage

**This program runs untrusted model-generated code.** Each sample executes in
a fresh interpreter with a `reliability_guard` that disables the most
destructive functions, but that is not a security sandbox. Run evaluations
inside a container or VM you are prepared to lose; a dockerized sandbox is the
next item on the roadmap.

The same caveat applies to grading: the completion shares its interpreter with
the code that runs the tests. The harness never reads the verdict from an
object the completion can reach (a pass requires the worker to exit with
status 0, which only happens after the tests ran through), but a completion
that goes looking for the harness's own references can still forge a pass.
Treat pass@k on adversarial completions with suspicion until the sandbox lands.

Generate samples and save them as JSON Lines, one sample per line:

```
{"task_id": "Corresponding HumanEval task ID", "completion": "Completion only without the prompt"}
```

`data/example_problem.jsonl` and `data/example_samples.jsonl` illustrate the
format. The snippet below writes completions for every task; supply your own
`generate_one_completion`:

```python
from human_eval.data import read_problems, write_jsonl

problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
```

Evaluate them with:

```
$ uv run evaluate_functional_correctness samples.jsonl
Reading samples...
32800it [00:01, 23787.50it/s]
Running test suites...
100%|...| 32800/32800 [16:11<00:00, 33.76it/s]
Writing results to samples.jsonl_results.jsonl...
100%|...| 32800/32800 [00:00<00:00, 42876.84it/s]
{'pass@1': ..., 'pass@10': ..., 'pass@100': ...}
```

A file ending in `<input_path>_results.jsonl` is written next to the input;
each row carries the original sample plus `passed` and the execution `result`,
one of `"passed"`, `"timed out"` or `"failed: <reason>"`.

As a sanity check, the bundled example samples give pass@1 = 0.5 (and, with
`--k=1,2,4`, pass@2 = 0.8 and pass@4 = 1.0):

```
$ uv run evaluate_functional_correctness data/example_samples.jsonl --problem_file=data/example_problem.jsonl --k=1,2,4
Reading samples...
Running test suites...
Writing results to data/example_samples.jsonl_results.jsonl...
{'pass@1': 0.4999999999999999, 'pass@2': 0.8, 'pass@4': 1.0}
```

There is no unbiased estimate of pass@k with fewer than k samples per task, so
such k are skipped. See `uv run evaluate_functional_correctness --help` for the
remaining options (`--n_workers`, `--timeout`, `--problem_file`).

## Development

```
$ make check        # ruff check, ruff format --check, mypy --strict, pytest
$ make test-fast    # skip the tests that spawn subprocesses
$ make coverage     # pytest with branch coverage
$ make format       # auto-fix lint findings and reformat
```

Layout:

- `human_eval/` - the upstream package: dataset loading, per-sample execution,
  the pass@k estimator and the `evaluate_functional_correctness` CLI.
- `codeeval/` - the harness package (currently the version only; modules land
  slice by slice).
- `data/` - `HumanEval.jsonl.gz` plus the example problem and samples.
- `tests/` - pytest suite; tests marked `slow` spawn worker processes.

Copy `.env.example` to `.env` if you want to configure a hosted model provider
later; nothing in the repository needs credentials.

## Known Issues

While evaluation uses very little memory, you might see the following error
message when the system is running out of RAM. Since this may cause some
correct programs to fail, we recommend that you free some memory and try again.
```
malloc: can't allocate region
```

## Citation

Please cite the upstream paper when using the HumanEval dataset or evaluator:

```
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

MIT. The original code and dataset are Copyright (c) OpenAI; see
[LICENSE](LICENSE). Changes in this fork are released under the same license.
