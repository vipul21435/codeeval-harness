import shutil
from pathlib import Path
from typing import Any

import pytest

from human_eval.data import read_problems

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture
def example_problem_file(tmp_path: Path) -> Path:
    """A private copy of data/example_problem.jsonl (one trivial task, id test/0)."""
    return Path(shutil.copy(DATA_DIR / "example_problem.jsonl", tmp_path))


@pytest.fixture
def example_samples_file(tmp_path: Path) -> Path:
    """A private copy of data/example_samples.jsonl.

    Copied so that the results file the evaluator writes next to it lands in
    tmp_path instead of the repository's data directory.
    """
    return Path(shutil.copy(DATA_DIR / "example_samples.jsonl", tmp_path))


@pytest.fixture
def example_problem(example_problem_file: Path) -> dict[str, Any]:
    return read_problems(example_problem_file)["test/0"]
