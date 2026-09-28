"""Hand-checked counts for codeeval.breakdown."""

import json
from collections import Counter
from pathlib import Path

import pytest

from codeeval.breakdown import Category, breakdown, categorize, read_results
from codeeval.errors import DataError


@pytest.mark.parametrize(
    ("result", "category"),
    [
        ("passed", Category.PASS),
        ("timed out", Category.TIMEOUT),
        ("failed: ", Category.FAIL),
        ("failed: AssertionError", Category.FAIL),
        ("failed: invalid syntax (<string>, line 1)", Category.SYNTAX_ERROR),
        (
            (
                "failed: expected an indented block after function definition on line 1 "
                "(<string>, line 2)"
            ),
            Category.SYNTAX_ERROR,
        ),
        ("failed: worker exited with signal 9", Category.SANDBOX_ERROR),
        ("failed: worker exited with status 137", Category.SANDBOX_ERROR),
    ],
)
def test_categorize(result: str, category: Category) -> None:
    assert categorize(result) is category


def test_runtime_error_mentioning_a_line_is_still_a_fail() -> None:
    assert categorize("failed: bad value at line 3") is Category.FAIL


def test_unknown_result_string_is_a_data_error() -> None:
    with pytest.raises(DataError, match="unknown result string"):
        categorize("ok")


RECORDS = [
    {"task_id": "T/0", "result": "passed"},
    {"task_id": "T/0", "result": "failed: "},
    {"task_id": "T/0", "result": "passed"},
    {"task_id": "T/1", "result": "timed out"},
    {"task_id": "T/1", "result": "failed: invalid syntax (<string>, line 1)"},
    {"task_id": "T/2", "result": "failed: worker exited with signal 9"},
]


def test_breakdown_counts_per_run_and_per_task() -> None:
    """3 tasks, 6 samples: 2 pass, 1 fail, 1 timeout, 1 syntax, 1 sandbox."""
    result = breakdown(RECORDS)
    assert result.samples == 6
    assert result.counts() == {
        "pass": 2,
        "fail": 1,
        "timeout": 1,
        "syntax_error": 1,
        "sandbox_error": 1,
    }
    assert result.tasks == {
        "T/0": Counter({Category.PASS: 2, Category.FAIL: 1}),
        "T/1": Counter({Category.TIMEOUT: 1, Category.SYNTAX_ERROR: 1}),
        "T/2": Counter({Category.SANDBOX_ERROR: 1}),
    }


def test_empty_breakdown_lists_every_category_at_zero() -> None:
    result = breakdown([])
    assert result.samples == 0
    assert list(result.counts()) == ["pass", "fail", "timeout", "syntax_error", "sandbox_error"]
    assert set(result.counts().values()) == {0}


def test_read_results_round_trips_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl_results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n\n", encoding="utf-8")
    assert read_results(path) == RECORDS
    assert breakdown(read_results(path)).samples == 6


def test_demo_shaped_records_are_passes_and_fails_only(tmp_path: Path) -> None:
    """Records as `verifybench demo` writes them: canonical passes, stubs `failed: `."""
    demo = [
        {
            "task_id": "HumanEval/0",
            "completion": "    return x\n",
            "backend": "mock",
            "index": 0,
            "result": "passed",
            "passed": True,
        },
        {
            "task_id": "HumanEval/0",
            "completion": "    raise NotImplementedError\n",
            "backend": "mock",
            "index": 1,
            "result": "failed: ",
            "passed": False,
        },
    ]
    path = tmp_path / "samples.jsonl_results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in demo), encoding="utf-8")
    result = breakdown(read_results(path))
    assert result.counts() == {
        "pass": 1,
        "fail": 1,
        "timeout": 0,
        "syntax_error": 0,
        "sandbox_error": 0,
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json}\n", r":1: invalid JSON"),
        ('{"task_id": "T/0"}\n', r":1: record needs a task_id string and a result string"),
        ('{"task_id": "T/0", "result": "passed"}\n[1, 2]\n', r":2: record needs task_id"),
        ('{"task_id": ["x"], "result": 5}\n', r":1: record needs a task_id string"),
        ('{"task_id": "T/0", "result": "passed"}\n{"result": 5}\n', r":2: record needs"),
        ('{"task_id": "T/0", "result": "crashed"}\n', r":1: unknown result string"),
        ("\n\n", "no result records"),
    ],
)
def test_read_results_names_the_bad_line(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DataError, match=message):
        read_results(path)


def test_breakdown_rejects_records_without_string_fields() -> None:
    with pytest.raises(DataError, match="record needs a task_id string and a result string"):
        breakdown([{"task_id": "T/0"}])
    with pytest.raises(DataError, match="record needs a task_id string"):
        breakdown([{"task_id": 3, "result": "passed"}])


def test_read_results_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no such results file"):
        read_results(tmp_path / "missing.jsonl")
