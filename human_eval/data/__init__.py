import gzip
import json
import os
from collections.abc import Iterable, Iterator
from importlib.resources import files
from typing import Any

# The dataset ships inside this package so that it is present after a wheel
# install, not only in a source checkout.
HUMAN_EVAL = str(files("human_eval.data").joinpath("HumanEval.jsonl.gz"))

PathLike = str | os.PathLike[str]


def read_problems(evalset_file: PathLike = HUMAN_EVAL) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: PathLike) -> Iterator[dict[str, Any]]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    filename = os.fspath(filename)
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp, gzip.open(gzfp, "rt") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(filename) as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)


def write_jsonl(filename: PathLike, data: Iterable[dict[str, Any]], append: bool = False) -> None:
    """
    Writes an iterable of dictionaries to jsonl
    """
    mode = "ab" if append else "wb"
    filename = os.path.expanduser(os.fspath(filename))
    if filename.endswith(".gz"):
        with open(filename, mode) as fp, gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
            for x in data:
                gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))
