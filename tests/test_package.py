"""Packaging smoke tests: both top-level packages import and the version is wired up."""

import re
import tomllib
from pathlib import Path

import codeeval
import human_eval

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert codeeval.__version__ == pyproject["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", codeeval.__version__)


def test_upstream_package_stays_importable() -> None:
    assert Path(human_eval.__file__).parent.name == "human_eval"
