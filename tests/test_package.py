"""Packaging smoke tests: both top-level packages import and the version is wired up."""

import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import codeeval
import human_eval

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert codeeval.__version__ == pyproject["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", codeeval.__version__)


def test_upstream_package_stays_importable() -> None:
    assert Path(human_eval.__file__).parent.name == "human_eval"


def _uv(*args: str) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    proc = subprocess.run([uv, *args], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc


@pytest.mark.slow
def test_built_wheel_ships_the_dataset_and_installs_cleanly(tmp_path: Path) -> None:
    """The editable install used for development hides packaging mistakes: build
    the wheel, check it carries the dataset, and read it from a venv that has
    only the wheel installed."""
    _uv("build", "--wheel", "--project", str(REPO_ROOT), "--out-dir", str(tmp_path))
    (wheel,) = tmp_path.glob("codeeval-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "human_eval/data/HumanEval.jsonl.gz" in names
    assert "human_eval/data/__init__.py" in names

    venv = tmp_path / "venv"
    _uv("venv", "--python", sys.executable, str(venv))
    python = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    # human_eval.data needs nothing outside the stdlib, so --no-deps keeps this offline.
    _uv("pip", "install", "--python", str(python), "--no-deps", str(wheel))

    probe = (
        "from human_eval.data import HUMAN_EVAL, read_problems; "
        "print(len(read_problems())); print(HUMAN_EVAL)"
    )
    proc = subprocess.run(
        [str(python), "-c", probe],
        cwd=tmp_path,  # not the checkout, so the package must come from the venv
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    count, dataset_path = proc.stdout.strip().splitlines()
    assert count == "164"
    assert Path(dataset_path).resolve().is_relative_to(venv.resolve())
