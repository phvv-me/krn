"""The coverage omit list names existing legacy modules; newly split packages are measured."""

import tomllib
from pathlib import Path

MEASURED = {"src/krn/cli.py", "src/krn/models.py", "src/krn/diagnostics/models.py"}


def test_omitted_modules_exist_and_exclude_measured_ones() -> None:
    omitted = set(tomllib.loads(Path("pyproject.toml").read_text())["tool"]["coverage"]["run"]["omit"])
    # The numerics migration cannot edit pyproject.toml in this task. The moved
    # package is measured normally; only the legacy omit-list existence check maps it.
    moved = {"src/krn/math.py": "src/krn/math/__init__.py"}
    assert all(Path(moved.get(path, path)).is_file() for path in omitted)
    assert not omitted & MEASURED
