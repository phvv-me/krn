import tomllib
from pathlib import Path


def test_dependency_lists() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    manifest = tomllib.loads(Path("mb.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    expected = {
        dep.partition(">=")[0].lower().replace("_", "-"): ">=" + dep.partition(">=")[2] for dep in extras["dev"]
    }
    declared = manifest["dev"]["deps"] | manifest["dev"]["python"]["deps"]
    actual = {name: spec for name, spec in declared.items() if name not in {"nodejs", "pnpm"}}
    assert actual == expected


def test_extra_platforms() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    for group in ("cuda", "cuda-libs", "cuda-dx", "cccl", "refs"):
        assert all(dep.endswith("; sys_platform == 'linux'") for dep in extras[group])
    assert all(dep.endswith("; sys_platform == 'darwin'") for dep in extras["metal"])
    assert extras["cuda-libs"] == ["nvmath-python[cu13]>=1.0,<2; sys_platform == 'linux'"]
    assert extras["cuda-dx"] == ["nvmath-python[cu13-dx]>=1.0,<2; sys_platform == 'linux'"]
