"""The public surface: examples import only public modules, and no name of `krn.__all__` is defined in `_internals`."""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

import krn
import krn.compiler.api
import krn.compiler.errors
import krn.emulate
import krn.emulator.emulate
import krn.errors
from tests.conftest import EXAMPLE_ROOT

PRIVATE = "_internals"
PUBLIC_MODULES = frozenset(
    {
        "krn.collections",
        "krn.compiler.api",
        "krn.data.sparse",
        "krn.data.sparse_build",
        "krn.emulate",
        "krn.errors",
        "krn.fast",
        "krn.fast.math",
        "krn.frontend.atomic",
        "krn.functools",
        "krn.hashlib",
        "krn.intrinsics",
        "krn.itertools",
        "krn.math",
        "krn.profile",
    }
)
"""The modules besides `krn` that an example may import. Everything else of `krn` that users meet is a name exported by
`krn` itself, such as `krn.host` or `krn.targets`; `krn.compiler`, `krn.facts`, `krn.runtime`, `krn.hw`, `krn.lib`,
`krn.data` and `krn._internals` are otherwise private to examples."""
EXAMPLES = sorted(EXAMPLE_ROOT.glob("[0-9][0-9]_*/[0-9][0-9]_*.py"))


def test_compat_is_not_a_public_module() -> None:
    spec = importlib.util.find_spec("krn.compat")
    assert spec is None or spec.origin is None  # a leftover empty folder on a synced host is a namespace package
    assert importlib.util.find_spec("krn._internals.compat") is not None


def _krn_imports(path: Path) -> set[str]:
    """Every `krn` module path a file imports, including imports inside functions: `from krn import host` is `krn.host`
    when `host` is a module of the package and plain `krn` when it is a name the package exports."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module == "krn":
            modules.update("krn" if alias.name in krn.__all__ else f"krn.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return {name for name in modules if name == "krn" or name.startswith("krn.")}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.relative_to(EXAMPLE_ROOT).with_suffix("").as_posix())
def test_example_imports_only_public_modules(path: Path) -> None:
    assert not [name for name in _krn_imports(path) - PUBLIC_MODULES - {"krn"}]


@pytest.mark.parametrize("module", sorted(PUBLIC_MODULES))
def test_a_public_module_exists_outside_internals(module: str) -> None:
    assert PRIVATE not in module
    assert importlib.import_module(module).__file__ is not None


@pytest.mark.parametrize("name", krn.__all__)
def test_public_name_is_defined_outside_internals(name: str) -> None:
    obj = getattr(krn, name)
    assert PRIVATE not in (getattr(obj, "__module__", None) or type(obj).__module__)


def test_errors_gathers_what_krn_raises_and_warns_with() -> None:
    assert krn.errors.__all__ == sorted(krn.errors.__all__)
    assert krn.errors.TargetError is krn.compiler.errors.TargetError
    assert issubclass(krn.errors.TargetError, krn.errors.CaptureError)
    assert krn.errors.PrecisionWarning is krn.PrecisionWarning
    assert krn.errors.PerformanceWarning is krn.PerformanceWarning
    assert all(getattr(krn.errors, name) for name in krn.errors.__all__)


def test_emulate_is_the_home_of_the_emulator_entry() -> None:
    assert krn.emulate.emulate is krn.emulator.emulate.emulate
    assert krn.emulate.Deadlock is krn.emulator.emulate.Deadlock
    assert krn.emulate.__all__ == ["Deadlock", "emulate"]


def test_the_compiler_api_exports_the_artifact_inspector() -> None:
    assert "inspect_kernel" in krn.compiler.api.__all__
    assert krn.compiler.api.inspect_kernel.__module__ == "krn.compiler.artifacts"
