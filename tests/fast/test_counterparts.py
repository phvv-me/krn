"""Every `krn.fast` function is a precision-losing twin of a strict `krn.math` function, and nothing else."""

import importlib
import inspect
import pkgutil
import re
import warnings
from pathlib import Path
from types import FunctionType, ModuleType

import pytest

import krn.fast
from krn.compiler import performance
from krn.compiler.api import capture
from krn.compiler.errors import PrecisionWarning
from krn.compiler.lower import emit
from krn.facts.fast_math import ACCURACY
from krn.facts.targets import Target

ARCHES = ("avx512", "sm_90a", "gfx942", "apple_m4")
FAST_ARCHES = ("avx512", "sm_90a", "apple_m4")  # AMD refuses fast.* until a HIP lowering is verified
LABELS = ("CUDA:", "Metal:", "HIP:", "C and emulator:")
# Vendors document these as composed errors with no fixed bound: strict xfails until a measured bound is written.
UNBOUNDED = {
    ("exp10", "apple"),
    ("log10", "apple"),
    ("pow", "nvidia"),
    ("pow", "apple"),
    ("tan", "nvidia"),
    ("tan", "apple"),
    ("tanh", "apple"),
}
BOUND = re.compile(r"<=\s*\d")
SOURCE = (
    "from krn import T, Vector, Matrix, fast, kernel, math\n"
    "@kernel\ndef run(a: Vector[T.f32], out: Matrix[T.f32]):\n"
    "    for i in range(a.shape[0]):\n"
    "        out[0, i] = fast.math.sqrt(a[i])\n"
    "        out[1, i] = fast.math.sqrt(a[i])\n"
    "        out[2, i] = fast.math.exp(a[i])\n"
    "        out[3, i] = math.sqrt(a[i])\n"
)


def fast_modules() -> list[ModuleType]:
    return [
        importlib.import_module(name) for _, name, _ in pkgutil.iter_modules(krn.fast.__path__, f"{krn.fast.__name__}.")
    ]


def fast_functions() -> list[tuple[str, FunctionType]]:
    return [
        (f"{module.__name__}.{name}", function)
        for module in fast_modules()
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module.__name__
    ]


FUNCTIONS = fast_functions()
PATHS = [path for path, _ in FUNCTIONS]


def contract(function: FunctionType) -> tuple[object, ...]:
    """Type parameters with their bounds, parameter names with annotations, and the return annotation.

    Two modules declare their own TypeVars, so the annotations are compared by their text and the bounds by value.
    """
    signature = inspect.signature(function)
    return (
        [(parameter.__name__, parameter.__bound__) for parameter in function.__type_params__],
        [(p.name, str(p.annotation)) for p in signature.parameters.values()],
        str(signature.return_annotation),
    )


def strict_twin(path: str) -> FunctionType:
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module.replace(krn.fast.__name__, "krn", 1)), name)


def test_the_walk_finds_the_fast_functions() -> None:
    assert len(FUNCTIONS) == 15
    assert {path.rsplit(".", 1)[1] for path in PATHS} == set(ACCURACY)


@pytest.mark.parametrize("path", PATHS)
def test_the_strict_twin_has_the_same_name_and_contract(path: str) -> None:
    fast_function = dict(FUNCTIONS)[path]
    strict = strict_twin(path)
    assert callable(strict)
    assert contract(fast_function) == contract(strict)


@pytest.mark.parametrize("path", PATHS)
def test_the_docstring_states_each_backend_and_names_the_twin(path: str) -> None:
    doc = dict(FUNCTIONS)[path].__doc__ or ""
    assert all(label in doc for label in LABELS)
    assert f"`krn.math.{path.rsplit('.', 1)[1]}`" in doc


@pytest.mark.parametrize(
    ("name", "vendor"),
    [
        pytest.param(name, vendor, marks=pytest.mark.xfail(strict=True, reason="composed error, no fixed bound"))
        if (name, vendor) in UNBOUNDED
        else (name, vendor)
        for name in sorted(ACCURACY)
        for vendor in ("nvidia", "amd", "apple")
    ],
)
def test_the_accuracy_table_lists_a_finite_bound(name: str, vendor: str) -> None:
    assert BOUND.search(ACCURACY[name][vendor]), ACCURACY[name][vendor]


@pytest.mark.parametrize("arch", FAST_ARCHES)
def test_a_fast_call_warns_once_per_kernel_kind_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str
) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    file = tmp_path / "precision.py"
    file.write_text(SOURCE)
    target = Target.of(arch)
    ir = capture(file, "run", target=target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        emit(ir, target)
        emit(ir, target)
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, PrecisionWarning) and "precision is reduced" in str(w.message)
    ]
    assert len(messages) == 2
    assert all(arch in message for message in messages)
    assert "fast.math.sqrt" in messages[0] and "fast.math.exp" in messages[1]


@pytest.mark.parametrize("arch", ARCHES)
def test_a_strict_call_never_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    file = tmp_path / "strict.py"
    file.write_text(SOURCE.replace("fast.math.", "math."))
    target = Target.of(arch)
    ir = capture(file, "run", target=target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        emit(ir, target)
    assert not [
        w for w in caught if issubclass(w.category, PrecisionWarning) and "precision is reduced" in str(w.message)
    ]
