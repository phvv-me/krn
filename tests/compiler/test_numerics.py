"""Strict arithmetic is invariant under neighboring call-site opt-ins."""

import inspect
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks, Kernel, Precision, PrecisionWarning, T, fast, math
from krn.cache.tuning import TuningDb
from krn.compiler import performance
from krn.compiler.api import capture
from krn.compiler.errors import CaptureError, TargetError
from krn.compiler.heuristics import decide
from krn.compiler.lower import emit
from krn.data.formats import Plain16, Plain32
from krn.emulator.emulate import Emulated
from krn.emulator.emulate import build as emulate
from krn.facts import catalog
from krn.facts.fast_math import accuracy, relaxation
from krn.facts.targets import Target
from krn.frontend.precision import Ladder
from krn.lib import lib, lib_cublaslt, lib_host
from krn.runtime.compile_options import CompileOptions
from krn.runtime.cpu import CPUProgram
from krn.runtime.settings import Settings
from tests.helpers.devices import MetalRun

ARCHES = ("avx512", "sm_90a", "gfx942", "apple_m4")
FAST_ARCHES = ("avx512", "sm_90a", "apple_m4")  # AMD refuses fast.* until a HIP lowering is verified
TARGET = Target.of("avx512")
LAUNCH = Blocks(1, threads=1)


def source(directory: Path, body: str, name: str = "run") -> Path:
    file = directory / f"{name}.py"
    file.write_text(
        "from krn import T, Vector, Matrix, fast, kernel, math\n"
        f"@kernel\ndef {name}(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Matrix[T.f32]):\n" + body
    )
    return file


@pytest.fixture(scope="module")
def strict_programs(tmp_path_factory: pytest.TempPathFactory) -> tuple[CPUProgram, Emulated]:
    file = source(
        tmp_path_factory.mktemp("strict"),
        "    for i in range(a.shape[0]):\n"
        "        out[0, i] = a[i] * b[i] + c[i]\n"
        "        out[1, i] = a[i] / b[i]\n"
        "        out[2, i] = math.sqrt(a[i])\n"
        "        out[3, i] = (a[i] + b[i]) + c[i]\n",
    )
    ir = capture(file, "run", target=TARGET)
    return CPUProgram(ir, emit(ir, TARGET)), emulate(ir, TARGET)


def exact(left: np.ndarray, right: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
    live = ~np.isnan(left)
    np.testing.assert_array_equal(left[live].view(np.uint32), right[live].view(np.uint32))


@given(st.lists(st.tuples(*(st.floats(width=32) for _ in range(3))), min_size=1, max_size=24))
@settings(max_examples=50, deadline=None)
def test_strict_ieee_random_and_boundaries(
    strict_programs: tuple[CPUProgram, Emulated], triples: list[tuple[float, float, float]]
) -> None:
    edges = [
        (2**-149, 1.0, 0.0),
        (-(2**-149), 0.5, -0.0),
        (2**-126, 0.5, 2**-149),
        (float("inf"), 0.0, float("-inf")),
        (float("nan"), 1.0, 2.0),
        (0.0, -0.0, -0.0),
        (1 + 2**-23, 1 - 2**-23, -1.0),
    ]
    a, b, c = np.array(triples + edges, np.float32).T.copy()
    expected, actual = np.zeros((4, len(a)), np.float32), np.zeros((4, len(a)), np.float32)
    native, emulator = strict_programs
    with np.errstate(all="ignore"):
        emulator.run(LAUNCH, [a, b, c, expected])
        native(a, b, c, actual, launch=LAUNCH)
    exact(actual, expected)
    assert actual[0, -1] == 0.0  # contraction would produce -2**-46


@pytest.mark.parametrize("arch", FAST_ARCHES)
def test_mixed_and_separate_calls(tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    file = source(tmp_path, "    out[0, 0] = math.sqrt(a[0])\n    out[1, 0] = fast.math.sqrt(a[0])\n")
    ir = capture(file, "run", target=target)
    text = emit(ir, target).source
    strict = "precise::sqrt(" if target.vendor == "apple" else "sqrtf("
    approximate_text = {
        "nvidia": "sqrt.approx.f32",
        "apple": "fast::sqrt(",
        "cpu": "_fast_sqrt_f32(",
    }[target.vendor]
    assert strict in text and approximate_text in text
    for name, expression in (("strict", "math.sqrt(a[0])"), ("approximate", "fast.math.sqrt(a[0])")):
        other = source(tmp_path, f"    out[0, 0] = {expression}\n", name)
        generated = emit(capture(other, name, target=target), target).source
        assert (approximate_text in generated) == (name == "approximate")
    rows = decide(ir, target, Settings()).rows()
    assert any(name == "numerics" and "fast.math.sqrt" in value for name, value, _ in rows)
    if target.vendor == "cpu":
        a = np.array([2**-149, 2.0, float("inf"), float("nan")], np.float32)
        for value in a:
            output = np.zeros((2, 1), np.float32)
            CPUProgram(ir, emit(ir, target))(np.array([value]), a, a, output, launch=LAUNCH)
            expected = np.full((1,), np.sqrt(value), np.float32)
            exact(output[0], expected)
            exact(output[1], expected)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("operation", fast.math.__all__)
def test_every_fast_operation_has_a_lowering_or_target_error(tmp_path: Path, arch: str, operation: str) -> None:
    target = Target.of(arch)
    arguments = "a[0], b[0]" if operation in {"div", "pow"} else "a[0]"
    file = source(tmp_path, f"    out[0, 0] = fast.math.{operation}({arguments})\n")
    if target.vendor == "amd":
        with pytest.raises(TargetError) as refusal:
            capture(file, "run", target=target)
        assert refusal.value.feature == f"fast.math.{operation}"
        assert refusal.value.instead == f"the strict `krn.math.{operation}`"
    else:
        ir = capture(file, "run", target=target)
        assert emit(ir, target).source
        if arch == "avx512":
            CPUProgram(ir, emit(ir, target))
    assert "ulp" in accuracy(operation, target)
    assert relaxation(operation, target) and getattr(math, operation)


STRICT_COUNTERPARTS = ("reciprocal", "rsqrt", "div", "exp10", "tan", "pow")
IEEE_EXACT = ("reciprocal", "rsqrt", "div")  # IEEE division and square root, so every backend gives the same bits


def counterpart_file(tmp_path: Path, operation: str) -> Path:
    arguments = "a[i], b[i]" if operation in {"div", "pow"} else "a[i]"
    body = "    for i in range(a.shape[0]):\n" + f"        out[0, i] = math.{operation}({arguments})\n"
    return source(tmp_path, body)


SPECIAL_A = np.array([0, -0.0, 2**-149, -1, 1, 0.5, 3, 1000, 2.0**-126, float("inf")], np.float32)
SPECIAL_B = np.array([0, -0.0, 2**-149, 0, 2, 3, 0.5, 2, 1, float("nan")], np.float32)
TINY = float(np.finfo(np.float32).tiny)


def without_denormals(values: np.ndarray) -> np.ndarray:
    """Apple GPUs flush float32 denormals to zero, so a device run gets the normal-range inputs only."""
    return np.where((values != 0) & (np.abs(values) < TINY), np.float32(TINY), values).astype(np.float32)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("operation", STRICT_COUNTERPARTS)
def test_strict_counterparts_lower_on_every_arch(tmp_path: Path, arch: str, operation: str) -> None:
    target = Target.of(arch)
    ir = capture(counterpart_file(tmp_path, operation), "run", target=target)
    assert emit(ir, target).source
    assert operation in math.__all__


def test_strict_div_refuses_integers(tmp_path: Path) -> None:
    file = tmp_path / "divide.py"
    file.write_text(
        "from krn import T, Vector, kernel, math\n@kernel\n"
        "def run(a: Vector[T.i32], out: Vector[T.i32]):\n    out[0] = math.div(a[0], a[1])\n"
    )
    with pytest.raises(CaptureError, match="div needs f32 or f64"):
        capture(file, "run", target=TARGET)


@pytest.mark.parametrize("operation", STRICT_COUNTERPARTS)
def test_strict_counterparts_match_the_emulator_on_cpu(tmp_path: Path, operation: str) -> None:
    ir = capture(counterpart_file(tmp_path, operation), "run", target=TARGET)
    expected, actual = np.zeros((1, len(SPECIAL_A)), np.float32), np.zeros((1, len(SPECIAL_A)), np.float32)
    with np.errstate(all="ignore"):
        emulate(ir, TARGET).run(LAUNCH, [SPECIAL_A, SPECIAL_B, SPECIAL_B, expected])
        CPUProgram(ir, emit(ir, TARGET))(SPECIAL_A, SPECIAL_B, SPECIAL_B, actual, launch=LAUNCH)
    assert_counterpart(operation, actual[0], expected[0])


@pytest.mark.parametrize("operation", STRICT_COUNTERPARTS)
def test_strict_counterparts_match_the_emulator_on_metal(
    tmp_path: Path, metal_program: Callable[..., MetalRun], operation: str
) -> None:
    ir = capture(counterpart_file(tmp_path, operation), "run", target=Target.of("apple_m4"))
    a, b = without_denormals(SPECIAL_A), without_denormals(SPECIAL_B)
    expected, actual = np.zeros((1, len(a)), np.float32), np.zeros((1, len(a)), np.float32)
    with np.errstate(all="ignore"):
        emulate(ir, Target.of("apple_m4")).run(LAUNCH, [a, b, b, expected])
        metal_program(ir).run(LAUNCH, [a, b, b, actual])
    assert_counterpart(operation, actual[0], expected[0], flushed=True)


def assert_counterpart(operation: str, actual: np.ndarray, expected: np.ndarray, *, flushed: bool = False) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0]))
    finite = np.isfinite(expected) & (expected != 0) & (np.abs(expected) >= TINY if flushed else True)
    if operation in IEEE_EXACT:
        exact(actual[finite], expected[finite])
    else:
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=2**-20)


def test_warning_identity_filters_source_and_missing_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    file = source(tmp_path, "    out[0, 0] = fast.math.sqrt(a[0])\n    out[1, 0] = fast.math.sqrt(b[0])\n")
    for arch in FAST_ARCHES:
        target = Target.of(arch)
        ir = capture(file, "run", target=target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PrecisionWarning)
            emit(ir, target)
            emit(ir, target)
        precision = [w for w in caught if issubclass(w.category, PrecisionWarning)]
        assert len(precision) == 1
        assert precision[0].filename == str(file) and precision[0].lineno == 4
        message = str(precision[0].message)
        for part in (
            "precision is reduced",
            "fast.math.sqrt",
            arch,
            "ulp",
            "strict alternative: math.sqrt",
            "emulator computes it precisely",
        ):
            assert part in message
    target = TARGET
    ir = capture(file, "run", target=target)
    monkeypatch.setattr(performance, "_seen", set())
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        with pytest.raises(PrecisionWarning):
            emit(ir, target)
        with pytest.raises(PrecisionWarning):
            emit(ir, target)
    monkeypatch.setattr(performance, "_seen", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", PrecisionWarning)
        emit(ir, target)
    assert not caught
    entry = replace(catalog.FUNCTIONS["fast.sqrt"], cpu="", note="test target has no approximate square root")
    monkeypatch.setitem(catalog.FUNCTIONS, entry.name, entry)
    with pytest.raises(TargetError) as failure:
        capture(file, "run", target=TARGET)
    error = failure.value
    assert error.feature == "fast.math.sqrt" and error.target == TARGET.arch
    assert error.reason and "math.sqrt" in error.instead


@pytest.mark.parametrize("arch", ARCHES)
def test_strict_flags_are_fixed(arch: str) -> None:
    flags = CompileOptions.resolve(Target.of(arch), {}, settings=Settings()).flags()
    assert not any(
        word in flag for flag in flags for word in ("fast-math", "use_fast_math", "fmad=true", "ffp-contract=fast")
    )
    assert "--fmad=false" in flags if arch.startswith("sm") else "-ffp-contract=off" in flags


def test_prototypes_keep_public_names_and_contracts() -> None:
    for name in fast.math.__all__:
        function = getattr(fast.math, name)
        assert function.__module__ == "krn.fast.math" and "ulp" in function.__doc__
        with pytest.raises(NotImplementedError):
            function(*(1.0 for _ in inspect.signature(function).parameters))
    for name in (
        "sqrt",
        "exp",
        "exp2",
        "log",
        "log2",
        "sin",
        "cos",
        "tanh",
        "floor",
        "ceil",
        "copysign",
        "isnan",
        "isfinite",
        "ldexp",
        "exp10",
        "log10",
        "tan",
        "pow",
        "reciprocal",
        "rsqrt",
        "div",
    ):
        function = getattr(math, name)
        with pytest.raises(NotImplementedError):
            function(*(1.0 for _ in inspect.signature(function).parameters))


@pytest.mark.parametrize("dtype", ("u32", "u64", "i32", "i64"))
@pytest.mark.parametrize("tiled", (False, True))
def test_integer_corner_cases(tmp_path: Path, dtype: str, tiled: bool) -> None:
    file = tmp_path / "integers.py"
    index = "slice(0, 8)" if tiled else "i"
    loop = "" if tiled else "    for i in range(a.shape[0]):\n"
    indent = "    " if tiled else "        "
    expressions = (f"a[{index}] << b[{index}]", f"a[{index}] >> b[{index}]")
    if dtype.startswith("u"):
        expressions += (f"a[{index}] // b[{index}]", f"a[{index}] % b[{index}]")
    for operation, expression in enumerate(expressions):
        file = tmp_path / f"integers_{operation}.py"
        file.write_text(
            "from krn import T, Vector, kernel\n@kernel\n"
            f"def run(a: Vector[T.{dtype}], b: Vector[T.{dtype}], out: Vector[T.{dtype}]):\n"
            + loop
            + indent
            + f"out[{index}] = {expression}\n"
        )
        width = int(dtype[1:])
        dt = np.dtype(f"{'u' if dtype.startswith('u') else ''}int{width}")
        a = np.array([0, 1, 2, 31, 16, 0, 7, 127], dt)
        b = np.array([0, 0, 1, width - 1, width, width + 1, 127, 1], dt)
        if dtype.startswith("i"):
            a[0], b[0] = -1, -1
        ir = capture(file, "run", target=TARGET)
        expected, actual = np.zeros_like(a), np.zeros_like(a)
        emulate(ir, TARGET).run(LAUNCH, [a, b, expected])
        CPUProgram(ir, emit(ir, TARGET))(a, b, actual, launch=LAUNCH)
        np.testing.assert_array_equal(actual, expected)
        if operation < 2:
            assert not actual[4:7].any()
        else:
            assert actual[0] == actual[1] == 0
        for arch in ARCHES[1:]:
            target = Target.of(arch)
            assert "? 0 :" in emit(capture(file, "run", target=target), target).source


@pytest.mark.parametrize("dtype", ("f32", "f64", "f16", "bf16"))
@pytest.mark.parametrize("arch", ARCHES)
def test_fma_types_and_native_refusals(tmp_path: Path, dtype: str, arch: str) -> None:
    target = Target.of(arch)
    file = tmp_path / "fused.py"
    file.write_text(
        "from krn import T, Vector, kernel, math\n@kernel\n"
        f"def fused(a: Vector[T.{dtype}], out: Vector[T.{dtype}]):\n"
        "    out[0] = math.fma(a[0], a[1], a[2])\n"
    )
    supported = dtype in ("f32", "f64") and not (dtype == "f64" and arch == "apple_m4")
    supported |= dtype == "f16" and arch != "avx512"
    supported |= dtype == "bf16" and arch == "sm_90a"
    if not supported:
        with pytest.raises(TargetError) as refused:
            capture(file, "fused", target=target)
        assert refused.value.target == arch and refused.value.reason and refused.value.instead
        return
    ir = capture(file, "fused", target=target)
    text = emit(ir, target).source
    assert "fma" in text
    if dtype in ("f32", "f64"):
        dt = np.dtype("float32" if dtype == "f32" else "float64")
        mantissa = 23 if dtype == "f32" else 52
        a = np.array([1 + 2.0**-mantissa, 1 - 2.0**-mantissa, -1], dt)
        result = np.zeros(1, dt)
        emulate(ir, target).run(LAUNCH, [a, result])
        assert result[0] == -(2.0 ** (-2 * mantissa))
        if arch == "avx512":
            actual = np.zeros_like(result)
            CPUProgram(ir, emit(ir, target))(a, actual, launch=LAUNCH)
            np.testing.assert_array_equal(actual, result)


def test_precision_floor_and_matrix_operation_choice(tmp_path: Path) -> None:

    ladder = Ladder("weights", (Plain16(), Plain32()))
    with pytest.raises(TargetError, match="below the 32-bit precision floor"):
        ladder.resolve(TARGET, Settings(precision=Precision(floor_bits=32, pins={"weights": "fp16"})))
    with pytest.raises(TargetError, match="floor"):
        ladder.resolve(TARGET, Settings(precision=Precision(floor_bits=64)))
    assert ladder.resolve(TARGET).storage.elem.bits >= 16
    problem = lib.Problem(lib.Op.MATMUL, (T.f32,) * 3, (1, 1, 1), lib.LibOptions())
    dispatcher = lib.Dispatcher(
        TARGET, {lib.Op.MATMUL: [lib_host.Host()]}, TuningDb(tmp_path / "tuning"), lib.Mode.AUTO, {}, "test"
    )
    assert dispatcher.choose(lib.MatmulCandidate, problem, capturing=False).name == "host"
    with pytest.raises(TargetError, match="matmul precision=tf32"):
        dispatcher.choose(lib.MatmulCandidate, replace(problem, precision=lib.Compute.TF32), capturing=False)
    assert lib_cublaslt._FORMATS[(T.f32,) * 3][0] == "COMPUTE_32F_PEDANTIC"
    key = lib_cublaslt.plan_key(problem, (1, 2, 3, None))
    assert key.matches(problem)
    assert not key.matches(replace(problem, precision=lib.Compute.TF32))


def test_strict_math_helpers_keep_their_python_contracts() -> None:
    assert cast(Kernel[..., float], math.clip).fn(2.0, 0.0, 1.0) == 1.0
    assert cast(Kernel[..., bool], math.in_range).fn(1.0, 0.0, 2.0)
    assert cast(Kernel[..., float], math.lerp).fn(0.0, 2.0, 0.5) == 1.0
    assert cast(Kernel[..., float], math.inverse_lerp).fn(0.0, 2.0, 1.0) == 0.5
    with pytest.raises(NotImplementedError):
        cast(Kernel[..., float], math.mean).fn(np.zeros(1, np.float32), 0, 1)


@pytest.mark.parametrize(
    "operation", ("sqrt", "rsqrt", "reciprocal", "div", "log", "log2", "log10", "exp", "exp2", "exp10")
)
def test_fast_precise_emulator_special_values(tmp_path: Path, operation: str) -> None:
    arguments = "a[i], b[i]" if operation == "div" else "a[i]"
    file = source(
        tmp_path, "    for i in range(a.shape[0]):\n" + f"        out[0, i] = fast.math.{operation}({arguments})\n"
    )
    ir = capture(file, "run", target=TARGET)
    a = np.array([0, -0.0, 2**-149, -1, 1, float("inf"), float("-inf"), float("nan"), 1000], np.float32)
    b = np.array([0, -0.0, 2**-149, 0, 0, float("inf"), 2, 1, 0], np.float32)
    expected, actual = np.zeros((1, len(a)), np.float32), np.zeros((1, len(a)), np.float32)
    with np.errstate(all="ignore"):
        emulate(ir, TARGET).run(LAUNCH, [a, b, b, expected])
        CPUProgram(ir, emit(ir, TARGET))(a, b, b, actual, launch=LAUNCH)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_max_ulp(actual[finite], expected[finite], maxulp=1)
    np.testing.assert_array_equal(np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("tiled", (False, True))
def test_integer_guards_evaluate_atomic_operands_once(tmp_path: Path, tiled: bool) -> None:
    file = tmp_path / "atomic_operands.py"
    access = "slice(0, 4)" if tiled else "0"
    file.write_text(
        "from krn import T, Vector, atomic, kernel\n@kernel\n"
        "def once(a: Vector[T.u32], count: Vector[T.u32], out: Vector[T.u32]):\n"
        f"    out[{access}] = a[{access}] // atomic.add(count, 0, T.u32(1))\n"
        f"    out[{access}] = a[{access}] << atomic.add(count, 0, T.u32(1))\n"
    )
    if tiled:
        for arch in ARCHES:
            with pytest.raises(CaptureError, match="bind scalar atomic results to local variables"):
                capture(file, "once", target=Target.of(arch))
        file = tmp_path / "atomic_locals.py"
        file.write_text(
            "from krn import T, Vector, atomic, kernel\n@kernel\n"
            "def once(a: Vector[T.u32], count: Vector[T.u32], out: Vector[T.u32]):\n"
            "    divisor = atomic.add(count, 0, T.u32(1))\n"
            "    out[slice(0, 4)] = a[slice(0, 4)] // divisor\n"
            "    amount = atomic.add(count, 0, T.u32(1))\n"
            "    out[slice(0, 4)] = a[slice(0, 4)] << amount\n"
        )
    ir = capture(file, "once", target=TARGET)
    a, output, count = np.full(4, 8, np.uint32), np.zeros(4, np.uint32), np.zeros(1, np.uint32)
    emulate(ir, TARGET).run(LAUNCH, [a, count, output])
    assert count[0] == 2
    assert np.all(output[: 4 if tiled else 1] == 16)
    expected = output.copy()
    count.fill(0)
    output.fill(0)
    CPUProgram(ir, emit(ir, TARGET))(a, count, output, launch=LAUNCH)
    assert count[0] == 2
    np.testing.assert_array_equal(output, expected)


def test_constant_integer_corner_cases_match_runtime_rule(tmp_path: Path) -> None:
    file = tmp_path / "constants.py"
    file.write_text(
        "from krn import T, Vector, kernel\n@kernel\n"
        "def constants(out: Vector[T.u32]):\n"
        "    out[0] = T.u32(1 // 0)\n    out[1] = T.u32(1 % 0)\n"
        "    out[2] = T.u32(1 << 64)\n    out[3] = T.u32(1 >> -1)\n"
    )
    for arch in ARCHES:
        target = Target.of(arch)
        ir = capture(file, "constants", target=target)
        assert emit(ir, target).source
        result = np.ones(4, np.uint32)
        emulate(ir, target).run(LAUNCH, [result])
        assert not result.any()
        if arch == "avx512":
            result.fill(1)
            CPUProgram(ir, emit(ir, target))(result, launch=LAUNCH)
            assert not result.any()
