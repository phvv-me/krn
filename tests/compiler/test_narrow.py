"""Narrow floats (fp8, fp6, fp4): the table reference, the composed conversions on the C and Apple backends, the NVIDIA
conversion instructions, and the software-fallback warning and refusal."""

import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks
from krn.compiler import narrow
from krn.compiler.api import capture
from krn.compiler.dtypes import fp
from krn.compiler.errors import TargetError
from krn.compiler.lower import emit
from krn.compiler.performance import PerformanceWarning
from krn.data.formats import E2M1, E2M3, E3M2, E4M3FN, E5M2, FloatFormat
from krn.facts.targets import CPU, Target
from krn.runtime.cpu import CPUProgram, CPURenderer
from krn.runtime.settings import configure
from tests.helpers.devices import CUDARun

if TYPE_CHECKING:
    from collections.abc import Callable

    from krn.compiler.ir import Kernel
    from krn.runtime.cuda_backend import CUDADevice
    from tests.helpers.devices import MetalRun

FORMATS = {"f8e4m3fn": E4M3FN, "f8e5m2": E5M2, "f6e2m3": E2M3, "f6e3m2": E3M2, "f4e2m1": E2M1}
TYPES = tuple(FORMATS)
COUNT = 64 * 8


def round_trip(directory: Path, ty: str) -> Path:
    file = directory / "kernel.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "@kernel\ndef k(a: Vector[T.f32], out: Vector[T.f32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n"
        f"        out[i] = a[i].to(T.{ty}).to(T.f32)\n"
    )
    return file


def inputs(fmt: FloatFormat, seed: int) -> np.ndarray:
    """Every value of the format with both signs, the midpoints and their float32 neighbours, and random values."""
    table = np.asarray(fmt.table, np.float32)
    levels = np.array(sorted({abs(v) for v in fmt.table if math.isfinite(v)}), np.float64)
    mids = ((levels[:-1] + levels[1:]) / 2).astype(np.float32)
    mids = np.concatenate([mids, -mids])
    near = np.concatenate([mids.view(np.uint32) + np.uint32(step) for step in (0, 1, 0xFFFFFFFF)]).view(np.float32)
    random = (np.random.default_rng(seed).standard_normal(COUNT) * 2.5 * levels.max()).astype(np.float32)
    special = [np.inf, -np.inf, np.nan, 0.0, -0.0]
    values = np.concatenate([table, near, random, np.array(special, np.float32)])
    return np.resize(values, -(-len(values) // 64) * 64)


def expected(fmt: FloatFormat, values: np.ndarray) -> np.ndarray:
    return narrow.reference_decode(fmt, narrow.reference_encode(fmt, values))


def assert_bits_equal(actual: np.ndarray, want: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(want))
    live = ~np.isnan(want)
    np.testing.assert_array_equal(actual[live].view(np.uint32), want[live].view(np.uint32))


def exact_code(fmt: FloatFormat, x: float) -> int:
    """The code of float32 `x` by an exact rational search over the finite values: the oracle of the table reference."""
    if math.isnan(x):
        return narrow.nan_code(fmt)
    sign = (np.float32(x).view(np.uint32) >> 31) << (fmt.bits - 1)
    magnitude = Fraction(min(abs(x), fmt.max))
    candidates = [(Fraction(fmt.table[c]), c) for c in range(1 << (fmt.bits - 1)) if math.isfinite(fmt.table[c])]
    return int(sign) | min(candidates, key=lambda vc: (abs(vc[0] - magnitude), vc[1] % 2))[1]


@pytest.mark.parametrize("fmt", FORMATS.values(), ids=FORMATS.keys())
@settings(max_examples=300, deadline=None)
@given(st.floats(width=32))
def test_reference_encode_is_the_exact_nearest_even_saturating_code(fmt: FloatFormat, x: float) -> None:
    assert int(narrow.reference_encode(fmt, np.float32(x))) == exact_code(fmt, float(np.float32(x)))


@pytest.mark.parametrize("fmt", FORMATS.values(), ids=FORMATS.keys())
def test_every_finite_code_round_trips_through_the_reference(fmt: FloatFormat) -> None:
    """Infinities and NaNs saturate or go to the canonical code, so only the finite codes come back as themselves."""
    codes = np.arange(1 << fmt.bits)
    values = narrow.reference_decode(fmt, codes)
    live = np.isfinite(values)
    np.testing.assert_array_equal(narrow.reference_encode(fmt, values[live]), codes[live])


@pytest.mark.parametrize("fmt", FORMATS.values(), ids=FORMATS.keys())
def test_reference_follows_the_ptx_saturation_rule(fmt: FloatFormat) -> None:
    """Overflow and infinity saturate; a NaN is the canonical code, or the positive maximum where there is no NaN."""
    negative_top = narrow.maximum_code(fmt) | (1 << (fmt.bits - 1))
    assert narrow.reference_encode(fmt, np.float32(np.inf)) == narrow.maximum_code(fmt)
    assert narrow.reference_encode(fmt, np.float32(-1e30)) == negative_top
    assert narrow.reference_encode(fmt, np.float32(np.nan)) == narrow.nan_code(fmt)
    assert narrow.reference_encode(fmt, np.float32(-0.0)) == 1 << (fmt.bits - 1)


def test_the_canonical_nan_codes_are_the_documented_ones() -> None:
    assert narrow.nan_code(E4M3FN) == 0x7F and narrow.nan_code(E5M2) == 0x7F
    assert narrow.nan_code(E2M1) == 0x7 and narrow.nan_code(E2M3) == 0x1F and narrow.nan_code(E3M2) == 0x1F


@pytest.mark.parametrize("ty", TYPES)
def test_composed_round_trip_matches_the_reference_on_the_c_backend(tmp_path: Path, ty: str) -> None:
    fmt, target = FORMATS[ty], Target.of("avx512")
    ir = capture(round_trip(tmp_path, ty), "k", target=target)
    values = inputs(fmt, seed=5)
    out = np.zeros_like(values)
    CPUProgram(ir, emit(ir, target))(values, out, launch=Blocks(1, threads=len(values)))
    assert_bits_equal(out, expected(fmt, values))


@pytest.mark.parametrize("ty", TYPES)
def test_composed_round_trip_matches_the_reference_on_the_apple_gpu(
    tmp_path: Path, metal_program: Callable[[Kernel], MetalRun], ty: str
) -> None:
    fmt = FORMATS[ty]
    ir = capture(round_trip(tmp_path, ty), "k", target=Target.of("apple_m4"))
    values = inputs(fmt, seed=6)
    out = np.zeros_like(values)
    metal_program(ir).run(Blocks(len(values) // 64, threads=64), [values, out])
    assert_bits_equal(out, expected(fmt, values))


@pytest.mark.parametrize("ty", TYPES)
def test_nvidia_round_trip_matches_the_reference(tmp_path: Path, gpu: CUDADevice, ty: str) -> None:
    """Native where the card has the conversion (silo sm_120a) and composed where it does not (crimson sm_86)."""
    fmt = FORMATS[ty]
    ir = capture(round_trip(tmp_path, ty), "k", target=gpu.nvidia)
    values = inputs(fmt, seed=7)
    out = np.zeros_like(values)
    CUDARun(gpu, ir).run(Blocks(len(values) // 64, threads=64), [values, out])
    assert_bits_equal(out, expected(fmt, values))


def emitted(directory: Path, ty: str, arch: str) -> tuple[str, list[str]]:
    """The source of the round trip for `arch` and the software-fallback warnings it raised."""
    target = Target.of(arch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source = emit(capture(round_trip(directory, ty), "k", target=target), target).source
    return source, [str(w.message) for w in caught if issubclass(w.category, PerformanceWarning)]


def test_apple_warns_once_per_conversion_kind_and_names_where_it_is_native(tmp_path: Path) -> None:
    _, messages = emitted(tmp_path, "f8e4m3fn", "apple_m4")
    assert len(messages) == 2
    assert any("convert f8e4m3fn to f32 runs as a 6-instruction" in m for m in messages)
    assert any("convert f32 to f8e4m3fn runs as a 17-instruction" in m for m in messages)
    assert all("native on NVIDIA GPUs at sm_89 and later" in m for m in messages)


def test_cpu_never_warns(tmp_path: Path) -> None:
    assert emitted(tmp_path, "f4e2m1", "avx512")[1] == []


def test_nvidia_fp8_is_native_from_sm_89_and_composed_before(tmp_path: Path) -> None:
    native, silent = emitted(tmp_path, "f8e5m2", "sm_89")
    assert silent == [] and "cvt.rn.satfinite.e5m2x2.f32" in native and "_native_encode_f8e5m2" in native
    composed, warned = emitted(tmp_path, "f8e5m2", "sm_86")
    assert len(warned) == 2 and "_encode_f8e5m2" in composed and "cvt.rn.satfinite" not in composed


@pytest.mark.parametrize("ty", ["f4e2m1", "f6e2m3", "f6e3m2"])
def test_nvidia_fp4_and_fp6_are_native_on_sm_120a(tmp_path: Path, ty: str) -> None:
    native, silent = emitted(tmp_path, ty, "sm_120a")
    assert silent == [] and "_native_decode_" in native


def test_nvidia_fp4_is_composed_on_sm_89_and_says_where_it_is_native(tmp_path: Path) -> None:
    _, warned = emitted(tmp_path, "f4e2m1", "sm_89")
    assert len(warned) == 2 and all("sm_100a, sm_101a and sm_120a" in m for m in warned)


def test_metal_uses_the_bit_cast_spelling(tmp_path: Path) -> None:
    source, _ = emitted(tmp_path, "f8e4m3fn", "apple_m4")
    assert "as_type<uint>(" in source and "as_type<float>(" in source


def test_warning_settings_silence_or_move_the_threshold(tmp_path: Path) -> None:
    with configure(**{"warn.warn_software_fallback": False}):
        assert emitted(tmp_path, "f8e4m3fn", "apple_m4")[1] == []
    with configure(**{"warn.software_fallback_min_instructions": 100}):
        assert emitted(tmp_path, "f8e4m3fn", "apple_m4")[1] == []


def test_a_narrow_float_without_a_composition_is_refused_with_the_four_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every narrow name the frontend exports has a composition, so the table entry is removed to reach the refusal."""
    monkeypatch.delitem(narrow.FORMATS, fp(4, 3))
    target = Target.of("sm_120a")
    with pytest.raises(TargetError) as error:
        emit(capture(round_trip(tmp_path, "f8e4m3fn"), "k", target=target), target)
    assert error.value.feature == "the software conversion of f8e4m3fn"
    assert error.value.target == "sm_120a"
    assert error.value.reason == "f8e4m3fn is a narrow float with no software composition"
    assert error.value.instead == "convert through f32 to one of f8e4m3fn, f8e5m2, f6e2m3, f6e3m2, f4e2m1"


@pytest.mark.parametrize("ty", ["f8e4m3fn", "f8e5m2"])
def test_fp8_is_refused_on_gfx942_whose_storage_is_fnuz(tmp_path: Path, ty: str) -> None:
    target = Target.of("gfx942")
    with pytest.raises(TargetError) as error:
        emit(capture(round_trip(tmp_path, ty), "k", target=target), target)
    assert error.value.feature == f"the fp8 conversion of {ty}"
    assert error.value.target == "gfx942"
    assert error.value.reason == "gfx942 stores fp8 as FNUZ, which krn does not model yet"
    assert error.value.instead == "use gfx950 or compute in f16"


def test_gfx950_composes_fp8_and_gfx942_composes_fp6(tmp_path: Path) -> None:
    """Each kernel has its own file: `Module.read` caches a parsed source by path, so a rewritten file is not reread."""
    (tmp_path / "fp8").mkdir()
    (tmp_path / "fp6").mkdir()
    assert emitted(tmp_path / "fp8", "f8e4m3fn", "gfx950")[1] != []
    assert emitted(tmp_path / "fp6", "f6e2m3", "gfx942")[1] != []


def test_sm_121a_stores_fp4_and_fp6_but_composes_until_a_card_confirms_the_cvt(tmp_path: Path) -> None:
    """GB10 is advertised with FP4, so storage stays; the native cvt is claimed only where the PTX text names it."""
    target = Target.of("sm_121a")
    assert target.supports(E2M1) and target.supports(E2M3) and target.supports(E3M2)
    assert target.supports(E4M3FN) and target.supports(E5M2)
    assert not narrow.native(target, fp(2, 1))
    _, warned = emitted(tmp_path, "f4e2m1", "sm_121a")
    assert len(warned) == 2


def test_sm_103a_keeps_fp4_storage_because_ptx_names_it_for_the_rs_form_and_mxf4() -> None:
    assert Target.of("sm_103a").supports(E2M1) and not narrow.native(Target.of("sm_103a"), fp(2, 1))


@pytest.mark.parametrize("ty", TYPES)
def test_a_narrow_value_is_one_byte_of_storage(ty: str) -> None:
    dtype = next(d for d, fmt in narrow.FORMATS.items() if narrow.NAMES[fmt] == ty)
    assert CPURenderer(CPU.of("avx512")).ctype(dtype) == "unsigned char"


ARITHMETIC = {
    "add": "out[i] = T.f32(a[i] + b[i])",
    "subtract": "out[i] = T.f32(a[i] - b[i])",
    "multiply": "out[i] = T.f32(a[i] * b[i])",
    "divide": "out[i] = T.f32(a[i] / b[i])",
    "negate": "out[i] = T.f32(-a[i])",
    "less": "out[i] = T.f32(1.0) if a[i] < b[i] else T.f32(0.0)",
    "equal": "out[i] = T.f32(1.0) if a[i] == b[i] else T.f32(0.0)",
}
ARITHMETIC_REASON = "narrow floats are storage formats: the operation would add the byte codes, not the values"
ARITHMETIC_INSTEAD = "convert with .to(T.f16) or T.f32 first, compute, then convert back"
PRODUCT_INSTEAD = "convert the operands to f16 with .to and multiply, or use a block-scaled QMatrix on sm_120a"


def arithmetic_kernel(directory: Path, ty: str, statement: str) -> Path:
    file = directory / "kernel.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        f"@kernel\ndef k(a: Vector[T.{ty}], b: Vector[T.{ty}], out: Vector[T.f32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n"
        f"        {statement}\n"
    )
    return file


def product_kernel(directory: Path, ty: str) -> Path:
    file = directory / "product.py"
    file.write_text(
        "from krn import T, Matrix, block, zeros, kernel\n@kernel\n"
        f"def f(a: Matrix[T.{ty}], b: Matrix[T.{ty}], c: Matrix[T.f32]) -> None:\n"
        "    rows, cols = block.tile(32, 32)\n    acc = zeros(T.f32, 32, 32)\n"
        "    for ks in range(0, a.shape[1], 32):\n"
        "        acc += a[rows, slice(ks, ks + 32)] @ b[slice(ks, ks + 32), cols]\n"
        "    c[rows, cols] = acc.to(T.f32)\n"
    )
    return file


@pytest.mark.parametrize("operation", ARITHMETIC)
@pytest.mark.parametrize("ty", TYPES)
def test_arithmetic_on_a_narrow_float_is_refused_at_the_operation(tmp_path: Path, ty: str, operation: str) -> None:
    target = Target.of("avx512")
    file = arithmetic_kernel(tmp_path, ty, ARITHMETIC[operation])
    with pytest.raises(TargetError) as error:
        emit(capture(file, "k", target=target), target)
    assert error.value.feature == f"arithmetic on {ty}"
    assert error.value.target == "avx512"
    assert error.value.reason == ARITHMETIC_REASON
    assert error.value.instead == ARITHMETIC_INSTEAD
    assert error.value.span is not None and error.value.span.line == 6


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a", "gfx950", "apple_m4"])
@pytest.mark.parametrize("ty", TYPES)
def test_arithmetic_on_a_narrow_float_is_refused_on_every_target(tmp_path: Path, arch: str, ty: str) -> None:
    target = Target.of(arch)
    with pytest.raises(TargetError) as error:
        emit(capture(arithmetic_kernel(tmp_path, ty, ARITHMETIC["add"]), "k", target=target), target)
    assert error.value.target == arch and error.value.feature == f"arithmetic on {ty}"


@pytest.mark.parametrize("ty", ["f6e2m3", "f6e3m2", "f4e2m1"])
def test_a_plain_product_of_fp6_or_fp4_is_refused_with_the_block_scale_alternative(tmp_path: Path, ty: str) -> None:
    with pytest.raises(TargetError) as error:
        capture(product_kernel(tmp_path, ty), "f", target=Target.of("sm_120a"))
    assert error.value.feature == f"`@` on {ty}"
    assert error.value.target == "sm_120a"
    assert error.value.reason == "fp6 and fp4 matrix units take a block scale, which a plain product does not carry"
    assert error.value.instead == PRODUCT_INSTEAD


@pytest.mark.parametrize("ty", ["f8e4m3fnuz", "f8e5m2fnuz"])
def test_a_plain_product_of_fnuz_fp8_is_refused_with_the_conversion_alternative(tmp_path: Path, ty: str) -> None:
    with pytest.raises(TargetError) as error:
        capture(product_kernel(tmp_path, ty), "f", target=Target.of("gfx950"))
    assert error.value.feature == f"`@` on {ty}"
    assert error.value.reason == "krn does not model the FNUZ convention in a matrix product yet"
    assert error.value.instead == "convert the operands to f16 with .to and multiply"


@pytest.mark.parametrize("arch", ["avx512", "sm_90a", "gfx950", "apple_m4"])
@pytest.mark.parametrize("ty", ["f8e4m3fnuz", "f8e5m2fnuz"])
def test_fnuz_conversions_are_refused_on_every_target(tmp_path: Path, arch: str, ty: str) -> None:
    target = Target.of(arch)
    with pytest.raises(TargetError) as error:
        emit(capture(round_trip(tmp_path, ty), "k", target=target), target)
    assert error.value.feature == f"the fp8 conversion of {ty}"
    assert error.value.target == arch
    assert error.value.reason == f"{ty} is the FNUZ convention, which krn composes on no target yet"
    assert error.value.instead == "convert through f32 to f8e4m3fn or f8e5m2, the OCP formats"


@pytest.mark.parametrize("ty", ["f8e4m3fnuz", "f8e5m2fnuz"])
def test_fnuz_loads_and_stores_stay_native_as_one_byte(tmp_path: Path, ty: str) -> None:
    file = tmp_path / "copy.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n@kernel\n"
        f"def k(a: Vector[T.{ty}], out: Vector[T.{ty}]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = a[i]\n"
    )
    target = Target.of("avx512")
    assert "unsigned char" in emit(capture(file, "k", target=target), target).source
