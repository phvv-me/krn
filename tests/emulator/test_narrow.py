"""The emulator's narrow conversions: the same bits as the C backend on every format, and the FNUZ refusal."""

import math
from pathlib import Path

import numpy as np
import pytest

from krn import Blocks, Items
from krn.compiler import narrow
from krn.compiler.api import capture
from krn.compiler.errors import TargetError
from krn.compiler.lower import emit
from krn.data.formats import E2M1, E2M3, E3M2, E4M3FN, E5M2, FloatFormat
from krn.emulator.emulate import build
from krn.facts.targets import Target
from krn.runtime.cpu import CPUProgram

FORMATS = {"f8e4m3fn": E4M3FN, "f8e5m2": E5M2, "f6e2m3": E2M3, "f6e3m2": E3M2, "f4e2m1": E2M1}
GPU, CPU = Target.of("sm_90a"), Target.of("avx512")


def round_trip(directory: Path, ty: str) -> Path:
    file = directory / "kernel.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "@kernel\ndef k(a: Vector[T.f32], out: Vector[T.f32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n"
        f"        out[i] = a[i].to(T.{ty}).to(T.f32)\n"
    )
    return file


def probes(fmt: FloatFormat, count: int = 200_000) -> np.ndarray:
    """Float32 inputs at every rounding boundary (both sides of each midpoint), specials, and random bit patterns."""
    table = np.asarray(fmt.table, np.float64)
    levels = np.unique(np.abs(table[np.isfinite(table)]))
    mids = ((levels[:-1] + levels[1:]) / 2).astype(np.float32)
    mids = np.concatenate([mids, -mids, levels.astype(np.float32), -levels.astype(np.float32)])
    bits = mids.view(np.uint32)
    near = np.concatenate([bits + np.uint32(step) for step in (0, 1, 0xFFFFFFFF)]).view(np.float32)
    random = np.random.default_rng(3).integers(0, 2**32, count, dtype=np.uint64).astype(np.uint32).view(np.float32)
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], np.float32)
    values = np.concatenate([near, random, specials])
    return np.resize(values, -(-len(values) // 64) * 64)


@pytest.mark.parametrize("ty", FORMATS)
def test_emulator_round_trip_matches_the_c_backend_bit_for_bit(tmp_path: Path, ty: str) -> None:
    values = probes(FORMATS[ty])
    compiled = np.zeros_like(values)
    ir = capture(round_trip(tmp_path, ty), "k", target=CPU)
    CPUProgram(ir, emit(ir, CPU))(values, compiled, launch=Blocks(1, threads=len(values)))
    emulated = np.zeros_like(values)
    build(capture(round_trip(tmp_path, ty), "k", target=GPU), GPU).run(Items(len(values)), [values, emulated])
    nan = np.isnan(compiled)
    np.testing.assert_array_equal(np.isnan(emulated), nan)
    np.testing.assert_array_equal(emulated[~nan].view(np.uint32), compiled[~nan].view(np.uint32))


@pytest.mark.parametrize("ty", FORMATS)
def test_emulator_round_trip_is_the_table_reference(tmp_path: Path, ty: str) -> None:
    fmt = FORMATS[ty]
    values = probes(fmt, count=4_096)
    emulated = np.zeros_like(values)
    build(capture(round_trip(tmp_path, ty), "k", target=GPU), GPU).run(Items(len(values)), [values, emulated])
    want = narrow.reference_decode(fmt, narrow.reference_encode(fmt, values))
    live = ~np.isnan(want)
    np.testing.assert_array_equal(emulated[live].view(np.uint32), want[live].view(np.uint32))


def test_emulator_saturates_and_maps_nan_to_the_canonical_code(tmp_path: Path) -> None:
    fmt = E4M3FN
    values = np.array([1e9, -1e9, np.inf, np.nan, 0.1], np.float32)
    emulated = np.zeros_like(values)
    build(capture(round_trip(tmp_path, "f8e4m3fn"), "k", target=GPU), GPU).run(Items(len(values)), [values, emulated])
    assert emulated[0] == 448.0 and emulated[1] == -448.0 and emulated[2] == 448.0
    assert math.isnan(emulated[3]) and emulated[4] == np.float32(0.1015625)
    assert narrow.nan_code(fmt) == 0x7F


def test_emulator_refuses_fp8_on_gfx942_with_the_same_text_as_the_compiler(tmp_path: Path) -> None:
    target = Target.of("gfx942")
    ir = capture(round_trip(tmp_path, "f8e5m2"), "k", target=target)
    with pytest.raises(TargetError) as error:
        build(ir, target)
    assert error.value.feature == "the fp8 conversion of f8e5m2"
    assert error.value.target == "gfx942"
    assert error.value.reason == "gfx942 stores fp8 as FNUZ, which krn does not model yet"
    assert error.value.instead == "use gfx950 or compute in f16"


def test_emulator_runs_gfx950_fp8_as_ocp(tmp_path: Path) -> None:
    target = Target.of("gfx950")
    values = np.array([1e9, 0.1], np.float32)
    emulated = np.zeros_like(values)
    build(capture(round_trip(tmp_path, "f8e4m3fn"), "k", target=target), target).run(
        Items(len(values)), [values, emulated]
    )
    assert emulated[0] == 448.0 and emulated[1] == np.float32(0.1015625)


@pytest.mark.parametrize("ty", FORMATS)
def test_emulator_refuses_arithmetic_on_a_narrow_float_with_the_compiler_text(tmp_path: Path, ty: str) -> None:
    file = tmp_path / "add.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        f"@kernel\ndef k(a: Vector[T.{ty}], b: Vector[T.{ty}], out: Vector[T.f32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = T.f32(a[i] + b[i])\n"
    )
    ir = capture(file, "k", target=GPU)
    with pytest.raises(TargetError) as compiled:
        emit(ir, GPU)
    with pytest.raises(TargetError) as emulated:
        build(ir, GPU)
    for field in ("feature", "target", "reason", "instead"):
        assert getattr(emulated.value, field) == getattr(compiled.value, field)
    span = emulated.value.span
    assert span is not None and span.line == 6 and span == compiled.value.span
