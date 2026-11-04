"""The scalar matrix fallback fuses each product into its accumulator, the one rounding a matrix instruction makes.

`(1 + 2**-12) ** 2` is `1 + 2**-11 + 2**-24`, which f32 cannot hold: a separate product rounds to `1 + 2**-11`, which
cancels `-(1 + 2**-11)` to zero, while the fused product keeps the `2**-24`. The emulator is the oracle, and the C
backend and the Apple GPU must agree with it bit for bit. The C backend runs one thread per block: its barrier is empty,
so staged tiles are only correct without concurrent threads.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks
from krn.compiler.api import capture, emit
from krn.compiler.dtypes import F32
from krn.compiler.emit import LIBM, Emitter
from krn.compiler.errors import TargetError
from krn.compiler.ir import Accumulate, Fn, Kernel
from krn.compiler.msl import AppleSimdgroup
from krn.compiler.validity import walk
from krn.emulator.emulate import build as emulate
from krn.emulator.emulate import fused
from krn.facts import catalog
from krn.facts.targets import Target
from krn.runtime.cpu import CPUProgram, CPURenderer
from tests.compiler.test_tensor_cores import fused_loop, source_of
from tests.helpers.devices import MetalRun

CPU = Target.of("avx512")
APPLE = Target.of("apple_m4")
ONE = 1 + 2.0**-12
CANCELLED = np.float32(2.0**-24)
GEMM = """
from krn import Matrix, block, kernel, zeros
from krn.frontend.types import f32

@kernel
def gemm(a: Matrix[f32], b: Matrix[f32], c: Matrix[f32]) -> None:
    rows, cols = block.tile(8, 8)
    acc = zeros(f32, 8, 8)
    acc += a[rows, slice(0, 2)] @ b[slice(0, 2), cols]
    c[rows, cols] = acc
"""


@pytest.fixture
def cancelling(tmp_path: Path) -> Path:
    file = tmp_path / "gemm.py"
    file.write_text(GEMM)
    return file


def cancelling_operands() -> tuple[np.ndarray, np.ndarray]:
    """Row `[-(1 + 2**-11), 1 + 2**-12]` against column `[1, 1 + 2**-12]`: the first product is exact."""
    a = np.tile(np.array([-(1 + 2.0**-11), ONE], np.float32), (8, 1))
    b = np.tile(np.array([[1.0], [ONE]], np.float32), (1, 8))
    return a, b


def same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_the_cancelling_product_stays_fused_on_the_cpu(cancelling: Path) -> None:
    a, b = cancelling_operands()
    kernel = capture(cancelling, "gemm", dtypes={"a": F32, "b": F32, "c": F32}, target=CPU)
    expected = np.zeros((8, 8), np.float32)
    emulate(kernel, CPU).run(Blocks(1, threads=1), [a.copy(), b.copy(), expected])
    actual = np.zeros((8, 8), np.float32)
    CPUProgram(kernel, emit(kernel, CPU))(a, b, actual, launch=Blocks(1, threads=1))
    assert np.all(expected == CANCELLED)
    same_bits(actual, expected)


def test_the_cancelling_product_stays_fused_on_the_apple_gpu(
    cancelling: Path, metal_program: Callable[[Kernel], MetalRun]
) -> None:
    a, b = cancelling_operands()
    kernel = capture(cancelling, "gemm", dtypes={"a": F32, "b": F32, "c": F32}, target=APPLE)
    expected = np.zeros((8, 8), np.float32)
    emulate(kernel, APPLE).run(Blocks(1, threads=64), [a.copy(), b.copy(), expected])
    actual = np.zeros((8, 8), np.float32)
    metal_program(kernel).run(Blocks(1, threads=64), [a.copy(), b.copy(), actual])
    assert np.all(expected == CANCELLED)
    same_bits(actual, expected)


@pytest.mark.parametrize("size", [32, 64])
def test_random_operands_match_the_emulator_on_the_apple_gpu(
    tmp_path: Path, metal_program: Callable[[Kernel], MetalRun], size: int
) -> None:
    """The 64 x 64 tile with f32 operands takes the memory-limited path that reads operands directly; 32 stages them."""
    kernel = capture(source_of(tmp_path / "gemm.py", size, size, size, "f32", generic=True), "gemm", target=APPLE)
    rng = np.random.default_rng(size)
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    expected, actual = (np.zeros((size, size), np.float32) for _ in range(2))
    emulate(kernel, APPLE).run(Blocks(1, threads=64), [a.copy(), b.copy(), expected])
    metal_program(kernel).run(Blocks(1, threads=64), [a.copy(), b.copy(), actual])
    same_bits(actual, expected)


def test_the_emitted_source_fuses_the_staged_and_the_direct_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct: list[Accumulate] = []
    original = Emitter._direct_product

    def record(self: Emitter, s: Accumulate) -> None:
        direct.append(s)
        original(self, s)

    monkeypatch.setattr(Emitter, "_direct_product", record)
    staged = capture(source_of(tmp_path / "staged.py", 32, 32, 32, "f32", generic=True), "gemm", target=APPLE)
    source = emit(staged, APPLE).source
    assert fused_loop(source) and "_s +=" not in source and not direct
    memory = capture(source_of(tmp_path / "direct.py", 64, 64, 64, "f32", generic=True), "gemm", target=APPLE)
    source = emit(memory, APPLE).source
    assert fused_loop(source) and "_s +=" not in source and len(direct) == 1


def test_the_cpu_fallback_fuses_the_product(cancelling: Path) -> None:
    kernel = capture(cancelling, "gemm", dtypes={"a": F32, "b": F32, "c": F32}, target=CPU)
    source = emit(kernel, CPU).source
    assert fused_loop(source) and "_s +=" not in source


def test_a_target_without_a_fused_multiply_add_refuses_the_fallback(
    cancelling: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = capture(cancelling, "gemm", dtypes={"a": F32, "b": F32, "c": F32}, target=CPU)
    monkeypatch.setattr(CPURenderer, "functions", {fn: name for fn, name in LIBM.items() if fn is not Fn.FMA})
    with pytest.raises(TargetError, match="no fused multiply-add for f32") as caught:
        emit(kernel, CPU)
    assert "use a target with a fused f32 multiply-add" in caught.value.instead


FINITE = st.floats(width=32, allow_nan=False, allow_infinity=False, min_value=-(2.0**60), max_value=2.0**60)


def exact_fma(a: float, b: float, c: float) -> np.float32:
    """The single rounding to f32 of `a * b + c`, by the catalog's exact reference."""
    reference = catalog.BY_NAME["fma"].reference
    assert reference is not None
    value = reference(a, b, c, bits=32, kind="f")
    assert isinstance(value, float)
    return np.float32(value)


@given(FINITE, FINITE, st.integers(-3, 3))
@settings(max_examples=400, deadline=None)
def test_the_emulator_fma_rounds_once_like_the_reference(a: float, b: float, steps: int) -> None:
    """`c` sits within a few f32 ulps of `-(a * b)`, where a separate product and sum differ from the fused result."""
    c = np.float32(-np.float32(np.float64(a) * np.float64(b)))
    for _ in range(abs(steps)):
        c = np.nextafter(c, np.float32(np.copysign(np.inf, steps)))
    expected = exact_fma(float(a), float(b), float(c))
    same_bits(np.asarray(fused(np.float32(a), np.float32(b), c)), np.asarray(expected))


def test_non_finite_and_float64_operands_keep_the_reference_rounding() -> None:
    assert np.isinf(fused(np.float32(np.inf), np.float32(1.0), np.float32(0.0)))
    assert np.isnan(fused(np.float32(np.nan), np.float32(1.0), np.float32(0.0)))
    assert fused(np.float64(ONE), np.float64(ONE), np.float64(-(1 + 2.0**-11))) == 2.0**-24


def test_the_emulator_fma_agrees_with_the_reference_elementwise() -> None:
    rng = np.random.default_rng(3)
    a, b = ((rng.standard_normal((6, 5)) * 10.0 ** rng.integers(-15, 15, (6, 5))).astype(np.float32) for _ in range(2))
    c = -(a.astype(np.float64) * b.astype(np.float64)).astype(np.float32) * np.float32(1 + 2.0**-20)
    expected = np.array(
        [
            [exact_fma(x, y, z) for x, y, z in zip(ra, rb, rc, strict=True)]
            for ra, rb, rc in zip(a.tolist(), b.tolist(), c.tolist(), strict=True)
        ],
        np.float32,
    )
    same_bits(np.asarray(fused(a, b, c)), expected)


def test_the_fallback_reason_names_the_fused_loop(tmp_path: Path) -> None:
    kernel = capture(source_of(tmp_path / "gemm.py", 32, 32, 32, "f32", generic=True), "gemm", target=APPLE)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    reason = AppleSimdgroup().plan(statement, APPLE)
    assert isinstance(reason, str) and "scalar loop of fused multiply-adds, one rounding per product" in reason
