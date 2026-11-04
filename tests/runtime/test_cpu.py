"""The whole frontend against numpy: capture, check, lower to C, compile with clang and run on random inputs."""

import numpy as np
import pytest

from krn import Blocks, Items
from krn.arithmetic import ceildiv
from krn.compiler.dtypes import Arithmetic
from tests.conftest import build_kernel

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("n", [1, 255, 256, 1000])
def test_saxpy(n: int) -> None:
    x, y = RNG.standard_normal(n, dtype=np.float32), RNG.standard_normal(n, dtype=np.float32)
    expected = np.float32(2.5) * x + y
    build_kernel("elementwise", "saxpy")(2.5, x, y, launch=Items(n))
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)  # clang fuses a * x + y into one fma


@pytest.mark.parametrize(("shape", "tile"), [((100, 70), 32), ((5, 3), 128), ((64, 64), 32)])
def test_tile_add_with_ragged_edges(shape: tuple[int, int], tile: int) -> None:
    a, b = (
        RNG.standard_normal(shape, dtype=np.float32),
        RNG.standard_normal(shape, dtype=np.float32),
    )
    c = np.full(shape, np.nan, dtype=np.float32)
    program = build_kernel("elementwise", "add", {"tiling": {"rows": tile, "cols": tile}})
    program(a, b, c, {}, launch=Blocks(ceildiv(shape[1], tile), ceildiv(shape[0], tile)))
    np.testing.assert_array_equal(c, a + b)


def softmax_reference(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32)
    e = np.exp(v - v.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float16)


@pytest.mark.parametrize(("rows", "cols", "tile_rows"), [(7, 64, 4), (6, 4096, 4), (1, 16, 4)])
def test_softmax_f16_with_f32_math(rows: int, cols: int, tile_rows: int) -> None:
    x = RNG.standard_normal((rows, cols)).astype(np.float16)
    y = np.zeros((rows, cols), dtype=np.float16)
    program = build_kernel("softmax", "softmax", {"t": {"rows": tile_rows, "cols": cols}})
    program(x, y, {}, launch=Blocks(ceildiv(rows, tile_rows)))
    np.testing.assert_allclose(y.astype(np.float32), softmax_reference(x).astype(np.float32), rtol=2e-3, atol=1e-4)


def test_softmax_of_ragged_columns_ignores_the_lanes_past_the_edge() -> None:
    """A 4 x 50 matrix in a 4 x 64 tile: the 14 lanes past the edge are invalid and add nothing to a denominator."""
    x = RNG.standard_normal((4, 50)).astype(np.float16) - 5
    y = np.zeros((4, 50), dtype=np.float16)
    build_kernel("softmax", "softmax", {"t": {"rows": 4, "cols": 64}})(x, y, {}, launch=Blocks(1))
    np.testing.assert_allclose(y.astype(np.float32), softmax_reference(x).astype(np.float32), rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize(
    ("m", "n", "k", "tiles"),
    [
        (70, 50, 45, (32, 32, 16)),
        (33, 17, 5, (16, 16, 8)),
        (64, 64, 64, (128, 128, 32)),
        (3, 130, 100, (128, 128, 32)),
    ],
)
def test_gemm_f16_accumulates_in_f32(m: int, n: int, k: int, tiles: tuple[int, int, int]) -> None:
    a, b = (
        RNG.standard_normal((m, k)).astype(np.float16),
        RNG.standard_normal((k, n)).astype(np.float16),
    )
    c = np.full((m, n), np.nan, dtype=np.float32)
    bm, bn, bk = tiles
    program = build_kernel("gemm", "gemm", {"t": {"bm": bm, "bn": bn, "bk": bk}})
    program(a, b, c, {}, launch=Blocks(ceildiv(n, bn), ceildiv(m, bm)))
    np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-4, atol=1e-4)


def test_run_dfa_with_struct_arguments_is_exact() -> None:
    states, segments = 7, 61
    table = RNG.integers(0, states, size=states * 256).astype(np.uint16)
    accept = RNG.integers(0, 2, size=states).astype(np.uint8)
    text = RNG.integers(0, 256, size=4000).astype(np.uint8)
    offsets = np.sort(np.concatenate([[0, len(text)], RNG.integers(0, len(text), size=segments - 1)])).astype(np.uint32)
    out = np.full(segments, 99, dtype=np.uint32)
    start = 2
    expected = np.zeros(segments, dtype=np.uint32)
    for i in range(segments):
        state = start
        for byte in text[offsets[i] : offsets[i + 1]]:
            state = int(table[state * 256 + int(byte)])
        expected[i] = accept[state]
    program = build_kernel("automaton", "run_dfa")
    program(
        {"table": table, "accept": accept, "start": start},
        {"text": text, "offsets": offsets, "out": out},
        launch=Items(segments),
    )
    np.testing.assert_array_equal(out, expected)


def test_integer_arithmetic_wraps_in_its_own_width() -> None:
    n = 1000
    x = RNG.integers(0, 256, size=n).astype(np.uint8)
    y, z, w = np.zeros(n, np.uint8), np.zeros(n, np.uint16), np.zeros(n, np.uint32)
    build_kernel("extra", "wrap")(x, y, z, w, launch=Items(n))
    np.testing.assert_array_equal(y, (x + x + np.uint8(100)).astype(np.uint8))
    np.testing.assert_array_equal(z, (x.astype(np.uint16) * np.uint16(300) - np.uint16(1)).astype(np.uint16))
    np.testing.assert_array_equal(w, z.astype(np.uint32) * np.uint32(70000) + x.astype(np.uint32))


def test_lossless_widening_is_exact() -> None:
    n = 500
    x, z = (
        RNG.integers(0, 256, size=n).astype(np.uint8),
        RNG.integers(0, 65536, size=n).astype(np.uint16),
    )
    w, f = RNG.integers(0, 2**31, size=n).astype(np.uint32), np.zeros(n, np.float32)
    expected_w = ((x.astype(np.uint32) + z.astype(np.uint32)) + w).astype(np.uint32)
    build_kernel("extra", "widen", arithmetic=Arithmetic.LOSSLESS)(x, z, w, f, launch=Items(n))
    np.testing.assert_array_equal(w, expected_w)
    np.testing.assert_array_equal(f, w.astype(np.float32) * np.float32(0.5))


def test_wrong_numpy_dtype_is_rejected_at_the_call() -> None:
    program = build_kernel("elementwise", "saxpy")
    with pytest.raises(TypeError, match="C-contiguous f32 array of rank 1"):
        program(1.0, np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float32), launch=Items(4))


def test_while_loop_and_branches() -> None:
    x = np.arange(1, 200, dtype=np.uint32)
    steps = np.zeros_like(x)

    def collatz(n: int) -> int:
        count = 0
        while n > 1:
            n = n // 2 if n % 2 == 0 else n * 3 + 1
            count += 1
        return count

    build_kernel("extra", "collatz")(x, steps, launch=Items(len(x)))
    np.testing.assert_array_equal(steps, [collatz(int(n)) for n in x])


def test_user_names_that_are_c_keywords_or_launch_variables_are_renamed() -> None:
    x = np.zeros(5, dtype=np.float32)
    build_kernel("extra", "shadow")(x, launch=Items(5))
    np.testing.assert_array_equal(x, np.ones(5, dtype=np.float32))


def test_branch_on_the_target_folds_and_forward_references_resolve() -> None:
    out = np.zeros(4, dtype=np.uint32)
    build_kernel("extra", "lanes", warp_size=16)(out, launch=Items(4))
    np.testing.assert_array_equal(out, [16] * 4)
    x, y = np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32)
    build_kernel("extra", "scale", {"cfg": {"factor": 3.0}})(x, y, {"shift": 0.5}, launch=Items(10))
    np.testing.assert_allclose(y, x * 3 + 0.5)
