"""Tile operations against numpy on the emulator and the C backend, with tensors whose edges cut the tiles, and the
flash attention example against a plain numpy softmax. Each kernel is compiled once and run on every drawn shape."""

import math
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks
from krn.compiler.api import CaptureError, capture
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from tests.conftest import EXAMPLE_ROOT, KERNELS, build_kernel
from tests.helpers.text import raises_literal

FILE = KERNELS / "tile_ops.py"
FLASH = EXAMPLE_ROOT / "04_libraries" / "08_flash_attention.py"
GPU, APPLE = Target.of("sm_90a"), Target.of("apple_m4")
CONSTS = {"t": {"rows": 8, "cols": 16}}  # the tile of every kernel in tile_ops.py
ONE_BLOCK = Blocks(1)
SEED = st.integers(0, 2**32 - 1)
SIZE = st.tuples(st.integers(1, 8), st.integers(1, 16))  # no larger than the tile, so the tensor's edge cuts the tile


@cache
def compiled(name: str) -> Callable[..., None]:
    """The C backend's program for one kernel of tile_ops.py, compiled once for all the shapes a test draws."""
    return build_kernel(FILE, name, CONSTS)


def on_both(
    name: str, arrays: list[np.ndarray], launch: Blocks = ONE_BLOCK
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run a kernel of tile_ops.py over copies of `arrays` on the emulator and on the C backend, and return both."""
    emulated = [array.copy() for array in arrays]
    with np.errstate(all="ignore"):
        emulate(capture(FILE, name, consts=CONSTS, target=GPU), GPU).run(launch, [{}, *emulated])
    native = [array.copy() for array in arrays]
    compiled(name)({}, *native, launch=launch)
    return emulated, native


def random(rng: np.random.Generator, shape: tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    return rng.standard_normal(shape).astype(dtype)


@settings(max_examples=12, deadline=None)
@given(size=SIZE, seed=SEED)
def test_the_transpose_of_a_ragged_tile_is_numpy_on_both_backends(size: tuple[int, int], seed: int) -> None:
    x = random(np.random.default_rng(seed), size)
    for result in on_both("transpose", [x, np.full(size[::-1], -7.0, np.float32)]):
        np.testing.assert_array_equal(result[1], x.T)


@settings(max_examples=12, deadline=None)
@given(size=SIZE, seed=SEED)
def test_where_on_a_comparison_selects_elementwise_on_both_backends(size: tuple[int, int], seed: int) -> None:
    x = random(np.random.default_rng(seed), size)
    expected = np.where(x > 0, x, np.float32(0.5))
    for result in on_both("select", [x, np.zeros(size, np.float32)]):
        np.testing.assert_array_equal(result[1], expected)


@settings(max_examples=12, deadline=None)
@given(size=SIZE, seed=SEED)
def test_the_four_comparisons_give_exact_indicators_on_both_backends(size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.round(random(rng, size)).astype(np.float32)  # many exact zeros, so the strict and loose forms differ
    outputs = [np.zeros(size, np.float32) for _ in range(4)]
    expected = [(x < 0), (x <= 0), (x > 0), (x >= 0)]
    for result in on_both("compare", [x, *outputs]):
        for got, want in zip(result[1:], expected, strict=True):
            np.testing.assert_array_equal(got, want.astype(np.float32))


@settings(max_examples=12, deadline=None)
@given(size=SIZE)
def test_arange_along_two_axes_broadcasts_to_a_grid_of_positions_on_both_backends(size: tuple[int, int]) -> None:
    rows, cols = size
    expected = np.arange(cols)[None, :] - (np.arange(rows)[:, None] + 3)
    for result in on_both("positions", [np.zeros(size, np.int32)]):
        np.testing.assert_array_equal(result[0], expected.astype(np.int32))


@settings(max_examples=12, deadline=None)
@given(size=SIZE, seed=SEED)
def test_a_bare_product_is_a_fresh_accumulator_reading_a_transposed_operand(size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, w = random(rng, size, np.float16), random(rng, (8, size[1]), np.float16)
    expected = x.astype(np.float32) @ w.astype(np.float32).T
    for result in on_both("product", [x, w, np.zeros((8, 8), np.float32)]):
        np.testing.assert_allclose(result[2][: size[0]], expected, rtol=1e-5, atol=1e-5)
        assert not result[2][size[0] :].any()  # rows past the tensor's edge are invalid, so they are not stored


@settings(max_examples=12, deadline=None)
@given(rows=st.integers(1, 8), cols=st.integers(1, 16), heads=st.integers(1, 3), seed=SEED)
def test_a_scalar_head_index_with_two_slices_is_a_rank_two_tile_on_both_backends(
    rows: int, cols: int, heads: int, seed: int
) -> None:
    x = random(np.random.default_rng(seed), (heads, rows, cols))
    for result in on_both("per_head", [x, np.zeros_like(x)], Blocks(1, heads)):
        np.testing.assert_array_equal(result[1], 2 * x)


@settings(max_examples=12, deadline=None)
@given(size=SIZE, seed=SEED)
def test_a_causal_where_keeps_the_lower_triangle_on_both_backends(size: tuple[int, int], seed: int) -> None:
    x = random(np.random.default_rng(seed), size)
    keep = np.arange(size[1])[None, :] <= np.arange(size[0])[:, None]
    for result in on_both("causal", [x, np.zeros(size, np.float32)]):
        np.testing.assert_array_equal(result[1], np.where(keep, x, np.float32(-1.0e30)))


def numpy_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    """softmax(Q K^T / sqrt(d)) V with a plain max-shifted softmax, in float64."""
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    s = q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape[-2:], bool)), s, -np.inf)
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)) @ v


@settings(max_examples=6, deadline=None)
@given(
    seq=st.integers(1, 40),
    heads=st.integers(1, 2),
    d=st.sampled_from((8, 16)),
    causal=st.booleans(),
    seed=SEED,
)
def test_flash_attention_matches_numpy_on_both_backends_for_ragged_sequences(
    seq: int, heads: int, d: int, causal: bool, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    q, k, v = (random(rng, (heads, seq, d), np.float16) for _ in range(3))
    consts = {"t": {"bq": 8, "bk": 8, "d": d, "causal": causal}}
    scale = math.log2(math.e) / math.sqrt(d)
    launch = Blocks(-(-seq // 8), heads)
    want = numpy_attention(q, k, v, causal)
    emulated = np.zeros((heads, seq, d), np.float16)
    with np.errstate(all="ignore"):
        kernel = capture(FLASH, "flash_attention", consts=consts, target=GPU)
        emulate(kernel, GPU).run(launch, [{}, q, k, v, emulated, scale])
    native = np.zeros((heads, seq, d), np.float16)
    build_kernel(FLASH, "flash_attention", consts)({}, q, k, v, native, scale, launch=launch)
    for got in (emulated, native):
        np.testing.assert_allclose(got.astype(np.float64), want, atol=3e-3)


@pytest.mark.hardware
def test_the_tile_operations_on_the_apple_gpu_match_numpy(metal_program) -> None:
    rng = np.random.default_rng(0)
    x = random(rng, (5, 11))
    y, s = np.zeros((11, 5), np.float32), np.zeros((5, 11), np.float32)
    metal_program(capture(FILE, "transpose", consts=CONSTS, target=APPLE)).run(Blocks(1), [{}, x, y])
    np.testing.assert_array_equal(y, x.T)
    metal_program(capture(FILE, "select", consts=CONSTS, target=APPLE)).run(Blocks(1), [{}, x, s])
    np.testing.assert_array_equal(s, np.where(x > 0, x, np.float32(0.5)))


@pytest.mark.hardware
def test_flash_attention_on_the_apple_gpu_matches_numpy(metal_program) -> None:
    rng = np.random.default_rng(1)
    heads, seq, d = 2, 37, 16
    q, k, v = (random(rng, (heads, seq, d), np.float16) for _ in range(3))
    out = np.zeros((heads, seq, d), np.float16)
    consts = {"t": {"bq": 8, "bk": 8, "d": d, "causal": True}}
    scale = math.log2(math.e) / math.sqrt(d)
    program = metal_program(capture(FLASH, "flash_attention", consts=consts, target=APPLE))
    program.run(Blocks(-(-seq // 8), heads), [{}, q, k, v, out, scale])
    np.testing.assert_allclose(out.astype(np.float64), numpy_attention(q, k, v, True), atol=3e-3)


def captured(tmp_path: Path, body: str) -> None:
    """Capture one kernel whose body is `body`, over a tile of a rank-1 and a rank-2 tensor."""
    source = (
        "from krn import Matrix, T, arange, kernel, where, zeros\n\n\n@kernel\n"
        "def bad(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:\n    t = x[0:8, 0:16]\n"
        + "".join(f"    {line}\n" for line in body.splitlines())
    )
    path = tmp_path / "bad.py"
    path.write_text(source)
    capture(path, "bad", target=GPU)


def test_transpose_refuses_a_tile_that_is_not_a_matrix(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "`transpose()` needs a rank-2 tile"):
        captured(tmp_path, "y[0:8, 0:16] = zeros(T.f32, 8).transpose()")


def test_where_refuses_a_condition_that_is_not_a_comparison(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "`where` needs a bool condition"):
        captured(tmp_path, "y[0:8, 0:16] = where(x[0:8, 0:16], x[0:8, 0:16], 0.0)")


def test_arange_refuses_an_axis_past_its_rank(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "`arange` axis 2 is out of range for rank 2"):
        captured(tmp_path, "y[0:8, 0:16] = arange(8, 2, rank=2)")


@settings(max_examples=12, deadline=None)
@given(size=SIZE, seed=SEED)
def test_a_tile_shape_axis_is_its_static_extent_on_both_backends(size: tuple[int, int], seed: int) -> None:
    x = random(np.random.default_rng(seed), size)
    for result in on_both("scaled_by_extent", [x, np.zeros(size, np.float32)]):
        np.testing.assert_array_equal(result[1], x * np.float32(CONSTS["t"]["cols"]))


def test_transpose_takes_no_arguments(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "`transpose()` takes no arguments"):
        captured(tmp_path, "y[0:8, 0:16] = t.transpose(0)")


def test_where_refuses_branches_of_different_kinds(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "`where` branches have types"):
        captured(tmp_path, "y[0:8, 0:16] = where(t > 0.0, t, t.to(T.i32))")


def test_a_product_refuses_operands_that_are_not_matrices(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "`@` needs two rank-2 tiles"):
        captured(tmp_path, "y[0:8, 0:16] = zeros(T.f32, 8) @ t")


def test_a_tile_shape_refuses_an_axis_it_does_not_have(tmp_path: Path) -> None:
    with raises_literal(CaptureError, "has 2 axes, no axis 2"):
        captured(tmp_path, "y[0:8, 0:16] = t * t.shape[2]")
