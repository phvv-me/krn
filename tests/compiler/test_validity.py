"""A lane read past the edge of its tensor is invalid, not zero: a reduction skips it, an elementwise operation keeps it
invalid, a matrix product counts it as zero and a store leaves it alone.

The C backend (one valid interval per axis, in C expressions) and the emulator (one boolean per lane) implement this
independently. Each must equal numpy computed on the unpadded arrays, and so equal each other: a lane of zero where the
reduction needed -inf, +inf or 1 is exactly what these data expose.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from krn import Tiles, targets
from krn.compiler.api import capture
from krn.emulator.emulate import build
from tests.conftest import KERNELS, build_kernel

FILE = KERNELS / "ragged.py"
GPU = targets.Target.of("sm_90a")
RNG = np.random.default_rng(5)
TILE = (8, 16)
SENTINEL = 99.0


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Ragged:
    """A kernel, its arguments (outputs start as NaN or the sentinel) and what numpy says the outputs hold."""

    name: str
    arguments: Callable[[], list[np.ndarray]]
    launch: Tiles
    expected: Callable[[list[np.ndarray]], list[tuple[int, np.ndarray]]]


def negative(shape: tuple[int, ...]) -> np.ndarray:
    return -RNG.uniform(1.0, 5.0, shape).astype(np.float32)


def positive(shape: tuple[int, ...]) -> np.ndarray:
    return RNG.uniform(1.0, 5.0, shape).astype(np.float32)


def nan(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, np.float32)


X = (5, 11)  # smaller than the 8 x 16 tile on both axes
ROWS = Tiles(X, TILE)
CASES = [
    Ragged(
        "row_max",
        lambda: [negative(X), nan((5, 1))],
        ROWS,
        lambda a: [(1, a[0].max(axis=1, keepdims=True))],
    ),
    Ragged(
        "row_min",
        lambda: [positive(X), nan((5, 1))],
        ROWS,
        lambda a: [(1, a[0].min(axis=1, keepdims=True))],
    ),
    Ragged(
        "row_sum",
        lambda: [positive(X), nan((5, 1))],
        ROWS,
        lambda a: [(1, a[0].sum(axis=1, keepdims=True))],
    ),
    Ragged(
        "row_prod",
        lambda: [positive(X) / 3 + 0.8, nan((5, 1))],
        ROWS,
        lambda a: [(1, a[0].prod(axis=1, keepdims=True))],
    ),
    Ragged("softmax_rows", lambda: [negative(X), nan(X)], ROWS, lambda a: [(1, softmax(a[0]))]),
    Ragged(
        "centered",
        lambda: [negative(X), nan(X)],
        ROWS,
        lambda a: [(1, a[0] - a[0].max(axis=1, keepdims=True))],
    ),
    Ragged(  # `out` is larger than the operands, so the lanes beyond them must keep their sentinel
        "larger_target",
        lambda: [positive(X), positive(X), np.full((9, 20), SENTINEL, np.float32)],
        Tiles((9, 20), TILE),
        lambda a: [(2, np.pad(np.maximum(a[0], a[1]) + 1, ((0, 4), (0, 9)), constant_values=SENTINEL))],
    ),
    Ragged(  # products of positive and negative data are negative, so a zero-padded column would win the row maximum
        "scores_max",
        lambda: [positive((5, 11)), negative((11, 13)), nan((5, 1))],
        Tiles((5, 13), TILE),
        lambda a: [(2, (a[0] @ a[1]).max(axis=1, keepdims=True))],
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_both_backends_equal_numpy_on_the_valid_lanes_only(case: Ragged) -> None:
    compiled, emulated = case.arguments(), None
    emulated = [a.copy() for a in compiled]
    reference = case.expected([a.copy() for a in compiled])
    build_kernel("ragged", case.name)(*compiled, launch=case.launch)
    build(capture(FILE, case.name, target=GPU), GPU).run(case.launch, emulated)
    for index, want in reference:
        np.testing.assert_allclose(compiled[index], want, rtol=2e-5, atol=1e-6, err_msg="C backend")
        np.testing.assert_allclose(emulated[index], want, rtol=2e-5, atol=1e-6, err_msg="emulator")
