"""Tile kernels whose tiles run past the edge of their tensors: the lanes beyond it are invalid, not zero."""

from krn import Const, Matrix, T, block, kernel, maximum, pipelined, struct, zeros


@struct
class Width:
    cols: Const[int] = 16


@kernel
def row_max(x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    out[rows, slice(0, 1)] = x[rows, cols].max(axis=1, keepdims=True)


@kernel
def row_min(x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    out[rows, slice(0, 1)] = x[rows, cols].min(axis=1, keepdims=True)


@kernel
def row_sum(x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    out[rows, slice(0, 1)] = x[rows, cols].sum(axis=1, keepdims=True)


@kernel
def row_prod(x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    out[rows, slice(0, 1)] = x[rows, cols].prod(axis=1, keepdims=True)


@kernel
def softmax_rows(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    y[rows, cols] = x[rows, cols].softmax(axis=1)


@kernel
def centered(x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    """A tile variable carries the validity of what it was assigned, so the maximum below skips the ragged lanes."""
    rows, cols = block.tile(8, 16)
    v = x[rows, cols]
    out[rows, cols] = v - v.max(axis=1, keepdims=True)


@kernel
def larger_target(x: Matrix[T.f32], y: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    """`out` is larger than `x`; the lanes of the sum beyond `x` or `y` are invalid, so they are not stored."""
    rows, cols = block.tile(8, 16)
    out[rows, cols] = maximum(x[rows, cols], y[rows, cols]) + 1.0


@kernel
def scores_max(a: Matrix[T.f32], b: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    """The largest score of each row of `a @ b`, for operands whose rows and columns run past their tensors."""
    rows, cols = block.tile(8, 16)
    acc = zeros(T.f32, 8, 16)
    for ks in pipelined(a.shape[1], 8):
        acc += a[rows, ks] @ b[ks, cols]
    out[rows, slice(0, 1)] = acc.max(axis=1, keepdims=True)


@kernel
def mean_row(t: Width, x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    """The mean of one row of `t.cols` lanes: the lanes past the edge of `x` are not counted."""
    rows, cols = block.tile(1, t.cols)
    out[rows, slice(0, 1)] = x[rows, cols].mean(axis=1, keepdims=True)


@kernel
def mean_past_edge(x: Matrix[T.f32], out: Matrix[T.f32]) -> None:
    """Every lane of this mean lies past the edge of `x`, so each row is NaN and that NaN is stored."""
    rows, cols = block.tile(4, 16)
    out[rows, slice(0, 1)] = x[rows, slice(16, 32)].mean(axis=1, keepdims=True)


@kernel
def mean_of_full_lanes(out: Matrix[T.f32]) -> None:
    """A mean of a tile that cannot be invalid, so the emulator and the C backend take the path with no valid mask."""
    rows, cols = block.tile(4, 16)
    out[rows, slice(0, 1)] = (zeros(T.f32, 4, 16) + 2.0).mean(axis=1, keepdims=True)
