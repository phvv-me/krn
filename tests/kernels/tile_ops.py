"""Tile operations the attention kernels need: transposes, `where` and comparisons, positions, bare products and tensors
read with one scalar index and two slices. The tiles are 8 by 16 and the tensors may be smaller, so the lanes past their
edges are invalid."""

from krn import Const, Matrix, T, Tensor, arange, block, kernel, struct, where


@struct
class Shape:
    rows: Const[int] = 8
    cols: Const[int] = 16


@kernel
def transpose(t: Shape, x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    y[0 : t.cols, 0 : t.rows] = x[0 : t.rows, 0 : t.cols].transpose()


@kernel
def select(t: Shape, x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    tile = x[0 : t.rows, 0 : t.cols]
    y[0 : t.rows, 0 : t.cols] = where(tile > 0.0, tile, 0.5)


@kernel
def compare(
    t: Shape,
    x: Matrix[T.f32],
    less: Matrix[T.f32],
    less_equal: Matrix[T.f32],
    greater: Matrix[T.f32],
    greater_equal: Matrix[T.f32],
) -> None:
    tile = x[0 : t.rows, 0 : t.cols]
    less[0 : t.rows, 0 : t.cols] = (tile < 0.0).to(T.f32)
    less_equal[0 : t.rows, 0 : t.cols] = (tile <= 0.0).to(T.f32)
    greater[0 : t.rows, 0 : t.cols] = (tile > 0.0).to(T.f32)
    greater_equal[0 : t.rows, 0 : t.cols] = (tile >= 0.0).to(T.f32)


@kernel
def positions(t: Shape, out: Matrix[T.i32]) -> None:
    rows = arange(t.rows, 0, rank=2, start=3)
    cols = arange(t.cols, 1, rank=2)
    out[0 : t.rows, 0 : t.cols] = cols - rows


@kernel
def product(t: Shape, x: Matrix[T.f16], w: Matrix[T.f16], y: Matrix[T.f32]) -> None:
    y[0 : t.rows, 0 : t.rows] = x[0 : t.rows, 0 : t.cols] @ w[0 : t.rows, 0 : t.cols].transpose()


@kernel
def per_head(t: Shape, x: Tensor[T.f32, int, int, int], y: Tensor[T.f32, int, int, int]) -> None:
    head = block.index().y
    y[head, 0 : t.rows, 0 : t.cols] = x[head, 0 : t.rows, 0 : t.cols] * 2.0


@kernel
def causal(t: Shape, x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows = arange(t.rows, 0, rank=2)
    cols = arange(t.cols, 1, rank=2)
    y[0 : t.rows, 0 : t.cols] = where(cols <= rows, x[0 : t.rows, 0 : t.cols], -1.0e30)


@kernel
def scaled_by_extent(t: Shape, x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    tile = x[0 : t.rows, 0 : t.cols]
    y[0 : t.rows, 0 : t.cols] = tile * tile.shape[1]
