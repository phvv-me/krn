"""Row softmax. TileLang's official online softmax is 28 non-blank lines, three of them arithmetic."""

from krn import Const, Matrix, T, block, kernel, struct


@struct
class RowTiling:
    rows: Const[int] = 4
    cols: Const[int] = 4096


@kernel
def softmax(x: Matrix[T.f16], y: Matrix[T.f16], t: RowTiling) -> None:
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    y[rows, cols] = x[rows, cols].to(T.f32).softmax(axis=1).to(T.f16)


@kernel
def softmax_steps(x: Matrix[T.f16], y: Matrix[T.f16], t: RowTiling) -> None:
    """The same function with its arithmetic spelled out: subtract the row maximum, exponentiate, normalize."""
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    v = x[rows, cols].to(T.f32)
    e = (v - v.max(axis=1, keepdims=True)).exp()
    y[rows, cols] = (e / e.sum(axis=1, keepdims=True)).to(T.f16)
