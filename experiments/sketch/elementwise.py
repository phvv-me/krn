"""One operation in both styles. TileLang's official add is 18 non-blank lines and its terse form 14."""

from krn import Const, Items, Matrix, T, Vector, block, grid, kernel, struct


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    """SIMT style: one thread per element, the way the body reads."""
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@struct
class Tiling:
    rows: Const[int] = 128
    cols: Const[int] = 128


@kernel
def add(a: Matrix[T.f32], b: Matrix[T.f32], c: Matrix[T.f32], tiling: Tiling) -> None:
    """Tile style: a block owns a tile, and slices say which. Staging, layout and bounds are the compiler's problem."""
    rows, cols = block.tile(tiling.rows, tiling.cols)
    c[rows, cols] = a[rows, cols] + b[rows, cols]


def run(x: Vector[T.f32], y: Vector[T.f32], a: Matrix[T.f32], b: Matrix[T.f32], c: Matrix[T.f32]) -> None:
    saxpy(T.f32(2.0), x, y, launch=Items(x.shape[0]))
    tiling = Tiling()
    add(a, b, c, tiling)
