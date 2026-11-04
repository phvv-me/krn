"""Kernels in the namespaced syntax: block.tile, pipelined slices, AUTO, templates, selectors and capture-time facts."""

from krn import (
    AUTO,
    Auto,
    Const,
    Feature,
    Matrix,
    T,
    Vector,
    WarpSpecialized,
    block,
    grid,
    kernel,
    pipelined,
    struct,
    target,
    targets,
    zeros,
)


@struct
class RowTiling:
    rows: Const[int] = 4
    cols: Const[int] = 16


@kernel
def doubled_rows(x: Matrix[T.f32], y: Matrix[T.f32], t: RowTiling) -> None:
    """One size is the tile along x, so a 1-D grid walks the rows."""
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    y[rows, cols] = x[rows, cols] * 2.0


@kernel
def corner_tile(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    """Two sizes are the rows along y and the columns along x."""
    rows, cols = block.tile(4, 8)
    y[rows, cols] = x[rows, cols] + 1.0


@struct
class Tiling:
    bm: Const[int] = 16
    bn: Const[int] = 16
    bk: Const[int] = 8
    stages: Const[int | Auto] = AUTO


@kernel
def gemm(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32], t: Tiling) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=t.stages):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def gemm_two_stages(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    acc = zeros(T.f32, 8, 8)
    for ks in pipelined(a.shape[1], 4, stages=2):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def gemm_warp_specialized(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    acc = zeros(T.f32, 8, 8)
    for ks in pipelined(a.shape[1], 4, schedule=WarpSpecialized()):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def gemm_template(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """The tiling is the configuration in the brackets: gemm_template[Tiling()](a, b, c)."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=t.stages):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def gemm_generic[F: T.FloatingPoint](t: Tiling, a: Matrix[F], b: Matrix[F], c: Matrix[T.f32]) -> None:
    """The operand type is a type parameter bound by the dtype of the tensors; the accumulator follows it."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.acc(a.dtype), t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc.to(c.dtype)


@struct
class Precision:
    dtype: Const[T.FloatingPoint] = T.fp.min


@kernel
def round_trip(p: Precision, x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    """Round to the narrowest float of the target, and back to float32: the configuration holds the selector."""
    rows, cols = block.tile(8, 8)
    y[rows, cols] = x[rows, cols].to(p.dtype).to(T.f32)


@kernel
def selected(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    low = x[rows, cols].to(T.fp.min)
    high = zeros(T.fp.max, 8, 8) + x[rows, cols].to(T.fp.max)
    y[rows, cols] = low.to(T.acc(T.fp.min)) + high.to(T.f32)


@kernel
def accumulators(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    counts = zeros(T.acc(T.u8), 8, 8)
    doubles = zeros(T.acc(T.f64), 8, 8)
    halves = zeros(T.acc(T.f16), 8, 8)
    y[rows, cols] = x[rows, cols] + counts.to(T.f32) + doubles.to(T.f32) + halves


@kernel
def eight_bits(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    ranged = x[rows, cols].to(T.fp.emax(8))
    precise = x[rows, cols].to(T.fp.mmax(8))
    y[rows, cols] = ranged.to(T.f32) + precise.to(T.f32)


@struct
class Flags:
    negate: Const[bool] = False


@kernel
def flagged(x: Vector[T.f32], y: Vector[T.f32], flags: Flags) -> None:
    """A `Const[bool]` folds the branch: the untaken side is never captured, so it may name anything."""
    i = grid.thread_rank()
    if i < x.shape[0]:
        if flags.negate:
            y[i] = -x[i]
        else:
            y[i] = x[i]


@kernel
def type_aliases(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    """A name may stand for a type and for AUTO, like for any other compile-time value."""
    rows, cols = block.tile(8, 8)
    wide = T.acc(T.fp.min)
    unrolled = AUTO
    acc = zeros(wide, 8, 8)
    for ks in pipelined(x.shape[1], 8, stages=unrolled):
        acc += x[rows, ks].to(T.f16).to(wide)
    y[rows, cols] = acc


@kernel
def capability(out: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < out.shape[0]:
        out[i] = T.u32(0)
        if target.has(Feature.TENSOR_CORE):
            out[i] += 1
        if isinstance(target, targets.CPU):
            out[i] += 10
        if target.has(targets.Feature.FLOAT64):
            out[i] += 100
