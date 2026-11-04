"""The two halves of a blocked LU in one language: a SIMT pivot search and a block-scaled tile update.

Over 95 percent of the flops are the update (tile style), but the pivot search and the panel are warp code, and both
must live in one program so the panel can quantize its output without a round trip (notes/lowprec-algorithms.md).
"""

from krn import Matrix, QMatrix, T, Vector, block, kernel, pipelined, warp, zeros
from krn.data import formats as formats

from .gemm import GemmTiling


@kernel
def pivot_search(a: Matrix[T.f32], col: T.u32, pivots: Vector[T.i32]) -> None:
    """One warp scans the column below the diagonal: largest magnitude, lowest row on ties."""
    best = T.f32(0.0)
    best_row = T.u32(0)
    for r in range(col + warp.lane(), a.shape[0], warp.size()):
        v = abs(a[r, col])
        if v > best:
            best, best_row = v, T.u32(r)
    best, best_row = warp.argmax(best, best_row)
    if warp.lane() == 0:
        pivots[col] = T.i32(best_row)


@kernel
def trailing_update[F: formats.BlockFormat](
    l21: QMatrix[F], u12: QMatrix[F], a22: Matrix[T.f32], t: GemmTiling
) -> None:
    """A22 -= L21 @ U12 with block-scaled operands and float32 accumulators."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(l21.shape[1], t.bk, stages=t.stages):
        acc += l21[rows, ks] @ u12[ks, cols]
    a22[rows, cols] -= acc
