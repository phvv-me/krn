"""GEMM. TileLang's official auto-pipelined fp16 GEMM is 17 non-blank lines: 7 of ABI prologue, 3 allocations, 3 copies.

The quantized GEMM is the same body. Its operands are QMatrix[F] and the instruction comes from the lowering registry
keyed by (format, accumulator, target).
"""

from krn import AUTO, Auto, Const, Matrix, QMatrix, T, WarpSpecialized, block, kernel, pipelined, struct, zeros
from krn.data import formats as formats


@struct
class GemmTiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32
    stages: Const[int | Auto] = AUTO


@kernel
def gemm(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32], t: GemmTiling) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=t.stages):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def gemm_warp_specialized(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32], t: GemmTiling) -> None:
    """The same body asking for producer and consumer warps: a request the target must be able to run (TMA)."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, schedule=WarpSpecialized()):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def qgemm[F: formats.BlockFormat](a: QMatrix[F], b: QMatrix[F], c: Matrix[T.f32], t: GemmTiling) -> None:
    """The same body. Operands are block-scaled, and the instruction comes from the lowering registry."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=t.stages):
        acc = a[rows, ks].mma(b[ks, cols], acc, promote_every=128)
    c[rows, cols] = acc


def run(a: QMatrix[formats.NVFP4], b: QMatrix[formats.NVFP4], c: Matrix[T.f32]) -> None:
    qgemm(a, b, c, GemmTiling())
