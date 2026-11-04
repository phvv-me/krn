"""Nineteen mistakes the checkers should reject, with what ty 0.0.95 and pyrefly 1.4.0 did (Python 3.15.2).

Both reject every one except E6 and E14, which only ty catches. E16 to E19 are the mistakes of the namespaced syntax.
"""

from typing import Literal

from krn import Blocks, Items, Matrix, QMatrix, T, Tile, Vector, block, grid, kernel, pipelined
from krn.data import formats as formats

from .elementwise import saxpy
from .gemm import GemmTiling, qgemm
from .named_axes import Hidden, Seq, Tiling, Tokens, Vocab, logits

type D128 = Literal[128]
type D256 = Literal[256]
type D64 = Literal[64]
type D32 = Literal[32]


@kernel
def bad_kernel(x: Vector[T.f32], h: Vector[T.f16], b: Vector[T.u8], n: T.u32) -> None:
    i = grid.thread_rank()
    x[i] = h[i]  # E1 f16 stored into an f32 tensor: ty and pyrefly reject
    _j = n + b[i]  # E2 u32 + u8: both reject
    x[i] = x[i] + n  # E3 f32 + u32: both reject
    if i < n:  # E4 u64 < u32: both reject
        x[i] = T.f32(0.0)


def launches(
    x: Vector[T.f32], y: Vector[T.f32], a: QMatrix[formats.NVFP4], b: QMatrix[formats.MXFP8], c: Matrix[T.f32]
) -> None:
    saxpy(2.0, x, y, launch=Items(8))  # E5 a float where an f32 is declared: both reject
    qgemm(a, b, c, GemmTiling(), launch=Items(8))  # E6 NVFP4 times MXFP8 in one product: ty rejects, pyrefly misses it


def tiles(
    p: Tile[T.f16, D128, D64],
    q: Tile[T.f16, D32, D256],
    r: Tile[T.f32, D128, D64],
    acc: Tile[T.f32, D128, D256],
    out: Matrix[T.f32],
    half: Matrix[T.f16],
) -> None:
    p.mma(q, acc)  # E7 inner dimensions 64 and 32: both reject
    p + r  # E8 f16 tile plus f32 tile: both reject
    p @ q  # E9 inner dimensions 64 and 32 through the operator
    out[0:128, 0:64] = p  # E10 f16 tile stored into an f32 tensor
    out[0:128, 0:64] = out[0, 0]  # E11 a scalar where a tile is expected
    half[0, 0] = out[0:128, 0:64]  # E12 a tile where a scalar is expected


# E6 is the only miss: pyrefly 1.4.0 drops the shared type variable F when a generic function goes through the
# ParamSpec of `kernel`. ty rejects it (as an invariance mismatch against the union NVFP4 | MXFP8), and the JIT
# rejects it in any case.
# E7 and E9 need classic invariant TypeVar and TypeVarTuple on Tile. With PEP 695 parameters the inferred variance is
# covariant and ty widens K to a union (64 | 32), so it accepts the mismatch. See the module docstring of memory.py.


@kernel
def bad_axes[M, N, K](h: Matrix[T.f16, M, K], w: Matrix[T.f16, N, K]) -> None:
    h[0:128, 0:32] @ w[
        0:128, 0:32
    ]  # E13 contracts M-by-K with N-by-K: the second axes match but the first of w is N, not K


def axes(h: Matrix[T.f16, Tokens, Hidden], wrong: Matrix[T.f16, Seq, Vocab], out: Matrix[T.f32, Tokens, Vocab]) -> None:
    logits(
        h, wrong, out, Tiling(), launch=Blocks(1)
    )  # E14 a Seq axis where Hidden is expected: ty rejects, pyrefly misses


def widths(a: T.u3, b: T.u4) -> None:
    a + b  # E15 a 3-bit integer plus a 4-bit integer


@kernel
def scaled(t: GemmTiling, c: Matrix[T.f32]) -> None:
    rows, cols, planes = block.tile(8, 8, 8)  # E16 three tile sizes: block.tile takes one or two
    for ks in pipelined(range(0, c.shape[1], t.bk), t.bk):  # E17 the extent is an int, not a range
        c[rows, ks] = c[planes, cols]


def configuration(c: Matrix[T.f32]) -> None:
    scaled[GemmTiling()](c)
    scaled(c)  # E18 a template launch without its configuration in the brackets
    scaled[GemmTiling()](c)  # E19 the tile of Tiles is a tuple, one size per axis
