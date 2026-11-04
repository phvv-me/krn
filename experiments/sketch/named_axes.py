"""Named axes as types: a kernel signature declares M, N and K, and the checker verifies the contraction by name.

The sizes stay dynamic. What is checked is that `a`'s second axis is the same axis as `b`'s first, so swapping `Seq` and
`Hidden` (both 4096 at runtime) is a static error, and no one writes a size or a transpose to find out.
"""

from krn import Const, Matrix, T, Tile, block, kernel, pipelined, struct, zeros


class Tokens: ...


class Hidden: ...


class Vocab: ...


class Seq: ...


@struct
class Tiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32


@kernel
def logits[M, N, K](h: Matrix[T.f16, M, K], w: Matrix[T.f16, K, N], out: Matrix[T.f32, M, N], t: Tiling) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc: Tile[T.f32, M, N] = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(h.shape[1], t.bk, stages=3):
        acc += h[rows, ks] @ w[ks, cols]
    out[rows, cols] = acc


def run(
    h: Matrix[T.f16, Tokens, Hidden],
    w: Matrix[T.f16, Hidden, Vocab],
    out: Matrix[T.f32, Tokens, Vocab],
) -> None:
    t = Tiling()
    logits(h, w, out, t)
