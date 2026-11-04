"""Flash attention 1, forward: softmax(Q K^T / sqrt(d)) V in one pass over the keys and values

Each block owns one query tile of one head and walks the key and value tiles once, keeping the running row maximum
`m`, the running row sum `l` and the output `acc`. When a new key tile raises the maximum, the old sums are rescaled by
`exp2(m - m_new)`, so the final `acc / l` is the exact softmax and the score matrix never leaves the block. The scores
carry `log2(e) / sqrt(d)`, which lets every exponential be `exp2`. A key tile that runs past the sequence is invalid
lanes, so it takes no part in the maximum, the sum or the product with V, and a causal mask is one `where` on the
positions of the two tiles.

Level: 4
Topics: attention, online softmax, batched heads, causal masks, ragged tiles
Needs: 04_libraries/01_gemm_tiles
"""

import asyncio
import logging
import math

import numpy as np

from krn import Const, T, Tensor, Tiles, arange, block, host, kernel, struct, where, zeros
from krn.compiler.api import CPUDevice

log = logging.getLogger("flash_attention")


@struct
class Tiling:
    bq: Const[int] = 16
    bk: Const[int] = 16
    d: Const[int] = 64
    causal: Const[bool] = False


@kernel
def flash_attention(
    t: Tiling,
    q: Tensor[T.f16, int, int, int],
    k: Tensor[T.f16, int, int, int],
    v: Tensor[T.f16, int, int, int],
    o: Tensor[T.f16, int, int, int],
    scale: T.f32,
) -> None:
    rows, head = block.tile(t.bq), block.index().y
    q_pos = arange(t.bq, 0, rank=2, start=block.index().x * t.bq)
    m, total, acc = zeros(T.f32, t.bq, 1).sub(1.0e30), zeros(T.f32, t.bq, 1), zeros(T.f32, t.bq, t.d)
    qt = q[head, rows, slice(0, t.d)]
    for c in range(0, k.shape[1], t.bk):
        s = (qt @ k[head, c : c + t.bk, slice(0, t.d)].transpose()).mul(scale)
        if t.causal:
            s = where(arange(t.bk, 1, rank=2, start=c) <= q_pos, s, -1.0e30)
        m_new = m.maximum(s.max(keepdims=True))
        p, corr = (s - m_new).exp2(), (m - m_new).exp2()
        total *= corr
        total += p.sum(keepdims=True)
        acc *= corr
        acc += p.to(T.f16) @ v[head, c : c + t.bk, slice(0, t.d)]
        m = m_new
    o[head, rows, slice(0, t.d)] = (acc / total).to(T.f16)


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    """softmax(Q K^T / sqrt(d)) V the plain way, in float64: the scores, the softmax, then the product."""
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    s = q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape[-2:], bool)), s, -np.inf)
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)) @ v


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    heads, seq = 2, 200  # 200 is not a multiple of the 16-row tile: the last key and query tiles are ragged
    for d, causal in ((64, False), (128, True)):
        q, k, v, o = (device.empty(T.f16, heads, seq, d) for _ in range(4))
        for x in (q, k, v):
            np.asarray(x)[:] = rng.standard_normal((heads, seq, d))
        scale = math.log2(math.e) / math.sqrt(d)
        flash_attention(Tiling(d=d, causal=causal), q, k, v, o, scale, launch=Tiles((heads, seq), (1, 16)))
        await device.synchronize()
        want = reference(np.asarray(q), np.asarray(k), np.asarray(v), causal)
        np.testing.assert_allclose(np.asarray(o).astype(np.float64), want, atol=2e-3)
        log.info("attention of %d heads over %d tokens, d=%d, causal=%s, agrees with numpy", heads, seq, d, causal)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
