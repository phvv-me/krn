"""Softmax for rows longer than a tile, with no padding value and no mask

Two passes over the row in column chunks: the first keeps a running maximum `m` and running sum `s` (the online form
of FlashAttention), the second writes the normalized values. `pipelined(extent, tile)` yields the chunks as slices
and the last one is ragged. An out-of-bounds lane is invalid and therefore neutral in `max`, `maximum` and `sum`: it
can never be the maximum and adds nothing, where padding with zeros would silently corrupt `max`. A kernel cannot
spell `-inf` yet, so the running maximum starts at a large finite value.

Level: 2
Topics: tiles, reductions, invalid lanes, chunked loops
Needs: 01_first_kernels/02_gelu
"""

import asyncio
import logging

import numpy as np

from krn import Const, Matrix, T, block, host, kernel, maximum, pipelined, struct, zeros
from krn.compiler.api import CPUDevice

log = logging.getLogger("softmax")


@struct
class RowTiling:
    rows: Const[int] = 4
    cols: Const[int] = 1024


@kernel
def softmax_long_rows(t: RowTiling, x: Matrix[T.f16], y: Matrix[T.f16]) -> None:
    rows = block.tile(t.rows)
    m = zeros(T.f32, t.rows, 1).sub(3.0e38)  # a finite stand-in for -inf, which a kernel cannot spell yet
    s = zeros(T.f32, t.rows, 1)
    for cs in pipelined(x.shape[1], t.cols):  # column chunks, the last one ragged
        v = x[rows, cs].to(T.f32)
        m_new = maximum(m, v.max(keepdims=True))
        s = s * (m - m_new).exp() + (v - m_new).exp().sum(keepdims=True)
        m = m_new
    for cs in pipelined(x.shape[1], t.cols):
        y[rows, cs] = (x[rows, cs].to(T.f32) - m).exp().div(s).to(T.f16)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    x, y = device.empty(T.f16, 6, 2500), device.empty(T.f16, 6, 2500)  # 2500 columns: two full chunks and a ragged one
    x.array[:] = rng.standard_normal((6, 2500)) * 4
    softmax_long_rows[RowTiling()](x, y)
    await device.synchronize()
    v = x.array.astype(np.float32)
    e = np.exp(v - v.max(axis=1, keepdims=True))
    np.testing.assert_allclose(y.array.astype(np.float32), e / e.sum(axis=1, keepdims=True), atol=1e-3)
    log.info("softmax of %d rows of %d columns agrees with numpy", *y.array.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
