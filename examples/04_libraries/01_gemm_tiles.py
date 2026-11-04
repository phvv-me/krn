"""GEMM from the algorithm alone: Tilus's minimal Blackwell matmul, written once for every target

Tilus v0 is a 60-line script for sm_100a: it stages tiles with `cp.async`, multiplies with `tcgen05.mma` into tensor
memory and tracks completion with an `mbarrier`. Take the instructions away and the algorithm is left: split C into
tiles, one block each, run the K loop adding `a_tile @ b_tile` into a float32 accumulator, round to float16 and
store. That is `block.tile`, `pipelined` and `@` here, and what `@` becomes is the target's job: `mma.sync`,
`wgmma`, `tcgen05`, MFMA, `simdgroup_matrix`, or FMA loops on a CPU. `experiments/gemm_gap.py` measures it against
cuBLASLt.

Level: 4
Topics: GEMM, tiles, pipelined loops, tensor cores
Needs: 01_first_kernels/02_gelu
"""

import asyncio
import logging

import numpy as np

from krn import Const, Matrix, block, f16, f32, host, kernel, pipelined, struct, zeros
from krn.compiler.api import CPUDevice

log = logging.getLogger("matmul")
CONFIGS = [(bm, bn, bk) for bm, bn in ((128, 64), (128, 128), (128, 256)) for bk in (16, 32, 64)]  # Tilus's space


@struct
class Tiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32


@kernel
def matmul(t: Tiling, a: Matrix[f16], b: Matrix[f16], c: Matrix[f16]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc.to(f16)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    m, n, k = 50, 38, 70  # none a multiple of a tile: the ragged edge is part of the product
    a = device.empty(f16, m, k)
    b = device.empty(f16, k, n)
    c = device.empty(f16, m, n)
    np.asarray(a)[:], np.asarray(b)[:] = rng.standard_normal((m, k)), rng.standard_normal((k, n))
    tiling = Tiling(bm=16, bn=16, bk=8)
    matmul[tiling](a, b, c)
    await device.synchronize()
    want = np.asarray(a).astype(np.float32) @ np.asarray(b).astype(np.float32)
    np.testing.assert_allclose(np.asarray(c).astype(np.float32), want, rtol=2e-2, atol=2e-2)
    log.info("matmul %dx%dx%d agrees with numpy", m, n, k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
