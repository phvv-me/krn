"""Torch-style tile methods: chain them, and update in place with a trailing underscore

Every elementwise method has two forms: `x.exp()` returns a new tile and `x.exp_()` updates the variable `x` and
returns it, so the in-place form is for a named tile and never for a temporary. In a kernel the tiles are registers
and the chain fuses into the surrounding code. The methods come from one table: unary (`exp`, `rsqrt`, `silu`,
`gelu` ...), binary (`add`, `mul`, `maximum` ...), `clip`, reductions (`sum`, `mean`, `max`, `logsumexp`, `norm`)
with `keepdims`, and `softmax`. Three kernels show the style: a softmax, an RMS norm and a gated activation.

Level: 2
Topics: tile methods, reductions, in-place updates
Needs: 01_first_kernels/02_gelu
"""

import asyncio
import logging

import numpy as np

from krn import Const, Matrix, T, Vector, block, host, kernel, struct
from krn.compiler.api import CPUDevice

log = logging.getLogger("tile_methods")


@struct
class RowTiling:
    rows: Const[int] = 4
    cols: Const[int] = 4096


@kernel
def softmax(t: RowTiling, x: Matrix[T.f16], y: Matrix[T.f16]) -> None:
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    y[rows, cols] = x[rows, cols].to(T.f32).softmax().to(T.f16)


@kernel
def rms_norm(t: RowTiling, x: Matrix[T.f16], weight: Vector[T.f16], y: Matrix[T.f16], eps: T.f32) -> None:
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    v = x[rows, cols].to(T.f32)
    scale = v.mul(v).mean(keepdims=True).add(eps).rsqrt()  # 1 / sqrt(mean(x^2) + eps), one fused chain
    y[rows, cols] = v.mul(scale).mul(weight[cols].to(T.f32)).to(T.f16)


@kernel
def swish_gate(t: RowTiling, gate: Matrix[T.f32], up: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    """silu(gate) * up, clipped: `v` is a variable, so the in-place forms update its register."""
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    v = gate[rows, cols].silu()
    v.mul_(up[rows, cols])
    y[rows, cols] = v.clip_(-65504.0, 65504.0)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    shape, tiling = (8, 1024), RowTiling(rows=4, cols=1024)
    x, weight, y = device.empty(T.f16, *shape), device.empty(T.f16, shape[1]), device.empty(T.f16, *shape)
    x.array[:], weight.array[:] = rng.standard_normal(shape), rng.uniform(0.5, 1.5, shape[1])
    v = x.array.astype(np.float32)

    softmax[tiling](x, y)
    await device.synchronize()
    e = np.exp(v - v.max(axis=1, keepdims=True))
    np.testing.assert_allclose(y.array.astype(np.float32), e / e.sum(axis=1, keepdims=True), atol=1e-3)

    rms_norm[tiling](x, weight, y, T.f32(1e-6))
    await device.synchronize()
    want = v / np.sqrt((v * v).mean(axis=1, keepdims=True) + 1e-6) * weight.array.astype(np.float32)
    np.testing.assert_allclose(y.array.astype(np.float32), want, atol=5e-3)

    gate, up, out = (device.empty(T.f32, *shape) for _ in range(3))
    gate.array[:], up.array[:] = rng.standard_normal(shape), rng.standard_normal(shape)
    swish_gate[tiling](gate, up, out)
    await device.synchronize()
    np.testing.assert_allclose(out.array, gate.array / (1 + np.exp(-gate.array)) * up.array, rtol=1e-4, atol=1e-5)
    log.info("softmax, rms_norm and swish_gate agree with numpy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
