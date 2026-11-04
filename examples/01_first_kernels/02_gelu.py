"""GELU with a bias in the tile style: a slice read is a load, a slice write a store

There is no `load`, `store`, mask or pointer. `x[rows, cols]` reads a tile, methods on tiles are elementwise and
chain as they do in PyTorch, and `y[rows, cols] = tile` writes it. `block.tile(rows, cols)` names the tile a block
owns, a `@struct` of `Const` fields is the configuration in the brackets (`gelu_bias[Tiling()](x, bias, y)`), and
the launch geometry follows from the output and the tile. The compiler owns staging, and the lanes beyond the edge
of the tensor are invalid: they read as nothing, take part in nothing and are not stored, so there is no remainder
code.

Level: 1
Topics: tiles, tile methods, configuration, invalid lanes
Needs: 01_first_kernels/01_saxpy
"""

import asyncio
import logging

import numpy as np

from krn import Const, Matrix, T, Vector, block, host, kernel, struct
from krn.compiler.api import CPUDevice

log = logging.getLogger("gelu")


@struct
class Tiling:
    rows: Const[int] = 32
    cols: Const[int] = 256


@kernel
def gelu_bias(t: Tiling, x: Matrix[T.f16], bias: Vector[T.f16], y: Matrix[T.f16]) -> None:
    """y = gelu(x + bias) with the sigmoid approximation x * sigmoid(1.702 x), computed in float32."""
    rows, cols = block.tile(t.rows, t.cols)
    v = x[rows, cols].to(T.f32).add(bias[cols].to(T.f32))  # the 1-D bias tile broadcasts along rows
    y[rows, cols] = v.div(v.mul(-1.702).exp().add(1.0)).to(T.f16)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    x, bias, y = device.empty(T.f16, 100, 300), device.empty(T.f16, 300), device.empty(T.f16, 100, 300)
    x.array[:], bias.array[:] = rng.standard_normal((100, 300)), rng.standard_normal(300)
    gelu_bias[Tiling()](x, bias, y)  # the geometry is inferred from y and the tile: 4 x 2 blocks, the edges ragged
    await device.synchronize()
    v = x.array.astype(np.float32) + bias.array.astype(np.float32)
    np.testing.assert_allclose(y.array.astype(np.float32), v / (1 + np.exp(-1.702 * v)), atol=4e-3)
    log.info("gelu_bias on %d x %d agrees with numpy", *y.array.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
