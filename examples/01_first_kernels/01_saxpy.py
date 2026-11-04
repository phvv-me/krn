"""SAXPY in the SIMT style, and what strict arithmetic asks of the code you write

One thread handles one element, the way CUDA, HIP and Metal programmers think, without pointer arithmetic, a launch
macro or a hand-written grid: `saxpy(a, x, y, launch=Items(n))` gives it one thread per item, and the same source
runs on NVIDIA, AMD, Apple and CPUs. Arithmetic is strict. The same type stays the same type (`u8 + u8` wraps as
`u8`), a Python literal adopts the other operand's type and must fit it, mixing two types is an error that names
both, widening is spelled `T.u32(byte)`, and integers and floats never mix. `ty` and `pyrefly` report these mistakes
while you type, for example `out[i] = data[i]` storing a `u8` into a `u32` tensor.

Level: 1
Topics: SIMT, strict arithmetic, launch geometry
Needs: none
"""

import asyncio
import logging

import numpy as np

from krn import Items, T, Vector, grid, host, kernel
from krn.compiler.api import CPUDevice

log = logging.getLogger("saxpy")


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    """y = a * x + y, rounded twice: strict arithmetic never fuses on its own.

    The compiler hints that `math.fma(a, x, y)` rounds once and is one instruction.
    """
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@kernel
def byte_checksum(data: Vector[T.u8], out: Vector[T.u32]) -> None:
    """A position-weighted checksum. `T.u32(...)` is the explicit widening that strict arithmetic asks for.

    The literal 31 adopts u32 and the product wraps modulo 2**32, which is exactly what a hash wants.
    """
    i = grid.thread_rank()
    if i < data.shape[0]:
        out[i] = T.u32(data[i]) * 31 + T.u32(i)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    n = 1000
    x, y = device.empty(T.f32, n), device.empty(T.f32, n)
    x.array[:], y.array[:] = rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)
    want = np.float32(2.0) * x.array + y.array
    saxpy(T.f32(2.0), x, y, launch=Items(n))  # one thread per item: manual thread indexing needs explicit geometry
    await device.synchronize()
    np.testing.assert_allclose(y.array, want, rtol=1e-6)

    data, out = device.empty(T.u8, n), device.empty(T.u32, n)
    data.array[:] = rng.integers(0, 256, n, dtype=np.uint8)
    byte_checksum(data, out, launch=Items(n))
    await device.synchronize()
    np.testing.assert_array_equal(out.array, data.array.astype(np.uint32) * 31 + np.arange(n, dtype=np.uint32))
    log.info("saxpy and byte_checksum agree with numpy on %d items", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
