"""The emulator against the backend: a differential test in ten lines

The emulator and the backend share the front end (capture and the typed IR) and nothing after it, so a disagreement
is a bug in one of the two lowerings or a float rounding difference, never in the program. The pair is a test oracle
for each other: when a kernel is wrong on a GPU, run it here first. If both agree the logic is fine and the problem
is in the vendor path (a missing barrier, a race, a layout); if the emulator is wrong too, put a `breakpoint()` in
the kernel. A SIMT kernel and a tile kernel run both ways, the backend inferring geometry and the emulator taking it
explicitly. The emulator is far slower, so it is for small inputs.

Level: 5
Topics: emulator, differential testing, launch inference
Needs: 03_memory_and_sync/02_block_reduction
"""

import asyncio
import logging
import time

import numpy as np

from krn import Items, Matrix, T, Tiles, Vector, block, host, items, kernel
from krn.compiler.api import CPUDevice
from krn.emulate import emulate

log = logging.getLogger("differential")


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        y[i] = a * x[i] + y[i]


@kernel
def gelu_bias(x: Matrix[T.f32], bias: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    y[rows, cols] = (x[rows, cols] + bias[rows, cols]).gelu()


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(1)

    x, y_emulated, y_backend = (device.empty(T.f32, 1000) for _ in range(3))
    x.array[:] = rng.standard_normal(1000)
    y_emulated.array[:] = y_backend.array[:] = rng.standard_normal(1000)
    started = time.perf_counter()
    emulate(saxpy, Items(1000), T.f32(2.0), x, y_emulated)
    middle = time.perf_counter()
    saxpy(T.f32(2.0), x, y_backend)
    await device.synchronize()
    ended = time.perf_counter()
    np.testing.assert_allclose(y_emulated.array, y_backend.array, rtol=1e-6)
    log.info(
        "saxpy: emulator and backend agree (emulator %.0f ms, backend %.2f ms)",
        1e3 * (middle - started),
        1e3 * (ended - middle),
    )

    shape = (20, 40)
    a, b = device.empty(T.f32, *shape), device.empty(T.f32, *shape)
    out_emulated, out_backend = device.empty(T.f32, *shape), device.empty(T.f32, *shape)
    a.array[:], b.array[:] = rng.standard_normal(shape), rng.standard_normal(shape)
    started = time.perf_counter()
    emulate(gelu_bias, Tiles(shape, (8, 16)), a, b, out_emulated)
    middle = time.perf_counter()
    gelu_bias(a, b, out_backend)
    await device.synchronize()
    ended = time.perf_counter()
    np.testing.assert_allclose(out_emulated.array, out_backend.array, rtol=1e-5, atol=1e-6)
    log.info(
        "gelu + bias: emulator and backend agree (emulator %.0f ms, backend %.2f ms)",
        1e3 * (middle - started),
        1e3 * (ended - middle),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
