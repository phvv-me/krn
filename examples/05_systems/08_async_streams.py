"""Overlapping device work and Python work with async/await, for the free-threaded build of Python

A launch returns at once and the device is one more thing to await: `launcher.submit(...)` gives an `Event` and
`await event` suspends only the awaiting task, resolved by the driver's completion callback and never by polling.
The current stream is a `ContextVar`, so `async with device.stream():` gives every task its own in-order queue and
leaving the block awaits it. Compilation is Python plus a vendor compiler, and `asyncio.to_thread` runs it on a
worker thread (truly parallel on 3.14t, serial but correct otherwise). The program warms both kernels in parallel,
then runs batches as tasks, at most four in flight, with a timeout and failures collected as an exception group.

Level: 5
Topics: async launches, streams, TaskGroup, free-threaded Python
Needs: 05_systems/07_cpu_backend
"""

import asyncio
import logging
from collections.abc import Sequence
from itertools import batched

import numpy as np

from krn import Const, Matrix, T, Vector, block, host, kernel, pipelined, struct, zeros
from krn.compiler.api import CPUDevice, CPUTensor

log = logging.getLogger("pipeline")

type Batch = tuple[
    CPUTensor[T.f16, int, int],
    CPUTensor[T.f16, int, int],
    CPUTensor[T.f32, int, int],
    CPUTensor[T.f32, int],
    CPUTensor[T.f16, int, int],
]


@struct
class Tiling:
    bm: Const[int] = 64
    bn: Const[int] = 64
    bk: Const[int] = 32


@kernel
def gemm(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def bias_gelu(t: Tiling, h: Matrix[T.f32], bias: Vector[T.f32], y: Matrix[T.f16]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    v = h[rows, cols].add(bias[cols])
    y[rows, cols] = v.div(v.mul(-1.702).exp().add(1.0)).to(T.f16)


async def run_batch(
    a: CPUTensor[T.f16, int, int],
    b: CPUTensor[T.f16, int, int],
    h: CPUTensor[T.f32, int, int],
    bias: CPUTensor[T.f32, int],
    y: CPUTensor[T.f16, int, int],
) -> None:
    """One batch on its own stream. Both launches return at once; leaving the block awaits the stream."""
    t = Tiling()
    async with host.Device.current().stream():
        gemm[t](a, b, h)
        bias_gelu[t](h, bias, y)


async def serve(batches: Sequence[Batch]) -> None:
    a, b, h, bias, y = batches[0]
    t = Tiling()
    await asyncio.gather(  # bind compiles for these argument types, off the event loop thread
        asyncio.to_thread(gemm[t].bind, a, b, h),
        asyncio.to_thread(bias_gelu[t].bind, h, bias, y),
    )
    try:
        async with asyncio.timeout(30), asyncio.TaskGroup() as group:
            for chunk in batched(batches, 4, strict=False):  # at most four batches in flight: memory stays bounded
                async with asyncio.TaskGroup() as wave:
                    for batch in chunk:
                        wave.create_task(run_batch(*batch))
                group.create_task(host.Device.current().synchronize())
    except* (TimeoutError, MemoryError) as failures:
        log.error("pipeline failed: %s", failures.exceptions)
        raise


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    batches: list[Batch] = []
    for _ in range(6):
        a, b = device.empty(T.f16, 70, 96), device.empty(T.f16, 96, 80)
        h, y, bias = device.empty(T.f32, 70, 80), device.empty(T.f16, 70, 80), device.empty(T.f32, 80)
        a.array[:], b.array[:], bias.array[:] = (rng.standard_normal(x.array.shape) for x in (a, b, bias))
        batches.append((a, b, h, bias, y))
    await serve(batches)
    for a, b, _, bias, y in batches:
        v = a.array.astype(np.float32) @ b.array.astype(np.float32) + bias.array
        np.testing.assert_allclose(y.array.astype(np.float32), v / (1 + np.exp(-1.702 * v)), rtol=3e-2, atol=3e-2)
    log.info("%d batches served on their own streams, all agree with numpy", len(batches))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
