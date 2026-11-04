"""Run for real: eight batches of GEMM then SAXPY on the CPU backend, eager, as async tasks and as graph replays

The kernels are captured from this very file, compiled with clang and run on a worker pool, and every result is
compared with numpy. The host code is what the GPU backends run: `host.Device.current()`, `async with
device.stream()`, `device.graph(...)`. Three schedules of the same work: `eager` (one stream, one launch after
another), `tasks` (one asyncio task per batch, each with its own stream through the `ContextVar` of the current
stream) and `graph` (the two launches of `step` recorded once and replayed per batch). On the free-threaded build
(`python3.14t`) the workers, the event loop and the Python between launches run in parallel and
`sys._is_gil_enabled()` is printed; read the speedups, not the absolute times.

Level: 5
Topics: CPU backend, streams, tasks, graphs, free-threaded Python
Needs: 04_libraries/01_gemm_tiles
"""

import asyncio
import logging
import sys
import time
from typing import NamedTuple

import numpy as np

from krn import AUTO, Auto, Const, Items, Matrix, T, Vector, block, grid, host, kernel, pipelined, struct, zeros
from krn.compiler.api import CPUDevice, CPUTensor

log = logging.getLogger("demo")
SIZE, BATCHES = 320, 8


@struct
class Tiling:
    bm: Const[int] = 32
    bn: Const[int] = 32
    bk: Const[int] = 16
    stages: Const[int | Auto] = AUTO


@kernel
def gemm(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=t.stages):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


class Batch(NamedTuple):
    a: CPUTensor[T.f16, int, int]
    b: CPUTensor[T.f16, int, int]
    c: CPUTensor[T.f32, int, int]
    c_flat: CPUTensor[T.f32, int]
    out_flat: CPUTensor[T.f32, int]


def make_batch(device: CPUDevice, seed: int) -> Batch:
    rng = np.random.default_rng(seed)
    a, b = device.empty(T.f16, SIZE, SIZE), device.empty(T.f16, SIZE, SIZE)
    a.array[:], b.array[:] = rng.standard_normal((SIZE, SIZE)), rng.standard_normal((SIZE, SIZE))
    c, out = device.empty(T.f32, SIZE, SIZE), device.empty(T.f32, SIZE * SIZE)
    c.array[:], out.array[:] = 0.0, 1.0
    return Batch(a, b, c, CPUTensor(c.array.reshape(-1)), out)


def step(
    a: CPUTensor[T.f16, int, int],
    b: CPUTensor[T.f16, int, int],
    c: CPUTensor[T.f32, int, int],
    c_flat: CPUTensor[T.f32, int],
    out_flat: CPUTensor[T.f32, int],
) -> None:
    """The unit of work: c = a @ b, then out += 0.5 * c. Two launches, ordered by their tensors."""
    t = Tiling()
    gemm[t](a, b, c)
    saxpy(T.f32(0.5), c_flat, out_flat, launch=Items(SIZE * SIZE))


def expected(batch: Batch) -> np.ndarray:
    return 1.0 + 0.5 * (batch.a.array.astype(np.float32) @ batch.b.array.astype(np.float32)).reshape(-1)


def reset(batches: list[Batch]) -> None:
    for batch in batches:
        batch.out_flat.array[:] = 1.0


def check(batches: list[Batch], label: str) -> float:
    error = max(float(np.abs(batch.out_flat.array - expected(batch)).max()) for batch in batches)
    log.info("  %-34s max |error| against numpy %.2e", label, error)
    return error


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    batches = [make_batch(device, seed) for seed in range(BATCHES)]
    log.info(
        "python %s, GIL enabled: %s, target %s, %d workers",
        sys.version.split()[0],
        sys._is_gil_enabled(),
        device.target.arch,
        device.settings.run.workers,
    )

    step(*batches[0])  # compile both kernels once, so the timings below measure running, not clang
    await device.synchronize()
    reset(batches)

    start = time.perf_counter()
    for batch in batches:
        step(*batch)
    await device.synchronize()
    eager = time.perf_counter() - start
    check(batches, "eager, one stream")

    async def run_batch(batch: Batch) -> None:
        async with device.stream():  # leaving the block awaits this stream
            step(*batch)

    reset(batches)
    start = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for batch in batches:
            group.create_task(run_batch(batch))
    tasks = time.perf_counter() - start
    check(batches, "tasks, one stream each")

    graph = device.graph(step, *batches[0])

    async def replay_batch(batch: Batch) -> None:
        async with device.stream():
            graph(*batch)  # new tensors, the recorded launches and their order

    reset(batches)
    start = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for batch in batches:
            group.create_task(replay_batch(batch))
    replay = time.perf_counter() - start
    check(batches, "graph replay, one stream each")

    log.info(
        "  eager %.0f ms, tasks %.0f ms (%.1fx), graph replay %.0f ms (%.1fx)",
        eager * 1e3,
        tasks * 1e3,
        eager / tasks,
        replay * 1e3,
        eager / replay,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
