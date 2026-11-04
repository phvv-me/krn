"""Atomic updates: a histogram, a running maximum, a wrapping ticket counter and a lock

`atomic.add(tensor, index, value)` updates one element as a single indivisible step and returns the value it had.
The same eleven names (`add sub and_ or_ xor exch inc dec max min cas`) mean the same on every vendor, and four
patterns need them: a histogram (many threads add 1 to a bin), a running maximum of floats (a compare-and-swap loop
on NVIDIA), a ticket counter (`atomic.inc` wraps at a limit) and a lock (`atomic.cas` returns 0 for exactly one
thread). The first half runs on the CPU backend; the second prints `atomic.plan` for every target: native, a CAS
loop, or refused with the reason (Apple has no 64-bit atomics).

Level: 3
Topics: atomics, CPU backend, atomic plan
Needs: 01_first_kernels/01_saxpy
"""

import asyncio
import logging

import numpy as np

from krn import Items, T, Vector, grid, host, kernel, targets
from krn.compiler.api import CPUDevice
from krn.frontend.atomic import AtomicOp, Strategy, atomic, plan

log = logging.getLogger("atomics")


@kernel
def histogram(values: Vector[T.i32], bins: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    if i < values.shape[0]:
        atomic.add(bins, values[i], 1)


@kernel
def running_max(x: Vector[T.f32], best: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        atomic.max(best, 0, x[i])


@kernel
def tickets(limit: T.u32, counter: Vector[T.u32], mine: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < mine.shape[0]:
        mine[i] = atomic.inc(counter, 0, limit)


@kernel
def take_lock(lock: Vector[T.i32], winners: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    if i < winners.shape[0]:
        old = atomic.cas(lock, 0, 0, 1)
        atomic.add(winners, 0, 1 if old == 0 else 0)


async def run_here() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)

    values = device.empty(T.i32, 1000)
    values.array[:] = rng.integers(0, 16, 1000)
    bins = device.empty(T.i32, 16)
    bins.array[:] = 0
    histogram(values, bins, launch=Items(1000))
    await device.synchronize()
    log.info("histogram equals bincount: %s", bool((bins.array == np.bincount(values.array, minlength=16)).all()))

    x = device.empty(T.f32, 500)
    x.array[:] = rng.standard_normal(500)
    best = device.empty(T.f32, 1)
    best.array[:] = -np.inf
    running_max(x, best, launch=Items(500))
    await device.synchronize()
    log.info("running max equals max: %s", bool(best.array[0] == x.array.max()))

    counter, mine = device.empty(T.u32, 1), device.empty(T.u32, 10)
    counter.array[:] = 0
    tickets(T.u32(3), counter, mine, launch=Items(10))
    await device.synchronize()
    log.info("tickets with limit 3: %s (counter ends at %d)", mine.array.tolist(), counter.array[0])

    lock, winners = device.empty(T.i32, 1), device.empty(T.i32, 1)
    lock.array[:] = winners.array[:] = 0
    take_lock(lock, winners, launch=Items(64))
    await device.synchronize()
    log.info("64 threads tried the lock, %d won it", winners.array[0])


def show_plans() -> None:
    chosen = [targets.Target.of(arch) for arch in ("sm_90a", "gfx942", "apple_m5", "avx512")]
    cases = [(op, dtype) for op, dtype in [(AtomicOp.ADD, "i32"), (AtomicOp.ADD, "f32"), (AtomicOp.ADD, "i64"),
                                           (AtomicOp.MAX, "f32"), (AtomicOp.MAX, "i32"), (AtomicOp.INC, "u32"),
                                           (AtomicOp.INC, "u64"), (AtomicOp.CAS, "u64")]]  # fmt: skip
    log.info("%-12s %s", "operation", "  ".join(f"{t.arch:>10}" for t in chosen))
    for op, dtype in cases:
        cells = []
        for target in chosen:
            match plan(op, dtype, target).strategy:
                case Strategy.NATIVE:
                    cells.append("native")
                case Strategy.CAS_LOOP:
                    cells.append("cas loop")
                case Strategy.REFUSED:
                    cells.append("refused")
        log.info("%-12s %s", f"{op}.{dtype}", "  ".join(f"{c:>10}" for c in cells))
    log.info("why: %s", plan(AtomicOp.ADD, "i64", targets.Target.of("apple_m5")).reason)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run_here())
    show_plans()
