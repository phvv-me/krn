"""Warp collectives: a butterfly sum, an inclusive scan, a ballot and a vote

A warp is the group of lanes that run in lockstep (32 on NVIDIA and Apple, 64 on AMD CDNA) and `warp.size()` gives
the width at capture, so the loops are written once. `warp.shfl_xor(x, m)` reads the value of lane `^ m` (a
butterfly), `warp.shfl_up(x, d)` the value of lane `- d` (a scan), `warp.ballot(p)` collects a bit per lane, and
`warp.all(p)` and `warp.any(p)` answer for the whole warp. A C loop over the threads cannot run a collective, so the
C backend refuses these kernels at capture and says why; the emulator runs each lane as a coroutine that switches at
the collective.

Level: 3
Topics: warps, shuffles, ballot, votes, emulator
Needs: 03_memory_and_sync/02_block_reduction
"""

import logging

import numpy as np

from krn import Items, T, Vector, grid, host, kernel, static_assert, warp
from krn.compiler.api import CPUDevice, capture
from krn.emulate import emulate
from krn.errors import CaptureError

log = logging.getLogger("warp")


@kernel
def butterfly_sum(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    static_assert(warp.size() == 32, "this butterfly is written for 32 lanes")
    i = grid.thread_rank()
    total = x[i]
    for step in range(5):
        total = total + warp.shfl_xor(total, 1 << step)
    if warp.lane() == 0:
        out[i // 32] = total


@kernel
def inclusive_scan(x: Vector[T.i32], out: Vector[T.i32]) -> None:
    static_assert(warp.size() == 32, "this scan is written for 32 lanes")
    i = grid.thread_rank()
    value = x[i]
    for step in range(5):
        other = warp.shfl_up(value, 1 << step)
        if warp.lane() >= 1 << step:
            value = value + other
    out[i] = value


@kernel
def votes(x: Vector[T.i32], ballots: Vector[T.u64], unanimous: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    positive = x[i] > 0
    mask = warp.ballot(positive)
    everyone = warp.all(positive)
    if warp.lane() == 0:
        ballots[i // 32] = mask
        unanimous[i // 32] = T.i32(1) if everyone else T.i32(0)


def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(3)

    x = device.empty(T.f32, 256)
    x.array[:] = rng.standard_normal(256)
    sums = device.empty(T.f32, 8)
    emulate(butterfly_sum, Items(256), x, sums)
    log.info(
        "butterfly sum equals numpy per warp: %s",
        bool(np.allclose(sums.array, x.array.reshape(8, 32).sum(1), atol=1e-5)),
    )

    counts = device.empty(T.i32, 256)
    counts.array[:] = rng.integers(0, 5, 256)
    scanned = device.empty(T.i32, 256)
    emulate(inclusive_scan, Items(256), counts, scanned)
    expected = np.cumsum(counts.array.reshape(8, 32), axis=1).reshape(-1)
    log.info("inclusive scan equals cumsum per warp: %s", bool((scanned.array == expected).all()))

    signs = device.empty(T.i32, 256)
    signs.array[:] = np.where(np.arange(256) % 4 == 0, 3, -1)
    signs.array[32:64] = 7
    ballots, unanimous = device.empty(T.u64, 8), device.empty(T.i32, 8)
    emulate(votes, Items(256), signs, ballots, unanimous)
    log.info("ballots %s, unanimous %s", [hex(int(b)) for b in ballots.array], unanimous.array.tolist())

    try:
        capture(__file__, "butterfly_sum", target=device.target)
    except CaptureError as refusal:
        log.info("the C backend says:\n%s", refusal)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
