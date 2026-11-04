"""Helper functions in kernels: itertools generators that cost what the hand-written loop costs

A helper is a `@kernel` function that capture inlines where it is called, or a generator that capture expands inside
the `for` loop consuming it, so it allocates nothing. `krn.itertools` brings `pairwise`, `windowed`, `groupby` and
friends over a row `x[start:stop]`, as a CSR row is read, with sizes and steps fixed at capture. Below, one thread
counts the rising steps of a row, measures the runs of equal keys and averages every window of three.

Level: 2
Topics: helper functions, itertools, generators
Needs: 02_tiles_and_types/04_python_in_kernels
"""

import asyncio
import itertools
import logging

import numpy as np

from krn import Blocks, T, Vector, host, kernel
from krn.compiler.api import CPUDevice
from krn.itertools import groupby, pairwise, windowed

log = logging.getLogger("helpers")


@kernel
def count_rises(x: Vector[T.f32], out: Vector[T.u32]) -> None:
    """How many steps of the row go up, with `pairwise` yielding each element and its successor."""
    rises = T.u32(0)
    for a, b in pairwise(x, 0, x.shape[0]):
        if b > a:
            rises += 1
    out[0] = rises


@kernel
def run_lengths(keys: Vector[T.u32], out: Vector[T.u32]) -> None:
    """The lengths of the runs of equal adjacent keys: `groupby` yields (key, start, stop) and allocates nothing."""
    for run, (_, lo, hi) in enumerate(groupby(keys, 0, keys.shape[0])):
        out[run] = T.u32(hi) - T.u32(lo)


@kernel
def moving_average(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    """The mean of every full window of three elements."""
    for i, (a, b, c) in enumerate(windowed(x, 0, x.shape[0], 3)):
        out[i] = (a + b + c) / 3.0


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    one = Blocks(1, threads=1)  # a single thread walks the whole row, as one CSR row is walked

    row = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], np.float32)
    x, rises = device.empty(T.f32, len(row)), device.empty(T.u32, 1)
    x.array[:] = row
    count_rises(x, rises, launch=one)
    await device.synchronize()
    assert rises.array[0] == np.sum(np.diff(row) > 0)

    keys = [0, 0, 1, 1, 1, 3, 3, 4, 4, 4]
    runs = [len(list(group)) for _, group in itertools.groupby(keys)]
    k, lengths = device.empty(T.u32, len(keys)), device.empty(T.u32, len(runs))
    k.array[:] = keys
    run_lengths(k, lengths, launch=one)
    await device.synchronize()
    assert lengths.array.tolist() == runs

    averages = device.empty(T.f32, len(row) - 2)
    moving_average(x, averages, launch=one)
    await device.synchronize()
    np.testing.assert_allclose(averages.array, np.convolve(row, np.ones(3) / 3, "valid"), rtol=1e-6)
    log.info("rises %d, run lengths %s, moving average %s", rises.array[0], lengths.array.tolist(), averages.array)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
