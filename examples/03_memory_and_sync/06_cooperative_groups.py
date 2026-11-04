"""Cooperative groups: tile sums and a block scan with the collectives of a group

`cg.this_thread_block()` is the group of the block and `cg.tiled_partition(group, 8)` splits it into tiles of eight
lanes. A group has the collectives of CUDA's cooperative groups, such as `reduce(value, cg.plus)` and
`inclusive_scan(value, cg.plus)`, so the kernel never spells a shuffle or a barrier. The collectives run lane by
lane in the emulator and are checked against numpy; the C backend refuses them at capture with the reason.

Level: 3
Topics: cooperative groups, reductions, scans, emulator
Needs: 03_memory_and_sync/05_warp_shuffles
"""

import logging

import numpy as np

from krn import Blocks, Vector, cg, grid, host, i32, kernel
from krn.compiler.api import CPUDevice, capture
from krn.emulate import emulate
from krn.errors import CaptureError

log = logging.getLogger("cg")


@kernel
def tile_sums(x: Vector[i32], out: Vector[i32]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), 8)
    i = grid.thread_rank()
    out[i] = tile.reduce(x[i], cg.plus)


@kernel
def block_scan(x: Vector[i32], out: Vector[i32]) -> None:
    group = cg.this_thread_block()
    i = grid.thread_rank()
    out[i] = group.inclusive_scan(x[i], cg.plus)


def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    x = device.empty(i32, 128)
    out = device.empty(i32, 128)
    x.array[:] = np.arange(128) % 7
    launch = Blocks(2, threads=64)
    emulate(tile_sums, launch, x, out)
    expected = np.repeat(x.array.reshape(-1, 8).sum(1), 8)
    np.testing.assert_array_equal(out.array, expected)
    log.info("tile sums over eight lanes match NumPy")
    emulate(block_scan, launch, x, out)
    expected = np.cumsum(x.array.reshape(2, 64), axis=1).reshape(-1)
    np.testing.assert_array_equal(out.array, expected)
    log.info("block scans over 64 threads match NumPy")
    try:
        capture(__file__, "tile_sums", target=device.target)
    except CaptureError as error:
        log.info("C backend: %s", error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
