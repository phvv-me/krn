"""A block-wide sum through shared memory and barriers, run thread by thread in the emulator

Threads of a block share an array (`shared_array`), and `block.sync()` makes every thread wait until all have
arrived. The reduction halves the active threads each round with a barrier between rounds, so a thread never reads a
slot its neighbor has not written. `block.count`, `block.all` and `block.any` are barriers that also return a vote
over the block. The C backend runs the threads one after the other and refuses a barrier at capture, because it
would not wait and the sum would silently be wrong; the emulator runs every thread as a coroutine that meets at each
barrier.

Level: 3
Topics: shared memory, barriers, block votes, emulator
Needs: 01_first_kernels/01_saxpy
"""

import logging

import numpy as np

from krn import Items, T, Vector, block, grid, host, kernel, shared_array, thread
from krn.compiler.api import CPUDevice, capture
from krn.emulate import emulate
from krn.errors import CaptureError

log = logging.getLogger("block")


@kernel
def block_sum(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    scratch = shared_array(T.f32, 256)
    t = thread.index().x
    scratch[t] = x[grid.thread_rank()]
    block.sync()
    stride = 128
    while stride > 0:
        if t < stride:
            scratch[t] = scratch[t] + scratch[t + stride]
        block.sync()
        stride = stride // 2
    if t == 0:
        out[block.index().x] = scratch[0]


@kernel
def block_votes(x: Vector[T.i32], counts: Vector[T.u32], anyone: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    positive = x[i] > 0
    n = block.count(positive)
    some = block.any(positive)
    if thread.index().x == 0:
        counts[block.index().x] = n
        anyone[block.index().x] = T.i32(1) if some else T.i32(0)


def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    x = device.empty(T.f32, 512)
    x.array[:] = np.random.default_rng(5).standard_normal(512)
    sums = device.empty(T.f32, 2)
    emulate(block_sum, Items(512), x, sums)
    log.info("block sums equal numpy: %s", bool(np.allclose(sums.array, x.array.reshape(2, 256).sum(1), atol=1e-4)))

    marks = device.empty(T.i32, 512)
    marks.array[:] = np.where(np.arange(512) % 5 == 0, 1, -1)
    marks.array[256:] = -1
    counts, anyone = device.empty(T.u32, 2), device.empty(T.i32, 2)
    emulate(block_votes, Items(512), marks, counts, anyone)
    log.info("positive per block %s, any positive %s", counts.array.tolist(), anyone.array.tolist())

    try:
        capture(__file__, "block_sum", target=device.target)
    except CaptureError as refusal:
        log.info("the C backend says:\n%s", refusal)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
