"""Two paths of a block at two different barriers: the emulator reports the deadlock

`block.sync()` waits for every thread of the block. If the even threads reach one barrier and the odd threads
another (`block.sync()` against `block.count(...)`), each group waits for the other forever, and on a GPU the kernel
hangs or goes on with garbage, the bug `compute-sanitizer --tool synccheck` exists for. The emulator runs the
threads as coroutines that meet at barriers, so it knows when none can make progress and raises `Deadlock`, naming
the barriers and the waiting threads. The first kernel has the bug; the second is the usual fix: every thread
reaches the same barrier and only the work is conditional.

Level: 3
Topics: barriers, deadlock, emulator
Needs: 03_memory_and_sync/02_block_reduction
"""

import logging

from krn import Items, T, Vector, block, host, kernel, thread
from krn.compiler.api import CPUDevice
from krn.emulate import Deadlock, emulate

log = logging.getLogger("deadlock")


@kernel
def broken(out: Vector[T.u32]) -> None:
    if thread.index().x % 2 == 0:
        block.sync()
    else:
        out[thread.index().x] = block.count(True)


@kernel
def fixed(out: Vector[T.u32]) -> None:
    odd = block.count(thread.index().x % 2 == 1)
    if thread.index().x % 2 == 0:
        out[thread.index().x] = odd


def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    out = device.empty(T.u32, 256)
    out.array[:] = 0
    try:
        emulate(broken, Items(256), out)
    except Deadlock as stuck:
        log.info("broken: %s", str(stuck).replace("; ", "\n        "))
    emulate(fixed, Items(256), out)
    log.info("fixed: every even thread read the count %d", int(out.array[0]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
