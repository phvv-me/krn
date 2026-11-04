"""print, assert and out-of-bounds checks in a kernel

Python's own statements keep their meaning on the device, with the costs of a device. `print(...)` is a `printf`
from every thread that reaches it, `assert cond, "text"` prints `file:line: assertion failed` with the block and
thread and then traps (Metal has no trap: the message prints and the thread goes on), and `x[i]` outside `x` raises
`IndexError` in the emulator naming the tensor, the index and the shape at the kernel line, the check
`compute-sanitizer` makes. One kernel does all three: the emulator shows what a failure looks like from Python, and
the CUDA source shows the `printf` and `__trap` it becomes. Nothing is launched on a GPU.

Level: 3
Topics: print, assert, bounds checks, emulator
Needs: 03_memory_and_sync/02_block_reduction
"""

import logging
import traceback

from krn import Items, T, Vector, grid, host, kernel, targets
from krn.compiler.api import CPUDevice, capture, emit
from krn.emulate import emulate

log = logging.getLogger("checks")


@kernel
def reciprocal(x: Vector[T.f32], out: Vector[T.f32], shift: T.u64) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        assert x[i] != 0.0, "x must not be zero"
        print("thread", i, "reads", x[i])
        out[i] = 1.0 / x[i + shift]


def where(error: BaseException) -> str:
    """The kernel's own frame in the traceback: file, line and the statement."""
    frame = [f for f in traceback.extract_tb(error.__traceback__) if f.filename == __file__][-1]
    return f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}  {frame.line}"


def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    x, out = device.empty(T.f32, 3), device.empty(T.f32, 3)
    out.array[:] = 0

    x.array[:] = [2.0, 4.0, 8.0]
    log.info("1. a clean run prints from every thread:")
    emulate(reciprocal, Items(3), x, out, T.u64(0))
    log.info("   out = %s", out.array.tolist())

    x.array[:] = [2.0, 0.0, 8.0]
    log.info("2. a zero trips the assert:")
    try:
        emulate(reciprocal, Items(3), x, out, T.u64(0))
    except AssertionError as failure:
        log.info("   AssertionError: %s", failure)

    x.array[:] = [2.0, 4.0, 8.0]
    log.info("3. an index past the end is a bounds error at the line that made it:")
    try:
        emulate(reciprocal, Items(3), x, out, T.u64(1))
    except IndexError as failure:
        log.info("   IndexError: %s", failure)
        log.info("   raised at %s", where(failure))

    target = targets.Target.of("sm_90a")
    source = emit(capture(__file__, "reciprocal", target=target), target).source
    log.info("4. the same kernel for sm_90a:")
    for line in source.splitlines():
        if any(word in line for word in ("printf", "__trap", "__global__")):
            log.info("   %s", line.strip())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
