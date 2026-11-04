"""Breakpoints an agent can use: a hook that writes the locals to a file and carries on

`pdb` waits for a person, and a breakpoint that waits is a hang for an agent running a command. Python lets the hook
behind `breakpoint()` be replaced with `PYTHONBREAKPOINT`, and `krn.compiler.debug.record` is a hook that writes the
file, the line, the function and the repr of every local of the calling frame to `breakpoint-N.json` in
`KRN_DEBUG_DUMP_DIR`, then returns. The kernel runs to the end and the agent reads the files; in the emulator the
locals are the thread's. `PYTHONBREAKPOINT=0` makes every `breakpoint()` do nothing, which a test run wants.

Level: 5
Topics: breakpoints, agents, PYTHONBREAKPOINT, recorded locals
Needs: 05_systems/02_breakpoints
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from krn import Items, T, Vector, configure, grid, host, kernel
from krn.compiler.api import CPUDevice
from krn.emulate import emulate

log = logging.getLogger("record")


@kernel
def clip(limit: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        value = x[i]
        if value > limit:
            breakpoint()
            value = limit
        y[i] = value


def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    x, y = device.empty(T.f32, 6), device.empty(T.f32, 6)
    x.array[:] = [0.5, 3.0, 1.0, 9.0, 2.0, 7.5]
    with tempfile.TemporaryDirectory(prefix="krn-record-") as folder:
        os.environ["PYTHONBREAKPOINT"] = "krn.compiler.debug.record"
        configure(**{"debug.dump_dir": Path(folder)})
        emulate(clip, Items(6), T.f32(2.5), x, y)
        for path in sorted(Path(folder).glob("breakpoint-*.json")):
            stop = json.loads(path.read_text())
            locals_ = stop["locals"]
            log.info("%s: %s line %d in %s  thread %s  i=%s value=%s", path.name, Path(stop["file"]).name, stop["line"],
                     stop["function"], locals_["_thread_id"], locals_["i"], locals_["value"])  # fmt: skip
    log.info("the kernel carried on: y = %s", y.array.tolist())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
