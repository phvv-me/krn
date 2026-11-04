"""breakpoint() inside a kernel: stop, look at the locals of one thread, step through the kernel's own lines

Write `breakpoint()` where you want to look. In the emulator it is Python's own: `pdb` opens on the kernel as
written, with the real file and line and the locals of that thread (`_block_id`, `_thread_id`). On NVIDIA it is a
`brkpt` instruction for `cuda-gdb`, on AMD and CPUs a debug trap for `rocgdb` or `gdb`, and Metal has none, so use
the emulator or the Xcode shader debugger. This file stops twice: with a hook that records each stop, then in a
child process under `python -m pdb` with scripted commands whose transcript is printed. For a person: `python -m pdb
<this file>`, then `c`, `p i`, `ll`, `n`.

Level: 5
Topics: breakpoints, pdb, emulator
Needs: 05_systems/01_emulator_vs_backend
"""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from krn import Items, T, Vector, debug, grid, host, kernel
from krn.compiler.api import CPUDevice
from krn.emulate import emulate

log = logging.getLogger("breakpoint")
HERE = Path(__file__).resolve()


@kernel
def saxpy_with_breakpoint(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        if x[i] > 2.0:
            breakpoint()
        y[i] = a * x[i] + y[i]


def arrays() -> tuple[CPUDevice, np.ndarray, np.ndarray]:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    return device, np.array([1.0, 2.5, 0.5, 4.0], np.float32), np.ones(4, np.float32)


def stops_recorded_by_a_hook() -> None:
    device, x_values, y_values = arrays()
    x, y = device.empty(T.f32, 4), device.empty(T.f32, 4)
    x.array[:], y.array[:] = x_values, y_values

    def hook(*_args: str, **_kwargs: str) -> None:
        frame = sys._getframe(1)
        log.info("  stop at %s:%d thread %s: i=%s x[i]=%s", Path(frame.f_code.co_filename).name, frame.f_lineno,
                 frame.f_locals["_thread_id"][0], frame.f_locals["i"], x_values[frame.f_locals["i"]])  # fmt: skip

    with patch.object(sys, "breakpointhook", hook):
        emulate(saxpy_with_breakpoint, Items(4), T.f32(2.0), x, y)
    log.info("  y after the run: %s", y.array.tolist())


def scripted_pdb() -> None:
    line = next(n for n, text in enumerate(HERE.read_text().splitlines(), 1) if text.strip() == "breakpoint()")
    where = debug.Location(str(HERE), line)
    commands = ("p i", "p x[i]", "p _thread_id", "ll", "n", "p y[i]")
    argv = debug.Pdb(breakpoints=(where,), commands=commands).argv(HERE, "--child")
    env = os.environ | {"PYTHONBREAKPOINT": "0"}  # only the line breakpoint stops; the call in the source is ignored
    done = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120, check=False)
    log.info("  $ %s ...", shlex.join(["python", *argv[1:7]]))
    log.info("%s", done.stdout.rstrip())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--child" in sys.argv:
        device, x_values, y_values = arrays()
        x, y = device.empty(T.f32, 4), device.empty(T.f32, 4)
        x.array[:], y.array[:] = x_values, y_values
        emulate(saxpy_with_breakpoint, Items(4), T.f32(2.0), x, y)
    else:
        log.info("1. a hook that records every stop")
        stops_recorded_by_a_hook()
        log.info("2. python -m pdb with scripted commands, as an agent would drive it")
        scripted_pdb()
