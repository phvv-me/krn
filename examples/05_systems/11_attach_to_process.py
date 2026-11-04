"""Attach to a process that is already running, ask it what it is doing, and dump its stacks when it is stuck

Python 3.14 can attach to a live interpreter without a debugger: `python -m pdb -p PID` for a person,
`sys.remote_exec` for a program. `debug.attach(pid)` injects a script that calls `debug.snapshot()` in the target
and returns a typed `Snapshot`: the interpreter and build kind, the `KRN_*` variables, what every watched object
reports (a CPU stream: its queue length, whether a worker drains it, the next task, any sticky error) and the stack
of every thread. The OS must let you trace the process, the interpreter versions and build kind must match, and the
target must reach a safe point; a process stuck in a driver call raises `TimeoutError`, and `debug.install()` there
arms `kill -USR1 PID` to dump all stacks. This file serves launches from a child on a CPU stream and attaches to it.

Level: 5
Topics: attach, snapshot, stuck processes, remote debugging
Needs: 05_systems/07_cpu_backend
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from krn import Items, T, Vector, debug, grid, host, kernel
from krn.compiler.api import CPUDevice

log = logging.getLogger("attach")


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


async def serve() -> None:
    """The child: launch work on the default stream forever, the way an engine would."""
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    x, y = device.empty(T.f32, 1 << 16), device.empty(T.f32, 1 << 16)
    x.array[:], y.array[:] = 1.0, 0.0
    debug.install()
    saxpy(T.f32(0.5), x, y, launch=Items(1 << 16))
    await device.synchronize()
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    while True:
        saxpy(T.f32(0.5), x, y, launch=Items(1 << 16))
        await device.synchronize()
        await asyncio.sleep(0.02)


def main() -> None:
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}
    command = [sys.executable, str(Path(__file__)), "--child"]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        started = time.perf_counter()
        found = debug.attach(child.pid, timeout=30)
        log.info("attached to pid %d in %.0f ms", found.pid, 1e3 * (time.perf_counter() - started))
        log.info(
            "python %s, free-threaded %s, argv %s",
            found.python,
            found.free_threaded,
            [Path(a).name for a in found.argv],
        )
        for obj in found.objects:
            log.info("watched: %s", obj)
        log.info("%d threads; the main thread is at:", len(found.threads))
        main_stack = next(iter(found.threads.values()))
        log.info("  %s", main_stack[-1].strip().replace("\n", " | "))
        child.send_signal(signal.SIGUSR1)  # faulthandler writes every stack to stderr, without the interpreter
        time.sleep(0.3)
    finally:
        child.kill()
        _, stderr = child.communicate()
    dump = stderr.splitlines()
    log.info("after SIGUSR1 the child's stderr has %d lines; the first two:", len(dump))
    for line in dump[:2]:
        log.info("  %s", line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--child" in sys.argv:
        asyncio.run(serve())
    else:
        main()
