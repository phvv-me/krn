"""A host device whose launches run on a GPU: how an example written against `host.Device.current()` runs on Metal or
on a card.

The tensors stay numpy arrays, which is what the examples read and assert on. Each launch captures its kernel for the
GPU's target, uploads the arguments, runs it and reads them back in place, as `devices.CUDARun` and
`MetalRun.run` do. `python -c "import mirror; mirror.main()" example.py` runs one example that way, with
`KRN_MIRROR` naming the GPU (`metal` or `cuda`) and `KRN_TARGET` its architecture, and prints the launches it made.
An example that compiles its own programs with `krn.compiler.api.build` and calls
`program(*arrays, launch=Items(n))` gets the same.
"""

import os
import runpy
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from importlib import import_module
from typing import cast
from unittest.mock import patch

import numpy as np

import krn.compiler.api
from krn import Blocks, Tiles
from krn.compiler.emit import Unit
from krn.compiler.heuristics import Decisions
from krn.compiler.ir import Kernel
from krn.compiler.space import geometry
from krn.facts.targets import CPU, Target
from krn.frontend.geometry import Geometry
from krn.runtime import api as runtime
from krn.runtime.cpu import CPUProgram, Value
from krn.runtime.host import Backend
from krn.runtime.runtime import CPUDevice
from tests.helpers.devices import CUDARun, MetalRun

TILE_THREADS = 128  # threads that share a tile when the launch of a tile kernel does not say, as the CUDA backend uses
VENDOR = {"metal": "apple", "cuda": "nvidia"}

type Runner = Callable[[Kernel], CUDARun | MetalRun]


def writable(value: Value) -> Value:
    """The runners write every array back, so an input a program only reads (`np.frombuffer` makes those) is copied."""
    if isinstance(value, np.ndarray):
        return value if value.flags.writeable else value.copy()
    if isinstance(value, Mapping):
        return {name: writable(part) for name, part in value.items()}
    return value


class Mirrored:
    """A kernel compiled for the GPU, in the shape of a `CPUProgram`: `program(*arrays, launch=launch)` runs it."""

    def __init__(self, device: MirrorDevice, kernel: Kernel) -> None:
        self.kernel, self.device, self.runner = kernel, device, device.make(kernel)
        self.unit = Unit("", kernel.name)  # the GPU program holds the source; the CPU runtime only reads the field

    def __call__(self, *arrays: Value, launch: Geometry | None = None) -> None:
        self.run(launch if launch is not None else geometry(self.kernel, arrays), arrays)

    def bind(self, launch: Geometry) -> Callable[..., None]:
        return lambda *arrays: self.run(launch, arrays)

    def run(self, launch: Geometry, values: Sequence[Value]) -> None:
        """A GPU runs a tile kernel with a block of threads that share the tile, where the CPU runs it with one."""
        blocks = launch.blocks if isinstance(launch, Tiles) else launch
        if self.kernel.tiled and isinstance(blocks, Blocks) and blocks.threads in (None, 1):
            launch = replace(blocks, threads=TILE_THREADS)
        with self.device.busy:
            self.runner.run(launch, [writable(value) for value in values])
            self.device.launches += 1

    def marshal(self, values: Sequence[Value]) -> tuple[list[Value], list]:
        """The values themselves: the GPU takes the arrays, not their addresses."""
        return list(values), []

    def function(self, *arguments: Value) -> None:
        """What `Call.run` invokes: the values, then the grid and threads the CPU launch resolved to. A program is
        shared by every launch of its kernel and signature, so the geometry comes from the call, not the program."""
        match arguments:
            case (*values, int() as x, int() as y, int() as z, int() as threads):
                self.run(Blocks(x, y, z, threads), values)
            case _:
                raise TypeError(f"expected the values and four integers of a geometry, got {arguments!r}")


class MirrorDevice(CPUDevice):
    """The host CPU's streams, events, graphs and numpy tensors, with every program run on a GPU instead."""

    def __init__(self, target: Target, make: Runner) -> None:
        super().__init__(CPU.of("avx512"))  # the pool and the streams are the host's; the programs below target the GPU
        self.target, self.make, self.launches, self.busy = target, make, 0, threading.Lock()

    def program(self, ir: Kernel, decisions: Decisions) -> CPUProgram:
        # the runtime reads a program's `kernel`, `unit`, `marshal` and `function`, which `Mirrored` has
        return cast(CPUProgram, Mirrored(self, ir))


def metal_runner(target: Target) -> Runner:
    """Programs on this Mac's GPU. PyObjC exists on macOS only, so it is imported where the Mac is chosen."""
    metal = import_module("krn.runtime.metal")

    gpu = metal.MetalDevice(target)
    return lambda kernel: MetalRun(gpu, kernel)


def cuda_runner(target: Target) -> Runner:
    """Programs on the first card. The backend needs cuda-python, so it is imported where a card is chosen."""
    backend = import_module("krn.runtime.cuda_backend").BACKEND

    card = backend.device(target)
    return lambda kernel: CUDARun(card, kernel)


class MirrorBackend(Backend):
    """The vendor of the GPU `KRN_MIRROR` names, found by the runtime through `runtime.MODULES` like any other."""

    vendor = "mirror"

    def device(self, target: Target) -> MirrorDevice:
        gpu = os.environ["KRN_MIRROR"]
        return MirrorDevice(target, metal_runner(target) if gpu == "metal" else cuda_runner(target))


BACKEND = MirrorBackend()


def main() -> None:
    """Run the example named in `sys.argv[1]` as a script on the mirrored GPU, then say how many launches it made."""
    runtime.MODULES[VENDOR[os.environ["KRN_MIRROR"]]] = __name__
    device = runtime.default_device()
    assert isinstance(device, MirrorDevice)
    try:
        with patch.object(krn.compiler.api, "build", lambda kernel, _target: Mirrored(device, kernel)):
            runpy.run_path(sys.argv[1], run_name="__main__")
    finally:
        sys.stdout.write(f"mirror: {device.launches} launches on {device.target.arch}\n")
