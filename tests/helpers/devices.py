"""What a test of a device shares with the others: the launch of a cooperative block and the comparison of a device
run with the emulator, the oracle for the IR. Integers must agree exactly and floats to 1e-5, since a device's libm
rounds differently."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from krn import Blocks, Tiles
from krn.compiler.api import capture
from krn.compiler.dtypes import Dtype
from krn.compiler.ir import Kernel
from krn.compiler.source import Module
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.frontend.geometry import Geometry
from krn.runtime.cpu import Value
from krn.runtime.runtime import DTYPES

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDATensor
    from krn.runtime.cuda_backend import Value as DeviceValue
    from krn.runtime.metal import MetalDevice, MetalTensor
    from krn.runtime.metal import Value as MetalValue

FILE = Path(__file__).resolve().parents[1] / "kernels" / "intrinsics.py"


class CUDARun:
    """A captured kernel run on a CUDA device through the backend, with the `run` of the other programs: the arrays
    go to the card, the kernel launches on the default stream, and the arrays are read back in place."""

    def __init__(self, gpu: CUDADevice, kernel: Kernel) -> None:
        self.gpu, self.program = gpu, gpu.program(kernel)

    def run(self, launch: Geometry, values: Sequence[Value]) -> None:
        uploaded: list[tuple[np.ndarray, CUDATensor]] = []
        arguments = [self.upload(value, uploaded) for value in values]
        stream = self.gpu.default_stream
        self.program.call(self.program.geometry(launch), arguments).enqueue(stream)
        stream.sync()
        for array, tensor in uploaded:
            array[...] = tensor.to_host(stream)

    def upload(self, value: Value, uploaded: list[tuple[np.ndarray, CUDATensor]]) -> DeviceValue:
        match value:
            case np.ndarray():
                uploaded.append((value, tensor := self.gpu.to_device(value)))
                return tensor
            case Mapping():
                return {name: self.upload(part, uploaded) for name, part in value.items()}
            case _:
                return value


class MetalRun:
    """Adapt numpy-based compiler conformance fixtures to the shared-buffer device."""

    def __init__(self, gpu: MetalDevice, kernel: Kernel) -> None:
        self.gpu, self.program = gpu, gpu.program(kernel)

    def run(self, launch: Geometry, values: Sequence[Value]) -> None:
        uploaded: list[tuple[np.ndarray, MetalTensor]] = []
        arguments = [self.upload(value, uploaded) for value in values]
        self.program.call(self.program.geometry(launch), arguments).enqueue(self.gpu.default_stream)
        self.gpu.default_stream.sync()
        for array, tensor in uploaded:
            array[...] = tensor.array

    def upload(self, value: Value, uploaded: list[tuple[np.ndarray, MetalTensor]]) -> MetalValue:
        match value:
            case np.ndarray():
                uploaded.append((value, tensor := self.gpu.to_device(value)))
                return tensor
            case Mapping():
                return {name: self.upload(part, uploaded) for name, part in value.items()}
            case _:
                return value


def cooperative(tiles: Tiles, threads: int = 128) -> Blocks:
    """One block per tile with `threads` threads sharing it, the way a card runs a tile kernel."""
    return replace(tiles.blocks, threads=threads)


def fresh(value: Value) -> Value:
    """A copy of an argument that a run may write into: its arrays, its structs' arrays, its numbers as they are."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {name: fresh(part) for name, part in value.items()}
    return value


def arrays(values: Sequence[Value]) -> list[np.ndarray]:
    """Every array among `values`, in order, those inside structs too."""
    found: list[np.ndarray] = []
    for value in values:
        if isinstance(value, np.ndarray):
            found.append(value)
        elif isinstance(value, Mapping):
            found += arrays(list(value.values()))
    return found


def dtypes(arguments: Mapping[str, Value]) -> dict[str, Dtype]:
    """The element type of each array argument by parameter path (`x`, `a.values`), the way `capture` binds them."""
    found: dict[str, Dtype] = {}
    for name, value in arguments.items():
        if isinstance(value, np.ndarray):
            found[name] = DTYPES[value.dtype]
        elif isinstance(value, Mapping):
            found.update(dtypes({f"{name}.{field}": part for field, part in value.items()}))
    return found


def bindings(file: Path, name: str, arguments: Sequence[Value]) -> dict[str, Dtype]:
    """`dtypes` of positional `arguments`, named by the parameters of the kernel `name` in `file`."""
    names = [parameter.arg for parameter in Module.read(file).kernel(name).args.args]
    return dtypes(dict(zip(names, arguments, strict=True)))


def compare(
    target: Target,
    launcher: Callable[[Kernel], CUDARun | MetalRun],
    name: str,
    launch: Geometry,
    arguments: list[Value],
    file: Path = FILE,
    structs: Mapping[str, str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The array arguments after `name` ran in the emulator and after `launcher` ran it on a device, from one start.

    structs: the class of each struct argument by parameter name, for a kernel written against a base class.
    """
    kernel = capture(file, name, structs=structs, dtypes=bindings(file, name, arguments), target=target)
    oracle, actual = ([fresh(a) for a in arguments] for _ in range(2))
    emulate(kernel, target).run(launch, oracle)
    launcher(kernel).run(launch, actual)
    return arrays(oracle), arrays(actual)


def agree(oracle: list[np.ndarray], actual: list[np.ndarray], outputs: tuple[int, ...] | None = None) -> None:
    for index in range(len(oracle)) if outputs is None else outputs:
        if oracle[index].dtype.kind == "f":
            assert np.allclose(oracle[index], actual[index], rtol=1e-5, atol=1e-6, equal_nan=True), index
        else:
            assert np.array_equal(oracle[index], actual[index]), index
