"""The CUDA vendor of the runtime: a device, streams, events and tensors over `cuda.core`, launches through the driver.

`cuda.core` owns what it has an object model for: the primary context, streams, events and stream-ordered memory. It
cannot pass a struct by value, so a launch is `cuda.bindings.driver.cuLaunchKernel` with the parameters packed here, one
C-aligned buffer per launch (a tensor is its pointer and one 64-bit size per axis, a struct is laid out as C lays it
out). NVRTC comes from `cuda.bindings.nvrtc`, so the backend needs the cp314 wheels `cuda-core` and `cuda-bindings` and
nothing else. Programs are cached by the hash of the generated source and the compiler flags, in memory and, with
`KRN_CACHE_DIR`, as cubins on disk.

Memory is stream ordered: a tensor is valid on the stream that allocated it and on any stream that waited for an event
recorded after the allocation, and it returns to the pool on its allocation stream when the last reference goes.
"""

import asyncio
import ctypes
import hashlib
import json
import logging
import os
import struct
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import is_dataclass
from functools import cache, partial
from typing import cast, overload

import numpy as np
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
from cuda.core import Buffer, EventOptions, ObjectCode
from cuda.core import Device as CoreDevice
from cuda.core import Event as CoreEvent
from cuda.core import Kernel as CoreKernel
from cuda.core import Stream as CoreStream

from krn import Kernel, Tensor, debug
from krn.compiler import space
from krn.compiler.cuda import CUDARenderer
from krn.compiler.dtypes import (
    BOOL,
    F16,
    F32,
    F64,
    I8X4,
    I16X2,
    PY_FLOAT,
    PY_INT,
    U8,
    U8X4,
    U16X2,
    U32,
    U64,
    Dtype,
    from_sketch,
    sint,
    to_sketch,
    uint,
)
from krn.compiler.emit import Emitter, Unit, dimensions
from krn.compiler.errors import TargetError
from krn.compiler.heuristics import Decisions, Extent, decide
from krn.compiler.ir import Kernel as IrKernel
from krn.compiler.ir import ParamType, StructType, TensorType
from krn.compiler.performance import Compilation, Facts, compilation
from krn.compiler.performance import check as check_performance
from krn.facts.targets import Nvidia, Target
from krn.frontend.geometry import Blocks, Geometry, Items
from krn.frontend.numerics import FloatingPoint, Scalar, Selector

from ..profile import launches as profile_launches
from .api import Launch
from .compile_options import CompileOptions, numerical_options
from .completion import Completion, Keeper, Waiter
from .cuda_context import ThreadContext
from .host import Backend, Bound, Device, Event, Graph, Settings, Stream, current_stream
from .runtime import DTYPES, Argument, SourceLauncher, Variant, _layout, numpy_dtype
from .settings import current_settings

log = logging.getLogger(__name__)

SHARED_LIMIT = 48 * 1024  # dynamic shared memory a launch may reserve without asking the driver for more
FORMATS = {
    BOOL: "?", U8: "B", uint(16): "H", U32: "I", U64: "Q", sint(8): "b", sint(16): "h", sint(32): "i", sint(64): "q",
    F16: "e", F32: "f", F64: "d", PY_INT: "q", PY_FLOAT: "d", I8X4: "I", U8X4: "I", I16X2: "I", U16X2: "I",
}  # fmt: skip

type Number = int | float
type Value = Number | CUDATensor | Mapping[str, Value]
type ArrayInterface = dict[str, int | str | tuple[int, ...] | tuple[int, bool] | None]
type Empty[T: Scalar | bool] = (
    CUDATensor[T, int] | CUDATensor[T, int, int] | CUDATensor[FloatingPoint, int] | CUDATensor[FloatingPoint, int, int]
)


class CUDAError(RuntimeError):
    """A driver or NVRTC call returned a status other than success."""


def check(status: cu.CUresult) -> None:
    if status:
        _, name = cu.cuGetErrorName(status)
        _, message = cu.cuGetErrorString(status)
        raise CUDAError(f"{(name or b'?').decode()}: {(message or b'unknown').decode()}")


class CUDAEvent(Completion):
    """A point in a stream that records the device clock, so `elapsed_since` works between any two.

    An event with subscribers is settled by the device's waiter, which blocks in the event's blocking sync.
    """

    def __init__(self, device: CUDADevice, core: CoreEvent) -> None:
        super().__init__()
        self.device, self.core = device, core

    def done(self) -> bool:
        self.device.activate()
        return self.core.is_done

    def subscribe(self, callback: Callable[[], None]) -> bool:
        done = super().subscribe(callback)
        if not done:
            self.device.waiter.watch(self)
        return done

    def _finish(self) -> None:
        self.device.activate()
        self.core.sync()

    def wait(self, timeout: float | None = None) -> None:
        """Block the calling thread until the event completes, the host side of `cudaEventSynchronize`."""
        self.device.activate()
        if timeout is None:
            self.core.sync()
            return
        deadline = time.monotonic() + timeout
        while not self.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"event not completed after {timeout} seconds")
            time.sleep(1e-4)

    def elapsed_since(self, earlier: Event) -> float:
        if not isinstance(earlier, CUDAEvent):
            raise TypeError(f"events of a CUDA device are compared with each other, not with {type(earlier).__name__}")
        if not (self.done() and earlier.done()):
            raise RuntimeError("elapsed_since needs two events that have completed")
        return self.core - earlier.core


class CUDAStream(Stream):
    """An in-order queue of device work. `handle` is the CUstream as an integer, for libraries that take a stream."""

    def __init__(self, device: CUDADevice, core: CoreStream) -> None:
        self.device, self.gpu = (
            device,
            device,
        )  # the contract types `device` as any Device, `gpu` as ours
        self.core, self.cu, self.handle = core, core.handle, int(core.handle)
        self._keeper = Keeper()
        debug.watch(self)

    def __cuda_stream__(self) -> tuple[int, int]:
        """The protocol of `cuda.core` and of torch's `ExternalStream`: (version, CUstream)."""
        return 0, self.handle

    @property
    def capturing(self) -> bool:
        self.gpu.activate()
        status, state = cu.cuStreamIsCapturing(self.cu)
        check(status)
        return state != cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE

    def describe(self) -> dict[str, debug.Json]:
        return {"type": "CUDAStream", "handle": self.handle, "capturing": self.capturing}

    def keep(self, name: str, arguments: tuple[object, ...]) -> None:
        """Hold a launch's arguments until a marker after it has completed; the device reads them until then."""
        self._keeper.keep(name, arguments)
        if self._keeper.full and not self.capturing:
            self.record()

    def pending_launches(self) -> tuple[str, ...]:
        return self._keeper.pending()

    def record(self) -> CUDAEvent:
        self.gpu.activate()
        event = CUDAEvent(self.gpu, self.core.record(options=EventOptions(timing_enabled=True, blocking_sync=True)))
        if not self.capturing:  # a capture's events cannot be queried, and a graph holds its own arguments
            self._keeper.mark(event)
        return event

    def wait(self, event: Event) -> None:
        if not isinstance(event, CUDAEvent):
            raise TypeError(f"a CUDA stream waits for CUDA events, not {type(event).__name__}")
        self.gpu.activate()
        self.core.wait(event.core)

    def sync(self) -> None:
        self.gpu.activate()
        self.core.sync()


class CUDATensor[T: Scalar | bool, *S](Tensor[T, *S]):
    """C-contiguous device memory with its element type and shape. `strides` are in bytes, as numpy's are."""

    def __init__[D: Scalar | bool](
        self, device: CUDADevice, buffer: Buffer, dtype: type[D], shape: tuple[int, ...]
    ) -> None:
        self.gpu, self.buffer, self.shape = device, buffer, shape
        # the element type is the type argument the caller promised, as in a CPUTensor
        self.dtype = cast(type[T], dtype)
        self.element, self.numpy, self.pointer = (
            from_sketch(dtype),
            numpy_dtype(dtype),
            int(buffer.handle),
        )

    @property
    def nbytes(self) -> int:
        return self.numpy.itemsize * int(np.prod(self.shape))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(int(np.prod(self.shape[axis + 1 :])) * self.numpy.itemsize for axis in range(len(self.shape)))

    @property
    def __cuda_array_interface__(self) -> ArrayInterface:
        return {
            "version": 3,
            "shape": self.shape,
            "typestr": self.numpy.str,
            "data": (self.pointer, False),
            "strides": None,
        }

    def to_host(self, stream: CUDAStream | None = None) -> np.ndarray:
        """Copy to a new numpy array, after the work queued on `stream` (default: the launch stream) has run."""
        stream = stream or self.gpu.launch_stream()
        self.gpu.activate()
        array = np.empty(self.shape, self.numpy)
        check(cu.cuMemcpyDtoHAsync(array.ctypes.data, self.pointer, self.nbytes, stream.cu)[0])
        stream.sync()
        return array

    def __repr__(self) -> str:
        return f"CUDATensor({self.numpy}{list(self.shape)} at {self.pointer:#x})"


def native(argument: Argument | Value) -> Value:
    """An argument as the packer wants it: numbers, tensors, and mappings for structs."""
    match argument:
        case CUDATensor() | int() | float():
            return argument
        case np.generic():
            return argument.item()
        case Scalar():
            return argument.value
        case Mapping():
            return {name: native(value) for name, value in argument.items()}
        case _ if is_dataclass(argument) and not isinstance(argument, type):
            return {name: native(getattr(argument, name)) for name, const in _layout(type(argument)) if not const}
        case _:
            raise TypeError(f"{argument!r} is not an argument a CUDA kernel takes")


def flatten(name: str, kind: ParamType, value: Value) -> Iterator[Number]:
    """The C values of one argument in order: the number, a tensor's pointer and sizes, or a struct's fields."""
    match kind, value:
        case Dtype(), int() | float():
            yield value
        case TensorType(), CUDATensor():
            if value.element != kind.dtype or len(value.shape) != kind.rank:
                raise TypeError(f"`{name}` must be a {kind.dtype} tensor of rank {kind.rank}, got {value}")
            yield value.pointer
            yield from value.shape
        case StructType(fields=()), _:
            return
        case StructType(), Mapping():
            for field, field_kind in kind.fields:
                yield from flatten(field, field_kind, value[field])
        case _:
            raise TypeError(f"argument `{name}` is {value!r}, expected {kind}")


def record_format(kind: ParamType) -> str:
    match kind:
        case Dtype():
            return FORMATS[kind]
        case TensorType():
            return "Q" + "q" * kind.rank
        case StructType():
            return "".join(record_format(field_kind) for _, field_kind in kind.fields)


class Parameters:
    """Where the C parameters of a kernel sit in one buffer: each starts on 8 bytes and a struct is laid out as C does.
    Tensors count one parameter per pointer and per size, a struct is one parameter."""

    def __init__(self, params: Sequence[tuple[str, ParamType]]) -> None:
        self.params = tuple(params)
        parts: list[str] = []
        self.offsets: list[int] = []
        for _, kind in self.params:
            if isinstance(kind, StructType) and not kind.fields:
                continue  # only Const fields: folded at capture, nothing to pass
            start = struct.calcsize("@" + "".join(parts))
            self.offsets += [start + 8 * i for i in range(1 + kind.rank if isinstance(kind, TensorType) else 1)]
            parts.append(record_format(kind) + "0Q")
        self.record = struct.Struct("@" + "".join(parts))

    def pack(self, arguments: Sequence[Value]) -> Packed:
        packed = Packed(self)
        packed.fill(self.values(arguments))
        return packed

    def values(self, arguments: Sequence[Value]) -> list[Number]:
        flat: list[Number] = []
        for (name, kind), argument in zip(self.params, arguments, strict=True):
            flat += flatten(name, kind, argument)
        return flat


class Packed:
    """The parameters of one launch in a buffer, and the table of their addresses that `cuLaunchKernel` reads."""

    __slots__ = ("address", "buffer", "layout", "table")

    def __init__(self, layout: Parameters) -> None:
        self.layout, self.buffer = layout, ctypes.create_string_buffer(layout.record.size)
        base = ctypes.addressof(self.buffer)
        self.table = (ctypes.c_void_p * len(layout.offsets))(*(base + offset for offset in layout.offsets))
        self.address = ctypes.addressof(self.table)

    def fill(self, values: Sequence[Number]) -> None:
        """Overwrite the values in place. A launch already queued keeps what it was given: the driver copies them."""
        self.layout.record.pack_into(self.buffer, 0, *values)


class CUDAProgram:
    """A kernel compiled for the device and loaded: everything a launch needs but the geometry and the arguments."""

    def __init__(
        self,
        ir: IrKernel,
        unit: Unit,
        function: CoreKernel,
        *,
        cooperative: bool,
        digest: str,
        decisions: Decisions,
        facts: Facts | None = None,
    ) -> None:
        self.ir, self.unit, self.function, self.cooperative, self.digest = (
            ir,
            unit,
            function,
            cooperative,
            digest,
        )
        self.handle, self.parameters = function.handle, Parameters(ir.params)
        self.decisions = decisions.floor(unit.threads)
        self.facts = facts or Facts()

    def geometry(self, launch: Geometry) -> tuple[int, int, int, int]:
        """The grid and the threads per block. A tile kernel needs a block that cooperates, where a CPU runs one."""
        if isinstance(launch, Blocks) and launch.threads is not None and launch.threads < self.unit.threads:
            raise ValueError(f"launch={launch}: tile product needs {self.unit.threads} threads")
        return dimensions(self.ir, launch, self.decisions)

    def call(self, geometry: tuple[int, int, int, int], arguments: Sequence[Argument | Value]) -> CUDACall:
        """The launch of this program over `geometry` (grid x, y, z and threads) with its arguments encoded."""
        arguments = tuple(arguments)
        return CUDACall(self, geometry, self.parameters.pack([native(arg) for arg in arguments]), arguments)

    def launch(self, stream: CUDAStream, grid: tuple[int, int, int], threads: int, packed: Packed) -> None:
        """Enqueue on `stream`. A cooperative launch makes `grid.sync()` safe for any grid the device can hold.

        A grid with a zero extent has no work, and CUDA rejects it, so it enqueues nothing, as the CPU backend does.
        """
        if 0 in grid:
            return
        stream.gpu.activate()
        shared = self.unit.shared_bytes
        if self.cooperative:
            result = cu.cuLaunchCooperativeKernel(self.handle, *grid, threads, 1, 1, shared, stream.cu, packed.address)
        else:
            result = cu.cuLaunchKernel(self.handle, *grid, threads, 1, 1, shared, stream.cu, packed.address, 0)
        check(result[0])


class CUDACall:
    """A launch with its arguments encoded: enqueueing it costs one driver call and allocates nothing.

    The Python arguments are kept too, so the stream can hold them while the device reads the pointers.
    """

    __slots__ = ("arguments", "grid", "packed", "program", "threads")

    def __init__(
        self,
        program: CUDAProgram,
        geometry: tuple[int, int, int, int],
        packed: Packed,
        arguments: tuple[Argument | Value, ...],
    ) -> None:
        x, y, z, self.threads = geometry
        self.program, self.grid, self.packed, self.arguments = program, (x, y, z), packed, arguments

    def enqueue(self, stream: CUDAStream) -> None:
        stream.keep(self.program.ir.name, self.arguments)
        session = profile_launches.session
        if session is None:
            self.program.launch(stream, self.grid, self.threads, self.packed)
        else:
            session.cuda_run(self, stream)


class CUDALauncher[**P](SourceLauncher[P, CUDAProgram, "CUDADevice"]):
    """A kernel with a launch geometry. The first call of a signature compiles it; later calls find it in a dict."""

    @staticmethod
    def extent(argument: Argument, write: bool) -> Extent:
        if not isinstance(argument, CUDATensor):
            raise TypeError(f"expected a CUDA tensor, got {type(argument).__name__}")
        return Extent(argument.pointer, argument.pointer + argument.nbytes, write)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self.device.dispatch(self.encode(self.ordered(*args, **kwargs)))

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> CUDAEvent:
        self(*args, **kwargs)
        return self.device.launch_stream().record()

    def bind(self, *args: P.args, **kwargs: P.kwargs) -> CUDABound[P]:
        return CUDABound(self, self.encode(self.ordered(*args, **kwargs)))

    def encode(self, args: tuple[Argument, ...]) -> CUDACall:
        variant = self.plan(args)
        program = variant.program
        geometry, _ = self.geometry(variant, args)
        return program.call(geometry, args)

    def geometry(
        self, variant: Variant[CUDAProgram], args: Sequence[Argument]
    ) -> tuple[tuple[int, int, int, int], str | None]:
        program = variant.program
        launch = self.launch if self.launch is not None else space.geometry(program.ir, args)
        geometry = program.geometry(launch)
        if self.launch is None and isinstance(launch, Items) and program.cooperative:
            capacity = self.device.resident_blocks(program, geometry[3])
            if geometry[0] > capacity:
                return (
                    capacity,
                    1,
                    1,
                    geometry[3],
                ), f"inferred grid {geometry[0]} capped at resident capacity {capacity}"
        return geometry, None

    def specialize(self, ir: IrKernel, decisions: Decisions) -> Variant[CUDAProgram]:
        program = self.device.program(ir, decisions)
        check_performance(ir, self.device.target.arch, decisions.settings.warn, program.facts)
        return Variant(program, decisions.floor(program.unit.threads), program.unit)


class CUDABound[**P](Bound[P]):
    """A launch whose arguments are encoded. A call is one driver call on the current stream."""

    def __init__(self, launcher: CUDALauncher[P], call: CUDACall) -> None:
        self._launcher, self._call = launcher, call

    def __call__(self) -> None:
        self._launcher.device.dispatch(self._call)

    def submit(self) -> CUDAEvent:
        self()
        return self._launcher.device.launch_stream().record()

    def rebind(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Encode other arguments. Launches already queued keep the old ones, as the driver copies parameters."""
        self._call = self._launcher.encode(self._launcher.ordered(*args, **kwargs))


class CUDAGraph[**P](Graph[P]):
    """The launches of one function captured from a stream and replayed with one driver call.

    A replay requires the arguments that were recorded: the pointers of recorded nodes are not patched.
    """

    def __init__(self, device: CUDADevice, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> None:
        self.device, self._fn, self._recorded = device, fn, (args, kwargs)
        self._executable: cu.CUgraphExec | None = None
        if device.settings.run.graphs:
            self._executable = self._capture(*args, **kwargs)
        else:
            log.info("run.graphs=False: %s replays as plain launches", getattr(fn, "__name__", fn))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        recorded_args, recorded_kwargs = self._recorded
        same = (
            len(args) == len(recorded_args)
            and all(a is b for a, b in zip(args, recorded_args, strict=True))
            and kwargs.keys() == recorded_kwargs.keys()
            and all(kwargs[name] is recorded_kwargs[name] for name in kwargs)
        )
        if self._executable is not None and same:
            session = profile_launches.session
            if session is not None:
                session.cuda_graph(self.device.target.arch)
            self.device.activate()
            check(cu.cuGraphLaunch(self._executable, self.device.launch_stream().cu)[0])
            return
        if self._executable is not None:
            raise TargetError(
                "graph replay",
                self.device.target.arch,
                "CUDA graph nodes retain their recorded arguments; krn does not patch node parameters",
                "record a new graph for these arguments or set run.graphs=False",
            )
        self._fn(*args, **kwargs)

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> CUDAEvent:
        self(*args, **kwargs)
        return self.device.launch_stream().record()

    def _capture(self, *args: P.args, **kwargs: P.kwargs) -> cu.CUgraphExec:
        self.device.activate()
        stream = self.device.stream()
        with stream:
            check(cu.cuStreamBeginCapture(stream.cu, cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)[0])
            try:
                self._fn(*args, **kwargs)
            except BaseException:  # leave the stream, and the graph it was making, as if nothing had started
                status, graph = cu.cuStreamEndCapture(stream.cu)
                if not status:
                    cu.cuGraphDestroy(graph)
                raise
            status, graph = cu.cuStreamEndCapture(stream.cu)
        check(status)
        status, executable = cu.cuGraphInstantiate(graph, 0)
        check(status)
        weakref.finalize(self, _release, self.device, executable, graph)
        return executable


def _release(device: CUDADevice, executable: cu.CUgraphExec, graph: cu.CUgraph) -> None:
    """Destroy a graph on the thread that collects it, which binds the device's context first."""
    device.activate()
    cu.cuGraphExecDestroy(executable)
    cu.cuGraphDestroy(graph)


def _make_current(ordinal: int) -> None:
    CoreDevice(ordinal).set_current()


class CUDADevice(Device):
    """One NVIDIA GPU: its streams, its stream-ordered memory and the cache of programs compiled for it."""

    def __init__(self, target: Nvidia, settings: Settings | None = None, ordinal: int = 0) -> None:
        self.target, self.nvidia, self.ordinal = target, target, ordinal
        self._settings_override = settings
        self._context = ThreadContext(partial(_make_current, ordinal))
        core = self.core
        found = core.compute_capability.major * 10 + core.compute_capability.minor
        if target.sm != found:
            raise CUDAError(f"target {target.arch} does not run on {core.name}, which is sm_{found}")
        self.name, self.cooperative = core.name, core.properties.cooperative_launch
        self.settings.validate(target)
        self.compile_options = CompileOptions.resolve(
            target,
            lineinfo=True if settings and settings.debug.lineinfo else None,
            debug=True if settings and settings.debug.debug else None,
        )
        self.waiter = Waiter(f"krn-cuda-waiter-{ordinal}")
        self._default_stream = CUDAStream(self, core.create_stream())
        self._launchers: dict[tuple[Hashable, Launch | None, str], CUDALauncher] = {}
        self._programs: dict[str, CUDAProgram] = {}
        self._builds: dict[str, threading.Lock] = {}
        self._resident: dict[tuple[str, int], int] = {}

    def activate(self) -> None:
        """Bind the device's context to the calling thread, once per thread. Each driver call the device makes starts
        with this, except those that reach `cuda.core` through `core`, which binds it first."""
        self._context.activate()

    @property
    def core(self) -> CoreDevice:
        """The `cuda.core` object of this GPU for the calling thread, whose context is bound."""
        self.activate()
        return CoreDevice(self.ordinal)

    @property
    def default_stream(self) -> CUDAStream:
        return self._default_stream

    def stream(self) -> CUDAStream:
        return CUDAStream(self, self.core.create_stream())

    def launch_stream(self) -> CUDAStream:
        """The stream launches go to: the current one of this thread or task, else the default stream."""
        match current_stream.get():
            case None:
                return self._default_stream
            case CUDAStream() as stream:
                return stream
        raise TypeError("the current stream belongs to another backend")

    def launcher[**P, R, S, **Q](self, kernel: Kernel[P, R, S, Q], launch: Launch | None = None) -> CUDALauncher[P]:
        snapshot = self._settings_override or current_settings()
        key = (kernel, launch, snapshot.key)
        try:
            return self._launchers[key]
        except KeyError:
            return self._launchers.setdefault(key, CUDALauncher(self, kernel, launch, snapshot))

    @overload
    def empty[T: Scalar | bool](self, dtype: type[T], length: int, /) -> CUDATensor[T, int]: ...
    @overload
    def empty[T: Scalar | bool](self, dtype: type[T], *shape: int) -> CUDATensor[T, int, int]: ...
    @overload
    def empty(self, dtype: Selector, length: int, /) -> CUDATensor[FloatingPoint, int]: ...
    @overload
    def empty(self, dtype: Selector, *shape: int) -> CUDATensor[FloatingPoint, int, int]: ...
    def empty[T: Scalar | bool](self, dtype: type[T] | Selector, *shape: int) -> Empty[T]:
        """Uninitialized memory allocated on the launch stream. A selector resolves for the target of this device."""
        resolved = dtype.resolve(self.target) if isinstance(dtype, Selector) else dtype
        size = numpy_dtype(resolved).itemsize * int(np.prod(shape))
        return CUDATensor(
            self,
            self.core.allocate(max(size, 1), stream=self.launch_stream().core),
            resolved,
            shape,
        )

    def to_device(self, array: np.ndarray, stream: CUDAStream | None = None) -> CUDATensor:
        """A device copy of `array`, enqueued on `stream` (default: the launch stream)."""
        array = np.ascontiguousarray(array)
        stream = stream or self.launch_stream()
        self.activate()
        try:
            dtype = to_sketch(DTYPES[array.dtype])
        except KeyError:
            raise TypeError(f"{array.dtype} has no representation on the device") from None
        tensor = self.empty(dtype, *array.shape)
        check(cu.cuMemcpyHtoDAsync(tensor.pointer, array.ctypes.data, array.nbytes, stream.cu)[0])
        return tensor

    def graph[**P](self, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> CUDAGraph[P]:
        return CUDAGraph(self, fn, *args, **kwargs)

    async def synchronize(self) -> None:
        """Wait for everything queued so far on every stream of the device."""
        await asyncio.get_running_loop().run_in_executor(None, self._synchronize)

    def dispatch(self, call: CUDACall) -> None:
        """Enqueue on the current stream."""
        call.enqueue(self.launch_stream())

    def resident_blocks(self, program: CUDAProgram, threads: int) -> int:
        key = program.digest, threads
        if key in self._resident:
            return self._resident[key]
        self.activate()
        status, function = cu.cuKernelGetFunction(program.handle)
        check(status)
        status, blocks = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, threads, program.unit.shared_bytes)
        check(status)
        capacity = blocks * self.core.properties.multiprocessor_count
        self._resident[key] = capacity
        return capacity

    def program(self, ir: IrKernel, decisions: Decisions | None = None) -> CUDAProgram:
        """The program of a captured kernel: found in a dict, else built once behind the lock of its content hash."""
        decisions = decisions or decide(ir, self.target, self.settings)
        options = CompileOptions.resolve(self.target, settings=decisions.settings)
        unit = Emitter(ir, CUDARenderer(self.nvidia, decisions), lineinfo=options.lineinfo).emit()
        decisions = decisions.floor(unit.threads)
        flags = options.flags()
        digest = hashlib.sha256("\0".join([nvrtc_version(), *flags, unit.source]).encode()).hexdigest()
        try:
            return self._programs[digest]
        except KeyError:
            pass
        with self._builds.setdefault(digest, threading.Lock()):
            if digest not in self._programs:
                self._programs[digest] = self._load(ir, unit, digest, flags, decisions)
            return self._programs[digest]

    def _synchronize(self) -> None:
        self.core.sync()

    def _load(self, ir: IrKernel, unit: Unit, digest: str, flags: tuple[str, ...], decisions: Decisions) -> CUDAProgram:
        self.activate()
        cooperative = ir.cooperative
        if cooperative and not self.cooperative:
            raise CUDAError(f"{ir.name} calls grid.sync(), which needs a cooperative launch; {self.name} has none")
        started = time.perf_counter()
        result = self._compile_result(unit, digest, flags, decisions.settings)
        function = ObjectCode.from_cubin(result.cubin, name=unit.entry).get_kernel(unit.entry)
        if unit.shared_bytes > SHARED_LIMIT:
            attribute = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            check(cu.cuKernelSetAttribute(attribute, unit.shared_bytes, function.handle, cu.CUdevice(self.ordinal))[0])
        threads = decisions.block.value
        blocks = function.occupancy.max_active_blocks_per_multiprocessor(threads, unit.shared_bytes)
        occupancy = blocks * threads / self.core.properties.max_threads_per_multiprocessor
        check_performance(
            ir,
            self.target.arch,
            decisions.settings.warn,
            Facts(registers=result.registers, spilled_bytes=result.spilled_bytes, occupancy=occupancy),
        )
        log.debug("%s: load %.1f ms", ir.name, 1e3 * (time.perf_counter() - started))
        return CUDAProgram(
            ir,
            unit,
            function,
            cooperative=cooperative,
            digest=digest,
            decisions=decisions,
            facts=Facts(registers=result.registers, spilled_bytes=result.spilled_bytes, occupancy=occupancy),
        )

    def _cubin(self, unit: Unit, digest: str, flags: tuple[str, ...], settings: Settings | None = None) -> bytes:
        return self._compile_result(unit, digest, flags, settings or self.settings).cubin

    def _compile_result(self, unit: Unit, digest: str, flags: tuple[str, ...], settings: Settings) -> Compilation:
        cache = settings.cache.dir
        path = cache / "cuda" / f"{digest}.cubin" if cache else None
        metadata = path.with_suffix(".json") if path else None
        if path is not None and path.exists() and metadata is not None and metadata.exists():
            facts = json.loads(metadata.read_text())
            return Compilation(
                cubin=path.read_bytes(), registers=facts["registers"], spilled_bytes=facts["spilled_bytes"]
            )
        result = compile_result(unit.source, flags)
        if path is not None and metadata is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(result.cubin)
            temporary.replace(path)
            temporary = metadata.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"registers": result.registers, "spilled_bytes": result.spilled_bytes}))
            temporary.replace(metadata)
        return result


@cache
def nvrtc_version() -> str:
    """The NVRTC that compiles, part of every cache key: a cubin from another compiler is another program."""
    status, major, minor = nvrtc.nvrtcVersion()
    _nvrtc(status, "version")
    return f"nvrtc {major}.{minor}"


def compile_cubin(source: str, flags: Sequence[str]) -> bytes:
    return compile_result(source, flags).cubin


def compile_result(source: str, flags: Sequence[str]) -> Compilation:
    """NVRTC through `cuda.bindings`: the cubin of CUDA C++ source under `numerical_options`, or a CUDAError with the
    compiler's log."""
    status, program = nvrtc.nvrtcCreateProgram(source.encode(), b"unit.cu", 0, [], [])
    _nvrtc(status, "create")
    try:
        options = [flag.encode() for flag in (*numerical_options(flags), "--ptxas-options=-v")]
        (status,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
        _, size = nvrtc.nvrtcGetProgramLogSize(program)
        text = bytearray(size)
        nvrtc.nvrtcGetProgramLog(program, text)
        compiler_log = bytes(text).rstrip(b"\0").decode(errors="replace")
        if status:
            raise CUDAError(f"nvrtcCompileProgram failed with {status.name}\n{compiler_log}")
        if compiler_log:
            log.debug("nvrtc: %s", compiler_log)
        _, size = nvrtc.nvrtcGetCUBINSize(program)
        cubin = bytearray(size)
        _nvrtc(nvrtc.nvrtcGetCUBIN(program, cubin)[0], "cubin")
        return compilation(bytes(cubin), compiler_log)
    finally:
        nvrtc.nvrtcDestroyProgram(program)


def _nvrtc(status: nvrtc.nvrtcResult, what: str) -> None:
    if status:
        raise CUDAError(f"nvrtc {what} failed with {status.name}")


class CUDABackend(Backend):
    """The `nvidia` vendor. One device per architecture, so every thread that asks for the default device shares it."""

    vendor = "nvidia"

    def __init__(self) -> None:
        self._devices: dict[str, CUDADevice] = {}
        self._lock = threading.Lock()

    def device(self, target: Target) -> CUDADevice:
        if not isinstance(target, Nvidia):
            raise TypeError(f"the nvidia backend builds devices for Nvidia targets, not {target.name}")
        with self._lock:
            if target.arch not in self._devices:
                self._devices[target.arch] = CUDADevice(target)
            return self._devices[target.arch]


BACKEND = CUDABackend()
