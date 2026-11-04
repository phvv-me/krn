"""Apple's host backend over PyObjC Metal and shared unified memory.

Each launch commits one command buffer, so enqueue never waits for a later flush. Events signal an MTLSharedEvent
and the device's waiter observes command completion, including errors. Graphs replay by re-encoding immutable calls.
Programs are cached by MSL source, language version, and math options. Fast math is disabled to preserve sqrt
accuracy and floating-point semantics. An explicit KRN_CACHE_DIR is refused: newLibraryWithData can
load a metallib, but MTLLibrary has no API to serialize a library compiled from source into one.
"""

import asyncio
import hashlib
import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, cast, overload

import Metal
import numpy as np

from krn import Kernel, Tensor
from krn.compiler import space
from krn.compiler.debug import Json
from krn.compiler.dtypes import (
    BF16,
    BOOL,
    I8X4,
    I16X2,
    PY_FLOAT,
    PY_INT,
    U8X4,
    U16X2,
    U32,
    Dtype,
    Kind,
    from_sketch,
    to_sketch,
    uint,
)
from krn.compiler.emit import Emitter, dimensions
from krn.compiler.errors import TargetError
from krn.compiler.heuristics import Decisions, Extent, decide
from krn.compiler.ir import Kernel as IrKernel
from krn.compiler.ir import StructType, TensorType
from krn.compiler.msl import LANGUAGE_VERSION, MSLRenderer
from krn.compiler.performance import Facts
from krn.compiler.performance import check as check_performance
from krn.facts.targets import Apple, Target
from krn.frontend.geometry import Blocks, Geometry
from krn.frontend.numerics import FloatingPoint, Scalar, Selector

from ..profile import launches as profile_launches
from .api import Backend, Bound, Device, Event, Graph, Launch, Settings, Stream, current_stream
from .compile_options import CompileOptions
from .completion import Keeper, Waiter
from .cpu import NUMPY
from .runtime import DTYPES, Argument, CPUEvent, SourceLauncher, Variant, _layout
from .settings import current_settings

SCALARS = NUMPY | {
    BF16: np.dtype(np.uint16),
    BOOL: np.dtype(np.bool_),
    I8X4: np.dtype(np.uint32),
    U8X4: np.dtype(np.uint32),
    I16X2: np.dtype(np.uint32),
    U16X2: np.dtype(np.uint32),
    PY_INT: np.dtype(np.int64),
    PY_FLOAT: np.dtype(np.float32),
}
type Value = int | float | MetalTensor | Mapping[str, Value]
type Empty[T: Scalar | bool] = (
    MetalTensor[T, int]
    | MetalTensor[T, int, int]
    | MetalTensor[FloatingPoint, int]
    | MetalTensor[FloatingPoint, int, int]
)
_recording: ContextVar[MetalCapture | None] = ContextVar("krn_metal_recording", default=None)


@dataclass(frozen=True, kw_only=True)
class MetalCapture:
    stream: MetalStream
    calls: list[MetalCall]


def _outside_capture() -> None:
    if _recording.get() is not None:
        raise RuntimeError("Metal graph capture accepts launches only")


class MetalError(RuntimeError):
    """Metal could not allocate, compile, or complete work."""


class MetalTensor[T: Scalar | bool, *S](Tensor[T, *S]):
    """Shared storage. The numpy view aliases the MTLBuffer; order GPU writes before reading it."""

    def __init__[D: Scalar | bool](
        self, device: MetalDevice, buffer: Metal.MTLBuffer, dtype: type[D], shape: tuple[int, ...]
    ) -> None:
        self.gpu, self.buffer, self.shape = device, buffer, shape
        # The generic element type is the type promised by empty, as in CPUTensor.
        self.dtype = cast(type[T], dtype)
        self.numpy = SCALARS[from_sketch(dtype)]
        self.nbytes = self.numpy.itemsize * int(np.prod(shape))
        self.array = np.frombuffer(buffer.contents().as_buffer(self.nbytes), self.numpy).reshape(shape)
        self.pointer = self.array.ctypes.data

    @property
    def strides(self) -> tuple[int, ...]:
        return self.array.strides

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        return self.array.astype(dtype or self.numpy, copy=bool(copy))

    def to_host(self, stream: MetalStream | None = None) -> np.ndarray:
        (stream or self.gpu.launch_stream()).sync()
        return self.array.copy()

    def __repr__(self) -> str:
        return f"MetalTensor({self.numpy}{list(self.shape)})"


class MetalEvent(CPUEvent):
    """A shared-event signal plus command completion, settled by the device's waiter."""

    def __init__(self, stream: MetalStream, commands: tuple[Metal.MTLCommandBuffer, ...]) -> None:
        super().__init__()
        self.device, self.shared, self.commands = stream.device, stream.shared, commands
        self.value = stream.sequence
        stream.device.waiter.watch(self)

    def _finish(self) -> None:
        error = None
        for command in self.commands:
            command.waitUntilCompleted()
            if (failure := command.error()) is not None:
                error = MetalError(str(failure))
        if error is not None:
            raise error

    def elapsed_since(self, earlier: Event) -> float:
        if not isinstance(earlier, MetalEvent) or earlier.device is not self.device:
            raise TypeError("Metal events must belong to the same device")
        if not self.done() or not earlier.done():
            raise RuntimeError("elapsed_since needs two events that have completed")
        self.wait()
        earlier.wait()
        return (self.commands[-1].GPUEndTime() - earlier.commands[-1].GPUEndTime()) * 1000


class MetalStream(Stream):
    """One in-order command queue. A command buffer per launch makes submission immediate."""

    def __init__(self, device: MetalDevice) -> None:
        self.device, self.queue = device, device.core.newCommandQueue()
        self.shared = device.core.newSharedEvent()
        self.sequence = 0
        self.commands: list[Metal.MTLCommandBuffer] = []
        self.lock = threading.RLock()
        self._keeper = Keeper()

    @property
    def capturing(self) -> bool:
        return _recording.get() is not None

    def describe(self) -> dict[str, Json]:
        return {"type": "MetalStream", "submitted": len(self.commands)}

    def pending_launches(self) -> tuple[str, ...]:
        return self._keeper.pending()

    def keep(self, name: str, arguments: tuple[object, ...]) -> None:
        """Hold a launch's buffers until a marker after it has completed; the GPU reads them until then."""
        self._keeper.keep(name, arguments)
        if self._keeper.full:
            self.record()

    def record(self) -> MetalEvent:
        _outside_capture()
        with self.lock:
            self.sequence += 1
            command = self.queue.commandBuffer()
            command.encodeSignalEvent_value_(self.shared, self.sequence)
            command.commit()
            self.commands.append(command)
            event = MetalEvent(self, tuple(self.commands))
            self._keeper.mark(event)
            return event

    def wait(self, event: Event) -> None:
        _outside_capture()
        if not isinstance(event, MetalEvent) or event.device is not self.device:
            raise TypeError("a Metal stream waits for events of the same device")
        with self.lock:
            command = self.queue.commandBuffer()
            command.encodeWaitForEvent_value_(event.shared, event.value)
            command.commit()
            self.commands.append(command)

    def sync(self) -> None:
        with self.lock:
            event = self.record()
            self.commands.clear()
        event.wait()


@dataclass(frozen=True, kw_only=True)
class MetalCall:
    program: MetalProgram
    geometry: tuple[int, int, int, int]
    buffers: tuple[MetalTensor, ...]
    scalars: bytes

    def __post_init__(self) -> None:
        x, y, z, threads = self.geometry
        if min(x, y, z) < 0 or threads <= 0:
            raise ValueError("Metal grid extents must be nonnegative and threads positive")

    def enqueue(self, stream: MetalStream) -> None:
        x, y, z, threads = self.geometry
        if 0 in (x, y, z):
            return
        if any(tensor.gpu is not stream.device for tensor in self.buffers):
            raise ValueError("Metal arguments must belong to the launch device")
        stream.keep(self.program.ir.name, self.buffers)
        with stream.lock:
            command = stream.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self.program.pipeline)
            for index, tensor in enumerate(self.buffers):
                encoder.setBuffer_offset_atIndex_(tensor.buffer, 0, index)
            if self.scalars:
                encoder.setBytes_length_atIndex_(self.scalars, len(self.scalars), len(self.buffers))
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(x, y, z), Metal.MTLSizeMake(threads, 1, 1)
            )
            encoder.endEncoding()
            session = profile_launches.session
            if session is not None:
                session.metal_enqueue(self, stream, command)
            command.commit()
            stream.commands.append(command)


class MetalProgram:
    """A compiled pipeline and scalar layout, shared by launches with different buffers and geometry."""

    def __init__(self, device: MetalDevice, ir: IrKernel, decisions: Decisions) -> None:
        self.ir = ir
        self.unit = Emitter(ir, MSLRenderer(device.apple, decisions), lineinfo=False).emit()
        self.decisions = decisions.floor(self.unit.threads)
        options = Metal.MTLCompileOptions.new()
        options.setLanguageVersion_(getattr(Metal, f"MTLLanguageVersion{LANGUAGE_VERSION[0]}_{LANGUAGE_VERSION[1]}"))
        options.setMathMode_(Metal.MTLMathModeSafe)
        options.setMathFloatingPointFunctions_(Metal.MTLMathFloatingPointFunctionsPrecise)
        library, error = device.core.newLibraryWithSource_options_error_(self.unit.source, options, None)
        if library is None:
            raise MetalError(f"{error}\n{self.unit.source}")
        pipeline, error = device.core.newComputePipelineStateWithFunction_error_(
            library.newFunctionWithName_(self.unit.entry), None
        )
        if pipeline is None:
            raise MetalError(str(error))
        self.pipeline = pipeline
        self.limit = pipeline.maxTotalThreadsPerThreadgroup()
        check_performance(ir, device.target.arch, decisions.settings.warn, Facts(thread_limit=self.limit))
        self.pointers, scalars = MSLRenderer.slots(ir)
        self.layout = np.dtype([(slot.name, SCALARS[slot.dtype]) for slot in scalars], align=True)

    def geometry(self, launch: Geometry, decisions: Decisions | None = None) -> tuple[int, int, int, int]:
        decisions = decisions or self.decisions
        if isinstance(launch, Blocks) and launch.threads is not None and launch.threads < self.unit.threads:
            raise ValueError(f"launch={launch}: tile product needs {self.unit.threads} threads")
        geometry = dimensions(self.ir, launch, self.decisions)
        if geometry[3] > self.limit:
            raise ValueError(f"{geometry[3]} threads exceeds the Metal pipeline limit {self.limit}")
        check_performance(
            self.ir,
            self.decisions.target,
            decisions.settings.warn,
            Facts(threads=geometry[3], warp_size=self.pipeline.threadExecutionWidth(), thread_limit=self.limit),
        )
        return geometry

    def call(self, geometry: tuple[int, int, int, int], arguments: Sequence[Value]) -> MetalCall:
        buffers: dict[str, MetalTensor] = {}
        scalars = np.zeros((), self.layout)
        for (name, kind), value in zip(self.ir.params, arguments, strict=True):
            self._bind(name, kind, value, buffers, scalars)
        return MetalCall(
            program=self,
            geometry=geometry,
            buffers=tuple(buffers[slot.name] for slot, _ in self.pointers),
            scalars=scalars.tobytes(),
        )

    def _bind(
        self,
        name: str,
        kind: Dtype | TensorType | StructType,
        value: Value,
        buffers: dict[str, MetalTensor],
        scalars: np.ndarray,
    ) -> None:
        match kind, value:
            case Dtype(), int() | float():
                scalars[name] = value
            case TensorType(), MetalTensor():
                storage = from_sketch(value.dtype)
                # BF16 buffers store raw ushort words, packed parameters u32 words; numpy has no BF16 dtype.
                words = U32 if kind.dtype.kind is Kind.PACKED else kind.dtype
                matches = storage == words or (kind.dtype == BF16 and storage == uint(16))
                if not matches or len(value.shape) != kind.rank:
                    raise TypeError(f"argument `{name}` does not match {kind}")
                buffers[name] = value
                for axis, size in enumerate(value.shape):
                    scalars[f"{name}_n{axis}"] = size
            case StructType(fields=()), _:
                return
            case StructType(), Mapping():
                for field, field_kind in kind.fields:
                    self._bind(f"{name}_{field}", field_kind, value[field], buffers, scalars)
            case _:
                raise TypeError(f"argument `{name}` is {value!r}, expected {kind}")


def native(argument: Argument) -> Value:
    """Marshal dataclasses recursively without moving tensor storage."""
    match argument:
        case MetalTensor() | int() | float():
            return argument
        case np.generic():
            return argument.item()
        case Scalar():
            return argument.value
        case _ if is_dataclass(argument) and not isinstance(argument, type):
            return {name: native(getattr(argument, name)) for name, const in _layout(type(argument)) if not const}
        case _:
            raise TypeError(f"{argument!r} is not an argument a Metal kernel takes")


class MetalLauncher[**P](SourceLauncher[P, MetalProgram, "MetalDevice"]):
    @staticmethod
    def extent(argument: Argument, write: bool) -> Extent:
        if not isinstance(argument, MetalTensor):
            raise TypeError(f"expected a Metal tensor, got {type(argument).__name__}")
        return Extent(argument.pointer, argument.pointer + argument.nbytes, write)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self.device.dispatch(self.encode(self.ordered(*args, **kwargs)))

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> MetalEvent:
        self(*args, **kwargs)
        return self.device.launch_stream().record()

    def bind(self, *args: P.args, **kwargs: P.kwargs) -> MetalBound[P]:
        return MetalBound(self, self.encode(self.ordered(*args, **kwargs)))

    def encode(self, args: tuple[Argument, ...]) -> MetalCall:
        variant = self.plan(args)
        geometry, _ = self.geometry(variant, args)
        return variant.program.call(geometry, [native(arg) for arg in args])

    def geometry(
        self, variant: Variant[MetalProgram], args: Sequence[Argument]
    ) -> tuple[tuple[int, int, int, int], str | None]:
        launch = self.launch if self.launch is not None else space.geometry(variant.program.ir, args)
        return variant.program.geometry(launch, variant.decisions), None

    def specialize(self, ir: IrKernel, decisions: Decisions) -> Variant[MetalProgram]:
        program = self.device.program(ir, decisions)
        return Variant(program, decisions.floor(program.unit.threads), program.unit)


class MetalBound[**P](Bound[P]):
    def __init__(self, launcher: MetalLauncher[P], call: MetalCall) -> None:
        self._launcher, self._call = launcher, call

    def __call__(self) -> None:
        self._launcher.device.dispatch(self._call)

    def submit(self) -> MetalEvent:
        self()
        return self._launcher.device.launch_stream().record()

    def rebind(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self._call = self._launcher.encode(self._launcher.ordered(*args, **kwargs))


class MetalGraph[**P](Graph[P]):
    """Record launches on one stream; replay requires the recorded arguments."""

    def __init__(self, device: MetalDevice, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> None:
        self.device, self.fn, self.args, self.kwargs = device, fn, args, kwargs
        self.calls: list[MetalCall] = []
        if device.settings.run.graphs:
            if _recording.get() is not None:
                raise RuntimeError("nested Metal graph capture")
            token = _recording.set(MetalCapture(stream=device.launch_stream(), calls=self.calls))
            try:
                fn(*args, **kwargs)
            finally:
                _recording.reset(token)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        same = (
            len(args) == len(self.args)
            and all(a is b for a, b in zip(args, self.args, strict=True))
            and kwargs.keys() == self.kwargs.keys()
            and all(kwargs[k] is self.kwargs[k] for k in kwargs)
        )
        if self.device.settings.run.graphs and same:
            for call in self.calls:
                self.device.dispatch(call)
        elif not self.device.settings.run.graphs:
            self.fn(*args, **kwargs)
        else:
            raise TargetError(
                "graph replay",
                self.device.target.arch,
                "Metal graph calls retain their recorded buffers and cannot rebind arguments",
                "record a new graph for these arguments or set run.graphs=False",
            )

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> MetalEvent:
        self(*args, **kwargs)
        return self.device.launch_stream().record()


class MetalDevice(Device):
    """The system default Apple GPU, shared buffers, queues, and a source-content program cache."""

    def __init__(self, target: Target, settings: Settings | None = None) -> None:
        if not isinstance(target, Apple):
            raise TypeError("Metal requires an Apple target")
        self._settings_override = settings
        self.target, self.apple = target, target
        self.settings.validate(target)
        self.compile_options = CompileOptions.resolve(
            target,
            lineinfo=True if settings and settings.debug.lineinfo else None,
            debug=True if settings and settings.debug.debug else None,
        )
        if (core := Metal.MTLCreateSystemDefaultDevice()) is None:
            raise MetalError("no Metal device")
        if not core.hasUnifiedMemory() or not core.supportsFamily_(1000 + target.family):
            raise MetalError(f"{target.arch} requires a unified-memory Apple{target.family} device")
        self.core, self.name = core, core.name()
        self.waiter = Waiter("krn-metal-waiter")
        self._streams: list[MetalStream] = []
        self._default_stream = self.stream()
        self._launchers: dict[tuple[Hashable, Launch | None, str], MetalLauncher] = {}
        self._programs: dict[str, MetalProgram] = {}
        self._lock = threading.RLock()

    @property
    def default_stream(self) -> MetalStream:
        return self._default_stream

    def stream(self) -> MetalStream:
        stream = MetalStream(self)
        self._streams.append(stream)
        return stream

    def launch_stream(self) -> MetalStream:
        match current_stream.get():
            case None:
                return self._default_stream
            case MetalStream(device=device) as stream if device is self:
                return stream
        raise TypeError("the current stream belongs to another device")

    def launcher[**P, R, S, **Q](self, kernel: Kernel[P, R, S, Q], launch: Launch | None = None) -> MetalLauncher[P]:
        with self._lock:
            snapshot = self._settings_override or current_settings()
            key = kernel, launch, snapshot.key
            if key not in self._launchers:
                self._launchers[key] = MetalLauncher(self, kernel, launch, snapshot)
            return self._launchers[key]

    if TYPE_CHECKING:

        @overload
        def empty[T: Scalar | bool](self, dtype: type[T], length: int, /) -> MetalTensor[T, int]: ...
        @overload
        def empty[T: Scalar | bool](self, dtype: type[T], *shape: int) -> MetalTensor[T, int, int]: ...
        @overload
        def empty(self, dtype: Selector, length: int, /) -> MetalTensor[FloatingPoint, int]: ...
        @overload
        def empty(self, dtype: Selector, *shape: int) -> MetalTensor[FloatingPoint, int, int]: ...

    def empty[T: Scalar | bool](self, dtype: type[T] | Selector, *shape: int) -> Empty[T]:
        _outside_capture()
        if any(size < 0 for size in shape):
            raise ValueError("negative tensor dimension")
        resolved = dtype.resolve(self.target) if isinstance(dtype, Selector) else dtype
        size = SCALARS[from_sketch(resolved)].itemsize * int(np.prod(shape))
        buffer = self.core.newBufferWithLength_options_(max(size, 1), Metal.MTLResourceStorageModeShared)
        if buffer is None:
            raise MetalError(f"could not allocate {size} bytes")
        return MetalTensor(self, buffer, resolved, shape)

    def to_device(self, array: np.ndarray) -> MetalTensor:
        try:
            dtype = to_sketch(DTYPES[array.dtype])
        except KeyError:
            raise TypeError(f"{array.dtype} has no representation on the device") from None
        tensor = self.empty(dtype, *array.shape)
        tensor.array[...] = array
        return tensor

    def graph[**P](self, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> MetalGraph[P]:
        return MetalGraph(self, fn, *args, **kwargs)

    async def synchronize(self) -> None:
        _outside_capture()
        recorded = [(stream, stream.record()) for stream in tuple(self._streams)]
        await asyncio.gather(*(event for _, event in recorded))
        for stream, event in recorded:
            with stream.lock:
                completed = {id(command) for command in event.commands}
                stream.commands[:] = [command for command in stream.commands if id(command) not in completed]

    def dispatch(self, call: MetalCall) -> None:
        if (capture := _recording.get()) is not None:
            if self.launch_stream() is not capture.stream:
                raise RuntimeError("Metal graph capture requires one stream on one device")
            capture.calls.append(call)
        else:
            call.enqueue(self.launch_stream())

    def program(self, ir: IrKernel, decisions: Decisions | None = None) -> MetalProgram:
        decisions = decisions or decide(ir, self.target, self.settings)
        decisions.settings.validate(self.target)
        unit = Emitter(ir, MSLRenderer(self.apple, decisions), lineinfo=False).emit()
        options = CompileOptions.resolve(self.target, settings=decisions.settings)
        digest = hashlib.sha256(f"{LANGUAGE_VERSION}\0{options}\0{unit.source}".encode()).hexdigest()
        with self._lock:
            if digest not in self._programs:
                self._programs[digest] = MetalProgram(self, ir, decisions)
            return self._programs[digest]


class MetalBackend(Backend):
    vendor = "apple"

    def __init__(self) -> None:
        self._devices: dict[str, MetalDevice] = {}
        self._lock = threading.Lock()

    def device(self, target: Target) -> MetalDevice:
        if not isinstance(target, Apple):
            raise TypeError(f"the apple backend requires an Apple target, not {target.name}")
        with self._lock:
            if target.arch not in self._devices:
                self._devices[target.arch] = MetalDevice(target)
            return self._devices[target.arch]


BACKEND = MetalBackend()
