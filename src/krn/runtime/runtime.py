"""The host side of krn.gpu on a CPU: the contract of `krn.runtime.api` made real with numpy, clang and threads.

A device owns a pool of worker threads. A stream is a serial lane, not a thread: a deque of tasks plus a flag that says
a pool thread is draining it, so a busy lane holds exactly one worker and an idle one holds none. A launch is a task. An
event is a marker task that stamps the clock and wakes its subscribers, which is how `await event` resolves an asyncio
future from a worker with `call_soon_threadsafe` and how one lane waits for another without blocking any thread. The
first call of a signature captures and compiles on the calling thread behind a per-key lock; every later call reads
plain dicts. A graph records the calls of a function once, derives dependencies from the tensors they read and write,
and replays the DAG on the pool.
"""

import asyncio
import hashlib
import inspect
import logging
import operator
import os
import sys
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache, cached_property, partial
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Self, cast, get_origin, overload

import numpy as np
from numpy.lib.array_utils import byte_bounds

from krn import Const, Kernel, Tensor, debug
from krn.compiler import space
from krn.compiler.capture import ConstValue, ConstValues, capture, roles
from krn.compiler.dtypes import Arithmetic, Dtype, Kind, from_sketch, to_sketch
from krn.compiler.emit import Emitter, Unit, dimensions
from krn.compiler.heuristics import Decisions, Explanation, Extent, aliased, decide, space_text
from krn.compiler.ir import Kernel as IrKernel
from krn.compiler.ir import StructType, TensorType
from krn.compiler.performance import Facts
from krn.compiler.performance import check as check_performance
from krn.facts.targets import CPU, Target
from krn.frontend.numerics import FloatingPoint, Scalar, Selector

from ..profile import launches as profile_launches
from .api import HostArgument, Launch
from .compile_options import CompileOptions
from .completion import Completion
from .cpu import NUMPY, STACK_BYTES, CPUProgram, CPURenderer, Native, Value
from .host import Backend, Bound, Device, Event, Graph, Launcher, Settings, Stream, current_stream
from .settings import current_settings, retained, snapshot_scope

if TYPE_CHECKING:
    import ctypes

    from _typeshed import DataclassInstance

    from ..profile.profiler import Profiler

log = logging.getLogger(__name__)
_diagnostics = threading.Lock()

DTYPES = {dtype: proto for proto, dtype in NUMPY.items() if proto.kind is not Kind.PACKED}  # a word is a u32 array
WORKER_STACK = STACK_BYTES + (2 << 20)  # the stack arrays of a tile kernel, plus room for the frames around them
SPIN_NS = 100_000  # how long a lane keeps its worker after running dry, before the worker sleeps (free-threaded only)

type Argument = HostArgument
type Signature = tuple[Hashable, ...]
type Empty[T: Scalar | bool] = (
    CPUTensor[T, int] | CPUTensor[T, int, int] | CPUTensor[FloatingPoint, int] | CPUTensor[FloatingPoint, int, int]
)

_recording: ContextVar[list[Call] | None] = ContextVar("krn_cpu_recording", default=None)


def numpy_dtype[S: Scalar | bool](dtype: type[S]) -> np.dtype:
    """The numpy dtype of a krn scalar type (`T.f32` is `fp[L[8], L[23], ieee]`)."""
    try:
        return NUMPY[from_sketch(dtype)]
    except KeyError, TypeError:
        raise TypeError(f"{dtype} has no host representation on a CPU") from None


class CPUTensor[T: Scalar | bool, *S](Tensor[T, *S]):
    """A C-contiguous numpy array as a krn tensor. `CPUTensor[f32, int](array)` wraps an array without a copy."""

    def __init__(self, array: np.ndarray) -> None:
        if not array.flags.c_contiguous:
            raise ValueError("a CPU tensor wraps a C-contiguous array; use np.ascontiguousarray first")
        if array.dtype not in DTYPES:
            raise TypeError(f"a CPU tensor holds {', '.join(map(str, DTYPES))}, not {array.dtype}")
        self.array, self.shape = array, array.shape
        # the element type is the type argument the caller promised, checked here against the array it wraps
        self.dtype = cast(type[T], to_sketch(DTYPES[array.dtype]))

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        return self.array if dtype is None else self.array.astype(dtype, copy=bool(copy))

    def __repr__(self) -> str:
        return f"CPUTensor({self.array.dtype}{list(self.shape)})"

    def to_host(self) -> np.ndarray:
        """Copy host storage after the caller has ordered pending writes."""
        return self.array.copy()


@cache
def _layout(struct: type[DataclassInstance]) -> tuple[tuple[str, bool], ...]:
    """The fields of a `@struct` class with a flag for the `Const` ones."""
    return tuple((field.name, get_origin(field.type) is Const) for field in fields(struct))


def _kind(argument: Argument) -> Hashable:
    """What compiling depends on in one argument: dtype and rank of an array, the Const values of a struct."""
    if isinstance(argument, CPUTensor):
        argument = argument.array
    if isinstance(argument, np.ndarray):
        return argument.dtype, argument.ndim
    if isinstance(argument, Tensor):  # another backend's tensor: the sketch type names the element
        return argument.dtype, len(argument.shape)
    if is_dataclass(argument) and not isinstance(argument, type):
        return type(argument), tuple(
            getattr(argument, name) if const else _kind(getattr(argument, name))
            for name, const in _layout(type(argument))
        )
    return None


def _value(argument: Argument) -> Value:
    """An argument as the prototype's marshalling wants it: arrays, Python numbers and mappings for structs."""
    match argument:
        case CPUTensor():
            return argument.array
        case np.generic():
            return argument.item()
        case Scalar():
            return argument.value
        case int() | float() | np.ndarray():
            return argument
        case _:
            return {name: _value(getattr(argument, name)) for name, const in _layout(type(argument)) if not const}


def _consts(argument: DataclassInstance) -> dict[str, ConstValue]:
    return {name: getattr(argument, name) for name, const in _layout(type(argument)) if const}


def _dtypes(names: Sequence[str], args: Sequence[Argument]) -> dict[str, Dtype]:
    """Tensor element types by parameter path, including dataclass fields such as `a.values`."""
    found: dict[str, Dtype] = {}
    for name, argument in zip(names, args, strict=True):
        if isinstance(argument, CPUTensor):
            argument = argument.array
        if isinstance(argument, np.ndarray):
            found[name] = DTYPES[argument.dtype]
        elif isinstance(argument, Tensor):
            found[name] = from_sketch(argument.dtype)
        elif is_dataclass(argument) and not isinstance(argument, type):
            layout = [field for field, const in _layout(type(argument)) if not const]
            found.update(_dtypes([f"{name}.{field}" for field in layout], [getattr(argument, f) for f in layout]))
    return found


def _structs(names: Sequence[str], args: Sequence[Argument]) -> dict[str, str]:
    """The class of each struct argument by parameter name: the format a kernel written for a base class runs on."""
    pairs = zip(names, args, strict=True)
    return {name: type(arg).__name__ for name, arg in pairs if is_dataclass(arg) and not isinstance(arg, type)}


def cpu_extent(argument: Argument, write: bool) -> Extent:
    array = argument.array if isinstance(argument, CPUTensor) else argument
    if not isinstance(array, np.ndarray):
        raise TypeError(f"expected a CPU tensor, got {type(argument).__name__}")
    low, high = byte_bounds(array)
    return Extent(low, high, write)


class Countdown:
    """An integer that threads decrement atomically."""

    __slots__ = ("_lock", "_value")

    def __init__(self, value: int) -> None:
        self._lock, self._value = threading.Lock(), value

    def decrement(self) -> int:
        """Subtract one and return the new value."""
        with self._lock:
            self._value -= 1
            return self._value


class WorkerPool:
    """Daemon threads that run callables from one queue. What a callable raises is logged and the thread goes on."""

    def __init__(self, size: int, spin_ns: int = 0) -> None:
        self._queue: SimpleQueue[Callable[[], None] | None] = SimpleQueue()
        self._size, self.spin_ns = size, spin_ns
        previous = threading.stack_size(WORKER_STACK)  # the setting is process-wide, so it is restored right after
        try:
            for index in range(size):
                threading.Thread(target=self._work, name=f"krn-cpu-{index}", daemon=True).start()
        finally:
            threading.stack_size(previous)

    def submit(self, task: Callable[[], None]) -> None:
        self._queue.put(task)

    def backlog(self) -> bool:
        """Whether tasks wait for a worker: a lane that lingers should give its thread up when this is true."""
        return not self._queue.empty()

    def close(self) -> None:
        """Let the threads leave once the tasks queued before this call have run."""
        for _ in range(self._size):
            self._queue.put(None)

    def _work(self) -> None:
        while (task := self._queue.get()) is not None:
            try:
                task()
            except Exception:  # the top of a thread's stack: one failed task must not take the worker with it
                log.exception("a task failed on a pool thread")


class CPUEvent(Completion):
    """Completion of a marker in a lane. Workers resolve it; awaiting tasks, other lanes and `wait` subscribe."""

    def __init__(self) -> None:
        super().__init__()
        self._stamp = 0

    def resolve(self, error: Exception | None = None) -> None:
        """Called by the worker that reached the marker: stamp the clock, then wake every subscriber."""
        if not self.done():
            self._stamp = time.perf_counter_ns()
        super().resolve(error)

    def elapsed_since(self, earlier: Event) -> float:
        if not isinstance(earlier, CPUEvent):
            raise TypeError(f"events of the CPU device are compared with each other, not with {type(earlier).__name__}")
        return (self._stamped() - earlier._stamped()) / 1e6

    def _stamped(self) -> int:
        with self._lock:
            if not self._done:
                raise RuntimeError("elapsed_since needs two events that have completed")
            return self._stamp


class Task(ABC):
    """One item of a lane's queue."""

    __slots__ = ()

    @abstractmethod
    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        """Do the work on a pool thread. False means the lane is parked: the task calls `lane.resume()` later.

        session: the profiling session that records the work, None when it records nothing.
        """

    def names(self) -> tuple[str, ...]:
        """The kernels this task launches, for the message of a drain that cannot finish."""
        return ()


class Call(Task):
    """A launch with its arguments marshalled. Never mutated, so a `Bound` enqueues one object again and again."""

    __slots__ = ("args", "arguments", "launcher", "live", "program")

    def __init__(
        self,
        launcher: CPULauncher,
        args: tuple[Argument, ...],
        program: CPUProgram,
        arguments: tuple[Native, ...],
        live: list[ctypes.Structure],
    ) -> None:
        self.launcher, self.args, self.program, self.arguments, self.live = (
            launcher,
            args,
            program,
            arguments,
            live,
        )

    def names(self) -> tuple[str, ...]:
        return (self.program.kernel.name,)

    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        if session is None:
            self.program.function(*self.arguments)
        else:
            session.cpu_run(self, lane)
        return True


class Marker(Task):
    """Completes an event, and carries the first error the lane met since the last marker."""

    __slots__ = ("event",)

    def __init__(self, event: CPUEvent) -> None:
        self.event = event

    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        self.event.resolve(lane.take_error())
        return True


class Wait(Task):
    """Parks the lane until an event of another lane completes."""

    __slots__ = ("event",)

    def __init__(self, event: CPUEvent) -> None:
        self.event = event

    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        return self.event.subscribe(lane.resume)


class Owned(Task):
    """A task queued while a session was active, recorded by that session whatever is active when it runs."""

    __slots__ = ("session", "task")

    def __init__(self, task: Task, session: Profiler) -> None:
        self.task, self.session = task, session

    def names(self) -> tuple[str, ...]:
        return self.task.names()

    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        return self.task.run(lane, self.session)


class CPUStream(Stream):
    """A serial lane. `_runner` is held while a pool thread drains the deque or while the lane is parked on an event."""

    def __init__(self, device: CPUDevice) -> None:
        self.device, self.pool = device, device.pool
        self._tasks: deque[Task] = deque()
        self._running: Task | None = None
        self._runner = threading.Lock()
        self._error: Exception | None = None
        debug.watch(self)

    def describe(self) -> dict[str, debug.Json]:
        return {
            "type": "CPUStream",
            "queued": len(self._tasks),
            "draining": self._runner.locked(),
            "next": type(self._tasks[0]).__name__ if self._tasks else None,
            "error": repr(self._error) if self._error else None,
        }

    def record(self) -> CPUEvent:
        event = CPUEvent()
        self.enqueue(Marker(event))
        return event

    def wait(self, event: Event) -> None:
        if not isinstance(event, CPUEvent):
            raise TypeError(f"a CPU stream waits for CPU events, not {type(event).__name__}")
        self.enqueue(Wait(event))

    def enqueue(self, task: Task) -> None:
        session = profile_launches.session
        if session is None:
            self._tasks.append(task)
        else:
            calls = (task,) if isinstance(task, Call) else task._calls if isinstance(task, GraphRun) else ()
            session.cpu_enqueue(calls, self)
            self._tasks.append(Owned(task, session))
        if self._runner.acquire(blocking=False):
            self.pool.submit(self._drain)

    def pending_launches(self) -> tuple[str, ...]:
        running = () if self._running is None else self._running.names()
        return (*running, *(name for task in self._tasks for name in task.names()))

    def resume(self) -> None:
        """Continue a parked lane on the pool. The flag stays held from the park until the deque runs dry."""
        self.pool.submit(self._drain)

    def fail(self, error: Exception) -> None:
        """Remember the first error. The next marker reports it, as a CUDA stream reports a sticky error."""
        log.error("a launch failed on its stream", exc_info=error)
        self._error = self._error or error

    def take_error(self) -> Exception | None:
        error, self._error = self._error, None
        return error

    def _drain(self) -> None:
        while True:
            while self._tasks or self._linger():  # this thread is the only consumer: a non-empty deque stays so
                task = self._running = self._tasks.popleft()
                try:
                    if not task.run(self, profile_launches.session):
                        return
                except Exception as error:  # the top of this thread's stack: the stream reports it at its next marker
                    self.fail(error)
                finally:
                    self._running = None
            self._runner.release()
            if not self._tasks or not self._runner.acquire(blocking=False):
                return  # a producer that appended after the check takes the flag itself and schedules a drain

    def _linger(self) -> bool:
        """Keep the lane for `spin_ns` after the deque runs dry and say whether a task arrived.

        The next launch of a host loop usually follows within microseconds, and a worker that went to sleep costs the
        producer a futex wake, tens of microseconds on a virtual machine. The flag stays held, so producers only append.
        """
        if not self.pool.spin_ns:
            return bool(self._tasks)
        deadline = time.perf_counter_ns() + self.pool.spin_ns
        while not (self._tasks or self.pool.backlog()) and time.perf_counter_ns() < deadline:
            os.sched_yield()
        return bool(self._tasks)


@dataclass(frozen=True, slots=True)
class Variant[Spec]:
    program: Spec
    decisions: Decisions
    unit: Unit


@dataclass(slots=True)
class Specialization[Spec]:
    ir: IrKernel
    settings: Settings
    variants: dict[bool, Variant[Spec]] = field(default_factory=dict)
    choices: dict[bool, Decisions] = field(default_factory=dict)
    tensors: int = 0

    def __post_init__(self) -> None:
        def count(kind: TensorType | StructType) -> int:
            return (
                1
                if isinstance(kind, TensorType)
                else sum(count(child) for _, child in kind.fields if isinstance(child, TensorType | StructType))
            )

        self.tensors = sum(count(kind) for _, kind in self.ir.params if isinstance(kind, TensorType | StructType))


class SourceLauncher[**P, Spec, D: Device](Launcher[P], ABC):
    """A kernel read from source and specialized once per signature, with explicit or per-call inferred geometry.

    The first call of a signature captures and builds a `Spec`; later calls find it in a dict. A backend says what a
    `Spec` is and how to make one.
    """

    def __init__[R, S, **Q](
        self, device: D, kernel: Kernel[P, R, S, Q], launch: Launch | None, settings: Settings
    ) -> None:
        self.device, self.launch, self.settings = device, launch, settings
        # A host launch discards callee return types and configured-call metadata; capture reads fn and name only.
        self.kernel = cast(Kernel[P], kernel)
        self.parameters = tuple(inspect.signature(kernel.fn).parameters)
        self._plans: dict[Signature, Specialization[Spec]] = {}
        self._lock = threading.RLock()

    @cached_property
    def path(self) -> Path:
        source = inspect.getsourcefile(self.kernel.fn)
        if source is None:
            raise ValueError(f"kernel {self.kernel.name} has no source file to capture from")
        return Path(source)

    def ordered(self, *args: P.args, **kwargs: P.kwargs) -> tuple[Argument, ...]:
        """The arguments in parameter order, with keyword arguments placed. A wrong count is a TypeError."""
        # P.args is opaque to the checkers; the type checker verified the call against the kernel's own signature
        positional, named = (
            cast(tuple[Argument, ...], args),
            cast(dict[str, Argument], dict(kwargs)),
        )
        return self._ordered(positional, named)

    def _ordered(self, positional: tuple[Argument, ...], named: dict[str, Argument]) -> tuple[Argument, ...]:
        if not named and len(positional) == len(self.parameters):
            return positional
        try:
            ordered = (
                *positional,
                *(named.pop(name) for name in self.parameters[len(positional) :]),
            )
        except KeyError as missing:
            raise TypeError(f"{self.kernel.name}() is missing the argument {missing}") from None
        if named or len(ordered) != len(self.parameters):
            raise TypeError(
                f"{self.kernel.name}() takes {len(self.parameters)} arguments, got {len(ordered) + len(named)}"
            )
        return ordered

    def captured(self, target: Target, args: Sequence[Argument]) -> IrKernel:
        """The IR of this kernel for `target`, specialized on the Const fields, element types and struct classes of
        `args`."""
        consts: ConstValues = {
            name: _consts(arg)
            for name, arg in zip(self.parameters, args, strict=True)
            if is_dataclass(arg) and not isinstance(arg, type)
        }
        return capture(
            self.path,
            self.kernel.name,
            consts=consts,
            dtypes=_dtypes(self.parameters, args),
            structs=_structs(self.parameters, args),
            target=target,
            arithmetic=Arithmetic.STRICT,
        )

    @retained
    def plan(self, args: Sequence[Argument]) -> Variant[Spec]:
        snapshot = self.device._settings_override or current_settings()
        signature = (snapshot.key, *map(_kind, args))
        try:
            record = self._plans[signature]
        except KeyError:
            with self._lock:
                if signature not in self._plans:
                    with snapshot_scope(snapshot):
                        self._plans[signature] = Specialization(self.captured(self.device.target, args), snapshot)
                record = self._plans[signature]
        overlap = (
            record.settings.warn.warn_on_unproven_aliasing
            and bool(record.ir.written)
            and record.tensors > 1
            and aliased([extent for _, extent in tensor_extents(record.ir, args, self.extent)])
        )
        try:
            decisions = record.choices[overlap]
        except KeyError:
            decisions = decide(
                record.ir,
                self.device.target,
                record.settings,
                aliased=overlap,
                launch=self.launch if self.launch is not None else space.geometry(record.ir, args),
            )
            record.choices[overlap] = decisions
        check_performance(record.ir, self.device.target.arch, record.settings.warn, Facts(unproven_aliasing=overlap))
        restricted = decisions.restrict.value
        try:
            return record.variants[restricted]
        except KeyError:
            with self._lock:
                if restricted not in record.variants:
                    variant = self.specialize(record.ir, decisions)
                    geometry, _ = self.geometry(variant, args)
                    gpu = self.device.target.vendor != "cpu"
                    check_performance(
                        record.ir,
                        self.device.target.arch,
                        record.settings.warn,
                        Facts(
                            threads=geometry[3] if gpu else None,
                            warp_size=self.device.target.warp_size
                            if self.device.target.vendor in {"nvidia", "amd"}
                            else None,
                            unproven_aliasing=overlap,
                        ),
                    )
                    record.variants[restricted] = variant
                    if record.settings.debug.decisions:
                        diagnostic = logging.getLogger("krn.heuristics")
                        with _diagnostics:
                            if not any(type(handler) is logging.StreamHandler for handler in diagnostic.handlers):
                                diagnostic.addHandler(logging.StreamHandler())
                            diagnostic.setLevel(logging.INFO)
                        diagnostic.info("%s", self.explanation(args, variant))
                return record.variants[restricted]

    @overload
    def explain(self, *args: P.args, **kwargs: P.kwargs) -> Explanation: ...
    @overload
    def explain(self, *args: Argument, launch: Launch | None = None, **kwargs: Argument) -> Explanation: ...
    def explain(self, *args: Argument, launch: Launch | None = None, **kwargs: Argument) -> Explanation:
        ordered = self._ordered(args, kwargs)
        if launch is not None and launch != self.launch:
            other = self.device.launcher(self.kernel, launch)
            if not isinstance(other, SourceLauncher):
                raise TypeError("explain needs a source launcher")
            return other.explanation(ordered, other.plan(ordered))
        return self.explanation(ordered, self.plan(ordered))

    def explanation(self, args: Sequence[Argument], variant: Variant[Spec]) -> Explanation:
        ir = self._plans[(variant.decisions.settings.key, *map(_kind, args))].ir
        geometry, cap = self.geometry(variant, args)
        extents = tensor_extents(ir, args, self.extent)
        pair = next(
            ((a, b) for i, (a, left) in enumerate(extents) for b, right in extents[i + 1 :] if left.conflicts(right)),
            None,
        )
        reason = f"overlapping parameters {pair[0]} and {pair[1]}" if pair else "arguments do not overlap"
        decisions = variant.decisions
        found = roles(
            self.path,
            self.kernel.name,
            consts={
                name: _consts(arg)
                for name, arg in zip(self.parameters, args, strict=True)
                if is_dataclass(arg) and not isinstance(arg, type)
            },
            dtypes=_dtypes(self.parameters, args),
            structs=_structs(self.parameters, args),
            target=self.device.target,
        )
        return Explanation(
            ir.name,
            self.device.target.arch,
            tuple(map(space_text, ir.space)),
            {
                "launch": str(self.launch if self.launch is not None else space.geometry(ir, args)),
                "grid": list(geometry[:3]),
                "threads": geometry[3],
                "inference": "explicit launch" if self.launch is not None else "inferred from kernel space",
            },
            tuple(decisions.rows()),
            f"{'restrict' if decisions.restrict.value else 'plain'}: {reason}; {decisions.restrict.why}",
            tuple(
                (
                    name,
                    size,
                    qualifier,
                    f"inline={decisions.inline.value}, limit={decisions.inline_limit.value}",
                )
                for name, size, qualifier in variant.unit.helpers
            ),
            tuple(
                (name, tuple(sorted(role.value for role in function_roles)))
                for name, function_roles in sorted(found.items())
            ),
            cap,
        )

    @abstractmethod
    def extent(self, argument: Argument, write: bool) -> Extent:
        """The storage occupied by one tensor argument on this backend."""

    @abstractmethod
    def geometry(
        self, variant: Variant[Spec], args: Sequence[Argument]
    ) -> tuple[tuple[int, int, int, int], str | None]:
        """Evaluate this call's geometry and any cooperative grid cap."""

    @abstractmethod
    def specialize(self, ir: IrKernel, decisions: Decisions) -> Variant[Spec]:
        """Build one pointer variant of an already captured signature."""


class CPULauncher[**P](SourceLauncher[P, CPUProgram, "CPUDevice"]):
    """A kernel with a launch geometry. The first call of a signature compiles it; later calls find it in a dict."""

    extent = staticmethod(cpu_extent)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self.device.dispatch(self.encode(self.ordered(*args, **kwargs)))

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> CPUEvent:
        return self.device.submit(self.encode(self.ordered(*args, **kwargs)))

    def bind(self, *args: P.args, **kwargs: P.kwargs) -> CPUBound[P]:
        return CPUBound(self, self.encode(self.ordered(*args, **kwargs)))

    def program(self, *args: Argument) -> CPUProgram:
        """The compiled program for these arguments, compiled now if this signature is new. Safe from any thread."""
        return self.plan(args).program

    def encode(self, args: tuple[Argument, ...]) -> Call:
        """Marshal once: the pointers, shapes, numbers and struct addresses the C function takes, plus the geometry."""
        variant = self.plan(args)
        program = variant.program
        geometry, _ = self.geometry(variant, args)
        natives, live = program.marshal([_value(arg) for arg in args])
        return Call(self, args, program, (*natives, *geometry), live)

    def geometry(
        self, variant: Variant[CPUProgram], args: Sequence[Argument]
    ) -> tuple[tuple[int, int, int, int], str | None]:
        launch = self.launch if self.launch is not None else space.geometry(variant.program.kernel, args)
        return dimensions(variant.program.kernel, launch, variant.decisions), None

    def specialize(self, ir: IrKernel, decisions: Decisions) -> Variant[CPUProgram]:
        program = self.device.program(ir, decisions)
        return Variant(program, decisions, program.unit)


class CPUBound[**P](Bound[P]):
    """A launch whose arguments are encoded. A call appends one existing object to a deque and allocates nothing."""

    def __init__(self, launcher: CPULauncher[P], call: Call) -> None:
        self._launcher, self._call = launcher, call

    def __call__(self) -> None:
        self._launcher.device.dispatch(self._call)

    def submit(self) -> CPUEvent:
        return self._launcher.device.submit(self._call)

    def rebind(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Replace the encoded arguments. Launches already queued keep the old ones, as a driver copies parameters."""
        self._call = self._launcher.encode(self._launcher.ordered(*args, **kwargs))


def tensor_extents(
    kernel: IrKernel,
    args: Sequence[Argument],
    extent: Callable[[Argument, bool], Extent] = cpu_extent,
) -> list[tuple[str, Extent]]:
    """Named extents, including nested structs; CUDA tensors expose a pointer and byte size."""
    found: list[tuple[str, Extent]] = []

    def visit(path: tuple[str, ...], kind: TensorType | StructType, argument: Argument) -> None:
        if isinstance(kind, StructType):
            for name, child in kind.fields:
                if isinstance(child, TensorType | StructType):
                    visit((*path, name), child, getattr(argument, name))
        else:
            found.append((".".join(path), extent(argument, path in kernel.written)))

    for (name, kind), argument in zip(kernel.params, args, strict=True):
        if isinstance(kind, TensorType | StructType):
            visit((name,), kind, argument)
    return found


def accesses(call: Call) -> list[Extent]:
    """Every tensor a call touches, including the tensors inside struct arguments. A written tensor is read as well."""
    return [extent for _, extent in tensor_extents(call.program.kernel, call.args)]


def dependencies(calls: Sequence[Call]) -> tuple[frozenset[int], ...]:
    """For each call, the earlier calls it must wait for, minus those another predecessor already implies."""
    touched = [accesses(call) for call in calls]
    reach: list[int] = []  # bit j set: call j is a direct or indirect predecessor
    direct: list[frozenset[int]] = []
    for i, mine in enumerate(touched):
        before = {j for j in range(i) if any(a.conflicts(b) for a in mine for b in touched[j])}
        implied = 0
        for j in before:
            implied |= reach[j]
        direct.append(frozenset(j for j in before if not implied >> j & 1))
        reach.append(implied | sum(1 << j for j in before))
    return tuple(direct)


@dataclass(frozen=True, slots=True)
class Dag:
    """The recorded calls contracted to chains: a call whose only predecessor has no other successor joins that
    predecessor's chain, so a chain runs on one thread with no counters, and counters guard only the joins."""

    chains: tuple[tuple[int, ...], ...]
    before: tuple[frozenset[int], ...]  # per chain, the chains it waits for
    after: tuple[tuple[int, ...], ...]  # per chain, the chains it releases
    starts: tuple[int, ...]  # chains that wait for nothing

    @classmethod
    def of(cls, predecessors: Sequence[frozenset[int]]) -> Self:
        fanout = Counter(parent for before in predecessors for parent in before)
        chains: list[list[int]] = []
        chain_of: dict[int, int] = {}
        for call, before in enumerate(predecessors):
            parent = next(iter(before)) if len(before) == 1 else None
            if parent is not None and fanout[parent] == 1:
                chain_of[call] = chain_of[parent]
                chains[chain_of[call]].append(call)
            else:
                chain_of[call] = len(chains)
                chains.append([call])
        waits_for = tuple(frozenset(chain_of[parent] for parent in predecessors[chain[0]]) for chain in chains)
        releases = tuple(
            tuple(k for k, waits in enumerate(waits_for) if chain in waits) for chain in range(len(chains))
        )
        starts = tuple(k for k, waits in enumerate(waits_for) if not waits)
        return cls(tuple(map(tuple, chains)), waits_for, releases, starts)


class GraphRun(Task):
    """One replay. The counters live here, so replays of one graph may overlap on different lanes.

    A chain whose last dependency just finished runs inline on that thread and its siblings go to the pool, so a chain
    of any length is one pool task. `_pending` counts the chains plus one hold for `run`, so whichever side finishes
    last, `run` returning True or the final chain, completes the replay exactly once. All state is made on the worker.
    """

    __slots__ = ("_calls", "_counters", "_dag", "_error", "_lane", "_pending", "_session")

    def __init__(self, dag: Dag, calls: Sequence[Call]) -> None:
        self._dag, self._calls = dag, calls

    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        self._lane, self._error, self._session = lane, None, session
        self._counters = [Countdown(len(before)) for before in self._dag.before]
        self._pending = Countdown(len(self._dag.chains) + 1)
        starts = self._dag.starts
        for start in starts[1:]:
            lane.pool.submit(partial(self._execute, start))
        if starts:
            self._execute(starts[0])
        return self._pending.decrement() == 0

    def _execute(self, chain: int | None) -> None:
        session = self._session
        while chain is not None:
            for index in self._dag.chains[chain]:
                call = self._calls[index]
                try:
                    if session is None:
                        call.program.function(*call.arguments)
                    else:
                        session.cpu_run(call, self._lane)
                except Exception as error:  # the top of a pool thread's stack: the replay reports it via its lane
                    self._error = self._error or error
                    log.error("graph call %d failed", index, exc_info=error)
            ready = [after for after in self._dag.after[chain] if self._counters[after].decrement() == 0]
            if self._pending.decrement() == 0:
                if self._error is not None:
                    self._lane.fail(self._error)
                self._lane.resume()
                return
            for other in ready[1:]:
                self._lane.pool.submit(partial(self._execute, other))
            chain = ready[0] if ready else None


@dataclass(frozen=True, slots=True)
class Binding:
    """The arguments a graph was last replayed with, and the calls encoded for them."""

    arguments: tuple[Argument, ...]
    calls: Sequence[Call]


class CPUGraph[**P](Graph[P]):
    """The launches of one function, recorded once. A replay runs the dependency DAG on the pool, no Python per node.

    Nodes depend on each other through the bytes they touch, as recorded. A replay with other tensors keeps that
    structure, so the new tensors must alias each other the way the recorded ones did.
    """

    def __init__(self, device: CPUDevice, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> None:
        self.device, self._fn = device, fn
        self._signature = inspect.signature(fn)
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        self._recorded = tuple(bound.arguments.values())
        self.calls: tuple[Call, ...] = ()
        if device.settings.run.graphs:
            self.calls = self._record(*args, **kwargs)
        else:
            log.info("run.graphs=False: %s replays as plain launches", getattr(fn, "__name__", fn))
        self.predecessors = dependencies(self.calls)
        self.dag = Dag.of(self.predecessors)
        self._binding = Binding(self._recorded, self.calls)

    @property
    def roots(self) -> tuple[int, ...]:
        """The recorded calls that wait for nothing."""
        return tuple(i for i, before in enumerate(self.predecessors) if not before)

    @property
    def edges(self) -> frozenset[tuple[int, int]]:
        """Dependencies as (earlier call, later call) pairs of the recorded order."""
        return frozenset((before, i) for i, deps in enumerate(self.predecessors) for before in deps)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if self.device.settings.run.graphs:
            self.device.dispatch(GraphRun(self.dag, self._calls_for(*args, **kwargs)))
        else:
            self._fn(*args, **kwargs)

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> CPUEvent:
        if _recording.get() is not None:
            raise RuntimeError("a graph cannot be replayed while another graph records")
        lane = self.device.launch_stream()
        self(*args, **kwargs)
        return lane.record()

    def _record(self, *args: P.args, **kwargs: P.kwargs) -> tuple[Call, ...]:
        if _recording.get() is not None:
            raise RuntimeError("graphs do not nest: one is already recording")
        token = _recording.set(calls := [])
        try:
            self._fn(*args, **kwargs)
        finally:
            _recording.reset(token)
        return tuple(calls)

    def _calls_for(self, *args: P.args, **kwargs: P.kwargs) -> Sequence[Call]:
        """The calls for these arguments: the previous replay's, or re-encoded with the tensors swapped."""
        current = self._binding
        if not kwargs and _same(
            cast(tuple[Argument, ...], args), current.arguments
        ):  # P.args is opaque to the checkers
            return current.calls
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        given = tuple(bound.arguments.values())
        if _same(given, current.arguments):
            return current.calls
        swaps = {
            id(old): new for old, new in zip(self._recorded, given, strict=True) if new is not old and _swappable(old)
        }
        calls = tuple(self._swapped(call, swaps) for call in self.calls)
        self._binding = Binding(given, calls)
        return calls

    @staticmethod
    def _swapped(call: Call, swaps: Mapping[int, Argument]) -> Call:
        if not any(id(arg) in swaps for arg in call.args):
            return call
        return call.launcher.encode(tuple(swaps.get(id(arg), arg) for arg in call.args))


def _same(left: Sequence[Argument], right: Sequence[Argument]) -> bool:
    """Whether two argument lists hold the very same objects."""
    return len(left) == len(right) and all(map(operator.is_, left, right, strict=True))


def _swappable(argument: Argument) -> bool:
    """Tensors and structs are replaced by identity. A number cannot be: `3 is 3` would swap unrelated arguments."""
    return isinstance(argument, CPUTensor | np.ndarray) or (is_dataclass(argument) and not isinstance(argument, type))


class CPUDevice(Device):
    """The host CPU: a worker pool, the streams on it, and the cache of compiled programs."""

    def __init__(self, target: CPU, settings: Settings | None = None) -> None:
        self.target, self.cpu = target, target
        self._settings_override = settings
        self.settings.validate(target)
        self.compile_options = CompileOptions.resolve(
            target,
            lineinfo=True if settings and settings.debug.lineinfo else None,
            debug=True if settings and settings.debug.debug else None,
        )
        # Spinning delays a host with a GIL; only a free-threaded build keeps workers after their lanes run dry.
        self.pool = WorkerPool(
            self.settings.run.workers or os.cpu_count() or 1, 0 if sys._is_gil_enabled() else SPIN_NS
        )
        self._default_stream = CPUStream(self)
        self._streams: weakref.WeakSet[CPUStream] = weakref.WeakSet()
        self._streams_lock = threading.Lock()
        self._launchers: dict[tuple[Hashable, Launch | None, str], CPULauncher] = {}
        self._programs: dict[Hashable, CPUProgram] = {}
        self._builds: dict[Hashable, threading.Lock] = {}

    @property
    def default_stream(self) -> CPUStream:
        return self._default_stream

    def stream(self) -> CPUStream:
        stream = CPUStream(self)
        with self._streams_lock:
            self._streams.add(stream)
        return stream

    def launch_stream(self) -> CPUStream:
        """The stream launches go to: the current one of this thread or task, else the default stream."""
        match current_stream.get():
            case None:
                return self.default_stream
            case CPUStream() as stream:
                return stream
        raise TypeError("the current stream belongs to another backend")

    def launcher[**P, R, S, **Q](self, kernel: Kernel[P, R, S, Q], launch: Launch | None = None) -> CPULauncher[P]:
        snapshot = self._settings_override or current_settings()
        key = (kernel, launch, snapshot.key)
        try:
            return self._launchers[key]
        except KeyError:
            return self._launchers.setdefault(key, CPULauncher(self, kernel, launch, snapshot))

    @overload
    def empty[T: Scalar | bool](self, dtype: type[T], length: int, /) -> CPUTensor[T, int]: ...
    @overload
    def empty[T: Scalar | bool](self, dtype: type[T], *shape: int) -> CPUTensor[T, int, int]: ...
    @overload
    def empty(self, dtype: Selector, length: int, /) -> CPUTensor[FloatingPoint, int]: ...
    @overload
    def empty(self, dtype: Selector, *shape: int) -> CPUTensor[FloatingPoint, int, int]: ...
    def empty[T: Scalar | bool](self, dtype: type[T] | Selector, *shape: int) -> Empty[T]:
        """Uninitialized memory. The contract types every result `Tensor[T, int, int]`, which a `Vector[f32]` kernel
        parameter rejects, so a single length returns a vector type. A selector such as `T.fp.min` resolves for the
        target of this device, which the contract's `type[T]` does not allow yet."""
        resolved = dtype.resolve(self.target) if isinstance(dtype, Selector) else dtype
        return CPUTensor(np.empty(shape, numpy_dtype(resolved)))

    def to_device(self, array: np.ndarray) -> CPUTensor:
        """A copy of `array`: host memory is the device, but a tensor must not alias what the caller keeps changing."""
        return CPUTensor(np.array(array, order="C"))

    def graph[**P](self, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> CPUGraph[P]:
        return CPUGraph(self, fn, *args, **kwargs)

    async def synchronize(self) -> None:
        """Wait for everything queued so far on every stream of this device."""
        with self._streams_lock:
            streams = [self.default_stream, *self._streams]
        await asyncio.gather(*(stream.record() for stream in streams))

    def dispatch(self, task: Task) -> None:
        """Enqueue on the current stream, or, while a graph records, append the call to the graph."""
        if (recording := _recording.get()) is None:
            self.launch_stream().enqueue(task)
        elif isinstance(task, Call):
            recording.append(task)
        else:
            raise RuntimeError("only kernel launches can be recorded in a graph")

    def submit(self, task: Task) -> CPUEvent:
        if _recording.get() is not None:
            raise RuntimeError("a launch cannot be awaited while a graph records it")
        lane = self.launch_stream()
        lane.enqueue(task)
        return lane.record()

    def program(self, ir: IrKernel, decisions: Decisions) -> CPUProgram:
        """The program of a signature: found in a dict without a lock, built once behind the lock of its key."""
        snapshot = decisions.settings
        options = CompileOptions.resolve(self.target, settings=snapshot)
        unit = Emitter(ir, CPURenderer(self.cpu, decisions), lineinfo=options.lineinfo).emit()
        key = hashlib.sha256(f"{options}\0{unit.source}".encode()).hexdigest()
        try:
            return self._programs[key]
        except KeyError:
            pass
        with self._builds.setdefault(key, threading.Lock()):
            try:
                return self._programs[key]
            except KeyError:
                return self._programs.setdefault(key, CPUProgram(ir, unit, self.cpu, snapshot.cache.dir, options))

    def close(self) -> None:
        """Stop the pool once the queued tasks have run. For tests and short-lived devices."""
        self.pool.close()


class CPUBackend(Backend):
    """The `cpu` vendor. One device per target, so every thread that asks for the default device gets the same pool."""

    vendor = "cpu"

    def __init__(self) -> None:
        self._devices: dict[tuple[str, int | None], CPUDevice] = {}
        self._lock = threading.Lock()

    def device(self, target: Target) -> CPUDevice:
        if not isinstance(target, CPU):
            raise TypeError(f"the cpu backend builds devices for CPU targets, not {target.name}")
        with self._lock:
            key = target.arch, current_settings().run.workers
            if key not in self._devices:
                self._devices[key] = CPUDevice(target)
            return self._devices[key]


BACKEND = CPUBackend()
