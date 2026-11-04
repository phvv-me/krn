"""The host side: devices, streams, events, launches and graphs. Python that stays out of the way of the device.

The rules this module is built around:

* Every enqueue is non-blocking and returns at once. Completion is an `Event`, and an `Event` is awaitable, so a device
  is one more thing an `asyncio` program waits for. A waiter thread resolves the future with
  `loop.call_soon_threadsafe`: it blocks in a blocking-sync event on CUDA or HIP, in a shared-event listener on Metal,
  and a finished task is the signal on the CPU pool. No host function is used on the completion path, because a host
  function is never called once the CUDA context has an error, and an `await` built on it would hang forever. On the
  free-threaded build the waiter runs Python in parallel with the event loop.
* The current stream is a `ContextVar`. Each asyncio task and each thread has its own, so concurrent tasks use
  separate streams without passing one around, and `async with device.stream():` scopes it to the task.
* A launch has a slow path (the first call: capture, compile, cache) and a fast path (every call after). The fast path
  is a prebound `Launcher` or `Bound`: arguments marshalled into one preallocated struct, one driver call, no kwargs
  parsing, no allocation. A graph removes even that driver call per kernel.

This module is the contract. A vendor implements `Backend`, `Device`, `Stream`, `Event`, `Launcher`, `Bound` and
`Graph` and registers the backend by vendor name; everything above (kernels, families, examples) is written against
these classes only. `Device.current()` finds the system with `detect()` and picks the backend of its vendor.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextvars import ContextVar, Token
from functools import cache, partial
from importlib import import_module
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Concatenate, Self, overload

from krn.compiler.debug import Described, Explanation
from krn.facts.targets import Target
from krn.frontend.geometry import Blocks, Items
from krn.frontend.memory import Tensor
from krn.frontend.numerics import FloatingPoint, Scalar, Selector
from krn.hw.system import System

from .settings import Settings, current_settings, retained

if TYPE_CHECKING:
    import numpy as np
    from _typeshed import DataclassInstance

    from krn.frontend.kernel import Kernel
    from krn.frontend.pipeline import Lane

type Launch = Items | Blocks
type HostArgument = Tensor | np.ndarray | np.generic | Scalar | int | float | DataclassInstance

log = logging.getLogger(__name__)


class Event(ABC):
    """A point in a stream's order. Awaiting it suspends the awaiting task only; `done()` never blocks.

    `subscribe` is the loop-neutral hook that awaiting is built on: its callback runs once, on the thread that resolves
    the event. A loop that is not asyncio bridges it with its own thread-safe re-entry, such as Trio's
    `TrioToken.run_sync_soon`.
    """

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def subscribe(self, callback: Callable[[], None]) -> bool:
        """Run once on the resolving thread; return True if already resolved without retaining the callback."""

    @property
    @abstractmethod
    def error(self) -> BaseException | None:
        """What the device reported for the work before this event, once it is done; None when that work succeeded."""

    def wait(self, timeout: float | None = None) -> None:
        """Block the host until completion; concrete device events implement this."""
        raise NotImplementedError

    def __await__(self) -> Generator[asyncio.Future[None]]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if self.subscribe(partial(self._wake, loop, future)):
            self._deliver(future)
        return future.__await__()

    def _wake(self, loop: asyncio.AbstractEventLoop, future: asyncio.Future[None]) -> None:
        try:
            loop.call_soon_threadsafe(self._deliver, future)
        except RuntimeError:  # the loop closed while its task waited, and nobody is left to resume
            log.debug("event completed after its event loop closed")

    def _deliver(self, future: asyncio.Future[None]) -> None:
        if future.done():  # the awaiting task was cancelled
            return
        if (error := self.error) is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    @abstractmethod
    def elapsed_since(self, earlier: Event) -> float:
        """Milliseconds between two events of one device, measured by the device clock."""


class Stream(Described, ABC):
    """An in-order queue of device work: a CUDA or HIP stream, a Metal command queue, a serial lane of the CPU pool."""

    device: Device

    @property
    def capturing(self) -> bool:
        """Is a graph being captured on this stream? Libraries must not plan, allocate or synchronize while it is."""
        return False

    @abstractmethod
    def record(self) -> Event: ...

    @abstractmethod
    def pending_launches(self) -> tuple[str, ...]:
        """The names of the launches queued on this stream that have not completed, oldest first."""

    @abstractmethod
    def wait(self, event: Event) -> None:
        """Make later work on this stream wait for an event of another stream, without blocking the host."""

    def __enter__(self) -> Self:
        """Make this the current stream of the running thread or task. Leaving restores the previous one."""
        _tokens.set((*_tokens.get(), current_stream.set(self)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        *outer, token = _tokens.get()
        current_stream.reset(token)
        _tokens.set(tuple(outer))

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)
        await self.record()


current_stream: ContextVar[Stream | None] = ContextVar("krn_current_stream", default=None)
_tokens: ContextVar[tuple[Token[Stream | None], ...]] = ContextVar("krn_stream_tokens", default=())


class Launcher[**P](ABC):
    """A kernel bound to a launch geometry. Calling it enqueues on the current stream and returns at once."""

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...

    @abstractmethod
    def explain(self, *args: P.args, **kwargs: P.kwargs) -> Explanation:
        """Describe capture roles, geometry and backend choices without launching."""

    @abstractmethod
    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Event:
        """Enqueue and return the completion event: `await launcher.submit(x, y)`."""

    @abstractmethod
    def bind(self, *args: P.args, **kwargs: P.kwargs) -> Bound[P]:
        """Marshal the arguments once. The returned launch is the fast path for loops that repeat a launch."""


class Bound[**P](ABC):
    """A launch whose arguments are already encoded. Calling it costs one driver call and no allocation."""

    @abstractmethod
    def __call__(self) -> None: ...

    @abstractmethod
    def submit(self) -> Event: ...

    @abstractmethod
    def rebind(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Overwrite the encoded pointers and scalars in place, for example with the next batch."""


class ConfiguredBound[S, **P](Bound[P]):
    """A `Bound` of a configured kernel with its configuration already part of the encoded arguments."""

    def __init__(self, inner: Bound[Concatenate[S, P]], config: S) -> None:
        self.inner, self.config = inner, config

    def __call__(self) -> None:
        self.inner()

    def submit(self) -> Event:
        return self.inner.submit()

    def rebind(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self.inner.rebind(self.config, *args, **kwargs)


class Configured[S, **P](Launcher[P]):
    """The launcher of a configured kernel: the configuration chosen in the brackets is passed as the first argument."""

    def __init__(self, inner: Launcher[Concatenate[S, P]], config: S) -> None:
        self.inner, self.config = inner, config

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self.inner(self.config, *args, **kwargs)

    def explain(self, *args: P.args, **kwargs: P.kwargs) -> Explanation:
        return self.inner.explain(self.config, *args, **kwargs)

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Event:
        return self.inner.submit(self.config, *args, **kwargs)

    def bind(self, *args: P.args, **kwargs: P.kwargs) -> Bound[P]:
        return ConfiguredBound(self.inner.bind(self.config, *args, **kwargs), self.config)


class Graph[**P](ABC):
    """Recorded launches that replay with one call per graph, typed by the function they were recorded from.

    CUDA uses stream capture, Metal records calls to re-encode, and CPU replays a dependency DAG on its worker pool.
    CUDA and Metal require the recorded arguments; rebinding raises TargetError. run.graphs=False explicitly selects
    plain launches on every runtime backend.
    """

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Replay with new arguments. The checker verifies them against the recorded function.

        CPU graphs swap tensors and structs by identity. CUDA and Metal require the recorded arguments; record a new
        graph for new arguments. Mutable tensor contents may change between replays on every backend.
        """

    @abstractmethod
    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Event: ...


class Device(ABC):
    """One GPU, one Apple GPU or the host CPU, with the target facts the compiler needs."""

    target: Target
    _settings_override: Settings | None = None

    @property
    def settings(self) -> Settings:
        """An explicit device snapshot, or the current launch snapshot."""
        return self._settings_override or current_settings()

    @settings.setter
    def settings(self, value: Settings) -> None:
        self._settings_override = value

    @classmethod
    def current(cls) -> Device:
        """The device kernels launch on. The first call detects the system and builds its backend's device."""
        return default_device()

    @property
    @abstractmethod
    def default_stream(self) -> Stream:
        """Where launches go when the thread or task has no current stream."""

    @abstractmethod
    def stream(self) -> Stream: ...

    @abstractmethod
    def to_device(self, array: np.ndarray) -> Tensor:
        """A device copy of a host array, with the element type of the array. The copy is enqueued, not awaited."""

    def copy[T: Scalar | bool](self, destination: Tensor[T, int], source: Tensor[T, int], *, after: Event) -> Event:
        """Enqueue a copy on a copy engine once `after` has resolved, and return the event that says it is done."""
        raise NotImplementedError(f"{type(self).__name__} has no copy engine yet")

    def lane(self) -> Lane:
        """A stream whose tasks are Python callables, for the host stages of a pipeline."""
        raise NotImplementedError(f"{type(self).__name__} has no host lanes yet")

    @abstractmethod
    def launcher[**P, R, S, **Q](self, kernel: Kernel[P, R, S, Q], launch: Launch | None = None) -> Launcher[P]:
        """Compile on first use and cache by specialization. With no launch, infer geometry on every call."""

    @overload
    def empty[T: Scalar | bool](self, dtype: type[T], length: int, /) -> Tensor[T, int]: ...
    @overload
    def empty[T: Scalar | bool](self, dtype: type[T], *shape: int) -> Tensor[T, int, int]: ...
    @overload
    def empty(self, dtype: Selector, length: int, /) -> Tensor[FloatingPoint, int]: ...
    @overload
    def empty(self, dtype: Selector, *shape: int) -> Tensor[FloatingPoint, int, int]: ...
    @abstractmethod
    def empty[T: Scalar | bool](
        self, dtype: type[T] | Selector, *shape: int
    ) -> Tensor[T, int] | Tensor[T, int, int] | Tensor[FloatingPoint, int] | Tensor[FloatingPoint, int, int]:
        """Uninitialized memory. One length gives a vector, two a matrix; a selector such as `T.fp.min` is resolved."""

    @abstractmethod
    def graph[**P](self, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> Graph[P]:
        """Record every launch `fn(*args, **kwargs)` makes, once. Replaying the graph needs no Python per launch."""

    @abstractmethod
    async def synchronize(self) -> None:
        """Wait for everything queued so far on every stream of the device, not only the calling task's."""


class Backend(ABC):
    """What a vendor implementation provides. Its module exposes one instance as `BACKEND`; nothing registers it."""

    vendor: ClassVar[str]

    @abstractmethod
    def device(self, target: Target) -> Device: ...


MODULES = {
    "cpu": "krn.runtime.runtime",
    "nvidia": "krn.runtime.cuda_backend",
    "amd": "krn.runtime.hip",
    "apple": "krn.runtime.metal",
}


class BackendUnavailable(RuntimeError):
    """The vendor was detected but its backend package is not installed."""


@retained
def default_device() -> Device:
    """The device of the detected target (or `KRN_TARGET`), built once per target and shared by every thread."""
    settings = current_settings()
    return device_of(Target.of(settings.run.target) if settings.run.target else System.detect().target)


@cache
def device_of(target: Target) -> Device:
    """The one device of `target` for this process.

    The backend is found, never registered: the vendor of the target names a module, which is imported on first use,
    so a program that never touches an AMD GPU never loads anything AMD.
    """
    try:
        backend: Backend = import_module(MODULES[target.vendor]).BACKEND
    except (ImportError, AttributeError) as error:
        raise BackendUnavailable(
            f"{target.arch} was detected but its backend {MODULES[target.vendor]!r} has no device yet: {error}. "
            "Set KRN_TARGET to a CPU target such as avx512 to use the C backend."
        ) from error
    return backend.device(target)
