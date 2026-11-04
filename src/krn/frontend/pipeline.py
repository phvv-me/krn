"""The pieces that turn streams into a pipeline: host lanes, tickets, bounded rings, a coalescing egress, transfers.

An LLM serving engine is one chain from a request to a token: tokenize, schedule, copy the plan to the device, replay
the step, copy the sampled tokens back, detokenize, stream out. The study of vLLM, SGLang, TensorRT-LLM, llama.cpp and
MLX (notes/round5/serving) found one structure under all five, and these types are that structure:

* a stage's output event is the next stage's wait event, and every stage is a `Lane`, a stream whose work may be
  Python, so host stages and device stages order against each other with the same `Event`;
* every in-flight buffer is a `Channel` of `Slot`s, a ring with two directions of event (`ready` for the consumer,
  `free` for the producer), which is also the credit-based backpressure;
* the sampled token stays on the device and the host learns it one step late, so a stage is planned while the previous
  one runs, and depth 2 is the whole benefit (depth 3 only adds stale tokens);
* a copy is elided when memory is shared, never the ordering: `route(...)` says ELIDE, COPY, STAGE or MAP from where
  the data is, where it must be, and what the machine's `Memory` allows, so the pipeline code is identical on a
  discrete GPU, GH200 and GB10, Apple silicon, an MI300A and a CPU alone.

Completion is signalled by a waiter thread that blocks in a blocking-sync event (CUDA, HIP) or a shared-event
listener (Metal), not by a host function: a host function never runs once the CUDA context has an error, so an
`await` built on it can hang forever. Only `Memory` and `route` are implemented here; the rest is the contract, and a
runnable toy engine that obeys it (15 tests, both interpreter builds) is in notes/round5/serving/proto.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto

from krn.facts.targets import CPU, Apple, Target
from krn.runtime.api import Device, Event, Stream

from .memory import Tensor
from .numerics import Scalar


class Ticket[R = None](Event):
    """An `Event` that carries a value or an error. It resolves exactly once, on the thread that reached it."""

    @property
    @abstractmethod
    def error(self) -> BaseException | None: ...

    @abstractmethod
    def result(self) -> R:
        """The value, or raises the error. Call it after the event is done."""

    @abstractmethod
    def on_done(self, callback: Callable[[], None]) -> None:
        """Run `callback` once, on any thread, when the event resolves. It must be short and must not block."""


@abstractmethod
def join(events: Sequence[Event]) -> Ticket[None]:
    """An event that resolves when all of `events` have, carrying the first error."""


class Lane(Stream):
    """A stream whose tasks are Python callables, ordered with device work by events.

    Each task runs in a copy of the submitter's context, so the current stream and any other `ContextVar` follow the
    task and not the worker thread (on the free-threaded build a plain thread inherits the context, on the GIL build it
    does not). A lane waits on a failed event by becoming poisoned: later tasks resolve with that error without running.
    """

    @abstractmethod
    def run[R](self, fn: Callable[[], R]) -> Ticket[R]:
        """Enqueue `fn` in stream order."""

    @abstractmethod
    def run_after[R](self, events: Sequence[Event], fn: Callable[[], R]) -> Ticket[R]:
        """Dataflow order: run `fn` when `events` have resolved, without parking the lane for tasks queued later."""

    @abstractmethod
    def wait(self, event: Event, *more: Event) -> None:
        """Park the lane until the events resolve (stream order: everything queued later waits too)."""

    @abstractmethod
    def fail(self, error: BaseException) -> None:
        """Poison the lane, as a sticky device error would."""


class Slot[T]:
    """One slot of a ring: the value, an event for the consumer and an event for the producer."""

    value: T
    ready: Event  # the producer's data is usable once this resolves
    free: Event  # the slot may be reused once this resolves

    @abstractmethod
    def publish(self, ready: Event) -> None:
        """Hand the slot to the consumer. Publishing twice without a release raises."""

    @abstractmethod
    def release(self, done: Event) -> None:
        """Return the slot: `free` resolves when `done` does, with its error."""


class Channel[T](ABC):
    """A ring of slots between two stages. Its length is the pipeline depth and the credit for backpressure."""

    @abstractmethod
    def peek(self) -> Slot[T]:
        """The next slot in ring order, without taking it."""

    @abstractmethod
    def acquire(self) -> Slot[T]: ...


class Latest[T](ABC):
    """The egress to one consumer: any thread puts, one asyncio task gets, and items merge when the consumer is behind.

    Memory per request stays bounded by one merged item, and the producing lane never blocks on a slow client. A
    doorbell shared by all `Latest`s wakes the event loop once per burst, not once per token.
    """

    @abstractmethod
    def put(self, item: T) -> None: ...

    @abstractmethod
    async def get(self) -> T: ...


class Place(StrEnum):
    HOST = auto()  # pageable host memory
    PINNED = auto()  # page-locked host memory the device can copy from at full speed
    DEVICE = auto()


class Move(StrEnum):
    ELIDE = auto()  # no copy: the same bytes are visible to both sides, only the ordering remains
    COPY = auto()  # a copy engine or a blit between pinned and device memory
    STAGE = auto()  # copy through a pinned staging buffer first
    MAP = auto()  # the device reads host memory in place (small data, mapped or unified addressing)


@dataclass(frozen=True, slots=True)
class Memory:
    """What the machine's memory allows. It comes from the facts of the machine, never from the program.

    shared: host and device see the same memory (Apple silicon, MI300A, GB10, GH200 with coherent access, a CPU alone).
    can_map: the device can read pinned host memory in place.
    """

    shared: bool
    can_map: bool = False

    @classmethod
    def of(cls, target: Target) -> Memory:
        """The default for a target: shared on Apple silicon and CPUs, discrete elsewhere unless facts say otherwise."""
        return cls(shared=isinstance(target, Apple | CPU), can_map=True)


def route(source: Place, destination: Place, memory: Memory, *, small: bool = False) -> Move:
    """How data moves between places on this machine. Elide the copy, never the ordering."""
    if memory.shared or source is destination:
        return Move.ELIDE
    if small and memory.can_map and PINNED in (source, destination):
        return Move.MAP
    return Move.COPY if PINNED in (source, destination) else Move.STAGE


PINNED = Place.PINNED


@dataclass(frozen=True, slots=True)
class Moved[T: Scalar]:
    """The result of a transfer: the view the consumer must read and the event that says it is ready."""

    view: Tensor[T, int]
    ready: Event


@dataclass(frozen=True, slots=True)
class Transfer:
    """Moves a tensor between places by the route the machine's memory allows; elided routes touch no engine.

    The returned `Moved` is what the next stage orders on: the source itself and `after` when the copy is elided, the
    destination and the copy's own event otherwise. The pipeline code is the same on every machine.
    """

    device: Device
    memory: Memory

    def __call__[T: Scalar](
        self,
        destination: Tensor[T, int],
        source: Tensor[T, int],
        after: Event,
        *,
        source_place: Place,
        place: Place,
    ) -> Moved[T]:
        if route(source_place, place, self.memory, small=True) is Move.ELIDE:
            return Moved(source, after)
        return Moved(destination, self.device.copy(destination, source, after=after))
