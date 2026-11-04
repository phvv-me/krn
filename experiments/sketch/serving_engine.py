"""An LLM serving engine as one pipeline of lanes and rings, from a request's text to its streamed tokens

One rule makes the chain one object: a stage's output event is the next stage's wait event. Every stage is a `Lane`
(a stream whose tasks may be Python) and every in-flight buffer is a `Channel` of `Slot`s with a `ready` event
toward the consumer and a `free` event toward the producer, so host and device stages order against each other with
the same `Event` and the whole chain is one dependency graph: tokenizer lane, scheduler lane, plan ring, compute (a
replayed graph), token ring, detokenizer lane, and a `Latest` per request to the streaming client. Copies are elided
on shared memory but events never are, so the code is identical on discrete, unified and CPU-only machines.

Level: 5
Topics: serving, lanes, rings, events, pipelines
Needs: 05_systems/08_async_streams, 05_systems/09_graphs
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial

from krn import T, Vector, host


@dataclass(frozen=True, slots=True)
class Delta:
    """What one request's client receives: a piece of text, or the end, or the engine's error."""

    text: str = ""
    finished: bool = False
    error: BaseException | None = None


@dataclass(slots=True)
class Request:
    rid: int
    tokens: list[int]
    max_new: int
    outbox: host.Latest[Delta]
    emitted: int = 0
    finished: bool = False


@dataclass(frozen=True, slots=True)
class Plan:
    step: int
    bucket: int
    requests: tuple[Request, ...]


@dataclass(slots=True)
class Buffers:
    """One ring slot's worth of memory: host metadata, its device twin, and the sampled tokens both ways."""

    meta_host: Vector[T.i32]
    meta_device: Vector[T.i32]
    sampled_device: Vector[T.i32]
    sampled_host: Vector[T.i32]


class Scheduler(ABC):
    """Pure Python state owned by one lane: waiting and running requests, device slots, the step number."""

    @abstractmethod
    def admit(self, request: Request) -> None: ...

    @abstractmethod
    def stop(self, request: Request) -> None: ...

    @abstractmethod
    def plan(self, step: int, meta: Vector[T.i32]) -> Plan | None:
        """Fill the slot's metadata for the next batch, padded to a recorded bucket, or None when idle."""

    @abstractmethod
    def step_done(self, step: int, finished: list[Request]) -> None: ...


class Model(ABC):
    """The forward pass and the sampler, recorded once per (bucket, ring slot) as a graph."""

    @abstractmethod
    def replay(self, bucket: int, slot: int) -> None: ...


@dataclass(slots=True)
class Engine:
    device: host.Device
    scheduler: Scheduler
    model: Model
    ring: host.Channel[Buffers]
    transfer: host.Transfer
    step: int = 0
    sched_lane: host.Lane = field(init=False)
    detok_lane: host.Lane = field(init=False)
    compute: host.Stream = field(init=False)
    armed: bool = False
    failed: BaseException | None = None

    def __post_init__(self) -> None:
        self.sched_lane, self.detok_lane, self.compute = self.device.lane(), self.device.lane(), self.device.stream()

    async def generate(self, request: Request) -> AsyncIterator[str]:
        """The client's view: admit the request, then stream deltas until it finishes or fails."""
        self.sched_lane.run(partial(self.scheduler.admit, request))
        self.kick()
        try:
            while not (delta := await request.outbox.get()).finished:
                if delta.error:
                    raise RuntimeError("the engine failed") from delta.error
                yield delta.text
            yield delta.text
        except asyncio.CancelledError, GeneratorExit:
            self.sched_lane.run(partial(self.scheduler.stop, request))  # idempotent; in-flight tokens are dropped
            raise

    def kick(self) -> None:
        """Arm one iteration, gated by the `free` event of the slot it will reuse: the credit that bounds the depth."""
        if not self.armed and not self.failed:
            self.armed = True
            self.sched_lane.run_after([self.ring.peek().free], self.iterate)

    def iterate(self) -> None:
        """On the scheduler lane: plan step n+1 while step n runs, launch it, and hand its tokens to the detokenizer."""
        self.armed = False
        slot = self.ring.peek()
        plan = self.scheduler.plan(self.step, slot.value.meta_host)  # pure Python, single owner of the scheduler state
        if plan is None:
            return  # idle: admit or step_done kicks again
        self.ring.acquire()
        self.step += 1
        buffers = slot.value
        meta = self.transfer(
            buffers.meta_device, buffers.meta_host, slot.free, source_place=host.Place.PINNED, place=host.Place.DEVICE
        )
        with self.compute:  # the current stream for every launch below
            self.compute.wait(meta.ready)  # output event = the next stage's wait event
            self.model.replay(plan.bucket, 0)
            done = self.compute.record()
        sampled = self.transfer(
            buffers.sampled_host, buffers.sampled_device, done, source_place=host.Place.DEVICE, place=host.Place.PINNED
        )
        slot.publish(sampled.ready)
        self.detok_lane.wait(sampled.ready)  # parks, never spins
        ticket = self.detok_lane.run(partial(self.consume, plan, slot))
        ticket.on_done(partial(self.fail_on, ticket))
        self.kick()  # step n+1 is planned while n runs

    def consume(self, plan: Plan, slot: host.Slot[Buffers]) -> None:
        """On the detokenizer lane: turn tokens into deltas, stop requests, report first, release the slot second."""
        finished = []
        for request in plan.requests:
            if not request.finished:
                request.emitted += 1
                request.finished = request.emitted >= request.max_new
                request.outbox.put(Delta(finished=request.finished))
                finished += [request] if request.finished else []
        self.sched_lane.run(partial(self.scheduler.step_done, plan.step, finished))
        slot.release(slot.ready)

    def fail_on(self, ticket: host.Ticket[None]) -> None:
        """Engine-scoped failure: poison, then fan the error out to every open request so no client hangs."""
        if error := ticket.error:
            self.failed = self.failed or error
            self.sched_lane.fail(error)
