"""A fixed pool of slots, each with its own device stream, lent to one task at a time."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

from ..runtime.api import Device, Stream

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Slot:
    """A lent buffer index and the stream used by its borrowing task."""

    index: int
    stream: Stream


class Slots:
    """A pool of `count` slots on the current device, for work that reuses buffers.

    `async with slots.acquire() as slot:` waits for a free slot. Its work goes to
    `slot.stream`. On exit the slot records its stream and returns to the pool when that event is done, so the device
    never reads a buffer that another task has already reused, and the exiting task never waits for the device.
    """

    def __init__(self, count: int) -> None:
        device = Device.current()
        self._free: asyncio.Queue[Slot] = asyncio.Queue()
        for index in range(count):
            self._free.put_nowait(Slot(index=index, stream=device.stream()))

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Slot]:
        slot = await self._free.get()
        try:
            yield slot
        finally:
            self._return(slot)

    def _return(self, slot: Slot) -> None:
        event = slot.stream.record()
        free = partial(self._free.put_nowait, slot)
        if event.subscribe(partial(_post, asyncio.get_running_loop(), free)):
            free()


def _post(loop: asyncio.AbstractEventLoop, function: Callable[[], None]) -> None:
    """Run `function` on `loop` from the thread that resolved an event. A loop that has closed has nobody to resume."""
    try:
        loop.call_soon_threadsafe(function)
    except RuntimeError:
        log.debug("a slot came back after its event loop closed")
