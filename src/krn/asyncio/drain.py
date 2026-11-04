"""Waiting for the device work queued on streams, bounded, and never abandoned half way."""

import asyncio
from collections.abc import Iterable

from ..runtime.api import Stream
from ..runtime.settings import current_settings


async def drain(streams: Iterable[Stream], timeout: float | None = None) -> None:
    """Wait until the work queued on `streams` so far has completed, or raise `TimeoutError` after `timeout` seconds.

    With no timeout the bound is `run.drain_seconds`. The error names each stream still busy and the launches pending on
    it. Cancelling the caller does not recall the device work: the wait goes on until the work completes, and only then
    is the cancellation raised, so the caller never resumes while a kernel still reads its tensors.
    """
    streams = tuple(streams)
    limit = current_settings().run.drain_seconds if timeout is None else timeout
    loop = asyncio.get_running_loop()
    waiting = asyncio.ensure_future(asyncio.gather(*(stream.record() for stream in streams)))
    deadline, cancelled = loop.time() + limit, False
    while not waiting.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            waiting.cancel()
            raise TimeoutError(_busy(streams, limit))
        try:
            await asyncio.wait((waiting,), timeout=remaining)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    waiting.result()


def _busy(streams: tuple[Stream, ...], limit: float) -> str:
    """The message of a drain that ran out of time: each stream with launches still pending, and what they are."""
    busy = [
        f"stream {index} ({stream.describe()['type']}) runs {', '.join(names)}"
        for index, stream in enumerate(streams)
        if (names := stream.pending_launches())
    ]
    return f"drain did not finish within {limit} seconds: {'; '.join(busy) or 'no launch is pending'}"
