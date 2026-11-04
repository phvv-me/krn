"""A sliding window over awaitables: at most `depth` in flight, results in submission order, and no waves."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Iterable
from itertools import islice


async def window[T](items: Iterable[Awaitable[T]], depth: int) -> AsyncIterator[T]:
    """Await `items` in order with at most `depth` of them started and not yet yielded.

    Each item starts as a task as soon as the window has room, and its result is yielded as soon as it and every earlier
    item are done. One slow item therefore never holds back the others the way a wave of `depth` items does. A failed
    item raises at its place in the order, and the items still in flight are cancelled: a cancelled item's device work
    goes on, and its slot returns to its pool when that work completes.
    """
    if depth < 1:
        raise ValueError(f"a window needs a depth of at least 1, not {depth}")
    source = iter(items)
    pending: deque[asyncio.Future[T]] = deque(asyncio.ensure_future(item) for item in islice(source, depth))
    try:
        while pending:
            result = await pending.popleft()
            pending.extend(asyncio.ensure_future(item) for item in islice(source, 1))
            yield result
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
