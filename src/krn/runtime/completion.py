"""Events that one thread resolves, the waiter that settles a device's events, and the arguments a stream keeps alive.

A host thread never polls a device. An event is resolved once by the thread that reached it, and its subscribers run on
that thread, so `Event.subscribe` costs one callback and no thread per await. A GPU event is settled by its device's one
`Waiter`, which blocks on the events in the order they were watched: an event that completes behind an older one on
another stream resolves when that older one does. A launch's tensors are held by its stream until a marker after the
launch is done, so a tensor freed by the host cannot be reused while the device still reads it.
"""

import logging
import threading
from collections import deque
from collections.abc import Callable
from queue import SimpleQueue

from krn.runtime.api import Event

log = logging.getLogger(__name__)
KEEP_LIMIT = 256  # launches a stream holds before it records a marker of its own, so arguments never pile up


class Completion(Event):
    """An event that one thread resolves once, with the error of the work before it when that work failed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self._done = False
        self._error: Exception | None = None

    def done(self) -> bool:
        return self._done

    @property
    def error(self) -> Exception | None:
        return self._error

    def resolve(self, error: Exception | None = None) -> None:
        """Mark the event done and run its subscribers on this thread. Only the first call has an effect."""
        with self._lock:
            if self._done:
                return
            self._done, self._error = True, error
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()

    def subscribe(self, callback: Callable[[], None]) -> bool:
        with self._lock:
            if not self._done:
                self._callbacks.append(callback)
                return False
        return True

    def wait(self, timeout: float | None = None) -> None:
        """Block the calling thread until the event completes, then raise the error of its work, if any."""
        arrived = threading.Event()
        if not self.subscribe(arrived.set) and not arrived.wait(timeout):
            raise TimeoutError(f"event not completed after {timeout} seconds")
        if self._error is not None:
            raise self._error

    def settle(self) -> None:
        """Block until the work before this event is finished, then resolve the event with what the device reported."""
        if self._done:  # a second subscriber of a watched event finds it settled already
            return
        try:
            self._finish()
        except Exception as failure:  # a waiter has no caller to raise to: the failure is the event's result
            log.debug("device completion failed", exc_info=True)
            self.resolve(failure)
        else:
            self.resolve()

    def _finish(self) -> None:
        """Block until completion and raise the device failure; used only by watched events."""
        raise NotImplementedError(f"{type(self).__name__} is resolved by the thread that reaches it, not settled")


class Waiter:
    """The one thread of a device that settles its completions, in the order they were watched."""

    def __init__(self, name: str) -> None:
        self._queue: SimpleQueue[Completion] = SimpleQueue()
        threading.Thread(target=self._run, name=name, daemon=True).start()

    def watch(self, completion: Completion) -> None:
        self._queue.put(completion)

    def _run(self) -> None:
        while True:
            completion = self._queue.get()
            try:
                completion.settle()
            except Exception:  # the top of the thread: a subscriber's failure must not stop the waiter for other events
                log.exception("a completion subscriber failed on the device waiter")


class Keeper:
    """Named launch arguments retained until the work that reads them completes.

    A launch's arguments wait with the next marker of their stream, and a marker's arguments are released, in order,
    once that marker's event is done. The tensors of a launch therefore outlive the device's use of them.
    """

    def __init__(self) -> None:
        self._open: list[tuple[str, tuple[object, ...]]] = []
        self._marks: deque[tuple[Event, list[tuple[str, tuple[object, ...]]]]] = deque()

    @property
    def full(self) -> bool:
        return len(self._open) >= KEEP_LIMIT

    def keep(self, name: str, arguments: tuple[object, ...]) -> None:
        self._open.append((name, arguments))

    def mark(self, event: Event) -> None:
        """Hold the open launches until `event`, and release every older marker that is done."""
        self._marks.append((event, self._open))
        self._open = []
        while self._marks and self._marks[0][0].done():
            self._marks.popleft()

    def pending(self) -> tuple[str, ...]:
        """Launches awaiting completion, from unfinished markers followed by unmarked launches."""
        marked = [(event, launches) for event, launches in self._marks if not event.done()]
        names = [name for _, launches in marked for name, _ in launches]
        return (*names, *(name for name, _ in self._open))
