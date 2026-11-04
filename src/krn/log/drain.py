"""Drains turn a launch's ring into `logging` records and dispatch them to the client's own loggers."""

import logging
from collections import deque
from collections.abc import Callable, Iterator
from enum import Enum, auto
from itertools import count
from logging import LogRecord
from queue import Empty, SimpleQueue
from threading import Lock, Thread

from .._internals.records import MessageFormatError
from .levels import logger_name
from .record import Record, decode_record
from .ring import Ring
from .sites import SiteTable

type Snapshot = Callable[[], tuple[bytes, int]]

FAILURE_LIMIT = 64
# A launch's records span several loggers, so its overflow warning goes to this fixed logger, the root of krn's tree.
OVERFLOW_LOGGER = "krn"
_HANDLER_ERRORS = logging.Handler()


class DrainerClosed(RuntimeError):
    """Refuses a snapshot submitted once the drain worker has begun to stop."""


class _State(Enum):
    NEW = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


class Drain:
    """The records of one launch, as `krn.<logger>` log records dispatched to the client's loggers.

    A record is named by its site's logger, `krn.saxpy` for the logger `saxpy`, and its kernel is the `kernel` extra.
    Its level is the site's, and it reaches a logger's handlers only when that logger enables the level, so a client
    that configures `krn.saxpy` controls both the compile level and the delivery. Records propagate to `krn` and the
    root like any others. An overflowed ring adds one WARNING record on the logger `krn`, which names the dropped count
    and the capacity. A record whose message cannot be rendered is delivered as an ERROR record on its own logger, which
    names the site and the raw argument words; the launch's other records are unaffected. The standard host fields
    (`thread`, `threadName`, `process`, `created`) are the drain's, so they date the drain, not the launch. The device
    block and thread are the extras `device_block` and `device_thread`.
    """

    def __init__(self, ring: Ring, table: SiteTable) -> None:
        self._ring = ring
        self._table = table

    def records(self, buffer: bytes, head: int) -> Iterator[LogRecord]:
        """The launch's records in sequence order, then the overflow warning when the ring dropped any."""
        size = self._ring.record_bytes
        kept = min(head, self._ring.capacity)
        for index in range(kept):
            yield _log_record(decode_record(buffer[index * size : (index + 1) * size], self._table, self._ring.slots))
        if dropped := head - kept:
            yield self._overflow(dropped)

    def run(self, buffer: bytes, head: int) -> None:
        """Dispatch the launch's records, each to its logger if that logger enables the record's level.

        A handler that raises does not stop the records after it. `logging.Handler.handleError` reports the error,
        printing a traceback only while `logging.raiseExceptions` is true, as the standard handlers do.
        """
        for log in self.records(buffer, head):
            logger = logging.getLogger(log.name)
            if logger.isEnabledFor(log.levelno):
                try:
                    logger.handle(log)
                except Exception:  # the handler boundary: logging reports it and the drain goes on
                    _HANDLER_ERRORS.handleError(log)

    def _overflow(self, dropped: int) -> LogRecord:
        message = f"{dropped} log records dropped from {self._ring.kernel} (ring capacity {self._ring.capacity})"
        return logging.getLogger(OVERFLOW_LOGGER).makeRecord(
            OVERFLOW_LOGGER,
            logging.WARNING,
            self._ring.file,
            self._ring.line,
            message,
            (),
            None,
            func=self._ring.kernel,
            extra={"kernel": self._ring.kernel},
        )


class Drainer:
    """Runs drains on one worker thread that polls for snapshots every `poll` seconds.

    A snapshot is a callable that returns the buffer and head once the launch has finished, so it may block. The state
    goes NEW, RUNNING, STOPPING, STOPPED. `submit` is accepted only while RUNNING, so every accepted snapshot is drained
    before `stop` returns. `start` on a RUNNING worker returns at once, and `start` while STOPPING raises RuntimeError,
    because the thread still runs a snapshot that may block. A snapshot that raises is retained with its sequence
    number, and the worker goes on with the queue. `stop` then raises the retained failures as one `ExceptionGroup`
    and forgets them. At most `FAILURE_LIMIT` are kept; the rest are counted in a note on the group.
    """

    def __init__(self, drain: Drain, *, poll: float = 0.05) -> None:
        self._drain = drain
        self._poll = poll
        self._lock = Lock()
        self._state = _State.NEW
        self._thread: Thread | None = None
        self._pending: SimpleQueue[tuple[int, Snapshot]] = SimpleQueue()
        self._sequences = count()
        self._failures: deque[tuple[int, Exception]] = deque(maxlen=FAILURE_LIMIT)
        self._discarded = 0

    @property
    def failures(self) -> tuple[tuple[int, Exception], ...]:
        """The retained failures as (snapshot sequence, error), oldest first, since the last `stop`."""
        with self._lock:
            return tuple(self._failures)

    def submit(self, snapshot: Snapshot) -> None:
        """Queue a snapshot. Raises RuntimeError before `start`, and DrainerClosed once the worker is stopping."""
        with self._lock:
            if self._state is _State.NEW:
                raise RuntimeError("start the drain worker before submitting a snapshot")
            if self._state is not _State.RUNNING:
                raise DrainerClosed("the drain worker is stopping or stopped, so the snapshot is refused")
            self._pending.put((next(self._sequences), snapshot))

    def start(self) -> None:
        """Start the worker, or return at once when it is already running."""
        with self._lock:
            if self._state is _State.STOPPING:
                raise RuntimeError("the drain worker is still running; stop it before starting it again")
            if self._state is _State.RUNNING:
                return
            self._state = _State.RUNNING
            self._thread = Thread(target=self._work, name="krn-log-drain", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Drain the queued snapshots, join the worker, then raise the retained failures as one `ExceptionGroup`.

        Raises TimeoutError when a snapshot still blocks the worker after `timeout` seconds. The thread stays
        referenced, so `start` refuses until a later `stop` has joined it.
        """
        with self._lock:
            if self._state is _State.RUNNING:
                self._state = _State.STOPPING
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"the drain worker is still blocked after {timeout} seconds, in a snapshot")
        if (group := self._take_failures()) is not None:
            raise group

    def _work(self) -> None:
        while True:
            try:
                sequence, snapshot = self._pending.get(timeout=self._poll)
            except Empty:
                if self._finish():
                    return
                continue
            self._process(sequence, snapshot)

    def _finish(self) -> bool:
        """Whether the worker may exit: STOPPING with nothing queued, which no later submit can change."""
        with self._lock:
            if self._state is _State.STOPPING and self._pending.empty():
                self._state = _State.STOPPED
                return True
            return False

    def _process(self, sequence: int, snapshot: Snapshot) -> None:
        try:
            self._drain.run(*snapshot())
        except Exception as error:  # the worker boundary: the failure is retained for stop
            error.add_note(f"in drain snapshot {sequence}")
            with self._lock:
                if len(self._failures) == FAILURE_LIMIT:
                    self._discarded += 1
                self._failures.append((sequence, error))

    def _take_failures(self) -> ExceptionGroup[Exception] | None:
        with self._lock:
            failures = list(self._failures)
            discarded, self._discarded = self._discarded, 0
            self._failures.clear()
        if not failures:
            return None
        group = ExceptionGroup(
            f"the drain worker failed in {len(failures)} snapshot(s)", [error for _, error in failures]
        )
        if discarded:
            group.add_note(f"the limit of {FAILURE_LIMIT} failures kept, so {discarded} older ones were dropped")
        return group


def _log_record(record: Record) -> LogRecord:
    site = record.site
    name = logger_name(site.logger)
    try:
        level, message = site.level, site.message.render(record.values)
    except MessageFormatError as error:
        level, message = logging.ERROR, _unrendered(record, error)
    return logging.getLogger(name).makeRecord(
        name,
        level,
        site.file,
        site.line,
        message,
        (),
        None,
        func=site.kernel,
        extra={
            "device_block": record.coordinates.block,
            "device_thread": record.coordinates.thread,
            "kernel": site.kernel,
            "seq": record.coordinates.seq,
        },
    )


def _unrendered(record: Record, error: MessageFormatError) -> str:
    site = record.site
    words = ", ".join(hex(word) for word in record.words)
    return (
        f"could not render the record at {site.file}:{site.line} in {site.kernel}: {error}; raw argument words: {words}"
    )
