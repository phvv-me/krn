"""Shared fixtures for the log tests: a ring of three sites, and loggers a test attaches handlers and levels to."""

import logging
from collections.abc import Iterator

import pytest

from krn._internals.records import Message, Tag
from krn.log.ring import Ring
from krn.log.sites import SiteTable

CAPACITY = 3
SLOTS = 2


class Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class Loggers:
    """The krn loggers one test attaches handlers to or sets levels on; `reset` restores them."""

    def __init__(self) -> None:
        self._levels: set[str] = set()
        self._handlers: list[tuple[logging.Logger, logging.Handler]] = []

    def attach(self, name: str, handler: logging.Handler) -> None:
        """Add the handler to the logger `name` and let that logger pass every level, as a client does."""
        self.level(name, logging.DEBUG)
        logging.getLogger(name).addHandler(handler)
        self._handlers.append((logging.getLogger(name), handler))

    def level(self, name: str, level: int) -> None:
        logging.getLogger(name).setLevel(level)
        self._levels.add(name)

    def reset(self) -> None:
        for logger, handler in self._handlers:
            logger.removeHandler(handler)
        for name in self._levels:
            logging.getLogger(name).setLevel(logging.NOTSET)


def extra(record: logging.LogRecord, name: str) -> int | str:
    """A field the drain passes as `extra=`, which LogRecord does not declare."""
    return record.__dict__[name]


def sample_sites() -> SiteTable:
    """Site 0 is `saxpy` at INFO with an int and a float; site 1 is `lane` at DEBUG with a uint; site 2 is `glyph`
    at INFO with a uint that a `%c` renders as a character only below 128."""
    sites = SiteTable()
    sites.add(
        logger="saxpy",
        level=logging.INFO,
        file="saxpy.py",
        line=10,
        kernel="saxpy",
        message=Message.of("x={} y={:.2f}", [Tag.INT, Tag.FLOAT64]),
    )
    sites.add(
        logger="lane",
        level=logging.DEBUG,
        file="saxpy.py",
        line=12,
        kernel="saxpy",
        message=Message.of("lane {}", [Tag.UINT]),
    )
    sites.add(
        logger="glyph",
        level=logging.INFO,
        file="saxpy.py",
        line=14,
        kernel="saxpy",
        message=Message.of("glyph {:c}", [Tag.UINT]),
    )
    return sites


@pytest.fixture
def table() -> SiteTable:
    return sample_sites()


@pytest.fixture
def ring() -> Ring:
    return Ring(kernel="saxpy", file="saxpy.py", line=5, capacity=CAPACITY, slots=SLOTS)


@pytest.fixture
def loggers() -> Iterator[Loggers]:
    """The loggers a test changes; their levels and handlers are restored after it."""
    changed = Loggers()
    yield changed
    changed.reset()


@pytest.fixture
def client(loggers: Loggers) -> Collector:
    """The client's `krn` logger: every krn record reaches it by propagation, so it collects them all."""
    collector = Collector()
    loggers.attach("krn", collector)
    return collector
