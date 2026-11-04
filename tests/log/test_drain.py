"""A drain turns a ring into ordered log records, counts what overflowed, and dispatches to the client's loggers."""

import logging
import os
import threading
import time
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import Scalar
from krn.log.drain import Drain
from krn.log.record import Coordinates, encode_record
from krn.log.ring import Ring
from krn.log.sites import SiteTable
from tests.log.conftest import CAPACITY, Collector, Loggers, extra, sample_sites


def buffer_from(ring: Ring, table: SiteTable, calls: Sequence[tuple[int, tuple[Scalar, ...]]]) -> bytes:
    """The ring's bytes after `calls`, each a (site id, values) pair in order, padded with zeros for the rest."""
    data = b"".join(
        encode_record(table[site], values, Coordinates(seq=index, block=index // 2, thread=index % 2), ring.slots)
        for index, (site, values) in enumerate(calls)
    )
    return data.ljust(ring.capacity * ring.record_bytes, b"\0")


def buffer_of(ring: Ring, table: SiteTable, count: int) -> bytes:
    """The ring's bytes after `count` kept calls, alternating sites 1 and 0, padded with zeros for the rest."""
    calls = [(1, (index,)) if index % 2 else (0, (index, index * 0.5)) for index in range(min(count, ring.capacity))]
    return buffer_from(ring, table, calls)


def test_records_keep_sequence_order_and_fields(ring: Ring, table: SiteTable) -> None:
    logs = list(Drain(ring, table).records(buffer_of(ring, table, 2), 2))
    assert [log.getMessage() for log in logs] == ["x=0 y=0.00", "lane 1"]
    first = logs[0]
    assert (first.name, first.levelno, first.pathname, first.lineno) == ("krn.saxpy", 20, "saxpy.py", 10)
    assert (extra(first, "device_block"), extra(first, "device_thread"), extra(first, "kernel")) == (0, 0, "saxpy")
    assert (extra(first, "seq"), first.funcName) == (0, "saxpy")
    second = logs[1]
    assert (second.name, second.levelno, second.lineno, extra(second, "seq")) == ("krn.lane", 10, 12, 1)
    assert (extra(second, "device_block"), extra(second, "device_thread")) == (0, 1)


def test_host_fields_are_the_drains_and_device_identity_is_extra(ring: Ring, table: SiteTable) -> None:
    before = time.time()
    first, second = Drain(ring, table).records(buffer_of(ring, table, 2), 2)
    assert (first.thread, first.threadName, first.process) == (
        threading.get_ident(),
        threading.current_thread().name,
        os.getpid(),
    )
    assert before <= first.created <= time.time()
    assert (extra(second, "device_block"), extra(second, "device_thread")) == (0, 1)
    assert second.thread == first.thread


def test_overflow_drops_are_counted_in_one_warning_on_krn(ring: Ring, table: SiteTable) -> None:
    logs = list(Drain(ring, table).records(buffer_of(ring, table, 3), 5))
    warnings = [log for log in logs if log.levelno == logging.WARNING]
    assert len(logs) == CAPACITY + 1 and len(warnings) == 1
    [warning] = warnings
    assert warning.getMessage() == "2 log records dropped from saxpy (ring capacity 3)"
    assert (warning.name, warning.pathname, warning.lineno) == ("krn", "saxpy.py", 5)
    assert (extra(warning, "kernel"), warning.funcName) == ("saxpy", "saxpy")
    assert all(log.levelno != logging.WARNING for log in logs[:CAPACITY])


@given(st.integers(0, 3 * CAPACITY))
def test_records_and_drops_always_add_up(head: int) -> None:
    ring = Ring(kernel="saxpy", file="saxpy.py", line=5, capacity=CAPACITY, slots=2)
    table = sample_sites()
    logs = list(Drain(ring, table).records(buffer_of(ring, table, head), head))
    records = [log for log in logs if log.levelno != logging.WARNING]
    assert [extra(log, "seq") for log in records] == list(range(min(head, CAPACITY)))
    assert sum(log.levelno == logging.WARNING for log in logs) == (head > CAPACITY)


def test_dispatch_respects_each_logger_level(ring: Ring, table: SiteTable, loggers: Loggers, client: Collector) -> None:
    loggers.level("krn.lane", logging.WARNING)
    Drain(ring, table).run(buffer_of(ring, table, 3), 4)
    assert [log.getMessage() for log in client.records] == [
        "x=0 y=0.00",
        "x=2 y=1.00",
        "1 log records dropped from saxpy (ring capacity 3)",
    ]
    assert [log.levelno for log in client.records] == [logging.INFO, logging.INFO, logging.WARNING]


def test_no_drop_means_no_warning(ring: Ring, table: SiteTable, client: Collector) -> None:
    Drain(ring, table).run(buffer_of(ring, table, 3), 3)
    assert len(client.records) == 3
    assert all(log.levelno != logging.WARNING for log in client.records)


def test_a_handler_on_the_site_logger_gets_that_kernels_records(ring: Ring, table: SiteTable, loggers: Loggers) -> None:
    saxpy, lane = Collector(), Collector()
    loggers.attach("krn.saxpy", saxpy)
    loggers.attach("krn.lane", lane)
    Drain(ring, table).run(buffer_of(ring, table, 2), 2)
    assert [log.getMessage() for log in saxpy.records] == ["x=0 y=0.00"]
    assert [(log.name, extra(log, "kernel")) for log in lane.records] == [("krn.lane", "saxpy")]


def test_records_reach_krn_handlers_by_propagation(ring: Ring, table: SiteTable, client: Collector) -> None:
    Drain(ring, table).run(buffer_of(ring, table, 2), 2)
    assert [log.name for log in client.records] == ["krn.saxpy", "krn.lane"]


def test_a_logger_inherits_its_level_from_krn(
    ring: Ring, table: SiteTable, loggers: Loggers, client: Collector
) -> None:
    loggers.level("krn", logging.WARNING)
    Drain(ring, table).run(buffer_of(ring, table, 2), 2)
    assert client.records == []
    loggers.level("krn.saxpy", logging.INFO)
    Drain(ring, table).run(buffer_of(ring, table, 2), 2)
    assert [log.name for log in client.records] == ["krn.saxpy"]


def test_an_unrenderable_record_is_an_error_in_its_place(ring: Ring, table: SiteTable, client: Collector) -> None:
    calls = [(2, (65,)), (2, (300,)), (1, (7,))]
    Drain(ring, table).run(buffer_from(ring, table, calls), 3)
    first, failed, last = client.records
    assert (first.getMessage(), first.name) == ("glyph A", "krn.glyph")
    assert last.getMessage() == "lane 7"
    assert (failed.name, failed.levelno, failed.pathname, failed.lineno) == ("krn.glyph", logging.ERROR, "saxpy.py", 14)
    assert (extra(failed, "seq"), extra(failed, "kernel"), failed.funcName) == (1, "saxpy", "saxpy")
    assert failed.getMessage().startswith("could not render the record at saxpy.py:14 in saxpy: %c takes a code")
    assert failed.getMessage().endswith("; raw argument words: 0x12c")


class FailsOnSequence(logging.Handler):
    """Keeps every record but the one whose ring sequence is 1, whose emit raises."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.kept: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if extra(record, "seq") == 1:
            raise ValueError("handler broke")
        self.kept.append(record)


@pytest.mark.parametrize("raise_exceptions", [True, False])
def test_failing_handler_keeps_the_other_records(
    ring: Ring,
    table: SiteTable,
    loggers: Loggers,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    raise_exceptions: bool,
) -> None:
    monkeypatch.setattr(logging, "raiseExceptions", raise_exceptions)
    handler = FailsOnSequence()
    loggers.attach("krn", handler)
    Drain(ring, table).run(buffer_of(ring, table, 3), 3)
    assert [extra(log, "seq") for log in handler.kept] == [0, 2]
    reported = "--- Logging error ---" in capsys.readouterr().err
    assert reported is raise_exceptions
