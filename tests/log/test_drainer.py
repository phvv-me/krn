"""The drain worker runs its snapshots, keeps and raises their failures, and refuses work once it is stopping."""

import threading
import time
from collections.abc import Callable
from threading import Barrier, Thread

import pytest

from krn.log.drain import FAILURE_LIMIT, Drain, Drainer, DrainerClosed
from krn.log.ring import Ring
from krn.log.sites import SiteTable
from tests.log.conftest import Collector, extra
from tests.log.test_drain import buffer_of

WORKER = "krn-log-drain"


def idle_snapshot(ring: Ring, table: SiteTable, count: int) -> Callable[[], tuple[bytes, int]]:
    buffer = buffer_of(ring, table, count)
    return lambda: (buffer, count)


def failing() -> tuple[bytes, int]:
    raise ValueError("snapshot failed")


def workers() -> int:
    return sum(thread.name == WORKER for thread in threading.enumerate())


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.001)


def test_submitted_snapshots_drain_before_stop(ring: Ring, table: SiteTable, client: Collector) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    for _ in range(3):
        drainer.submit(idle_snapshot(ring, table, 1))
    drainer.stop()
    assert len(client.records) == 3


def test_idle_worker_polls_until_work_arrives(ring: Ring, table: SiteTable, client: Collector) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    time.sleep(0.02)
    drainer.submit(idle_snapshot(ring, table, 2))
    drainer.stop()
    assert [extra(log, "seq") for log in client.records] == [0, 1]


def test_records_carry_the_workers_host_fields(ring: Ring, table: SiteTable, client: Collector) -> None:
    seen: list[int] = []
    buffer = buffer_of(ring, table, 1)

    def snapshot() -> tuple[bytes, int]:
        seen.append(threading.get_ident())
        return buffer, 1

    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.submit(snapshot)
    drainer.stop()
    [log] = client.records
    assert (log.thread, log.threadName) == (seen[0], WORKER)
    assert (extra(log, "device_block"), extra(log, "device_thread")) == (0, 0)


def test_start_while_running_keeps_one_worker(ring: Ring, table: SiteTable) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.start()
    assert workers() == 1
    drainer.stop()
    assert workers() == 0


def test_concurrent_starts_make_one_worker(ring: Ring, table: SiteTable) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    barrier = Barrier(8)
    errors: list[RuntimeError] = []

    def start() -> None:
        barrier.wait()
        try:
            drainer.start()
        except RuntimeError as error:
            errors.append(error)

    threads = [Thread(target=start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and workers() == 1
    drainer.stop()
    assert workers() == 0


def test_submissions_racing_stop_are_accepted_exactly_when_drained(
    ring: Ring, table: SiteTable, client: Collector
) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    accepted: list[None] = []
    refused: list[DrainerClosed] = []

    def submitter() -> None:
        while True:
            try:
                drainer.submit(idle_snapshot(ring, table, 1))
            except DrainerClosed as error:
                refused.append(error)
                return
            accepted.append(None)

    threads = [Thread(target=submitter) for _ in range(4)]
    for thread in threads:
        thread.start()
    wait_until(lambda: len(accepted) >= 20)
    drainer.stop()
    for thread in threads:
        thread.join()
    assert len(refused) == 4
    assert len(client.records) == len(accepted)
    assert workers() == 0


def test_blocked_snapshot_times_out_and_refuses_restart(ring: Ring, table: SiteTable, client: Collector) -> None:
    release = threading.Event()
    buffer = buffer_of(ring, table, 1)

    def blocked() -> tuple[bytes, int]:
        release.wait(10)
        return buffer, 1

    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.submit(blocked)
    with pytest.raises(TimeoutError, match="blocked"):
        drainer.stop(timeout=0.05)
    with pytest.raises(TimeoutError, match="blocked"):
        drainer.stop(timeout=0.01)
    with pytest.raises(RuntimeError, match="still running"):
        drainer.start()
    with pytest.raises(DrainerClosed):
        drainer.submit(idle_snapshot(ring, table, 1))
    release.set()
    drainer.stop(timeout=5)
    assert len(client.records) == 1
    drainer.start()
    drainer.stop()


def test_stop_is_idempotent_and_closes_submission(ring: Ring, table: SiteTable, client: Collector) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.submit(idle_snapshot(ring, table, 1))
    drainer.stop()
    drainer.stop()
    assert len(client.records) == 1 and workers() == 0
    assert issubclass(DrainerClosed, RuntimeError)
    with pytest.raises(DrainerClosed, match="stopping or stopped"):
        drainer.submit(idle_snapshot(ring, table, 1))


def test_stop_without_start_is_a_no_op(ring: Ring, table: SiteTable) -> None:
    drainer = Drainer(Drain(ring, table))
    drainer.stop()
    with pytest.raises(RuntimeError, match="before submitting"):
        drainer.submit(idle_snapshot(ring, table, 1))


def test_restart_after_stop_drains_again(ring: Ring, table: SiteTable, client: Collector) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.submit(idle_snapshot(ring, table, 1))
    drainer.stop()
    drainer.start()
    drainer.submit(idle_snapshot(ring, table, 1))
    drainer.stop()
    assert len(client.records) == 2


def test_a_failing_snapshot_does_not_strand_the_queued_ones(ring: Ring, table: SiteTable, client: Collector) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.submit(failing)
    drainer.submit(idle_snapshot(ring, table, 2))
    with pytest.raises(ExceptionGroup, match="failed in 1 snapshot") as info:
        drainer.stop()
    [error] = info.value.exceptions
    assert (str(error), error.__notes__) == ("snapshot failed", ["in drain snapshot 0"])
    assert [extra(log, "seq") for log in client.records] == [0, 1]
    assert drainer.failures == ()


def test_failure_is_visible_while_later_work_runs(ring: Ring, table: SiteTable, client: Collector) -> None:
    hold = threading.Event()
    buffer = buffer_of(ring, table, 1)

    def held() -> tuple[bytes, int]:
        hold.wait(10)
        return buffer, 1

    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    drainer.submit(failing)
    drainer.submit(held)
    wait_until(lambda: bool(drainer.failures))
    [(sequence, error)] = drainer.failures
    assert (sequence, str(error)) == (0, "snapshot failed")
    hold.set()
    with pytest.raises(ExceptionGroup):
        drainer.stop()
    assert len(client.records) == 1


def test_failures_beyond_the_limit_are_counted_in_a_note(ring: Ring, table: SiteTable) -> None:
    drainer = Drainer(Drain(ring, table), poll=0.001)
    drainer.start()
    for _ in range(FAILURE_LIMIT + 1):
        drainer.submit(failing)
    with pytest.raises(ExceptionGroup) as info:
        drainer.stop()
    assert len(info.value.exceptions) == FAILURE_LIMIT
    assert info.value.exceptions[0].__notes__ == ["in drain snapshot 1"]
    assert info.value.__notes__ == [f"the limit of {FAILURE_LIMIT} failures kept, so 1 older ones were dropped"]
