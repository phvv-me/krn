"""The per-thread binding of a device's context, with a fake bind so that it runs without a card."""

import threading

import pytest

from krn.runtime.cuda_context import ThreadContext


def test_a_thread_binds_once_and_each_new_thread_binds_its_own() -> None:
    bound: list[int] = []
    context = ThreadContext(lambda: bound.append(threading.get_ident()))

    def drive() -> None:
        for _ in range(3):
            context.activate()

    drive()
    worker = threading.Thread(target=drive)
    worker.start()
    worker.join()
    assert bound == [threading.get_ident(), worker.ident]


def test_a_bind_that_fails_is_retried_by_the_next_call() -> None:
    attempts: list[str] = []

    def flaky() -> None:
        attempts.append("try")
        if len(attempts) == 1:
            raise RuntimeError("no device")

    context = ThreadContext(flaky)
    with pytest.raises(RuntimeError, match="no device"):
        context.activate()
    context.activate()
    context.activate()
    assert attempts == ["try", "try"]
