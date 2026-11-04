"""Compile levels read the client's `krn.<logger>` loggers, so a kernel's cache key changes with its client's levels."""

import logging
from collections.abc import Iterator

import pytest

from krn.log.levels import compile_level, enabled, level_key, logger_name


@pytest.fixture
def levels() -> Iterator[list[str]]:
    """Loggers whose levels one test sets; their levels are reset after it."""
    touched: list[str] = []
    yield touched
    for name in touched:
        logging.getLogger(logger_name(name)).setLevel(logging.NOTSET)


def test_a_logger_is_the_krn_name_the_client_configures() -> None:
    assert logger_name("saxpy") == "krn.saxpy"
    assert logging.getLogger(logger_name("saxpy")) is logging.getLogger("krn.saxpy")


def test_compile_level_is_the_clients_effective_level(levels: list[str]) -> None:
    logging.getLogger("krn.lane").setLevel(logging.WARNING)
    levels.append("lane")
    assert compile_level("lane") == logging.WARNING
    assert compile_level("other") == logging.getLogger("krn").getEffectiveLevel()
    logging.getLogger("krn").setLevel(logging.DEBUG)
    try:
        assert compile_level("other") == logging.DEBUG
    finally:
        logging.getLogger("krn").setLevel(logging.NOTSET)


def test_level_key_changes_with_the_clients_level(levels: list[str]) -> None:
    before = level_key(["lane", "lane", "saxpy"])
    logging.getLogger("krn.lane").setLevel(logging.ERROR)
    levels.append("lane")
    after = level_key(["saxpy", "lane"])
    assert before != after
    assert after == (("lane", logging.ERROR), ("saxpy", compile_level("saxpy")))


def test_enabled_compares_against_the_compile_level() -> None:
    assert enabled(logging.INFO, logging.INFO)
    assert enabled(logging.ERROR, logging.INFO)
    assert not enabled(logging.DEBUG, logging.INFO)
