"""Compile levels: the client's own `logging` loggers decide which kernel calls are compiled in.

A logger named `saxpy` in krn is `logging.getLogger("krn.saxpy")`, so a client configures it like any other logger.
The kernel a call sits in is record metadata, never part of the name.
"""

import logging
from collections.abc import Iterable


def logger_name(logger: str) -> str:
    """The `logging` name of a krn logger, the one `krn.log.get` returns a logger for."""
    return f"krn.{logger}"


def compile_level(logger: str) -> int:
    """The effective level of the client's logger, read when a kernel is captured; krn has no level of its own."""
    return logging.getLogger(logger_name(logger)).getEffectiveLevel()


def level_key(loggers: Iterable[str]) -> tuple[tuple[str, int], ...]:
    """The loggers a program uses with their compile levels, for its cache key: a changed level recompiles."""
    return tuple(sorted({(logger, compile_level(logger)) for logger in loggers}))


def enabled(level: int, compile_level: int) -> bool:
    """Whether a call at `level` is compiled in when the program was compiled at `compile_level`."""
    return level >= compile_level
