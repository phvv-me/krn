"""Copies to block-shared memory; destinations become readable after wait."""

from krn.frontend.numerics import Scalar, todo


def copy[T: Scalar](dst: T, src: T, count: int) -> None:
    """Start copying `count` (1, 2 or 4) consecutive 4-byte elements from `src` to the shared `dst`."""
    todo()


def commit() -> None:
    """Close the group of copies this thread started since the last commit."""
    todo()


def wait(pending: int) -> None:
    """Block until at most `pending` committed groups of this thread are still in flight."""
    todo()
