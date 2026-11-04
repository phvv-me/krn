"""The thread: its coordinates in the block (`threadIdx`) and the fences that order its own memory writes.

Read an axis straight from the call, `thread.index().x`. Bodies are signature-only."""

from ..numerics import todo
from ..scopes import Dim3


def index() -> Dim3:
    """Coordinates of this thread inside its block."""
    return todo()


def fence_block() -> None:
    """Order this thread's writes for the block before its later memory operations."""
    return todo()


def fence_device() -> None:
    """Order this thread's writes for the device before its later memory operations."""
    return todo()


def fence_system() -> None:
    """Order this thread's writes across devices and the host."""
    return todo()
