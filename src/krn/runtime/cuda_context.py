"""The CUDA context a thread drives a device through.

The driver binds a context to one thread at a time, and a call on a thread that never bound it fails with
CUDA_ERROR_INVALID_CONTEXT. Each thread that drives a device binds it once, before its first call, so a program can be
compiled, loaded and launched from any thread a program runs on.
"""

import threading
from collections.abc import Callable


class _Bound(threading.local):
    done = False


class ThreadContext:
    """A device's context, bound by `bind` on each thread the first time that thread calls `activate`."""

    def __init__(self, bind: Callable[[], None]) -> None:
        self._bind, self._bound = bind, _Bound()

    def activate(self) -> None:
        """A thread-local read after the first call on a thread, so the launch path costs no driver call."""
        if not self._bound.done:
            self._bind()
            self._bound.done = True
