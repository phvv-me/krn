from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ...result.models import ProcessReading

_MAXLEN = 4096


@dataclass(frozen=True, kw_only=True)
class SpanFrame:
    """One live span with a bounded, mutable telemetry buffer."""

    name: str
    path: str
    thread: int
    device_start_ns: int
    finish_marker: Callable[[], None] | None
    samples: deque[ProcessReading] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
