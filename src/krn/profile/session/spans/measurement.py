from dataclasses import dataclass

from ...result.models import ProcessReading


@dataclass(frozen=True, kw_only=True)
class SpanMeasurement:
    """Completed span data summarized when a result is requested."""

    name: str
    wall_ms: float
    samples: tuple[ProcessReading, ...]
