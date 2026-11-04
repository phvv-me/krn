from dataclasses import dataclass

from ..enums import Activity
from .enums import Feature


@dataclass(frozen=True, kw_only=True)
class Collection:
    """Collection policy retained beside profiling evidence.

    features: independently enabled collectors.
    activities: CUPTI kinds retained when ACTIVITY is enabled.
    device_index: selected entry in the profiler's GPU sequence.
    sample_interval_ms: device telemetry polling interval.
    max_spans: retained span and window limit; excess records are counted.
    auto: modules whose functions receive automatic annotations.
    """

    features: Feature = Feature.DEFAULT
    activities: Activity = Activity.DEFAULT
    device_index: int = 0
    sample_interval_ms: int = 50
    max_spans: int = 100_000
    auto: tuple[str, ...] = ()
    native: bool = False

    def __post_init__(self) -> None:
        if self.device_index < 0 or self.sample_interval_ms <= 0 or self.max_spans <= 0:
            raise ValueError("device_index must be nonnegative; sample_interval_ms and max_spans must be positive")
