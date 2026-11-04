# What one Apple GPU exposes for timing and counters, read from the `MTLDevice` it is given.

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import Metal
else:
    try:
        import Metal
    except ImportError:
        Metal = None

# Sampling points by name, as the counter API spells them: the `MTLCounterSamplingPoint` constants.
_POINTS = {
    "stage": "MTLCounterSamplingPointAtStageBoundary",
    "dispatch": "MTLCounterSamplingPointAtDispatchBoundary",
    "draw": "MTLCounterSamplingPointAtDrawBoundary",
    "blit": "MTLCounterSamplingPointAtBlitBoundary",
}
_TIMESTAMP_SET = "timestamp"
_STATISTIC_SETS = frozenset({"statistic", "stageutilization"})


@dataclass(frozen=True, kw_only=True)
class Capabilities:
    """The counter sets, their counters, the sampling points and capture permission one device offers.

    counter_sets: each set's name mapped to its counter names. sampling: the names in `_POINTS` the device supports.
    capture: whether `MTLCaptureManager` accepts a GPU trace document from this process.
    """

    chip: str
    counter_sets: Mapping[str, tuple[str, ...]]
    sampling: frozenset[str]
    capture: bool

    @classmethod
    def probe(cls, device: Metal.MTLDevice) -> Capabilities:
        """Read the device's counter API. Reads only: no sample buffer is created and no work is submitted."""
        manager = Metal.MTLCaptureManager.sharedCaptureManager()
        return cls(
            chip=device.name(),
            counter_sets={s.name(): tuple(c.name() for c in s.counters()) for s in device.counterSets()},
            sampling=frozenset(
                point
                for point, attribute in _POINTS.items()
                if device.supportsCounterSampling_(getattr(Metal, attribute))
            ),
            capture=bool(manager.supportsDestination_(Metal.MTLCaptureDestinationGPUTraceDocument)),
        )

    @property
    def timestamps(self) -> bool:
        """Whether the timestamp counter set exists, the only per-work time the public counter API offers."""
        return _TIMESTAMP_SET in self.counter_sets

    @property
    def stage_sampling(self) -> bool:
        return "stage" in self.sampling

    @property
    def dispatch_sampling(self) -> bool:
        return "dispatch" in self.sampling

    @property
    def statistics(self) -> bool:
        """Whether an ALU, memory or occupancy counter set exists, which the public API would sample per encoder."""
        return bool(_STATISTIC_SETS & self.counter_sets.keys())

    def names(self) -> str:
        """The counter set names, or `none`, for an explained refusal."""
        return ", ".join(sorted(self.counter_sets)) or "none"
