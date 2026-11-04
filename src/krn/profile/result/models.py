from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


def known_mean(values: Iterable[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def known_max[T: (int, float)](values: Iterable[T | None]) -> T | None:
    return max((value for value in values if value is not None), default=None)


@dataclass(frozen=True, kw_only=True)
class ProcessReading:
    """One device telemetry sample attributed to a profiled span.

    memory_used_bytes: the profiled process's own device memory. The other fields stay
        whole-device signals, since no device sensor splits them per process.
    """

    unit_name: str = ""
    memory_used_bytes: int | None = None
    gpu_util_pct: float | None = None
    memory_util_pct: float | None = None
    power_w: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True, kw_only=True)
class RegionSummary:
    """Wall time and aggregated device telemetry for one occurrence of a profiled region.

    samples: snapshots taken during the region; the memory, util and power fields are their
        peak or mean, temperature their max. counters: hardware counter totals measured over the region, empty
        when none were read, never zeros standing in for a missing reading.
    """

    name: str
    wall_ms: float
    samples: int = 0
    peak_memory_bytes: int | None = None
    avg_memory_bytes: int | None = None
    avg_util_pct: float | None = None
    avg_memory_util_pct: float | None = None
    avg_power_w: float | None = None
    max_temp_c: float | None = None
    counters: dict[str, float] = field(default_factory=dict, hash=False)

    @classmethod
    def from_snaps(cls, name: str, wall_ms: float, snaps: Sequence[ProcessReading]) -> RegionSummary:
        if not snaps:
            return cls(name=name, wall_ms=wall_ms)
        memory = [s.memory_used_bytes for s in snaps]
        mean_memory = known_mean(memory)
        return cls(
            name=name,
            wall_ms=wall_ms,
            samples=len(snaps),
            peak_memory_bytes=known_max(memory),
            avg_memory_bytes=None if mean_memory is None else int(mean_memory),
            avg_util_pct=known_mean(s.gpu_util_pct for s in snaps),
            avg_memory_util_pct=known_mean(s.memory_util_pct for s in snaps),
            avg_power_w=known_mean(s.power_w for s in snaps),
            max_temp_c=known_max(s.temperature_c for s in snaps),
        )


@dataclass(frozen=True, kw_only=True)
class RegionStat:
    """One region name's aggregate over all its calls: the readable row of a profile."""

    name: str
    calls: int
    total_ms: float
    avg_ms: float
    peak_memory_bytes: int | None
    max_util_pct: float | None
    max_power_w: float | None

    @classmethod
    def aggregate(cls, summaries: Sequence[RegionSummary]) -> list[RegionStat]:
        """Collapse per-occurrence summaries into per-name stats, slowest total first."""
        groups: defaultdict[str, list[RegionSummary]] = defaultdict(list)
        for summary in summaries:
            groups[summary.name].append(summary)
        stats = [
            cls(
                name=name,
                calls=len(rows),
                total_ms=sum(r.wall_ms for r in rows),
                avg_ms=sum(r.wall_ms for r in rows) / len(rows),
                peak_memory_bytes=known_max(r.peak_memory_bytes for r in rows),
                max_util_pct=known_max(r.avg_util_pct for r in rows),
                max_power_w=known_max(r.avg_power_w for r in rows),
            )
            for name, rows in groups.items()
        ]
        return sorted(stats, key=lambda s: s.total_ms, reverse=True)
