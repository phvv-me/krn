# The `Profile`, the result of a profiling session, and what you do with it.

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from ...hw.peaks import PeakRates, PeakSource
from .._constants import BYTES_PER_KIBIBYTE, NANOSECONDS_PER_MILLISECOND
from ..launches import LaunchRecord
from ..trace import ActivityRecord, BottleneckReport, KernelTrace, MemcpyTrace, RegionWindow
from .efficiency import EfficiencyReport
from .enums import DeviceEvidence
from .models import RegionStat, RegionSummary
from .perfetto import write_trace
from .timeline import DeviceTimeline

if TYPE_CHECKING:
    from os import PathLike

SCHEMA_VERSION = 2  # 1 is the mainboard-era file without launch, peak or schema fields


@dataclass(frozen=True, kw_only=True)
class Profile:
    """One immutable result containing only evidence that was observed.

    Span timings, process GPU telemetry and native activities are independently optional, and
    a detected but unused GPU creates no output, so `device_evidence` says whether silence
    was expected.
    """

    host: str = ""
    device: str = ""
    device_evidence: DeviceEvidence = DeviceEvidence.UNSOUGHT
    summaries: tuple[RegionSummary, ...] = ()
    windows: tuple[RegionWindow, ...] = ()
    kernels: tuple[KernelTrace, ...] = ()
    memcpys: tuple[MemcpyTrace, ...] = ()
    activities: tuple[ActivityRecord, ...] = ()  # memset/runtime/driver/sync/... when enabled
    dropped_spans: int = 0
    dropped_activities: int = 0
    dropped_launches: int = 0
    launches: tuple[LaunchRecord, ...] = ()
    peak: PeakRates | None = None
    coherent: bool | None = None
    arch_key: str | None = None
    compute_units: int | None = None

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Profile:
        """Load a profile saved by :meth:`save`, or a mainboard-era file that predates the launch fields.

        Files without `schema_version` are version 1; a version this module does not read is refused.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        version = raw.get("schema_version", 1)
        if version not in (1, SCHEMA_VERSION):
            raise ValueError(f"profile schema {version} is unreadable: this krn reads 1 and {SCHEMA_VERSION}")
        peak = raw.get("peak")
        return cls(
            host=raw["host"],
            device=raw["device"],
            device_evidence=DeviceEvidence(raw["device_evidence"]),
            summaries=tuple(RegionSummary(**r) for r in raw["summaries"]),
            windows=tuple(RegionWindow(**r) for r in raw["windows"]),
            kernels=tuple(KernelTrace(**r) for r in raw["kernels"]),
            memcpys=tuple(MemcpyTrace(**r) for r in raw["memcpys"]),
            activities=tuple(ActivityRecord(**r) for r in raw["activities"]),
            launches=tuple(
                LaunchRecord(**(r | {"geometry": tuple(r["geometry"]), "dtypes": tuple(r["dtypes"])}))
                for r in raw.get("launches", ())
            ),
            dropped_spans=raw["dropped_spans"],
            dropped_activities=raw["dropped_activities"],
            dropped_launches=raw.get("dropped_launches", 0),
            coherent=raw.get("coherent"),
            arch_key=raw.get("arch_key"),
            compute_units=raw.get("compute_units"),
            peak=None
            if peak is None
            else PeakRates(
                memory_bandwidth_gbs=peak["memory_bandwidth_gbs"],
                memory_bandwidth_source=None
                if peak["memory_bandwidth_source"] is None
                else PeakSource(peak["memory_bandwidth_source"]),
                flops=peak["flops"],
                flops_sources={k: PeakSource(v) for k, v in peak["flops_sources"].items()},
            ),
        )

    @staticmethod
    def _region_text(stats: list[RegionStat]) -> str:
        """Plain-text per-name table for `report`."""
        if not stats:
            return "No regions recorded."
        rows = [f"{'region':<30}{'calls':>6}{'total ms':>10}{'avg ms':>9}{'peak MB':>10}"]
        rows += [
            f"{s.name:<30}{s.calls:>6d}{s.total_ms:>10.2f}{s.avg_ms:>9.2f}"
            f"{'?' if s.peak_memory_bytes is None else f'{s.peak_memory_bytes / BYTES_PER_KIBIBYTE**2:.1f}':>10}"
            for s in stats
        ]
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.report()

    def _device_section(self) -> str:
        if self.device_evidence is DeviceEvidence.ABSENT:
            return "Device\nno device evidence collected: asked for it and observed none"
        return ""

    def _gpu_section(self) -> str:
        if not (self.kernels or self.memcpys or self.activities or self.launches):
            return ""
        report = self.trace_report()
        return (
            "GPU activity\n"
            f"{len(self.kernels)} kernels, {len(self.memcpys)} copies, "
            f"{len(self.activities)} other activities, "
            f"{report.total_kernel_ns / NANOSECONDS_PER_MILLISECOND:.2f} ms kernel time"
            + (f"\n{len(self.launches)} runtime launches" if self.launches else "")
        )

    def _limit_section(self) -> str:
        drops = []
        if self.dropped_spans:
            drops.append(f"{self.dropped_spans} oldest spans dropped")
        if self.dropped_activities:
            drops.append(f"{self.dropped_activities} GPU activities dropped")
        if self.dropped_launches:
            drops.append(f"{self.dropped_launches} oldest launches dropped")
        if not drops:
            return ""
        return "Capture limit\n" + "\n".join(drops)

    def _spans_section(self) -> str:
        if not self.summaries:
            return ""
        return f"Spans\n{Profile._region_text(self.stats())}"

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """The slowest region names by total wall time."""
        return self.stats()[:top]

    def efficiency(
        self,
        *,
        sm_count: int | None = None,
        peak_bandwidth_gbs: float | None = None,
        bytes_moved: int | None = None,
        blocks_per_sm: int = 1,
        top: int = 12,
    ) -> EfficiencyReport:
        """Per-kernel launch shape, wave quantisation and achieved bandwidth.

        Exposes a grid whose final wave leaves most block slots empty, which a duration
        ranking or the timeline reads as busy.
        """
        payloads = [r.bytes_moved for r in self.launches]
        durations = [r.duration_ns for r in self.launches]
        payload = None if not payloads or None in payloads else sum(v for v in payloads if v is not None)
        elapsed = None if not durations or None in durations else sum(v for v in durations if v is not None)
        return EfficiencyReport.build(
            self.kernels,
            sm_count=self.compute_units if sm_count is None else sm_count,
            peak_bandwidth_gbs=peak_bandwidth_gbs
            if peak_bandwidth_gbs is not None
            else (self.peak.memory_bandwidth_gbs if self.peak is not None else None),
            bytes_moved=bytes_moved if bytes_moved is not None else payload,
            measured_ns=elapsed,
            blocks_per_sm=blocks_per_sm,
            top=top,
        )

    def perfetto(self, path: str | PathLike[str]) -> None:
        """Write a Perfetto/Chrome timeline (open at ui.perfetto.dev)."""
        write_trace(self, path)

    def report(self) -> str:
        """A plain-text report containing only populated evidence sections."""
        sections = [
            self._spans_section(),
            self._gpu_section(),
            self._device_section(),
            self._limit_section(),
        ]
        return "\n\n".join(section for section in sections if section) or "No profiling data collected."

    def save(self, path: str | PathLike[str]) -> None:
        """Persist to JSON so a later run can :meth:`load` it."""
        raw = {field.name: getattr(self, field.name) for field in fields(self) if field.name != "peak"}
        raw = {
            key: [asdict(item) for item in value] if isinstance(value, tuple) else value for key, value in raw.items()
        }
        raw["schema_version"] = SCHEMA_VERSION
        raw["peak"] = (
            None
            if self.peak is None
            else {
                "memory_bandwidth_gbs": self.peak.memory_bandwidth_gbs,
                "memory_bandwidth_source": self.peak.memory_bandwidth_source,
                "flops": dict(self.peak.flops),
                "flops_sources": dict(self.peak.flops_sources),
            }
        )
        Path(path).write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    def stats(self) -> list[RegionStat]:
        """Per-name aggregates (calls/total/avg/peak), slowest total first."""
        return RegionStat.aggregate(self.summaries)

    def timeline(self, top_gaps: int = 10) -> DeviceTimeline:
        """Busy/idle accounting over observed activity, with the longest idle windows.

        A pipeline that reads kernel-bound is often idle between launches. Kernels timed only by a
        device event pair have no position, so they are counted in `unpositioned` and not in busy time.
        """
        return DeviceTimeline.from_traces(self.kernels, self.memcpys, top_gaps=top_gaps)

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Deep GPU-time ranking (compute/copy split, hot regions and kernels)."""
        return BottleneckReport.from_traces(self.windows, self.kernels, self.memcpys, top)
