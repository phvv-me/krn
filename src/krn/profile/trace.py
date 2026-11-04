# Native activity records, span attribution, and bottleneck ranking.

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import TracebackType

from ..compiler.errors import TargetError
from .enums import Activity
from .formats.protocols import KernelActivity, MemcpyActivity

_NANOSECONDS_PER_MICROSECOND = 1000.0


_MEMCPY_KIND = dict(
    enumerate(("unknown", "HtoD", "DtoH", "HtoA", "AtoH", "AtoA", "AtoD", "DtoA", "DtoD", "HtoH", "PtoP"))
)
_MAX_THREADS_PER_BLOCK = 1024  # the hardware cap, the occupancy-proxy denominator
# Clock of a kernel timed only by a device event pair: its duration is real, its position is not.
UNPOSITIONED = "cuda-events"


@dataclass(frozen=True, kw_only=True)
class KernelTrace:
    """One GPU kernel execution in device-clock nanoseconds, with its CUPTI launch shape.

    correlation_id: the runtime call that launched it, a separate host-clock activity; the
        join that attributes a kernel to its callsite rather than only to its region.
    region: the span path open where it was enqueued, empty when it must be binned by time.
    An event-timed kernel (clock `UNPOSITIONED`) carries start_ns 0 and its duration as end_ns;
    timelines, busy time and time binning skip it, while duration aggregates keep it.
    """

    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    correlation_id: int = 0
    grid: str = ""
    block: str = ""
    static_shared_mem: int = 0
    dynamic_shared_mem: int = 0
    registers: int = 0
    device_id: int | None = None
    context_id: int | None = None
    stream_id: int | None = None
    clock: str = "device"
    region: str = ""

    @classmethod
    def from_activity(cls, act: KernelActivity) -> KernelTrace:
        """Build from a CUPTI CONCURRENT_KERNEL activity (snake_case attributes)."""
        return cls(
            name=act.name,
            start_ns=act.start,
            end_ns=act.end,
            grid=f"{act.grid_x}x{act.grid_y}x{act.grid_z}",
            block=f"{act.block_x}x{act.block_y}x{act.block_z}",
            static_shared_mem=act.static_shared_memory,
            dynamic_shared_mem=act.dynamic_shared_memory,
            registers=act.registers_per_thread,
            correlation_id=getattr(act, "correlation_id", 0),
            device_id=getattr(act, "device_id", None),
            context_id=getattr(act, "context_id", None),
            stream_id=getattr(act, "stream_id", None),
            clock="cupti",
        )

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def positioned(self) -> bool:
        """Whether start and end are real device timestamps rather than a duration alone."""
        return self.clock != UNPOSITIONED

    @property
    def duration_us(self) -> float:
        return self.duration_ns / _NANOSECONDS_PER_MICROSECOND

    @property
    def occupancy_pct(self) -> float:
        """Threads per block over the hardware max, the static upper bound on occupancy.

        The CUPTI activity record carries the launch config but not achieved occupancy.
        """
        return 100.0 * self.threads_per_block / _MAX_THREADS_PER_BLOCK

    @property
    def threads_per_block(self) -> int:
        """The product of the block dimensions, a non-numeric one counting as 1."""
        return math.prod(int(dim) if dim.isdigit() else 1 for dim in self.block.split("x"))


@dataclass(frozen=True, kw_only=True)
class MemcpyTrace:
    """One memory copy in device-clock nanoseconds, with its direction and bytes moved.

    correlation_id: the runtime call that issued the copy, as on a kernel.
    """

    kind: str = "unknown"
    start_ns: int = 0
    end_ns: int = 0
    correlation_id: int = 0
    bytes_moved: int = 0
    device_id: int | None = None
    context_id: int | None = None
    stream_id: int | None = None
    clock: str = "device"

    @classmethod
    def from_activity(cls, act: MemcpyActivity) -> MemcpyTrace:
        """Build from a CUPTI MEMCPY activity (snake_case attributes)."""
        return cls(
            kind=_MEMCPY_KIND.get(act.copy_kind, f"kind_{act.copy_kind}"),
            start_ns=act.start,
            end_ns=act.end,
            bytes_moved=getattr(act, "bytes", 0),
            correlation_id=getattr(act, "correlation_id", 0),
            device_id=getattr(act, "device_id", None),
            context_id=getattr(act, "context_id", None),
            stream_id=getattr(act, "stream_id", None),
            clock="cupti",
        )

    @property
    def bandwidth_gbps(self) -> float:
        return self.bytes_moved / self.duration_ns if self.duration_ns > 0 else 0.0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, kw_only=True)
class ActivityRecord:
    """A timed activity of a kind beyond kernel/memcpy.

    kind: the `Activity.label` (`memset`, `runtime`, `driver`, `sync`, ...).
    name: the API function for runtime/driver, else the kind label.
    """

    kind: str = ""
    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    correlation_id: int = 0
    clock: str = "device"

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, kw_only=True)
class RegionWindow:
    """A region's device-clock window, used to attribute kernels by timestamp."""

    name: str
    start_ns: int
    end_ns: int
    wall_ns: int
    clock: str = "device"


class TraceCollector:
    """No-op deep-trace collector; a vendor backend overrides it to gather records.

    Enter starts collection and exit drains it; the profiler bins the device-clock records into
    regions afterwards.
    """

    @property
    def device_index(self) -> int:
        """Return the captured CUDA-visible ordinal, not a physical NVML index."""
        raise RuntimeError("this collector has no synchronized CUDA device")

    def __enter__(self) -> TraceCollector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def activities(self) -> list[ActivityRecord]:
        """Generic timed records for the enabled non-kernel/memcpy activity kinds."""
        return []

    def checkpoint(self, activities: Activity) -> tuple[int, int]:
        """Drain one context; return its delivered-record and lifetime-loss counters.

        The requested kinds must actually be enabled; a collector without windows refuses.
        """
        raise RuntimeError("synchronized activity windows are unavailable on this backend")

    def dropped(self) -> int:
        """Number of native records discarded by a bounded capture buffer."""
        return 0

    def flush(self) -> None:
        """Deliver buffered records, without clearing, so reads see them."""

    def kernels(self, *, since: int | None = None, until: int | None = None) -> list[KernelTrace]:
        return []

    def memcpys(self, *, since: int | None = None, until: int | None = None) -> list[MemcpyTrace]:
        return []

    def reset(self) -> None:
        """Drain and discard collected records, starting a fresh measurement window."""

    def stop(self) -> None:
        """Drain and stop collection (a single device sync happens here)."""


def busy_runs(spans: Iterable[tuple[int, int, str]]) -> list[tuple[int, int, str, str]]:
    """The union of half-open `(start_ns, end_ns, name)` spans as busy runs, empty spans dropped.

    A run is `(start_ns, end_ns, first, last)`: the span that began it and the one that reached
    its end, which name the two sides of the gap beside it.
    """
    runs: list[tuple[int, int, str, str]] = []
    for start, end, name in sorted(span for span in spans if span[1] > span[0]):
        if not runs or start > runs[-1][1]:
            runs.append((start, end, name, name))
        elif end > runs[-1][1]:
            runs[-1] = (runs[-1][0], end, runs[-1][2], name)
    return runs


def region_walls(windows: Iterable[RegionWindow]) -> dict[str, int]:
    """Wall time per region name, counting each occurrence of a name on one clock at most once in time.

    Occurrences are taken in start order; one that starts before the last counted occurrence ends is a
    nested or overlapping re-entry of that time and adds nothing. Disjoint occurrences add.
    """
    walls: Counter[str] = Counter()
    covered: dict[tuple[str, str], int] = {}
    for window in sorted(windows, key=lambda w: (w.clock, w.name, w.start_ns, -w.end_ns)):
        key = window.clock, window.name
        if window.start_ns < covered.get(key, -math.inf):
            continue
        walls[window.name] += window.wall_ns
        covered[key] = window.end_ns
    return dict(walls)


def busy_ns(spans: Iterable[tuple[int, int]]) -> int:
    """How long the device was busy: the union of the half-open `(start_ns, end_ns)` spans.

    A share against wall must divide this, never a summed duration. A sum counts concurrent or
    nested work (two streams, a copy under a kernel, a launching call beside its kernel) once
    per record, which answers how much WORK was done. The factor is structural, not
    statistical: reproducibility's `recovery_cost` read device shares of 1.908 and 1.860 of its
    own wall from a sum, and its 2026-08-29 referee reproduced that at ten launches.
    """
    runs = busy_runs((start, end, "") for start, end in spans)
    return sum(end - start for start, end, _, _ in runs)


def common_clock(clocks: Iterable[str]) -> None:
    names = {clock for clock in clocks if clock != UNPOSITIONED}
    if len(names) > 1:
        raise TargetError(
            "combined timeline",
            ", ".join(sorted(names)),
            "these clock domains have no calibrated synchronization transform",
            "profile one target at a time or inspect the separate Perfetto processes",
        )


@dataclass(frozen=True, kw_only=True)
class HotKernel:
    """A kernel name's share of total kernel time."""

    name: str
    calls: int
    total_ns: int
    avg_ns: float
    share_pct: float


@dataclass(frozen=True, kw_only=True)
class HotRegion:
    """A region's share of total kernel time (kernels binned by timestamp)."""

    name: str
    kernel_count: int
    kernel_ns: int
    wall_ns: int
    share_pct: float


@dataclass(frozen=True, kw_only=True)
class BottleneckReport:
    """Where GPU time goes: compute-vs-copy split, hot regions and hot kernels.

    total_kernel_ns/total_memcpy_ns: summed WORK time, one entry per traced record.
    device_busy_ns: CLOCK time busy with either (see `busy_ns`), the only one of the three a
        share against wall may divide.
    """

    total_kernel_ns: int
    total_memcpy_ns: int
    total_memcpy_bytes: int
    device_busy_ns: int
    compute_pct: float
    memcpy_pct: float
    hot_regions: tuple[HotRegion, ...]
    hot_kernels: tuple[HotKernel, ...]

    @classmethod
    def from_traces(
        cls,
        windows: Sequence[RegionWindow],
        kernels: Sequence[KernelTrace],
        memcpys: Sequence[MemcpyTrace],
        top: int = 10,
    ) -> BottleneckReport:
        """Bin kernels into region windows by start timestamp and rank the hot spots."""
        common_clock(span.clock for span in (*kernels, *memcpys))
        total_kernel = sum(k.duration_ns for k in kernels)
        total_memcpy = sum(m.duration_ns for m in memcpys)
        denom = total_kernel + total_memcpy or 1
        share = total_kernel or 1
        walls = region_walls(windows)
        by_region = (
            (
                k.region or cls._region_of(k.start_ns, [w for w in windows if w.clock == k.clock]),
                k.duration_ns,
            )
            for k in kernels
        )
        by_kernel = ((k.name, k.duration_ns) for k in kernels)
        return cls(
            total_kernel_ns=total_kernel,
            total_memcpy_ns=total_memcpy,
            total_memcpy_bytes=sum(m.bytes_moved for m in memcpys),
            device_busy_ns=busy_ns(
                [
                    *((span.start_ns, span.end_ns) for span in kernels if span.positioned),
                    *((span.start_ns, span.end_ns) for span in memcpys),
                ]
            ),
            compute_pct=100.0 * total_kernel / denom,
            memcpy_pct=100.0 * total_memcpy / denom,
            hot_regions=tuple(
                HotRegion(
                    name=name,
                    kernel_count=calls,
                    kernel_ns=ns,
                    wall_ns=walls.get(name, 0),
                    share_pct=100.0 * ns / share,
                )
                for name, calls, ns in cls._hottest(by_region, top)
            ),
            hot_kernels=tuple(
                HotKernel(
                    name=name,
                    calls=calls,
                    total_ns=ns,
                    avg_ns=ns / calls,
                    share_pct=100.0 * ns / share,
                )
                for name, calls, ns in cls._hottest(by_kernel, top)
            ),
        )

    @staticmethod
    def _hottest(timed: Iterable[tuple[str, int]], top: int) -> list[tuple[str, int, int]]:
        """`(name, calls, total_ns)` per name, the `top` largest totals first."""
        calls: Counter[str] = Counter()
        nanos: Counter[str] = Counter()
        for name, ns in timed:
            calls[name] += 1
            nanos[name] += ns
        return [(name, calls[name], ns) for name, ns in nanos.most_common(top)]

    @staticmethod
    def _region_of(start_ns: int, windows: Sequence[RegionWindow]) -> str:
        """The narrowest window holding `start_ns`, since nested regions share the outer's.

        A kernel in no window (e.g. work awaited by a `synchronize` outside any region) is
        labeled `(outside regions)`, so unattributed GPU time stays visible.
        """
        best, best_span = "(outside regions)", None
        for window in windows:
            if not window.start_ns <= start_ns < window.end_ns:
                continue
            span = window.end_ns - window.start_ns
            if best_span is None or span < best_span:
                best, best_span = window.name, span
        return best
