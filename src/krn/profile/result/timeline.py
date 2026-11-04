from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from rich.table import Table

from .._constants import NANOSECONDS_PER_MILLISECOND
from ..trace import KernelTrace, MemcpyTrace, busy_runs, common_clock

_MIN_GAP_NS = 10_000


@dataclass(frozen=True, kw_only=True)
class DeviceGap:
    """One window in which no kernel or copy was executing.

    after: the activity that ended the busy run before this gap.
    before: the activity that starts the next busy run.
    """

    start_ns: int
    end_ns: int
    after: str = ""
    before: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / NANOSECONDS_PER_MILLISECOND


@dataclass(frozen=True, kw_only=True)
class DeviceTimeline:
    """How much of the wall clock the device was actually working.

    span_ns: first activity start to last activity end.
    busy_ns: the union of activity intervals, so overlapping kernels count once.
    gaps: idle windows, longest first.
    unpositioned: kernels timed only by a device event pair; their duration is in the totals but they
        have no place on the timeline, so busy and idle time exclude them.
    """

    span_ns: int = 0
    busy_ns: int = 0
    activities: int = 0
    gaps: tuple[DeviceGap, ...] = ()
    unpositioned: int = 0

    @classmethod
    def from_traces(
        cls,
        kernels: Sequence[KernelTrace],
        memcpys: Sequence[MemcpyTrace] = (),
        *,
        top_gaps: int = 10,
        min_gap_ns: int = _MIN_GAP_NS,
    ) -> DeviceTimeline:
        """Build the timeline from observed activity.

        top_gaps: how many idle windows to list, longest first.
        min_gap_ns: shorter windows are launch jitter rather than a stall: counted as idle,
            never listed.
        """
        common_clock(span.clock for span in (*kernels, *memcpys))
        unpositioned = sum(not k.positioned for k in kernels)
        spans = [(k.start_ns, k.end_ns, k.name) for k in kernels if k.positioned and k.end_ns > k.start_ns]
        spans += [(m.start_ns, m.end_ns, "memcpy") for m in memcpys if m.end_ns > m.start_ns]
        if not spans:
            return cls(unpositioned=unpositioned)
        runs = busy_runs(spans)
        gaps = [
            DeviceGap(start_ns=ended, end_ns=started, after=last, before=first)
            for (_, ended, _, last), (started, _, first, _) in pairwise(runs)
            if started - ended >= min_gap_ns
        ]

        # Latest end, not the last span's: a long kernel can outlast a later, shorter one.
        return cls(
            span_ns=runs[-1][1] - runs[0][0],
            busy_ns=sum(end - start for start, end, _, _ in runs),
            activities=len(spans),
            gaps=tuple(sorted(gaps, key=lambda g: g.end_ns - g.start_ns, reverse=True)[:top_gaps]),
            unpositioned=unpositioned,
        )

    @property
    def idle_ns(self) -> int:
        return max(self.span_ns - self.busy_ns, 0)

    @property
    def occupancy_pct(self) -> float:
        return 100.0 * self.busy_ns / self.span_ns if self.span_ns else 0.0

    def __rich__(self) -> Table:
        table = Table(title="device timeline", title_style="bold")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("activities", f"{self.activities}")
        table.add_row("span", f"{self.span_ns / NANOSECONDS_PER_MILLISECOND:.2f} ms")
        table.add_row("busy", f"{self.busy_ns / NANOSECONDS_PER_MILLISECOND:.2f} ms")
        table.add_row("idle", f"{self.idle_ns / NANOSECONDS_PER_MILLISECOND:.2f} ms")
        table.add_row("occupancy", f"{self.occupancy_pct:.1f}%")
        if self.unpositioned:
            table.add_row("unpositioned kernels", f"{self.unpositioned} (durations only, not placed)")
        for gap in self.gaps:
            table.add_row(f"gap {gap.after} → {gap.before}", f"{gap.duration_ms:.2f} ms")
        return table

    def __str__(self) -> str:
        return (
            f"span {self.span_ns / NANOSECONDS_PER_MILLISECOND:.1f} ms, "
            f"busy {self.busy_ns / NANOSECONDS_PER_MILLISECOND:.1f} ms, "
            f"occupancy {self.occupancy_pct:.1f}%, {len(self.gaps)} listed gaps, "
            f"{self.unpositioned} kernels without timestamps"
        )
