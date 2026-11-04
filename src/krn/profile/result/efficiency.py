# Per-kernel launch efficiency: waves, tail quantisation and achieved bandwidth.

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import JustifyMethod

    from ..trace import KernelTrace

_NS_PER_MS = 1_000_000


def _grid_blocks(shape: str) -> int:
    """The total extent of a launch shape, 0 when it has no digits.

    CUPTI writes `110x1x1`; parenthesised and comma-separated spellings are accepted too, so a
    backend that formats differently does not silently report zero.
    """
    cleaned = shape.replace("(", " ").replace(")", " ").replace(",", " ").replace("x", " ")
    digits = [int(part) for part in cleaned.split() if part.isdigit()]
    return math.prod(digits) if digits else 0


@dataclass(frozen=True, kw_only=True)
class KernelEfficiency:
    """Launch shape and efficiency for one kernel name, aggregated over its calls.

    waves: grid blocks divided by the device's block slots. A whole number means every
        block slot finishes together; a fractional one leaves a partly idle tail wave.
    tail_waste_pct: share of the kernel's time spent in that partial final wave, assuming
        every block costs the same.
    """

    name: str
    calls: int
    total_ms: float
    blocks: int
    block_threads: int
    registers: int
    waves: float | None
    tail_waste_pct: float | None

    @classmethod
    def aggregate(
        cls, kernels: Sequence[KernelTrace], sm_count: int | None, *, blocks_per_sm: int = 1
    ) -> list[KernelEfficiency]:
        """Rank kernels that ran by total device time, each with its launch shape.

        blocks_per_sm: resident blocks per multiprocessor. One gives the coarsest wave count,
            the one that matters for a grid far smaller than the machine.
        """
        slots = None if sm_count is None else sm_count * blocks_per_sm
        groups: defaultdict[str, list[KernelTrace]] = defaultdict(list)
        for kernel in kernels:
            if kernel.end_ns > kernel.start_ns:
                groups[kernel.name].append(kernel)

        rows: list[KernelEfficiency] = []
        for name, traces in groups.items():
            blocks = max(_grid_blocks(t.grid) for t in traces)
            waves = blocks / slots if slots else None
            whole = 0 if waves is None else int(waves)
            partial = 0 if waves is None else waves - whole
            rows.append(
                cls(
                    name=name,
                    calls=len(traces),
                    total_ms=sum(t.end_ns - t.start_ns for t in traces) / _NS_PER_MS,
                    blocks=blocks,
                    block_threads=_grid_blocks(traces[0].block),
                    registers=traces[0].registers,
                    waves=waves,
                    tail_waste_pct=None
                    if waves is None
                    else (0.0 if partial == 0 else (1 - partial) / (whole + 1) * 100),
                )
            )
        return sorted(rows, key=lambda r: r.total_ms, reverse=True)


@dataclass(frozen=True, kw_only=True)
class EfficiencyReport:
    """Launch efficiency across every kernel observed, with achieved bandwidth.

    bytes_moved: application-supplied payload size, so achieved bandwidth is reported
        against what the work actually required rather than a guess from the trace.
    """

    rows: tuple[KernelEfficiency, ...] = ()
    sm_count: int | None = None
    peak_bandwidth_gbs: float | None = None
    bytes_moved: int | None = None
    measured_ns: int | None = None

    @classmethod
    def build(
        cls,
        kernels: Sequence[KernelTrace],
        *,
        sm_count: int | None,
        peak_bandwidth_gbs: float | None = None,
        bytes_moved: int | None = None,
        measured_ns: int | None = None,
        blocks_per_sm: int = 1,
        top: int = 12,
    ) -> EfficiencyReport:
        rows = KernelEfficiency.aggregate(kernels, sm_count, blocks_per_sm=blocks_per_sm)
        return cls(
            rows=tuple(rows[:top]),
            sm_count=sm_count,
            peak_bandwidth_gbs=peak_bandwidth_gbs,
            bytes_moved=bytes_moved,
            measured_ns=sum(k.duration_ns for k in kernels if k.duration_ns > 0)
            if measured_ns is None
            else measured_ns,
        )

    @property
    def achieved_gbs(self) -> float | None:
        """Payload bytes over total device time."""
        nanos = self.measured_ns if self.measured_ns is not None else int(self.total_ms * 1e6)
        if nanos <= 0 or self.bytes_moved is None:
            return None
        return self.bytes_moved / nanos

    @property
    def bandwidth_utilisation_pct(self) -> float | None:
        achieved = self.achieved_gbs
        if not self.peak_bandwidth_gbs or achieved is None:
            return None
        return 100.0 * achieved / self.peak_bandwidth_gbs

    @property
    def total_ms(self) -> float:
        return self.measured_ns / 1e6 if self.measured_ns is not None else sum(row.total_ms for row in self.rows)

    @staticmethod
    def _readable(name: str) -> str:
        """A kernel name with the Itanium mangling and JIT suffix trimmed to its first three parts.

        numba appends a long content hash to every symbol, so the informative part is near the
        front.
        """
        if not name.startswith("_ZN"):
            return name
        parts: list[str] = []
        index = 3
        # Each part is spelled `<length><identifier>`, so `5numba` is `numba`.
        while index < len(name) and name[index].isdigit():
            end = index
            while end < len(name) and name[end].isdigit():
                end += 1
            length = int(name[index:end])
            index = end + length
            parts.append(name[end:index])
        return ".".join(parts[:3]) if parts else name

    def __rich__(self) -> Table:
        title = f"launch efficiency ({self.sm_count if self.sm_count is not None else '?'} compute units"
        if (achieved := self.achieved_gbs) is not None:
            title += f", {achieved:.0f} GB/s achieved"
        if (utilisation := self.bandwidth_utilisation_pct) is not None:
            title += f" = {utilisation:.1f}% of peak"
        table = Table(title=title + ")", title_style="bold")
        columns: tuple[tuple[str, JustifyMethod], ...] = (
            ("kernel", "left"),
            ("calls", "right"),
            ("total ms", "right"),
            ("blocks", "right"),
            ("thr/blk", "right"),
            ("waves", "right"),
            ("tail %", "right"),
        )
        for column, justify in columns:
            table.add_column(column, justify=justify)
        for row in self.rows:
            table.add_row(
                EfficiencyReport._readable(row.name)[:40],
                f"{row.calls}",
                f"{row.total_ms:.2f}",
                f"{row.blocks}",
                f"{row.block_threads}",
                "?" if row.waves is None else f"{row.waves:.2f}",
                "?" if row.tail_waste_pct is None else f"{row.tail_waste_pct:.1f}",
            )
        return table
