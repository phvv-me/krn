"""Chrome trace-event JSON, with independent processes for unaligned clock domains."""

import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from ..formats.protocols import Json, TraceEvent

if TYPE_CHECKING:
    from . import Profile

_NS_PER_US = 1000.0
_REGIONS, _KERNELS, _MEMCPYS, _ACTIVITIES = 1, 2, 3, 4
_TRACKS = {_REGIONS: "regions", _KERNELS: "GPU kernels", _MEMCPYS: "GPU memcpy", _ACTIVITIES: "API & activity"}


def _meta(name: str, tid: int, label: str, pid: int) -> TraceEvent:
    return {"ph": "M", "name": name, "pid": pid, "tid": tid, "args": {"name": label}}


def _timed(name: str, tid: int, start_ns: int, dur_ns: float, pid: int, args: dict[str, Json]) -> TraceEvent:
    return {
        "ph": "X",
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": start_ns / _NS_PER_US,
        "dur": dur_ns / _NS_PER_US,
        "args": args,
    }


def _origins(profile: Profile) -> dict[str, int]:
    origins: dict[str, int] = {}
    placed = [kernel for kernel in profile.kernels if kernel.positioned]
    for record in (*profile.windows, *placed, *profile.memcpys, *profile.activities):
        origins[record.clock] = min(origins.get(record.clock, record.start_ns), record.start_ns)
    return origins or {"host": 0}


def write_trace(profile: Profile, path: str | PathLike[str]) -> None:
    """Normalize each clock independently; no host/device synchronization is implied."""
    placed = [kernel for kernel in profile.kernels if kernel.positioned]
    origins = _origins(profile)
    processes = {clock: index for index, clock in enumerate(sorted(origins))}
    events: list[TraceEvent] = []
    for clock, pid in processes.items():
        label = f"{profile.device or 'krn'} ({clock}; independent origin)"
        events.append(_meta("process_name", 0, label, pid))
        events.extend(_meta("thread_name", tid, label, pid) for tid, label in _TRACKS.items())
    for window in profile.windows:
        events.append(
            _timed(
                window.name,
                _REGIONS,
                window.start_ns - origins[window.clock],
                window.end_ns - window.start_ns,
                processes[window.clock],
                {},
            )
        )
    for kernel in placed:
        events.append(
            _timed(
                kernel.name,
                _KERNELS,
                kernel.start_ns - origins[kernel.clock],
                kernel.duration_ns,
                processes[kernel.clock],
                {"grid": kernel.grid, "block": kernel.block, "registers": kernel.registers},
            )
        )
    for copy in profile.memcpys:
        events.append(
            _timed(
                copy.kind,
                _MEMCPYS,
                copy.start_ns - origins[copy.clock],
                copy.duration_ns,
                processes[copy.clock],
                {"bytes": copy.bytes_moved, "GB/s": round(copy.bandwidth_gbps, 1)},
            )
        )
    for activity in profile.activities:
        events.append(
            _timed(
                activity.name,
                _ACTIVITIES,
                activity.start_ns - origins[activity.clock],
                activity.duration_ns,
                processes[activity.clock],
                {"kind": activity.kind, "correlation": activity.correlation_id},
            )
        )
    if not profile.windows:
        elapsed_ns = 0
        for summary in profile.summaries:
            events.append(
                _timed(
                    summary.name,
                    _REGIONS,
                    elapsed_ns,
                    summary.wall_ms * 1e6,
                    0,
                    {"layout": "sequential summaries; timestamps unavailable"},
                )
            )
            elapsed_ns += int(summary.wall_ms * 1e6)
    Path(path).write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}), encoding="utf-8", newline="\n")
