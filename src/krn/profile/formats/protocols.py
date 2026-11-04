# The trace-event shapes profiling writes, and the CUPTI activity records it reads, typed by
# shape since CUPTI's buffer callback hands over records its bindings leave untyped.

from typing import Protocol

# JSON values accepted by the Chrome/Perfetto trace-event writer.
type Json = str | int | float | bool | list[Json] | dict[str, Json] | None
type TraceEvent = dict[str, Json]


class TimedActivity(Protocol):
    """The fields every CUPTI record carries: the runtime `kind` and device-clock window.

    `name`/`cbid`/`correlation_id` are absent on some kinds, so collectors read them with
    `getattr`.
    """

    kind: int
    start: int
    end: int


class KernelActivity(TimedActivity, Protocol):
    """A CUPTI CONCURRENT_KERNEL record: launch shape plus the device-clock window."""

    name: str
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    static_shared_memory: int
    dynamic_shared_memory: int
    registers_per_thread: int


class MemcpyActivity(TimedActivity, Protocol):
    """A CUPTI MEMCPY record: direction code and device-clock window (`bytes` via getattr)."""

    copy_kind: int


class RawActivity(KernelActivity, MemcpyActivity, Protocol):
    """The opaque CUPTI record before kind dispatch, statically exposing every field.

    Only the subset valid for its runtime `kind` is meaningful; the superset lets a record
    reach its kind-specific reader without a cast.
    """
