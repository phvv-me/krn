"""Runtime bridge. Inactive call sites perform only a None comparison."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..compiler.ir import Kernel, ParamType, StructType, TensorType
from .work import counts

if TYPE_CHECKING:
    from .profiler import Profiler

session: Profiler | None = None


@dataclass(frozen=True, kw_only=True)
class LaunchRecord:
    """One dispatch; unknown dynamic work and device durations stay None.

    device_key: process-local identity of the runtime device object that dispatched it, None when unknown.
    region: the span path open on the enqueuing thread, the empty string outside any span.
    flops_by_dtype: arithmetic operations keyed by the dtype that computed them; a total below `flops`
        means some arithmetic had no known dtype.
    """

    name: str
    geometry: tuple[int, int, int, int]
    dtypes: tuple[str, ...]
    target: str
    stream: int
    host_enqueue_ns: int
    flops: int | None = None
    bytes_moved: int | None = None
    flops_by_dtype: dict[str, int] = field(default_factory=dict, hash=False)
    start_ns: int | None = None
    end_ns: int | None = None
    timing: str = "unknown"
    elapsed_ns: int | None = None
    device_key: int | None = None
    region: str = ""

    def __post_init__(self) -> None:
        if min(self.geometry[:3]) < 0 or self.geometry[3] <= 0:
            raise ValueError("grid dimensions must be nonnegative and block size positive")
        if self.host_enqueue_ns < 0 or any(v is not None and v < 0 for v in (self.flops, self.bytes_moved)):
            raise ValueError("timestamps and work counts must be nonnegative")
        if any(v < 0 for v in self.flops_by_dtype.values()):
            raise ValueError("work counts must be nonnegative")
        if (self.start_ns is None) != (self.end_ns is None):
            raise ValueError("execution timestamps must be supplied together")
        if self.start_ns is not None and self.end_ns is not None and self.end_ns < self.start_ns:
            raise ValueError("execution end precedes start")

    @property
    def duration_ns(self) -> int | None:
        return self.elapsed_ns if self.start_ns is None or self.end_ns is None else self.end_ns - self.start_ns


def signature(kind: ParamType) -> tuple[str, ...]:
    match kind:
        case TensorType(dtype=dtype):
            return (str(dtype),)
        case StructType(fields=fields):
            return tuple(dtype for _, field in fields for dtype in signature(field))
        case _:
            return (str(kind),)


def record(
    ir: Kernel,
    geometry: tuple[int, int, int, int],
    target: str,
    stream: int,
    *,
    device_key: int | None = None,
    region: str = "",
) -> LaunchRecord:
    work = counts(ir, geometry)
    return LaunchRecord(
        name=ir.name,
        geometry=geometry,
        dtypes=tuple(dtype for _, kind in ir.params for dtype in signature(kind)),
        target=target,
        stream=stream,
        host_enqueue_ns=time.perf_counter_ns(),
        flops=work.flops,
        bytes_moved=work.bytes_moved,
        flops_by_dtype=work.flops_by_dtype,
        device_key=device_key,
        region=region,
    )
