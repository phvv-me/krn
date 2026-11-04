import time
from types import TracebackType
from typing import TYPE_CHECKING

import psutil

from ...hw.enums import DeviceKind
from ..result.models import known_max

if TYPE_CHECKING:
    from ...hw.system import System

_NANOSECONDS_PER_SECOND = 1e9
_BYTES_PER_GIB = 2**30


class Meter:
    """Times a region and tracks peak host and GPU memory, in gibibytes, across samples.

    Memory is sampled at enter, at every explicit `sample()`, and at exit; no background
    thread runs; the caller passes the `Machine()` it reads. A GPU sample is the sum over every
    GPU, and None when any of them reads unknown, so an unread card never looks like free memory.
    """

    @property
    def host_delta_gb(self) -> float:
        """Host memory growth from the first to the last sample."""
        return self.host_used_gb[-1] - self.host_used_gb[0] if self.host_used_gb else 0.0

    @property
    def peak_gpu_gb(self) -> float | None:
        """Highest total GPU memory in use across known samples, None when no sample was known."""
        return known_max(self.gpu_used_gb)

    @property
    def peak_host_gb(self) -> float:
        return max(self.host_used_gb, default=0.0)

    def __enter__(self) -> Meter:
        self.start_ns = time.perf_counter_ns()
        self.sample()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.sample()
        self.elapsed_s = (time.perf_counter_ns() - self.start_ns) / _NANOSECONDS_PER_SECOND

    def __init__(self, machine: System) -> None:
        self.machine = machine
        self.host_used_gb: list[float] = []
        self.gpu_used_gb: list[float | None] = []
        self.start_ns = 0
        self.elapsed_s = 0.0

    def sample(self) -> None:
        """Capture one host reading and one summed GPU memory reading from the live machine."""
        memory = psutil.virtual_memory()
        self.host_used_gb.append((memory.total - memory.available) / _BYTES_PER_GIB)
        readings = [gpu.snapshot().memory_used_bytes for gpu in self.machine.devices if gpu.kind is DeviceKind.GPU]
        known = [reading for reading in readings if reading is not None]
        self.gpu_used_gb.append(sum(known) / _BYTES_PER_GIB if readings and len(known) == len(readings) else None)
