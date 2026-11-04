"""Hardware counters through perf_event_open(2), called through the C library the loader opens.

perf_event_open has no libc wrapper, so the struct layout is spelled here and checked by a test. A hybrid Intel CPU
has one PMU per core type (cpu_core, cpu_atom): a generic event is opened on each and summed, so a thread that never
ran on a core type reads a true zero there. Counters follow the calling thread and the threads it creates while open.
"""

import ctypes
import errno
import os
import platform
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Self

from patos import value_dispatch

from ....compiler.errors import TargetError
from ....hw.libraries import Libraries, LibraryHandle
from ....hw.platforms import otherwise, system_key

PMU_ROOT = Path("/sys/bus/event_source/devices")
PARANOID = Path("/proc/sys/kernel/perf_event_paranoid")
CORE_PMUS = ("cpu_core", "cpu_atom", "cpu")
PERF_TYPE_HARDWARE = 0
ATTR_SIZE = 112  # PERF_ATTR_SIZE_VER5, the largest layout every kernel since 4.1 accepts
ERRNO_LOCATION = "__error" if sys.platform == "darwin" else "__errno_location"
SYSCALL_NUMBER = {"x86_64": 298, "aarch64": 241}
HYBRID_SHIFT = 32  # bits 32-39 of a hardware config name the core PMU on a hybrid CPU
FLAG_DISABLED = 1 << 0
FLAG_INHERIT = 1 << 1
FLAG_EXCLUDE_KERNEL = 1 << 5
FLAG_EXCLUDE_HV = 1 << 6
FORMAT_ENABLED_RUNNING = (1 << 0) | (1 << 1)
FLAG_FD_CLOEXEC = 1 << 3
IOC_ENABLE, IOC_DISABLE, IOC_RESET = 0x2400, 0x2401, 0x2403
GENERIC_EVENTS: Mapping[str, int] = {
    "cycles": 0,
    "instructions": 1,
    "cache_references": 2,
    "cache_misses": 3,
    "branches": 4,
    "branch_misses": 5,
}
DERIVED: Mapping[str, tuple[str, str]] = {
    "ipc": ("instructions", "cycles"),
    "cache_miss_rate": ("cache_misses", "cache_references"),
    "branch_miss_rate": ("branch_misses", "branches"),
}


class _Attr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
    ]


def _errno() -> int:
    return getattr(_libc(), ERRNO_LOCATION)().contents.value


@cache
def _libc() -> LibraryHandle:
    """The C library the loader opened; errno is read through its `__errno_location`, as ctypes' use_errno would."""
    handle = Libraries.load("libsystem").handle
    getattr(handle, ERRNO_LOCATION).restype = ctypes.POINTER(ctypes.c_int)
    handle.syscall.restype = ctypes.c_long
    handle.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong]
    handle.ioctl.restype = ctypes.c_int
    return handle


def paranoid() -> int | None:
    """The kernel's perf_event_paranoid level, None when this kernel does not expose it."""
    try:
        return int(PARANOID.read_text())
    except OSError, ValueError:
        return None


def pmu_type(name: str, root: Path = PMU_ROOT) -> int | None:
    """The dynamic PMU type of `name`, None when the kernel has no such PMU."""
    try:
        return int((root / name / "type").read_text())
    except OSError, ValueError:
        return None


def core_pmus(root: Path = PMU_ROOT) -> dict[str, int]:
    """The CPU core PMUs present: cpu_core and cpu_atom on a hybrid CPU, cpu otherwise."""
    found = {name: pmu_type(name, root) for name in CORE_PMUS}
    return {name: kind for name, kind in found.items() if kind is not None}


@dataclass(frozen=True, kw_only=True)
class Reading:
    """A counter's value with the time it was enabled and the time it held a hardware slot.

    `scaled` extrapolates a multiplexed counter; one that never held a slot keeps its raw zero.
    """

    value: int
    enabled: int
    running: int

    @property
    def scaled(self) -> float:
        if 0 < self.running < self.enabled:
            return self.value * self.enabled / self.running
        return float(self.value)


def refusal(number: int, feature: str, *, system_wide: bool) -> TargetError:
    """The explained error for a failed perf_event_open, naming the sysctl or capability that would allow it.

    A system-wide counter (pid -1) needs perf_event_paranoid 0 or CAP_PERFMON; an own-thread counter needs 2 or lower.
    """
    if number in (errno.EACCES, errno.EPERM):
        scope = "system-wide" if system_wide else "own-thread"
        remedy = (
            "lower kernel.perf_event_paranoid to 0, or grant CAP_PERFMON to the Python binary"
            if system_wide
            else "lower kernel.perf_event_paranoid to 2 or below, or grant CAP_PERFMON"
        )
        return TargetError(
            feature,
            "CPU",
            f"{scope} perf_event_open was refused ({os.strerror(number)}) at perf_event_paranoid={paranoid()}",
            remedy,
        )
    if number in (errno.ENOENT, errno.ENODEV):
        return TargetError(feature, "CPU", "the PMU does not offer this event", "count a generic event instead")
    return TargetError(
        feature,
        "CPU",
        f"perf_event_open failed with {os.strerror(number)} (errno {number})",
        "check the event encoding and the kernel's perf support",
    )


@value_dispatch(kind="system")
def open_raw(kind: int, config: int, *, pid: int, cpu: int, feature: str) -> int:
    """Open one disabled counter and return its file descriptor; every system but Linux refuses."""
    raise TargetError(
        feature,
        "CPU",
        "perf_event_open is a Linux interface, and macOS offers no unprivileged counter API",
        "profile on Linux, or use the wall-clock launch timing that the CPU profiler records",
    )


otherwise(open_raw, "Darwin")


@open_raw.register("Linux")
def _open_linux(kind: int, config: int, *, pid: int, cpu: int, feature: str) -> int:
    number = SYSCALL_NUMBER.get(platform.machine())
    if number is None:
        raise TargetError(
            feature,
            "CPU",
            f"no perf_event_open syscall number for {platform.machine()}",
            "profile on x86_64 or aarch64",
        )
    attr = _Attr(type=kind, size=ATTR_SIZE, config=config, read_format=FORMAT_ENABLED_RUNNING)
    attr.flags = FLAG_DISABLED | FLAG_INHERIT | FLAG_EXCLUDE_KERNEL | FLAG_EXCLUDE_HV
    descriptor = _libc().syscall(number, ctypes.byref(attr), pid, cpu, -1, FLAG_FD_CLOEXEC)
    if descriptor < 0:
        raise refusal(_errno(), feature, system_wide=pid == -1)
    return descriptor


@dataclass(frozen=True, kw_only=True)
class Counter:
    """One open event on one PMU, read through its file descriptor."""

    fd: int
    pmu: str
    event: str

    def read(self) -> Reading:
        value, enabled, running = struct.unpack("<3Q", os.read(self.fd, struct.calcsize("<3Q")))
        return Reading(value=value, enabled=enabled, running=running)

    def control(self, request: int) -> None:
        if _libc().ioctl(self.fd, request, 0) < 0:
            raise OSError(_errno(), os.strerror(_errno()), self.event)

    def close(self) -> None:
        os.close(self.fd)


def open_hardware(event: str, pmu: str, kind: int, *, hybrid: bool, pid: int, cpu: int) -> Counter:
    """Open one generic event on one core PMU; hybrid CPUs name the PMU in the config's high bits."""
    config = GENERIC_EVENTS[event] | (kind << HYBRID_SHIFT if hybrid else 0)
    fd = open_raw(
        PERF_TYPE_HARDWARE, config, pid=pid, cpu=cpu, feature=f"{event} counter on {pmu}", system=system_key()
    )
    return Counter(fd=fd, pmu=pmu, event=event)


class HardwareCounters:
    """Generic events counted for the calling thread and the threads it creates while open.

    Enter around a launch: `read()` sums every core PMU per event, and `metrics()` derives ipc, cache_miss_rate and
    branch_miss_rate, omitting any whose denominator counted zero. Counters open only on entry, so an inactive
    profiler allocates nothing here. A refused counter closes the ones already open and raises.

    events: generic names from GENERIC_EVENTS. pmus: core PMU name to type; read from sysfs when omitted.
    """

    def __init__(self, events: Sequence[str] = tuple(GENERIC_EVENTS), *, pmus: Mapping[str, int] | None = None) -> None:
        unknown = [event for event in events if event not in GENERIC_EVENTS]
        if unknown:
            raise ValueError(f"unknown generic counter events {unknown}; choose from {tuple(GENERIC_EVENTS)}")
        self.events = tuple(events)
        self.pmus = core_pmus() if pmus is None else dict(pmus)
        self._counters: list[Counter] = []

    def __enter__(self) -> Self:
        if not self.pmus:
            raise TargetError(
                "hardware counters",
                "CPU",
                "no CPU core PMU is under /sys/bus/event_source/devices (macOS and VMs without a PMU have none)",
                "profile on Linux with a PMU, or use the wall-clock launch timing that the CPU profiler records",
            )
        hybrid = len(self.pmus) > 1
        try:
            for pmu, kind in self.pmus.items():
                for event in self.events:
                    self._counters.append(open_hardware(event, pmu, kind, hybrid=hybrid, pid=0, cpu=-1))
            for counter in self._counters:
                counter.control(IOC_RESET)
                counter.control(IOC_ENABLE)
        except TargetError, OSError:
            self._release()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            for counter in self._counters:
                counter.control(IOC_DISABLE)
        finally:
            self._release()

    def _release(self) -> None:
        for counter in self._counters:
            counter.close()
        self._counters.clear()

    def read(self) -> dict[str, float]:
        """Each event's count summed over the core PMUs, scaled where the kernel multiplexed it."""
        totals = dict.fromkeys(self.events, 0.0)
        for counter in self._counters:
            totals[counter.event] += counter.read().scaled
        return totals

    @staticmethod
    def metrics(counts: Mapping[str, float]) -> dict[str, float]:
        """Ratios of counts that were measured and whose denominator is nonzero; others are absent."""
        return {
            name: counts[numerator] / counts[denominator]
            for name, (numerator, denominator) in DERIVED.items()
            if numerator in counts and denominator in counts and counts[denominator] > 0
        }
