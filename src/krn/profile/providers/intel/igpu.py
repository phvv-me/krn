"""Device-level health signals of an Intel integrated GPU, read from i915 sysfs and, when granted, its PMU.

krn cannot launch kernels on an Intel GPU yet, so nothing here is kernel-level: a device sampler reads these signals
while spans are open. The GT awake share comes from rc6 residency, so it is an upper bound on engine busy time, not a
busy percentage. The i915 PMU (engine busy, actual frequency) needs perf_event_paranoid 0 or CAP_PERFMON for a
system-wide counter and raises an explained TargetError otherwise.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ....compiler.errors import TargetError
from ....hw import native
from ....hw.devices.base import DeviceLimits, DeviceMemory, Process, ReadingDevice, Telemetry
from ....hw.enums import DeviceKind, Vendor
from ....hw.platforms import system_key
from ..cpu.perf import PMU_ROOT, Counter, open_raw, pmu_type

DRM_ROOT = Path("/sys/class/drm")
INTEL_PCI_VENDOR = 0x8086
CONFIG = re.compile(r"config=(0x[0-9a-fA-F]+|\d+)")


def igpu_cards(root: Path = DRM_ROOT) -> tuple[Path, ...]:
    """Intel GPU card directories that expose a GT, the i915 and xe layout alike."""
    return tuple(
        card
        for card in native.paths(root, "card*")
        if card.name.removeprefix("card").isdigit()
        and native.read_int(card / "device" / "vendor") == INTEL_PCI_VENDOR
        and (card / "gt" / "gt0").is_dir()
    )


@dataclass(frozen=True, kw_only=True)
class Frequencies:
    """Actual and requested GT frequency with the maximum the driver allows, in MHz."""

    actual_mhz: float | None
    requested_mhz: float | None
    max_mhz: float | None


class RC6Meter:
    """Awake share of the GT from the cumulative rc6 residency, measured since the previous reading.

    The first reading has no interval and reports None. Each later one reports 100 minus the share of the elapsed
    time spent in rc6, clamped to 0 to 100.
    """

    def __init__(self, gt: Path, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.gt = gt
        self.clock = clock
        self._previous: tuple[int, int] | None = None

    def awake_percent(self) -> float | None:
        residency_ms = native.read_int(self.gt / "rc6_residency_ms")
        if residency_ms is None:
            return None
        now = self.clock()
        previous, self._previous = self._previous, (residency_ms, now)
        if previous is None or now <= previous[1]:
            return None
        resident = (residency_ms - previous[0]) * 1e6
        elapsed = now - previous[1]
        return min(100.0, max(0.0, 100.0 * (1.0 - resident / elapsed)))


def frequencies(card: Path) -> Frequencies:
    """i915 exports its frequencies beside the gt directory, at card level."""
    return Frequencies(
        actual_mhz=_mhz(card / "gt_act_freq_mhz"),
        requested_mhz=_mhz(card / "gt_cur_freq_mhz"),
        max_mhz=_mhz(card / "gt_RP0_freq_mhz"),
    )


def _mhz(path: Path) -> float | None:
    value = native.read_int(path)
    return None if value is None else float(value)


def _reading(card: Path, meter: RC6Meter) -> Telemetry:
    current = frequencies(card)
    return Telemetry(
        clock_mhz=current.actual_mhz,
        max_clock_mhz=current.max_mhz,
        utilization_percent=meter.awake_percent(),
    ).attributed("i915-sysfs")


def _processes(bus: str) -> tuple[Process, ...]:
    return tuple(Process(pid=pid, memory_bytes=memory) for pid, memory in native.drm_processes(bus))


def igpu_devices(root: Path = DRM_ROOT, clock: Callable[[], int] = time.monotonic_ns) -> tuple[ReadingDevice, ...]:
    """One device per Intel GPU card, with its sysfs telemetry and the DRM clients that use it."""
    devices = []
    for card in igpu_cards(root):
        gt = card / "gt" / "gt0"
        bus = native.pci_id(card / "device")
        devices.append(
            ReadingDevice(
                vendor=Vendor.INTEL,
                kind=DeviceKind.GPU,
                index=int(card.name.removeprefix("card")),
                name=f"Intel GPU {native.read_text(card / 'device' / 'device') or 'unknown'}",
                source="i915-sysfs",
                driver="i915",
                pci_bus_id=bus,
                memory_facts=DeviceMemory(unified=True),
                limit_facts=DeviceLimits(max_clock_mhz=frequencies(card).max_mhz),
                telemetry_reader=partial(_reading, card, RC6Meter(gt, clock)),
                process_reader=partial(_processes, bus),
            )
        )
    return tuple(devices)


@dataclass(frozen=True, kw_only=True)
class I915Event:
    """One i915 PMU event by name and the config the kernel exports for it."""

    name: str
    config: int


def i915_event(name: str, root: Path = PMU_ROOT) -> I915Event:
    """The named i915 event; the kernel exports each as a `config=` line under the PMU's events directory."""
    match = CONFIG.search(native.read_text(root / "i915" / "events" / name) or "")
    if match is None:
        raise TargetError(
            f"i915 {name}",
            "Intel GPU",
            "the i915 PMU does not export this event on this kernel",
            "read gt_act_freq_mhz or rc6_residency_ms from sysfs instead",
        )
    return I915Event(name=name, config=int(match.group(1), 0))


def open_i915(event: I915Event, *, cpu: int = 0, root: Path = PMU_ROOT) -> Counter:
    """Open an i915 event system-wide on `cpu`, as the PMU requires: perf_event_paranoid 0 or CAP_PERFMON."""
    kind = pmu_type("i915", root)
    if kind is None:
        raise TargetError(
            f"i915 {event.name}", "Intel GPU", "the kernel has no i915 PMU", "load the i915 driver with perf support"
        )
    fd = open_raw(kind, event.config, pid=-1, cpu=cpu, feature=f"i915 {event.name}", system=system_key())
    return Counter(fd=fd, pmu="i915", event=event.name)


def busy_percent(first_ns: float, second_ns: float, elapsed_ns: float) -> float:
    """Engine busy share between two readings of a busy-time counter in nanoseconds, clamped to 0 to 100."""
    if elapsed_ns <= 0:
        raise ValueError("elapsed time must be positive")
    return min(100.0, max(0.0, 100.0 * (second_ns - first_ns) / elapsed_ns))
