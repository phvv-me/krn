"""DRM discovery remains useful without a vendor runtime."""

import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from .. import native
from ..enums import Vendor
from .base import Availability, DeviceLimits, DeviceMemory, PCILink, Probe, Process, ReadingDevice, Telemetry


def _card_devices(root: str | Path) -> tuple[Path, ...]:
    """Device links of GPU cards; connector entries such as card1-DP-1 also link to a device but are not cards."""
    cards = (card for card in native.paths(root, "card*") if card.name.removeprefix("card").isdigit())
    return tuple(card / "device" for card in cards if (card / "device").exists())


def _reading(path: Path) -> Telemetry:
    def sensor(pattern: str, scale: float) -> float | None:
        values = native.paths(path / "hwmon", pattern)
        value = native.read_int(values[0]) if values else None
        return value / scale if value is not None else None

    pwm = sensor("hwmon*/pwm1", 1)
    pwm_max = sensor("hwmon*/pwm1_max", 1)
    clock = sensor("hwmon*/freq1_input", 1e6)
    reading = Telemetry(
        temperature_c=sensor("hwmon*/temp1_input", 1000),
        slowdown_temperature_c=sensor("hwmon*/temp1_crit", 1000),
        fan_percent=100 * pwm / (pwm_max or 255) if pwm is not None else None,
        power_w=sensor("hwmon*/power1_average", 1e6),
        power_limit_w=sensor("hwmon*/power1_cap", 1e6),
        clock_mhz=clock,
        memory_used_bytes=native.read_int(path / "mem_info_vram_used"),
        utilization_percent=native.read_int(path / "gpu_busy_percent")
        if native.read_int(path / "vendor") != Vendor.INTEL
        else None,
    ).attributed("hwmon/drm")
    if clock is None and (gt_clock := native.read_int(path.parent / "gt_act_freq_mhz")) is not None:
        reading = replace(reading, clock_mhz=gt_clock, sources={**reading.sources, "clock_mhz": "drm/gt"})
    if psutil is None:
        return reading
    driver = (path / "driver").resolve().name
    try:
        temperatures = psutil.sensors_temperatures().get(driver, [])
        fans = psutil.sensors_fans().get(driver, [])
    except AttributeError, OSError, NotImplementedError:
        return reading
    values = {}
    if temperatures:
        values["temperature_c"] = temperatures[0].current
    if fans:
        values["fan_rpm"] = fans[0].current
    return replace(
        reading,
        temperature_c=values.get("temperature_c", reading.temperature_c),
        fan_rpm=values.get("fan_rpm", reading.fan_rpm),
        sources={**reading.sources, **dict.fromkeys(values, "psutil")},
    )


def _processes(path: Path) -> tuple[Process, ...]:
    return tuple(Process(pid=pid, memory_bytes=memory) for pid, memory in native.drm_processes(native.pci_id(path)))


@dataclass(frozen=True, kw_only=True)
class GenericGPU(ReadingDevice):
    """A GPU whose driver exposes only DRM files."""

    energy_path: Path | None = None

    def sample(self, *, interval: float) -> Telemetry:
        reading = super().sample(interval=interval)
        if self.energy_path is None:
            return reading
        first = native.read_int(self.energy_path)
        started = time.monotonic()
        time.sleep(interval)
        second = native.read_int(self.energy_path)
        elapsed = time.monotonic() - started
        if first is None or second is None or second < first or elapsed <= 0:
            return reading
        return replace(
            reading, power_w=(second - first) / 1e6 / elapsed, sources={**reading.sources, "power_w": "hwmon.energy"}
        )


class DRMProbe(Probe[GenericGPU]):
    def __init__(self, root: str | Path = "/sys/class/drm") -> None:
        self.root = root

    def available(self) -> Availability:
        found = bool(_card_devices(self.root))
        return Availability(available=found, reason=None if found else "drm: no readable devices")

    def discover(self) -> tuple[GenericGPU, ...]:
        devices = []
        for path in _card_devices(self.root):
            vendor_id = native.read_int(path / "vendor")
            vendor = Vendor(vendor_id) if vendor_id in Vendor else Vendor.UNKNOWN
            index = int(path.parent.name.removeprefix("card"))
            identity = native.read_text(path / "device") or "unknown"
            speed = (native.read_text(path / "current_link_speed") or "").split()
            generation = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
            node = native.read_int(path / "numa_node")
            boot = native.read_int(path / "boot_vga")
            energy = native.paths(path / "hwmon", "hwmon*/energy1_input")
            driver = path / "driver"
            devices.append(
                GenericGPU(
                    vendor=vendor,
                    index=index,
                    name=f"{vendor.name} GPU {identity}",
                    pci_bus_id=native.pci_id(path),
                    product_id=native.read_int(path / "device"),
                    driver=driver.resolve().name if driver.exists() else None,
                    source="drm",
                    boot_display=bool(boot) if boot is not None else None,
                    memory_facts=DeviceMemory(total=native.read_int(path / "mem_info_vram_total")),
                    limit_facts=DeviceLimits(max_clock_mhz=native.read_int(path.parent / "gt_RP0_freq_mhz")),
                    pci=PCILink(
                        generation=generation.get(speed[0]) if speed else None,
                        width=native.read_int(path / "current_link_width"),
                        numa_node=node if node is not None and node >= 0 else None,
                    ),
                    telemetry_reader=partial(_reading, path),
                    process_reader=partial(_processes, path),
                    energy_path=energy[0] if energy else None,
                )
            )
        return tuple(devices)
