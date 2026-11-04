"""Process-cached inventory; live readings remain explicit."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import cache, cached_property, partial
from typing import TYPE_CHECKING

from patos import value_dispatch

from krn.facts.targets import Target
from krn.runtime.settings import current_settings

from .devices.amd import AMDProbe
from .devices.apple import AppleProbe
from .devices.base import Device, Probe, ReadingDevice, Telemetry, same_device
from .devices.generic import DRMProbe
from .devices.intel import IntelProbe
from .devices.nvidia import NvidiaProbe
from .devices.sysman import SysmanProbe
from .devices.webgpu import WGPUProbe
from .enums import DeviceKind, Vendor
from .exceptions import HardwareError, ProbeError
from .host import CPU, OS, Limits, Memory, PythonRuntime, Volume
from .insights import RULES, Insight
from .libraries import available_libraries
from .platforms import system_key
from .topology import Topology, detect_topology

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)

_RUNTIME_MASKS = {"HIP_VISIBLE_DEVICES": Vendor.AMD, "ZE_AFFINITY_MASK": Vendor.INTEL}


def _combined(primary: Callable[[], Telemetry], fallback: Callable[[], Telemetry]) -> Telemetry:
    first, second = primary(), fallback()
    values = {
        f.name: getattr(first, f.name) if getattr(first, f.name) is not None else getattr(second, f.name)
        for f in fields(Telemetry)
        if f.name != "sources"
    }
    sources = {
        name: first.sources.get(name, "") if getattr(first, name) is not None else second.sources.get(name, "")
        for name, value in values.items()
        if value is not None
    }
    return Telemetry(**values, sources=sources)


def _first[T](primary: T | None, fallback: T | None) -> T | None:
    return primary if primary is not None else fallback


def _fill[T: DataclassInstance](primary: T, fallback: T) -> T:
    """A fact group with each unknown field taken from the fallback group."""
    return replace(
        primary, **{f.name: _first(getattr(primary, f.name), getattr(fallback, f.name)) for f in fields(primary)}
    )


def _merged(primary: ReadingDevice, fallback: ReadingDevice) -> ReadingDevice:
    """The same physical device seen by two sources: primary readings, with unknown facts filled from the fallback."""
    return replace(
        primary,
        uuid=_first(primary.uuid, fallback.uuid),
        pci_bus_id=_first(primary.pci_bus_id, fallback.pci_bus_id),
        product_id=_first(primary.product_id, fallback.product_id),
        driver=_first(primary.driver, fallback.driver),
        architecture=_first(primary.architecture, fallback.architecture),
        backend=_first(primary.backend, fallback.backend),
        description=_first(primary.description, fallback.description),
        boot_display=_first(primary.boot_display, fallback.boot_display),
        coherence_facts=_first(primary.coherence_facts, fallback.coherence_facts),
        memory_facts=_fill(primary.memory_facts, fallback.memory_facts),
        limit_facts=_fill(primary.limit_facts, fallback.limit_facts),
        pci=_fill(primary.pci, fallback.pci),
        telemetry_reader=partial(_combined, primary.telemetry, fallback.telemetry),
        process_reader=primary.processes if primary.process_reader else fallback.processes,
    )


@value_dispatch(kind="system")
def discovery_order() -> tuple[Probe[ReadingDevice], ...]:
    raise ProbeError(
        f"Unsupported system; known keys: {tuple(discovery_order.registry)}", "Use a supported operating system"
    )


@discovery_order.register("Darwin")
def _mac_probes() -> tuple[Probe[ReadingDevice], ...]:
    return AppleProbe(), IntelProbe(), WGPUProbe()


@discovery_order.register("Linux")
def _linux_probes() -> tuple[Probe[ReadingDevice], ...]:
    return NvidiaProbe(), AMDProbe(), IntelProbe(), SysmanProbe(), DRMProbe(), WGPUProbe()


@dataclass(frozen=True, kw_only=True)
class System:
    os: OS
    cpu: CPU
    memory: Memory
    volumes: tuple[Volume, ...]
    python: PythonRuntime
    limits: Limits
    devices: tuple[Device, ...] = ()
    probe_failures: tuple[str, ...] = ()

    @cached_property
    def topology(self) -> Topology:
        return detect_topology(system=self.os.system)

    @cached_property
    def libraries(self) -> frozenset[str]:
        """Vendor kernel libraries available to adapters on this host."""
        return available_libraries(self.os.system)

    @classmethod
    @cache
    def detect(cls) -> System:
        limits = Limits.detect()
        cpu = CPU.detect()
        memory = Memory.detect(limits)
        devices: list[ReadingDevice] = []
        failures = []
        for probe in discovery_order(system=system_key()):
            try:
                status = probe.available()
                if not status:
                    failures.append(status.reason or f"{type(probe).__name__}: unavailable")
                    continue
                for fallback in probe.discover():
                    match = next((i for i, d in enumerate(devices) if same_device(d, fallback)), None)
                    if match is None:
                        devices.append(fallback)
                    else:
                        devices[match] = _merged(devices[match], fallback)
            except (RuntimeError, OSError, ValueError, AttributeError) as error:
                failures.append(f"{type(probe).__name__}: {error}")
                logger.debug("Hardware source unavailable: %s", error)
        return cls(
            os=OS.detect(),
            cpu=cpu,
            memory=memory,
            volumes=Volume.detect_all(),
            python=PythonRuntime.detect(),
            limits=limits,
            devices=tuple(devices),
            probe_failures=tuple(failures),
        )

    @cached_property
    def gpus(self) -> tuple[Device, ...]:
        return tuple(d for d in self.devices if d.kind == DeviceKind.GPU)

    @cached_property
    def by_vendor(self) -> dict[Vendor, tuple[Device, ...]]:
        return {vendor: tuple(d for d in self.devices if d.vendor == vendor) for vendor in Vendor}

    @cached_property
    def visible_devices(self) -> tuple[Device, ...]:
        """Devices the probes expose; CUDA visibility is already in the probe, so only the other masks apply here."""
        result = tuple(d for d in self.devices if d.visible)
        for key, mask in self.limits.masks:
            if key not in _RUNTIME_MASKS:
                continue
            vendor = _RUNTIME_MASKS[key]
            tokens = mask.split(",")
            if vendor == Vendor.INTEL:
                tokens = [token.partition(".")[0] for token in tokens]
            result = tuple(
                d
                for d in result
                if d.vendor != vendor
                or str(d.index) in tokens
                or (d.uuid is not None and any(d.uuid.startswith(token) for token in tokens if token))
            )
        return result

    @cached_property
    def default_gpu(self) -> Device | None:
        """The runtime's default device: the visible GPU with the lowest runtime ordinal, then inventory order."""
        candidates = [d for d in self.visible_devices if d.kind == DeviceKind.GPU and d.target is not None]
        ordered = sorted(candidates, key=lambda d: (d.ordinal is None, d.ordinal or 0))
        return ordered[0] if ordered else None

    @cached_property
    def targets(self) -> tuple[Target, ...]:
        return tuple(d.target for d in self.gpus if d.target is not None)

    @cached_property
    def total_device_memory(self) -> int:
        """Sum device budgets; unified devices can overlap host memory."""
        return sum(d.memory.total or 0 for d in self.devices)

    @property
    def target(self) -> Target:
        """Honor the explicit compiler override without changing the inventory."""
        if override := current_settings().run.target:
            return Target.of(override)
        if self.default_gpu is not None and self.default_gpu.target is not None:
            return self.default_gpu.target
        if self.cpu.architecture not in {"x86_64", "AMD64"}:
            raise HardwareError(
                f"No supported compiler target for {self.cpu.architecture}",
                "Install a supported GPU backend or set KRN_TARGET for cross-compilation",
            )
        features = self.cpu.features
        return Target.of("spr" if "amx_tile" in features else "avx512" if "avx512f" in features else "avx2")

    def device(self, index_or_uuid: int | str) -> Device:
        matches = tuple(
            d
            for d in self.devices
            if (d.index == index_or_uuid if isinstance(index_or_uuid, int) else d.uuid == index_or_uuid)
        )
        if len(matches) != 1:
            raise HardwareError(f"Device {index_or_uuid!r} has {len(matches)} matches", "Select a unique device UUID")
        return matches[0]

    def require(self, vendor: Vendor) -> tuple[Device, ...]:
        devices = self.by_vendor[vendor]
        if not devices:
            raise HardwareError(f"No {vendor.name} device detected", "Check the driver and device visibility masks")
        return devices

    def insights(self) -> tuple[Insight, ...]:
        return tuple(insight for rule in RULES for insight in rule(self))

    def describe(self) -> str:
        lines = [
            f"{self.os.distribution} {self.os.distribution_version} ({self.os.architecture})",
            f"CPU: {self.cpu.name}; {self.cpu.logical_cores} logical cores; {self.limits.usable_cpus:g} usable",
            f"Memory: {self.memory.total} bytes; page size {self.memory.page_size}",
            f"Python: {self.python.version} {self.python.implementation}; GIL {self.python.gil_enabled}",
        ]
        lines.extend(
            f"GPU {d.vendor.name}:{d.index}: {d.name}; {d.architecture or 'unknown architecture'}; "
            f"{d.memory.total} bytes"
            for d in self.gpus
        )
        lines.extend(f"Volume {v.path}: {v.free}/{v.total} bytes free" for v in self.volumes)
        return "\n".join(lines)
