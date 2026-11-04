"""Common static facts and fresh device readings."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from math import isfinite
from types import MappingProxyType

from krn.facts.targets import Dtype, Feature, Target

from ..enums import DeviceKind, Vendor
from ..peaks import PeakRates, resolve

_BRANDS = (("apple", Vendor.APPLE), ("nvidia", Vendor.NVIDIA), ("amd", Vendor.AMD), ("intel", Vendor.INTEL))


def canonical_pci(address: str) -> str:
    """CUDA and NVML may pad the PCI domain to different widths."""
    parts = address.rsplit(":", 2)
    if len(parts) != 3:
        return address.lower()
    return f"{int(parts[0], 16):04x}:{parts[1].lower()}:{parts[2].lower()}"


def brand_vendor(text: str) -> Vendor:
    """Vendor named by a device or CPU string; UNKNOWN when no brand appears."""
    lowered = text.lower()
    return next((vendor for brand, vendor in _BRANDS if brand in lowered), Vendor.UNKNOWN)


def _name_head(name: str) -> str:
    """Product name without trademarks and suffixes such as '/PCIe/SSE2' or '(RPL-S)'."""
    return re.split(r"[/(]", re.sub(r"\((?:R|TM|C)\)", "", name, flags=re.IGNORECASE))[0].strip().lower()


@dataclass(frozen=True, kw_only=True)
class DeviceMemory:
    total: int | None = None
    unified: bool | None = None
    recommended_working_set: int | None = None
    max_buffer_length: int | None = None

    def __post_init__(self) -> None:
        if self.total is not None and self.total < 0:
            raise ValueError("memory total must be non-negative")


@dataclass(frozen=True, kw_only=True)
class DeviceLimits:
    compute_units: int | None = None
    subgroup_size: int | None = None
    subgroup_sizes: tuple[int, ...] | None = None
    max_threads_per_block: int | None = None
    shared_memory_per_block: int | None = None
    registers_per_block: int | None = None
    max_clock_mhz: float | None = None


@dataclass(frozen=True, kw_only=True)
class PCILink:
    generation: int | None = None
    width: int | None = None
    numa_node: int | None = None
    cpu_affinity: tuple[int, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class NVLink:
    index: int
    active: bool
    version: tuple[int, int]


@dataclass(frozen=True, kw_only=True)
class Telemetry:
    temperature_c: float | None = None
    slowdown_temperature_c: float | None = None
    fan_percent: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    clock_mhz: float | None = None
    max_clock_mhz: float | None = None
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None
    memory_utilization_percent: float | None = None
    throttle_reasons: tuple[str, ...] | None = None
    performance_state: int | None = None
    fan_rpm: float | None = None
    sources: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        self.__dict__["sources"] = MappingProxyType(dict(self.sources))

    def attributed(self, source: str) -> Telemetry:
        return replace(
            self,
            sources={f.name: source for f in fields(self) if f.name != "sources" and getattr(self, f.name) is not None},
        )


@dataclass(frozen=True, kw_only=True)
class Process:
    pid: int
    name: str | None = None
    memory_bytes: int | None = None


@dataclass(frozen=True, kw_only=True)
class Device(ABC):
    vendor: Vendor
    kind: DeviceKind = DeviceKind.GPU
    index: int
    name: str
    uuid: str | None = None
    pci_bus_id: str | None = None
    architecture: str | None = None
    source: str
    backend: str | None = None
    description: str | None = None
    boot_display: bool | None = None
    memory_facts: DeviceMemory = DeviceMemory()
    limit_facts: DeviceLimits = DeviceLimits()
    capabilities: frozenset[str] = frozenset()
    pci: PCILink = PCILink()
    peak_facts: PeakRates = field(default_factory=PeakRates)
    coherence_facts: bool | None = None
    # PCI product id (the device half of vendor:product), kernel or adapter reported.
    product_id: int | None = None
    # Kernel driver bound to the device, such as i915 or nvidia.
    driver: str | None = None
    # False when the owning runtime hides the physical device from this process.
    visible: bool = True
    # Ordinal the owning runtime uses among the devices it exposes; CUDA ordinal 0 is the default device.
    ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("device index must be non-negative")
        if self.pci_bus_id is not None:
            self.__dict__["pci_bus_id"] = canonical_pci(self.pci_bus_id)

    @cached_property
    def memory(self) -> DeviceMemory:
        return self.memory_facts

    @cached_property
    def limits(self) -> DeviceLimits:
        return self.limit_facts

    @cached_property
    def arch_key(self) -> str:
        """Stable dot-free compiler architecture, or normalized probe identity."""
        target = self.target
        return (target.arch if target is not None else self.architecture or "unknown").lower().replace(".", "_")

    @cached_property
    def coherent(self) -> bool | None:
        """Whether host memory is a cache-coherent peer; unknown is not false."""
        if self.coherence_facts is not None:
            return self.coherence_facts
        if self.memory.unified is False:
            return False
        return None

    @cached_property
    def peak(self) -> PeakRates:
        return resolve(self.arch_key, self.name, self.limits.compute_units, self.limits.max_clock_mhz, self.peak_facts)

    @cached_property
    def features(self) -> frozenset[str]:
        target = self.target
        if target is None:
            return self.capabilities
        features: set[str] = {feature.value.replace("_", "-") for feature in target.features}
        if Feature.TENSOR_CORE in target.features:
            features.add("tensor-cores")
        names = {Dtype.F16: "fp16", Dtype.BF16: "bf16", Dtype.I8: "int8"}
        features.update(names[dtype] for dtype in target.storage if dtype in names)
        if any(dtype.value.startswith("f8") for dtype in target.storage):
            features.add("fp8")
        return self.capabilities | frozenset(features)

    @cached_property
    def link(self) -> PCILink:
        return self.pci

    @cached_property
    def target(self) -> Target | None:
        if self.architecture is None:
            return None
        try:
            return Target.of(self.architecture)
        except ValueError, KeyError:
            return None

    @abstractmethod
    def telemetry(self) -> Telemetry:
        raise NotImplementedError

    @abstractmethod
    def processes(self) -> tuple[Process, ...] | None:
        raise NotImplementedError

    @abstractmethod
    def nvlinks(self) -> tuple[NVLink, ...] | None:
        raise NotImplementedError

    def sample(self, *, interval: float) -> Telemetry:
        """Read telemetry; rate-based sources override this with two timed samples."""
        if not isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        return self.telemetry()

    def snapshot(self) -> Telemetry:
        """One telemetry call without waiting; independently queried sensors are not atomic.

        Timed energy-derived power requires sample(interval=...) instead.
        Unknown readings stay None, with their existing field provenance.
        """
        return self.telemetry()


def same_device(first: Device, second: Device) -> bool:
    """Whether two probe records describe one physical device.

    Vendor and kind must agree. The first identity both records carry decides: the PCI bus id, then the PCI
    product id, then the name head (text before a '/' or '(' suffix, with '(R)' and '(TM)' removed, lowercased).
    Two different bus ids are different devices even when their names match.
    """
    if (first.vendor, first.kind) != (second.vendor, second.kind):
        return False
    if first.pci_bus_id is not None and second.pci_bus_id is not None:
        return first.pci_bus_id == second.pci_bus_id
    if first.product_id is not None and second.product_id is not None:
        return first.product_id == second.product_id
    return _name_head(first.name) == _name_head(second.name)


@dataclass(frozen=True, kw_only=True)
class ReadingDevice(Device):
    telemetry_reader: Callable[[], Telemetry] = field(default=Telemetry, repr=False, compare=False)
    process_reader: Callable[[], tuple[Process, ...] | None] | None = field(default=None, repr=False, compare=False)
    nvlink_reader: Callable[[], tuple[NVLink, ...] | None] | None = field(default=None, repr=False, compare=False)

    def telemetry(self) -> Telemetry:
        return self.telemetry_reader()

    def processes(self) -> tuple[Process, ...] | None:
        return self.process_reader() if self.process_reader is not None else None

    def nvlinks(self) -> tuple[NVLink, ...] | None:
        return self.nvlink_reader() if self.nvlink_reader is not None else None


@dataclass(frozen=True, kw_only=True)
class Availability:
    available: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.available


def availability[T](source: str, read: Callable[[], T]) -> Availability:
    try:
        result = read()
        return Availability(available=bool(result), reason=None if result else f"{source}: no devices")
    except (RuntimeError, OSError, ValueError, AttributeError) as error:
        return Availability(available=False, reason=f"{source}: {error}")


class Probe[D: Device](ABC):
    @abstractmethod
    def available(self) -> Availability:
        raise NotImplementedError

    @abstractmethod
    def discover(self) -> tuple[D, ...]:
        raise NotImplementedError
