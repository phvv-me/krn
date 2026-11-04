"""Metal identity and limits; IOKit supplies the GPU core count."""

from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING

from krn.facts.targets import Target

if TYPE_CHECKING:
    from Metal import MTLDevice

try:
    import Metal
except ImportError:
    Metal = None

try:
    import mlx.core as mlx
except ImportError:
    mlx = None

from .. import apple_native as native
from ..enums import Vendor
from .base import Availability, DeviceLimits, DeviceMemory, Probe, ReadingDevice, Telemetry, availability


def _reading() -> Telemetry:
    reading = Telemetry(
        temperature_c=native.gpu_temperature(),
        memory_used_bytes=native.registry_int("PerformanceStatistics", "In use system memory"),
        utilization_percent=native.registry_int("PerformanceStatistics", "Device Utilization %"),
    ).attributed("iokit")
    return replace(
        reading, sources={**reading.sources, **({"temperature_c": "smc"} if reading.temperature_c is not None else {})}
    )


class AppleDevice(ReadingDevice):
    """Metal working-set memory is a budget, not dedicated VRAM."""

    @cached_property
    def coherent(self) -> bool | None:
        # Apple Silicon unified memory is cache coherent; an arbitrary Metal
        # GPU's unified allocation flag alone does not establish this.
        # https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
        return self.memory.unified

    def sample(self, *, interval: float) -> Telemetry:
        power = native.gpu_power(interval=interval)
        reading = self.telemetry()
        return replace(
            reading,
            power_w=power,
            sources={**reading.sources, **({"power_w": "ioreport"} if power is not None else {})},
        )

    @cached_property
    def target(self) -> Target | None:
        generation = next(iter(self.name.partition("Apple M")[2].split()), "")
        try:
            return Target.of(f"apple_m{generation}")
        except ValueError, KeyError:
            return None


class AppleProbe(Probe[ReadingDevice]):
    def available(self) -> Availability:
        if Metal is None:
            return Availability(available=False, reason="Metal: import unavailable")
        return availability("Metal", Metal.MTLCreateSystemDefaultDevice)

    def discover(self) -> tuple[ReadingDevice, ...]:
        if Metal is None:
            return ()
        default = Metal.MTLCreateSystemDefaultDevice()
        devices = tuple(Metal.MTLCopyAllDevices())
        if not devices and default is not None:
            devices = (default,)
        return tuple(self._device(device, index) for index, device in enumerate(devices))

    def _device(self, device: MTLDevice, index: int) -> ReadingDevice:
        brand = device.name()
        vendor = next(
            (
                vendor
                for prefix, vendor in (
                    ("Apple", Vendor.APPLE),
                    ("AMD", Vendor.AMD),
                    ("Intel", Vendor.INTEL),
                    ("NVIDIA", Vendor.NVIDIA),
                )
                if brand.startswith(prefix)
            ),
            Vendor.UNKNOWN,
        )
        if vendor != Vendor.APPLE:
            return ReadingDevice(
                vendor=vendor,
                index=index,
                name=brand,
                uuid=str(device.registryID()),
                source="metal",
                memory_facts=DeviceMemory(
                    total=device.recommendedMaxWorkingSetSize(), unified=device.hasUnifiedMemory()
                ),
                limit_facts=DeviceLimits(
                    max_threads_per_block=device.maxThreadsPerThreadgroup().width,
                    shared_memory_per_block=device.maxThreadgroupMemoryLength(),
                ),
            )
        return self._apple(device, index)

    def _apple(self, device: MTLDevice, index: int) -> AppleDevice:
        family = max((family for family in range(7, 12) if device.supportsFamily_(1000 + family)), default=0)
        brand = device.name()
        generation = next(iter(brand.partition("Apple M")[2].split()), "")
        architecture = f"apple_m{generation}" if generation.isdigit() else None
        info = _mlx_info()
        mlx_name, mlx_arch = info.get("device_name"), info.get("architecture")
        return AppleDevice(
            vendor=Vendor.APPLE,
            index=index,
            name=mlx_name if isinstance(mlx_name, str) else device.name(),
            uuid=str(device.registryID()),
            architecture=mlx_arch if isinstance(mlx_arch, str) else architecture,
            source="metal",
            memory_facts=DeviceMemory(
                total=_integer(info.get("memory_size")) or device.recommendedMaxWorkingSetSize(),
                unified=device.hasUnifiedMemory(),
                recommended_working_set=_integer(info.get("max_recommended_working_set_size"))
                or device.recommendedMaxWorkingSetSize(),
                max_buffer_length=_integer(info.get("max_buffer_length")),
            ),
            limit_facts=DeviceLimits(
                compute_units=native.gpu_core_count(),
                max_threads_per_block=device.maxThreadsPerThreadgroup().width,
                shared_memory_per_block=device.maxThreadgroupMemoryLength(),
            ),
            capabilities=frozenset({f"apple-family-{family}"} if family else set())
            | frozenset({"fp16", "unified-memory"} if device.hasUnifiedMemory() else {"fp16"}),
            telemetry_reader=_reading,
        )


def _integer(value: str | int | None) -> int | None:
    return value if isinstance(value, int) else None


def _mlx_info() -> dict[str, str | int]:
    if mlx is None:
        return {}
    try:
        return mlx.device_info()
    except RuntimeError, AttributeError, OSError:
        return {}
