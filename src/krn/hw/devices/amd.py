"""HIP properties and optional SMI management readings."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from ..exceptions import ProbeError

if TYPE_CHECKING:
    from amdsmi import ProcessorHandle

try:
    from hip import hip
except ImportError:
    hip = None

try:
    import amdsmi
    from amdsmi import AmdSmiException
except ImportError:
    amdsmi = None
    AmdSmiException = RuntimeError

from ..enums import Vendor
from ..peaks import PeakRates, PeakSource, bandwidth, positive
from .base import Availability, DeviceLimits, DeviceMemory, Probe, ReadingDevice, Telemetry, availability


def _reading(handle: ProcessorHandle) -> Telemetry:
    smi = amdsmi
    if smi is None:
        return Telemetry()
    activity = _smi(lambda: smi.amdsmi_get_gpu_activity(handle)) or {}
    clocks = _smi(lambda: smi.amdsmi_get_clock_info(handle, smi.AmdSmiClkType.GFX)) or {}
    return Telemetry(
        utilization_percent=_number(activity.get("gfx_activity")),
        clock_mhz=_number(clocks.get("clk")),
        max_clock_mhz=_number(clocks.get("max_clk")),
        memory_utilization_percent=_number(activity.get("umc_activity")),
    ).attributed("amdsmi")


def _number(value: str | int | None) -> int | None:
    return value if isinstance(value, int) else None


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _smi[T](read: Callable[[], T]) -> T | None:
    try:
        return read()
    except RuntimeError, AmdSmiException, AttributeError, OSError:
        return None


def _reader(bus: str) -> Callable[[], Telemetry]:
    if amdsmi is None:
        return Telemetry
    try:
        amdsmi.amdsmi_init()
        handle = amdsmi.amdsmi_get_processor_handle_from_bdf(bus)
    except RuntimeError, AmdSmiException:
        return Telemetry
    return partial(_reading, handle)


class AMDDevice(ReadingDevice):
    """An AMD compute device reported by HIP."""


def _peak(handle: ProcessorHandle) -> PeakRates:
    smi = amdsmi
    if smi is None:
        return PeakRates()
    # gpu_metrics reports bandwidth at MAX memory clock in decimal GB/s;
    # vram_info's value is at CURRENT clock, unsuitable as a static peak.
    # https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html
    reader = getattr(smi, "amdsmi_get_gpu_metrics_info", None)
    metrics = (_smi(lambda: reader(handle)) if reader is not None else None) or {}
    value = positive(lambda: metrics["vram_max_bandwidth"])
    return PeakRates(
        memory_bandwidth_gbs=value, memory_bandwidth_source=PeakSource.LIBRARY if value is not None else None
    )


class AMDProbe(Probe[AMDDevice]):
    def available(self) -> Availability:
        if hip is None:
            if amdsmi is None:
                return Availability(available=False, reason="hip and amdsmi: imports unavailable")
            return availability("amdsmi", self._management_devices)
        return availability("hip", self.discover)

    def _management_devices(self) -> tuple[AMDDevice, ...]:
        smi = amdsmi
        if smi is None:
            return ()
        try:
            smi.amdsmi_init()
            handles = smi.amdsmi_get_processor_handles()
        except RuntimeError, AmdSmiException, AttributeError, OSError:
            return ()
        result = []
        for handle in handles:
            identity = _smi(lambda handle=handle: smi.amdsmi_get_gpu_asic_info(handle)) or {}
            name, arch = identity.get("market_name"), identity.get("target_graphics_version")
            result.append(
                AMDDevice(
                    vendor=Vendor.AMD,
                    index=len(result),
                    name=name if isinstance(name, str) else "AMD GPU",
                    architecture=arch if isinstance(arch, str) else None,
                    source="amdsmi",
                    peak_facts=_peak(handle),
                    # gfx942 also covers discrete MI300X; it is not enough.
                    coherence_facts=True if isinstance(name, str) and "MI300A" in name.upper() else None,
                    uuid=_smi(lambda handle=handle: smi.amdsmi_get_gpu_device_uuid(handle)),
                    pci_bus_id=_smi(lambda handle=handle: smi.amdsmi_get_gpu_device_bdf(handle)),
                    memory_facts=DeviceMemory(
                        total=_smi(
                            lambda handle=handle: smi.amdsmi_get_gpu_memory_total(handle, smi.AmdSmiMemoryType.VRAM)
                        )
                    ),
                    telemetry_reader=partial(_reading, handle),
                )
            )
        return tuple(result)

    def discover(self) -> tuple[AMDDevice, ...]:
        management = self._management_devices()
        if hip is None:
            return management
        result = []
        count = hip.hipGetDeviceCount()
        if isinstance(count, tuple):
            status, count = count
            if status:
                raise ProbeError("HIP device count failed", "Check the ROCm driver")
        for index in range(count):
            props = hip.hipGetDeviceProperties(index)
            if isinstance(props, tuple):
                status, p = props
                if status:
                    raise ProbeError("HIP properties failed", "Check the ROCm driver")
            else:
                p = props
            bus = f"{p.pciDomainID:04x}:{p.pciBusID:02x}:{p.pciDeviceID:02x}.0"
            identity = next((device for device in management if device.pci_bus_id == bus), None)
            name = identity.name if identity is not None else _text(p.name)
            result.append(
                AMDDevice(
                    vendor=Vendor.AMD,
                    index=index,
                    name=name,
                    uuid=identity.uuid if identity is not None else None,
                    architecture=_text(p.gcnArchName).split(":")[0],
                    source="hip",
                    peak_facts=identity.peak_facts
                    if identity is not None and identity.peak_facts.memory_bandwidth_gbs is not None
                    else bandwidth(
                        positive(lambda p=p: getattr(p, "memoryClockRate", None)),
                        positive(lambda p=p: getattr(p, "memoryBusWidth", None)),
                    ),
                    coherence_facts=True if "MI300A" in name.upper() else (False if not p.integrated else None),
                    pci_bus_id=bus,
                    telemetry_reader=_reader(bus),
                    memory_facts=DeviceMemory(total=p.totalGlobalMem, unified=bool(p.integrated)),
                    limit_facts=DeviceLimits(
                        compute_units=p.multiProcessorCount,
                        subgroup_size=p.warpSize,
                        max_threads_per_block=p.maxThreadsPerBlock,
                        shared_memory_per_block=p.sharedMemPerBlock,
                        max_clock_mhz=p.clockRate / 1000,
                    ),
                )
            )
        return tuple(result)
