"""CUDA properties and NVML readings joined by PCI identity."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core import Device as CoreDevice
    from cuda.core.system import Device as ManagementDevice

try:
    from cuda import core
    from cuda.core import system
except ImportError:
    core = None
    system = None

try:
    from cuda.bindings import driver
except ImportError:
    driver = None

try:
    from cuda.bindings import nvml
    from cuda.bindings.nvml import NvmlError
except ImportError:
    nvml = None
    NvmlError = RuntimeError

from ..enums import Vendor
from ..exceptions import ProbeError
from ..libraries import Libraries
from ..peaks import PeakRates, bandwidth, positive
from .base import (
    Availability,
    DeviceLimits,
    DeviceMemory,
    NVLink,
    PCILink,
    Probe,
    Process,
    ReadingDevice,
    Telemetry,
    availability,
    canonical_pci,
)


def _optional[T](read: Callable[[], T]) -> T | None:
    try:
        return read()
    except RuntimeError, ValueError, AttributeError, NvmlError:
        return None


def _power(index: int, limit: bool) -> float | None:
    if nvml is None:
        return None
    handle = nvml.device_get_handle_by_index_v2(index)
    value = nvml.device_get_power_management_limit(handle) if limit else nvml.device_get_power_usage(handle)
    return value / 1000


def _cuda_count() -> int:
    """Devices the CUDA runtime exposes after CUDA_VISIBLE_DEVICES; NVML lists every physical device."""
    if driver is None or driver.cuInit(0)[0] != driver.CUresult.CUDA_SUCCESS:
        return 0
    status, count = driver.cuDeviceGetCount()
    return count if status == driver.CUresult.CUDA_SUCCESS else 0


def _visible(device_type: type[CoreDevice], count: int) -> dict[str, tuple[int, CoreDevice]]:
    """CUDA devices by PCI bus id with their ordinal; an ordinal the runtime cannot open counts as hidden."""
    devices = {}
    for ordinal in range(count):
        try:
            device = device_type(ordinal)
        except RuntimeError, ValueError:
            continue
        devices[canonical_pci(device.pci_bus_id)] = (ordinal, device)
    return devices


def _reading(device: ManagementDevice) -> Telemetry:
    clock = _optional(lambda: device.get_clock("sm"))
    reasons = _optional(lambda: device.current_clock_event_reasons)
    reading = Telemetry(
        temperature_c=_optional(lambda: device.temperature.get_sensor()),
        slowdown_temperature_c=_optional(lambda: device.temperature.get_threshold("slowdown")),
        fan_percent=_optional(lambda: device.get_fan(0).speed) if _optional(lambda: device.num_fans) else None,
        clock_mhz=_optional(clock.get_current_mhz) if clock is not None else None,
        max_clock_mhz=_optional(clock.get_max_mhz) if clock is not None else None,
        utilization_percent=_optional(lambda: device.utilization.gpu),
        memory_used_bytes=_optional(lambda: device.memory_info.used),
        memory_total_bytes=_optional(lambda: device.memory_info.total),
        memory_free_bytes=_optional(lambda: getattr(device.memory_info, "free", None)),
        memory_utilization_percent=_optional(lambda: getattr(device.utilization, "memory", None)),
        throttle_reasons=tuple(reason.name for reason in reasons if reason.name != "GPU_IDLE")
        if reasons is not None
        else None,
        performance_state=_optional(lambda: device.performance_state),
        power_w=_optional(lambda: _power(device.index, False)),
        power_limit_w=_optional(lambda: _power(device.index, True)),
    ).attributed("cuda.core.system")
    return replace(
        reading,
        sources={
            name: "cuda.bindings.nvml" if name in {"power_w", "power_limit_w"} else source
            for name, source in reading.sources.items()
        },
    )


def _processes(device: ManagementDevice) -> tuple[Process, ...] | None:
    return _optional(
        lambda: tuple(
            Process(pid=process.pid, memory_bytes=process.used_gpu_memory)
            for process in device.compute_running_processes
        )
    )


def _nvlinks(device: ManagementDevice) -> tuple[NVLink, ...] | None:
    return _optional(
        lambda: tuple(
            NVLink(index=index, active=link.state, version=link.version)
            for index, link in enumerate(device.get_nvlinks())
        )
    )


def _device(index: int, management: ManagementDevice, cuda: tuple[int, CoreDevice] | None) -> NvidiaDevice:
    """A physical device; CUDA facts exist only while the CUDA runtime exposes the device to this process."""
    ordinal, compute = (None, None) if cuda is None else cuda
    # Product identity is available even when a visibility mask hides the GPU.
    # https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
    coherent = True if any(key in management.name for key in ("GH200", "GB200")) else None
    architecture, limits, peak, integrated = None, DeviceLimits(), PeakRates(), None
    if compute is not None:
        props = compute.properties
        integrated = props.integrated
        major, minor = compute.compute_capability
        capability = major * 10 + minor
        architecture = f"sm_{capability}{'a' if capability >= 90 else ''}"
        limits = DeviceLimits(
            compute_units=props.multiprocessor_count,
            subgroup_size=props.warp_size,
            max_threads_per_block=props.max_threads_per_block,
            shared_memory_per_block=props.max_shared_memory_per_block,
            registers_per_block=props.max_registers_per_block,
            max_clock_mhz=props.clock_rate / 1000,
        )
        peak = bandwidth(
            positive(lambda props=props: getattr(props, "memory_clock_rate", None)),
            positive(lambda props=props: getattr(props, "global_memory_bus_width", None)),
        )
        # ATS plus direct host access identifies coherent CUDA systems
        # even when the GPU name omits the Grace system's product name.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-system-allocator
        if getattr(props, "pageable_memory_access_uses_host_page_tables", False) and getattr(
            props, "direct_managed_mem_access_from_host", False
        ):
            coherent = True
        if coherent is None and not integrated:
            coherent = False
    return NvidiaDevice(
        vendor=Vendor.NVIDIA,
        index=index,
        name=management.name,
        uuid=management.uuid,
        pci_bus_id=management.pci_bus_id,
        visible=compute is not None,
        ordinal=ordinal,
        architecture=architecture,
        source="cuda",
        limit_facts=limits,
        peak_facts=peak,
        coherence_facts=coherent,
        memory_facts=DeviceMemory(total=management.memory_info.total, unified=integrated),
        pci=PCILink(
            generation=management.pci_info.max_link_generation,
            width=management.pci_info.max_link_width,
            numa_node=_optional(lambda device=management: device.numa_node_id),
            cpu_affinity=_optional(lambda device=management: tuple(device.get_cpu_affinity())),
        ),
        telemetry_reader=partial(_reading, management),
        process_reader=partial(_processes, management),
        nvlink_reader=partial(_nvlinks, management),
    )


class NvidiaDevice(ReadingDevice):
    """One physical NVIDIA device, independent of CUDA visibility order."""


class NvidiaProbe(Probe[NvidiaDevice]):
    def available(self) -> Availability:
        if core is None or system is None:
            return Availability(available=False, reason="cuda.core: import unavailable")
        if not Libraries.available("nvml"):
            return availability("nvml", lambda: Libraries.load("nvml"))
        return availability("cuda.core.system", system.get_num_devices)

    def discover(self) -> tuple[NvidiaDevice, ...]:
        if core is None or system is None:
            return ()
        try:
            count = system.get_num_devices()
        except NvmlError as error:
            raise ProbeError("NVIDIA management driver unavailable", "Check the NVIDIA driver installation") from error
        # CUDA ordinals follow CUDA_VISIBLE_DEVICES, so NVML indices never address CUDA devices directly.
        visible = _visible(core.Device, _cuda_count())
        managed = [system.Device(index=index) for index in range(count)]
        return tuple(
            _device(index, management, visible.get(canonical_pci(management.pci_bus_id)))
            for index, management in enumerate(managed)
        )
