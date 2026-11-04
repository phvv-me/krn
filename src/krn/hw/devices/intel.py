"""SYCL properties with DRM as the telemetry fallback."""

from collections.abc import Callable

try:
    import dpctl
except ImportError:
    dpctl = None

try:
    import openvino
except ImportError:
    openvino = None

from ..enums import DeviceKind, Vendor
from .base import Availability, DeviceLimits, DeviceMemory, Probe, ReadingDevice, availability, brand_vendor


class IntelDevice(ReadingDevice):
    """An Intel GPU reported by SYCL."""


class IntelProbe(Probe[IntelDevice]):
    def available(self) -> Availability:
        if dpctl is None and openvino is None:
            return Availability(available=False, reason="Intel: OpenVINO and SYCL imports unavailable")
        return availability("Intel", self.discover)

    def discover(self) -> tuple[IntelDevice, ...]:
        devices = self._openvino()
        if dpctl is None or any(d.kind == DeviceKind.GPU for d in devices):
            return devices
        return devices + tuple(
            IntelDevice(
                vendor=Vendor.INTEL,
                index=index,
                name=d.name,
                source="sycl",
                memory_facts=DeviceMemory(total=d.global_mem_size),
                capabilities=frozenset({"fp64"} if d.has_aspect_fp64 else set()),
                limit_facts=DeviceLimits(
                    compute_units=d.max_compute_units,
                    subgroup_sizes=tuple(d.sub_group_sizes),
                    max_threads_per_block=d.max_work_group_size,
                    shared_memory_per_block=d.local_mem_size,
                    max_clock_mhz=d.max_clock_frequency,
                ),
            )
            for index, d in enumerate(dpctl.get_devices(device_type="gpu"))
            if "intel" in d.vendor.lower()
        )

    def _openvino(self) -> tuple[IntelDevice, ...]:
        if openvino is None:
            return ()
        try:
            core = openvino.Core()
            names = core.available_devices
        except RuntimeError:
            return ()
        devices = []
        for name in names:
            if not name.startswith(("CPU", "GPU", "NPU")):
                continue
            kind = {"CPU": DeviceKind.CPU, "GPU": DeviceKind.GPU, "NPU": DeviceKind.NPU}[name.partition(".")[0]]
            suffix = name.partition(".")[2].partition(".")[0]
            pci = _property(lambda key=name: core.get_property(key, "DEVICE_PCI_INFO"))
            device_type = _property(lambda key=name: core.get_property(key, "DEVICE_TYPE"))
            total = (
                _property(lambda key=name: core.get_property(key, "GPU_DEVICE_TOTAL_MEM_SIZE"))
                if kind == DeviceKind.GPU
                else _property(lambda key=name: core.get_property(key, "NPU_DEVICE_TOTAL_MEM_SIZE"))
                if kind == DeviceKind.NPU
                else None
            )
            capabilities = _property(lambda key=name: core.get_property(key, "OPTIMIZATION_CAPABILITIES")) or []
            full_name = _property(lambda key=name: core.get_property(key, "FULL_DEVICE_NAME")) or name
            # OpenVINO publishes no PCI vendor id, so the brand of its device name is the only vendor signal.
            vendor = Vendor.INTEL if kind == DeviceKind.NPU else brand_vendor(full_name)
            if kind == DeviceKind.GPU and vendor != Vendor.INTEL:
                continue  # another vendor's GPU belongs to its own probe
            devices.append(
                IntelDevice(
                    vendor=vendor,
                    kind=kind,
                    index=int(suffix) if suffix.isdigit() else 0,
                    name=full_name,
                    source="openvino",
                    architecture=_property(lambda key=name: core.get_property(key, "DEVICE_ARCHITECTURE")),
                    pci_bus_id=f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.{pci.function}"
                    if pci is not None
                    else None,
                    memory_facts=DeviceMemory(
                        total=total, unified=device_type.name == "INTEGRATED" if device_type is not None else None
                    ),
                    limit_facts=DeviceLimits(
                        compute_units=_property(lambda key=name: core.get_property(key, "GPU_EXECUTION_UNITS_COUNT"))
                        if kind == DeviceKind.GPU
                        else None
                    ),
                    capabilities=frozenset(value.lower().replace("_", "-") for value in capabilities),
                )
            )
        return tuple(devices)


def _property[T](read: Callable[[], T]) -> T | None:
    try:
        return read()
    except RuntimeError:
        return None
