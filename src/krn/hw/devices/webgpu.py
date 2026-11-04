"""Portable adapter identity and WebGPU limits."""

from collections.abc import Mapping

try:
    import wgpu
except ImportError:
    wgpu = None

from ..enums import DeviceKind, Vendor
from .base import Availability, DeviceLimits, DeviceMemory, Probe, ReadingDevice, availability, brand_vendor

_SOFTWARE_RENDERERS = ("llvmpipe", "lavapipe", "swiftshader")


def _software(info: Mapping[str, str | int], name: str) -> bool:
    text = f"{name} {info.get('description', '')}".lower()
    return str(info.get("adapter_type", "")).lower() == "cpu" or any(key in text for key in _SOFTWARE_RENDERERS)


class WGPUProbe(Probe[ReadingDevice]):
    """Adapters the WebGPU runtime enumerates, including software renderers.

    The PCI vendor id, when the adapter reports one, names the vendor; the brand in the name is the fallback.
    Software renderers (adapter type CPU, llvmpipe, lavapipe, SwiftShader) keep kind SOFTWARE.
    """

    def available(self) -> Availability:
        if wgpu is None:
            return Availability(available=False, reason="wgpu: import unavailable")
        return availability("wgpu", wgpu.gpu.enumerate_adapters_sync)

    def discover(self) -> tuple[ReadingDevice, ...]:
        if wgpu is None:
            return ()
        devices = []
        for index, adapter in enumerate(wgpu.gpu.enumerate_adapters_sync()):
            info, limits = adapter.info, adapter.limits
            name = str(info.get("device") or info.get("description") or "Unknown GPU")
            vendor_id = info.get("vendor_id")
            if vendor_id in Vendor and vendor_id != 0:
                vendor = Vendor(vendor_id)
            else:
                vendor = brand_vendor(str(info.get("vendor") or name))
            architecture = str(info.get("architecture") or "") or None
            minimum, maximum = info.get("stackgroup_min_size"), info.get("stackgroup_max_size")
            devices.append(
                ReadingDevice(
                    vendor=vendor,
                    kind=DeviceKind.SOFTWARE if _software(info, name) else DeviceKind.GPU,
                    index=index,
                    name=name,
                    architecture=architecture,
                    product_id=info.get("device_id") or None,
                    source="wgpu",
                    backend=str(info.get("backend_type") or ""),
                    description=str(info.get("description") or ""),
                    memory_facts=DeviceMemory(max_buffer_length=limits.get("max-buffer-size")),
                    limit_facts=DeviceLimits(
                        max_threads_per_block=limits.get("max-compute-invocations-per-workgroup"),
                        shared_memory_per_block=limits.get("max-compute-workgroup-storage-size"),
                        subgroup_sizes=(minimum, maximum)
                        if isinstance(minimum, int) and isinstance(maximum, int)
                        else None,
                    ),
                )
            )
        return tuple(devices)
