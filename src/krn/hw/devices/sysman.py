"""Intel Sysman identity and live sensors through the pyzes binding."""

import ctypes
from collections.abc import Callable
from functools import partial

try:
    import pyzes
except ImportError, OSError:  # pyzes opens libze_loader while importing, so a missing runtime raises OSError
    pyzes = None

from ..enums import Vendor
from ..exceptions import ProbeError
from .base import Availability, DeviceLimits, Probe, ReadingDevice, Telemetry, availability


def _handles(
    read: Callable[[ctypes._Pointer[ctypes.c_uint32], ctypes.Array[ctypes.c_void_p] | None], int],
    handle_type: type[ctypes.c_void_p],
) -> tuple[ctypes.c_void_p, ...]:
    count = ctypes.c_uint32()
    if read(ctypes.pointer(count), None):
        return ()
    handles = (handle_type * count.value)()
    return () if read(ctypes.pointer(count), handles) else tuple(handles)


def _reading(device: ctypes.c_void_p) -> Telemetry:
    if pyzes is None:
        return Telemetry()
    try:
        temperatures = _handles(partial(pyzes.zesDeviceEnumTemperatureSensors, device), pyzes.zes_temp_handle_t)
        powers = _handles(partial(pyzes.zesDeviceEnumPowerDomains, device), pyzes.zes_pwr_handle_t)
        temperature, power, average = ctypes.c_double(), ctypes.c_uint32(), ctypes.c_uint32()
        thermal = temperatures and not pyzes.zesTemperatureGetState(temperatures[0], ctypes.pointer(temperature))
        powered = powers and not pyzes.zesPowerGetUsage(powers[0], ctypes.pointer(power), ctypes.pointer(average))
        return Telemetry(
            temperature_c=temperature.value if thermal else None, power_w=power.value / 1000 if powered else None
        ).attributed("pyzes")
    except OSError, RuntimeError, AttributeError, ValueError, NotImplementedError:
        return Telemetry()


class SysmanProbe(Probe[ReadingDevice]):
    def available(self) -> Availability:
        if pyzes is None:
            return Availability(available=False, reason="pyzes: import unavailable")
        return availability("pyzes", self.discover)

    def discover(self) -> tuple[ReadingDevice, ...]:
        if pyzes is None:
            return ()
        if pyzes.zesInit(0):
            raise ProbeError("Sysman initialization failed", "Check the Level Zero driver and Sysman permissions")
        drivers = _handles(pyzes.zesDriverGet, pyzes.zes_driver_handle_t)
        result = []
        for driver in drivers:
            for handle in _handles(partial(pyzes.zesDeviceGet, driver), pyzes.zes_device_handle_t):
                properties = pyzes.zes_device_properties_t()
                properties.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
                if pyzes.zesDeviceGetProperties(handle, ctypes.pointer(properties)):
                    continue
                core = properties.core
                pci = pyzes.zes_pci_properties_t()
                pci.stype = pyzes.ZES_STRUCTURE_TYPE_PCI_PROPERTIES
                bus = None
                if not pyzes.zesDevicePciGetProperties(handle, ctypes.pointer(pci)):
                    address = pci.address
                    bus = f"{address.domain:04x}:{address.bus:02x}:{address.device:02x}.{address.function}"
                result.append(
                    ReadingDevice(
                        vendor=Vendor.INTEL,
                        index=len(result),
                        name=core.name.decode().rstrip("\0"),
                        source="pyzes",
                        pci_bus_id=bus,
                        limit_facts=DeviceLimits(
                            compute_units=core.numSlices * core.numSubslicesPerSlice * core.numEUsPerSubslice
                        ),
                        telemetry_reader=partial(_reading, handle),
                    )
                )
        return tuple(result)
