import ctypes
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from krn.hw import apple_native as native
from krn.hw import topology
from krn.hw.devices import apple, generic, sysman
from krn.hw.devices.base import ReadingDevice, Telemetry
from krn.hw.enums import Vendor


def test_sampling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    device = ReadingDevice(vendor=Vendor.UNKNOWN, name="GPU", index=0, source="test")
    assert device.sample(interval=1) == Telemetry()
    for interval in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="interval"):
            device.sample(interval=interval)
        with pytest.raises(ValueError, match="interval"):
            native.gpu_power(interval=interval)
    monkeypatch.setattr(native, "gpu_power", lambda **kwargs: 2)
    gpu = apple.AppleDevice(vendor=Vendor.APPLE, name="Apple M4", index=0, source="metal")
    assert gpu.sample(interval=0.1).sources == {"power_w": "ioreport"}
    monkeypatch.setattr(native, "gpu_power", lambda **kwargs: None)
    assert gpu.sample(interval=0.1).power_w is None
    drm = generic.GenericGPU(vendor=Vendor.UNKNOWN, name="GPU", index=0, source="drm")
    assert drm.sample(interval=1) == Telemetry()
    energy = tmp_path / "energy"
    energy.write_text("1000000")
    drm = replace(drm, energy_path=energy)
    monkeypatch.setattr(generic.time, "sleep", lambda interval: energy.write_text("3000000"))
    times = iter([0, 2])
    monkeypatch.setattr(generic.time, "monotonic", lambda: next(times))
    assert drm.sample(interval=2).power_w == 1
    times = iter([0, 0])
    assert drm.sample(interval=2).power_w is None
    times = iter([0, 2])
    energy.write_text("5000000")
    assert drm.sample(interval=2).power_w is None
    times = iter([0, 2])
    energy.unlink()
    assert drm.sample(interval=2).power_w is None


def test_sensor_preference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "device"
    path.mkdir()
    (path / "driver").symlink_to(tmp_path / "amdgpu")
    monkeypatch.setattr(
        generic,
        "psutil",
        SimpleNamespace(
            sensors_temperatures=lambda: {"amdgpu": [SimpleNamespace(current=50)]},
            sensors_fans=lambda: {"amdgpu": [SimpleNamespace(current=1000)]},
        ),
    )
    reading = generic._reading(path)
    assert reading.temperature_c == 50 and reading.fan_rpm == 1000
    assert reading.sources == {"temperature_c": "psutil", "fan_rpm": "psutil"}
    monkeypatch.setattr(
        generic,
        "psutil",
        SimpleNamespace(
            sensors_temperatures=lambda: {},
            sensors_fans=lambda: {},
        ),
    )
    assert generic._reading(path) == Telemetry()
    monkeypatch.setattr(generic, "psutil", None)
    assert generic._reading(path) == Telemetry()
    monkeypatch.setattr(native, "gpu_temperature", lambda: 42)
    monkeypatch.setattr(native, "registry_int", lambda *args: None)
    assert apple._reading().sources == {"temperature_c": "smc"}


def test_smc(monkeypatch: pytest.MonkeyPatch) -> None:
    io = SimpleNamespace(
        **{
            name: Mock(return_value=0)
            for name in (
                "IOServiceMatching",
                "IOServiceGetMatchingService",
                "IOServiceOpen",
                "IOConnectCallStructMethod",
                "IOObjectRelease",
                "IOServiceClose",
            )
        }
    )
    libc = SimpleNamespace(mach_task_self=Mock(return_value=1))
    monkeypatch.setattr(native.Libraries, "load", lambda name: SimpleNamespace(handle=io if name == "iokit" else libc))
    assert native.gpu_temperature() is None
    io.IOServiceGetMatchingService.return_value = 1
    io.IOServiceOpen.return_value = 1
    assert native.gpu_temperature() is None
    io.IOServiceOpen.return_value = 0
    assert native.gpu_temperature() is None
    monkeypatch.setattr(native, "_smc_float", lambda *args: 50)
    assert native.gpu_temperature() == 50
    monkeypatch.setattr(native.Libraries, "load", Mock(side_effect=OSError))
    assert native.gpu_temperature() is None and native.gpu_power(interval=0.01) is None


def test_smc_float() -> None:
    call = Mock(return_value=1)
    io = Mock(IOConnectCallStructMethod=call)
    assert native._smc_float(io, 1, "Tg04") is None
    call.return_value = 0
    assert native._smc_float(io, 1, "Tg04") is None

    def fill(
        connection: int,
        selector: int,
        request: ctypes.c_void_p,
        size: int,
        response: ctypes.c_void_p,
        output_size: ctypes.c_void_p,
    ) -> int:
        value = ctypes.cast(response, ctypes.POINTER(native.SMCKeyData)).contents
        value.info.data_size, value.info.data_type = 4, int.from_bytes(b"flt ", "big")
        value.data[:4] = struct.pack("<f", 42)
        return 0

    call.side_effect = fill
    assert native._smc_float(io, 1, "Tg04") == 42
    call.side_effect = [0, 1]
    # Pre-populate the info on the first call before failing the read.
    count = []

    def failed_read(
        connection: int,
        selector: int,
        request: ctypes.c_void_p,
        size: int,
        response: ctypes.c_void_p,
        output_size: ctypes.c_void_p,
    ) -> int:
        count.append(1)
        return fill(connection, selector, request, size, response, output_size) if len(count) == 1 else 1

    call.side_effect = failed_read
    assert native._smc_float(io, 1, "Tg04") is None


def test_ioreport(monkeypatch: pytest.MonkeyPatch) -> None:
    cf = Mock(
        **{
            name: Mock(return_value=1)
            for name in (
                "CFRelease",
                "CFStringCreateWithCString",
                "CFStringGetCString",
                "CFDictionaryGetValue",
                "CFArrayGetCount",
                "CFArrayGetValueAtIndex",
            )
        }
    )
    report = SimpleNamespace(
        **{
            name: Mock(return_value=1)
            for name in (
                "IOReportCopyAllChannels",
                "IOReportCreateSubscription",
                "IOReportCreateSamples",
                "IOReportCreateSamplesDelta",
                "IOReportChannelGetChannelName",
                "IOReportChannelGetUnitLabel",
                "IOReportSimpleGetIntegerValue",
            )
        }
    )
    report.IOReportChannelGetUnitLabel.return_value = 2
    report.IOReportSimpleGetIntegerValue.return_value = 4000000000
    labels = {1: b"GPU Energy", 2: b"nJ"}

    def string(handle: int, buffer: ctypes.Array[ctypes.c_char], size: int, encoding: int) -> bool:
        value = labels[handle] + b"\0"
        ctypes.memmove(buffer, value, len(value))
        return True

    cf.CFStringGetCString.side_effect = string
    monkeypatch.setattr(
        native.Libraries, "load", lambda name: SimpleNamespace(handle=cf if name == "corefoundation" else report)
    )
    monkeypatch.setattr(native.time, "sleep", lambda interval: None)
    monkeypatch.setattr(native.time, "monotonic", Mock(side_effect=[0, 2]))
    assert native.gpu_power(interval=1) == 2
    assert cf.CFRelease.call_count == 6

    def subscription(
        parent: ctypes.c_void_p, channels: int, subscribed: ctypes.c_void_p, flags: int, context: ctypes.c_void_p
    ) -> int:
        ctypes.cast(subscribed, ctypes.POINTER(ctypes.c_void_p)).contents.value = 2
        return 1

    report.IOReportCreateSubscription.side_effect = subscription
    monkeypatch.setattr(native.time, "monotonic", Mock(side_effect=[0, 2]))
    assert native.gpu_power(interval=1) == 2
    cf.CFDictionaryGetValue.return_value = 0
    monkeypatch.setattr(native.time, "monotonic", Mock(side_effect=[0, 2]))
    assert native.gpu_power(interval=1) is None
    cf.CFDictionaryGetValue.return_value = 1
    for label in (b"unknown", b"GPU Energy"):
        labels[1] = label
        labels[2] = b"unknown"
        monkeypatch.setattr(native.time, "monotonic", Mock(side_effect=[0, 2]))
        assert native.gpu_power(interval=1) is None
    report.IOReportCopyAllChannels.return_value = 0
    assert native.gpu_power(interval=1) is None
    assert native._cf_string(cf, 0) == ""
    cf.CFStringGetCString.return_value = False
    cf.CFStringGetCString.side_effect = None
    assert native._cf_string(cf, 1) == ""


def test_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    assert topology.detect_topology().unavailable_reason
    monkeypatch.setattr(topology, "pyhwloc", None)
    assert topology.detect_topology(system="Linux").unavailable_reason

    class Cache:
        cache_depth = 2
        cache_type = 0
        size = 1024

    fake = Mock()
    fake.__enter__ = Mock(return_value=fake)
    fake.__exit__ = Mock(return_value=False)
    fake.iter_numa_nodes.return_value = [SimpleNamespace(os_index=0, local_memory=1000)]
    fake.iter_all_breadth_first.return_value = [Cache(), SimpleNamespace()]
    fake.iter_pci_devices.return_value = [SimpleNamespace(attr=SimpleNamespace(domain=0, bus=1, dev=0, func=0))]
    monkeypatch.setattr(topology, "pyhwloc", SimpleNamespace(Topology=lambda: fake))
    monkeypatch.setattr(topology, "Cache", Cache)
    monkeypatch.setattr(topology, "TypeFilter", SimpleNamespace(KEEP_ALL=1))
    result = topology.detect_topology(system="Linux")
    assert result.numa_nodes == ((0, 1000),) and result.pci_devices == ("0000:01:00.0",)
    assert result.caches == (("l2-0", 1024),)
    fake.load.side_effect = OSError("missing")
    assert topology.detect_topology(system="Linux").unavailable_reason == "pyhwloc: missing"
    monkeypatch.setattr(topology, "Cache", None)
    assert topology.detect_topology(system="Linux").unavailable_reason


def test_sysman_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysman, "pyzes", None)
    assert not sysman.SysmanProbe().available() and sysman.SysmanProbe().discover() == ()
    assert sysman._reading(ctypes.c_void_p()) == Telemetry()


class SysmanHandle(ctypes.c_void_p):
    pass


class SysmanCore(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 16),
        ("numSlices", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
    ]


class SysmanProperties(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("core", SysmanCore)]


class PCIAddress(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("bus", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("function", ctypes.c_uint32),
    ]


class PCIProperties(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("address", PCIAddress)]


def test_sysman(monkeypatch: pytest.MonkeyPatch) -> None:
    def handles(count: ctypes._Pointer[ctypes.c_uint32], values: ctypes.Array[ctypes.c_void_p] | None) -> int:
        count.contents.value = 1
        if values is not None:
            values[0] = SysmanHandle(1)
        return 0

    def temperature(handle: ctypes.c_void_p, result: ctypes._Pointer[ctypes.c_double]) -> int:
        result.contents.value = 42
        return 0

    def power(
        handle: ctypes.c_void_p, instant: ctypes._Pointer[ctypes.c_uint32], average: ctypes._Pointer[ctypes.c_uint32]
    ) -> int:
        instant.contents.value = 2000
        return 0

    fake = SimpleNamespace(
        zesInit=Mock(return_value=0),
        zesDriverGet=handles,
        zesDeviceGet=lambda device, count, values: handles(count, values),
        zesDeviceEnumTemperatureSensors=lambda device, count, values: handles(count, values),
        zesDeviceEnumPowerDomains=lambda device, count, values: handles(count, values),
        zes_driver_handle_t=SysmanHandle,
        zes_device_handle_t=SysmanHandle,
        zes_temp_handle_t=SysmanHandle,
        zes_pwr_handle_t=SysmanHandle,
        zes_device_properties_t=lambda: SysmanProperties(core=SysmanCore(b"Intel GPU", 2, 4, 8)),
        zes_pci_properties_t=lambda: PCIProperties(address=PCIAddress(0, 1, 0, 0)),
        ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES=1,
        ZES_STRUCTURE_TYPE_PCI_PROPERTIES=2,
        zesDeviceGetProperties=Mock(return_value=0),
        zesDevicePciGetProperties=Mock(return_value=0),
        zesTemperatureGetState=temperature,
        zesPowerGetUsage=power,
    )
    monkeypatch.setattr(sysman, "pyzes", fake)
    probe = sysman.SysmanProbe()
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.name == "Intel GPU" and gpu.limits.compute_units == 64 and gpu.pci_bus_id == "0000:01:00.0"
    reading = gpu.telemetry()
    assert reading.temperature_c == 42 and reading.power_w == 2
    assert reading.utilization_percent is None and reading.sources == {"temperature_c": "pyzes", "power_w": "pyzes"}
    fake.zesDevicePciGetProperties.return_value = 1
    assert probe.discover()[0].pci_bus_id is None
    fake.zesDeviceGetProperties.return_value = 1
    assert probe.discover() == ()
    fake.zesInit.return_value = 1
    assert not probe.available()
    monkeypatch.setattr(fake, "zesDeviceEnumTemperatureSensors", Mock(side_effect=RuntimeError))
    assert gpu.telemetry() == Telemetry()
    assert sysman._handles(lambda count, values: 1, SysmanHandle) == ()
    codes = iter([0, 1])
    assert sysman._handles(lambda count, values: next(codes), SysmanHandle) == ()
    monkeypatch.setattr(fake, "zesDeviceEnumTemperatureSensors", lambda device, count, values: 1)
    monkeypatch.setattr(fake, "zesDeviceEnumPowerDomains", lambda device, count, values: 1)
    assert gpu.telemetry() == Telemetry()
