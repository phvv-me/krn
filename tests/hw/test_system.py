import platform
from dataclasses import FrozenInstanceError, dataclass, replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from krn import configure
from krn.hw import system as system_module
from krn.hw.devices.base import Availability, DeviceMemory, PCILink, ReadingDevice, Telemetry, same_device
from krn.hw.enums import DeviceKind, Vendor
from krn.hw.exceptions import HardwareError
from krn.hw.install import install_plan
from krn.hw.system import System


@dataclass(frozen=True, kw_only=True)
class Counted:
    calls: ClassVar[list[int]] = []

    @cached_property
    def value(self) -> int:
        self.calls.append(1)
        return len(self.calls)


def test_frozen_cached() -> None:
    counted = Counted()
    assert counted.value == counted.value == 1
    assert Counted.calls == [1]
    with pytest.raises(FrozenInstanceError):
        setattr(Telemetry(temperature_c=1), "temperature_c".lower(), 2)
    with pytest.raises(FrozenInstanceError):
        setattr(counted, "other".lower(), 1)


def test_inventory(inventory: System, monkeypatch: pytest.MonkeyPatch) -> None:
    device = ReadingDevice(
        vendor=Vendor.NVIDIA,
        index=0,
        name="GPU",
        uuid="GPU-1",
        source="cuda",
        architecture="sm_120a",
        memory_facts=DeviceMemory(total=100),
    )
    system = replace(inventory, devices=(device,))
    assert system.device(0) is device and system.device("GPU-1") is device
    assert system.require(Vendor.NVIDIA) == system.gpus == (device,)
    assert system.total_device_memory == 100
    assert system.targets == (device.target,)
    assert device.memory is device.memory and device.link is device.pci
    assert device.limits is device.limit_facts and "fp8" in device.features
    assert device.telemetry() == Telemetry() and device.telemetry() is not device.telemetry()
    assert device.processes() is None
    assert system.describe() == system.describe()
    assert system.describe().splitlines()[0] == "Linux 1 (x86_64)"
    assert system.describe() == (
        "Linux 1 (x86_64)\nCPU: CPU; 8 logical cores; 8 usable\nMemory: 1024 bytes; page size 4096\n"
        "Python: 3.14 CPython; GIL True\nGPU NVIDIA:0: GPU; sm_120a; 100 bytes\nVolume /: 50/100 bytes free"
    )
    configure(**{"run.target": None})
    assert system.target.arch == "sm_120a"
    assert inventory.target.arch == "avx2"
    assert install_plan(system).extras == ("cuda", "cuda-libs")
    assert install_plan(inventory).command == 'pip install "krn"'
    assert ReadingDevice(vendor=Vendor.UNKNOWN, index=0, name="?", source="drm").target is None
    assert (
        ReadingDevice(vendor=Vendor.UNKNOWN, index=0, name="?", source="drm", architecture="unsupported").target is None
    )
    for selector in (2, "missing"):
        with pytest.raises(HardwareError, match="Select a unique"):
            system.device(selector)
    with pytest.raises(HardwareError, match="visibility"):
        inventory.require(Vendor.APPLE)


def test_live() -> None:
    system = System.detect()
    assert system is System.detect()
    assert system.cpu.logical_cores >= 1 and system.memory.total is not None and system.memory.total > 0
    assert system.memory.page_size > 0
    assert all(d.memory.total is None or d.memory.total > 0 for d in system.gpus)


@pytest.mark.parametrize("vendor", [Vendor.NVIDIA, Vendor.AMD, Vendor.INTEL, Vendor.APPLE])
def test_live_vendor(vendor: Vendor) -> None:
    devices = System.detect().by_vendor[vendor]
    if not devices:
        pytest.skip(f"{vendor.name} unavailable to this process on {platform.system()}")
    assert all(d.name and (d.memory.total is None or d.memory.total > 0) for d in devices)


def test_no_commands() -> None:
    for path in Path("src/krn/hw").rglob("*.py"):
        assert "subprocess" not in path.read_text(), path


def test_detection(inventory: System, monkeypatch: pytest.MonkeyPatch) -> None:
    first = ReadingDevice(
        vendor=Vendor.AMD,
        index=0,
        name="AMD",
        source="hip",
        pci_bus_id="bus",
        telemetry_reader=lambda: Telemetry(clock_mhz=1000),
    )
    fallback = ReadingDevice(
        vendor=Vendor.AMD,
        index=0,
        name="AMD",
        source="drm",
        pci_bus_id="bus",
        telemetry_reader=lambda: Telemetry(temperature_c=42),
    )
    extra = ReadingDevice(vendor=Vendor.UNKNOWN, index=1, name="Unknown", source="drm", pci_bus_id="other")
    monkeypatch.setattr(system_module.CPU, "detect", lambda: inventory.cpu)
    monkeypatch.setattr(
        system_module,
        "discovery_order",
        lambda **kwargs: (
            SimpleNamespace(available=lambda: Availability(available=True), discover=lambda: (first,)),
            SimpleNamespace(available=lambda: Availability(available=True), discover=lambda: (fallback, extra)),
        ),
    )
    System.detect.cache_clear()
    try:
        detected = System.detect()
        assert len(detected.devices) == 2
        assert detected.devices[0].telemetry().temperature_c == 42
        assert detected.devices[0].telemetry().clock_mhz == 1000

        def fail() -> tuple[ReadingDevice, ...]:
            raise RuntimeError("driver absent")

        monkeypatch.setattr(
            system_module,
            "discovery_order",
            lambda **kwargs: (SimpleNamespace(available=lambda: Availability(available=True), discover=fail),),
        )
        System.detect.cache_clear()
        assert System.detect().probe_failures
    finally:
        System.detect.cache_clear()


def test_targets_and_plans(inventory: System, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(**{"run.target": None})
    arm_cpu = replace(inventory, cpu=replace(inventory.cpu, architecture="arm64"))
    with pytest.raises(HardwareError, match="cross-compilation"):
        _ = arm_cpu.target
    for feature, arch in (("amx_tile", "spr"), ("avx512f", "avx512")):
        system = replace(inventory, cpu=replace(inventory.cpu, features=frozenset({feature})))
        assert system.target.arch == arch
    configure(**{"run.target": "sm_86"})
    assert inventory.target.arch == "sm_86"
    monkeypatch.setattr(system_module, "available_libraries", lambda system: frozenset({"cublas"}))
    assert inventory.libraries == frozenset({"cublas"})
    amd = ReadingDevice(vendor=Vendor.AMD, index=0, name="GPU", source="hip", architecture="gfx942")
    system = replace(inventory, devices=(amd,))
    assert install_plan(system).extras == ("hip",)
    assert "fp8" in amd.features
    assert not ReadingDevice(vendor=Vendor.UNKNOWN, index=0, name="?", source="drm").features
    cpu = ReadingDevice(vendor=Vendor.INTEL, index=0, name="CPU", source="test", architecture="avx2")
    assert "tensor-cores" not in cpu.features and "fp8" not in cpu.features
    assert cpu.nvlinks() is None
    arm = replace(system, os=replace(system.os, architecture="aarch64"))
    assert install_plan(arm).extras == () and install_plan(arm).notes
    mask = replace(inventory.limits, masks=(("HIP_VISIBLE_DEVICES", "0"),))
    assert replace(system, limits=mask).visible_devices == (amd,)
    intel = ReadingDevice(vendor=Vendor.INTEL, index=0, name="iGPU", source="drm")
    visible = replace(
        inventory, devices=(intel,), limits=replace(inventory.limits, masks=(("ZE_AFFINITY_MASK", "0.1"),))
    )
    assert visible.visible_devices == (intel,)


def test_same_device() -> None:
    def gpu(name: str, pci_bus_id: str | None = None, product_id: int | None = None) -> ReadingDevice:
        return ReadingDevice(
            vendor=Vendor.NVIDIA, index=0, name=name, source="probe", pci_bus_id=pci_bus_id, product_id=product_id
        )

    cuda = gpu("NVIDIA GeForce RTX 5080", "0000:01:00.0")
    assert same_device(cuda, gpu("NVIDIA GeForce RTX 5080", "0000:01:00.0"))
    assert not same_device(cuda, gpu("NVIDIA GeForce RTX 5080", "0000:02:00.0"))
    assert same_device(gpu("NVIDIA GeForce RTX 5080"), gpu("NVIDIA GeForce RTX 5080/PCIe/SSE2"))
    assert same_device(gpu("Intel(R) Graphics (RPL-S)"), gpu("Intel Graphics"))
    assert not same_device(gpu("NVIDIA GeForce RTX 5080"), gpu("NVIDIA GeForce RTX 3090"))
    assert not same_device(gpu("NVIDIA A", product_id=1), gpu("NVIDIA A", product_id=2))
    assert same_device(gpu("Intel(R) Graphics", product_id=0xA780), gpu("INTEL GPU 0xa780", product_id=0xA780))
    assert not same_device(cuda, replace(cuda, vendor=Vendor.AMD))
    assert not same_device(cuda, replace(cuda, kind=DeviceKind.SOFTWARE))


def test_merged_detection(inventory: System, monkeypatch: pytest.MonkeyPatch) -> None:
    drm = ReadingDevice(
        vendor=Vendor.INTEL,
        index=1,
        name="INTEL GPU 0xa780",
        source="drm",
        pci_bus_id="0000:00:02.0",
        product_id=0xA780,
        driver="i915",
        pci=PCILink(width=16),
    )
    twin = replace(drm, index=2, pci_bus_id="0000:03:00.0")
    wgpu = ReadingDevice(
        vendor=Vendor.INTEL,
        index=0,
        name="Intel(R) Graphics (RPL-S)",
        source="wgpu",
        product_id=0xA780,
        architecture="xe-lpg",
        backend="Vulkan",
        memory_facts=DeviceMemory(max_buffer_length=1024),
        telemetry_reader=lambda: Telemetry(utilization_percent=10),
    )
    monkeypatch.setattr(system_module.CPU, "detect", lambda: inventory.cpu)
    available = Availability(available=True)
    monkeypatch.setattr(
        system_module,
        "discovery_order",
        lambda **kwargs: (
            SimpleNamespace(available=lambda: available, discover=lambda: (drm, twin)),
            SimpleNamespace(available=lambda: available, discover=lambda: (wgpu,)),
        ),
    )
    System.detect.cache_clear()
    try:
        first, second = System.detect().devices
    finally:
        System.detect.cache_clear()
    assert first.name == "INTEL GPU 0xa780" and first.driver == "i915" and first.product_id == 0xA780
    assert first.architecture == "xe-lpg" and first.backend == "Vulkan"
    assert first.memory.max_buffer_length == 1024 and first.pci.width == 16
    assert first.telemetry().utilization_percent == 10
    assert second.pci_bus_id == "0000:03:00.0" and second.architecture is None


def test_probe_visibility_and_ordinals(inventory: System) -> None:
    cuda = ReadingDevice(
        vendor=Vendor.NVIDIA, index=0, name="GPU", uuid="GPU-1", source="cuda", architecture="sm_86", ordinal=1
    )
    lowest = replace(cuda, index=1, uuid="GPU-2", architecture="sm_120a", ordinal=0)
    hidden = replace(cuda, index=2, uuid="GPU-3", architecture=None, visible=False, ordinal=None)
    limits = replace(inventory.limits, masks=(("CUDA_VISIBLE_DEVICES", "2"),))
    system = replace(inventory, devices=(cuda, lowest, hidden), limits=limits)
    assert system.visible_devices == (cuda, lowest)
    assert system.default_gpu is lowest and system.targets == (cuda.target, lowest.target)
    software = ReadingDevice(vendor=Vendor.UNKNOWN, kind=DeviceKind.SOFTWARE, index=3, name="llvmpipe", source="wgpu")
    cpu_only = replace(inventory, devices=(software,))
    assert cpu_only.gpus == () and cpu_only.visible_devices == (software,) and cpu_only.default_gpu is None
