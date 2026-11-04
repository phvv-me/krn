"""Roofline facts use cited tables, never storage capabilities as rates."""

import logging
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from math import isfinite
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn.facts.peaks import DEVICE_RATES, RATES
from krn.facts.targets import registry
from krn.hw.devices import amd, apple, nvidia
from krn.hw.devices.base import DeviceLimits, DeviceMemory, ReadingDevice, Telemetry
from krn.hw.enums import Vendor
from krn.hw.peaks import PeakRates, PeakSource, bandwidth, positive, resolve
from krn.hw.system import System


@given(st.sampled_from(tuple(RATES)), st.integers(1, 512), st.floats(1, 5000))
def test_table(arch: str, units: int, clock: float) -> None:
    assert set(RATES) == set(registry())
    result = resolve(arch, "unknown", units, clock, PeakRates())
    row = RATES[arch]
    if row is None:
        assert result == PeakRates()
        return
    assert row.reference.startswith("https://")
    assert all(isfinite(rate) and rate > 0 for rate in row.operations_per_cycle.values())
    assert result.flops_sources == dict.fromkeys(result.flops, PeakSource.TABLE)
    for low, high in (("f64", "f32"), ("tf32", "f16"), ("f16", "fp8")):
        if low in result.flops and high in result.flops:
            assert result.flops[low] <= result.flops[high]
    for dtype, ops in row.operations_per_cycle.items():
        assert result.flops[dtype] == ops * units * clock / 1000


@pytest.mark.parametrize("arch", tuple(RATES))
def test_every_target_explicit(arch: str) -> None:
    row = RATES[arch]
    device = ReadingDevice(
        vendor=Vendor.UNKNOWN,
        index=0,
        name="GPU",
        source="fake",
        architecture=arch,
        limit_facts=DeviceLimits(compute_units=2, max_clock_mhz=1000),
    )
    assert device.arch_key == arch
    assert bool(device.peak.flops) == (row is not None)
    assert device.peak.memory_bandwidth_gbs is None


@given(st.sampled_from(tuple(DEVICE_RATES)), st.integers(1, 512), st.floats(1, 5000))
def test_product_table(key: tuple[str, str], units: int, clock: float) -> None:
    arch, name = key
    rates = resolve(arch, f"NVIDIA GeForce {name}", units, clock, PeakRates())
    row = DEVICE_RATES[key]
    assert row.reference.startswith("https://")
    assert all(value > 0 and isfinite(value) for value in row.operations_per_cycle.values())
    assert rates.flops["f64"] <= rates.flops["f32"] if "f64" in rates.flops else True
    assert rates.flops["tf32"] <= rates.flops["f16"]
    if "fp8" in rates.flops:
        assert rates.flops["f16"] <= rates.flops["fp8"]
    for dtype, ops in row.operations_per_cycle.items():
        assert rates.flops[dtype] == ops * units * clock / 1000
    assert rates.flops_sources == dict.fromkeys(rates.flops, PeakSource.TABLE)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_rates(value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        PeakRates(flops={"f32": value}, flops_sources={"f32": PeakSource.TABLE})
    with pytest.raises(ValueError, match="positive and finite"):
        PeakRates(memory_bandwidth_gbs=value, memory_bandwidth_source=PeakSource.TABLE)


def test_provenance_and_immutability() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        PeakRates(memory_bandwidth_gbs=1)
    with pytest.raises(ValueError, match="supplied together"):
        PeakRates(memory_bandwidth_source=PeakSource.TABLE)
    with pytest.raises(ValueError, match="exactly one source"):
        PeakRates(flops={"f32": 1})
    values = {"f32": 1.0}
    sources = {"f32": PeakSource.LIBRARY}
    library = PeakRates(
        flops=values, flops_sources=sources, memory_bandwidth_gbs=42, memory_bandwidth_source=PeakSource.LIBRARY
    )
    values["f32"] = 2
    sources.clear()
    result = resolve("sm_90a", "Apple M4 Pro", 2, 1000, library)
    assert result.flops["f32"] == 1 and result.flops_sources["f32"] == PeakSource.LIBRARY
    assert result.flops["f16"] == 8192 and result.flops_sources["f16"] == PeakSource.TABLE
    assert result.memory_bandwidth_gbs == 42 and result.memory_bandwidth_source == PeakSource.LIBRARY
    assert isinstance(result.flops, type(RATES)) and isinstance(result.flops_sources, type(RATES))
    with pytest.raises(FrozenInstanceError):
        setattr(result, "memory_bandwidth_gbs".lower(), 2)
    assert PeakSource.MEASURED.value == "measured"


@pytest.mark.parametrize(("units", "clock"), [(None, 1000), (1, None), (0, 1000), (1, 0)])
def test_incomplete_device(units: int | None, clock: float | None) -> None:
    assert not resolve("sm_86", "unknown", units, clock, PeakRates()).flops


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), float("nan"), 2**64 - 1])
def test_unavailable_number(value: float | None) -> None:
    assert positive(lambda: value) is None


@pytest.mark.parametrize("error", [AttributeError, KeyError, RuntimeError, OSError, ValueError, TypeError])
def test_failed_number(error: type[Exception]) -> None:
    assert positive(Mock(side_effect=error)) is None


def test_bandwidth_and_snapshot() -> None:
    assert bandwidth(None, 256) == bandwidth(1000, None) == PeakRates()
    assert bandwidth(9_750_000, 384).memory_bandwidth_gbs == 936
    reader = Mock(
        return_value=Telemetry(
            utilization_percent=50,
            temperature_c=60,
            power_w=100,
            memory_used_bytes=20,
            memory_total_bytes=100,
            memory_free_bytes=80,
            memory_utilization_percent=30,
        ).attributed("fake")
    )
    device = ReadingDevice(vendor=Vendor.UNKNOWN, index=0, name="GPU", source="fake", telemetry_reader=reader)
    assert device.arch_key == "unknown" and device.coherent is None
    assert device.peak is device.peak
    assert device.snapshot() is reader.return_value
    reader.assert_called_once_with()
    assert device.snapshot().sources["memory_utilization_percent"] == "fake"
    assert replace(device, architecture="ZEN.4").arch_key == "zen_4"
    assert replace(device, memory_facts=DeviceMemory(unified=False)).coherent is False
    assert replace(device, coherence_facts=True).coherent is True
    assert replace(device, coherence_facts=False).coherent is False


def _cuda(
    monkeypatch: pytest.MonkeyPatch,
    cuda_devices: Callable[[int], None],
    name: str,
    integrated: bool,
    host_tables: bool = False,
    direct_access: bool = False,
) -> nvidia.NvidiaDevice:
    props = SimpleNamespace(
        multiprocessor_count=82,
        warp_size=32,
        max_threads_per_block=1024,
        max_shared_memory_per_block=49152,
        max_registers_per_block=65536,
        clock_rate=1695000,
        integrated=integrated,
        memory_clock_rate=9750000,
        global_memory_bus_width=384,
        pageable_memory_access_uses_host_page_tables=host_tables,
        direct_managed_mem_access_from_host=direct_access,
    )
    compute = SimpleNamespace(pci_bus_id="0000:01:00.0", properties=props, compute_capability=(8, 6))
    management = SimpleNamespace(
        index=0,
        name=name,
        uuid="GPU-1",
        pci_bus_id=compute.pci_bus_id,
        memory_info=SimpleNamespace(total=100, used=20, free=80),
        utilization=SimpleNamespace(gpu=50, memory=30),
        pci_info=SimpleNamespace(max_link_generation=4, max_link_width=16),
    )
    cuda_devices(1)
    monkeypatch.setattr(nvidia, "core", SimpleNamespace(Device=lambda i: compute))
    monkeypatch.setattr(nvidia, "system", SimpleNamespace(get_num_devices=lambda: 1, Device=lambda **kw: management))
    return nvidia.NvidiaProbe().discover()[0]


@pytest.mark.parametrize(
    ("name", "integrated", "coherent"),
    [
        ("RTX 3090", False, False),
        ("GH200", False, True),
        ("GB200", True, True),
        ("unknown", True, None),
    ],
)
def test_cuda_library(
    monkeypatch: pytest.MonkeyPatch,
    cuda_devices: Callable[[int], None],
    name: str,
    integrated: bool,
    coherent: bool | None,
) -> None:
    device = _cuda(monkeypatch, cuda_devices, name, integrated)
    assert device.peak.memory_bandwidth_gbs == 936
    assert device.peak.memory_bandwidth_source == PeakSource.LIBRARY
    assert device.peak.flops["f32"] == pytest.approx(35581.44)
    assert device.peak.flops_sources["f32"] == PeakSource.TABLE
    assert device.coherent is coherent
    sample = device.snapshot()
    assert sample.memory_total_bytes == 100 and sample.memory_free_bytes == 80
    assert sample.memory_utilization_percent == 30


@pytest.mark.parametrize("direct_access", [False, True])
def test_cuda_ats(monkeypatch: pytest.MonkeyPatch, cuda_devices: Callable[[int], None], direct_access: bool) -> None:
    device = _cuda(monkeypatch, cuda_devices, "NVIDIA B200", False, host_tables=True, direct_access=direct_access)
    assert device.coherent is direct_access


def _smi(monkeypatch: pytest.MonkeyPatch, name: str) -> SimpleNamespace:
    fake = SimpleNamespace(
        amdsmi_init=lambda: None,
        amdsmi_get_processor_handles=lambda: [1],
        amdsmi_get_gpu_asic_info=lambda handle: {"market_name": name, "target_graphics_version": "gfx942"},
        amdsmi_get_gpu_device_uuid=lambda handle: "AMD-1",
        amdsmi_get_gpu_device_bdf=lambda handle: "0000:01:00.0",
        amdsmi_get_gpu_memory_total=lambda *args: 100,
        AmdSmiMemoryType=SimpleNamespace(VRAM=0),
        amdsmi_get_gpu_metrics_info=lambda handle: {"vram_max_bandwidth": 5300},
        amdsmi_get_processor_handle_from_bdf=lambda bus: 1,
        amdsmi_get_gpu_activity=lambda handle: {"gfx_activity": 50, "umc_activity": 30},
        amdsmi_get_clock_info=lambda *args: {},
        AmdSmiClkType=SimpleNamespace(GFX=0),
    )
    monkeypatch.setattr(amd, "amdsmi", fake)
    return fake


def test_smi_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd, "hip", None)
    fake = _smi(monkeypatch, "AMD Instinct MI300A")
    (device,) = amd.AMDProbe().discover()
    assert device.coherent is True
    assert device.peak.memory_bandwidth_gbs == 5300
    assert device.peak.memory_bandwidth_source == PeakSource.LIBRARY
    assert device.snapshot().memory_utilization_percent == 30
    monkeypatch.setattr(fake, "amdsmi_get_gpu_metrics_info", lambda handle: {})
    assert amd.AMDProbe().discover()[0].peak.memory_bandwidth_gbs is None
    monkeypatch.setattr(amd, "amdsmi", None)
    assert amd._peak(Mock()) == PeakRates()


@pytest.mark.parametrize(
    ("name", "integrated", "coherent"),
    [
        ("MI300A", True, True),
        ("MI300X", False, False),
        ("unknown", True, None),
    ],
)
def test_hip_library(monkeypatch: pytest.MonkeyPatch, name: str, integrated: bool, coherent: bool | None) -> None:
    fake = _smi(monkeypatch, name)
    props = SimpleNamespace(
        name=name,
        gcnArchName=b"gfx942:sramecc+",
        pciDomainID=0,
        pciBusID=1,
        pciDeviceID=0,
        totalGlobalMem=100,
        integrated=integrated,
        multiProcessorCount=304,
        warpSize=64,
        maxThreadsPerBlock=1024,
        sharedMemPerBlock=65536,
        clockRate=2100000,
        memoryClockRate=2600000,
        memoryBusWidth=8192,
    )
    monkeypatch.setattr(
        amd, "hip", SimpleNamespace(hipGetDeviceCount=lambda: 1, hipGetDeviceProperties=lambda i: props)
    )
    (device,) = amd.AMDProbe().discover()
    assert device.coherent is coherent and device.arch_key == "gfx942"
    assert device.peak.memory_bandwidth_gbs == 5300  # SMI takes precedence over HIP.
    assert device.peak.flops["fp8"] == pytest.approx(2614886.4)
    monkeypatch.setattr(fake, "amdsmi_get_gpu_metrics_info", lambda handle: {})
    assert amd.AMDProbe().discover()[0].peak.memory_bandwidth_gbs == pytest.approx(5324.8)


def test_metal_library(monkeypatch: pytest.MonkeyPatch) -> None:
    metal = SimpleNamespace(
        name=lambda: "Apple M4 Pro",
        registryID=lambda: 1,
        supportsFamily_=lambda family: family <= 1009,
        hasUnifiedMemory=lambda: True,
        recommendedMaxWorkingSetSize=lambda: 100,
        maxThreadsPerThreadgroup=lambda: SimpleNamespace(width=1024),
        maxThreadgroupMemoryLength=lambda: 32768,
    )
    monkeypatch.setattr(
        apple,
        "Metal",
        SimpleNamespace(
            MTLCreateSystemDefaultDevice=lambda: metal,
            MTLCopyAllDevices=lambda: [metal],
        ),
    )
    monkeypatch.setattr(apple, "mlx", SimpleNamespace(device_info=lambda: {"architecture": "applegpu_g16s"}))
    monkeypatch.setattr(apple.native, "gpu_core_count", lambda: 16)
    (device,) = apple.AppleProbe().discover()
    assert device.arch_key == "apple_m4" and device.coherent is True
    assert device.architecture == "applegpu_g16s"
    assert device.peak.memory_bandwidth_gbs == 273 and device.peak.memory_bandwidth_source == PeakSource.TABLE
    assert device.peak.flops == {}
    metal.name = lambda: "Apple M5"
    assert apple.AppleProbe().discover()[0].peak == PeakRates()


@pytest.mark.hardware
def test_live_roofline_facts() -> None:
    devices = System.detect().gpus
    assert devices
    for device in devices:
        assert device.arch_key and "." not in device.arch_key
        peak = device.peak
        logging.getLogger(__name__).info(
            "%s arch=%s units=%s clock_MHz=%s coherent=%s peak=%s",
            device.name,
            device.arch_key,
            device.limits.compute_units,
            device.limits.max_clock_mhz,
            device.coherent,
            peak,
        )
        assert peak is device.peak
        if peak.memory_bandwidth_gbs is not None:
            assert peak.memory_bandwidth_gbs > 0 and peak.memory_bandwidth_source is not None
        assert all(value > 0 for value in peak.flops.values())
        assert set(peak.flops) == set(peak.flops_sources)
        if device.arch_key in {"sm_86", "sm_90a", "sm_120a"}:
            assert peak.memory_bandwidth_gbs is not None and peak.flops
        if device.name == "Apple M4 Pro":
            assert peak.memory_bandwidth_gbs == 273 and device.coherent is True
        snapshot = device.snapshot()
        if snapshot.utilization_percent is not None:
            assert 0 <= snapshot.utilization_percent <= 100
        if snapshot.memory_used_bytes is not None:
            assert snapshot.memory_used_bytes >= 0
