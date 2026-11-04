from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from krn.hw import insights
from krn.hw.devices.base import DeviceMemory, ReadingDevice, Telemetry
from krn.hw.enums import DeviceKind, Severity, Vendor
from krn.hw.libraries import LibraryNotFound
from krn.hw.system import System


@pytest.mark.parametrize("rule", insights.RULES)
def test_rules(inventory: System, rule: Callable[[System], tuple[insights.Insight, ...]]) -> None:
    gpu = ReadingDevice(
        vendor=Vendor.NVIDIA,
        name="GPU",
        index=0,
        uuid="GPU-1",
        source="drm",
        memory_facts=DeviceMemory(total=100),
        telemetry_reader=lambda: Telemetry(
            temperature_c=90, slowdown_temperature_c=95, memory_used_bytes=91, throttle_reasons=("hot",)
        ),
    )
    hidden = replace(gpu, index=1, name="Hidden", visible=False)
    software = ReadingDevice(vendor=Vendor.UNKNOWN, kind=DeviceKind.SOFTWARE, index=2, name="llvmpipe", source="wgpu")
    changed = replace(
        inventory,
        devices=(gpu, hidden, software),
        os=replace(inventory.os, rosetta=True),
        limits=replace(inventory.limits, usable_cpus=2),
        volumes=(replace(inventory.volumes[0], free=9),),
    )
    assert rule(changed)
    assert not rule(inventory) or rule is insights.python_build
    assert all(i.code and i.message and i.fix for i in changed.insights())


def test_readings(inventory: System, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insights.Libraries, "load", Mock())  # IOKit exists only on macOS
    gpu = ReadingDevice(
        vendor=Vendor.APPLE,
        name="GPU",
        index=0,
        source="metal",
        architecture="apple_m4",
        memory_facts=DeviceMemory(total=100),
        telemetry_reader=lambda: Telemetry(temperature_c=91, slowdown_temperature_c=95),
    )
    system = replace(inventory, devices=(gpu,))
    assert len(insights.device_health(system)) == 1
    assert not insights.unsupported_devices(system)
    assert not insights.missing_sources(system)


def test_no_health_warning(inventory: System) -> None:
    gpu = ReadingDevice(
        vendor=Vendor.APPLE,
        name="GPU",
        index=0,
        source="metal",
        memory_facts=DeviceMemory(total=100),
        telemetry_reader=lambda: Telemetry(temperature_c=40, slowdown_temperature_c=95, memory_used_bytes=20),
    )
    assert not insights.device_health(replace(inventory, devices=(gpu,)))


@pytest.mark.parametrize(
    ("telemetry", "fragment"),
    [
        (Telemetry(throttle_reasons=("HW_THERMAL_SLOWDOWN",)), "HW_THERMAL_SLOWDOWN active"),
        (Telemetry(throttle_reasons=("HW_POWER_BRAKE_SLOWDOWN",)), "HW_POWER_BRAKE_SLOWDOWN active"),
        (
            Telemetry(temperature_c=91, slowdown_temperature_c=95, utilization_percent=0),
            "91 C is within 5 C of the 95 C slowdown temperature",
        ),
        (
            Telemetry(throttle_reasons=("SW_POWER_CAP",), utilization_percent=80),
            "SW_POWER_CAP active at 80% utilization",
        ),
    ],
)
def test_throttling_names_its_condition(inventory: System, telemetry: Telemetry, fragment: str) -> None:
    gpu = ReadingDevice(vendor=Vendor.NVIDIA, name="GPU", index=0, source="cuda", telemetry_reader=lambda: telemetry)
    (insight,) = insights.device_health(replace(inventory, devices=(gpu,)))
    assert insight.code == "gpu-throttling" and fragment in insight.message


@pytest.mark.parametrize(
    "telemetry",
    [
        Telemetry(throttle_reasons=("SW_POWER_CAP", "HW_SLOWDOWN"), utilization_percent=0, temperature_c=40),
        Telemetry(
            throttle_reasons=("HW_SLOWDOWN",), utilization_percent=90, temperature_c=40, slowdown_temperature_c=95
        ),
        Telemetry(throttle_reasons=("SYNC_BOOST",), utilization_percent=90),
    ],
)
def test_benign_throttle_reasons_stay_silent(inventory: System, telemetry: Telemetry) -> None:
    gpu = ReadingDevice(vendor=Vendor.NVIDIA, name="GPU", index=0, source="cuda", telemetry_reader=lambda: telemetry)
    assert not insights.device_health(replace(inventory, devices=(gpu,)))


def test_software_renderer_is_informational(inventory: System) -> None:
    software = ReadingDevice(vendor=Vendor.UNKNOWN, kind=DeviceKind.SOFTWARE, index=0, name="llvmpipe", source="wgpu")
    (insight,) = insights.software_renderers(replace(inventory, devices=(software,)))
    assert insight.severity == Severity.INFO and insight.subject == "llvmpipe"
    assert not insights.unsupported_devices(replace(inventory, devices=(software,)))


def test_intel_missing_runtimes_once_per_gpu(
    monkeypatch: pytest.MonkeyPatch, inventory: System, tmp_path: Path
) -> None:
    def load(name: str) -> None:
        raise LibraryNotFound(name, (f"lib{name}.so",), (str(tmp_path / name),), f"fix {name}")

    monkeypatch.setattr(insights.Libraries, "load", load)
    igpu = ReadingDevice(vendor=Vendor.INTEL, name="Intel(R) Graphics (RPL-S)", index=0, source="wgpu")
    (insight,) = insights.missing_sources(replace(inventory, devices=(igpu,)))
    assert insight.code == "missing-source" and insight.severity == Severity.WARNING
    assert "'level_zero'" in insight.message and "'opencl'" in insight.message
    assert "fix level_zero" in insight.fix and "fix opencl" in insight.fix
    monkeypatch.setattr(insights.Libraries, "load", lambda name: None)
    (insight,) = insights.missing_sources(replace(inventory, devices=(igpu,)))
    assert insight.message == "GPU detected without its compute binding" and "krn[intel]" in insight.fix
    cpu = ReadingDevice(vendor=Vendor.INTEL, kind=DeviceKind.CPU, index=0, name="Intel CPU", source="openvino")
    assert not insights.missing_sources(replace(inventory, devices=(cpu,)))
