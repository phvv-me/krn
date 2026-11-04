import ast
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from patos import value_dispatch

from krn.hw import apple_native, libraries, native, system
from krn.hw.devices import apple, base, webgpu
from krn.hw.devices.base import DeviceMemory, ReadingDevice, Telemetry
from krn.hw.enums import DeviceKind, Vendor
from krn.hw.exceptions import ProbeError
from krn.hw.install import install_plan
from krn.hw.platforms import otherwise


def test_dispatch() -> None:
    @value_dispatch(kind="system")
    def read(value: int) -> int:
        return value

    @read.register("Linux")
    def linux(value: int) -> int:
        return value + 1

    otherwise(read, "Darwin")
    assert read(2, system="Linux") == 3 and read(2, system="Darwin") == 2 and read(2) == 2
    with pytest.raises(ValueError, match="unknown system"):
        read(2, system="other")
    assert len(system.discovery_order(system="Linux")) == 6
    assert len(system.discovery_order(system="Darwin")) == 3
    with pytest.raises(ProbeError, match="known keys"):
        system.discovery_order()


def test_sources_and_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not base.availability("empty", lambda: None)
    failed = base.availability("broken", Mock(side_effect=OSError("absent")))
    assert not failed and "absent" in (failed.reason or "")
    with pytest.raises(ValueError, match="non-negative"):
        DeviceMemory(total=-1)
    with pytest.raises(ValueError, match="non-negative"):
        ReadingDevice(vendor=Vendor.UNKNOWN, name="bad", index=-1, source="test")
    assert base.canonical_pci("00000000:AB:01.0") == "0000:ab:01.0"
    assert base.canonical_pci("BUS") == "bus"
    assert Telemetry(power_w=2).attributed("test").sources == {"power_w": "test"}
    assert getattr(Telemetry().sources, "__setitem__", None) is None
    for architecture in ("sm_86", "avx2"):
        gpu = ReadingDevice(vendor=Vendor.UNKNOWN, name="GPU", index=0, source="test", architecture=architecture)
        assert gpu.features
    monkeypatch.setattr(apple, "mlx", SimpleNamespace(device_info=Mock(side_effect=RuntimeError)))
    assert apple._mlx_info() == {}
    assert apple.AppleDevice(vendor=Vendor.APPLE, name="Apple M4 Pro", index=0, source="metal").target is not None
    assert apple.AppleDevice(vendor=Vendor.APPLE, name="Apple GPU", index=0, source="metal").target is None
    monkeypatch.setattr(apple_native.Libraries, "load", Mock(side_effect=OSError))
    assert apple_native.registry_int("gpu-core-count") is None
    assert apple_native.performance_cores() == (None, None)
    library = SimpleNamespace(sysctlbyname=Mock(return_value=0))
    monkeypatch.setattr(apple_native.Libraries, "load", lambda name: SimpleNamespace(handle=library))
    assert apple_native.performance_cores() == (0, 0)
    library.sysctlbyname.return_value = 1
    assert apple_native.performance_cores() == (None, None)


def test_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(libraries, "find_spec", lambda name: None)
    assert not libraries.importable("missing")
    monkeypatch.setattr(libraries, "find_spec", Mock(side_effect=ImportError))
    assert not libraries.importable("missing.child")
    monkeypatch.setattr(libraries, "importable", lambda name: False)
    monkeypatch.setattr(libraries.Libraries, "available", lambda name: name in {"cublas", "accelerate"})
    assert libraries.available_libraries("Linux") == frozenset({"cublas"})
    assert libraries.available_libraries("Darwin") == frozenset({"cublas", "Accelerate"})
    monkeypatch.setattr(libraries.Libraries, "available", lambda name: name == "cublas")
    assert libraries.available_libraries("Darwin") == frozenset({"cublas"})
    monkeypatch.setattr(libraries, "importable", lambda name: True)
    assert "MetalPerformanceShaders" in libraries.available_libraries("Linux")


def test_webgpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webgpu, "wgpu", None)
    assert not webgpu.WGPUProbe().available() and webgpu.WGPUProbe().discover() == ()
    adapters = [
        SimpleNamespace(info=info, limits={"max-buffer-size": 100})
        for info in (
            {"device": "Apple M4 Pro", "vendor_id": 0, "stackgroup_min_size": 4, "stackgroup_max_size": 128},
            {"vendor_id": Vendor.INTEL, "description": "Intel GPU", "architecture": "xe"},
            {},
        )
    ]
    monkeypatch.setattr(webgpu, "wgpu", SimpleNamespace(gpu=SimpleNamespace(enumerate_adapters_sync=lambda: adapters)))
    assert webgpu.WGPUProbe().available()
    first, second, unknown = webgpu.WGPUProbe().discover()
    assert first.vendor == Vendor.APPLE and first.limits.subgroup_sizes == (4, 128)
    assert first.memory.max_buffer_length == 100 and second.architecture == "xe"
    assert unknown.vendor == Vendor.UNKNOWN


@pytest.mark.parametrize(
    "module,dependency",
    [
        ("host", "psutil"),
        ("host", "archspec.cpu"),
        ("devices.apple", "mlx.core"),
        ("devices.intel", "openvino"),
        ("devices.webgpu", "wgpu"),
        ("devices.generic", "psutil"),
        ("devices.sysman", "pyzes"),
        ("topology", "pyhwloc"),
    ],
)
def test_missing_imports(module: str, dependency: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, f"krn.hw.{module}")
    monkeypatch.setitem(sys.modules, dependency, None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    runpy.run_module(f"krn.hw.{module}")


def test_plans(inventory: system.System) -> None:
    gpu = ReadingDevice(vendor=Vendor.INTEL, kind=DeviceKind.NPU, name="NPU", index=0, source="openvino")
    assert install_plan(replace(inventory, devices=(gpu,))).extras == ("intel",)
    assert install_plan(replace(inventory, devices=(replace(gpu, kind=DeviceKind.CPU),))).extras == ()
    amd_cpu = replace(gpu, vendor=Vendor.AMD, kind=DeviceKind.CPU, name="Threadripper")
    assert install_plan(replace(inventory, devices=(amd_cpu,))).extras == ()
    assert install_plan(replace(inventory, devices=(replace(amd_cpu, kind=DeviceKind.GPU),))).extras == ("hip",)
    assert "probe" in install_plan(inventory).optional_extras
    with pytest.raises(ValueError, match="unknown system"):
        install_plan(replace(inventory, os=replace(inventory.os, system="other")))
    mac = replace(
        inventory,
        devices=(replace(gpu, vendor=Vendor.APPLE),),
        os=replace(inventory.os, system="Darwin", architecture="arm64"),
    )
    assert install_plan(mac).extras == ("metal",)
    assert install_plan(replace(mac, devices=())).extras == ()


def test_readers(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000")
    assert native.cgroup_limits(tmp_path) == (None, None)


def test_no_platform_chains() -> None:
    for path in Path("src/krn/hw").rglob("*.py"):
        source = path.read_text()
        assert "subprocess" not in source and "find_library" not in source
        if path.name in {"dispatch.py", "platforms.py"}:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            compared = ast.unparse(node)
            assert not any(key in compared for key in ("platform.system()", "sys.platform", ".os.system =="))


def test_topology_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("pyhwloc", "pyhwloc.hwobject", "pyhwloc.topology"):
        fake = ModuleType(name)
        fake.__dict__.update(Cache=type("Cache", (), {}), TypeFilter=type("TypeFilter", (), {}))
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.delitem(sys.modules, "krn.hw.topology")
    runpy.run_module("krn.hw.topology")
