from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from krn.hw.devices import amd, apple, generic, intel, nvidia, webgpu
from krn.hw.devices.base import Telemetry
from krn.hw.enums import DeviceKind, Vendor
from krn.hw.exceptions import ProbeError
from krn.hw.system import System


def test_apple(monkeypatch: pytest.MonkeyPatch) -> None:
    device = SimpleNamespace(
        name=lambda: "Apple M4",
        registryID=lambda: 10,
        supportsFamily_=lambda family: family <= 1009,
        recommendedMaxWorkingSetSize=lambda: 1000,
        hasUnifiedMemory=lambda: True,
        maxThreadsPerThreadgroup=lambda: SimpleNamespace(width=1024),
        maxThreadgroupMemoryLength=lambda: 32768,
    )
    monkeypatch.setattr(
        apple, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: device, MTLCopyAllDevices=lambda: [device])
    )
    monkeypatch.setattr(apple.native, "gpu_core_count", lambda: 16)
    monkeypatch.setattr(apple.native, "registry_int", lambda *args: 5)
    monkeypatch.setattr(apple, "mlx", SimpleNamespace(device_info=lambda: {}))
    probe = apple.AppleProbe()
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.architecture == "apple_m4" and gpu.limits.compute_units == 16
    assert gpu.memory.total == 1000 and gpu.memory.unified
    assert gpu.telemetry().utilization_percent == 5
    monkeypatch.setattr(
        apple, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: None, MTLCopyAllDevices=lambda: [])
    )
    assert probe.discover() == ()
    monkeypatch.setattr(apple, "Metal", None)
    assert not probe.available() and probe.discover() == ()


def test_nvidia(monkeypatch: pytest.MonkeyPatch, cuda_devices: Callable[[int], None]) -> None:
    properties = SimpleNamespace(
        multiprocessor_count=10,
        warp_size=32,
        max_threads_per_block=1024,
        max_shared_memory_per_block=32768,
        max_registers_per_block=65536,
        clock_rate=1500000,
        integrated=False,
    )
    compute = SimpleNamespace(pci_bus_id="0000:01:00.0", properties=properties, compute_capability=(12, 0))
    management = SimpleNamespace(
        index=0,
        name="NVIDIA",
        uuid="GPU-1",
        pci_bus_id=compute.pci_bus_id,
        memory_info=SimpleNamespace(total=100, used=20),
        numa_node_id=0,
        pci_info=SimpleNamespace(max_link_generation=5, max_link_width=16),
        num_fans=1,
        get_fan=lambda i: SimpleNamespace(speed=40),
        get_clock=lambda kind: SimpleNamespace(get_current_mhz=lambda: 1000, get_max_mhz=lambda: 1500),
        temperature=SimpleNamespace(get_sensor=lambda: 60, get_threshold=lambda kind: 95),
        get_cpu_affinity=lambda: [0, 1],
        get_nvlinks=lambda: [SimpleNamespace(state=True, version=(4, 0))],
        utilization=SimpleNamespace(gpu=50),
        current_clock_event_reasons=[],
        performance_state=0,
        compute_running_processes=[SimpleNamespace(pid=12, used_gpu_memory=20)],
    )
    cuda_devices(1)
    monkeypatch.setattr(nvidia, "core", SimpleNamespace(Device=lambda i: compute))
    monkeypatch.setattr(
        nvidia, "system", SimpleNamespace(Device=lambda **kwargs: management, get_num_devices=lambda: 1)
    )
    probe = nvidia.NvidiaProbe()
    monkeypatch.setattr(nvidia.Libraries, "available", lambda name: True)
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.architecture == "sm_120a" and gpu.limits.subgroup_size == 32
    assert gpu.telemetry().memory_used_bytes == 20
    assert gpu.processes() == (nvidia.Process(pid=12, memory_bytes=20),)
    assert gpu.nvlinks() == (nvidia.NVLink(index=0, active=True, version=(4, 0)),)
    assert gpu.link.cpu_affinity == (0, 1)
    management.memory_info.used = 30
    assert gpu.telemetry().memory_used_bytes == 30
    monkeypatch.setattr(
        nvidia,
        "nvml",
        SimpleNamespace(
            device_get_handle_by_index_v2=lambda index: 1,
            device_get_power_usage=lambda handle: 1000,
            device_get_power_management_limit=lambda handle: 2000,
        ),
    )
    assert gpu.telemetry().sources["power_w"] == "cuda.bindings.nvml"
    assert gpu.telemetry().sources["memory_used_bytes"] == "cuda.core.system"
    monkeypatch.setattr(nvidia, "core", None)
    assert not probe.available() and probe.discover() == ()


def test_amd(monkeypatch: pytest.MonkeyPatch) -> None:
    props = SimpleNamespace(
        name="MI300",
        gcnArchName="gfx942:sramecc+",
        pciDomainID=0,
        pciBusID=1,
        pciDeviceID=0,
        totalGlobalMem=100,
        integrated=1,
        multiProcessorCount=100,
        warpSize=64,
        maxThreadsPerBlock=1024,
        sharedMemPerBlock=65536,
        clockRate=1500000,
    )
    monkeypatch.setattr(
        amd, "hip", SimpleNamespace(hipGetDeviceCount=lambda: 1, hipGetDeviceProperties=lambda i: props)
    )
    probe = amd.AMDProbe()
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.architecture == "gfx942" and gpu.memory.unified
    assert gpu.limits.registers_per_block is None
    assert gpu.telemetry() == Telemetry()
    assert amd._text(b"gfx942") == "gfx942"
    monkeypatch.setattr(
        amd,
        "amdsmi",
        SimpleNamespace(
            amdsmi_init=lambda: None,
            amdsmi_get_processor_handle_from_bdf=lambda bus: 1,
            amdsmi_get_gpu_activity=lambda handle: {"gfx_activity": 50},
            amdsmi_get_clock_info=lambda *args: {"clk": 1000, "max_clk": "N/A"},
            amdsmi_get_gpu_vram_usage=lambda handle: {"vram_used": 20},
            AmdSmiClkType=SimpleNamespace(GFX=0),
        ),
    )
    monkeypatch.setattr(
        amd, "hip", SimpleNamespace(hipGetDeviceCount=lambda: (0, 1), hipGetDeviceProperties=lambda i: (0, props))
    )
    smi_gpu = probe.discover()[0]
    assert smi_gpu.telemetry().utilization_percent == 50
    monkeypatch.setattr(amd, "hip", SimpleNamespace(hipGetDeviceCount=lambda: (1, 0)))
    with pytest.raises(ProbeError):
        probe.discover()
    monkeypatch.setattr(
        amd, "hip", SimpleNamespace(hipGetDeviceCount=lambda: 1, hipGetDeviceProperties=lambda i: (1, props))
    )
    with pytest.raises(ProbeError):
        probe.discover()
    monkeypatch.setattr(amd, "amdsmi", None)
    assert smi_gpu.telemetry() == Telemetry()
    monkeypatch.setattr(amd, "hip", None)
    assert not probe.available() and probe.discover() == ()


def test_intel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel, "openvino", None)
    device = SimpleNamespace(
        name="Intel UHD",
        vendor="Intel",
        global_mem_size=100,
        max_compute_units=32,
        has_aspect_fp64=True,
        sub_group_sizes=[8, 16, 32],
        max_work_group_size=256,
        local_mem_size=32768,
        max_clock_frequency=1000,
    )
    monkeypatch.setattr(intel, "dpctl", SimpleNamespace(get_devices=lambda **kwargs: [device]))
    probe = intel.IntelProbe()
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.vendor == Vendor.INTEL and gpu.limits.compute_units == 32
    monkeypatch.setattr(intel, "dpctl", None)
    assert not probe.available() and probe.discover() == ()


def test_openvino(monkeypatch: pytest.MonkeyPatch) -> None:
    names = {
        "CPU": "AMD Ryzen 9",
        "GPU.0": "Intel device",
        "GPU.1": "NVIDIA GeForce RTX 3090 (dGPU)",
        "NPU": "Intel NPU",
    }
    properties = {
        "DEVICE_ARCHITECTURE": "xe",
        "DEVICE_TYPE": SimpleNamespace(name="INTEGRATED"),
        "DEVICE_PCI_INFO": SimpleNamespace(domain=0, bus=0, device=2, function=0),
        "GPU_DEVICE_TOTAL_MEM_SIZE": 100,
        "NPU_DEVICE_TOTAL_MEM_SIZE": 200,
        "GPU_EXECUTION_UNITS_COUNT": 32,
        "OPTIMIZATION_CAPABILITIES": ["FP16", "INT8"],
    }
    core = SimpleNamespace(
        available_devices=["CPU", "GPU.0", "GPU.1", "NPU", "OTHER"],
        get_property=lambda key, prop: names[key] if prop == "FULL_DEVICE_NAME" else properties[prop],
    )
    monkeypatch.setattr(intel, "openvino", SimpleNamespace(Core=lambda: core))
    monkeypatch.setattr(intel, "dpctl", SimpleNamespace(get_devices=Mock(side_effect=AssertionError)))
    probe = intel.IntelProbe()
    assert probe.available()
    cpu, gpu, npu = probe.discover()
    assert cpu.kind.value == "cpu" and cpu.vendor == Vendor.AMD and cpu.memory.total is None
    assert gpu.vendor == Vendor.INTEL and gpu.memory.total == 100 and gpu.memory.unified
    assert gpu.pci_bus_id == "0000:00:02.0" and npu.vendor == Vendor.INTEL and npu.kind.value == "npu"
    assert "fp16" in npu.features and npu.memory.total == 200
    monkeypatch.setattr(intel, "dpctl", SimpleNamespace(get_devices=lambda device_type: []))
    core.get_property = Mock(side_effect=RuntimeError)
    assert [d.kind.value for d in probe.discover()] == ["cpu", "npu"]
    assert probe.discover()[1].pci_bus_id is None
    monkeypatch.setattr(intel, "dpctl", None)
    monkeypatch.setattr(intel, "openvino", SimpleNamespace(Core=Mock(side_effect=RuntimeError)))
    assert probe.discover() == ()


def test_drm(tmp_path: Path) -> None:
    path = tmp_path / "card0" / "device"
    path.mkdir(parents=True)
    for key, value in {
        "vendor": "0x8086",
        "device": "0x1234",
        "numa_node": "-1",
        "mem_info_vram_total": "100",
        "mem_info_vram_used": "30",
        "gpu_busy_percent": "20",
        "current_link_speed": "8.0 GT/s PCIe",
        "current_link_width": "16",
    }.items():
        (path / key).write_text(value)
    monitor = path / "hwmon" / "hwmon0"
    monitor.mkdir(parents=True)
    for key, value in {"temp1_input": "42000", "power1_average": "1000000", "power1_cap": "2000000"}.items():
        (monitor / key).write_text(value)
    probe = generic.DRMProbe(tmp_path)
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.link.generation == 3 and gpu.link.width == 16 and gpu.link.numa_node is None
    assert gpu.telemetry().temperature_c == 42 and gpu.telemetry().power_w == 1
    (path / "mem_info_vram_used").write_text("50")
    assert gpu.telemetry().memory_used_bytes == 50
    (path / "vendor").write_text("0xabcd")
    assert probe.discover()[0].vendor == Vendor.UNKNOWN
    assert gpu.processes() == ()


def test_nvidia_gaps(monkeypatch: pytest.MonkeyPatch, cuda_devices: Callable[[int], None]) -> None:
    assert nvidia._optional(Mock(side_effect=RuntimeError)) is None
    monkeypatch.setattr(
        nvidia,
        "nvml",
        SimpleNamespace(
            device_get_handle_by_index_v2=lambda i: 1,
            device_get_power_usage=lambda h: 1000,
            device_get_power_management_limit=lambda h: 2000,
        ),
    )
    assert nvidia._power(0, False) == 1 and nvidia._power(0, True) == 2
    management = SimpleNamespace(
        name="GPU",
        uuid="GPU-1",
        pci_bus_id="0000:01:00.0",
        numa_node_id=0,
        get_cpu_affinity=lambda: [],
        memory_info=SimpleNamespace(total=100),
        pci_info=SimpleNamespace(max_link_generation=4, max_link_width=16),
    )
    cuda_devices(1)
    monkeypatch.setattr(nvidia, "core", SimpleNamespace(Device=Mock(side_effect=RuntimeError)))
    monkeypatch.setattr(
        nvidia, "system", SimpleNamespace(Device=lambda **kwargs: management, get_num_devices=lambda: 1)
    )
    (gpu,) = nvidia.NvidiaProbe().discover()
    assert gpu.architecture is None and gpu.limits.compute_units is None and not gpu.visible
    monkeypatch.setattr(nvidia, "core", SimpleNamespace(Device=lambda i: SimpleNamespace(pci_bus_id="other")))
    assert nvidia.NvidiaProbe().discover()[0].architecture is None
    monkeypatch.setattr(nvidia, "system", SimpleNamespace(get_num_devices=Mock(side_effect=nvidia.NvmlError(1))))
    with pytest.raises(ProbeError, match="NVIDIA driver"):
        nvidia.NvidiaProbe().discover()


def test_cuda_count(monkeypatch: pytest.MonkeyPatch) -> None:
    success = SimpleNamespace(CUresult=SimpleNamespace(CUDA_SUCCESS=0))
    monkeypatch.setattr(nvidia, "driver", None)
    assert nvidia._cuda_count() == 0
    monkeypatch.setattr(nvidia, "driver", SimpleNamespace(**vars(success), cuInit=lambda flags: (1,)))
    assert nvidia._cuda_count() == 0
    monkeypatch.setattr(
        nvidia,
        "driver",
        SimpleNamespace(**vars(success), cuInit=lambda flags: (0,), cuDeviceGetCount=lambda: (1, 0)),
    )
    assert nvidia._cuda_count() == 0
    monkeypatch.setattr(
        nvidia,
        "driver",
        SimpleNamespace(**vars(success), cuInit=lambda flags: (0,), cuDeviceGetCount=lambda: (0, 2)),
    )
    assert nvidia._cuda_count() == 2


def _compute(capability: tuple[int, int], pci: str) -> SimpleNamespace:
    properties = SimpleNamespace(
        multiprocessor_count=10,
        warp_size=32,
        max_threads_per_block=1024,
        max_shared_memory_per_block=32768,
        max_registers_per_block=65536,
        clock_rate=1500000,
        integrated=False,
    )
    return SimpleNamespace(pci_bus_id=pci, properties=properties, compute_capability=capability)


def _physical(pci: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        uuid=f"GPU-{pci}",
        pci_bus_id=pci,
        numa_node_id=0,
        get_cpu_affinity=lambda: [],
        memory_info=SimpleNamespace(total=100),
        pci_info=SimpleNamespace(max_link_generation=4, max_link_width=16),
    )


@pytest.mark.parametrize(
    ("exposed", "visible", "architectures", "default"),
    [
        (("0000:02:00.0",), (False, True), (None, "sm_86"), "sm_86"),
        (("0000:01:00.0",), (True, False), ("sm_120a", None), "sm_120a"),
        ((), (False, False), (None, None), None),
        (("0000:02:00.0", "0000:01:00.0"), (True, True), ("sm_120a", "sm_86"), "sm_86"),
    ],
)
def test_nvidia_masks(
    monkeypatch: pytest.MonkeyPatch,
    cuda_devices: Callable[[int], None],
    inventory: System,
    exposed: tuple[str, ...],
    visible: tuple[bool, ...],
    architectures: tuple[str | None, ...],
    default: str | None,
) -> None:
    """NVML lists both cards; the CUDA runtime exposes `exposed` in its own ordinal order."""
    computes = {"0000:01:00.0": _compute((12, 0), "0000:01:00.0"), "0000:02:00.0": _compute((8, 6), "0000:02:00.0")}
    physical = [_physical("0000:01:00.0", "RTX 5080"), _physical("0000:02:00.0", "RTX 3090")]
    monkeypatch.setattr(nvidia, "core", SimpleNamespace(Device=lambda ordinal: computes[exposed[ordinal]]))
    monkeypatch.setattr(
        nvidia, "system", SimpleNamespace(get_num_devices=lambda: 2, Device=lambda index: physical[index])
    )
    cuda_devices(len(exposed))
    devices = nvidia.NvidiaProbe().discover()
    assert tuple(d.name for d in devices) == ("RTX 5080", "RTX 3090")
    assert tuple(d.visible for d in devices) == visible
    assert tuple(d.architecture for d in devices) == architectures
    system = replace(inventory, devices=devices)
    assert (system.default_gpu.architecture if system.default_gpu is not None else None) == default
    assert system.visible_devices == tuple(d for d in devices if d.visible)


def test_drm_connectors(tmp_path: Path) -> None:
    card = tmp_path / "card1"
    (card / "device").mkdir(parents=True)
    (card / "device" / "vendor").write_text("0x8086")
    (card / "device" / "device").write_text("0xa780")
    (card / "gt_act_freq_mhz").write_text("400")
    (card / "gt_RP0_freq_mhz").write_text("1450")
    (tmp_path / "drivers").mkdir()
    (tmp_path / "drivers" / "i915").mkdir()
    (card / "device" / "driver").symlink_to(tmp_path / "drivers" / "i915", target_is_directory=True)
    connector = tmp_path / "card1-DP-1"
    connector.mkdir()
    (connector / "device").symlink_to(card, target_is_directory=True)
    (tmp_path / "card2").mkdir()
    probe = generic.DRMProbe(tmp_path)
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.vendor == Vendor.INTEL and gpu.index == 1 and gpu.driver == "i915" and gpu.product_id == 0xA780
    assert gpu.limits.max_clock_mhz == 1450 and gpu.telemetry().clock_mhz == 400
    assert gpu.telemetry().sources["clock_mhz"] == "drm/gt"
    assert not generic.DRMProbe(tmp_path / "missing").available()


def _adapter(info: dict[str, str | int], limits: dict[str, int] | None = None) -> SimpleNamespace:
    return SimpleNamespace(info=info, limits=limits or {"max-buffer-size": 1024})


def test_wgpu_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    adapters = [
        _adapter(
            {
                "device": "NVIDIA GeForce RTX 5080",
                "vendor_id": 0x10DE,
                "adapter_type": "DiscreteGPU",
                "backend_type": "Vulkan",
                "device_id": 0x2C02,
                "vendor": "NVIDIA",
                "architecture": "",
                "description": "driver",
                "stackgroup_min_size": 4,
                "stackgroup_max_size": 128,
            }
        ),
        _adapter(
            {
                "device": "NVIDIA GeForce RTX 5080/PCIe/SSE2",
                "vendor_id": 0x10DE,
                "adapter_type": "DiscreteGPU",
                "backend_type": "OpenGL",
                "device_id": 0,
            }
        ),
        _adapter(
            {
                "device": "llvmpipe (LLVM 20.1, 256 bits)",
                "vendor_id": 0x10005,
                "adapter_type": "CPU",
                "backend_type": "Vulkan",
                "device_id": 0,
            }
        ),
        _adapter(
            {
                "device": "Intel(R) Graphics (RPL-S)",
                "vendor_id": 0x8086,
                "device_id": 0xA780,
                "adapter_type": "IntegratedGPU",
                "backend_type": "Vulkan",
                "architecture": "xe-lpg",
            }
        ),
        _adapter(
            {
                "device": "SwiftShader Device (Subzero)",
                "vendor_id": 0,
                "adapter_type": "Unknown",
                "backend_type": "Vulkan",
            }
        ),
        _adapter(
            {
                "device": "Apple M4 Pro",
                "vendor_id": 0,
                "vendor": "",
                "adapter_type": "IntegratedGPU",
                "backend_type": "Metal",
            }
        ),
    ]
    monkeypatch.setattr(webgpu, "wgpu", SimpleNamespace(gpu=SimpleNamespace(enumerate_adapters_sync=lambda: adapters)))
    probe = webgpu.WGPUProbe()
    assert probe.available()
    nvidia_vulkan, nvidia_gl, software, intel_gpu, swiftshader, apple = probe.discover()
    assert (nvidia_vulkan.vendor, nvidia_vulkan.kind, nvidia_vulkan.product_id) == (
        Vendor.NVIDIA,
        DeviceKind.GPU,
        0x2C02,
    )
    assert nvidia_gl.product_id is None and nvidia_gl.backend == "OpenGL"
    assert software.kind == DeviceKind.SOFTWARE and software.vendor == Vendor.UNKNOWN
    assert swiftshader.kind == DeviceKind.SOFTWARE
    assert (intel_gpu.vendor, intel_gpu.product_id, intel_gpu.architecture) == (Vendor.INTEL, 0xA780, "xe-lpg")
    assert apple.vendor == Vendor.APPLE and apple.kind == DeviceKind.GPU and apple.memory.max_buffer_length == 1024
    monkeypatch.setattr(webgpu, "wgpu", None)
    assert not probe.available() and probe.discover() == ()


def test_smi_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    assert amd._smi(Mock(side_effect=RuntimeError)) is None
    monkeypatch.setattr(amd, "amdsmi", SimpleNamespace(amdsmi_init=Mock(side_effect=RuntimeError)))
    assert amd._reader("bus")() == Telemetry()


def test_apple_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    device = SimpleNamespace(
        name=lambda: "Apple GPU",
        registryID=lambda: 1,
        supportsFamily_=lambda family: False,
        recommendedMaxWorkingSetSize=lambda: 100,
        hasUnifiedMemory=lambda: False,
        maxThreadsPerThreadgroup=lambda: SimpleNamespace(width=512),
        maxThreadgroupMemoryLength=lambda: 16384,
    )
    monkeypatch.setattr(
        apple, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: device, MTLCopyAllDevices=lambda: [])
    )
    monkeypatch.setattr(apple, "mlx", None)
    monkeypatch.setattr(apple.native, "gpu_core_count", lambda: None)
    assert apple.AppleProbe().discover()[0].architecture is None
    device.name = lambda: "AMD Radeon"
    assert apple.AppleProbe().discover()[0].vendor == Vendor.AMD


def test_amd_management_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        amdsmi_init=lambda: None,
        amdsmi_get_processor_handles=lambda: [1],
        amdsmi_get_gpu_asic_info=lambda handle: {"market_name": "MI300", "target_graphics_version": "gfx942"},
        amdsmi_get_gpu_device_uuid=lambda handle: "AMD-1",
        amdsmi_get_gpu_device_bdf=lambda handle: "0000:01:00.0",
        amdsmi_get_gpu_memory_total=lambda handle, memory: 100,
        AmdSmiMemoryType=SimpleNamespace(VRAM=0),
    )
    monkeypatch.setattr(amd, "amdsmi", fake)
    monkeypatch.setattr(amd, "hip", None)
    probe = amd.AMDProbe()
    assert probe.available()
    (gpu,) = probe.discover()
    assert gpu.name == "MI300" and gpu.uuid == "AMD-1" and gpu.memory.total == 100
    monkeypatch.setattr(fake, "amdsmi_get_gpu_asic_info", lambda handle: {})
    assert probe.discover()[0].name == "AMD GPU" and probe.discover()[0].architecture is None
    monkeypatch.setattr(fake, "amdsmi_get_processor_handles", Mock(side_effect=OSError))
    assert not probe.available()
