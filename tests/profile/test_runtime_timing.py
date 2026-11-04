from unittest.mock import Mock

import pytest

from krn import Blocks, T
from krn.compiler.api import CPUDevice
from krn.compiler.errors import TargetError
from krn.hw.devices.base import ReadingDevice
from krn.hw.enums import Vendor
from krn.profile import Profiler, launches
from krn.profile.spans import active

from .test_launches import twice


def test_metal_exact_command_timing_and_completion_failure(device: CPUDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    program = device.launcher(twice, Blocks(2, threads=16)).program(x, y)
    gpu = ReadingDevice(name="fake Metal", architecture="apple_m4", source="fake", vendor=Vendor.APPLE, index=0)
    stream = Mock()
    stream.device.target.arch = "apple_m4"
    call = Mock()
    call.program.ir, call.geometry = program.kernel, (2, 1, 1, 16)
    command = Mock()
    command.error.return_value = None
    command.GPUStartTime.return_value = 2.0
    command.GPUEndTime.return_value = 2.000001
    with Profiler(gpus=(gpu,)) as profiler:
        profiler.metal_enqueue(call, stream, command)
    result = profiler.result()
    assert result.kernels[0].name == "twice"
    assert result.kernels[0].duration_ns == 1000
    assert result.launches[0].timing == "metal-gpu" and result.peak is gpu.peak
    command.error.return_value = "GPU failure"
    with pytest.raises(RuntimeError, match="GPU failure"), Profiler(gpus=(gpu,)) as failed:
        failed.metal_enqueue(call, stream, command)
    assert not failed.active and launches.session is None and active() is None


def test_cuda_event_durations_are_not_fabricated_absolute_timestamps(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    program = device.launcher(twice, Blocks(2, threads=16)).program(x, y)
    call, stream = Mock(), Mock()
    call.program.ir, call.grid, call.threads = program.kernel, (2, 1, 1), 16
    stream.device.target.arch = "sm_90"
    before, after = Mock(), Mock()
    after.elapsed_since.return_value = 0.005
    stream.record.side_effect = [before, after]
    with Profiler() as profiler:
        profiler.cuda_run(call, stream)
    record = profiler.result().launches[0]
    assert record.duration_ns == 5000 and record.start_ns is None and record.end_ns is None
    assert record.timing == "cuda-events"
    kernel = profiler.result().kernels[0]
    assert (kernel.name, kernel.duration_ns, kernel.positioned) == ("twice", 5000, False)
    call.program.launch.assert_called_once_with(stream, call.grid, 16, call.packed)
    with pytest.raises(TargetError, match=r"run\.graphs=False"):
        profiler.cuda_graph("sm_90")
