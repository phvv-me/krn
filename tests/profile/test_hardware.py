from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, Items, T
from krn.profile import Feature, Profiler, launches, span
from krn.profile.providers.nvidia.collector import _MAX_RECORDS
from krn.profile.providers.nvidia.delivery import Delivery
from krn.profile.providers.nvidia.tracer import NVTXTracer
from krn.profile.spans import active
from tests.helpers.example_gemm import GemmTiling, gemm
from tests.helpers.example_kernels import saxpy
from tests.helpers.example_softmax import RowTiling, softmax

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDAEvent
    from krn.runtime.metal import MetalDevice

from .test_launches import twice

_THREADS = 32 * 32


@pytest.mark.hardware
def test_real_metal_launches_are_exact_ordered_and_correct(metal_gpu: MetalDevice) -> None:
    x = metal_gpu.to_device(np.full(_THREADS, 3, dtype=np.float32))
    y = metal_gpu.empty(T.f32, _THREADS)
    launch = metal_gpu.launcher(twice, Blocks(16, threads=64)).bind(x, y)
    with Profiler(native=True) as profiler:
        for _ in range(3):
            launch.submit().wait()
        first = profiler.result()
        again = profiler.result()
    result = profiler.result()
    assert (y.to_host() == 6).all()
    assert len(result.launches) == 3 and len(first.launches) == len(again.launches) == 3
    assert [r.name for r in result.launches] == ["twice"] * 3
    assert {r.geometry for r in result.launches} == {(16, 1, 1, 64)}
    for record in result.launches:
        assert record.timing == "metal-gpu" and record.start_ns is not None and record.end_ns is not None
        assert 0 < record.start_ns < record.end_ns
    for earlier, later in pairwise(result.launches):
        assert earlier.start_ns is not None and later.start_ns is not None and earlier.start_ns <= later.start_ns
    record = result.launches[0]
    assert (record.flops, record.bytes_moved, record.flops_by_dtype) == (_THREADS, _THREADS * 8, {"f32": _THREADS})
    assert [k.clock for k in result.kernels] == ["metal"] * 3
    assert all(k.duration_ns > 0 for k in result.kernels)
    assert active() is None and launches.session is None


@pytest.mark.hardware
def test_real_metal_capture_nests_and_names_its_region(metal_gpu: MetalDevice) -> None:
    x = metal_gpu.to_device(np.full(_THREADS, 3, dtype=np.float32))
    y = metal_gpu.empty(T.f32, _THREADS)
    launch = metal_gpu.launcher(twice, Blocks(16, threads=64)).bind(x, y)
    with Profiler(native=True) as profiler:
        launch.submit().wait()
        _, inner = Profiler.capture(lambda: (launch.submit().wait(), 0)[1])
        with span("stage"):
            launch.submit().wait()
    assert len(inner.launches) == 1 and inner.kernels[0].clock == "metal"
    result = profiler.result()
    assert len(result.launches) == 3
    assert [r.region for r in result.launches] == ["", "", "stage"]
    assert [hot.name for hot in result.trace_report().hot_regions if hot.name == "stage"] == ["stage"]


@pytest.mark.hardware
def test_real_metal_failure_inside_a_session_leaves_nothing_behind(metal_gpu: MetalDevice) -> None:
    x = metal_gpu.to_device(np.full(_THREADS, 3, dtype=np.float32))
    y = metal_gpu.empty(T.f32, _THREADS)
    launch = metal_gpu.launcher(twice, Blocks(16, threads=64)).bind(x, y)
    with pytest.raises(ZeroDivisionError), Profiler(native=True) as profiler:
        launch.submit().wait()
        raise ZeroDivisionError
    assert len(profiler.launches) == 1 and not profiler._metal_pending
    assert active() is None and launches.session is None


@pytest.mark.hardware
def test_real_cuda_runtime_events_time_each_launch(gpu: CUDADevice) -> None:
    x, y = gpu.empty(T.f32, 1024), gpu.empty(T.f32, 1024)
    launch = gpu.launcher(twice, Blocks(16, threads=64)).bind(x, y)
    with Profiler() as profiler:
        launch.submit().wait()
    record = profiler.result().launches[0]
    assert record.timing == "cuda-events" and record.elapsed_ns and record.elapsed_ns > 0
    assert record.start_ns is None and record.name == "twice"


@pytest.mark.hardware
def test_real_cuda_capture_has_named_positive_kernel_time(gpu: CUDADevice) -> None:
    x, y = gpu.empty(T.f32, 1024), gpu.empty(T.f32, 1024)
    launch = gpu.launcher(twice, Blocks(16, threads=64)).bind(x, y)
    launch.submit().wait()
    _, profile = Profiler.capture(launch)
    assert any("twice" in record.name and record.duration_ns > 0 for record in profile.kernels)


_TILE = 64
_GEMM_SHAPE = (256, 256, 256)  # m, n, k
_KINDS = Profiler.Activity.KERNEL | Profiler.Activity.MEMCPY | Profiler.Activity.RUNTIME | Profiler.Activity.DRIVER


def _cuda_runs(gpu: CUDADevice) -> tuple[Callable[[], CUDAEvent], ...]:
    """One saxpy, one tiled f16 GEMM and one row softmax, each a bound launch returning its event."""
    n = 1 << 16
    x, y = gpu.to_device(np.ones(n, np.float32)), gpu.to_device(np.ones(n, np.float32))
    m, cols, k = _GEMM_SHAPE
    a, b = gpu.to_device(np.ones((m, k), np.float16)), gpu.to_device(np.ones((k, cols), np.float16))
    c = gpu.to_device(np.zeros((m, cols), np.float32))
    rows = 4
    sx = gpu.to_device(np.ones((rows, 4096), np.float16))
    sy = gpu.to_device(np.zeros((rows, 4096), np.float16))
    saxpy_launch = gpu.launcher(saxpy, Items(n)).bind(T.f32(2.0), x, y)
    gemm_launch = gpu.launcher(gemm, Blocks(cols // _TILE, m // _TILE)).bind(a, b, c, GemmTiling(bm=_TILE, bn=_TILE))
    softmax_launch = gpu.launcher(softmax, Blocks(rows // 4)).bind(sx, sy, RowTiling(rows=4, cols=4096))
    return saxpy_launch.submit, gemm_launch.submit, softmax_launch.submit


@pytest.mark.hardware
def test_real_cupti_kernels_carry_launch_shape_memory_and_device_clock(gpu: CUDADevice) -> None:
    saxpy_run, gemm_run, softmax_run = _cuda_runs(gpu)
    with Profiler(native=True, activities=_KINDS) as profiler:
        copied = gpu.to_device(np.ones(1 << 16, np.float32))
        for run in (saxpy_run, gemm_run, softmax_run):
            run().wait()
        result = profiler.result()
    assert copied.to_host().sum() == 1 << 16
    kernels = {name: next(k for k in result.kernels if name in k.name) for name in ("saxpy", "gemm", "softmax")}
    records = {record.name: record for record in result.launches}
    for name, kernel in kernels.items():
        assert kernel.grid == "x".join(map(str, records[name].geometry[:3])), name
        assert kernel.block == f"{records[name].geometry[3]}x1x1", name
        assert kernel.clock == "cupti" and kernel.positioned and kernel.registers > 0, name
        assert 0 < kernel.start_ns < kernel.end_ns and kernel.device_id is not None and kernel.stream_id is not None
    assert kernels["gemm"].dynamic_shared_mem > 0
    assert len(result.kernels) == 3
    copies = [m for m in result.memcpys if m.kind == "HtoD"]
    assert copies and all(m.bytes_moved > 0 and m.clock == "cupti" for m in copies)


@pytest.mark.hardware
def test_real_cupti_kernels_join_their_launch_calls_by_correlation_id(gpu: CUDADevice) -> None:
    saxpy_run = _cuda_runs(gpu)[0]
    with Profiler(native=True, activities=_KINDS) as profiler:
        for _ in range(4):
            saxpy_run().wait()
        result = profiler.result()
    calls = [a for a in result.activities if "LaunchKernel" in a.name or "LaunchCooperativeKernel" in a.name]
    ids = [kernel.correlation_id for kernel in result.kernels]
    assert len(calls) == len(ids) == 4
    assert len(set(ids)) == 4 and all(i > 0 for i in ids)
    assert sorted(ids) == sorted(call.correlation_id for call in calls)


@pytest.mark.hardware
def test_real_cupti_sessions_run_back_to_back_and_nest_without_leaking_a_subscriber(gpu: CUDADevice) -> None:
    saxpy_run = _cuda_runs(gpu)[0]
    with Profiler(native=True) as first:
        saxpy_run().wait()
        first_kernels = len(first.result().kernels)
    with Profiler(native=True) as second:
        saxpy_run().wait()
        _, inner = Profiler.capture(lambda: (saxpy_run().wait(), 0)[1])
        saxpy_run().wait()
        second_kernels = len(second.result().kernels)
    with Profiler(native=True) as third:
        saxpy_run().wait()
        third_kernels = len(third.result().kernels)
    assert (first_kernels, len(inner.kernels), second_kernels, third_kernels) == (1, 1, 3, 1)
    assert Delivery.active == [] and launches.session is None and active() is None


@pytest.mark.hardware
def test_real_cupti_window_refuses_records_overwritten_by_a_full_buffer(gpu: CUDADevice) -> None:
    saxpy_run = _cuda_runs(gpu)[0]
    with Profiler(native=True) as profiler:
        with pytest.raises(RuntimeError, match="overwritten"):
            Profiler.capture(lambda: [saxpy_run() for _ in range(_MAX_RECORDS + 1000)] and 0)
        _, after = Profiler.capture(lambda: (saxpy_run().wait(), 0)[1])
    assert len(after.kernels) == 1 and profiler.result().dropped_activities > 0


@pytest.mark.hardware
def test_real_cuda_marker_spans_open_nvtx_ranges_around_their_launches(gpu: CUDADevice) -> None:
    saxpy_run = _cuda_runs(gpu)[0]
    with Profiler(native=True, features=Feature.SPANS | Feature.ACTIVITY | Feature.MARKERS) as profiler:
        with span("stage"):
            saxpy_run().wait()
        result = profiler.result()
    assert isinstance(profiler.spans.tracer, NVTXTracer)
    assert [record.region for record in result.launches] == ["stage"]
    assert [window.name for window in result.windows] == ["stage"]
