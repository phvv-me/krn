import ctypes
import threading
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest

from krn.compiler.errors import TargetError
from krn.hw.devices.base import ReadingDevice
from krn.hw.enums import Vendor
from krn.hw.system import System
from krn.profile import Activity, Feature, KernelTrace, MemcpyTrace, Profiler, Tracer
from krn.profile.launches import LaunchRecord
from krn.profile.providers.nvidia import bindings
from krn.profile.providers.nvidia import collector as collector_module
from krn.profile.providers.nvidia import tracer as tracer_module
from krn.profile.providers.nvidia.collector import CUPTICollector
from krn.profile.providers.nvidia.constants import CONCURRENT_KERNEL, CUPTI_KIND, MEMCPY
from krn.profile.providers.nvidia.delivery import Delivery, RawGeneric
from krn.profile.providers.nvidia.tracer import NVTXTracer


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    enabled: list[int] = []
    disabled: list[int] = []
    fake = SimpleNamespace(
        ActivityKind=SimpleNamespace(**{name: index for index, name in enumerate(CUPTI_KIND.values(), 1)}),
        CallbackDomain=SimpleNamespace(RUNTIME_API=1, DRIVER_API=2),
        activity_enable=enabled.append,
        activity_disable=disabled.append,
        activity_flush_all=lambda _: None,
        activity_register_callbacks=lambda *args: None,
        activity_get_num_dropped_records=lambda context, stream, pointer: None,
        get_device_id=lambda _: 0,
        get_context_id=lambda _: 1,
        get_callback_name=lambda domain, cbid: "callback",
        get_timestamp=lambda: 10,
    )
    runtime = SimpleNamespace(
        cudaDeviceSynchronize=lambda: (0,), cudaGetDevice=lambda: (0, 0), cudaError_t=SimpleNamespace(cudaSuccess=0)
    )
    driver = SimpleNamespace(cuCtxGetCurrent=lambda: (0, 1))
    monkeypatch.setattr(bindings, "cupti", fake, raising=False)
    monkeypatch.setattr(bindings, "runtime", runtime, raising=False)
    monkeypatch.setattr(bindings, "driver", driver, raising=False)
    monkeypatch.setattr(bindings, "has_cupti", True)
    monkeypatch.setattr(bindings, "has_cuda", True)
    monkeypatch.setattr(Delivery, "active", [])
    monkeypatch.setattr(Delivery, "registered", False)
    monkeypatch.setattr(Delivery, "labels", {})
    monkeypatch.setattr(Delivery, "domains", {})
    monkeypatch.setattr(tracer_module, "_supported_kinds", None)
    fake.enabled, fake.disabled = enabled, disabled
    return fake


def kernel_activity(**extra):
    fields = dict(
        kind=CONCURRENT_KERNEL,
        name="gemm",
        start=100,
        end=200,
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        static_shared_memory=0,
        dynamic_shared_memory=0,
        registers_per_thread=16,
        device_id=0,
        context_id=1,
        stream_id=2,
    )
    return SimpleNamespace(**(fields | extra))


def test_cupti_delivers_typed_records_and_preserves_windows(api, monkeypatch: pytest.MonkeyPatch) -> None:
    assert NVTXTracer.is_available()
    tracer = NVTXTracer()
    assert tracer.timestamp() == 10
    assert tracer.clock() == "cupti"
    assert tracer.supported() is Activity.ALL
    assert tracer.supported() is Activity.ALL
    with tracer.collect(Activity.DEFAULT) as collector:
        assert isinstance(collector, CUPTICollector)
        assert collector.device_index == 0
        assert collector.checkpoint(Activity.DEFAULT) == (0, 0)
        Delivery.completed([kernel_activity(correlation_id=7)])
        copy = SimpleNamespace(
            kind=MEMCPY, start=90, end=100, copy_kind=1, bytes=64, device_id=0, context_id=1, stream_id=2
        )
        Delivery.completed([copy])
        Delivery.labels[99] = "runtime"
        Delivery.domains[99] = 1
        Delivery.completed([SimpleNamespace(kind=99, start=1, end=2, cbid=3)])
        assert collector.checkpoint(Activity.KERNEL) == (3, 0)
        assert collector.kernels(since=0, until=1)[0].name == "gemm"
        assert collector.memcpys(since=0)[0].bandwidth_gbps == 6.4
        assert collector.activities()[0].name == "callback"
        collector.scope = (0, 0, 2)
        with pytest.raises(RuntimeError, match="foreign or unknown"):
            collector.kernels(since=0)
        collector.scope = (0, 0, 1)
        assert collector.dropped() == 0
        collector.reset()
        assert collector.kernels() == []
        with pytest.raises(RuntimeError, match="reset or overwritten"):
            collector.kernels(since=0)
        with pytest.raises(RuntimeError, match="nested"):
            CUPTICollector().__enter__()
    collector.stop()
    assert not Delivery.active and not collector.running
    assert Delivery.requested() == (8 * 1024 * 1024, 0)
    Delivery.completed([kernel_activity()])


def test_cupti_start_errors_and_context_checks(api, monkeypatch: pytest.MonkeyPatch) -> None:
    collector = CUPTICollector(Activity.KERNEL)
    with pytest.raises(RuntimeError):
        _ = collector.device_index
    with pytest.raises(RuntimeError, match="running collector"):
        collector.checkpoint(Activity.KERNEL)
    monkeypatch.setattr(bindings, "has_cuda", False)
    with pytest.raises(RuntimeError, match="required"):
        collector_module._synchronize()
    with pytest.raises(RuntimeError, match="required"):
        collector._scope()
    monkeypatch.setattr(bindings, "has_cuda", True)
    monkeypatch.setattr(bindings.runtime, "cudaDeviceSynchronize", lambda: (2,))
    with pytest.raises(RuntimeError, match="status 2"):
        collector.__enter__()
    assert not collector.running and collector.enabled_kinds == ()
    monkeypatch.setattr(bindings.runtime, "cudaDeviceSynchronize", lambda: (0,))
    monkeypatch.setattr(bindings.runtime, "cudaGetDevice", lambda: (2, 0))
    with pytest.raises(RuntimeError, match="cudaGetDevice"):
        collector._scope()
    monkeypatch.setattr(bindings.runtime, "cudaGetDevice", lambda: (0, 0))
    monkeypatch.setattr(bindings.driver, "cuCtxGetCurrent", lambda: (1, 0))
    with pytest.raises(RuntimeError, match="cuCtxGetCurrent"):
        collector._scope()
    monkeypatch.setattr(bindings.driver, "cuCtxGetCurrent", lambda: (0, 1))
    with collector:
        with pytest.raises(RuntimeError, match="not actually enabled"):
            collector.checkpoint(Activity.MEMCPY)
        collector.owner_thread = -1
        with pytest.raises(RuntimeError, match="issuing thread"):
            collector.flush()
        collector.owner_thread = threading.get_ident()
        collector.scope = (0, 0, 2)
        with pytest.raises(RuntimeError, match="issuing thread"):
            collector.flush()
        collector.scope = (0, 0, 1)
        collector.callback_failed = True
        with pytest.raises(RuntimeError, match="tainted"):
            collector.flush()
        collector.callback_failed = False


def test_cupti_loss_validation_and_generic_names(api, monkeypatch: pytest.MonkeyPatch) -> None:
    collector = CUPTICollector()
    collector.scope = (0, 0, 1)
    collector.records = deque(maxlen=1)
    valid = KernelTrace(name="a", start_ns=1, end_ns=2, device_id=0, context_id=1, stream_id=1)
    collector.append(valid)
    collector.append(valid)
    assert collector.dropped() == 1 and collector.lost_records == 0
    with pytest.raises(RuntimeError, match="overwritten"):
        collector._records(since=0, until=None)
    assert len(collector._records(since=1, until=None)) == 1
    collector._checked((valid,))
    generic = RawGeneric(kind_id=1, kind="runtime", name="named", cbid=None, start_ns=0, end_ns=1, correlation_id=0)
    collector._checked((generic,))
    assert collector.activity_name(generic) == "named"
    assert collector.activity_name(replace(generic, name=None)) == "runtime"
    Delivery.domains[1] = 2
    monkeypatch.setattr(bindings, "has_cupti", False)
    assert collector.activity_name(replace(generic, name=None, cbid=1)) == "runtime"
    for bad in (replace(valid, start_ns=0), replace(valid, stream_id=None), replace(valid, end_ns=0)):
        with pytest.raises(RuntimeError, match="incomplete"):
            collector._checked((bad,))
    collector.scope = None
    with pytest.raises(RuntimeError, match="foreign"):
        collector._checked((valid,))
    assert Delivery._copied(SimpleNamespace(kind=999)) is None
    assert MemcpyTrace.from_activity(SimpleNamespace(start=0, end=1, copy_kind=999)).kind == "kind_999"
    collector._retire()

    def dropped(context: int, stream: int, address: int) -> None:
        ctypes.c_size_t.from_address(address).value = 3

    monkeypatch.setattr(bindings.cupti, "activity_get_num_dropped_records", dropped)
    collector.flush()
    assert collector.dropped() == 4


def test_nvtx_markers_capabilities_and_missing_extras(api, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bindings, "nvtx", SimpleNamespace(start_range=lambda name: name, end_range=calls.append), raising=False
    )
    monkeypatch.setattr(bindings, "has_nvtx", True)
    NVTXTracer().start("outer")()
    assert calls == ["outer"]
    monkeypatch.setattr(bindings, "has_nvtx", False)
    with pytest.raises(TargetError, match=r"krn\[perf\]"):
        NVTXTracer().start("missing")
    monkeypatch.setattr(bindings, "has_cupti", False)
    assert not NVTXTracer.is_available()
    with pytest.raises(TargetError, match=r"krn\[perf\]"):
        NVTXTracer().collect()
    assert NVTXTracer().timestamp() > 0
    assert NVTXTracer().clock() == "host"
    monkeypatch.setattr(bindings, "has_cupti", True)
    first = next(iter(CUPTI_KIND.values()))

    def enable(kind: int) -> None:
        if kind == getattr(api.ActivityKind, first):
            raise NotImplementedError

    monkeypatch.setattr(api, "activity_enable", enable)
    supported = NVTXTracer().supported()
    assert Activity.KERNEL not in supported
    with pytest.raises(TargetError, match="available activity kinds"):
        NVTXTracer().collect(Activity.KERNEL)
    assert NVTXTracer()._resolve(Activity.MEMCPY) is Activity.MEMCPY
    assert isinstance(Tracer.detect(present=frozenset({Vendor.NVIDIA})), NVTXTracer)
    assert type(Tracer.detect()) is Tracer


def test_native_profiler_capture_windows_and_loss(api, monkeypatch: pytest.MonkeyPatch) -> None:
    device = ReadingDevice(vendor=Vendor.NVIDIA, name="fake CUDA", index=0, architecture="sm_90", source="fake")
    with Profiler(gpus=(device,), features=Feature.ACTIVITY, activities=Activity.DEFAULT) as profiler:
        assert isinstance(profiler.collector, CUPTICollector)
        answer, result = Profiler.capture(lambda: (Delivery.completed([kernel_activity()]), 3)[1])
        assert answer == 3 and result.kernels[0].name == "gemm"
        assert profiler.result().kernels == result.kernels
        profiler.collector.scope = (1, 0, 1)
        with pytest.raises(RuntimeError, match="selected device"):
            profiler._capturable(Activity.KERNEL)
        profiler.collector.scope = (0, 0, 1)
        assert profiler._retrieve(since=0, until=1, activities=Activity.MEMCPY) == ((), ())

        def lose() -> None:
            assert isinstance(profiler.collector, CUPTICollector)
            profiler.collector.lost_records += 1

        with pytest.raises(RuntimeError, match="lost 1"):
            Profiler.capture(lose)
        with pytest.raises(RuntimeError, match="different CUDA-visible"):
            Profiler.capture(lambda: None, device_index=1)
    machine = replace(System.detect(), devices=())
    monkeypatch.setattr(System, "detect", lambda: machine)
    assert Profiler.capture(lambda: 4)[0] == 4
    with Profiler() as profiler:
        assert Profiler.capture(lambda: 5)[0] == 5
        with pytest.raises(RuntimeError, match="did not enable"):
            Profiler.capture(lambda: None, activities=Activity.MEMCPY)

        class Derived(Profiler):
            pass

        with pytest.raises(RuntimeError, match="cannot provide"):
            Derived.capture(lambda: None)
    assert isinstance(Tracer()._open(Activity.KERNEL), collector_module.TraceCollector)
    with Profiler(gpus=(device,), native=True) as profiler:
        assert isinstance(profiler.collector, CUPTICollector)


def _launch() -> LaunchRecord:
    return LaunchRecord(name="a", geometry=(1, 1, 1, 1), dtypes=(), target="sm_90", stream=1, host_enqueue_ns=0)


def test_whole_session_reads_refuse_a_foreign_context(api) -> None:
    device = ReadingDevice(vendor=Vendor.NVIDIA, name="fake CUDA", index=0, architecture="sm_90", source="fake")
    with Profiler(gpus=(device,), features=Feature.ACTIVITY, activities=Activity.DEFAULT, native=True) as profiler:
        Delivery.completed([kernel_activity(context_id=9)])
        with pytest.raises(RuntimeError, match="foreign"):
            profiler.result()


def test_native_window_is_refused_only_when_retention_passed_its_launches(api) -> None:
    device = ReadingDevice(vendor=Vendor.NVIDIA, name="fake CUDA", index=0, architecture="sm_90", source="fake")
    with Profiler(
        gpus=(device,), features=Feature.ACTIVITY, activities=Activity.DEFAULT, native=True, max_spans=1
    ) as profiler:
        profiler._append_launch(_launch())

        def overflowing() -> None:
            profiler._append_launch(_launch())
            profiler._append_launch(_launch())
            Delivery.completed([kernel_activity()])

        with pytest.raises(RuntimeError, match="outside the retained"):
            Profiler.capture(overflowing)

        def complete() -> None:
            profiler._append_launch(_launch())
            Delivery.completed([kernel_activity()])

        _, window = Profiler.capture(complete)
        assert len(window.launches) == 1 and window.kernels[0].name == "gemm"
