"""Session admission, bounded retention, settlement after failure, and launch windows."""

import threading
from collections.abc import Sequence
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from krn import Blocks, T
from krn.compiler.api import CPUDevice, CPUEvent, CPUStream
from krn.hw.devices.base import ReadingDevice
from krn.hw.enums import Vendor
from krn.profile import Profile, Profiler, launches
from krn.profile.spans import active
from krn.runtime.runtime import Call

from .test_launches import twice


class Recorder:
    """A session that counts the work its hooks take and runs each call, as the profiler does."""

    def __init__(self) -> None:
        self.queued = 0
        self.ran: list[Call] = []

    def cpu_enqueue(self, calls: Sequence[Call], stream: CPUStream) -> None:
        self.queued += len(calls)

    def cpu_run(self, call: Call, stream: CPUStream) -> None:
        call.program.function(*call.arguments)
        self.ran.append(call)


def _gpu() -> ReadingDevice:
    return ReadingDevice(vendor=Vendor.APPLE, name="fake Metal", architecture="apple_m4", index=0, source="fake")


def _metal_call(device: CPUDevice):
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    program = device.launcher(twice, Blocks(2, threads=16)).program(x, y)
    call, stream = Mock(), Mock()
    call.program.ir, call.geometry = program.kernel, (2, 1, 1, 16)
    stream.device.target.arch = "apple_m4"
    return call, stream


def _command(start: float = 2.0, end: float = 2.000001, error: str | None = None) -> Mock:
    command = Mock()
    command.error.return_value = error
    command.GPUStartTime.return_value = start
    command.GPUEndTime.return_value = end
    return command


def test_launch_after_exit_still_runs_but_records_nothing(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler() as profiler:
        bound()
    call, stream = Mock(), Mock()
    call.program.launch = Mock()
    call.grid, call.threads, call.packed = (2, 1, 1), 16, ()
    stream.device.target.arch = "sm_90"
    profiler.cuda_run(call, stream)
    call.program.launch.assert_called_once_with(stream, call.grid, 16, call.packed)
    stream.record.assert_not_called()
    assert len(profiler.result().launches) == 1


def test_work_admitted_before_exit_is_settled_and_recorded(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler() as profiler:
        bound.submit()
    records = profiler.result().launches
    assert len(records) == 1
    assert records[0].timing == "cpu-wall" and records[0].duration_ns is not None
    assert not profiler._cpu_pending and not profiler._cpu_streams


def test_exit_releases_every_native_reference_even_under_concurrent_hooks(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    profiler = Profiler()
    profiler.__enter__()
    failures: list[BaseException] = []

    def hammer() -> None:
        try:
            for _ in range(200):
                profiler.cpu_enqueue((bound._call,), device.default_stream)
                profiler.cpu_run(bound._call, device.default_stream)
        except (AttributeError, KeyError, RuntimeError, ValueError) as error:  # reported by the assertion below
            failures.append(error)

    worker = threading.Thread(target=hammer)
    worker.start()
    profiler.__exit__(None, None, None)
    worker.join(timeout=30)
    assert not worker.is_alive() and failures == []
    assert not profiler._cpu_pending and not profiler._cpu_streams and profiler._cpu_queued == 0
    assert launches.session is None and active() is None


def test_failed_settlement_keeps_completed_records_once_and_releases_the_rest(device: CPUDevice) -> None:
    call, stream = _metal_call(device)
    done, failing, pending = _command(), _command(error="GPU failure"), _command()
    profiler = Profiler(gpus=(_gpu(),))
    with pytest.raises(RuntimeError, match="GPU failure"), profiler:
        profiler.metal_enqueue(call, stream, done)
        profiler.metal_enqueue(call, stream, failing)
        profiler.metal_enqueue(call, stream, pending)
    assert len(profiler.launches) == 1
    assert not profiler._metal_pending and not profiler._cuda_pending
    pending.waitUntilCompleted.assert_called_once()
    assert not profiler.active and launches.session is None


def test_native_pending_is_bounded_by_max_spans_without_losing_records(device: CPUDevice) -> None:
    call, stream = _metal_call(device)
    profiler = Profiler(gpus=(_gpu(),), max_spans=2)
    with profiler:
        for _ in range(3):
            profiler.metal_enqueue(call, stream, _command())
            assert len(profiler._metal_pending) <= 2
    assert len(profiler.launches) == 2 and profiler.dropped_launches == 1
    assert profiler.result().dropped_launches == 1


def test_queued_cpu_calls_past_the_bound_are_still_recorded(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler(max_spans=2) as profiler:
        for _ in range(3):
            profiler.cpu_enqueue((bound._call,), device.default_stream)
        assert profiler._cpu_queued == 2
        for _ in range(3):
            profiler.cpu_run(bound._call, device.default_stream)
    assert len(profiler.launches) == 2 and profiler.dropped_launches == 1
    assert all(record.timing == "cpu-wall" for record in profiler.launches)
    assert profiler._cpu_queued == 0 and not profiler._cpu_pending


def test_runtime_window_keeps_its_launches_when_older_ones_were_overwritten(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler(max_spans=1):
        bound()
        _, window = Profiler.capture(lambda: (bound(), 0)[1])
    assert [record.name for record in window.launches] == ["twice"]
    assert window.dropped_launches == 1


def test_nested_runtime_windows_are_views_of_one_buffer(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler() as profiler:
        bound()

        def outer() -> None:
            bound()
            _, inner = Profiler.capture(lambda: (bound(), 0)[1])
            assert len(inner.launches) == 1

        _, outer_window = Profiler.capture(outer)
    assert len(outer_window.launches) == 2
    assert len(profiler.result().launches) == 3


def test_window_refused_once_retention_passed_its_first_launch(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler(max_spans=1):
        bound()
        with pytest.raises(RuntimeError, match="overwritten"):
            Profiler.capture(lambda: (bound(), bound(), 0)[2])


def test_marker_only_session_discovers_the_host_device(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[frozenset] = []
    gpu = _gpu()

    def detect(*, present: frozenset) -> Mock:
        seen.append(present)
        return Mock(collect=Mock())

    monkeypatch.setattr("krn.profile.profiler.System.detect", lambda: Mock(visible_devices=(gpu,)))
    monkeypatch.setattr("krn.profile.profiler.Tracer.detect", staticmethod(detect))
    with Profiler(features=Profiler.Feature.MARKERS):
        pass
    assert seen == [frozenset({Vendor.APPLE})]


def test_hooks_after_exit_execute_without_recording(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    x.array.fill(3)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler() as profiler:
        pass
    profiler.cpu_run(bound._call, device.default_stream)
    assert (y.array == 6).all()
    call, stream = _metal_call(device)
    profiler.metal_enqueue(call, stream, _command())
    assert not profiler._metal_pending and profiler.result().launches == ()


def test_failed_cuda_settlement_waits_on_the_rest_and_releases_them(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    program = device.launcher(twice, Blocks(2, threads=16)).program(x, y)
    call, stream = Mock(), Mock()
    call.program.ir, call.grid, call.threads = program.kernel, (2, 1, 1), 16
    stream.device.target.arch = "sm_90"
    failing, pending = Mock(), Mock()
    failing.wait.side_effect = RuntimeError("cudaErrorIllegalAddress")
    stream.record.side_effect = [Mock(), failing, Mock(), pending]
    profiler = Profiler()
    with pytest.raises(RuntimeError, match="IllegalAddress"), profiler:
        profiler.cuda_run(call, stream)
        profiler.cuda_run(call, stream)
    pending.wait.assert_called_once()
    assert not profiler._cuda_pending and not profiler.launches


def test_untimed_launch_is_kept_without_inventing_a_kernel(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    ir = device.launcher(twice, Blocks(2, threads=16)).program(x, y).kernel
    untimed = launches.record(ir, (2, 1, 1, 16), "avx512", 1)
    profile = Profiler()._with_launches(Profile(), [untimed])
    assert profile.launches == (untimed,) and profile.kernels == ()


def test_device_identity_names_only_an_unambiguous_device() -> None:
    gpu = _gpu()
    profiler = Profiler(gpus=(gpu, replace(gpu, index=1)))
    shared = launches.LaunchRecord(
        name="a", geometry=(1, 1, 1, 1), dtypes=(), target="apple_m4", stream=1, host_enqueue_ns=0, device_key=7
    )
    assert profiler._with_launches(Profile(), [shared]).device == "unknown"
    assert profiler._with_launches(Profile(), [replace(shared, device_key=None)]).device == "unknown"
    assert profiler._with_launches(Profile(), [replace(shared, target="zz_unknown_target")]).device == "unknown"
    assert profiler._with_launches(Profile(), [replace(shared, target="avx512")]).device == "cpu"
    assert Profiler(gpus=(gpu,))._with_launches(Profile(), [shared]).device == "fake Metal"


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(calls=st.integers(min_value=1, max_value=4), graphed=st.booleans())
def test_queued_work_is_recorded_by_the_session_that_queued_it(device: CPUDevice, calls: int, graphed: bool) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    x.array.fill(3)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)

    def once() -> None:
        bound()

    launch = device.graph(once) if graphed else bound
    stream, gate = device.stream(), CPUEvent()
    owner, other = Recorder(), Recorder()
    with patch.object(launches, "session", owner), stream:
        stream.wait(gate)  # parks the lane, so the launches below stay queued while `other` is active
        for _ in range(calls):
            launch()
    with patch.object(launches, "session", other):
        gate.resolve()
        stream.record().wait(30)
    assert owner.queued == calls and len(owner.ran) == calls
    assert other.ran == [] and other.queued == 0
    assert (y.array == 6).all()
