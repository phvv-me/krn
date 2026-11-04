import os
import threading
from dataclasses import replace

import pytest

from krn.hw.devices.base import Process, ReadingDevice, Telemetry
from krn.hw.enums import DeviceKind, Vendor
from krn.hw.system import System
from krn.profile import Diagnosis, Feature, Profiler, Tracer, span
from krn.profile.bottleneck import annotate
from krn.profile.bottleneck.core import gpu_busy, profile
from krn.profile.health.meter import Meter
from krn.profile.recording import DeviceSampler, SpanRecorder
from krn.profile.session import Collection


def device() -> ReadingDevice:
    return ReadingDevice(
        name="fake",
        architecture="test",
        source="fake",
        vendor=Vendor.UNKNOWN,
        index=0,
        telemetry_reader=lambda: Telemetry(
            utilization_percent=80,
            memory_used_bytes=95,
            memory_total_bytes=100,
            memory_utilization_percent=70,
            power_w=5,
            temperature_c=40,
        ),
        process_reader=lambda: (Process(pid=-1), Process(pid=os.getpid(), memory_bytes=50)),
    )


def test_sampler_boundary_periodic_failures_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = DeviceSampler(interval_ms=1)
    recorder = SpanRecorder(Collection(features=Feature.DEVICE), tracer=Tracer(), snapshot=sampler.snapshot)
    assert sampler.snapshot("empty") is None
    sampler._sample_once(recorder)
    token = recorder.enter("a")
    sampler._sample_once(recorder)
    sampler.gpu = device()
    sampler._sample_once(recorder)
    assert recorder.frames[token].samples[0].memory_used_bytes == 50
    recorder.exit(token, wall_ns=100)
    token = recorder.enter("boundary")
    recorder.exit(token, wall_ns=200)
    assert len(recorder.measurements) == 2
    sampler.start(recorder)
    sampler.stop()
    assert sampler.thread is None
    sampler.stop_event.clear()
    # The polling loop passes a timeout; adapt the deterministic two-tick clock.
    ticks = iter([False, True])
    monkeypatch.setattr(sampler.stop_event, "wait", lambda _: next(ticks))
    sampler._poll(recorder)
    sampler.gpu = replace(device(), process_reader=None)
    assert sampler.snapshot("absent") is None

    def failed() -> Telemetry:
        raise OSError("sensor unavailable")

    sampler.gpu = replace(device(), telemetry_reader=failed)
    assert sampler.snapshot("failed") is None
    empty = SpanRecorder(Collection(features=Feature.DEVICE), tracer=Tracer(), snapshot=lambda _: None)
    empty.exit(empty.enter("none"), wall_ns=1)
    assert not empty.measurements
    windowed = SpanRecorder(
        Collection(features=Feature.ACTIVITY, activities=Profiler.Activity.KERNEL, max_spans=1),
        tracer=Tracer(),
        snapshot=lambda _: None,
    )
    for _ in range(2):
        windowed.exit(windowed.enter("a"), wall_ns=2)
    assert windowed.dropped == 1 and not windowed.measurements
    marked = SpanRecorder(Collection(features=Feature.MARKERS), tracer=Tracer(), snapshot=lambda _: None)
    marked.exit(marked.enter("base marker"), wall_ns=1)
    with (
        Profiler(gpus=(device(),), features=Feature.DEVICE | Feature.SPANS, sample_interval_ms=1) as profiler,
        span("measured"),
    ):
        pass
    assert profiler.result().summaries[0].samples == 1
    with pytest.raises(ValueError, match="outside"), Profiler(gpus=(device(),), device_index=1):
        pass


def test_health_and_bottleneck_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    machine = replace(System.detect(), devices=(device(),))
    meter = Meter(machine)
    assert meter.host_delta_gb == meter.peak_host_gb == 0 and meter.peak_gpu_gb is None
    with meter:
        meter.sample()
    assert meter.elapsed_s > 0 and meter.peak_host_gb > 0
    assert Diagnosis.diagnose(meter, None).reason == "unknown: no GPU to diagnose"
    assert Diagnosis.diagnose(meter, device()).host_offload
    assert "process count" in Diagnosis.diagnose(meter, replace(device(), process_reader=None)).unknown
    meter.gpu_used_gb[:] = [99]
    meter.host_used_gb[:] = [1, 10]
    verdict = Diagnosis.of(meter, Telemetry(utilization_percent=10, throttle_reasons=("power",)), capacity_gb=100)
    assert verdict.near_oom and verdict.gpu_underutilized and verdict.host_offload and verdict.throttled
    assert verdict.reason.startswith("near OOM")
    verdict = Diagnosis.of(meter, Telemetry(), capacity_gb=0)
    assert not verdict.near_oom and not verdict.gpu_underutilized
    assert gpu_busy(None) is False
    assert gpu_busy(device()) is True
    assert gpu_busy(replace(device(), telemetry_reader=Telemetry)) is False
    assert gpu_busy(
        replace(
            device(),
            telemetry_reader=lambda: Telemetry(utilization_percent=0, memory_used_bytes=95, memory_total_bytes=100),
        )
    )
    annotate._tracer = None
    assert annotate.tracer() is annotate.tracer()
    assert profile(lambda: 1, iters=1, warmup=1, sync=lambda: None, gpu=device()).iterations == 1
    assert profile(lambda: 1, iters=0).iterations == 0
    assert profile(lambda: 1, kinds=Profiler.Activity(0)).iterations == 1


def test_blocked_sampler_worker_is_kept_and_refuses_a_second_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DeviceSampler, "_JOIN_SECONDS", 0.01)
    entered, release = threading.Event(), threading.Event()

    def blocked() -> Telemetry:
        entered.set()
        release.wait(5)
        return Telemetry()

    sampler = DeviceSampler(interval_ms=1)
    sampler.gpu = replace(device(), telemetry_reader=blocked)
    recorder = SpanRecorder(Collection(features=Feature.DEVICE), tracer=Tracer(), snapshot=sampler.snapshot)
    recorder.enter("open")
    sampler.start(recorder)
    assert entered.wait(5)
    sampler.stop()
    assert sampler.thread is not None and sampler.thread.is_alive()
    with pytest.raises(RuntimeError, match="blocked"):
        sampler.start(recorder)
    release.set()
    sampler.thread.join(5)
    sampler.stop()
    assert sampler.thread is None


def test_meter_reads_only_gpus_and_stays_unknown_when_one_is_unread() -> None:
    cpu = replace(device(), kind=DeviceKind.CPU)
    meter = Meter(replace(System.detect(), devices=(cpu, device())))
    meter.sample()
    assert meter.gpu_used_gb == [95 / 2**30] and meter.peak_gpu_gb == 95 / 2**30
    unread = Meter(replace(System.detect(), devices=(device(), replace(device(), telemetry_reader=Telemetry))))
    unread.sample()
    assert unread.gpu_used_gb == [None] and unread.peak_gpu_gb is None


@pytest.mark.parametrize(
    ("telemetry", "host_gb", "capacity_gb", "processes", "reason"),
    [
        (Telemetry(throttle_reasons=("power",)), (1, 1), None, 1, "throttled: power"),
        (Telemetry(throttle_reasons=()), (1, 1), None, 3, "host offload: 3 processes share the GPU"),
        (Telemetry(throttle_reasons=()), (1, 10), None, 1, "host offload: host memory grew 9.0 GB"),
        (Telemetry(throttle_reasons=(), utilization_percent=5), (1, 1), None, 1, "GPU underutilized: 5% compute"),
        (
            Telemetry(),
            (1, 1),
            None,
            None,
            "unknown: GPU memory, GPU utilization, throttle state, process count unavailable",
        ),
    ],
)
def test_each_health_reason_renders_its_evidence(
    telemetry: Telemetry, host_gb: tuple[float, float], capacity_gb: float | None, processes: int | None, reason: str
) -> None:
    meter = Meter(System.detect())
    meter.host_used_gb[:] = list(host_gb)
    meter.gpu_used_gb[:] = [None]
    assert Diagnosis.of(meter, telemetry, capacity_gb=capacity_gb, process_count=processes).reason == reason


def test_a_fully_observed_quiet_device_is_healthy() -> None:
    meter = Meter(System.detect())
    meter.host_used_gb[:] = [1, 1]
    meter.gpu_used_gb[:] = [50]
    verdict = Diagnosis.of(
        meter, Telemetry(throttle_reasons=(), utilization_percent=90), capacity_gb=100, process_count=1
    )
    assert verdict.reason == "healthy" and verdict.unknown == ()
