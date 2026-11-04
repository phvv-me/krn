import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn.compiler.dtypes import F32
from krn.compiler.errors import TargetError
from krn.compiler.ir import StructType, TensorType
from krn.hw.devices.base import ReadingDevice
from krn.hw.enums import Vendor
from krn.hw.peaks import PeakRates, PeakSource
from krn.profile import (
    Activity,
    ActivityRecord,
    Bound,
    DeviceEvidence,
    DeviceTimeline,
    EfficiencyReport,
    KernelEfficiency,
    KernelTrace,
    MemcpyTrace,
    Profile,
    ProfileReport,
    RegionStat,
    RegionSummary,
    RegionWindow,
    TraceCollector,
    Tracer,
    benchmark,
    compare,
    profile_stages,
)
from krn.profile.launches import LaunchRecord, signature
from krn.profile.result.models import ProcessReading
from krn.profile.session import Collection, Feature
from krn.profile.studies.dispatch import arch_config
from krn.profile.studies.study import Point, Row, Study


def test_results_roundtrip_export_and_reports(tmp_path: Path) -> None:
    snaps = [ProcessReading(memory_used_bytes=100, gpu_util_pct=70, memory_util_pct=80, power_w=5, temperature_c=42)]
    summary = RegionSummary.from_snaps("a", 2, snaps)
    assert (summary.samples, summary.avg_memory_bytes, summary.max_temp_c) == (1, 100, 42)
    missing = RegionSummary.from_snaps("unknown", 1, [ProcessReading()])
    assert missing.peak_memory_bytes is None and missing.avg_util_pct is None
    assert ProfileReport.from_profile(Profile(summaries=(missing,)), iterations=1).avg_memory_bytes is None
    stats = RegionStat.aggregate([summary, summary])
    assert stats[0].calls == 2
    peak = PeakRates(
        memory_bandwidth_gbs=10,
        memory_bandwidth_source=PeakSource.TABLE,
        flops={"f32": 100},
        flops_sources={"f32": PeakSource.LIBRARY},
    )
    launch = LaunchRecord(
        name="a",
        geometry=(10, 1, 1, 32),
        dtypes=("f32",),
        target="cpu",
        stream=1,
        host_enqueue_ns=1,
        flops=20,
        flops_by_dtype={"f32": 20},
        bytes_moved=100,
        start_ns=1,
        end_ns=101,
    )
    profile = Profile(
        device="test",
        device_evidence=DeviceEvidence.COLLECTED,
        kernels=(KernelTrace(name="a", start_ns=1, end_ns=101, grid="10x1x1", block="32x1x1"),),
        memcpys=(MemcpyTrace(start_ns=2, end_ns=4, bytes_moved=10),),
        activities=(ActivityRecord(name="runtime", kind="runtime", start_ns=1, end_ns=2),),
        summaries=(summary,),
        windows=(RegionWindow(name="a", start_ns=0, end_ns=200, wall_ns=200),),
        launches=(launch,),
        peak=peak,
        dropped_activities=1,
        dropped_spans=1,
        dropped_launches=1,
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    assert Profile.load(path) == profile
    assert "Spans" in str(profile) and "GPU activity" in str(profile) and "Capture limit" in str(profile)
    assert profile.bottlenecks()[0].name == "a"
    assert profile.timeline().activities == 2
    assert profile.trace_report().hot_kernels[0].name == "a"
    profile.perfetto(path)
    events = json.loads(path.read_text())["traceEvents"]
    assert {event["ph"] for event in events} == {"M", "X"}
    assert {event["name"] for event in events} >= {"runtime", "unknown", "a"}
    no_peak = Profile(summaries=(summary,))
    no_peak.save(path)
    assert Profile.load(path) == no_peak
    no_peak.perfetto(path)
    assert Profile._region_text([]) == "No regions recorded."
    assert str(Profile()) == "No profiling data collected."
    assert "observed none" in str(Profile(device_evidence=DeviceEvidence.ABSENT))

    efficiency = profile.efficiency(sm_count=4)
    assert efficiency.achieved_gbs == 1
    assert efficiency.bandwidth_utilisation_pct == 10
    assert efficiency.total_ms == pytest.approx(0.0001)
    assert efficiency.__rich__().row_count == 1
    assert profile.efficiency().__rich__().row_count == 1
    assert EfficiencyReport().__rich__().row_count == 0
    assert EfficiencyReport().achieved_gbs is None
    assert EfficiencyReport().bandwidth_utilisation_pct is None
    all_time = EfficiencyReport.build(
        [KernelTrace(name=str(i), start_ns=1, end_ns=1001) for i in range(13)], sm_count=4, bytes_moved=13000, top=1
    )
    assert len(all_time.rows) == 1 and all_time.total_ms == 0.013 and all_time.achieved_gbs == 1
    assert EfficiencyReport(bytes_moved=100, measured_ns=10).bandwidth_utilisation_pct is None
    assert profile.efficiency(bytes_moved=20, peak_bandwidth_gbs=1).achieved_gbs == 0.2
    assert replace(profile, launches=()).efficiency().achieved_gbs is None
    assert replace(profile, launches=(replace(launch, bytes_moved=None),)).efficiency().achieved_gbs is None
    assert replace(profile, launches=(replace(launch, start_ns=None, end_ns=None),)).efficiency().achieved_gbs == 1
    assert KernelTrace(block="32xunknown").threads_per_block == 32
    assert KernelTrace(block="32").occupancy_pct == 3.125
    assert KernelTrace(start_ns=0, end_ns=1000).duration_us == 1
    assert ActivityRecord(start_ns=1, end_ns=4).duration_ns == 3
    assert MemcpyTrace().bandwidth_gbps == 0
    rows = KernelEfficiency.aggregate(
        [*profile.kernels, KernelTrace(), KernelTrace(name="b", start_ns=1, end_ns=3, grid="bad", block="(4, 1)")], 5
    )
    assert rows[0].waves == 2
    assert rows[1].blocks == 0
    waste = KernelEfficiency.aggregate(profile.kernels, 4)[0].tail_waste_pct
    assert waste is not None and waste > 0
    assert EfficiencyReport._readable("_ZN5numba3foo3barE") == "numba.foo.bar"
    assert EfficiencyReport._readable("_ZNbad") == "_ZNbad"
    assert EfficiencyReport._readable("plain") == "plain"
    assert EfficiencyReport._readable("_ZN123") == ""
    assert DeviceTimeline().__rich__().row_count == 5
    timeline = DeviceTimeline.from_traces(
        [KernelTrace(name="a", start_ns=0, end_ns=1), KernelTrace(name="b", start_ns=20001, end_ns=20002)]
    )
    assert timeline.idle_ns == 20000 and timeline.occupancy_pct > 0
    assert timeline.gaps[0].duration_ms == 0.02
    assert timeline.__rich__().row_count == 6
    assert "occupancy" in str(timeline)
    assert DeviceTimeline().occupancy_pct == 0

    report = ProfileReport.from_profile(
        profile, iterations=1, supported=Activity.KERNEL.value, requested=Activity.DEFAULT.value
    )
    assert report.bound is Bound.MEMORY and report.unavailable == ("memcpy",)
    assert report.achieved_bandwidth_gbps == 1
    assert "bandwidth" in str(report)
    assert ProfileReport.from_profile(Profile(), iterations=0).bound is Bound.UNKNOWN
    assert "No kernels" in str(ProfileReport(device_evidence=DeviceEvidence.ABSENT))
    for flops, payload, bound in (
        (2000, 1, Bound.COMPUTE),
        (2, 0, Bound.COMPUTE),
        (0, 0, Bound.UNKNOWN),
        (None, 1, Bound.UNKNOWN),
        (1, None, Bound.UNKNOWN),
    ):
        assert (
            ProfileReport._roofline(
                replace(
                    profile,
                    launches=(replace(launch, flops=flops, flops_by_dtype={"f32": flops or 0}, bytes_moved=payload),),
                )
            )
            is bound
        )
    assert ProfileReport._roofline(replace(profile, peak=None)) is Bound.UNKNOWN
    assert ProfileReport._roofline(replace(profile, peak=PeakRates())) is Bound.UNKNOWN
    assert (
        ProfileReport._roofline(replace(profile, launches=(replace(launch, flops_by_dtype={"bf16": 20}),)))
        is Bound.UNKNOWN
    )
    assert (
        ProfileReport._roofline(replace(profile, launches=(replace(launch, flops_by_dtype={"f32": 10}),)))
        is Bound.UNKNOWN
    )
    assert ProfileReport._classify(kernel_ns=1, memcpy_ns=2, mem_util=0, gpu_util=0) is Bound.MEMORY
    assert ProfileReport._classify(kernel_ns=2, memcpy_ns=1, mem_util=0, gpu_util=0) is Bound.COMPUTE
    assert ProfileReport._classify(kernel_ns=0, memcpy_ns=0, mem_util=2, gpu_util=1) is Bound.MEMORY
    assert ProfileReport._classify(kernel_ns=0, memcpy_ns=0, mem_util=1, gpu_util=2) is Bound.COMPUTE
    assert ProfileReport._unavailable(supported=None, requested=1) == ()
    assert ProfileReport._unavailable(supported=1, requested=None) == ()


def test_base_tracer_and_collector_are_explicitly_empty() -> None:
    tracer = Tracer()
    assert tracer.is_available() is False
    assert tracer.start("empty")() is None and tracer.timestamp() > 0
    with pytest.raises(TargetError, match="no native collector"):
        tracer.collect()
    with TraceCollector() as collector:
        assert collector.kernels() == collector.memcpys() == collector.activities() == []
        assert collector.dropped() == 0
        collector.flush()
        collector.reset()
        with pytest.raises(RuntimeError):
            _ = collector.device_index
        with pytest.raises(RuntimeError):
            collector.checkpoint(Activity.KERNEL)
    assert signature(F32) == ("f32",)
    assert signature(TensorType(F32, 1)) == ("f32",)
    assert signature(StructType("x", (("field", F32),))) == ("f32",)
    assert Activity.KERNEL.label == "kernel" and Activity(0).label == "activity"


def test_clock_domains_are_not_joined_by_timestamp(tmp_path: Path) -> None:
    host = KernelTrace(name="cpu", start_ns=100, end_ns=200, clock="host")
    metal = KernelTrace(name="gpu", start_ns=1_000_000, end_ns=1_000_010, clock="metal")
    windows = (RegionWindow(name="host span", start_ns=0, end_ns=2_000_000, wall_ns=100, clock="host"),)
    profile = Profile(kernels=(metal,), windows=windows)
    assert profile.trace_report().hot_regions[0].name == "(outside regions)"
    mixed = replace(profile, kernels=(host, metal))
    for report in (mixed.timeline, mixed.trace_report):
        with pytest.raises(TargetError, match="clock domains"):
            report()
    path = tmp_path / "clocks.json"
    mixed.perfetto(path)
    events = json.loads(path.read_text())["traceEvents"]
    process_ids = {event["pid"] for event in events}
    assert len(process_ids) == 2
    names = [event["args"]["name"] for event in events if event["name"] == "process_name"]
    assert all("independent origin" in name for name in names)
    Profile().perfetto(path)
    assert json.loads(path.read_text())["traceEvents"]


@dataclass(frozen=True, kw_only=True)
class Case(Point):
    value: int = 1

    @property
    def label(self) -> str:
        return str(self.value)


def test_benchmarks_stages_and_study(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier: list[int] = []
    sample = benchmark(lambda: 1, iters=2, warmup=1, sync=lambda: barrier.append(1))
    assert sample.mean_us >= sample.min_us >= 0 and sample.runs == 2 and len(barrier) == 3
    assert benchmark(lambda: None, iters=1, warmup=0).runs == 1
    for iters, warmup in ((0, 1), (1, -1)):
        with pytest.raises(ValueError):
            benchmark(lambda: None, iters=iters, warmup=warmup)
    assert len(compare({"a": lambda: 1}, iters=1, warmup=0)) == 1
    assert compare({}) == []
    stages = profile_stages({"a": lambda: 1}, iters=1, warmup=0)
    assert "a" in str(stages)
    stages.show()
    assert str(type(stages)()) == "No stages profiled."
    monkeypatch.setattr(Tracer, "detect", lambda **_: Tracer())
    monkeypatch.setattr(Tracer, "supported", lambda self: Activity.KERNEL)
    stages = profile_stages({"a": lambda: 1}, iters=1, warmup=0, trace=Activity.KERNEL, sync=lambda: None)
    assert stages.profile is not None and "Spans" in str(stages)
    assert profile_stages({"a": lambda: 1}, iters=1, warmup=0, trace=Activity.KERNEL).profile is not None
    collection = Collection(features=Feature.SPANS)
    rows = Study.over([Case()], collection=collection).run(lambda _: None)
    assert rows[0].seconds >= 0 and not rows[0].has_evidence
    assert Row(label="x", point=Case(), profile=Profile(kernels=(KernelTrace(),))).has_evidence
    assert Study.over([]).run(lambda _: None) == ()
    assert Study.over([Case()]).collection == Collection()
    assert Study.over([Case()], collection=collection).run(lambda _: None, warm=False)
    assert arch_config({}, default=1) == 1
    assert (
        arch_config(
            {"x": 2},
            default=1,
            gpu=ReadingDevice(name="x", architecture="x", vendor=Vendor.UNKNOWN, index=0, source="fake"),
        )
        == 2
    )
    for invalid in (
        lambda: Collection(device_index=-1),
        lambda: Collection(sample_interval_ms=0),
        lambda: Collection(max_spans=0),
    ):
        with pytest.raises(ValueError):
            invalid()


def test_mainboard_era_profile_loads_with_defaulted_launch_fields(tmp_path: Path) -> None:
    mainboard_format = {
        "host": "h",
        "device": "d",
        "device_evidence": "collected",
        "summaries": [{"name": "a", "wall_ms": 2.0, "samples": 0, "peak_memory_bytes": 0}],
        "windows": [{"name": "a", "start_ns": 1, "end_ns": 5, "wall_ns": 4}],
        "kernels": [{"name": "k", "start_ns": 1, "end_ns": 5, "grid": "1x1x1", "block": "32x1x1"}],
        "memcpys": [],
        "activities": [],
        "dropped_spans": 0,
        "dropped_activities": 0,
    }
    path = tmp_path / "mainboard.json"
    path.write_text(json.dumps(mainboard_format), encoding="utf-8")
    loaded = Profile.load(path)
    assert loaded.kernels[0].name == "k" and loaded.launches == () and loaded.peak is None
    assert (loaded.dropped_launches, loaded.coherent, loaded.arch_key, loaded.compute_units) == (0, None, None, None)
    loaded.save(tmp_path / "krn.json")
    assert json.loads((tmp_path / "krn.json").read_text(encoding="utf-8"))["schema_version"] == 2
    assert Profile.load(tmp_path / "krn.json") == loaded


def test_unknown_profile_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 9 is unreadable"):
        Profile.load(path)


def test_mixed_precision_roofline_sums_each_dtype_at_its_own_ceiling() -> None:
    peak = PeakRates(
        memory_bandwidth_gbs=10,
        memory_bandwidth_source=PeakSource.TABLE,
        flops={"f32": 100, "f16": 400},
        flops_sources={"f32": PeakSource.LIBRARY, "f16": PeakSource.LIBRARY},
    )
    launch = LaunchRecord(
        name="m",
        geometry=(1, 1, 1, 1),
        dtypes=(),
        target="cpu",
        stream=1,
        host_enqueue_ns=0,
        flops=40_010,
        bytes_moved=1000,
    )
    profile = Profile(peak=peak, launches=(replace(launch, flops_by_dtype={"f32": 40_000, "f16": 10}),))
    assert ProfileReport._roofline(profile) is Bound.COMPUTE
    light = replace(launch, flops=200, flops_by_dtype={"f32": 100, "f16": 100}, bytes_moved=10_000)
    assert ProfileReport._roofline(replace(profile, launches=(light,))) is Bound.MEMORY
    unknown = replace(light, flops_by_dtype={"f64": 200})
    assert ProfileReport._roofline(replace(profile, launches=(unknown,))) is Bound.UNKNOWN


@given(flops=st.integers(0, 10**6), payload=st.integers(1, 10**6))
def test_single_dtype_roofline_is_the_ridge_comparison(flops: int, payload: int) -> None:
    peak = PeakRates(
        memory_bandwidth_gbs=10,
        memory_bandwidth_source=PeakSource.TABLE,
        flops={"f32": 100},
        flops_sources={"f32": PeakSource.LIBRARY},
    )
    launch = LaunchRecord(
        name="r",
        geometry=(1, 1, 1, 1),
        dtypes=(),
        target="cpu",
        stream=1,
        host_enqueue_ns=0,
        flops=flops,
        flops_by_dtype={"f32": flops},
        bytes_moved=payload,
    )
    expected = Bound.COMPUTE if flops * 10 >= payload * 100 else Bound.MEMORY
    assert ProfileReport._roofline(Profile(peak=peak, launches=(launch,))) is expected
