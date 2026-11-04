from dataclasses import replace

import pytest

from krn import Blocks, T
from krn.compiler.api import CPUDevice
from krn.compiler.errors import TargetError
from krn.hw.system import System
from krn.profile import Activity, Feature, Profiler
from krn.profile.collector import RuntimeCollector
from krn.profile.providers.apple.tracer import SignpostTracer
from krn.profile.providers.cpu.tracer import CPUTracer

from .test_launches import twice


def test_runtime_collector_owns_or_borrows_session(device: CPUDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    launch = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    empty = RuntimeCollector()
    with pytest.raises(RuntimeError, match="enter"):
        empty.kernels()
    with pytest.raises(RuntimeError, match="running"):
        empty.checkpoint(Activity.KERNEL)
    with CPUTracer().collect(Activity.KERNEL) as collector:
        assert isinstance(collector, RuntimeCollector)
        assert collector.device_index == 0
        with pytest.raises(RuntimeError, match="entered twice"):
            collector.__enter__()
        with pytest.raises(RuntimeError, match="only kernel"):
            collector.checkpoint(Activity.MEMCPY)
        launch()
        until, lost = collector.checkpoint(Activity.KERNEL)
        assert (until, lost) == (1, 0)
        assert collector.kernels(since=0, until=1)[0].name == "twice"
        assert collector.dropped() == 0
        collector.reset()
        assert collector.kernels() == []
        with pytest.raises(RuntimeError, match="reset"):
            collector.kernels(since=0)
        with pytest.raises(RuntimeError, match="reset"):
            collector.kernels(since=2)
    collector.stop()
    assert collector.kernels() == []
    with Profiler(features=Feature.SPANS), pytest.raises(RuntimeError, match="requires a Profiler"):
        RuntimeCollector().__enter__()
    with Profiler() as owner:
        with SignpostTracer().collect(Activity.KERNEL) as borrowed:
            launch()
            assert borrowed.kernels()[0].duration_ns > 0
        launch()
        assert len(borrowed.kernels()) == 1
        borrowed.reset()
        assert borrowed.kernels() == []
        assert owner.active
    machine = replace(System.detect(), devices=())
    monkeypatch.setattr(System, "detect", lambda: machine)
    with Profiler(native=True) as owner:
        launch()
    assert owner.result().kernels[0].name == "twice"
    with pytest.raises(TargetError, match=r"Feature\.SPANS"):
        CPUTracer().start("native marker")
