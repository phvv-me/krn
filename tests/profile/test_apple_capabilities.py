import runpy
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from krn.compiler.errors import TargetError
from krn.profile.providers.apple import capabilities
from krn.profile.providers.apple.capabilities import Capabilities
from krn.profile.providers.apple.signals import Signal, require

if TYPE_CHECKING:
    import Metal


class FakeSet:
    def __init__(self, name: str, counters: tuple[str, ...]) -> None:
        self._name, self._counters = name, counters

    def name(self) -> str:
        return self._name

    def counters(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=lambda counter=counter: counter) for counter in self._counters]


class FakeDevice:
    def __init__(self, *, sets: tuple[tuple[str, tuple[str, ...]], ...] = (), points: tuple[str, ...] = ()) -> None:
        self.sets = [FakeSet(name, counters) for name, counters in sets]
        self.points = frozenset(points)

    def name(self) -> str:
        return "Apple M4 Pro"

    def counterSets(self) -> list[FakeSet]:
        return self.sets

    def supportsCounterSampling_(self, point: str) -> bool:
        return point in self.points


def as_device(fake: FakeDevice) -> Metal.MTLDevice:
    """MTLDevice is an Objective-C protocol that PyObjC does not let a test subclass, so the cast is here."""
    return cast("Metal.MTLDevice", fake)


def fake_metal(*, capture: bool) -> SimpleNamespace:
    manager = SimpleNamespace(supportsDestination_=lambda destination: capture)
    return SimpleNamespace(
        MTLCaptureManager=SimpleNamespace(sharedCaptureManager=lambda: manager),
        MTLCaptureDestinationGPUTraceDocument="gpu-trace",
        MTLCounterSamplingPointAtStageBoundary="stage",
        MTLCounterSamplingPointAtDispatchBoundary="dispatch",
        MTLCounterSamplingPointAtDrawBoundary="draw",
        MTLCounterSamplingPointAtBlitBoundary="blit",
    )


@pytest.fixture
def metal(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    fake = fake_metal(capture=False)
    monkeypatch.setattr(capabilities, "Metal", fake)
    return fake


TIMESTAMP = Capabilities(
    chip="Apple M4 Pro", counter_sets={"timestamp": ("GPUTimestamp",)}, sampling=frozenset({"stage"}), capture=False
)


def test_probe_reads_sets_points_and_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capabilities, "Metal", fake_metal(capture=True))
    device = FakeDevice(sets=(("timestamp", ("GPUTimestamp",)),), points=("stage",))
    caps = Capabilities.probe(as_device(device))
    assert caps == replace(TIMESTAMP, capture=True)
    assert caps.timestamps and caps.stage_sampling
    assert not caps.dispatch_sampling and not caps.statistics
    assert caps.names() == "timestamp"


def test_statistic_sets_and_an_empty_device(metal: SimpleNamespace) -> None:
    stats = Capabilities.probe(
        as_device(FakeDevice(sets=(("stageutilization", ("ALU",)),), points=("stage", "dispatch")))
    )
    assert stats.statistics and stats.dispatch_sampling and not stats.timestamps
    empty = Capabilities.probe(as_device(FakeDevice()))
    assert empty.names() == "none" and not empty.statistics and not empty.stage_sampling


def test_require_names_the_feature_target_reason_and_alternative(metal: SimpleNamespace) -> None:
    require(Signal.ENCODER_TIME, TIMESTAMP)
    with pytest.raises(TargetError, match="dispatch boundaries") as caught:
        require(Signal.DISPATCH_TIME, TIMESTAMP)
    assert caught.value.feature == "GPU time per dispatch from counter samples"
    assert caught.value.target == "Apple M4 Pro" and "GPUStartTime" in caught.value.instead
    with pytest.raises(TargetError, match="counter sets: timestamp"):
        require(Signal.UTILIZATION, TIMESTAMP)
    with pytest.raises(TargetError, match="MetalCaptureEnabled"):
        require(Signal.CAPTURE, TIMESTAMP)
    require(Signal.CAPTURE, replace(TIMESTAMP, capture=True))
    require(Signal.UTILIZATION, replace(TIMESTAMP, counter_sets={"statistic": ("ALU",)}))


def test_encoder_time_refuses_without_the_timestamp_set(metal: SimpleNamespace) -> None:
    without = replace(TIMESTAMP, counter_sets={})
    with pytest.raises(TargetError, match=r"\[none\]"):
        require(Signal.ENCODER_TIME, without)
    with pytest.raises(TargetError, match=r"\[no points\]"):
        require(Signal.ENCODER_TIME, replace(TIMESTAMP, sampling=frozenset()))


def test_metal_bindings_are_optional_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "Metal", None)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        assert runpy.run_module(capabilities.__name__)["Metal"] is None
