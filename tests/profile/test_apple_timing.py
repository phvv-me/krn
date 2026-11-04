import runpy
import struct
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from krn.compiler.errors import TargetError
from krn.profile.providers.apple import capabilities, timing
from krn.profile.providers.apple.timing import StageTimer, clock_offset_ns, command_span

from .test_apple_capabilities import FakeDevice, FakeSet, as_device, fake_metal

if TYPE_CHECKING:
    import Metal


class FakeCommand:
    def __init__(self, start: float, end: float) -> None:
        self.start, self.end = start, end

    def GPUStartTime(self) -> float:
        return self.start

    def GPUEndTime(self) -> float:
        return self.end


def as_command(fake: FakeCommand) -> Metal.MTLCommandBuffer:
    """MTLCommandBuffer is an Objective-C protocol that PyObjC does not let a test subclass, so the cast is here."""
    return cast("Metal.MTLCommandBuffer", fake)


class FakeClockDevice(FakeDevice):
    def __init__(self, cpu: int, gpu: int) -> None:
        super().__init__()
        self.pair = (cpu, gpu)

    def sampleTimestamps_gpuTimestamp_(self, cpu: None, gpu: None) -> tuple[int, int]:
        return self.pair


class FakeBuffer:
    def __init__(self, ticks: list[int]) -> None:
        self.ticks = ticks

    def resolveCounterRange_(self, window: tuple[int, int]) -> bytes:
        start, length = window
        return struct.pack(f"<{length}Q", *self.ticks[start : start + length])


class FakeAttachment:
    def __init__(self) -> None:
        self.buffer: FakeBuffer | None = None
        self.start: int | None = None
        self.end: int | None = None

    def setSampleBuffer_(self, buffer: FakeBuffer) -> None:
        self.buffer = buffer

    def setStartOfEncoderSampleIndex_(self, index: int) -> None:
        self.start = index

    def setEndOfEncoderSampleIndex_(self, index: int) -> None:
        self.end = index


class FakeDescriptor:
    def __init__(self) -> None:
        self.counter_set: FakeSet | None = None
        self.count: int | None = None
        self.storage: str | None = None
        self.attachment = FakeAttachment()

    @classmethod
    def alloc(cls) -> FakeDescriptor:
        return cls()

    def init(self) -> FakeDescriptor:
        return self

    def setCounterSet_(self, counter_set: FakeSet) -> None:
        self.counter_set = counter_set

    def setSampleCount_(self, count: int) -> None:
        self.count = count

    def setStorageMode_(self, storage: str) -> None:
        self.storage = storage

    @classmethod
    def computePassDescriptor(cls) -> FakeDescriptor:
        return cls()

    def sampleBufferAttachments(self) -> FakeDescriptor:
        return self

    def objectAtIndexedSubscript_(self, index: int) -> FakeAttachment:
        return self.attachment


class StageDevice(FakeDevice):
    def __init__(self, *, ticks: list[int], error: str | None = None) -> None:
        super().__init__(sets=(("timestamp", ("GPUTimestamp",)),), points=("stage",))
        self.ticks, self.error = ticks, error
        self.requested: FakeDescriptor | None = None
        self.buffer: FakeBuffer | None = None

    def newCounterSampleBufferWithDescriptor_error_(self, descriptor: FakeDescriptor, _: None) -> tuple:
        self.requested = descriptor
        if self.error is not None:
            return None, self.error
        self.buffer = FakeBuffer(self.ticks)
        return self.buffer, None


@pytest.fixture
def metal(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    fake = fake_metal(capture=False)
    fake.MTLCounterSampleBufferDescriptor = FakeDescriptor
    fake.MTLComputePassDescriptor = FakeDescriptor
    fake.MTLStorageModeShared = "shared"
    monkeypatch.setattr(capabilities, "Metal", fake)
    monkeypatch.setattr(timing, "Metal", fake)
    monkeypatch.setattr(timing, "NSMakeRange", lambda location, length: (location, length))
    return fake


def test_command_span_is_nanoseconds_on_the_host_clock() -> None:
    assert command_span(as_command(FakeCommand(1.5, 1.5000001))) == (1_500_000_000, 1_500_000_100)


def test_clock_offset_is_the_median_gpu_minus_cpu() -> None:
    assert clock_offset_ns(as_device(FakeClockDevice(cpu=100, gpu=107))) == 7
    assert clock_offset_ns(as_device(FakeClockDevice(cpu=100, gpu=100))) == 0


def test_stage_timer_samples_each_encoder_at_its_own_slots(metal: SimpleNamespace) -> None:
    device = StageDevice(ticks=[10, 20, 30, 40, 50, 60])
    timer = StageTimer(as_device(device), capacity=3)
    requested = device.requested
    assert timer.buffer is device.buffer and requested is not None and requested.counter_set is not None
    assert (requested.count, requested.counter_set.name(), requested.storage) == (6, "timestamp", "shared")
    descriptor = timer.descriptor(1)
    assert isinstance(descriptor, FakeDescriptor)
    assert (descriptor.attachment.start, descriptor.attachment.end) == (2, 3)
    assert descriptor.attachment.buffer is timer.buffer
    assert timer.read(1) == (30, 40)
    assert timer.read(2) == (50, 60)


def test_stage_timer_refuses_indexes_outside_its_capacity(metal: SimpleNamespace) -> None:
    timer = StageTimer(as_device(StageDevice(ticks=list(range(4)))), capacity=2)
    for index in (-1, 2):
        with pytest.raises(IndexError, match="outside the 2 samples"):
            timer.descriptor(index)


def test_stage_timer_names_the_failure_and_the_alternative(metal: SimpleNamespace) -> None:
    device = StageDevice(ticks=[], error="could not allocate")
    with pytest.raises(TargetError, match="could not allocate") as caught:
        StageTimer(as_device(device), capacity=1)
    assert "command_span" in caught.value.instead


def test_stage_timer_requires_the_timestamp_set_before_allocating(metal: SimpleNamespace) -> None:
    with pytest.raises(TargetError, match="counter sets"):
        StageTimer(as_device(FakeDevice(points=("stage",))), capacity=1)


def test_timing_bindings_are_optional_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "Metal", None)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        module = runpy.run_module(timing.__name__)
    assert module["Metal"] is None and module["NSMakeRange"] is None
