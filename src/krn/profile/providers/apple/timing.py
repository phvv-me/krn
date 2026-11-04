# GPU start and end of work on the device clock: command buffers, and timestamp counter pairs per encoder.

import struct
from statistics import median
from typing import TYPE_CHECKING

from ....compiler.errors import TargetError
from .capabilities import Capabilities
from .signals import Signal, require

if TYPE_CHECKING:
    import Metal
    from Foundation import NSMakeRange
else:
    try:
        import Metal
        from Foundation import NSMakeRange
    except ImportError:
        Metal = None
        NSMakeRange = None

_CLOCK_SAMPLES = 16


def command_span(command: Metal.MTLCommandBuffer) -> tuple[int, int]:
    """The GPU start and end of a completed command buffer, in nanoseconds on the host clock."""
    return round(command.GPUStartTime() * 1e9), round(command.GPUEndTime() * 1e9)


def clock_offset_ns(device: Metal.MTLDevice) -> int:
    """The median of GPU minus CPU time over paired samples, which is the residual between the two clocks.

    The pair is sampled together by the driver, so this is a measurement to report, not a transform to apply.
    """
    pairs = [device.sampleTimestamps_gpuTimestamp_(None, None) for _ in range(_CLOCK_SAMPLES)]
    return round(median(gpu - cpu for cpu, gpu in pairs))


class StageTimer:
    """Per-encoder GPU start and end from a timestamp counter sampled at each encoder's stage boundaries.

    The timer owns `capacity` pairs of samples. `descriptor(i)` is the pass descriptor for encoder `i`, and
    `read(i)` returns its span once the command buffer has completed. On this device the ticks are nanoseconds
    that agree with `command_span` to one 41.7 ns tick, which the hardware tests check.
    """

    def __init__(self, device: Metal.MTLDevice, *, capacity: int) -> None:
        caps = Capabilities.probe(device)
        require(Signal.ENCODER_TIME, caps)
        descriptor = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
        descriptor.setCounterSet_(self._timestamp_set(device))
        descriptor.setSampleCount_(2 * capacity)
        descriptor.setStorageMode_(Metal.MTLStorageModeShared)
        buffer, error = device.newCounterSampleBufferWithDescriptor_error_(descriptor, None)
        if buffer is None:
            raise TargetError("GPU time per encoder from counter samples", caps.chip, str(error), "use command_span")
        self.buffer, self.capacity = buffer, capacity

    @staticmethod
    def _timestamp_set(device: Metal.MTLDevice) -> Metal.MTLCounterSet:
        return next(s for s in device.counterSets() if s.name() == "timestamp")

    def descriptor(self, index: int) -> Metal.MTLComputePassDescriptor:
        """A pass descriptor whose encoder samples its start and end into slots `2 * index` and `2 * index + 1`."""
        if not 0 <= index < self.capacity:
            raise IndexError(f"encoder {index} is outside the {self.capacity} samples of this timer")
        descriptor = Metal.MTLComputePassDescriptor.computePassDescriptor()
        attachment = descriptor.sampleBufferAttachments().objectAtIndexedSubscript_(0)
        attachment.setSampleBuffer_(self.buffer)
        attachment.setStartOfEncoderSampleIndex_(2 * index)
        attachment.setEndOfEncoderSampleIndex_(2 * index + 1)
        return descriptor

    def read(self, index: int) -> tuple[int, int]:
        """The encoder's `(start, end)` in nanoseconds; call only after its command buffer has completed."""
        raw = bytes(self.buffer.resolveCounterRange_(NSMakeRange(2 * index, 2)))
        start, end = struct.unpack("<2Q", raw)
        return start, end
