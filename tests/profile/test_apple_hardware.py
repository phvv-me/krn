from typing import TYPE_CHECKING

import pytest

from krn.profile.providers.apple.capabilities import Capabilities
from krn.profile.providers.apple.timing import StageTimer, clock_offset_ns, command_span

if TYPE_CHECKING:
    from krn.runtime.metal import MetalDevice

Metal = pytest.importorskip("Metal", reason="PyObjC and the Metal framework are macOS only")

_TOUCH = """
#include <metal_stdlib>
using namespace metal;
kernel void touch(device uint* out [[buffer(0)]], constant uint& rounds [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    uint v = i;
    for (uint k = 0; k < rounds; ++k) v = v * 1664525u + 1013904223u;
    out[i] = v;
}
"""


@pytest.mark.hardware
def test_real_counter_api_is_the_timestamp_set_at_stage_boundaries(metal_gpu: MetalDevice) -> None:
    caps = Capabilities.probe(metal_gpu.core)
    assert caps.timestamps and caps.stage_sampling
    assert not caps.dispatch_sampling and not caps.statistics


@pytest.mark.hardware
def test_real_stage_counters_agree_with_command_buffer_spans(metal_gpu: MetalDevice) -> None:
    core = metal_gpu.core
    library, error = core.newLibraryWithSource_options_error_(_TOUCH, None, None)
    assert library is not None, error
    pipeline, error = core.newComputePipelineStateWithFunction_error_(library.newFunctionWithName_("touch"), None)
    assert pipeline is not None, error
    out = core.newBufferWithLength_options_(4096 * 4, Metal.MTLResourceStorageModeShared)
    assert out is not None
    rounds = core.newBufferWithBytes_length_options_(b"\x40\x00\x00\x00", 4, Metal.MTLResourceStorageModeShared)
    timer = StageTimer(core, capacity=8)
    queue = core.newCommandQueue()
    for index in range(8):
        command = queue.commandBuffer()
        encoder = command.computeCommandEncoderWithDescriptor_(timer.descriptor(index))
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(out, 0, 0)
        encoder.setBuffer_offset_atIndex_(rounds, 0, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(32, 1, 1), Metal.MTLSizeMake(128, 1, 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        start, end = command_span(command)
        counter_start, counter_end = timer.read(index)
        assert end - start > 0 and counter_end - counter_start > 0
        assert abs(counter_start - start) < 1_000 and abs(counter_end - end) < 1_000


@pytest.mark.hardware
def test_real_clock_offset_is_under_a_microsecond(metal_gpu: MetalDevice) -> None:
    assert abs(clock_offset_ns(metal_gpu.core)) < 1_000
