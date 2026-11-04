"""Hardware facts as compile-time variables: one kernel for a 32-lane warp, a 64-lane wave and a 16-lane CPU vector.

`target.warp_size` is a constant while the kernel is captured, so the loop below unrolls to 5 steps on NVIDIA, 6 on AMD
and 4 on an AVX-512 CPU. A kernel that needs something the target lacks fails at capture, naming the missing feature.
"""

from krn import (
    Feature,
    T,
    Vector,
    block,
    grid,
    kernel,
    requires,
    shared_array,
    static_assert,
    static_warn,
    target,
    warp,
)


@kernel
def butterfly_steps(width: int) -> int:
    """Rounds of an xor butterfly over `width` lanes."""
    return width.bit_length() - 1


@kernel
def lane_sum(x: T.f32) -> T.f32:
    """The sum over the lockstep lanes, whatever their number."""
    static_assert(target.warp_size & (target.warp_size - 1) == 0, "the lane count must be a power of two")
    for step in range(butterfly_steps(target.warp_size)):
        x += warp.shfl_xor(x, 1 << step)
    return x


@kernel
@requires(Feature.SHARED_MEMORY, Feature.WARP_SHUFFLE)
def block_sum(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    """Warp sums, then one slot per warp in shared memory, then the first warp adds those."""
    static_warn(target.shared_bytes >= 4 * 1024, "shared memory is small on this target, expect spills to global")
    slots = shared_array(T.f32, 1024 // target.warp_size)
    i = grid.thread_rank()
    total = lane_sum(x[i] if i < x.shape[0] else T.f32(0.0))
    if warp.lane() == 0:
        slots[warp.index()] = total
    block.sync()
    if warp.index() == 0:
        out[block.index().x] = lane_sum(slots[warp.lane()] if warp.lane() < slots.shape[0] else T.f32(0.0))
