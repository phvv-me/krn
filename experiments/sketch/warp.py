"""Warp-level code: a ballot prefix written by hand and one intrinsic with a lowering per vendor.

TileLang's official ballot prefix (`examples/deep_select`) is 12 lines and untyped.
"""

from krn import OUT, T, intrinsic, kernel, msl, popcount, ptx, warp
from krn.facts import targets as targets


@kernel
def exclusive_sum(count: T.u32, bits: int) -> T.u32:
    """Exclusive sum of lane counts below `bits` bits, one ballot per bit plane instead of a shuffle scan."""
    below = warp.lanemask_lt()
    prefix = T.u32(0)
    for bit in range(bits):
        votes = warp.ballot(((count >> bit) & 1) != 0)
        prefix += popcount(votes & below) << bit
    return prefix


@intrinsic
def reduce_add(x: T.u32) -> T.u32:
    """Sum of x over the warp. This signature is the contract every lowering must meet."""
    raise NotImplementedError


@reduce_add.lower(targets.Nvidia, when=lambda t: t.sm >= 80)
def reduce_add_nvidia(x: T.u32) -> T.u32:
    return ptx(t"redux.sync.add.u32 {OUT}, {x}, 0xffffffff;", T.u32)


@reduce_add.lower(targets.Apple)
def reduce_add_apple(x: T.u32) -> T.u32:
    return msl(t"simd_sum({x})", T.u32)


@reduce_add.lower(targets.Target)
def reduce_add_portable(x: T.u32) -> T.u32:
    for step in range(warp.size().bit_length() - 1):
        x += warp.shfl_xor(x, 1 << step)
    return x
