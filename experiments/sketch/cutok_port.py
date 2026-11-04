"""Two real cutok kernels re-expressed with unchanged logic: a 16-byte-key hash probe and a block prefix.

Sources: cutoken/stages/merge/device/vocabulary.py lines 109-122 and giant_merger.py lines 128-143. The originals use
numba-shaped `u32`, `i64` and `Struct` types, so the port is nearly a rename.
What it must also keep is census-equal PTX.
"""

from krn import Const, T, Tensor, block, kernel, shared_array, struct, warp

WARPS = 8


@kernel
def splitmix(x: T.u64) -> T.u64:
    raise NotImplementedError


@struct
class Direct:
    """Open addressing on a 16-byte key. A zero payload marks an empty slot."""

    slots: Tensor[T.u64, int]
    mask: T.u64
    words: Const[int] = 3

    def lookup(self, low: T.u64, high: T.u64, length: T.u64) -> T.i64:
        slot = splitmix(low ^ splitmix(high ^ length)) & self.mask
        while True:
            base = slot * self.words
            payload = self.slots[base + 2]
            if payload == 0:
                return T.i64(-1)
            if self.slots[base] == low and self.slots[base + 1] == high and (payload >> 32) == length:
                return T.i64(payload & 0xFFFFFFFF)
            slot = (slot + 1) & self.mask


@struct
class Prefix:
    before: T.i32
    total: T.i32


@kernel
def block_prefix(count: T.i32) -> Prefix:
    """The tokens the warps before this one write, and the block's total."""
    counts = shared_array(T.i32, WARPS)
    if warp.lane() == 0:
        counts[warp.index()] = count
    block.sync()
    before = T.i32(0)
    total = T.i32(0)
    for index in range(WARPS):
        if index < warp.index():
            before += counts[index]
        total += counts[index]
    return Prefix(before=before, total=total)
