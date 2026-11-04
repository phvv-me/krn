"""Top-1 per row with one warp per row, written once for 32-lane warps, 64-lane waves and CPU vectors.

The warp is the lockstep group, and its width is a compile-time fact of the target: 32 on NVIDIA and Apple, 64 on AMD
CDNA, and the number of float32 lanes of a vector register on a CPU (8 on AVX2, 16 on AVX-512). Nothing below says
32. The loop stride is `warp.size()`, the lane masks are always 64 bits (`Mask`), so a 64-lane ballot fits, and
`warp.argmax` is a typed intrinsic with a lowering per target (a shuffle tree, a DPP tree, `simd_max` plus a ballot,
a horizontal reduction).

Compile-time checks, evaluated while the kernel is captured:

* `static_assert` on the width fails the capture for a target whose width the algorithm does not support;
* `@requires(Feature.WARP_VOTE)` fails with the missing feature and the targets that have it, instead of an
  assembler error later;
* `static_warn` reports a slow case without refusing it.

Launch: `row_argmax(x, values, indices, launch=Items(x.shape[0] * 32))` (32 threads per row on a 32-lane target).

Python features used: decorator stacking, tuple assignment of a (value, index) pair, and `math.inf` as a typed constant.
"""

import math

from krn import Feature, Matrix, T, Vector, grid, kernel, requires, static_assert, static_warn, target, warp


@kernel
@requires(Feature.WARP_VOTE)
def row_argmax(x: Matrix[T.f32], values: Vector[T.f32], indices: Vector[T.i32]) -> None:
    """values[r] and indices[r] hold the largest element of row r and the lowest column that holds it."""
    static_assert(warp.size() in (8, 16, 32, 64), "warp.argmax is written for widths from 8 to 64")
    static_warn(x.shape[1] >= warp.size(), f"rows shorter than {target.warp_size} leave lanes idle")
    row = grid.thread_rank() // warp.size()
    best, best_col = T.f32(-math.inf), T.u32(0)
    for c in range(warp.lane(), x.shape[1], warp.size()):
        v = x[row, c]
        if v > best:
            best, best_col = v, T.u32(c)
    best, best_col = warp.argmax(best, best_col)
    if warp.lane() == 0:
        values[row] = best
        indices[row] = T.i32(best_col)
