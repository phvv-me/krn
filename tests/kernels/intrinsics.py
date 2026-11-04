"""Kernels that use the intrinsic vocabulary: atomics, bit operations, math helpers, warp collectives and debug aids."""

from krn import (
    Matrix,
    T,
    Vector,
    atomic,
    block,
    cbrt,
    first_set,
    fma,
    grid,
    items,
    kernel,
    leading_zeros,
    popcount,
    reverse_bits,
    thread,
    warp,
)


@kernel
def histogram(values: Vector[T.i32], bins: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    if i < values.shape[0]:
        atomic.add(bins, values[i], 1)


@kernel
def largest(x: Vector[T.f32], best: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        atomic.max(best, 0, x[i])


@kernel
def smallest(x: Vector[T.i64], low: Vector[T.i64]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        atomic.min(low, 0, x[i])


@kernel
def wrapping_counter(limit: T.u32, counter: Vector[T.u32], seen: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < seen.shape[0]:
        seen[i] = atomic.inc(counter, 0, limit)


@kernel
def take_lock(lock: Vector[T.i32], winners: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    if i < winners.shape[0]:
        old = atomic.cas(lock, 0, 0, 1)
        atomic.add(winners, 0, 1 if old == 0 else 0)


@kernel
def bit_tricks(
    x: Vector[T.u32],
    ones: Vector[T.u32],
    reversed_bits: Vector[T.u32],
    zeros: Vector[T.u32],
    low: Vector[T.u32],
) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        ones[i] = popcount(x[i])
        reversed_bits[i] = reverse_bits(x[i])
        zeros[i] = leading_zeros(x[i])
        low[i] = first_set(x[i])


@kernel
def math_helpers(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        v = x[i]
        out[i] = fma(v, 2.0, 1.0) + cbrt(v) + v.log10() + v.trunc()


@kernel
def warp_sum(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    total = x[i]
    for step in range(5):
        total = total + warp.shfl_xor(total, 1 << step)
    if warp.lane() == 0:
        out[i // 32] = total


@kernel
def votes(x: Vector[T.i32], ballots: Vector[T.u64], everyone: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    positive = x[i] > 0
    mask = warp.ballot(positive)
    unanimous = warp.all(positive)
    if warp.lane() == 0:
        ballots[i // 32] = mask
        everyone[i // 32] = 1 if unanimous else 0


@kernel
def ordered(a: Vector[T.f32], flag: Vector[T.i32]) -> None:
    a[thread.index().x] = 1.0
    thread.fence_device()
    block.sync()
    flag[0] = 1


@kernel
def debug_aids(x: Vector[T.i32], out: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        assert x[i] >= 0, "x must not be negative"
        print("thread", i, "reads", x[i])
        breakpoint()
        out[i] = x[i]


@kernel
def block_votes(x: Vector[T.i32], counts: Vector[T.u32], anyone: Vector[T.i32]) -> None:
    i = grid.thread_rank()
    positive = x[i] > 0
    n = block.count(positive)
    some = block.any(positive)
    if thread.index().x == 0:
        counts[block.index().x] = n
        anyone[block.index().x] = 1 if some else 0


@kernel
def reads_past_the_end(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    out[i] = x[i + 1]


@kernel
def split_barrier(out: Vector[T.i32]) -> None:
    if thread.index().x % 2 == 0:
        block.sync()
    else:
        out[thread.index().x] = warp.ballot(True).to(T.i32)


@kernel
def tiles_wide(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    y[rows, cols] = x[rows, cols] + 1.0


@kernel
def group_ranks(out: Matrix[T.u64]) -> None:
    i = grid.thread_rank()
    out[i, 0] = T.u64(block.thread_rank())
    out[i, 1] = T.u64(warp.lane())
    out[i, 2] = T.u64(warp.index())
    out[i, 3] = T.u64(grid.block_rank())
    out[i, 4] = T.u64(block.num_threads())
    out[i, 5] = grid.num_threads()
    out[i, 6] = T.u64(grid.num_blocks())


@kernel
def segmented_shuffles(x: Vector[T.i32], out: Matrix[T.i32], width: T.u32, amount: T.u32) -> None:
    i = grid.thread_rank()
    out[i, 0] = warp.shfl_xor(x[i], amount, width)
    out[i, 1] = warp.shfl_up(x[i], delta=amount, width=width)
    out[i, 2] = warp.shfl_down(x=x[i], delta=amount, width=width)
    out[i, 3] = warp.shfl(src=amount, width=width, x=x[i])


@kernel
def lane_masks(out: Matrix[T.u64]) -> None:
    i = grid.thread_rank()
    out[i, 0] = warp.lanemask_lt()
    out[i, 1] = warp.lanemask_le()
    out[i, 2] = warp.lanemask_eq()
    out[i, 3] = warp.lanemask_ge()
    out[i, 4] = warp.lanemask_gt()


@kernel
def masked_groups(out: Vector[T.u64]) -> None:
    i = grid.thread_rank()
    lane = warp.lane()
    if lane < 8:
        value = warp.ballot(True, mask=T.u64(255))
    else:
        value = warp.ballot(True, mask=T.u64(4294967040))
    out[i] = value


@kernel
def matching_lanes(x: Vector[T.i32], masks: Vector[T.u64], unanimous: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    mask, equal = warp.match_all(x=x[i])
    masks[i] = mask
    unanimous[i] = 1 if equal else 0


@kernel
def group_items(out: Vector[T.u32]) -> None:
    for i in items(out.shape[0]):
        out[i] += 1
