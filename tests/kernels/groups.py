"""Cooperative Groups exercised through source inlining."""

from krn import Const, Matrix, Scalar, Vector, cg, grid, kernel, shared_array, struct, warp
from krn.frontend.types import i32, u32, u64


@struct
class Tiling:
    lanes: Const[int]
    operation: Const[int] = 0
    collective: Const[int] = 0


@kernel
def tile_collective[T: Scalar](t: Tiling, x: Vector[T], out: Vector[T]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), t.lanes)
    op = cg.plus
    if t.operation == 1:
        op = cg.less
    if t.operation == 2:
        op = cg.greater
    if t.operation == 3:
        op = cg.bit_and
    if t.operation == 4:
        op = cg.bit_or
    if t.operation == 5:
        op = cg.bit_xor
    i = grid.thread_rank()
    if t.collective == 0:
        out[i] = cg.reduce(tile, x[i], op)
    if t.collective == 1:
        out[i] = cg.inclusive_scan(tile, x[i], op)
    if t.collective == 2:
        out[i] = cg.exclusive_scan(tile, x[i], op)


@kernel
def tile_indices(t: Tiling, out: Matrix[u64]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), t.lanes)
    i = grid.thread_rank()
    out[i, 0] = u64(tile.size())
    out[i, 1] = u64(tile.thread_rank())
    out[i, 2] = u64(tile.meta_group_rank())
    out[i, 3] = u64(tile.meta_group_size())
    out[i, 4] = tile.mask()


@kernel
def tile_votes(t: Tiling, x: Vector[u32], out: Matrix[u64]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), t.lanes)
    tile.sync()
    i = grid.thread_rank()
    out[i, 0] = tile.ballot(x[i] > 0)
    out[i, 1] = u64(tile.any(x[i] > 0))
    out[i, 2] = u64(tile.all(x[i] > 0))


@kernel
def tile_matching(t: Tiling, x: Vector[u32], out: Vector[u64]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), t.lanes)
    i = grid.thread_rank()
    out[i] = tile.match_any(x[i])


@kernel
def tile_shuffles(t: Tiling, x: Vector[u32], out: Matrix[u32]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), t.lanes)
    i = grid.thread_rank()
    out[i, 0] = tile.shfl(x[i], 1)
    out[i, 1] = tile.shfl_up(x[i], 1)
    out[i, 2] = tile.shfl_down(x[i], 1)
    out[i, 3] = tile.shfl_xor(x[i], 1)


@kernel
def block_collective[T: Scalar](t: Tiling, x: Vector[T], out: Vector[T]) -> None:
    group = cg.this_thread_block()
    op = cg.plus
    if t.operation == 1:
        op = cg.less
    if t.operation == 2:
        op = cg.greater
    i = grid.thread_rank()
    if t.collective == 0:
        out[i] = group.reduce(x[i], op)
    if t.collective == 1:
        out[i] = group.inclusive_scan(x[i], op)
    if t.collective == 2:
        out[i] = group.exclusive_scan(x[i], op)


@kernel
def block_indices(out: Matrix[u32]) -> None:
    group = cg.this_thread_block()
    i = grid.thread_rank()
    out[i, 0] = group.size()
    out[i, 1] = group.thread_rank()


@kernel
def coalesced(x: Vector[i32], out: Matrix[i32]) -> None:
    i = grid.thread_rank()
    if x[i] > 0:
        group = cg.coalesced_threads()
        out[i, 0] = group.reduce(x[i], cg.plus)
        out[i, 1] = i32(group.thread_rank())
        out[i, 2] = i32(group.size())
        out[i, 3] = group.inclusive_scan(x[i], cg.plus)
    else:
        out[i, 0] = -1
        out[i, 1] = -1
        out[i, 2] = -1
        out[i, 3] = -1


@kernel
def partitioned[T: Scalar](t: Tiling, x: Vector[T], out: Vector[T]) -> None:
    tile = cg.tiled_partition(cg.this_thread_block(), warp.size())
    i = grid.thread_rank()
    if t.operation == 0:
        group = cg.labeled_partition(tile, x[i] % 3)
    else:
        group = cg.binary_partition(tile, x[i] > 0)
    out[i] = group.reduce(x[i], cg.plus)


@kernel
def store_total(out: Vector[u32], total: u32) -> None:
    out[grid.thread_rank()] = total


@kernel
def copied(x: Vector[u32], out: Vector[u32]) -> None:
    group = cg.this_thread_block()
    shared = shared_array(u32, 256)
    cg.memcpy_async(group, shared, x, 256)
    cg.wait(group)
    i = group.thread_rank()
    while i < 256:
        out[i] = shared[255 - i]
        i = i + group.size()


@kernel
def invoked(x: Vector[u32], out: Vector[u32]) -> None:
    group = cg.this_thread_block()
    total = group.reduce(x[grid.thread_rank()], cg.plus)
    cg.invoke_one(group, store_total, out, total)


@kernel
def grid_barrier(x: Vector[u32], out: Vector[u32]) -> None:
    group = cg.this_grid()
    i = group.thread_rank()
    x[i] = u32(i)
    group.sync()
    out[i] = x[(i + 64) % group.size()]


@kernel
def clustered(out: Matrix[u32]) -> None:
    group = cg.this_cluster()
    i = grid.thread_rank()
    out[i, 0] = group.block_rank()
    out[i, 1] = group.num_blocks()
    group.sync()


@kernel
def coalesced_functions(x: Vector[u32], out: Matrix[u64]) -> None:
    i = grid.thread_rank()
    if x[i] > 0:
        group = cg.coalesced_threads()
        group.sync()
        out[i, 0] = u64(group.shfl(x[i], 1))
        out[i, 1] = u64(group.shfl_up(x[i], 2))
        out[i, 2] = u64(group.shfl_down(x[i], 2))
        out[i, 3] = u64(group.shfl(x[i], int(group.size())))
        out[i, 4] = u64(group.shfl(x[i], -1))
        out[i, 5] = group.ballot(x[i] % 3 == 0)
        out[i, 6] = u64(group.any(x[i] % 3 == 0))
        out[i, 7] = u64(group.all(x[i] % 3 == 0))
        out[i, 8] = u64(group.exclusive_scan(x[i], cg.plus))
        out[i, 9] = group.mask()


@kernel
def coalesced_matching(x: Vector[u32], out: Vector[u64]) -> None:
    i = grid.thread_rank()
    if x[i] > 0:
        group = cg.coalesced_threads()
        out[i] = group.match_any(x[i] % 3)


@kernel
def grid_indices(out: Matrix[u64]) -> None:
    group = cg.this_grid()
    i = group.thread_rank()
    out[i, 0] = group.block_rank()
    out[i, 1] = group.num_blocks()
    out[i, 2] = group.num_threads()
    out[i, 3] = group.size()


@kernel
def repeated_grid_barrier(x: Vector[u32], out: Vector[u32]) -> None:
    shared = shared_array(u32, 64)
    i = grid.thread_rank()
    local = grid.thread_rank() % 64
    shared[local] = u32(i)
    block = cg.this_thread_block()
    group = cg.this_grid()
    for phase in range(3):
        block.sync()
        x[i] = shared[63 - local] + u32(phase)
        group.sync()
        out[i] = x[(i + 64) % group.size()]
        group.sync()
