"""Cooperative Groups, in the names of CUDA's `cooperative_groups`, written in the kernel language itself.

A group value says which threads take part in a collective, so `reduce` and the scans name their participants instead
of assuming the warp. The structs are small values whose methods delegate to `warp`, `block` and `grid`; capture reads
this file like `sparse.py` and inlines them, so one lowering serves every target and the emulator runs them for free.

    tile = cg.tiled_partition(cg.this_thread_block(), 8)
    total = tile.reduce(x[i], cg.plus)          # also cg.reduce(tile, x[i], cg.plus)

How each group lowers: a `Tile` of up to `warp.size()` lanes by shuffles over its segment of the warp; a `ThreadBlock`
by shuffles inside each warp and one slot per warp in shared memory (the block must be a whole number of warps);
`Coalesced`, `labeled_partition` and `binary_partition` by the active mask or `match_any` of the warp; `Grid` by the
grid barrier, which needs a cooperative launch; `Cluster` by the cluster barrier of sm_90a and later. A target that
cannot do one refuses it at capture, naming the reason and the architectures that can. A method that CUDA documents
for tiles only (`reduce`, the scans) is extended to blocks. `exclusive_scan` gives rank 0 a zero.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import overload

from ..kernel import kernel, static_assert
from ..memory import Vector
from ..records import Const, struct
from ..scopes import first_set, popcount, shared_array
from ..types import Integer, Scalar, u32, u64
from . import block, cluster, grid, warp

type Operator[T] = Callable[[T, T], T]

# `Scalar` has no operators for the checkers, so each operator is an overload for the typed face and a plain body for
# capture, which reads the body and types it by the arguments.


@overload
def plus[T: Scalar](a: T, b: T) -> T: ...


@overload
def plus(a: int, b: int) -> int: ...


@kernel
def plus(a, b):
    return a + b


@overload
def less[T: Scalar](a: T, b: T) -> T: ...


@overload
def less(a: int, b: int) -> int: ...


@kernel
def less(a, b):
    return min(a, b)


@overload
def greater[T: Scalar](a: T, b: T) -> T: ...


@overload
def greater(a: int, b: int) -> int: ...


@kernel
def greater(a, b):
    return max(a, b)


@overload
def bit_and[T: Integer](a: T, b: T) -> T: ...


@overload
def bit_and(a: int, b: int) -> int: ...


@kernel
def bit_and(a, b):
    return a & b


@overload
def bit_or[T: Integer](a: T, b: T) -> T: ...


@overload
def bit_or(a: int, b: int) -> int: ...


@kernel
def bit_or(a, b):
    return a | b


@overload
def bit_xor[T: Integer](a: T, b: T) -> T: ...


@overload
def bit_xor(a: int, b: int) -> int: ...


@kernel
def bit_xor(a, b):
    return a ^ b


@struct
class Group(ABC):
    """Threads participating in a block, tile, or active-warp collective."""

    @abstractmethod
    def size(self) -> u32: ...

    @abstractmethod
    def thread_rank(self) -> u32: ...

    @abstractmethod
    def sync(self) -> None: ...

    @abstractmethod
    def mask(self) -> u64: ...

    @abstractmethod
    def reduce[T: Scalar](self, x: T, op: Operator[T]) -> T: ...

    @abstractmethod
    def inclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T: ...

    @abstractmethod
    def exclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T: ...


@struct
class ThreadBlock(Group):
    """Block collectives require a multiple of the target warp size, at most 1024 threads."""

    def size(self) -> u32:
        return block.num_threads()

    def thread_rank(self) -> u32:
        return block.thread_rank()

    def sync(self) -> None:
        block.sync()

    def mask(self) -> u64:
        return u64(0xFFFFFFFFFFFFFFFF) >> (64 - warp.size())

    def reduce[T: Scalar](self, x: T, op: Operator[T]) -> T:
        tile = tiled_partition(self, warp.size())
        dtype = type(x)
        slots = shared_array(dtype, 1024 // warp.size())
        x = tile.reduce(x, op)
        if warp.lane() == 0:
            slots[warp.index()] = x
        block.sync()
        x = slots[0]
        for w in range(1, self.size() // warp.size()):
            x = op(x, slots[w])
        block.sync()
        return x

    def inclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T:
        tile = tiled_partition(self, warp.size())
        dtype = type(x)
        slots = shared_array(dtype, 1024 // warp.size())
        x = tile.inclusive_scan(x, op)
        if warp.lane() == warp.size() - 1:
            slots[warp.index()] = x
        block.sync()
        if warp.index() > 0:
            carry = slots[0]
            for w in range(1, warp.index()):
                carry = op(carry, slots[w])
            x = op(carry, x)
        block.sync()
        return x

    def exclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T:
        tile = tiled_partition(self, warp.size())
        dtype = type(x)
        slots = shared_array(dtype, 1024 // warp.size())
        total = tile.inclusive_scan(x, op)
        if warp.lane() == warp.size() - 1:
            slots[warp.index()] = total
        x = tile.shfl_up(total, 1)
        if warp.lane() == 0:
            x = type(x)(0)
        block.sync()
        if warp.index() > 0:
            carry = slots[0]
            for w in range(1, warp.index()):
                carry = op(carry, slots[w])
            x = carry if warp.lane() == 0 else op(carry, x)
        block.sync()
        return x


@struct
class Tile(Group):
    """`tiled_partition(parent, lanes)`: aligned segments of `lanes` threads of a warp, ranked inside the segment."""

    parent: Group
    lanes: Const[int]

    def size(self) -> u32:
        return u32(self.lanes)

    def thread_rank(self) -> u32:
        return warp.lane() % self.lanes

    def meta_group_rank(self) -> u32:
        return self.parent.thread_rank() // self.lanes

    def meta_group_size(self) -> u32:
        return self.parent.size() // self.lanes

    def mask(self) -> u64:
        # a tile as wide as the warp is every lane, so it needs no lane arithmetic (and no local, which costs scratch)
        return (u64(0xFFFFFFFFFFFFFFFF) >> (64 - self.lanes)) << (
            0 if self.lanes == warp.size() else int(warp.lane() - self.thread_rank())
        )

    def sync(self) -> None:
        warp.sync(self.mask())

    def shfl[T: Scalar](self, x: T, rank: int) -> T:
        return warp.shfl(x, rank, self.lanes, self.mask())

    def shfl_up[T: Scalar](self, x: T, delta: int) -> T:
        return warp.shfl_up(x, delta, self.lanes, self.mask())

    def shfl_down[T: Scalar](self, x: T, delta: int) -> T:
        return warp.shfl_down(x, delta, self.lanes, self.mask())

    def shfl_xor[T: Scalar](self, x: T, lane_mask: int) -> T:
        return warp.shfl_xor(x, lane_mask, self.lanes, self.mask())

    def any(self, predicate: bool) -> bool:
        return warp.any(predicate, self.mask())

    def all(self, predicate: bool) -> bool:
        return warp.all(predicate, self.mask())

    def ballot(self, predicate: bool) -> u64:
        return (warp.ballot(predicate, self.mask()) >> int(warp.lane() - self.thread_rank())) & (
            u64(0xFFFFFFFFFFFFFFFF) >> (64 - self.lanes)
        )

    def match_any[T: Scalar](self, x: T) -> u64:
        return (warp.match_any(x, self.mask()) >> int(warp.lane() - self.thread_rank())) & (
            u64(0xFFFFFFFFFFFFFFFF) >> (64 - self.lanes)
        )

    def reduce[T: Scalar](self, x: T, op: Operator[T]) -> T:
        for offset in [self.lanes >> k for k in range(1, 7) if self.lanes >> k != 0]:
            x = op(x, self.shfl_xor(x, offset))
        return x

    def inclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T:
        for offset in [1 << k for k in range(6) if (1 << k) < self.lanes]:
            y = self.shfl_up(x, int(offset))
            if self.thread_rank() >= offset:
                x = op(y, x)
        return x

    def exclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T:
        """Rank zero receives zero, including for operators whose identity differs."""
        x = self.inclusive_scan(x, op)
        x = self.shfl_up(x, 1)
        if self.thread_rank() == 0:
            x = type(x)(0)
        return x


@struct
class Coalesced(Group):
    """The threads of a warp that are together now (`coalesced_threads`) or share a label (`labeled_partition`).

    members: the absolute lane mask of the group; a rank counts the members below the calling lane.
    """

    members: u64

    def size(self) -> u32:
        return popcount(self.members)

    def thread_rank(self) -> u32:
        return popcount(self.members & warp.lanemask_lt())

    def mask(self) -> u64:
        return self.members

    def sync(self) -> None:
        warp.sync(self.members)

    def _lane(self, rank: u32) -> u32:
        m = self.members
        for _r in range(rank):
            m = m & (m - 1)
        return u32(first_set(m) - 1)

    def shfl[T: Scalar](self, x: T, rank: int) -> T:
        lane = warp.lane()
        if rank >= 0 and rank < int(self.size()):
            lane = self._lane(u32(rank))
        return warp.shfl(x, int(lane), warp.size(), self.members)

    def shfl_up[T: Scalar](self, x: T, delta: int) -> T:
        rank = self.thread_rank()
        source = rank
        if rank >= delta:
            source = rank - delta
        return self.shfl(x, int(source))

    def shfl_down[T: Scalar](self, x: T, delta: int) -> T:
        rank = self.thread_rank()
        source = rank
        if rank + delta < self.size():
            source = rank + delta
        return self.shfl(x, int(source))

    def any(self, predicate: bool) -> bool:
        return warp.any(predicate, self.members)

    def all(self, predicate: bool) -> bool:
        return warp.all(predicate, self.members)

    def _relative(self, absolute: u64) -> u64:
        result = u64(0)
        remaining = self.members
        rank = u32(0)
        while remaining != 0:
            bit = remaining & (u64(0) - remaining)
            if absolute & bit != 0:
                result = result | (u64(1) << int(rank))
            remaining = remaining & (remaining - 1)
            rank = rank + 1
        return result

    def ballot(self, predicate: bool) -> u64:
        absolute = warp.ballot(predicate, self.members)
        return self._relative(absolute)

    def match_any[T: Scalar](self, x: T) -> u64:
        absolute = warp.match_any(x, self.members)
        return self._relative(absolute)

    def inclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T:
        offset = u32(1)
        while offset < self.size():
            y = self.shfl_up(x, int(offset))
            if self.thread_rank() >= offset:
                x = op(y, x)
            offset = offset * 2
        return x

    def reduce[T: Scalar](self, x: T, op: Operator[T]) -> T:
        x = self.inclusive_scan(x, op)
        return self.shfl(x, int(self.size() - 1))

    def exclusive_scan[T: Scalar](self, x: T, op: Operator[T]) -> T:
        """Rank zero receives zero for every operator."""
        x = self.inclusive_scan(x, op)
        x = self.shfl_up(x, 1)
        if self.thread_rank() == 0:
            x = type(x)(0)
        return x


@struct
class Grid:
    """`this_grid()`: every thread of the launch. `sync` needs every block resident (a cooperative launch)."""

    def sync(self) -> None:
        grid.sync()

    def thread_rank(self) -> u64:
        return grid.thread_rank()

    def block_rank(self) -> u64:
        return u64(grid.block_rank())

    def num_threads(self) -> u64:
        return grid.num_threads()

    def num_blocks(self) -> u64:
        return u64(grid.num_blocks())

    def size(self) -> u64:
        return self.num_threads()


@struct
class Cluster:
    """`this_cluster()`: the blocks of a thread block cluster, on targets with `Feature.CLUSTERS`."""

    def sync(self) -> None:
        cluster.sync()

    def block_rank(self) -> u32:
        return cluster.block_rank()

    def num_blocks(self) -> u32:
        return cluster.num_blocks()


@kernel
def this_thread_block() -> ThreadBlock:
    return ThreadBlock()


@kernel
def this_grid() -> Grid:
    return Grid()


@kernel
def this_cluster() -> Cluster:
    return Cluster()


@kernel
def tiled_partition(parent: Group, lanes: int) -> Tile:
    static_assert(
        lanes > 0 and lanes & (lanes - 1) == 0 and lanes <= warp.size(),
        "tile size must be a capture-time power of two up to the target warp size; "
        "a larger tile needs shared memory, which this lowering does not use",
    )
    return Tile(parent, lanes)


@kernel
def coalesced_threads() -> Coalesced:
    members = warp.active_mask()
    return Coalesced(members)


@kernel
def labeled_partition[T: Scalar](parent: Group, label: T) -> Coalesced:
    members = warp.match_any(label, parent.mask())
    return Coalesced(members)


@kernel
def binary_partition(parent: Group, predicate: bool) -> Coalesced:
    members = warp.match_any(u32(predicate), parent.mask())
    return Coalesced(members)


@kernel
def reduce[T: Scalar](group: Group, x: T, op: Operator[T]) -> T:
    return group.reduce(x, op)


@kernel
def inclusive_scan[T: Scalar](group: Group, x: T, op: Operator[T]) -> T:
    return group.inclusive_scan(x, op)


@kernel
def exclusive_scan[T: Scalar](group: Group, x: T, op: Operator[T]) -> T:
    return group.exclusive_scan(x, op)


@kernel
def memcpy_async[T: Scalar](group: Group, dst: Vector[T], src: Vector[T], count: int) -> None:
    """Copy elements cooperatively; visibility is guaranteed after wait(group)."""
    i = group.thread_rank()
    while i < count:
        dst[i] = src[i]
        i = i + group.size()


@kernel
def wait(group: Group) -> None:
    group.sync()


@kernel
def invoke_one[*Args](group: Group, fn: Callable[[*Args], None], *args: *Args) -> None:
    """Rank zero calls fn without communication inside it; lowered natively at capture."""
    if group.thread_rank() == 0:
        fn(*args)
