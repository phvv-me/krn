"""The block: its coordinates in the grid (`blockIdx`), its shape (`blockDim`), the linear rank of the calling thread
and the barriers. `block.tile` and `block.roles` are the sketch's own additions for tile kernels and warp roles.
A thread rank is linear over x, y and z with x varying fastest, which is how CUDA numbers the threads of a warp. Bodies
are signature-only."""

from typing import overload

from ..numerics import Scalar, todo, u32
from ..scopes import Dim3, Roles


def index() -> Dim3:
    """Coordinates of this block inside the grid."""
    return todo()


def dim() -> Dim3:
    """Threads per axis of this block."""
    return todo()


def num_threads() -> u32:
    """Number of threads over all three axes."""
    return todo()


def thread_rank() -> u32:
    """Linear rank inside the block, with x varying fastest."""
    return todo()


def roles() -> Roles:
    """Split the warps of this block into roles: `async with block.roles() as roles: roles.spawn(...)`."""
    return todo()


@overload
def tile(size: int, /) -> slice: ...
@overload
def tile(rows: int, cols: int, /) -> tuple[slice, slice]: ...
def tile(*sizes: int) -> slice | tuple[slice, slice]:
    """The tile of the output this block owns: `rows, cols = block.tile(bm, bn)` (rows from y, columns from x)."""
    return todo()


def sync() -> None:
    """A barrier for every thread of the block. All of them must reach it, or the behavior is undefined."""
    return todo()


def count(predicate: bool) -> u32:
    """A barrier that returns how many threads of the block have `predicate` true."""
    return todo()


def all(predicate: bool) -> bool:
    """A barrier that returns whether `predicate` is true for every thread of the block."""
    return todo()


def any(predicate: bool) -> bool:
    """A barrier that returns whether `predicate` is true for at least one thread of the block."""
    return todo()


def sum[T: Scalar](x: T) -> T:
    """Block-wide reduction, written once as warp reductions plus shared memory."""
    return todo()
