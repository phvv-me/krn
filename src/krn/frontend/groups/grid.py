"""The grid: its shape (`gridDim`), the linear ranks of the calling thread and block, and the grid barrier.

Ranks are linear over x, y and z with x varying fastest (`grid_group::thread_rank()` of Cooperative Groups), so
`grid.thread_rank()` is the index of one item per thread whatever the launch shape. Bodies are signature-only."""

from ..numerics import todo, u32, u64
from ..scopes import Dim3


def dim() -> Dim3:
    """Blocks per axis of the grid."""
    return todo()


def num_threads() -> u64:
    """Number of threads over every block and axis of the grid."""
    return todo()


def num_blocks() -> u32:
    """Number of blocks over all three axes."""
    return todo()


def block_rank() -> u32:
    """Linear block rank, with x varying fastest."""
    return todo()


def thread_rank() -> u64:
    """Linear thread rank over every block and axis of the grid."""
    return todo()


def sync() -> None:
    """A barrier for the whole grid. It needs a cooperative launch, where every block is resident at once.

    The CUDA and HIP text is a header-free atomic barrier that is correct only while every block is resident, so keep
    the grid within what the device holds (a cooperative launch enforces it). Apple and CPU targets refuse it."""
    return todo()
