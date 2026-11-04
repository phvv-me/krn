"""NVIDIA cluster distributed shared memory; requires a cluster launch."""

from krn.frontend.numerics import Scalar, todo, u32


def load[T: Scalar](x: T, rank: u32) -> T:
    """The element `x` of a shared array as it is in the block of cluster rank `rank`."""
    return todo()


def store[T: Scalar](x: T, rank: u32, value: T) -> None:
    """Write `value` to the element `x` of a shared array in the block of cluster rank `rank`."""
    todo()
