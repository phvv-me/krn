"""Shared scope types, bit and math helpers, and memory declarations.

Execution functions live in `groups`; `Mask` covers both 32-lane and 64-lane warps.
"""

from collections.abc import Coroutine
from types import TracebackType

from .memory import Tensor
from .numerics import Integer, Scalar, f32, todo, u32, u64

type Mask = u64


class Dim3:
    x: u32
    y: u32
    z: u32


class Roles:
    """The warps of a block assigned to concurrent roles, written as the tasks of a task group.

    `spawn(coroutine, warps=n)` runs the coroutine on `n` warps; leaving the `async with` joins them all. A producer
    that streams tiles into a ring and a consumer that multiplies them are the warp-specialized GEMM of Hopper and
    Blackwell, and the same code is a plain producer/consumer loop on a target without independent warps, where the
    compiler runs the roles one after the other (`Feature.BULK_COPY` is not needed for correctness, only for speed).
    """

    async def __aenter__(self) -> Roles:
        return todo()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return todo()

    def spawn(self, role: Coroutine[None, None, None], *, warps: int = 1) -> None:
        return todo()


def popcount(x: Integer) -> u32:
    """The number of bits set."""
    return todo()


def reverse_bits[T: Integer](x: T) -> T:
    """The bit pattern read backwards: 0b10110110 becomes 0b01101101 in 8 bits."""
    return todo()


def leading_zeros(x: Integer) -> u32:
    """The number of zero bits above the highest set bit; the width of the type for 0."""
    return todo()


def first_set(x: Integer) -> u32:
    """The position of the lowest set bit counting from 1, and 0 for 0."""
    return todo()


def fma[T: Scalar](a: T, b: T | float, c: T | float) -> T:
    """`a * b + c` with one rounding, in the type of the operands (`fma.rn.f32`, `v_fma_f32`, `fma`)."""
    return todo()


def cbrt[T: Scalar](x: T) -> T:
    """The cube root."""
    return todo()


def log10[T: Scalar](x: T) -> T:
    """The base 10 logarithm."""
    return todo()


def exp10[T: Scalar](x: T) -> T:
    """10 to the power x."""
    return todo()


def trunc[T: Scalar](x: T) -> T:
    """Round toward zero."""
    return todo()


def sleep(nanoseconds: int) -> None:
    """Suspend the thread for about this long. A target without a sleep instruction returns at once."""
    return todo()


def exp(x: f32) -> f32:
    return todo()


def shared_array[T: Scalar](dtype: type[T], size: int) -> Tensor[T, int]:
    """Block-shared memory, sized at compile time. Every thread of the block sees the same array."""
    return todo()


def local_array[T: Scalar](dtype: type[T], size: int) -> Tensor[T, int]:
    """Per-thread memory, sized at compile time. Every thread sees its own array."""
    return todo()


def constant[T: Scalar](values: Tensor[T, int]) -> Tensor[T, int]:
    """Constant memory filled from a host array while the kernel is captured. Every thread sees the same values."""
    return todo()
