"""itertools and more-itertools over the row `x[start:stop]` of a vector: generators that capture expands in place.

    for a, b in pairwise(x, start, stop):
        ...

A row is a vector with its `start` and `stop`, as a CSR row is; `start <= stop` holds, an empty row has `start == stop`.
Sizes and steps are capture-time constants, and unsigned bounds never wrap: every count is built with `max`.
A helper that only moves elements is generic in their type (`S` is the element type of the vector); the ones that add or
compare them (`accumulate`, `running_max`, `running_min`) take f32, since a static checker cannot add an `S`.
"""

from collections.abc import Iterator

from krn import Matrix, Vector, kernel
from krn.frontend.types import Idx, Scalar, f32, u32


@kernel
def chunked(stop: u32 | int, size: int) -> Iterator[tuple[Idx, Idx]]:
    """The (start, valid) of each chunk of `size` positions below `stop`; the last chunk may be shorter."""
    for start in range(0, stop, size):
        yield start, min(size, stop - start)


@kernel
def batched(stop: u32 | int, size: int) -> Iterator[tuple[Idx, Idx]]:
    """Half-open chunk bounds; unlike `chunked`, the second result is the stop position."""
    for start in range(0, stop, size):
        yield start, start + min(size, stop - start)


@kernel
def windowed[S: Scalar](
    x: Vector[S], start: u32 | int, stop: u32 | int, size: int, step: int = 1
) -> Iterator[tuple[S, ...]]:
    """Full windows only, without more-itertools' padding; `size` and `step` are positive constants."""
    windows = max(stop - start, size - 1) - (size - 1)
    for i in range(start, start + windows, step):
        yield tuple(x[i + k] for k in range(size))


@kernel
def pairwise[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int) -> Iterator[tuple[S, S]]:
    """Each element with the next one."""
    for i in range(start, max(stop, start + 1) - 1):
        yield x[i], x[i + 1]


@kernel
def accumulate(x: Vector[f32], start: u32 | int, stop: u32 | int) -> Iterator[f32]:
    """The running sum, one value per element."""
    total = 0.0
    for i in range(start, stop):
        total += x[i]
        yield total


@kernel
def running_max(x: Vector[f32], start: u32 | int, stop: u32 | int) -> Iterator[f32]:
    """Running maximum; NaN propagates from either operand, as in kernel `max`."""
    best = x[start] if start < stop else f32(0.0)
    for i in range(start, stop):
        best = max(best, x[i])
        yield best


@kernel
def running_min(x: Vector[f32], start: u32 | int, stop: u32 | int) -> Iterator[f32]:
    """Running minimum; NaN propagates from either operand, as in kernel `min`."""
    best = x[start] if start < stop else f32(0.0)
    for i in range(start, stop):
        best = min(best, x[i])
        yield best


@kernel
def groupby[K: Scalar](keys: Vector[K], start: u32 | int, stop: u32 | int) -> Iterator[tuple[K, Idx, Idx]]:
    """Adjacent equal keys as (key, run_start, run_stop), without allocating groups or accepting a key function."""
    i = start
    while i < stop:
        key = keys[i]
        j = i + 1
        while j < stop and keys[j] == key:
            j += 1
        yield key, i, j
        i = j


@kernel
def chain[S: Scalar](
    a: Vector[S],
    a_start: u32 | int,
    a_stop: u32 | int,
    b: Vector[S],
    b_start: u32 | int,
    b_stop: u32 | int,
) -> Iterator[S]:
    """Concatenate two rows; arbitrary iterable arguments cannot cross a device call."""
    for i in range(a_start, a_stop):
        yield a[i]
    for i in range(b_start, b_stop):
        yield b[i]


@kernel
def flatten[S: Scalar](m: Matrix[S], first_row: u32 | int, stop_row: u32 | int) -> Iterator[S]:
    """Row-major entries of a matrix's half-open row interval."""
    for r in range(first_row, stop_row):
        for c in range(m.shape[1]):
            yield m[r, c]


@kernel
def interleave[S: Scalar](a: Vector[S], b: Vector[S], start: u32 | int, stop: u32 | int) -> Iterator[S]:
    """Alternate two rows with identical bounds."""
    for i in range(start, stop):
        yield a[i]
        yield b[i]


@kernel
def intersperse[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, separator: S) -> Iterator[S]:
    """Insert a separator between entries, never before the first or after the last."""
    for i in range(start, stop):
        if i > start:
            yield separator
        yield x[i]


@kernel
def ilen(start: u32 | int, stop: u32 | int, step: int = 1) -> Idx:
    """Length of `range(start, stop, step)` at run time, without iterating; `step` is a positive constant."""
    return (max(stop, start) - start + step - 1) // step


@kernel
def take[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, n: u32 | int) -> Iterator[S]:
    """At most `n` entries from the beginning of the row."""
    for i in range(start, start + min(stop - start, n)):
        yield x[i]


@kernel
def tail[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, n: u32 | int) -> Iterator[S]:
    """At most `n` entries from the end, preserving their order."""
    for i in range(stop - min(stop - start, n), stop):
        yield x[i]


@kernel
def first[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, default: S) -> S:
    """Use the explicit default for an empty row; the unused load is not evaluated."""
    return x[start] if start < stop else default


@kernel
def last[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, default: S) -> S:
    """Use the explicit default for an empty row."""
    return x[stop - 1] if start < stop else default


@kernel
def nth[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, n: u32 | int, default: S) -> S:
    """Zero-based row offset `n`, or the explicit default when outside the row."""
    return x[start + n] if n < stop - start else default


@kernel
def partition[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, size: int) -> Iterator[tuple[S, ...]]:
    """Nonoverlapping tuples of a positive constant size, dropping the short tail."""
    for i in range(start, start + (stop - start) // size * size, size):
        yield tuple(x[i + k] for k in range(size))


@kernel
def drop[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, n: u32 | int) -> Iterator[S]:
    """Skip up to `n` entries at the beginning of the row."""
    for i in range(start + min(stop - start, n), stop):
        yield x[i]


@kernel
def take_nth[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, step: int) -> Iterator[S]:
    """Every `step`-th entry, including the first; `step` is a positive constant."""
    for i in range(start, stop, step):
        yield x[i]


@kernel
def second[S: Scalar](x: Vector[S], start: u32 | int, stop: u32 | int, default: S) -> S:
    """The second entry, with an explicit default instead of toolz's exception on a short row."""
    return nth(x, start, stop, 1, default)


@kernel
def range_right(start: u32 | int, stop: u32 | int, step: int = 1) -> Iterator[u32]:
    """Reverse a half-open range with a positive constant step; an inverted interval is empty."""
    for i in reversed(range(start, stop, step)):
        yield u32(i)
