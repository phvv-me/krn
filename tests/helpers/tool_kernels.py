"""Small launchable kernels, one per helper: the oracle targets of the tests and the launches of the example.

They live beside the helpers because capture follows a relative import to inline a device function, but refuses an
absolute `from krn.itertools import chunked` for a generator. A kernel writes its output into the caller's
`out` in order; no kernel allocates a result. The ones that only move elements are generic in their type: `S` is the
element type of the vectors, bound when the kernel is captured.
"""

from krn import Vector, items, kernel
from krn.frontend.memory import Matrix
from krn.frontend.types import Scalar, f32, u32
from krn.functools import count_by, max_by, min_by
from krn.itertools import (
    accumulate,
    batched,
    chain,
    chunked,
    drop,
    first,
    flatten,
    groupby,
    ilen,
    interleave,
    intersperse,
    last,
    nth,
    pairwise,
    partition,
    range_right,
    running_max,
    running_min,
    second,
    tail,
    take,
    take_nth,
    windowed,
)
from krn.itertools import take as toolz_take
from krn.math import clip, in_range, inverse_lerp, lerp, mean


@kernel
def run_chunked(n: u32, out: Vector[u32]) -> None:
    j = 0
    for start, valid in chunked(n, 2):
        out[j] = u32(start)
        out[j + 1] = u32(valid)
        j += 2


@kernel
def run_batched(n: u32, out: Vector[u32]) -> None:
    j = 0
    for start, stop in batched(n, 2):
        out[j] = u32(start)
        out[j + 1] = u32(stop)
        j += 2


@kernel
def run_windowed[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    j = 0
    for a, b, c in windowed(x, start, n, 3):
        out[j] = a
        out[j + 1] = b
        out[j + 2] = c
        j += 3


@kernel
def run_windowed_step[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    j = 0
    for a, b, c in windowed(x, start, n, 3, 2):
        out[j] = a
        out[j + 1] = b
        out[j + 2] = c
        j += 3


@kernel
def run_pairwise[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    j = 0
    for a, b in pairwise(x, start, n):
        out[j] = a
        out[j + 1] = b
        j += 2


@kernel
def run_accumulate(x: Vector[f32], start: u32, n: u32, out: Vector[f32]) -> None:
    for j, total in enumerate(accumulate(x, start, n)):
        out[j] = total


@kernel
def run_running_max(x: Vector[f32], start: u32, n: u32, out: Vector[f32]) -> None:
    for j, value in enumerate(running_max(x, start, n)):
        out[j] = value


@kernel
def run_running_min(x: Vector[f32], start: u32, n: u32, out: Vector[f32]) -> None:
    for j, value in enumerate(running_min(x, start, n)):
        out[j] = value


@kernel
def run_groupby[K: Scalar](keys: Vector[K], start: u32, n: u32, out: Vector[u32]) -> None:
    j = 0
    for key, lo, hi in groupby(keys, start, n):
        out[j] = u32(key)
        out[j + 1] = u32(lo)
        out[j + 2] = u32(hi)
        j += 3


@kernel
def run_chain[S: Scalar](a: Vector[S], b: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(chain(a, start, n, b, start, n)):
        out[j] = value


@kernel
def run_flatten[S: Scalar](m: Matrix[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(flatten(m, start, n)):
        out[j] = value


@kernel
def run_interleave[S: Scalar](a: Vector[S], b: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(interleave(a, b, start, n)):
        out[j] = value


@kernel
def run_intersperse[S: Scalar](x: Vector[S], fill: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(intersperse(x, start, n, fill[0])):
        out[j] = value


@kernel
def run_ilen(start: u32, n: u32, out: Vector[u32]) -> None:
    out[0] = u32(ilen(start, n))


@kernel
def run_ilen_step(start: u32, n: u32, out: Vector[u32]) -> None:
    out[0] = u32(ilen(start, n, 3))


@kernel
def run_take[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(take(x, start, n, 3)):
        out[j] = value


@kernel
def run_tail[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(tail(x, start, n, 3)):
        out[j] = value


@kernel
def run_first[S: Scalar](x: Vector[S], fill: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    out[0] = first(x, start, n, fill[0])


@kernel
def run_last[S: Scalar](x: Vector[S], fill: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    out[0] = last(x, start, n, fill[0])


@kernel
def run_nth[S: Scalar](x: Vector[S], fill: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    out[0] = nth(x, start, n, 2, fill[0])


@kernel
def run_partition[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    j = 0
    for a, b in partition(x, start, n, 2):
        out[j] = a
        out[j + 1] = b
        j += 2


@kernel
def run_drop[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(drop(x, start, n, 3)):
        out[j] = value


@kernel
def run_take_nth[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(take_nth(x, start, n, 3)):
        out[j] = value


@kernel
def run_second[S: Scalar](x: Vector[S], fill: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    out[0] = second(x, start, n, fill[0])


@kernel
def run_toolz_take[S: Scalar](x: Vector[S], start: u32, n: u32, out: Vector[S]) -> None:
    for j, value in enumerate(toolz_take(x, start, n, 3)):
        out[j] = value


@kernel
def run_clip(x: Vector[f32], n: u32, out: Vector[f32]) -> None:
    for i in range(n):
        out[i] = clip(x[i], f32(5.0), f32(15.0))


@kernel
def run_in_range(x: Vector[f32], low: Vector[f32], high: Vector[f32], n: u32, out: Vector[u32]) -> None:
    for i in range(n):
        out[i] = u32(in_range(x[i], low[i], high[i]))


@kernel
def run_lerp(a: Vector[f32], b: Vector[f32], fraction: Vector[f32], n: u32, out: Vector[f32]) -> None:
    for i in range(n):
        out[i] = lerp(a[i], b[i], fraction[i])


@kernel
def run_inverse_lerp(a: Vector[f32], b: Vector[f32], value: Vector[f32], n: u32, out: Vector[f32]) -> None:
    for i in range(n):
        out[i] = inverse_lerp(a[i], b[i], value[i])


@kernel
def run_mean(x: Vector[f32], start: u32, n: u32, out: Vector[f32]) -> None:
    out[0] = mean(x, start, n)


@kernel
def run_min_by(keys: Vector[f32], start: u32, n: u32, out: Vector[u32]) -> None:
    out[0] = u32(min_by(keys, start, n))


@kernel
def run_max_by(keys: Vector[f32], start: u32, n: u32, out: Vector[u32]) -> None:
    out[0] = u32(max_by(keys, start, n))


@kernel
def run_count_by(keys: Vector[u32], start: u32, n: u32, out: Vector[u32]) -> None:
    count_by(keys, start, n, out)


@kernel
def run_range_right(start: u32, n: u32, out: Vector[u32]) -> None:
    for j, value in enumerate(range_right(start, n)):
        out[j] = value


@kernel
def run_range_right_step(start: u32, n: u32, out: Vector[u32]) -> None:
    for j, value in enumerate(range_right(start, n, 5)):
        out[j] = value


@kernel
def run_min_by_rows(keys: Vector[f32], indptr: Vector[u32], n: u32, out: Vector[u32]) -> None:
    """The realistic shape: a thread to a CSR row, the helper reading the row's bounds."""
    for r in items(n):
        out[r] = u32(min_by(keys, indptr[r], indptr[r + 1]))


@kernel
def unit_clip(value: f32) -> f32:
    """`functools.partial(clip, minimum=0, maximum=1)` at capture: a device function that binds two arguments."""
    return clip(value, f32(0.0), f32(1.0))


@kernel
def run_unit_clip(x: Vector[f32], n: u32, out: Vector[f32]) -> None:
    for i in range(n):
        out[i] = unit_clip(x[i])
