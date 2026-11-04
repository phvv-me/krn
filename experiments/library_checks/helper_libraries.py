"""Helper libraries inside kernels: itertools, more-itertools, toolz and es-toolkit as device functions and generators.

A helper is a `@kernel` function that capture inlines where it is called, or a generator that capture expands inside
the `for` loop consuming it, so it costs what the hand-written loop costs and allocates nothing:

    for a, b in pairwise(x, start, stop):         # itertools.pairwise over the row x[start:stop]
    for key, lo, hi in groupby(keys, start, stop): # itertools.groupby on adjacent keys, as (key, run start, run stop)
    for window in windowed(x, start, stop, 3):     # more_itertools.windowed, full windows, a tuple of registers

A row is a vector with its `start` and `stop`, as a CSR row is. Sizes and steps are capture-time constants. The helpers
that only move elements are generic in the element type; the ones that add or compare them take f32.

    krn.itertools   chunked batched windowed pairwise accumulate running_max running_min groupby chain
                                flatten interleave intersperse ilen take tail first last nth
    krn.itertools       partition drop take_nth second, and take tail first last nth interleave re-exported
    krn.functools   clip in_range lerp inverse_lerp mean min_by max_by count_by range_right

The kernels below are imported from `tests.helpers.tool_kernels`: capture inlines a helper reached by a relative import,
and supports absolute `from krn.itertools import chunked` for a generator, so the kernels sit beside the
helpers until that gap closes. Not offered, because a device call cannot take a predicate, a function, a generator or a
register sequence: compose, pipe, curry, partial, juxt, reduceby, groupby(key), accumulate(func), min/max with a key,
map_reduce, locate, unique and every dict function. `notes/round9/helpers-status.md` lists each with its reason.

Run from the workspace root: `KRN_TARGET=avx512 mb run python experiments/library_checks/helper_libraries.py`.
The teaching version, three kernels long, is `examples/02_tiles_and_types/05_helper_functions.py`.

Python features used: generators, `yield`, PEP 695 type parameters, tuples of registers, `enumerate`, `reversed`,
conditional expressions, short-circuit `and`, relative imports.
"""

import asyncio
import itertools as it
import logging
import shutil
import statistics
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from krn import Blocks
from krn import itertools as kernel_itertools
from krn.compiler.api import CaptureError, CPUDevice, capture
from krn.frontend.types import f32, i32, u8, u32
from krn.runtime import host as host
from tests.helpers import tool_kernels as k

log = logging.getLogger("helpers")

ROW = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
OTHER = [2 * v for v in ROW]
KEYS = [0, 0, 1, 1, 1, 3, 3, 4, 4, 4]
FILL = 99


def flat(rows: Iterable[Iterable[int]]) -> list[int]:
    return list(it.chain.from_iterable(rows))


# The original of each element-moving helper, as the stdlib expression with its meaning (more-itertools and toolz are
# not dependencies of the examples).
MOVES: dict[str, tuple[str, Callable[[list[int]], list[int]]]] = {
    "pairwise": ("itertools.pairwise", lambda r: flat(it.pairwise(r))),
    "windowed": ("more_itertools.windowed, full windows", lambda r: flat(r[i : i + 3] for i in range(len(r) - 2))),
    "windowed_step": (
        "more_itertools.windowed(step=2)",
        lambda r: flat(r[i : i + 3] for i in range(0, len(r) - 2, 2)),
    ),
    "partition": ("toolz.partition", lambda r: flat(zip(*[iter(r)] * 2, strict=False))),
    "take": ("more_itertools.take", lambda r: list(it.islice(r, 3))),
    "tail": ("more_itertools.tail", lambda r: list(deque(r, maxlen=3))),
    "drop": ("toolz.drop", lambda r: list(it.islice(r, 3, None))),
    "take_nth": ("toolz.take_nth", lambda r: list(it.islice(r, 0, None, 3))),
    "intersperse": ("more_itertools.intersperse", lambda r: [v for x in r for v in (FILL, x)][1:]),
    "first": ("more_itertools.first", lambda r: [next(iter(r), FILL)]),
    "last": ("more_itertools.last", lambda r: [next(reversed(r), FILL)]),
    "nth": ("more_itertools.nth", lambda r: [next(it.islice(r, 2, None), FILL)]),
    "second": ("toolz.second", lambda r: [next(it.islice(r, 1, None), FILL)]),
}
NEEDS_FILL = {"intersperse", "first", "last", "nth", "second"}

REFUSED = """from collections.abc import Iterator
from krn import Vector, kernel
from krn.frontend.types import f32, u32
from .itertools import chunked, windowed

@kernel
def {name}(x: Vector[f32], n: u32, size: u32) -> None:
    for value in {call}:
        pass
"""

ABSOLUTE = """from krn import Vector, kernel
from krn.frontend.types import u32
from krn.itertools import chunked

@kernel
def run(n: u32, out: Vector[u32]) -> None:
    for start, valid in chunked(n, 2):
        out[start] = valid
"""


async def main() -> None:
    cpu = host.Device.current()
    assert isinstance(cpu, CPUDevice)
    launch = Blocks(1, threads=1)
    start, n = u32(0), u32(len(ROW))

    async def check(name: str, got: np.ndarray, expected: Sequence[float], original: str) -> None:
        await cpu.synchronize()
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=0, equal_nan=True, err_msg=name)
        log.info("%-14s agrees with %s", name, original)

    for label, dtype in (("f32", f32), ("i32", i32), ("u8", u8)):
        x, fill = cpu.empty(dtype, 10), cpu.empty(dtype, 1)
        x.array[:], fill.array[:] = ROW, FILL
        for name, (original, reference) in MOVES.items():
            expected = reference(ROW)
            out = cpu.empty(dtype, len(expected))
            helper = getattr(k, f"run_{name}")
            helper(*((x, fill) if name in NEEDS_FILL else (x,)), start, n, out, launch=launch)
            await check(f"{name}[{label}]", out.array, expected, original)

    x, y = cpu.empty(f32, 10), cpu.empty(f32, 10)
    key, wide, counts = cpu.empty(u8, 10), cpu.empty(u32, 10), cpu.empty(u32, 5)
    x.array[:], y.array[:], key.array[:], wide.array[:], counts.array[:] = ROW, OTHER, KEYS, KEYS, 0
    for name, original, expected, helper in (
        ("accumulate", "itertools.accumulate", list(it.accumulate(ROW)), k.run_accumulate),
        ("running_max", "itertools.accumulate(max)", list(it.accumulate(ROW, max)), k.run_running_max),
        ("running_min", "itertools.accumulate(min)", list(it.accumulate(ROW, min)), k.run_running_min),
        ("mean", "statistics.mean", [statistics.mean(ROW)], k.run_mean),
    ):
        out = cpu.empty(f32, len(expected))
        helper(x, start, n, out, launch=launch)
        await check(name, out.array, expected, original)
    for name, original, expected, helper in (
        ("chain", "itertools.chain", ROW + OTHER, k.run_chain),
        ("interleave", "more_itertools.interleave", flat(zip(ROW, OTHER, strict=True)), k.run_interleave),
    ):
        out = cpu.empty(f32, len(expected))
        helper(x, y, start, n, out, launch=launch)
        await check(name, out.array, expected, original)

    matrix = cpu.empty(f32, 2, 10)
    matrix.array[:] = [ROW, OTHER]
    out = cpu.empty(f32, 20)
    k.run_flatten(matrix, start, u32(2), out, launch=launch)
    await check("flatten", out.array, flat([ROW, OTHER]), "more_itertools.flatten")

    runs, lo = [], 0
    for value, group in it.groupby(KEYS):
        runs += (value, lo, lo := lo + len(list(group)))
    out = cpu.empty(u32, len(runs))
    k.run_groupby(key, start, n, out, launch=launch)
    await check("groupby[u8]", out.array, runs, "itertools.groupby")
    k.run_count_by(wide, start, n, counts, launch=launch)
    await check("count_by", counts.array, [Counter(KEYS)[i] for i in range(5)], "collections.Counter")

    chunks = list(it.batched(range(10), 2, strict=False))
    for name, expected, original in (
        ("chunked", [v for c in chunks for v in (c[0], len(c))], "itertools.batched, lengths"),
        ("batched", [v for c in chunks for v in (c[0], c[-1] + 1)], "itertools.batched, bounds"),
        ("ilen", [len(range(0, 10))], "more_itertools.ilen(range)"),
        ("ilen_step", [len(range(0, 10, 3))], "more_itertools.ilen(range)"),
        ("range_right", list(reversed(range(10))), "es-toolkit rangeRight"),
        ("range_right_step", list(reversed(range(0, 10, 5))), "es-toolkit rangeRight"),
    ):
        out = cpu.empty(u32, len(expected))
        arguments = (n,) if name in ("chunked", "batched") else (start, n)
        getattr(k, f"run_{name}")(*arguments, out, launch=launch)
        await check(name, out.array, expected, original)

    positions = cpu.empty(u32, 1)
    for name, expected in (
        ("min_by", min(range(10), key=ROW.__getitem__)),
        ("max_by", max(range(10), key=ROW.__getitem__)),
    ):
        getattr(k, f"run_{name}")(x, start, n, positions, launch=launch)
        await check(name, positions.array, [expected], f"{name.removesuffix('_by')}(range(n), key=row.__getitem__)")

    # The documented examples of es-toolkit.dev/reference/math: clamp, lerp, inverseLerp, inRange.
    low, high, third, out = (cpu.empty(f32, 5) for _ in range(4))
    low.array[:], high.array[:], third.array[:] = [0, 10, 100, 0, 5], [100, 20, 0, 100, 5], [0.5, 0.25, 0.25, 1.5, 0]
    k.run_lerp(low, high, third, u32(4), out, launch=launch)
    await check("lerp", out.array[:4], [50, 12.5, 75, 150], "es-toolkit lerp")
    third.array[:] = [50, 12.5, 75, 150, 5]
    k.run_inverse_lerp(low, high, third, u32(5), out, launch=launch)
    await check("inverse_lerp", out.array, [0.5, 0.25, 0.25, 1.5, 0], "es-toolkit inverseLerp")
    low.array[:4], high.array[:4], third.array[:4] = [2, 2, 2, -5], [5, 5, 5, -1], [3, 1, 5, -3]
    inside = cpu.empty(u32, 4)
    k.run_in_range(third, low, high, u32(4), inside, launch=launch)
    await check("in_range", inside.array, [1, 0, 0, 1], "es-toolkit inRange")
    samples = cpu.empty(f32, 3)
    samples.array[:] = [-2, 0.5, 4]
    for name in ("clip", "unit_clip"):
        getattr(k, f"run_{name}")(samples, u32(3), out, launch=launch)
        await check(
            name,
            out.array[:3],
            [min(max(v, 5 if name == "clip" else 0), 15 if name == "clip" else 1) for v in samples.array],
            "es-toolkit clamp",
        )

    indptr = cpu.empty(u32, 4)
    indptr.array[:] = [0, 2, 2, 10]
    rows = cpu.empty(u32, 3)
    k.run_min_by_rows(x, indptr, u32(3), rows, launch=launch)
    bounds = [0, 2, 2, 10]
    await check(
        "min_by rows",
        rows.array,
        [min(range(a, b), key=ROW.__getitem__, default=a) for a, b in it.pairwise(bounds)],
        "min(range(row), key=)",
    )

    with tempfile.TemporaryDirectory() as folder:
        assert kernel_itertools.__file__ is not None
        shutil.copy(Path(kernel_itertools.__file__), Path(folder) / "itertools.py")
        for index, (name, call) in enumerate(
            (("runtime chunk size", "chunked(n, size)"), ("runtime window width", "windowed(x, 0, n, size)"))
        ):
            (Path(folder) / f"refused{index}.py").write_text(REFUSED.format(name="run", call=call))  # parsed once each
            report(folder, f"refused{index}.py", name)
        (Path(folder) / "absolute.py").write_text(ABSOLUTE)
        capture(Path(folder) / "absolute.py", "run")


def report(folder: str, file: str, what: str) -> None:
    """Capture `file`'s kernel `run` and log why it is refused; a refusal that does not happen is a bug."""
    try:
        capture(Path(folder) / file, "run")
    except CaptureError as error:
        log.info("%s is refused: %s", what, str(error).splitlines()[0].split("error: ", 1)[-1].split(";")[0])
    else:
        raise AssertionError(f"{what} was not refused")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
