"""Python's own vocabulary inside a kernel: folds, enumerate, zip, register sequences and match

A kernel reads like the Python it is written in, and capture turns each construct into the loops and values the
hardware has. `max(abs(v) for v in row)` is a loop that keeps the larger, `enumerate` and `zip` become counters,
`reversed` and negative `range` steps walk down safely for unsigned bounds, a list or tuple is a fixed-length
sequence of registers, and `match x.dtype` captures only the arm that matches. What cannot be known at capture (a
runtime length, a set, a dict) is refused with the fix named. Everything runs on the CPU backend, and the target
match in the emulator, against numpy.

Level: 2
Topics: Python vocabulary, generators, register sequences, match
Needs: 01_first_kernels/01_saxpy
"""

import asyncio
import importlib.util
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np

from krn import Items, Matrix, T, Vector, f32, host, items, kernel, sum, target, targets, u32
from krn.compiler.api import CPUDevice, capture
from krn.emulate import emulate
from krn.errors import CaptureError

log = logging.getLogger("vocabulary")


@kernel
def row_stats(x: Matrix[f32], stats: Matrix[f32], votes: Matrix[u32]) -> None:
    """A row to a thread; each fold is a loop over the row that keeps what its name says."""
    for r in items(x.shape[0]):
        stats[r, 0] = max(abs(x[r, c]) for c in range(x.shape[1]))
        stats[r, 1] = min(x[r, c] for c in range(x.shape[1]))
        stats[r, 2] = sum(x[r, c] for c in range(x.shape[1]) if x[r, c] > 0.0)
        votes[r, 0] = u32(any(x[r, c] < 0.0 for c in range(x.shape[1])))
        votes[r, 1] = u32(all(x[r, c] < 3.0 for c in range(x.shape[1])))


@kernel
def neighbors(x: Vector[f32], i: u32, width: u32) -> Iterator[tuple[T.Idx, f32]]:
    """The (offset, value) pairs of `width` neighbors from `i`: a loop that iterates a generator inlines its body."""
    for k in range(width):
        yield k, x[i + k]


@kernel
def numbered(x: Vector[f32], count: u32, width: u32, out: Matrix[f32]) -> None:
    """`enumerate` counts beside any loop, here the generator above, and takes a start like Python's."""
    for i in items(count):
        for n, (k, v) in enumerate(neighbors(x, i, width), 1):
            out[i, k] = v * f32(n)


@kernel
def pair_products(x: Vector[f32], out: Vector[f32]) -> None:
    """Two strided ranges step together: x[i] x[i + 1] + x[i + 2] x[i + 3] + ..., four pairs from a runtime start."""
    for i in items(out.shape[0]):
        out[i] = sum(x[a] * x[b] for a, b in zip(range(i, i + 8, 2), range(i + 1, i + 9, 2), strict=False))


@kernel
def horner(coeff: Vector[f32], x: Vector[f32], up: Vector[f32], down: Vector[f32]) -> None:
    """A polynomial from its highest power down, written both ways: `reversed(range(n))` and a negative step."""
    for i in items(x.shape[0]):
        high = f32(0.0)
        low = f32(0.0)
        for c in reversed(range(coeff.shape[0])):
            high = high * x[i] + coeff[c]
        for c in range(coeff.shape[0] - 1, -1, -1):
            low = low * x[i] + coeff[c]
        up[i] = high
        down[i] = low


@kernel
def taps(x: Vector[f32], smooth: Vector[f32], spread: Vector[f32], scaled: Matrix[f32]) -> None:
    """Four neighbors in registers: `zip` with a tuple of weights, unpacking, a comprehension, an unrolled loop."""
    for i in items(smooth.shape[0]):
        around = [x[i], x[i + 1], x[i + 2], x[i + 3]]
        smooth[i] = sum(w * v for w, v in zip((0.1, 0.2, 0.3, 0.4), around, strict=False))
        first, *rest = around
        centered = [v - first for v in rest]
        spread[i] = max(centered) - min(centered)
        for k, v in enumerate(around):
            scaled[i, k] = v * f32(len(around) - k)


@kernel
def width[F: T.FloatingPoint](x: Vector[F], bits: Vector[u32]) -> None:
    """The width of the element type in bits, from the type alone: only the matching arm is captured."""
    for i in items(bits.shape[0]):
        match x.dtype:
            case T.f16 | T.bf16:
                bits[i] = 16
            case T.f32:
                bits[i] = 32
            case _:
                bits[i] = 64


@kernel
def lanes(out: Vector[u32]) -> None:
    """The family of the target is a fact of the capture: 1 on a CPU, 2 on NVIDIA and AMD, 3 elsewhere."""
    for i in items(out.shape[0]):
        match target:
            case targets.CPU():
                out[i] = 1
            case targets.Nvidia() | targets.AMD():
                out[i] = 2
            case _:
                out[i] = 3


# Kept as source text for the capture and runtime demonstrations below.
FLATTEN = """
from itertools import chain

from krn import Matrix, Vector, items, kernel
from krn.frontend.types import f32


@kernel
def interleave(x: Vector[f32], y: Vector[f32], flat: Matrix[f32], padded: Matrix[f32]) -> None:
    for i in items(flat.shape[0]):
        rows = [[x[i], x[i + 1]], [y[i], y[i + 1]]]
        for k, v in enumerate(chain.from_iterable(rows)):
            flat[i, k] = v
        for k, v in enumerate(chain.from_iterable(row + [1.0] for row in rows)):
            padded[i, k] = v
"""

REFUSED_KERNEL = """
from krn import Vector, kernel
from krn.frontend.types import f32


@kernel
def refused(x: Vector[f32], n: int) -> None:
    {body}
"""

REFUSED = {
    "a list of a run-time length": "xs = [x[i] for i in range(n)]",
    "a set": "xs = {x[0], x[1]}",
    "a dict": "xs = {0: x[0]}",
    "a run-time match subject": "match x[0]:\n        case 1:\n            pass",
    "an index that is not a constant": "xs = [x[0], x[1]]\n    v = xs[n]",
    "an append": "xs = [x[0]]\n    xs.append(x[1])",
    "an empty max": "v = max(x[i] for i in range(0))",
}


def load(path: Path, source: str) -> ModuleType:
    """Import kernels from source text through a file of their own, since a kernel is captured from its file."""
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def main() -> None:
    cpu = host.Device.current()
    assert isinstance(cpu, CPUDevice)
    rng = np.random.default_rng(61)

    def check(name: str, got: np.ndarray, expected: np.ndarray) -> None:
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6, err_msg=name)
        log.info("%-14s agrees with numpy", name)

    rows, columns = 40, 9
    x = cpu.empty(f32, rows, columns)
    x.array[:] = rng.standard_normal((rows, columns)) * 2
    stats, votes = cpu.empty(f32, rows, 3), cpu.empty(u32, rows, 2)
    row_stats(x, stats, votes)
    await cpu.synchronize()
    a = x.array
    check("max, min, sum", stats.array, np.stack([np.abs(a).max(1), a.min(1), np.where(a > 0, a, 0).sum(1)], 1))
    np.testing.assert_array_equal(votes.array, np.stack([(a < 0).any(1), (a < 3.0).all(1)], 1))
    log.info("%-14s agrees with numpy", "any, all")

    v = cpu.empty(f32, 64)
    v.array[:] = rng.standard_normal(64)
    out = cpu.empty(f32, 20, 5)
    numbered(v, u32(20), u32(5), out)
    await cpu.synchronize()
    check("enumerate", out.array, np.stack([v.array[i : i + 5] * np.arange(1, 6) for i in range(20)]))

    out = cpu.empty(f32, 20)
    pair_products(v, out)
    await cpu.synchronize()
    check(
        "zip of ranges",
        out.array,
        np.array([(v.array[i : i + 8 : 2] * v.array[i + 1 : i + 9 : 2]).sum() for i in range(20)]),
    )

    coeff, points = cpu.empty(f32, 5), cpu.empty(f32, 30)
    coeff.array[:] = rng.standard_normal(5)
    points.array[:] = rng.uniform(-1, 1, 30)
    up, down = cpu.empty(f32, 30), cpu.empty(f32, 30)
    horner(coeff, points, up, down)
    await cpu.synchronize()
    want = np.polynomial.polynomial.polyval(points.array.astype(np.float64), coeff.array.astype(np.float64))
    check("reversed", up.array, want)
    check("negative step", down.array, want)

    smooth, spread, scaled = cpu.empty(f32, 30), cpu.empty(f32, 30), cpu.empty(f32, 30, 4)
    taps(v, smooth, spread, scaled)
    await cpu.synchronize()
    windows = np.stack([v.array[i : i + 4] for i in range(30)])
    check("smooth", smooth.array, windows @ np.array([0.1, 0.2, 0.3, 0.4], np.float32))
    check("spread", spread.array, (windows[:, 1:] - windows[:, :1]).max(1) - (windows[:, 1:] - windows[:, :1]).min(1))
    check("scaled", scaled.array, windows * np.array([4, 3, 2, 1], np.float32))

    for name, dtype in (("f16", T.f16), ("f32", T.f32), ("f64", T.f64)):
        sample, bits = cpu.empty(dtype, 1), cpu.empty(u32, 3)
        width(sample, bits)
        await cpu.synchronize()
        assert bits.array.tolist() == [sample.array.itemsize * 8] * 3
        log.info("%-14s the match gave %d bits", f"width {name}", bits.array[0])

    here, there = cpu.empty(u32, 4), cpu.empty(u32, 4)
    lanes(here)
    emulate(lanes, Items(4), there)
    await cpu.synchronize()
    assert here.array.tolist() == [1] * 4 and there.array.tolist() == [2] * 4
    log.info("lanes: the C backend (a CPU target) took arm 1, the emulator (sm_90a) arm 2")

    folder = Path(tempfile.mkdtemp())
    interleave = load(folder / "flatten.py", FLATTEN).interleave
    y = cpu.empty(f32, 40)
    y.array[:] = rng.standard_normal(40)
    flat, padded = cpu.empty(f32, 20, 4), cpu.empty(f32, 20, 6)
    interleave(v, y, flat, padded, launch=Items(20))
    await cpu.synchronize()
    pairs = [np.stack([s.array[i : i + 2] for i in range(20)]) for s in (v, y)]
    check("PEP 798 flat", flat.array, np.concatenate(pairs, 1))
    check("PEP 798 padded", padded.array, np.concatenate([pairs[0], np.ones((20, 1)), pairs[1], np.ones((20, 1))], 1))

    for number, (what, body) in enumerate(REFUSED.items()):
        path = folder / f"refused{number}.py"  # a file is parsed once, so each refusal gets its own
        path.write_text(REFUSED_KERNEL.format(body=body))
        try:
            capture(path, "refused")
        except CaptureError as error:
            log.info("%s is refused: %s", what, str(error).splitlines()[0].split("error: ", 1)[-1])
        else:
            raise AssertionError(f"{what} was not refused")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
