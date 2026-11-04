"""Our sparse products against the vendors' libraries, on a matrix set whose rule is written down before any run.

The intended set is the SpMV suite of Williams et al. (SC 2007): the whole SuiteSparse group `Williams`, 14 matrices,
none left out. `WILLIAMS` and `fetch` are here for it. On 2026-11-01 the file host of the collection answered 403 to
every request from both of our machines, and its site says "the collection is offline because of 100x higher downloads
than normal... there is no known mirror". So the run below is on a **substitute that is not that suite**, fixed by this
rule before the first run, whole and without picking:

- the finite-difference Laplacians of the structured family of Bell and Garland (SC 2009): 3-point on a line of 10^6
  unknowns, 5-point and 9-point on a 1000 x 1000 grid, 7-point and 27-point on a 100 x 100 x 100 grid. The family is
  theirs and the sizes are ours;
- uniform random rows of 4, 32 and 256 entries on 10^5 rows and 10^5 columns (seed 0; a position may repeat);
- power-law rows: 2 * 10^5 rows whose lengths are Zipf(1.6) capped at 50,000, columns uniform (seed 0), which is the
  skew of a web graph;
- one dense 2000 x 2000 matrix stored as CSR, the longest rows there are.

The rest of the rule holds for either set. Values are float32, indices 32 bit, the format CSR, and `x` is standard
normal with seed 0 (a dense block of `columns` of them for a matrix product).

- Every eligible implementation is reported: our static choice, thread and warp SpMV variants, cuSPARSE's default
  and each applicable algorithm, and Accelerate's only one. A separate `lib` row eagerly tunes on these operands in
  an empty cache, reports its winner against the static choice, and includes dispatch cost. The plan, descriptors and
  upload are made once and are not in the time. Accelerate's copy to column-major order is in its time.
- A time is `repeats` products queued back to back on resident data, chosen so a trial lasts about 50 ms, and the figure
  is the median of 5 trials. On a card the host's cost of a launch is in it, equally for both sides. GFLOP/s is
  2 * entries * columns / seconds. The error is the largest absolute difference from a float64 reference divided by the
  largest magnitude of the reference, and a product that exceeds 1e-3 is reported as wrong, never as fast. A product
  shorter than 30 us is marked launch-bound: the loop makes one driver call per product for both sides, and that call
  is a few microseconds, so no ranking is read from it. The resident size is the matrix plus the two vectors, to be
  compared with the L2 cache of the card, because a product that fits in it is not limited by memory.
- Every number is written down, the slow ones too.

Run on a Mac: `mb run python -m experiments.sparse_bench cpu [--columns 16]`.
Run on silo:  `mb run --on silo python -m experiments.sparse_bench cuda [--columns 16]`.
"""

import ctypes
import itertools
import json
import logging
import platform
import statistics
import tarfile
import time
import urllib.request
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

import cyclopts
import numpy as np
from numpy.typing import NDArray

from experiments.paths import RESULTS
from krn import T, configure
from krn.compiler.api import CPUDevice
from krn.data.sparse import Csr
from krn.data.sparse_build import CsrArrays, csr_from_triples, csr_on
from krn.frontend.memory import Tensor
from krn.lib import lib as lib
from krn.lib import lib_nvidia as lib_nvidia
from krn.lib.lib import LibOptions, Op, Problem, SpmmOperands, SpmvOperands
from krn.lib.lib_accelerate import Accelerate, AccelerateSpmm, AccelerateSpmv
from krn.lib.lib_cusparse import CusparseSpmv
from krn.lib.lib_ours import OursSpmm, OursSpmv
from krn.runtime import host as runtime_host
from krn.runtime.settings import current_settings

try:
    from krn.runtime import cuda_backend as cuda_backend
except ImportError:
    cuda_backend = None

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDAStream, CUDATensor

log = logging.getLogger("sparse_bench")
app = cyclopts.App()
WILLIAMS = (
    "cant", "consph", "cop20k_A", "dense2", "mac_econ_fwd500", "mc2depi", "pdb1HYS",
    "pwtk", "qcd5_4", "rail4284", "rma10", "scircuit", "shipsec1", "webbase-1M",
)  # fmt: skip
SUITE = (
    "laplace3", "laplace5", "laplace9", "laplace7", "laplace27",
    "uniform4", "uniform32", "uniform256", "powerlaw", "dense2000",
)  # fmt: skip
URL = "https://sparse-files.engr.tamu.edu/MM/Williams/{name}.tar.gz"
CACHE = (current_settings().cache.dir or Path.home() / ".cache" / "krn") / "suitesparse"
TRIAL_SECONDS, TRIALS, WRONG = 0.05, 5, 1e-3
LAUNCH_BOUND = 30e-6  # a product shorter than this is mostly the host's launch call, whoever's it is
type Json = str | int | float | bool | list[Json] | dict[str, Json] | None


class Entries(NamedTuple):
    rows: int
    columns: int
    row: np.ndarray
    column: np.ndarray
    value: np.ndarray


class Measure(NamedTuple):
    seconds: float
    error: float


def fetch(name: str) -> Path:
    """The Matrix Market file of a matrix, downloaded and unpacked once into the cache."""
    path = CACHE / name / f"{name}.mtx"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        archive = CACHE / f"{name}.tar.gz"
        urllib.request.urlretrieve(URL.format(name=name), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(CACHE, filter="data")
        archive.unlink()
    return path


def read_matrix_market(path: Path) -> Entries:
    """A coordinate file as triples, both triangles of a symmetric one, a pattern as ones."""
    with path.open("rb") as handle:
        banner = handle.readline().decode().lower().split()
        line = handle.readline()
        while line.startswith(b"%"):
            line = handle.readline()
        rows, columns, count = (int(token) for token in line.split())
        numbers = np.fromstring(handle.read().decode("ascii"), sep=" ")
    field, symmetry = banner[3], banner[4]
    table = numbers.reshape(count, 2 if field == "pattern" else 3)
    row, column = table[:, 0].astype(np.int64) - 1, table[:, 1].astype(np.int64) - 1
    value = np.ones(count) if field == "pattern" else table[:, 2]
    if symmetry == "symmetric":
        below = row != column
        row, column, value = (
            np.concatenate([row, column[below]]),
            np.concatenate([column, row[below]]),
            np.concatenate([value, value[below]]),
        )
    elif symmetry != "general":
        raise ValueError(f"{path.name}: {symmetry} matrices are not in this suite")
    return Entries(rows, columns, row, column, value)


def reference(entries: Entries, x: np.ndarray) -> np.ndarray:
    """matrix @ x in float64, for a vector or a matrix `x`."""
    columns = [x] if x.ndim == 1 else list(x.T)
    products = [
        np.bincount(entries.row, weights=entries.value * c[entries.column], minlength=entries.rows) for c in columns
    ]
    return products[0] if x.ndim == 1 else np.stack(products, axis=1)


def error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got.astype(np.float64) - want)) / max(float(np.max(np.abs(want))), 1e-30))


def median_seconds(timed: Callable[[int], float]) -> float:
    """`timed(repeats)` is the seconds per product over that many; pick repeats for a trial of about 50 ms."""
    probe = timed(2)
    repeats = max(2, int(TRIAL_SECONDS / max(probe, 1e-9)))
    return statistics.median(timed(repeats) for _ in range(TRIALS))


def queued(step: Callable[[], None], settle: Callable[[], None]) -> Callable[[int], float]:
    """`repeats` calls of `step` back to back, then `settle`: the seconds per call."""

    def timed(repeats: int) -> float:
        started = time.perf_counter()
        for _ in range(repeats):
            step()
        settle()
        return (time.perf_counter() - started) / repeats

    return timed


def results(name: str, a: CsrArrays, columns: int, measures: dict[str, Measure]) -> dict[str, Json]:
    entries = len(a.values)
    flops = 2 * entries * max(columns, 1)
    resident_mb = (entries * 8 + (a.rows + 1) * 4 + 4 * (a.columns + a.rows) * max(columns, 1)) / 1e6
    log.info("%-16s %9d x %-9d %9d entries   %6.1f MB resident", name, a.rows, a.columns, entries, resident_mb)
    for label, measure in measures.items():
        verdict = "WRONG" if measure.error > WRONG else "launch-bound" if measure.seconds < LAUNCH_BOUND else ""
        log.info(
            "    %-14s %10.3f ms %8.2f GFLOP/s   error %.1e %s",
            label, measure.seconds * 1e3, flops / measure.seconds / 1e9, measure.error, verdict,
        )  # fmt: skip
    return {
        "matrix": name,
        "rows": a.rows,
        "columns": a.columns,
        "entries": entries,
        "mean_row": entries / a.rows,
        "resident_mb": resident_mb,
        "measures": {label: m._asdict() | {"gflops": flops / m.seconds / 1e9} for label, m in measures.items()},
    }


def stencil(shape: tuple[int, ...], offsets: list[tuple[int, ...]]) -> Entries:
    """The Laplacian of a regular grid: `len(offsets)` on the diagonal and -1 for each neighbor that exists."""
    points = np.indices(shape).reshape(len(shape), -1)
    index = np.arange(points.shape[1])
    rows, columns, values = [index], [index], [np.full(len(index), float(len(offsets)))]
    for offset in offsets:
        neighbor = points + np.array(offset)[:, None]
        inside = np.all((neighbor >= 0) & (neighbor < np.array(shape)[:, None]), axis=0)
        rows.append(index[inside])
        columns.append(np.ravel_multi_index(tuple(neighbor[:, inside]), shape))
        values.append(np.full(int(inside.sum()), -1.0))
    n = int(np.prod(shape))
    return Entries(n, n, np.concatenate(rows), np.concatenate(columns), np.concatenate(values))


def box(dimensions: int) -> list[tuple[int, ...]]:
    """Every neighbor of a grid point, the diagonal ones too."""
    return [o for o in itertools.product((-1, 0, 1), repeat=dimensions) if any(o)]


def axes(dimensions: int) -> list[tuple[int, ...]]:
    return [tuple(s * (i == axis) for i in range(dimensions)) for axis in range(dimensions) for s in (-1, 1)]


def random_rows(rows: int, columns: int, lengths: np.ndarray, rng: np.random.Generator) -> Entries:
    row = np.repeat(np.arange(rows), lengths)
    return Entries(rows, columns, row, rng.integers(0, columns, len(row)), rng.standard_normal(len(row)))


def synthetic(name: str) -> Entries:
    rng = np.random.default_rng(0)
    match name:
        case "laplace3":
            return stencil((1_000_000,), axes(1))
        case "laplace5":
            return stencil((1000, 1000), axes(2))
        case "laplace9":
            return stencil((1000, 1000), box(2))
        case "laplace7":
            return stencil((100, 100, 100), axes(3))
        case "laplace27":
            return stencil((100, 100, 100), box(3))
        case "uniform4" | "uniform32" | "uniform256":
            return random_rows(100_000, 100_000, np.full(100_000, int(name.removeprefix("uniform"))), rng)
        case "powerlaw":
            return random_rows(200_000, 200_000, np.minimum(rng.zipf(1.6, 200_000), 50_000), rng)
        case "dense2000":
            return random_rows(2000, 2000, np.full(2000, 2000), rng)._replace(column=np.tile(np.arange(2000), 2000))
    raise ValueError(f"{name} is neither in the synthetic set nor in the Williams group")


def load(name: str) -> tuple[Entries, CsrArrays]:
    entries = read_matrix_market(fetch(name)) if name in WILLIAMS else synthetic(name)
    return entries, csr_from_triples(entries.rows, entries.columns, entries.row, entries.column, entries.value)


def operand(rng: np.random.Generator, size: int, columns: int) -> np.ndarray:
    return rng.standard_normal((size, columns) if columns else (size,)).astype(np.float32)


def save(out: Path | None, machine: dict[str, Json], rows: list[dict[str, Json]], columns: int) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"machine": machine, "columns": columns, "rows": rows}, indent=1))


def cpu_steps(
    device: CPUDevice, a: CsrArrays, x: np.ndarray, accelerate: bool
) -> tuple[dict[str, Callable[[], None]], NDArray[np.float32]]:
    """The products to time on the CPU backend, by candidate, and the array that holds their result."""
    matrix, stream = csr_on(device, a), device.default_stream
    options, shape = LibOptions.of(device.target), (a.rows, a.columns, len(a.values))
    steps: dict[str, Callable[[], None]] = {}
    if x.ndim == 2:
        tx, result = device.empty(T.f32, *x.shape), device.empty(T.f32, a.rows, x.shape[1])
        tx.array[...] = x
        problem = Problem(Op.SPMM, (T.f32,) * 3, (*shape, x.shape[1]), options, layout="Csr")
        spmm = SpmmOperands(matrix, tx, result)
        steps["ours"] = lambda: OursSpmm().run(problem, spmm, stream)
        if accelerate:
            steps["accelerate"] = lambda: AccelerateSpmm().run(problem, spmm, stream)
    else:
        vx, vresult = device.empty(T.f32, len(x)), device.empty(T.f32, a.rows)
        vx.array[...] = x
        problem = Problem(Op.SPMV, (T.f32,) * 3, shape, options, layout="Csr")
        spmv = SpmvOperands(matrix, vx, vresult)
        steps["ours"] = lambda: OursSpmv().run(problem, spmv, stream)
        if accelerate:
            steps["accelerate"] = lambda: AccelerateSpmv().run(problem, spmv, stream)
        result = vresult
    return steps, np.asarray(result, dtype=np.float32)


@app.command
def cpu(columns: int = 0, out: Path | None = None, matrices: tuple[str, ...] = SUITE) -> None:
    """Our kernels on the CPU backend (one thread per launch) against Accelerate, where the framework loads."""
    device = runtime_host.Device.current()
    assert isinstance(device, CPUDevice)
    try:
        ctypes.CDLL(Accelerate.path)
        accelerate = True
    except OSError:
        accelerate = False
        log.info("no Accelerate here")
    rng, rows = np.random.default_rng(0), []
    for name in matrices:
        entries, a = load(name)
        x = operand(rng, a.columns, columns)
        steps, result = cpu_steps(device, a, x, accelerate)
        want, measures = reference(entries, x), {}

        def settle() -> None:
            device.default_stream.record().wait(600)

        for label, step in steps.items():
            step()
            settle()
            measures[label] = Measure(median_seconds(queued(step, settle)), error(result, want))
        rows.append(results(name, a, columns, measures))
    save(out, {"platform": platform.platform(), "machine": platform.machine(), "ours_threads": 1}, rows, columns)


def l2_megabytes(device: CUDADevice) -> float:
    """The L2 cache of the card, which a resident matrix is to be compared with."""
    api = lib_nvidia.driver()
    status, size = api.cuDeviceGetAttribute(api.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device.ordinal)
    if status:
        raise RuntimeError(f"cuDeviceGetAttribute returned {status}")
    return size / 1e6


def resident(device: CUDADevice, a: CsrArrays) -> Csr:
    """The matrix on the card."""
    return Csr(
        values=device.to_device(a.values),
        rows=T.u32(a.rows),
        columns=T.u32(a.columns),
        indptr=device.to_device(a.indptr),
        indices=device.to_device(a.indices),
    )


class CUDABench[O: lib.OperationOperands]:
    """One resident product, measured directly and through dispatch on the same stream."""

    def __init__(
        self,
        kind: type[lib.Candidate[O]],
        problem: Problem,
        operands: O,
        stream: CUDAStream,
        result: CUDATensor,
        want: np.ndarray,
        step: Callable[[], Tensor | None],
    ) -> None:
        self.kind, self.problem, self.operands = kind, problem, operands
        self.stream, self.result, self.want, self.step = stream, result, want, step

    def _time(self, step: Callable[[], Tensor | None], repeats: int) -> float:
        begin = self.stream.record()
        for _ in range(repeats):
            step()
        end = self.stream.record()
        end.wait(600)
        return end.elapsed_since(begin) / 1e3 / repeats  # the device clock reads milliseconds

    def measure(self) -> tuple[dict[str, Measure], dict[str, Json]]:
        with TemporaryDirectory(prefix="krn-sparse-") as cache, configure(**{"lib.tune": "eager", "cache.dir": cache}):
            lib._dispatcher.cache_clear()
            try:
                return self._measure()
            finally:
                for thread in lib.dispatcher().measuring:
                    thread.join(600)
                    if thread.is_alive():
                        raise TimeoutError("sparse library tuner did not finish in 600 seconds")
                lib._dispatcher.cache_clear()

    def _measure(self) -> tuple[dict[str, Measure], dict[str, Json]]:
        dispatch, measures = lib.dispatcher(), {}
        dispatch.tune = "off"
        for candidate in dispatch.candidates[self.problem.op]:
            if not isinstance(candidate, self.kind):
                continue
            reason = candidate.unavailable(dispatch.target) or candidate.refusal(self.problem, dispatch.target)
            if reason:
                log.info("    %s: %s", candidate.name, reason)
                continue
            step = partial(candidate.run, self.problem, self.operands, self.stream)
            try:
                step()
            except lib.Declined as refusal:
                log.info("    %s: %s", candidate.name, refusal)
                dispatch.declined.setdefault((self.kind, self.problem), {})[candidate.name] = str(refusal)
                dispatch.choices.pop((self.kind, self.problem), None)
                continue
            self.stream.record().wait(600)
            seconds = median_seconds(partial(self._time, step))
            measures[candidate.name] = Measure(seconds, error(self.result.to_host(self.stream), self.want))
        static = dispatch.choose(self.kind, self.problem, capturing=False)
        dispatch.tune = "eager"
        self.step()
        for thread in dispatch.measuring:
            thread.join(600)
            if thread.is_alive():
                raise TimeoutError("sparse library tuner did not finish in 600 seconds")
        choice = dispatch.choices[self.kind, self.problem]
        chosen = dispatch.tuning.winners.get(choice.key, choice.pool[0].name)
        dispatch.tune = "off"  # a missing winner must not start another tuner inside the timing loop
        seconds = median_seconds(partial(self._time, self.step))
        measures["lib"] = Measure(seconds, error(self.result.to_host(self.stream), self.want))
        static_seconds, chosen_seconds = measures[static.name].seconds, measures[chosen].seconds
        ratio = chosen_seconds / static_seconds
        log.info(
            "    lib chose %s: %.3f ms; static %s: %.3f ms; ratio %.3f %s; through lib.* %.3f ms",
            chosen,
            chosen_seconds * 1e3,
            static.name,
            static_seconds * 1e3,
            ratio,
            "SLOWER >10%" if ratio > 1.1 else "",
            seconds * 1e3,
        )
        return measures, {
            "chosen": chosen,
            "static": static.name,
            "chosen_seconds": chosen_seconds,
            "static_seconds": static_seconds,
            "ratio": ratio,
            "slower": ratio > 1.1,
            "lib_seconds": seconds,
            "dispatch_seconds": seconds - chosen_seconds,
        }


@app.command
def cuda(
    columns: int = 0, out: Path | None = None, matrices: tuple[str, ...] = SUITE, host: str = platform.node()
) -> None:
    """All eligible resident candidates and the eagerly tuned `lib` call, including dispatch cost."""
    if cuda_backend is None:
        raise RuntimeError("the CUDA backend dependencies are unavailable")
    device = runtime_host.Device.current()
    assert isinstance(device, cuda_backend.CUDADevice)
    stream = device.default_stream
    rng, rows = np.random.default_rng(0), []
    for name in matrices:
        entries, a = load(name)
        x = operand(rng, a.columns, columns)
        matrix = resident(device, a)
        tx = device.to_device(x)
        options, shape = LibOptions.of(device.target), (a.rows, a.columns, len(a.values))
        if columns:
            result = device.empty(T.f32, a.rows, columns)
            problem = Problem(Op.SPMM, (T.f32,) * 3, (*shape, columns), options, layout="Csr")
            bench = CUDABench(
                lib.SpmmCandidate,
                problem,
                SpmmOperands(matrix, tx, result),
                stream,
                result,
                reference(entries, x),
                partial(lib.spmm, matrix, tx, out=result),
            )
        else:
            result = device.empty(T.f32, a.rows)
            problem = Problem(Op.SPMV, (T.f32,) * 3, shape, options, layout="Csr")
            bench = CUDABench(
                lib.SpmvCandidate,
                problem,
                SpmvOperands(matrix, tx, result),
                stream,
                result,
                reference(entries, x),
                partial(lib.spmv, matrix, tx, out=result),
            )
        measures, summary = bench.measure()
        row = results(name, a, columns, measures)
        row["lib"] = summary
        rows.append(row)
    capability = device.core.compute_capability
    machine: dict[str, Json] = {
        "platform": platform.platform(),
        "gpu": device.name,
        "capability": [capability.major, capability.minor],
        "l2_mb": l2_megabytes(device),
    }
    operation = f"spmm{columns}" if columns else "spmv"
    save(out or RESULTS / f"sparse_lib_{operation}_{host}.json", machine, rows, columns)


@app.command
def trace(matrix: str = "uniform32", calls: int = 20) -> None:
    """`calls` SpMV products by cuSPARSE's default algorithm and nothing else, to run under `nsys profile`."""
    if cuda_backend is None:
        raise RuntimeError("the CUDA backend dependencies are unavailable")
    device = runtime_host.Device.current()
    assert isinstance(device, cuda_backend.CUDADevice)
    _, a = load(matrix)
    csr, x = resident(device, a), device.to_device(operand(np.random.default_rng(0), a.columns, 0))
    result = device.empty(T.f32, a.rows)
    shape, options = (a.rows, a.columns, len(a.values)), LibOptions.of(device.target)
    problem = Problem(Op.SPMV, (T.f32,) * 3, shape, options, layout="Csr")
    product = partial(CusparseSpmv().run, problem, SpmvOperands(csr, x, result), device.default_stream)
    product()  # the plan, its buffer and the preprocessing are not what is traced
    device.default_stream.record().wait(60)
    for _ in range(calls):
        product()
    device.default_stream.record().wait(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
