"""Measure backend decisions on resident operands, then apply a rule fixed before collecting results.

For each decision and each backend family (cuda: silo, crimson and the GH200; c: every host's C backend;
metal: the Mac), and for every kernel of the set on every host of the family, take ratio = median time of
the value / the smallest median time among the values of that decision on that kernel and host. The score
of a value is the geometric mean of its ratios over all those cells of the family. The default of the
family is the value with the lowest score. It replaces the current default only when its score is at
least 3 percent lower than the current default's score, so noise cannot flip a default. A value whose
result differs from the numpy reference beyond the stated tolerance on any cell is excluded. For restrict
the question is whether `always` beats `never` by that margin on any family; if it does not anywhere, the
second compiled variant costs compile time for nothing and the recommendation is to default to `never`;
if it does, the default stays
`auto` (the overlap rule), because `auto` only ever chooses `always` when it is safe.

A cell is the fastest of three fresh devices with fresh operands, each measured as the median of seven trials after a
0.4 s burn-in, because where a worker thread or an allocation lands moves one kernel's time by 30 percent on a card
and by a factor of 2.4 on a hybrid-core CPU (silo: the same program ran 1941 and 4601 us) and that must not decide
a default.
Metal is omitted: MetalProgram does not accept Decisions, and this benchmark uses the launch path under test.
The C overlap measurement reports unbound host launch cost with and without the extent check, plus its difference.
Tolerance is max absolute error / max absolute float64 reference <= 1e-3. Invalid or rejected cells are retained.
"""

import json
import logging
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING
from unittest.mock import patch

import cyclopts
import numpy as np

from experiments.conformance import facts
from experiments.paths import RESULTS
from krn import Const, Kernel, Matrix, Tensor, Vector, block, items, kernel, struct
from krn.compiler.heuristics import Explanation
from krn.data.sparse import Csr, spmv
from krn.data.sparse_build import CsrArrays
from krn.facts.targets import CPU, Nvidia, Target
from krn.frontend.types import f32, u32
from krn.hw import system as system
from krn.lib.lib_kernels import MatmulTiling, matmul_epilogue
from krn.runtime.runtime import Argument, CPUDevice, CPUTensor, SourceLauncher
from krn.runtime.settings import Run, Settings

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

try:
    from krn.runtime import cuda_backend as cuda_backend
except ImportError:
    cuda_backend = None

try:
    from krn.runtime import metal as metal
except ImportError:
    metal = None

LOG = logging.getLogger("heuristics_bench")
APP = cyclopts.App()

TOLERANCE = 1e-3
OPTIONS = {"compiler": ("default",)}
# the defaults the first run was measured against; it chose restrict never and unroll 2 (fe3-status.md section 4)
CURRENT = {"compiler": "default"}


@kernel
def user_helper(value: f32) -> f32:
    return value * 0.5 + 0.25


@kernel
def element(x: Vector[f32], out: Vector[f32]) -> None:
    for i in items(x.shape[0]):
        out[i] = max(user_helper(x[i]), 0.0)


@kernel
def reduction(t: ReductionTiling, x: Matrix[f32], out: Tensor[f32, *tuple[int, ...]]) -> None:
    rows, columns = block.tile(1, t.width)
    out[rows] = x[rows, columns].sum(axis=1)


@struct
class ReductionTiling:
    width: Const[int] = 1024


@kernel
def three(x: Vector[f32], y: Vector[f32], out: Vector[f32]) -> None:
    for i in items(x.shape[0]):
        out[i] = x[i] + y[i]


@dataclass(frozen=True)
class Case[**P, R, S, **Q]:
    name: str
    entry: Kernel[P, R, S, Q]
    arrays: tuple[np.ndarray | CsrArrays | ReductionTiling | MatmulTiling, ...]
    reference: np.ndarray


@dataclass
class Row:
    group: str
    series: str
    value: float
    note: str
    decision: str
    setting: str
    family: str
    error: float | None
    trials: list[float]


def cases(family: str, small: bool = False) -> list[Case]:
    rng = np.random.default_rng(0)
    gpu = family == "cuda"
    n = 257 if small else 1 << (25 if gpu else 22)
    x = rng.standard_normal(n, dtype=np.float32)
    value = x.astype(np.float64) * 0.5 + 0.25
    result: list[Case] = [Case("element", element, (x, np.empty_like(x)), np.maximum(value, 0.0))]
    rows, width = (8, 32) if small else (8192, 2048) if gpu else (2048, 1024)
    matrix = rng.standard_normal((rows, width), dtype=np.float32)
    result.append(
        Case(
            "reduction",
            reduction,
            (ReductionTiling(width), matrix, np.empty(rows, np.float32)),
            matrix.sum(1, dtype=np.float64),
        )
    )
    n = 16 if small else 2048 if gpu else 256
    dtype = np.float16 if gpu else np.float32
    a, b = (rng.standard_normal((n, n)).astype(dtype) for _ in range(2))
    result.append(
        Case(
            "tile",
            matmul_epilogue,
            (MatmulTiling(bk=16), a, b, np.empty(n, dtype), np.empty((n, n), dtype)),
            a.astype(np.float64) @ b.astype(np.float64),
        )
    )
    n = 19 if small else 200000 if gpu else 8192
    rng = np.random.default_rng(0)
    indices = rng.integers(0, n, (n, 32), dtype=np.uint32)
    values = rng.standard_normal((n, 32), dtype=np.float32)
    x = rng.standard_normal(n, dtype=np.float32)
    csr = CsrArrays(values.ravel(), n, n, np.arange(n + 1, dtype=np.uint32) * 32, indices.ravel())
    reference = (values.astype(np.float64) * x[indices].astype(np.float64)).sum(1)
    result.append(Case("spmv", spmv, (csr, x, np.empty_like(x)), reference))
    return result


def setting(decision: str, value: str) -> Settings:
    """Collect the compiler rules as one baseline; historical knob sweeps are no longer supported."""
    if (decision, value) != ("compiler", "default"):
        raise ValueError(f"{decision}={value}: compiler decisions have no user override")
    return Settings(run=Run(workers=1))


def resident(device: CPUDevice | CUDADevice, a: CsrArrays) -> Csr:
    """The matrix on the device. `csr_on` writes through numpy and so serves only the CPU device."""
    return Csr(
        values=device.to_device(a.values),
        rows=u32(a.rows),
        columns=u32(a.columns),
        indptr=device.to_device(a.indptr),
        indices=device.to_device(a.indices),
    )


def upload(
    device: CPUDevice | CUDADevice, arrays: Sequence[np.ndarray | CsrArrays | ReductionTiling | MatmulTiling]
) -> tuple[Argument, ...]:
    return tuple(
        resident(device, a)
        if isinstance(a, CsrArrays)
        else a
        if isinstance(a, ReductionTiling | MatmulTiling)
        else device.to_device(a)
        for a in arrays
    )


def download(value: Argument) -> np.ndarray:
    if isinstance(value, CPUTensor):
        return value.array
    if cuda_backend is not None and isinstance(value, cuda_backend.CUDATensor):
        return value.to_host()
    raise TypeError(f"expected an output tensor, got {type(value).__name__}")


def trials(device: CPUDevice | CUDADevice, launch: Callable[[], None], repeats: int) -> list[float]:
    stream = device.launch_stream()
    samples = []
    for _ in range(7):
        start = stream.record()
        for _ in range(repeats):
            launch()
        end = stream.record()
        end.wait(60)
        samples.append(end.elapsed_since(start) * 1000 / repeats)
    return samples


def burn_in(device: CPUDevice | CUDADevice, launch: Callable[[], None], seconds: float = 0.4) -> None:
    """Keep the device busy so its clocks are up: a card that idled while the previous variant compiled runs the first
    trials slow (measured: a 1.3 times outlier in the first configuration of a group on silo)."""
    stream, deadline = device.launch_stream(), time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(16):
            launch()
        stream.record().wait(60)


def measure(
    device: CPUDevice | CUDADevice, entry: Kernel, args: tuple[Argument, ...], reference: np.ndarray
) -> tuple[list[float], float]:
    bound = device.launcher(entry).bind(*args)
    for _ in range(3):
        bound()
    device.launch_stream().record().wait(60)
    actual = download(args[-1])
    error = float(np.max(np.abs(actual.astype(np.float64) - reference)) / max(float(np.max(np.abs(reference))), 1e-30))
    burn_in(device, bound)
    probe = trials(device, bound, 1)
    repeats = max(1, min(10000, round(30000 / statistics.median(probe))))
    return trials(device, bound, repeats), error


def best_device(
    family: str, target: Target | None, case: Case, settings: Settings
) -> tuple[list[float], float, Explanation]:
    """The fastest of three fresh devices with fresh operands. Where a worker thread or an allocation lands moves one
    kernel's time by 30 percent on a card and by a factor of 2.4 on a hybrid-core CPU (measured on silo: the same C
    program ran 1941 and 4601 us, the same CUDA program 563 and 431 us), and that must not decide a default."""
    found = []
    for _ in range(3):
        if family == "cuda":
            assert cuda_backend is not None and isinstance(target, Nvidia)
            device = cuda_backend.CUDADevice(target, settings)
        else:
            device = CPUDevice(CPU.of("avx512"), settings)
        try:
            args = upload(device, case.arrays)
            launcher = device.launcher(case.entry)
            assert isinstance(launcher, SourceLauncher)
            explanation = launcher.explain(*args)
            found.append((*measure(device, case.entry, args, case.reference), explanation))
        finally:
            if isinstance(device, CPUDevice):
                device.close()
    return min(found, key=lambda result: statistics.median(result[0]))


def overlap_cost() -> list[Row]:
    device = CPUDevice(CPU.of("avx512"), Settings(run=Run(workers=1)))
    try:
        args = tuple(CPUTensor[f32, int](np.ones(1, np.float32)) for _ in range(3))
        launcher = device.launcher(three)
        launcher(*args)

        def sample() -> list[float]:
            times = []
            for _ in range(7):
                start = time.perf_counter_ns()
                for _ in range(1000):
                    launcher(*args)
                times.append((time.perf_counter_ns() - start) / 1e6)
                device.launch_stream().record().wait(60)
            return times

        checked = sample()
        with patch("krn.runtime.runtime.tensor_extents", new=lambda *_args: []):
            unchecked = sample()
        difference = [a - b for a, b in zip(checked, unchecked, strict=True)]
        return [
            Row(
                "three tensors",
                name,
                statistics.median(values),
                "unbound host enqueue; 1000 launches/trial",
                "overlap",
                name,
                "c",
                0,
                values,
            )
            for name, values in (("checked", checked), ("unchecked", unchecked), ("added", difference))
        ]
    finally:
        device.close()


def versions() -> dict[str, str]:
    found = facts()
    result = {name: str(found[name]) for name in ("platform", "cpu", "python", "clang")}
    result["cpu"] = found["cpu"] or "unavailable (host model query denied)"
    result["gpu_and_driver"] = "; ".join(found["gpus"]) if cuda_backend is not None else "unavailable"
    if metal is not None:
        try:
            result["gpu_and_driver"] = f"{metal.MetalDevice(Target.of('apple_m4')).name}; {found['platform']}"
        except metal.MetalError:
            result["gpu_and_driver"] = "unavailable (no Metal device exposed)"
    for name in ("numpy", "cuda-core", "cuda-bindings", "nvidia-cuda-nvrtc"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


@APP.command
def collect(host: str) -> None:
    rows: list[Row] = []
    explanations = {}
    families = ["c"]
    target = next((t for t in system.System.detect().targets if t.vendor == "nvidia"), None)
    if isinstance(target, Nvidia):
        families.append("cuda")
    for family in families:
        for case in cases(family):
            for decision, values in OPTIONS.items():
                if decision == "unroll" and case.name not in {"element", "spmv"}:
                    continue  # only kernels with an `items` loop have items to put on a thread
                for value in values:
                    try:
                        samples, error, explanation = best_device(family, target, case, setting(decision, value))
                        note = "incorrect: excluded" if not math.isfinite(error) or error > TOLERANCE else ""
                        if (
                            not note
                            and family == "c"
                            and decision == "block"
                            and value != "default"
                            and case.name in {"reduction", "tile"}
                        ):
                            note = "C tile runs as one thread; override ignored"
                        rows.append(
                            Row(
                                case.name,
                                f"{decision}={value}",
                                statistics.median(samples),
                                note,
                                decision,
                                value,
                                family,
                                error if math.isfinite(error) else None,
                                samples,
                            )
                        )
                        if value == CURRENT[decision] and decision == "inline":
                            explanations[f"{family}/{case.name}"] = explanation.to_json()
                        LOG.info(
                            "%s %s %s=%s: %.3f us, error %.3g",
                            family,
                            case.name,
                            decision,
                            value,
                            statistics.median(samples),
                            error,
                        )
                    except ValueError as error:
                        rows.append(
                            Row(
                                case.name,
                                f"{decision}={value}",
                                0,
                                f"skipped: {error}",
                                decision,
                                value,
                                family,
                                None,
                                [],
                            )
                        )
    rows.extend(overlap_cost())
    document = {
        "schema": "kl-bench/1",
        "host": host,
        "date": datetime.now().astimezone().isoformat(),
        "versions": versions(),
        "name": "heuristics",
        "kind": "latency",
        "unit": "us",
        "lower_is_better": True,
        "title": "Backend heuristic decisions",
        "rows": [asdict(row) for row in rows],
        "explanations": explanations,
        "note": __doc__.split("\n\n")[-1].strip(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"heuristics_{host}.json"
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    LOG.info("wrote %s", path)


@APP.command
def choose() -> None:
    cells: dict[tuple[str, str], dict[tuple[str, str], dict[str, Row]]] = defaultdict(lambda: defaultdict(dict))
    excluded: set[tuple[str, str, str]] = set()
    for path in sorted(RESULTS.glob("heuristics_*.json")):
        document = json.loads(path.read_text())
        for raw in document["rows"]:
            row = Row(**raw)
            if row.decision == "overlap":
                continue
            key = row.family, row.decision
            if row.note.startswith("incorrect") or (
                row.error is not None and (not math.isfinite(row.error) or row.error > TOLERANCE)
            ):
                excluded.add((*key, row.setting))
            if row.trials and row.error is not None and row.error <= TOLERANCE:
                cells[key][document["host"], row.group][row.setting] = row
    restrict_wins = False
    for (family, decision), family_cells in sorted(cells.items()):
        scores = {}
        for value in OPTIONS[decision]:
            if (family, decision, value) in excluded or any(value not in cell for cell in family_cells.values()):
                continue
            ratios = [
                cell[value].value
                / min(row.value for name, row in cell.items() if (family, decision, name) not in excluded)
                for cell in family_cells.values()
            ]
            scores[value] = statistics.geometric_mean(ratios)
        if not scores or CURRENT[decision] not in scores:
            LOG.info("%s %s: insufficient common cells", family, decision)
            continue
        winner = min(scores, key=lambda value: scores[value])
        margin = scores[winner] <= 0.97 * scores[CURRENT[decision]]
        LOG.info(
            "%s %s: scores %s; winner %s; clears 3%%: %s",
            family,
            decision,
            ", ".join(f"{value}={score:.5f}" for value, score in scores.items()),
            winner,
            margin,
        )
        if decision == "restrict" and "always" in scores and "never" in scores:
            wins = scores["always"] <= 0.97 * scores["never"]
            restrict_wins |= wins
            LOG.info("%s restrict always beats never by 3%%: %s", family, wins)
    LOG.info("restrict recommendation from available families: %s", "auto" if restrict_wins else "never")


@APP.command
def check() -> None:
    device = CPUDevice(CPU.of("avx512"), Settings(run=Run(workers=1)))
    try:
        for case in cases("c", small=True):
            args = upload(device, case.arrays)
            device.launcher(case.entry).submit(*args).wait(60)
            np.testing.assert_allclose(download(args[-1]), case.reference, rtol=TOLERANCE, atol=TOLERANCE)
            LOG.info("%s passed", case.name)
    finally:
        device.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    APP()
