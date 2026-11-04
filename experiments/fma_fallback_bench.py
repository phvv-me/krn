"""The scalar matrix fallback and the sparse row kernels, with each fused multiply-add and its separate spelling.

After, every product in the fallback and in the sparse kernels is one `fma`, one rounding per product. Before, the same
kernels spelled each `fma(a, b, c)` as a product rounded and then added, which is the arithmetic the fallback loop
(`_s += a * b`) and the sparse generator sums (`sum(v * x)`) had. The before form is that spelling, switched on for one
run by patching the renderer's `math` for `Fn.FMA`. The old loop shapes are not rebuilt, so the difference measured is
the rounding spelling of the same kernels.

Each case runs on the CPU backend (one thread per block, which its empty barrier needs) or on the Apple GPU, and its
output is compared with a float64 reference. Times are the median of 5 trials of queued launches, as in `sparse_bench`.
The table goes to the log at INFO. Run on the Mac: `mb run python -m experiments.fma_fallback_bench cpu`, and
`mb run python -m experiments.fma_fallback_bench apple`.
"""

import logging
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import cyclopts
import numpy as np

from experiments.sparse_bench import error, median_seconds, queued
from krn import Matrix, T, Tiles, block, kernel, zeros
from krn.compiler.api import capture, emit
from krn.compiler.dtypes import Dtype
from krn.compiler.emit import Renderer
from krn.compiler.ir import Fn, Kernel
from krn.data import sparse, sparse_build
from krn.facts.targets import Target
from krn.frontend.geometry import Geometry, Items
from krn.runtime.cpu import CPUProgram, Value
from krn.runtime.metal import MetalDevice, MetalTensor
from krn.runtime.metal import Value as MetalValue
from tests.helpers.devices import bindings

log = logging.getLogger("fma_fallback_bench")
app = cyclopts.App(name="fma_fallback_bench", help=__doc__.splitlines()[0])
THIS, SPARSE = Path(__file__), Path(sparse.__file__)
CPU, APPLE = Target.of("avx512"), Target.of("apple_m4")
SIZE, ROWS, PER_ROW, WIDTH = 256, 4096, 16, 16
GEMM_THREADS = {"cpu": 1, "apple": 256}  # the C backend's block barrier is empty, so one thread runs a block


@kernel
def gemm(a: Matrix[T.f32], b: Matrix[T.f32], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(32, 32)
    acc = zeros(T.f32, 32, 32)
    for ks in range(0, a.shape[1], 32):
        acc += a[rows, slice(ks, ks + 32)] @ b[slice(ks, ks + 32), cols]
    c[rows, cols] = acc


class Variant(NamedTuple):
    """One run of a kernel: its file and name, and whether each fma is spelled as a product and a sum."""

    file: Path
    kernel: str
    separate: bool


@dataclass(frozen=True, kw_only=True)
class Case:
    """A kernel with its arguments, the index of its output argument and that output's float64 reference."""

    name: str
    kernel: str
    file: Path
    arguments: list[Value]
    output: int
    reference: np.ndarray
    structs: dict[str, str]
    launch: dict[str, Geometry]

    def variant(self, *, separate: bool) -> Variant:
        return Variant(self.file, self.kernel, separate)


@contextmanager
def separate_rounding(enabled: bool) -> Generator[None]:
    """Every `fma` of the kernel is spelled as a product rounded, then added, for the duration of the compile."""
    if not enabled:
        yield
        return
    original = Renderer.math

    def spelled(self: Renderer, fn: Fn, dtype: Dtype, args: list[str]) -> str:
        if fn is Fn.FMA:
            return f"({args[0]} * {args[1]} + {args[2]})"
        return original(self, fn, dtype, args)

    with patch.object(Renderer, "math", spelled):
        yield


def random_csr(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A dense ROWS x ROWS matrix with about PER_ROW entries a row, and the same matrix in CSR."""
    dense = np.zeros((ROWS, ROWS), np.float32)
    entries = rng.standard_normal(ROWS * PER_ROW)
    dense[np.repeat(np.arange(ROWS), PER_ROW), rng.integers(0, ROWS, ROWS * PER_ROW)] = entries
    return dense, sparse_build.csr_arrays(dense)._asdict()


def gemm_case(rng: np.random.Generator) -> Case:
    a = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    b = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    blocks = Tiles((SIZE, SIZE), (32, 32)).blocks
    return Case(
        name=f"gemm{SIZE}",
        kernel="gemm",
        file=THIS,
        arguments=[a, b, np.zeros((SIZE, SIZE), np.float32)],
        output=2,
        reference=a.astype(np.float64) @ b.astype(np.float64),
        structs={},
        launch={backend: replace(blocks, threads=threads) for backend, threads in GEMM_THREADS.items()},
    )


def spmv_case(rng: np.random.Generator, name: str) -> Case:
    """`spmv_vector` needs warp shuffles, which the C backend does not have, so it runs on the Apple GPU only."""
    dense, matrix = random_csr(rng)
    x = rng.standard_normal(ROWS).astype(np.float32)
    launch: dict[str, Geometry] = (
        {"cpu": Items(ROWS), "apple": Items(ROWS)} if name == "spmv" else {"apple": Items(ROWS * 32)}
    )
    return Case(
        name=name,
        kernel=name,
        file=SPARSE,
        arguments=[matrix, x, np.zeros(ROWS, np.float32)],
        output=2,
        reference=dense.astype(np.float64) @ x.astype(np.float64),
        structs={"a": "Csr"},
        launch=launch,
    )


def spmm_case(rng: np.random.Generator) -> Case:
    dense, matrix = random_csr(rng)
    b = rng.standard_normal((ROWS, WIDTH)).astype(np.float32)
    return Case(
        name="spmm",
        kernel="spmm",
        file=SPARSE,
        arguments=[matrix, b, np.zeros((ROWS, WIDTH), np.float32)],
        output=2,
        reference=dense.astype(np.float64) @ b.astype(np.float64),
        structs={"a": "Csr"},
        launch={backend: Items(ROWS * WIDTH) for backend in ("cpu", "apple")},
    )


def sddmm_case(rng: np.random.Generator) -> Case:
    _, pattern = random_csr(rng)
    left = rng.standard_normal((ROWS, 16)).astype(np.float32)
    right = rng.standard_normal((16, ROWS)).astype(np.float32)
    rows = np.repeat(np.arange(ROWS), np.diff(pattern["indptr"]))
    reference = np.einsum("kd,dk->k", left[rows].astype(np.float64), right[:, pattern["indices"]].astype(np.float64))
    return Case(
        name="sddmm",
        kernel="sddmm",
        file=SPARSE,
        arguments=[pattern, left, right, np.zeros(pattern["values"].shape[0], np.float32)],
        output=3,
        reference=reference,
        structs={"pattern": "Csr"},
        launch={backend: Items(ROWS) for backend in ("cpu", "apple")},
    )


def cases() -> list[Case]:
    rng = np.random.default_rng(0)
    return [
        gemm_case(rng),
        spmv_case(rng, "spmv"),
        spmv_case(rng, "spmv_vector"),
        spmm_case(rng),
        sddmm_case(rng),
    ]


def compile_kernel(case: Case, variant: Variant, target: Target) -> Kernel:
    return capture(
        variant.file,
        variant.kernel,
        structs=case.structs,
        dtypes=bindings(variant.file, variant.kernel, case.arguments),
        target=target,
    )


def measure_cpu(case: Case, variant: Variant) -> tuple[float, float]:
    ir = compile_kernel(case, variant, CPU)
    with separate_rounding(variant.separate):
        program = CPUProgram(ir, emit(ir, CPU))
    arguments = [a.copy() if isinstance(a, np.ndarray) else a for a in case.arguments]
    launch = case.launch["cpu"]

    def step() -> None:
        program(*arguments, launch=launch)

    seconds = median_seconds(queued(step, lambda: None))
    step()
    output = arguments[case.output]
    assert isinstance(output, np.ndarray)
    return seconds, error(output, case.reference)


def upload(device: MetalDevice, value: Value) -> MetalValue:
    """Each array on the device, each struct as a dict of its arrays, and each scalar as it is."""
    match value:
        case np.ndarray():
            return device.to_device(value)
        case Mapping():
            return {name: upload(device, part) for name, part in value.items()}
        case _:
            return value


def measure_apple(case: Case, variant: Variant, device: MetalDevice) -> tuple[float, float]:
    ir = compile_kernel(case, variant, APPLE)
    with separate_rounding(variant.separate):
        program = device.program(ir)
    arguments = [upload(device, value) for value in case.arguments]
    call, stream = program.geometry(case.launch["apple"]), device.default_stream

    def step() -> None:
        program.call(call, arguments).enqueue(stream)

    seconds = median_seconds(queued(step, lambda: stream.sync()))
    step()
    stream.sync()
    output = arguments[case.output]
    assert isinstance(output, MetalTensor)
    return seconds, error(output.array, case.reference)


def report(backend: str, measure: Callable[[Case, Variant], tuple[float, float]]) -> None:
    header = ("case", "backend", "before us", "after us", "after/before", "err before", "err after")
    log.info("%-12s %-7s %12s %12s %11s %12s %12s", *header)
    for case in cases():
        if backend not in case.launch:
            continue
        before_seconds, before_error = measure(case, case.variant(separate=True))
        after_seconds, after_error = measure(case, case.variant(separate=False))
        log.info(
            "%-12s %-7s %12.1f %12.1f %11.2f %12.2e %12.2e",
            case.name,
            backend,
            before_seconds * 1e6,
            after_seconds * 1e6,
            after_seconds / before_seconds,
            before_error,
            after_error,
        )  # fmt: skip


@app.command
def cpu() -> None:
    """Every case on the CPU backend, one thread per block."""
    report("cpu", measure_cpu)


@app.command
def apple() -> None:
    """Every case on the Apple GPU through Metal."""
    device = MetalDevice(APPLE)
    report("apple", lambda case, variant: measure_apple(case, variant, device))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
