"""Zero-cost check of the CPU backend: krn's generated C against hand-written C and BLAS, one thread.

Each kernel runs on the calling thread through `CPUProgram`, so the perf_event counters that `HardwareCounters` opens
for this thread see its instructions and cycles. Hand-written C is compiled by the same `compile_c` and flags, so the
only difference is the source. BLAS (numpy) is multithreaded and only its wall time is reported.

Run on the Linux CPU host: `mb run --on silo python -m experiments.zero_cost_cpu`.
"""

import ctypes
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from krn import Items, T, Vector, grid, items, kernel
from krn.compiler.capture import capture
from krn.compiler.lower import build
from krn.facts.targets import CPU
from krn.hw.libraries import Libraries
from krn.profile.providers.cpu.perf import HardwareCounters
from krn.runtime.cpu import CPUProgram, compile_c

log = logging.getLogger("zero_cost_cpu")
TARGET = CPU.of("avx2")
REPEATS = 30
SAXPY_N = 1 << 22
DOT_N = 1 << 22
GEMM_N = 192

HANDWRITTEN = """
void saxpy(float a, const float *restrict x, float *restrict y, long n) {
    for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}
void dot(const float *restrict x, const float *restrict y, float *restrict out, long n) {
    float acc = 0.0f;
    for (long i = 0; i < n; i++) acc += x[i] * y[i];
    out[0] = acc;
}
void gemm(const float *restrict a, const float *restrict b, float *restrict c, long n) {
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            float acc = 0.0f;
            for (long k = 0; k < n; k++) acc += a[i * n + k] * b[k * n + j];
            c[i * n + j] = acc;
        }
}
"""


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@kernel
def dot(x: Vector[T.f32], y: Vector[T.f32], out: Vector[T.f32]) -> None:
    acc = T.f32(0.0)
    for i in items(x.shape[0]):
        acc = acc + x[i] * y[i]
    out[0] = acc


@kernel
def gemm(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], n: T.u64) -> None:
    t = grid.thread_rank()
    if t < n * n:
        row = t // n
        col = t % n
        acc = T.f32(0.0)
        for k in range(n):
            acc = acc + a[row * n + k] * b[k * n + col]
        c[t] = acc


@dataclass(frozen=True, kw_only=True)
class Measure:
    """Median per-call counts and wall time over `REPEATS` calls, in-process on this thread."""

    name: str
    instructions: float
    cycles: float
    wall_ns: float

    @property
    def ipc(self) -> float | None:
        return self.instructions / self.cycles if self.cycles else None


def measure(name: str, call: Callable[[], object]) -> Measure:
    call()
    walls: list[float] = []
    with HardwareCounters(["cycles", "instructions"]) as counters:
        for _ in range(REPEATS):
            started = time.perf_counter_ns()
            call()
            walls.append(time.perf_counter_ns() - started)
        counts = counters.read()
    return Measure(
        name=name,
        instructions=counts["instructions"] / REPEATS,
        cycles=counts["cycles"] / REPEATS,
        wall_ns=statistics.median(walls),
    )


def handwritten() -> ctypes.CDLL:
    library, _ = compile_c(HANDWRITTEN, TARGET)
    handle = Libraries.load(str(library)).handle
    handle.saxpy.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    handle.dot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    handle.gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    return handle


def pointer(array: np.ndarray) -> int:
    return array.ctypes.data


def compiled(name: str) -> CPUProgram:
    ir = capture(Path(__file__), name, target=TARGET)
    return build(ir, TARGET)


def run() -> list[tuple[str, Measure, Measure, Measure]]:
    rng = np.random.default_rng(0)
    native = handwritten()
    x, y = rng.random(SAXPY_N, dtype=np.float32), rng.random(SAXPY_N, dtype=np.float32)
    a = 2.0
    saxpy_krn = compiled("saxpy")
    dot_krn = compiled("dot")
    gemm_krn = compiled("gemm")
    m = rng.random((GEMM_N, GEMM_N), dtype=np.float32)
    k = rng.random((GEMM_N, GEMM_N), dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    c = np.zeros(GEMM_N * GEMM_N, dtype=np.float32)
    rows = [
        (
            "saxpy",
            measure("krn", lambda: saxpy_krn(a, x, y, launch=Items(SAXPY_N))),
            measure("hand C", lambda: native.saxpy(float(a), pointer(x), pointer(y), SAXPY_N)),
            measure("numpy", lambda: np.add(a * x, y, out=y)),
        ),
        (
            "dot",
            measure("krn", lambda: dot_krn(x, y, out, launch=Items(1))),
            measure("hand C", lambda: native.dot(pointer(x), pointer(y), pointer(out), DOT_N)),
            measure("numpy", lambda: np.dot(x, y)),
        ),
        (
            "gemm",
            measure("krn", lambda: gemm_krn(m.ravel(), k.ravel(), c, GEMM_N, launch=Items(GEMM_N * GEMM_N))),
            measure("hand C", lambda: native.gemm(pointer(m), pointer(k), pointer(c), GEMM_N)),
            measure("numpy", lambda: m @ k),
        ),
    ]
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    for name, krn, hand, blas in run():
        log.info("%s: krn ipc %s, hand C ipc %s", name, krn.ipc, hand.ipc)
        log.info(
            "%s: instructions krn/hand %.3f, cycles krn/hand %.3f, wall krn/hand %.3f, wall numpy/hand %.3f",
            name,
            krn.instructions / hand.instructions,
            krn.cycles / hand.cycles,
            krn.wall_ns / hand.wall_ns,
            blas.wall_ns / hand.wall_ns,
        )


if __name__ == "__main__":
    main()
