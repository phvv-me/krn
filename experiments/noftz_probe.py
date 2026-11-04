"""Does `atom.global.add.noftz.f32` keep subnormals in one instruction, so that it can replace krn's float32 CAS loop?

Each form of a float32 add runs on the card under krn's production NVRTC flags (`CompileOptions` through
`compile_cubin`), and every result is compared bit for bit with the emulator: numpy float32 rounds to nearest and keeps
subnormals. The contended cases add powers of two, so every order of the adds gives the same exact sum, and that sum is
the oracle. The uncontended cases add one subnormal to a subnormal start in each word.

Remaining numerical-contract and hardware validation work is tracked in ROADMAP.html.

    mb run --on silo python -m experiments.noftz_probe --out /tmp/noftz_silo.json
    mb run --on crimson python -m experiments.noftz_probe --out /tmp/noftz_crimson.json
"""

import ctypes
import json
import logging
import os
import platform
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import cyclopts
import numpy as np
from cuda.bindings import driver as cu
from cuda.core import ObjectCode

from krn.compiler.cuda import cas_loop
from krn.compiler.dtypes import F32
from krn.facts.targets import Target
from krn.frontend.atomic import AtomicOp
from krn.hw import system
from krn.runtime.compile_options import CompileOptions
from krn.runtime.cuda_backend import BACKEND, CUDAError, check, compile_cubin, nvrtc_version
from krn.runtime.nvrtc import Nvidia as NVRTC
from krn.runtime.nvrtc import NvidiaError

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDAStream

log = logging.getLogger("noftz_probe")
app = cyclopts.App()

ARCHS = ("sm_80", "sm_86", "sm_90a", "sm_120a")
ULP = 2.0**-149  # the smallest positive float32 subnormal
TINY = float(np.finfo(np.float32).tiny)
BLOCK, GRID = 256, 64  # the contended global case: GRID blocks of BLOCK threads add into one word
REPS_GLOBAL, REPS_SHARED = 4, 256  # adds per thread, so both contended cases make 65536 atomics per launch
SPREAD = 1 << 22  # the uncontended cases: one add into each of this many words
WARMUP = 2
NAMED_START = np.array([0.0, 1e-40, 5e-45, -1e-40], np.float32)
NAMED_VALS = np.array([1e-40, 5e-45, -1e-40, -1e-40], np.float32)
GLOBAL, SHARED = "global", "shared"
type Argument = ctypes.c_void_p | ctypes.c_int


@dataclass(frozen=True, kw_only=True)
class Form:
    """One float32 add: `body` is the inside of `float add(float* p, float v)`, which returns the old value."""

    name: str
    body: str
    spaces: frozenset[str]


ATOM_GLOBAL = (
    'float r; asm volatile("atom.global.add.noftz.f32 %0, [%1], %2;" : "=f"(r) : "l"(p), "f"(v) : "memory"); return r;'
)
RED_GLOBAL = 'asm volatile("red.global.add.noftz.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); return 0.0f;'
ATOM_SHARED = (
    "unsigned a = (unsigned)__cvta_generic_to_shared(p); float r; "
    'asm volatile("atom.shared.add.noftz.f32 %0, [%1], %2;" : "=f"(r) : "r"(a), "f"(v) : "memory"); return r;'
)
ATOM_GENERIC = (
    'float r; asm volatile("atom.add.noftz.f32 %0, [%1], %2;" : "=f"(r) : "l"(p), "f"(v) : "memory"); return r;'
)
FORMS = (
    Form(name="krn CAS loop", body=cas_loop(AtomicOp.ADD, F32, "float"), spaces=frozenset({GLOBAL, SHARED})),
    Form(name="atomicAdd", body="return atomicAdd(p, v);", spaces=frozenset({GLOBAL, SHARED})),
    Form(name="atom.global.add.noftz.f32", body=ATOM_GLOBAL, spaces=frozenset({GLOBAL})),
    Form(name="red.global.add.noftz.f32", body=RED_GLOBAL, spaces=frozenset({GLOBAL})),
    Form(name="atom.shared.add.noftz.f32", body=ATOM_SHARED, spaces=frozenset({SHARED})),
    Form(name="atom.add.noftz.f32 (generic)", body=ATOM_GENERIC, spaces=frozenset({GLOBAL, SHARED})),
)
KERNELS = """
extern "C" __global__ void spread(float* out, const float* vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    add(out + i, vals[i]);
}
extern "C" __global__ void contended(float* out, const float* vals, int reps) {
    float v = vals[blockIdx.x * blockDim.x + threadIdx.x];
    for (int r = 0; r < reps; ++r) add(out, v);
}
extern "C" __global__ void spread_shared(float* out, const float* vals) {
    __shared__ float cell[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    cell[t] = out[i];
    __syncthreads();
    add(cell + t, vals[i]);
    __syncthreads();
    out[i] = cell[t];
}
extern "C" __global__ void contended_shared(float* out, const float* vals, int reps) {
    __shared__ float cell;
    if (threadIdx.x == 0) cell = 0.0f;
    __syncthreads();
    float v = vals[threadIdx.x];
    for (int r = 0; r < reps; ++r) add(&cell, v);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = cell;
}
"""


@dataclass(frozen=True, kw_only=True)
class Case:
    """One launch: its start state, its operands, the bits it must leave behind and the atomics it makes."""

    kernel: str
    grid: int
    atomics: int
    start: np.ndarray
    vals: np.ndarray
    expected: np.ndarray
    reps: int = 0  # the contended kernels take a repeat count; the spread kernels take none


@dataclass(frozen=True, kw_only=True)
class Verdict:
    """Whether NVRTC accepts a form for one architecture, as PTX and as a cubin (which ptxas must accept)."""

    arch: str
    form: str
    ptx: str
    cubin: str


@dataclass(frozen=True, kw_only=True)
class Measure:
    form: str
    space: str
    case: str
    atomics: int
    ns_per_atomic: float
    runs: int
    bad_runs: int
    subnormal: int
    kept: int
    flushed: int
    example: str


def source_of(form: Form) -> str:
    return f"__device__ __forceinline__ float add(float* p, float v) {{ {form.body} }}\n{KERNELS}"


def refusal(error: CUDAError | NvidiaError) -> str:
    lines = str(error).splitlines()
    return next((line.strip() for line in lines if "error" in line), lines[0])


def verdicts(nvrtc: NVRTC) -> list[Verdict]:
    rows = []
    for arch in ARCHS:
        flags = CompileOptions.resolve(Target.of(arch)).flags()
        for form in FORMS:
            try:
                ptx_text = nvrtc.compile(source_of(form), arch, "ptx", flags).data.decode()
                version = re.search(r"\.version (\d+\.\d+)", ptx_text)
                ptx = f"PTX {version.group(1)}" if version else "PTX (no version line)"
            except NvidiaError as error:
                ptx = refusal(error)
            try:
                compile_cubin(source_of(form), flags)
                cubin = "accepted"
            except CUDAError as error:
                cubin = refusal(error)
            rows.append(Verdict(arch=arch, form=form.name, ptx=ptx, cubin=cubin))
            log.info("%-8s %-30s ptx: %-60s cubin: %s", arch, form.name, ptx, cubin)
    return rows


def subnormals(rng: np.random.Generator, count: int) -> np.ndarray:
    """Random subnormal float32 values of both signs."""
    bits = rng.integers(1, 1 << 23, count, dtype=np.uint32)
    sign = rng.integers(0, 2, count, dtype=np.uint32) << 31
    return (bits | sign).view(np.float32)


def contended(rng: np.random.Generator, space: str) -> Case:
    """Every thread adds its own signed power of two (1, 2, 4 or 8 ulps) many times into one word that starts at zero.
    All partial sums are exact subnormals, so every order of the adds gives the same bits."""
    shared = space == SHARED
    threads, reps = (BLOCK, REPS_SHARED) if shared else (BLOCK * GRID, REPS_GLOBAL)
    k = rng.choice([-1, 1], threads) * 2 ** (np.arange(threads) % 4)
    exact = np.array([np.ldexp(float(int(k.sum()) * reps), -149)], np.float32)
    return Case(
        kernel="contended_shared" if shared else "contended",
        grid=1 if shared else GRID,
        atomics=threads * reps,
        start=np.zeros(1, np.float32),
        vals=(k * ULP).astype(np.float32),
        expected=exact,
        reps=reps,
    )


def spread(rng: np.random.Generator, space: str) -> Case:
    """One add into each word, from a subnormal start, with the named subnormal cases in the first four words."""
    shared = space == SHARED
    start, vals = subnormals(rng, SPREAD), subnormals(rng, SPREAD)
    start[: len(NAMED_START)], vals[: len(NAMED_VALS)] = NAMED_START, NAMED_VALS
    with np.errstate(all="ignore"):
        expected = start + vals
    return Case(
        kernel="spread_shared" if shared else "spread",
        grid=SPREAD // BLOCK,
        atomics=SPREAD,
        start=start,
        vals=vals,
        expected=expected,
    )


def denormal_counts(expected: np.ndarray, got: np.ndarray) -> tuple[int, int, int]:
    """Expected subnormal results, those whose bits came back equal, and those read back as zero."""
    small = (expected != 0) & (np.abs(expected) < TINY)
    kept = small & (got.view(np.uint32) == expected.view(np.uint32))
    return int(small.sum()), int(kept.sum()), int((small & (got == 0)).sum())


def first_difference(expected: np.ndarray, got: np.ndarray) -> str:
    wrong = np.flatnonzero(got.view(np.uint32) != expected.view(np.uint32))
    if wrong.size == 0:
        return ""
    i = int(wrong[0])
    want, have = expected.view(np.uint32)[i], got.view(np.uint32)[i]
    return f"[{i}] expected {expected[i]!r} ({want:#010x}) got {got[i]!r} ({have:#010x})"


def launch(stream: CUDAStream, function: cu.CUkernel, case: Case, arguments: list[Argument]) -> None:
    table = (ctypes.c_void_p * len(arguments))(*(ctypes.addressof(argument) for argument in arguments))
    check(cu.cuLaunchKernel(function, case.grid, 1, 1, BLOCK, 1, 1, 0, stream.cu, ctypes.addressof(table), 0)[0])


def measure(gpu: CUDADevice, function: cu.CUkernel, case: Case, repeats: int) -> tuple[list[float], int, np.ndarray]:
    """Device-event times in milliseconds, the runs whose bits differed from the expected ones, and the first result."""
    stream = gpu.default_stream
    out, vals = gpu.to_device(case.start), gpu.to_device(case.vals)
    arguments: list[Argument] = [ctypes.c_void_p(out.pointer), ctypes.c_void_p(vals.pointer)]
    if case.reps:
        arguments.append(ctypes.c_int(case.reps))
    times, bad, first = [], 0, []
    for index in range(WARMUP + repeats):
        check(cu.cuMemcpyHtoDAsync(out.pointer, case.start.ctypes.data, case.start.nbytes, stream.cu)[0])
        begin = stream.record()
        launch(stream, function, case, arguments)
        end = stream.record()
        end.wait(timeout=600)
        if index >= WARMUP:
            times.append(end.elapsed_since(begin))
        got = out.to_host(stream)
        first = first or [got]
        bad += int(not np.array_equal(got.view(np.uint32), case.expected.view(np.uint32)))
    return times, bad, first[0]


def driver_version() -> str:
    status, version = cu.cuDriverGetVersion()
    check(status)
    return f"{version // 1000}.{version % 1000 // 10}"


@app.default
def probe(repeats: int = 20, out: Path | None = None) -> None:
    """The acceptance table for every architecture, then each form on this card, `repeats` timed launches each."""
    target = next((t for t in system.System.detect().targets if t.vendor == "nvidia"), None)
    if target is None:
        raise RuntimeError("a CUDA card is required")
    gpu = BACKEND.device(target)
    nvrtc = NVRTC(Path(os.environ["KRN_NVRTC_LIB"]))
    rng = np.random.default_rng(0)
    cases = {space: (contended(rng, space), spread(rng, space)) for space in (GLOBAL, SHARED)}
    for space, (cell, many) in cases.items():
        pairwise = np.sum(np.repeat(cell.vals, cell.reps), dtype=np.float32)
        log.info(
            "%s: contended exact %r, numpy float32 sum %r; uncontended %d atomics",
            space, cell.expected[0], pairwise, many.atomics,
        )  # fmt: skip
    accepted = verdicts(nvrtc)
    rows = []
    for form in FORMS:
        try:
            cubin = compile_cubin(source_of(form), gpu.compile_options.flags())
            module = ObjectCode.from_cubin(cubin, name="noftz_probe")
        except CUDAError as error:
            log.info("%s does not build on %s: %s", form.name, target.arch, refusal(error))
            continue
        for space in (GLOBAL, SHARED):
            if space not in form.spaces:
                continue
            for case in cases[space]:
                function = module.get_kernel(case.kernel).handle
                times, bad, first = measure(gpu, function, case, repeats)
                subnormal, kept, flushed = denormal_counts(case.expected, first)
                median_ns = statistics.median(times) * 1e6 / case.atomics
                row = Measure(
                    form=form.name,
                    space=space,
                    case="contended" if case.reps else "uncontended",
                    atomics=case.atomics,
                    ns_per_atomic=median_ns,
                    runs=WARMUP + repeats,
                    bad_runs=bad,
                    subnormal=subnormal,
                    kept=kept,
                    flushed=flushed,
                    example=first_difference(case.expected, first),
                )
                rows.append(row)
                log.info(
                    "%-30s %-6s %-11s %10.3f ns/atomic bad %d/%d subnormal %d kept %d flushed %d %s",
                    row.form, row.space, row.case, row.ns_per_atomic, row.bad_runs, row.runs,
                    row.subnormal, row.kept, row.flushed, row.example,
                )  # fmt: skip
    report = {
        "date": date.today().isoformat(),
        "host": platform.node(),
        "gpu": gpu.name,
        "arch": target.arch,
        "nvrtc": nvrtc_version(),
        "driver": driver_version(),
        "acceptance": [asdict(v) for v in accepted],
        "measures": [asdict(m) for m in rows],
    }
    if out:
        out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
