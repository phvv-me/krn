"""Compile latency of one kernel per backend and launch overhead per call, written as normalized benchmark files.

The kernels are the first four of the renderers' common list (`tests/units.py`): saxpy, a tile add, a softmax and a tile
GEMM. For each one the script times, on the machine it runs on:

- the C backend: capture, rendering, and clang with `-O2 -march=native` (the source gets a unique comment so the
  program cache of `cpu.py` never answers; median of 3);
- MSL on Apple: `newLibraryWithSource` and the pipeline state (the first build is reported as cold, the median of the
  next 3 as warm);
- NVRTC for the card's architecture (median of 5 after one cold build), where `KRN_NVRTC_LIB` points at it;
- clang's AMDGPU backend for gfx942 and gfx950, the way `tests/test_hip.py` calls it (median of 3).

Launch overhead is the time of one call of an already built program on one element, a thousand elements and a million,
in microseconds (median of 200 calls after 20 warm-up calls): the cost of the call, not of the kernel. CUDA launches
belong to the CUDA backend track and are not measured here.

Run on the Mac, silo or crimson (each writes `results/compile_backends_<host>.json` and
`results/launch_overhead_<host>.json`):

    mb run --no-install python -m experiments.backend_latency collect mac
    mb run --no-install python -m experiments.backend_latency collect silo
"""

import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from functools import partial
from importlib import import_module
from pathlib import Path

import cyclopts
import numpy as np

import experiments.results_schema as schema
from experiments.conformance import RAW, between, emit, facts, locked_remote, stream
from experiments.paths import RESULTS
from krn import Items
from krn.compiler.api import capture
from krn.compiler.api import emit as emit_unit
from krn.compiler.capture import ConstValues
from krn.compiler.msl import LANGUAGE_VERSION
from krn.facts import targets as targets
from krn.facts.targets import CPU, Target
from krn.runtime.cpu import CLANG, CPUProgram, compile_c
from krn.runtime.nvrtc import Nvidia

LOG = logging.getLogger("backend_latency")
ROOT = Path(__file__).resolve().parents[1]

SKETCH, KERNEL_TESTS = ROOT / "experiments" / "sketch", ROOT / "tests" / "kernels"
APP = cyclopts.App(name="backend_latency", help=__doc__.splitlines()[0])
KERNELS: tuple[tuple[str, str, ConstValues | None], ...] = (
    ("elementwise", "saxpy", None),
    ("elementwise", "add", {"tiling": {"rows": 32, "cols": 32}}),
    ("softmax", "softmax", {"t": {"rows": 2, "cols": 1024}}),
    ("gemm", "gemm", {"t": {"bm": 64, "bn": 64, "bk": 16}}),
)
CPU_ARCH = {"mac": "avx512", "silo": "avx2", "crimson": "avx2", "gh200": "avx512"}
CARD_ARCH = {"silo": "sm_120a", "crimson": "sm_86", "gh200": "sm_90a"}
SHIM = ROOT / "tests" / "hip_shim"


def source_path(file: str) -> Path:
    return KERNEL_TESTS / f"{file}.py" if (KERNEL_TESTS / f"{file}.py").exists() else SKETCH / f"{file}.py"


def median_ms(samples: list[float]) -> float:
    return round(1000 * statistics.median(samples), 3)


def time_call[T](call: Callable[[], T], repeats: int, warmup: int = 0) -> list[float]:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return samples


class Measurements:
    """Rows of the two normalized files for one host."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.compile_rows: list[schema.BenchRow] = []
        self.launch_rows: list[schema.BenchRow] = []

    def add_compile(self, kernel: str, series: str, value: float, note: str = "") -> None:
        self.compile_rows.append({"group": kernel, "series": series, "value": value, "note": note})

    def c_backend(self) -> None:
        target = replace(CPU.of(CPU_ARCH[self.host]), warp_size=16)
        for file, name, consts in KERNELS:
            started = time.perf_counter()
            kernel = capture(source_path(file), name, consts=consts, target=target)
            unit = emit_unit(kernel, target)
            front = time.perf_counter() - started
            clang = [self.unique_clang(unit.source, name, salt) for salt in range(3)]
            self.add_compile(name, "C: capture and render", median_ms([front]))
            self.add_compile(name, "C: clang -O2 -march=native", median_ms(clang))

    @staticmethod
    def unique_clang(source: str, name: str, salt: int) -> float:
        """Seconds clang takes for `source` made unique by a comment, so the program cache never answers."""
        started = time.perf_counter()
        compile_c(f"{source}\n/* {name} {salt} {time.time_ns()} */")
        return time.perf_counter() - started

    def metal(self) -> None:
        Metal = import_module("Metal")
        metal = import_module("krn.runtime.metal")
        GPU = metal.MetalDevice

        gpu = GPU(Target.of("apple_m4"))
        options = Metal.MTLCompileOptions.new()
        options.setLanguageVersion_(getattr(Metal, f"MTLLanguageVersion{LANGUAGE_VERSION[0]}_{LANGUAGE_VERSION[1]}"))

        def unique_build(source: str, entry: str, salt: int) -> float:
            """Seconds to build a library and a pipeline from `source` made unique by a comment."""
            started = time.perf_counter()
            library, error = gpu.core.newLibraryWithSource_options_error_(
                f"{source}\n// {salt} {time.time_ns()}\n", options, None
            )
            if library is None:
                raise RuntimeError(str(error))
            gpu.core.newComputePipelineStateWithFunction_error_(library.newFunctionWithName_(entry), None)
            return time.perf_counter() - started

        for file, name, consts in KERNELS:
            kernel = capture(source_path(file), name, consts=consts, target=gpu.target)
            unit = emit_unit(kernel, gpu.target, lineinfo=False)

            unique = [unique_build(unit.source, unit.entry, salt) for salt in range(3)]
            gpu.program(kernel)
            cached = time_call(lambda k=kernel: gpu.program(k), 5)
            self.add_compile(name, "MSL: library and pipeline, unique source", median_ms(unique))
            self.add_compile(
                name,
                "MSL: same source again (system shader cache)",
                median_ms(cached),
                "the Metal compiler caches by source",
            )

    def nvrtc(self) -> None:

        compiler = Nvidia(Path(os.environ["KRN_NVRTC_LIB"]))
        arch = CARD_ARCH[self.host]
        target = targets.Target.of(arch)
        for file, name, consts in KERNELS:
            source = emit_unit(
                capture(source_path(file), name, consts=consts, target=target), target, lineinfo=False
            ).source
            cold = compiler.compile(source, arch, "cubin").seconds
            warm = [compiler.compile(source, arch, "cubin").seconds for _ in range(5)]
            self.add_compile(name, f"NVRTC cubin {arch}: first build (cold)", median_ms([cold]))
            self.add_compile(name, f"NVRTC cubin {arch}: warm", median_ms(warm))

    def amdgpu(self) -> None:
        for file, name, consts in KERNELS:
            for arch in ("gfx942", "gfx950"):
                target = targets.Target.of(arch)
                source = emit_unit(
                    capture(source_path(file), name, consts=consts, target=target), target, lineinfo=False
                ).source
                with tempfile.TemporaryDirectory() as directory:
                    unit = Path(directory) / f"{name}.hip"
                    unit.write_text(source)
                    command = [
                        CLANG[0],
                        "-x",
                        "hip",
                        f"--offload-arch={arch}",
                        "--offload-device-only",
                        "-nogpulib",
                        "-nogpuinc",
                        "-I",
                        str(SHIM),
                        "-S",
                        "-o",
                        f"{directory}/unit.s",
                        str(unit),
                    ]

                    def compile_once(command=command) -> None:
                        subprocess.run(command, capture_output=True, text=True, timeout=240, check=True)

                    self.add_compile(
                        name,
                        f"clang AMDGPU {arch}",
                        median_ms(time_call(compile_once, 3, 1)),
                        "a clang process, hipRTC is not installed",
                    )

    def launches(self) -> None:
        target = replace(CPU.of(CPU_ARCH[self.host]), warp_size=16)
        kernel = capture(source_path("elementwise"), "saxpy", target=target)
        program = CPUProgram(kernel, emit_unit(kernel, target))
        for size in (1, 1000, 1_000_000):
            x, y = np.ones(size, np.float32), np.ones(size, np.float32)
            call = program.bind(Items(size))
            samples = time_call(partial(call, 2.0, x, y), 200, 20)
            self.launch_rows.append(
                {
                    "group": f"saxpy of {size} elements",
                    "series": "C backend call",
                    "value": round(1e6 * statistics.median(samples), 2),
                }
            )
        if self.host == "mac":
            metal = import_module("krn.runtime.metal")
            GPU = metal.MetalDevice

            gpu = GPU(Target.of("apple_m4"))
            metal_kernel = capture(source_path("elementwise"), "saxpy", target=gpu.target)
            program = gpu.program(metal_kernel)
            for size in (1, 1000, 1_000_000):
                x, y = np.ones(size, np.float32), np.ones(size, np.float32)
                arguments = [2.0, gpu.to_device(x), gpu.to_device(y)]
                call = program.call(program.geometry(Items(size)), arguments)
                samples = time_call(
                    lambda call=call: (call.enqueue(gpu.default_stream), gpu.default_stream.sync()), 200, 20
                )
                self.launch_rows.append(
                    {
                        "group": f"saxpy of {size} elements",
                        "series": "Metal call, synchronous",
                        "value": round(1e6 * statistics.median(samples), 2),
                    }
                )

    def documents(self, versions: dict[str, str]) -> dict[str, schema.BenchFile]:
        """The two normalized benchmark files of this host, by name."""
        today = date.today().isoformat()
        compile_name, launch_name = f"compile_backends_{self.host}", f"launch_overhead_{self.host}"
        return {
            compile_name: {
                "schema": "kl-bench/1",
                "name": compile_name,
                "title": f"Compile latency of four kernels by backend on {self.host}",
                "host": self.host,
                "date": today,
                "unit": "ms",
                "lower_is_better": True,
                "kind": "compile",
                "versions": versions,
                "rows": self.compile_rows,
                "note": "Median of 3 to 5 builds; a clang process stands in for hipRTC; NVRTC and AMDGPU rows exist "
                "only where those compilers do.",
            },
            launch_name: {
                "schema": "kl-bench/1",
                "name": launch_name,
                "title": f"Launch overhead per call on {self.host}",
                "host": self.host,
                "date": today,
                "unit": "us",
                "lower_is_better": True,
                "kind": "latency",
                "versions": versions,
                "rows": self.launch_rows,
                "note": "Median of 200 calls of a built program; the call, not the kernel. CUDA launches are "
                "measured by the CUDA backend track.",
            },
        }


def attempt(label: str, step: Callable[[], None]) -> None:
    """Run one backend's measurement; a backend that this machine lacks is logged and left out, never fatal."""
    try:
        step()
    except (ImportError, OSError, KeyError, subprocess.CalledProcessError, RuntimeError) as error:
        LOG.warning("%s left out: %s: %s", label, type(error).__name__, str(error)[:200])


@APP.command
def measure(host: str) -> None:
    """Measure on this machine and print the two documents (the remote half of `collect`).

    host: the name the rows are recorded under: mac, silo or crimson.
    """
    rows = Measurements(host)
    attempt("C backend", rows.c_backend)
    attempt("launch overhead", rows.launches)
    if host == "mac":
        attempt("Metal compile", rows.metal)
    else:
        attempt("NVRTC", rows.nvrtc)
    attempt("AMDGPU", rows.amdgpu)
    versions = {key: str(value) for key, value in facts().items() if key in ("cpu", "clang", "python", "platform")}
    versions |= {k: v for k, v in facts()["packages"].items() if v}
    emit("latency", json.dumps(rows.documents(versions)))


@APP.command
def collect(host: str, timeout: int = 1500) -> None:
    """Run `measure` on a host and write its two files into `evidence/results`.

    host: mac (here), silo or crimson (through mb, under the GPU lock).
    timeout: seconds for the run.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    if host == "mac":
        paths = ":".join(str(ROOT / folder) for folder in ("src", ".", "experiments/sketch"))
        text = stream(
            [
                "mb",
                "run",
                "--no-install",
                "env",
                f"PYTHONPATH={paths}",
                "python",
                "-m",
                "experiments.backend_latency",
                "measure",
                host,
            ],
            timeout,
        )
    else:
        command = [
            "env",
            "PYTHONPATH=src:.:experiments/sketch",
            "python",
            "-m",
            "experiments.backend_latency",
            "measure",
            host,
        ]
        text = locked_remote(host, [command], timeout)[0]
    (RAW / f"{host}.latency.log").write_text(text)
    write_documents(text)


def write_documents(text: str) -> None:
    """Write the benchmark files that `measure` printed between its markers into `evidence/results`."""
    documents: dict[str, schema.BenchFile] = json.loads(between(text, "latency"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (RESULTS / f"{name}.json").write_text(json.dumps(document, indent=1) + "\n")
        LOG.info("wrote %s (%s rows)", name, len(document["rows"]))


@APP.command
def ingest(log: Path) -> None:
    """Write the benchmark files of a `measure` run made elsewhere (a PBS job), found between markers in `log`.

    log: a file holding the output of `backend_latency.py measure`.
    """
    write_documents(log.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    APP()
