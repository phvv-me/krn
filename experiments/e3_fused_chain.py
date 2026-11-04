"""E3 of round 6: does our own CPU renderer reach the speed of a hand-fused loop on the 25-op chain?

The chain is a GELU-style tanh and `exp(-y * y) * y + 1` over a million floats. Four implementations, timed on the
machine that runs this file, one thread each unless the name says OpenMP: numpy, the hand-fused C++ loop (the baseline
of round 6, `-O3 -march=native -ffast-math`), our C backend at its own flags, and our backend at the baseline's flags.
Round 6 measured XLA at 607 us against 578 us for the loop on four threads; the rule it set was "within 1.5x".

Run: mb run python -m experiments.e3_fused_chain [--out evidence/results/e3.json]
"""

import ctypes
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cyclopts
import numpy as np

from krn import Items
from krn.compiler.api import build, capture
from krn.facts import targets as targets
from krn.runtime import cpu as backend

log = logging.getLogger("e3")
app = cyclopts.App()
N = 1 << 20
FUSED = """
#include <cmath>
#include <cstdint>
extern "C" void chain(const float* x, const float* b, float* out, int64_t n) {
#pragma omp parallel for simd schedule(static) if (n > 65536)
  for (int64_t i = 0; i < n; ++i) {
    float y = x[i] + b[i];
    y = 0.5f * y * (1.0f + std::tanh(0.7978845608f * (y + 0.044715f * y * y * y)));
    out[i] = std::exp(-y * y) * y + 1.0f;
  }
}
"""


def reference(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = x + b
    y = 0.5 * y * (1.0 + np.tanh(0.7978845608 * (y + 0.044715 * y * y * y)))
    return np.exp(-y * y) * y + 1.0


def median_us[TypingResult](call: Callable[[], TypingResult], repeats: int = 200) -> float:
    for _ in range(5):
        call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call()
        times.append(time.perf_counter_ns() - started)
    return round(statistics.median(times) / 1e3, 1)


def ours(
    flags: tuple[str, ...], x: np.ndarray, b: np.ndarray, out: np.ndarray, compiler: str | None = None
) -> Callable[[], None]:
    """Our C backend compiled with `flags`, by `compiler` (the backend's clang when not given)."""
    saved = backend.CLANG
    backend.CLANG = (compiler or saved[0], *flags, "-shared", "-fPIC", "-fwrapv", "-x", "c")
    try:
        target = replace(targets.CPU.of("avx512"), warp_size=16)
        program = build(capture(Path(__file__).with_name("e3_chain_kernel.py"), "chain", target=target), target)
    finally:
        backend.CLANG = saved
    launch = program.bind(Items(N))
    return lambda: launch(x, b, out)


def fused(openmp: bool, x: np.ndarray, b: np.ndarray, out: np.ndarray) -> Callable[[], None] | None:
    """The hand-fused loop, or None where this machine's C++ compiler cannot build it."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        return None
    folder = Path(tempfile.mkdtemp(prefix="e3-"))
    (folder / "fused.cc").write_text(FUSED)
    flags = ["-O3", "-march=native", "-ffast-math", "-fPIC", "-shared", *(["-fopenmp"] if openmp else [])]
    done = subprocess.run(
        [compiler, *flags, str(folder / "fused.cc"), "-o", str(folder / "fused.so")],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        log.warning("hand-fused (openmp=%s) did not build: %s", openmp, done.stderr.strip().splitlines()[-1:])
        return None
    library = ctypes.CDLL(str(folder / "fused.so"))
    library.chain.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64]
    return lambda: library.chain(x.ctypes.data, b.ctypes.data, out.ctypes.data, N)


@app.default
def main(out: Path | None = None) -> None:
    rng = np.random.default_rng(0)
    x, b = rng.standard_normal(N, dtype=np.float32), rng.standard_normal(N, dtype=np.float32)
    want = reference(x, b)
    result: dict[str, str | int | dict[str, float] | None] = {
        "machine": f"{platform.machine()} {platform.system()}",
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    candidates: dict[str, Callable[[], np.ndarray | int | None] | None] = {}
    outputs = {name: np.zeros_like(x) for name in ("numpy", "ours_default", "ours_fast", "fused_1t", "fused_openmp")}
    candidates["numpy"] = lambda: reference(x, b)
    candidates["ours_default"] = ours(("-O2", "-march=native", "-mtune=generic"), x, b, outputs["ours_default"])
    candidates["ours_fast"] = ours(("-O3", "-march=native", "-ffast-math"), x, b, outputs["ours_fast"])
    outputs["ours_libmvec"] = np.zeros_like(x)
    try:  # clang vectorizes libm calls only when told which vector library to call
        fast = ("-O3", "-march=native", "-ffast-math", "-fveclib=libmvec")
        candidates["ours_libmvec"] = ours((*fast, "-lm", "-lmvec"), x, b, outputs["ours_libmvec"])
    except (subprocess.CalledProcessError, OSError) as error:
        log.warning("ours with libmvec did not build: %s", error)
        candidates["ours_libmvec"] = None
    outputs["ours_gcc"] = np.zeros_like(x)
    gcc = shutil.which("gcc")
    candidates["ours_gcc"] = (
        ours(("-O3", "-march=native", "-ffast-math"), x, b, outputs["ours_gcc"], gcc) if gcc else None
    )
    candidates["fused_1t"] = fused(False, x, b, outputs["fused_1t"])
    candidates["fused_openmp"] = fused(True, x, b, outputs["fused_openmp"])
    for name, call in candidates.items():
        if call is None:
            result[name] = None
            continue
        us = median_us(call)
        call()
        error = float(np.max(np.abs(outputs[name] - want))) if name != "numpy" else 0.0
        result[name] = {"us_median": us, "max_abs_err": error}
        log.info("%-14s %9.1f us   max error %.2e", name, us, error)
    base = result["fused_1t"]
    if isinstance(base, dict):
        for name in ("ours_default", "ours_fast", "ours_libmvec", "ours_gcc"):
            row = result[name]
            if isinstance(row, dict):
                row["vs_fused_1t"] = round(row["us_median"] / base["us_median"], 2)
    log.info("%s", json.dumps(result))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
