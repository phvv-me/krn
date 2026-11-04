"""E3, second question: why is our generated C 8x slower than the hand-fused loop on x86?

The kernel is the same chain. Four loops over the same body, each with and without `#include <math.h>` (which, under
`-ffast-math`, is what tells gcc that glibc has SIMD versions of expf and tanhf), compiled by gcc with
`-O3 -march=native -ffast-math`:
`generated` is what our CPU renderer writes (block and thread nest, a guard on the global index), `flat` runs one loop
over the global index and derives the block and thread from it, `guarded` is a plain loop with the guard kept and
`plain` is the hand-fused loop. Run: mb run python -m experiments.e3_loops
"""

import ctypes
import itertools
import logging
import shutil
import statistics
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np

log = logging.getLogger("e3_loops")
N = 1 << 20
BODY = """
        float y = x[i] + b[i];
        y = (0.5f * y) * (1.0f + __builtin_tanhf(0.7978845608f * (y + (((0.044715f * y) * y) * y))));
        out[i] = (__builtin_expf((-y) * y) * y) + 1.0f;
"""
SIGNATURE = (
    "void chain(const float* restrict x, long long n, const float* restrict b, long long bn, "
    "float* restrict out, long long on, long long gx, long long gy, long long gz, long long bdim)"
)
LOOPS = {
    "generated": """
    for (long long bz = 0; bz < gz; ++bz) for (long long by = 0; by < gy; ++by) for (long long bx = 0; bx < gx; ++bx)
    for (long long tx = 0; tx < bdim; ++tx) {
        unsigned long long i = ((unsigned long long)bx * bdim + tx);
        if (i < (unsigned long long)n) {"""
    + BODY
    + "}}",
    "flat": """
    for (long long g = 0; g < gx * bdim; ++g) {
        long long bx = g / bdim, tx = g % bdim; (void)bx; (void)tx;
        long long i = g;
        if (i < n) {"""
    + BODY
    + "}}",
    "guarded": """
    for (long long i = 0; i < gx * bdim; ++i) {
        if (i < n) {"""
    + BODY
    + "}}",
    "plain": """
    for (long long i = 0; i < n; ++i) {"""
    + BODY
    + "}",
}


def median_us(call, repeats: int = 100) -> float:
    for _ in range(5):
        call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call()
        times.append(time.perf_counter_ns() - started)
    return round(statistics.median(times) / 1e3, 1)


def main() -> None:
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise SystemExit("no C compiler")
    rng = np.random.default_rng(0)
    x, b = rng.standard_normal(N, dtype=np.float32), rng.standard_normal(N, dtype=np.float32)
    for name, header in itertools.product(LOOPS, ("", "#include <math.h>\n")):
        loop = LOOPS[name]
        name = f"{name}{'+math.h' if header else ''}"
        folder = Path(tempfile.mkdtemp(prefix="e3-"))
        (folder / "k.c").write_text(f"{header}{SIGNATURE} {{{loop}\n}}\n")
        built = subprocess.run(
            [
                compiler,
                "-O3",
                "-march=native",
                "-ffast-math",
                "-shared",
                "-fPIC",
                str(folder / "k.c"),
                "-o",
                str(folder / "k.so"),
                "-lm",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if built.returncode:
            log.warning("%s did not build: %s", name, built.stderr.strip().splitlines()[-1:])
            continue
        function = ctypes.CDLL(str(folder / "k.so")).chain
        out = np.zeros_like(x)
        args = (x.ctypes.data, N, b.ctypes.data, N, out.ctypes.data, N, N // 256, 1, 1, 256)
        function.argtypes = [ctypes.c_void_p, ctypes.c_longlong] * 3 + [ctypes.c_longlong] * 4
        log.info("%-18s %9.1f us", name, median_us(partial(function, *args)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
