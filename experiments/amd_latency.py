"""AMDGPU compile latency with the LLVM that hipRTC wraps, as a proxy: OpenCL C to a gfx942 or gfx950 code object.

No ROCm install and no GPU are needed. clang runs as a process, so each sample includes about one process
start, which the `empty` baseline measures. Run: python amd_latency.py <clang> <ld.lld> <out.json>
"""

import json
import logging
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.paths import RESULTS
from experiments.units import DFA_OPENCL, SAXPY_OPENCL, synthetic

log = logging.getLogger("amd_latency")

GFX = ("gfx942", "gfx950")
EMPTY = "__kernel void k(__global int* p) { p[0] = 1; }"


@dataclass(frozen=True)
class Row:
    case: str
    lines: int
    gfx: str
    compile_ms_median: float
    compile_ms_min: float
    link_ms_median: float
    code_object_bytes: int
    samples: int


def build(clang: str, lld: str, source: str, gfx: str, workdir: Path) -> tuple[float, float, int]:
    """Compile to an object, then link it into a code object. Returns seconds for each and the final size."""
    src, obj, out = workdir / "unit.cl", workdir / "unit.o", workdir / "unit.hsaco"
    src.write_text(source)
    started = time.perf_counter()
    subprocess.run(
        [
            clang,
            "--target=amdgcn-amd-amdhsa",
            f"-mcpu={gfx}",
            "-x",
            "cl",
            "-cl-std=CL2.0",
            "-O2",
            "-nogpulib",
            "-Xclang",
            "-finclude-default-header",
            "-c",
            str(src),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    compiled = time.perf_counter()
    subprocess.run([lld, "-shared", str(obj), "-o", str(out)], check=True, capture_output=True, text=True, timeout=60)
    linked = time.perf_counter()
    return compiled - started, linked - compiled, out.stat().st_size


def main(clang: str, lld: str, out: Path) -> None:
    version = subprocess.run([clang, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    log.info("%s", version)
    units: dict[str, tuple[str, int]] = {
        "empty": (EMPTY, 1),
        "saxpy": (SAXPY_OPENCL, SAXPY_OPENCL.count("\n") + 1),
        "dfa_structs": (DFA_OPENCL, DFA_OPENCL.count("\n") + 1),
    }
    for functions in (14, 70, 320):
        source, lines = synthetic(functions, dialect="opencl")
        units[f"synthetic_{lines}_lines"] = (source, lines)
    rows: list[Row] = []
    with tempfile.TemporaryDirectory() as tmp:
        for case, (source, lines) in units.items():
            for gfx in GFX:
                reps = 12 if lines < 200 else 6 if lines < 2000 else 3
                samples = [build(clang, lld, source, gfx, Path(tmp)) for _ in range(reps + 1)][1:]
                row = Row(
                    case,
                    lines,
                    gfx,
                    1000 * statistics.median(s[0] for s in samples),
                    1000 * min(s[0] for s in samples),
                    1000 * statistics.median(s[1] for s in samples),
                    samples[-1][2],
                    reps,
                )
                rows.append(row)
                log.info(
                    "%-28s %5d lines %-7s clang %8.1f ms (min %.1f)  lld %5.1f ms  %6d bytes",
                    case,
                    lines,
                    gfx,
                    row.compile_ms_median,
                    row.compile_ms_min,
                    row.link_ms_median,
                    row.code_object_bytes,
                )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clang": version, "rows": [asdict(r) for r in rows]}, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1], sys.argv[2], (Path(sys.argv[3]) if len(sys.argv) > 3 else RESULTS / "amd_latency.json"))
