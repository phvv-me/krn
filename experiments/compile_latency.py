"""NVRTC and nvJitLink compile latency on a machine with no GPU, cold (fresh process) and warm (same process).

Run with Python 3.14: python compile_latency.py run <nvidia-cu13 dir> <out.json>
A fresh process per cold sample matters: the first compile loads NVRTC's builtins and builds its internal tables.
"""

import json
import logging
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.paths import RESULTS
from experiments.units import COOP_HEADER_CUDA, DFA_CUDA, FP16_HEADER_CUDA, REDUX_CUDA, SAXPY_CUDA, synthetic
from krn.runtime.nvrtc import Compiled, Nvidia, NvJitLinkInput

log = logging.getLogger("compile_latency")

type Target = tuple[str, str]  # (kind, arch): ptx, cubin or ltoir+link
TARGETS: tuple[Target, ...] = (
    ("ptx", "compute_90"),
    ("cubin", "sm_90a"),
    ("cubin", "sm_120a"),
    ("ltoir+link", "sm_90a"),
)
COLD_SAMPLES = 5


def cases() -> dict[str, tuple[str, int]]:
    """Name to (source, line count)."""
    small = {
        "saxpy": SAXPY_CUDA,
        "dfa_structs": DFA_CUDA,
        "inline_ptx_redux": REDUX_CUDA,
        "header_cuda_fp16": FP16_HEADER_CUDA,
        "header_cooperative_groups": COOP_HEADER_CUDA,
    }
    out = {name: (src, src.count("\n") + 1) for name, src in small.items()}
    for functions in (14, 70, 320):
        source, lines = synthetic(functions, dialect="cuda")
        out[f"synthetic_{lines}_lines"] = (source, lines)
    return out


def compile_target(nv: Nvidia, source: str, target: Target, include: Path) -> dict[str, float]:
    kind, arch = target
    options = (f"-I{include}", f"-I{include / 'cccl'}")
    if kind == "ltoir+link":
        ir: Compiled = nv.compile(source, arch, "ltoir", options)
        linked = nv.link(arch, [(NvJitLinkInput.LTOIR, ir.data)])
        return {"compile_s": ir.seconds, "link_s": linked.seconds, "total_s": ir.seconds + linked.seconds}
    out = nv.compile(source, arch, "ptx" if kind == "ptx" else "cubin", options)
    return {"compile_s": out.seconds, "total_s": out.seconds}


def child(lib_dir: Path, case: str, target: Target) -> None:
    """One cold sample in a fresh process: the very first compile after loading the libraries."""
    nv = Nvidia(lib_dir)
    source, _ = cases()[case]
    sample = compile_target(nv, source, target, lib_dir.parent / "include")
    sys.stdout.write(json.dumps({"load_s": nv.load_seconds, **sample}) + "\n")


@dataclass(frozen=True)
class Row:
    case: str
    lines: int
    kind: str
    arch: str
    cold_total_ms_median: float
    cold_load_ms_median: float
    warm_total_ms_median: float
    warm_total_ms_min: float
    warm_total_ms_p90: float
    warm_samples: int


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def run(lib_dir: Path, out: Path) -> None:
    nv = Nvidia(lib_dir)
    log.info("NVRTC %s.%s, libraries from %s", *nv.nvrtc_version, lib_dir)
    include = lib_dir.parent / "include"
    rows: list[Row] = []
    for case, (source, lines) in cases().items():
        for target in TARGETS:
            try:
                cold = [
                    json.loads(
                        subprocess.run(
                            [sys.executable, "-m", "experiments.compile_latency", "child", str(lib_dir), case, *target],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=300,
                        ).stdout
                    )
                    for _ in range(COLD_SAMPLES)
                ]
                for _ in range(2):
                    compile_target(nv, source, target, include)
                reps = 20 if lines < 200 else 8 if lines < 2000 else 4
                warm = [compile_target(nv, source, target, include)["total_s"] for _ in range(reps)]
            except (subprocess.CalledProcessError, RuntimeError) as error:
                log.warning("%s %s: %s", case, target, str(error)[-300:].replace("\n", " "))
                continue
            row = Row(
                case,
                lines,
                *target,
                1000 * statistics.median(c["total_s"] for c in cold),
                1000 * statistics.median(c["load_s"] for c in cold),
                1000 * statistics.median(warm),
                1000 * min(warm),
                1000 * percentile(warm, 0.9),
                reps,
            )
            rows.append(row)
            log.info(
                "%-28s %5d lines %-10s %-10s cold %8.1f ms  warm median %8.1f ms (min %.1f)",
                case,
                lines,
                row.kind,
                row.arch,
                row.cold_total_ms_median,
                row.warm_total_ms_median,
                row.warm_total_ms_min,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"nvrtc": nv.nvrtc_version, "rows": [asdict(r) for r in rows]}, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1] == "child":
        child(Path(sys.argv[2]), sys.argv[3], (sys.argv[4], sys.argv[5]))
    else:
        run(Path(sys.argv[2]), (Path(sys.argv[3]) if len(sys.argv) > 3 else RESULTS / "nvrtc_latency.json"))
