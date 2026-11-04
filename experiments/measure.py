"""Measure the frontend: capture, IR, emit, NVRTC and clang times per kernel, sizes, and the sketch's metrics.

    mb run --no-install python -m experiments.measure run <nvidia-cu13 lib dir> [--out evidence/results]

Cold numbers come from a fresh process per sample (a child of this script), warm numbers from one process.
Every batch records the load average, because other agents share the machine; a noisy batch is repeated.
"""

import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cyclopts

from experiments import metrics
from experiments.paths import RESULTS
from krn.compiler.api import capture, emit
from krn.compiler.capture import Builder
from krn.compiler.dtypes import Arithmetic
from krn.compiler.source import Module
from krn.facts.targets import Nvidia, Target
from krn.runtime.cpu import CLANG
from krn.runtime.nvrtc import Nvidia as Nvrtc

log = logging.getLogger("measure")
app = cyclopts.App()
ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "experiments" / "sketch"
COLD, WARM, NOISY = 5, 10, 1.25
TARGETS: dict[str, Target] = {
    "cuda_sm_90a": Target.of("sm_90a"),
    "cuda_sm_120a": Target.of("sm_120a"),
    "cpu": Target.of("avx512"),
}
CPU = TARGETS["cpu"]
type Consts = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Case:
    label: str
    file: str
    name: str
    consts: Consts | None = None
    tilelang_s: float | None = None  # warm NVRTC cubin time of the TileLang kernel, round3/tilelang-compile-only.md

    @property
    def path(self) -> Path:
        return SKETCH / f"{self.file}.py"


CASES = (
    Case("saxpy", "elementwise", "saxpy"),
    Case("add", "elementwise", "add", {"tiling": {"rows": 32, "cols": 32}}, 0.83),
    Case("softmax", "softmax", "softmax", {"t": {"rows": 2, "cols": 4096}}, 0.83),
    Case("gemm", "gemm", "gemm", None, 1.43),
    Case("run_dfa", "automaton", "run_dfa"),
    Case("block_sum", "target_variables", "block_sum"),
)
BY_LABEL = {c.label: c for c in CASES}


def load() -> float:
    return os.getloadavg()[0]


def median_ms(samples: list[float]) -> float:
    return 1000 * statistics.median(samples)


def timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    return fn(), time.perf_counter() - started


def frontend_once(case: Case, target: Target) -> dict[str, float]:
    """Seconds for parse, check plus IR, and emit of one kernel for one target."""
    module, parse = timed(lambda: Module(case.path))
    kernel, ir = timed(
        lambda: Builder(module, target, Arithmetic.STRICT, case.consts or {}, {}, {}).kernel(module.kernel(case.name))
    )
    _, emitted = timed(lambda: emit(kernel, target))
    return {"parse": parse, "ir": ir, "emit": emitted}


@app.command
def child_frontend(label: str, target: str) -> None:
    """One cold sample: the first capture and emit after the imports of a fresh process."""
    sys.stdout.write(json.dumps(frontend_once(BY_LABEL[label], TARGETS[target])) + "\n")


@app.command
def child_nvrtc(lib_dir: Path, source: Path, arch: str, kind: str) -> None:
    """One cold sample: the first NVRTC compile after loading the libraries of a fresh process."""
    nvrtc = Nvrtc(lib_dir)
    out = nvrtc.compile(source.read_text(), arch, "ptx" if kind == "ptx" else "cubin")
    sys.stdout.write(json.dumps({"load": nvrtc.load_seconds, "total": out.seconds, "bytes": len(out.data)}) + "\n")


def cold(*argv: str) -> list[dict[str, float]]:
    command = [sys.executable, "-m", "experiments.measure"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [ROOT, ROOT / "src"]))}
    return [
        json.loads(subprocess.run([*command, *argv], capture_output=True, text=True, check=True, env=env).stdout)
        for _ in range(COLD)
    ]


@dataclass(frozen=True)
class Batch:
    ms: float
    spread: float  # slowest over fastest sample
    load: float


def batches(measure: Callable[[], float]) -> Batch:
    """Run a warm batch up to three times while it is noisy (spread above 1.25) and keep the quietest."""
    runs: list[Batch] = []
    for _ in range(3):
        before = load()
        samples = [measure() for _ in range(WARM)]
        runs.append(Batch(median_ms(samples), max(samples) / min(samples), before))
        if runs[-1].spread < NOISY:
            break
    return min(runs, key=lambda b: b.spread)


@dataclass
class Row:
    label: str
    target: str
    cold_ms: dict[str, float]
    warm_ms: dict[str, float]
    cuda_lines: int = 0
    source_bytes: int = 0
    load: float = 0.0


def frontend_row(case: Case, name: str, target: Target) -> Row:
    samples = cold("child-frontend", case.label, name)
    cold_ms = {k: median_ms([s[k] for s in samples]) for k in samples[0]}
    frontend_once(case, target)
    warm = {k: median_ms([frontend_once(case, target)[k] for _ in range(30)]) for k in ("parse", "ir", "emit")}
    source = emit(capture(case.path, case.name, consts=case.consts, target=target), target).source
    return Row(case.label, name, cold_ms, warm, source.count("\n"), len(source), load())


def warm_batch(nvrtc: Nvrtc, source: str, arch: str, kind: str) -> Batch:
    nvrtc.compile(source, arch, "ptx" if kind == "ptx" else "cubin")
    return batches(lambda: nvrtc.compile(source, arch, "ptx" if kind == "ptx" else "cubin").seconds)


def nvrtc_rows(case: Case, nvrtc: Nvrtc, lib_dir: Path, work: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for name in ("cuda_sm_90a", "cuda_sm_120a"):
        target = TARGETS[name]
        assert isinstance(target, Nvidia)
        source = emit(capture(case.path, case.name, consts=case.consts, target=target), target).source
        file = work / f"{case.label}_{name}.cu"
        file.write_text(source)
        for kind, arch in (("cubin", f"sm_{target.sm}a"), ("ptx", f"compute_{target.sm}a")):
            samples = cold("child-nvrtc", str(lib_dir), str(file), arch, kind)
            warm = warm_batch(nvrtc, source, arch, kind)
            rows.append({
                "label": case.label, "target": name, "kind": kind, "bytes": int(samples[0]["bytes"]),
                "cold_ms": median_ms([s["total"] for s in samples]),
                "cold_load_ms": median_ms([s["load"] for s in samples]),
                "warm_ms": warm.ms, "warm_spread": warm.spread, "load": warm.load,
                "tilelang_warm_ms": 1000 * case.tilelang_s if case.tilelang_s else 0,
            })  # fmt: skip
    return rows


def clang_row(case: Case) -> dict[str, str | float | int]:
    source = emit(capture(case.path, case.name, consts=case.consts, target=CPU), CPU).source
    runs = [
        timed(lambda: subprocess.run([*CLANG, "-o", "/dev/null", "-"], input=source, text=True, check=True))[1]
        for _ in range(COLD)
    ]
    return {
        "label": case.label,
        "lines": source.count("\n"),
        "bytes": len(source),
        "clang_ms": median_ms(runs),
        "load": load(),
    }


UNITS = {
    "saxpy": ("elementwise", ["saxpy"]),
    "add": ("elementwise", ["add", "Tiling"]),
    "softmax": ("softmax", ["softmax", "RowTiling"]),
    "gemm": ("gemm", ["gemm", "GemmTiling"]),
    "run_dfa": ("automaton", ["run_dfa", "Dfa", "Segments"]),
    "block_sum": ("target_variables", ["block_sum", "lane_sum", "butterfly_steps"]),
}  # the sketch source each kernel needs: the kernel, its config struct, its helpers


@app.command
def sketch_metrics(out: Path) -> None:
    """NLOC, CCN, tokens and API symbols of each sketch kernel next to those of the CUDA and C it expands to."""
    members = metrics.api_members(sorted((ROOT / "src" / "krn").rglob("*.py")))
    rows = []
    for case in CASES:
        _, names = UNITS[case.label]
        ranges = {n: (s, e) for n, s, e in metrics.kernels_in(case.path)}
        parts = [metrics.measure("krn-gpu", case.path, *ranges[n], members) for n in names]
        row = {
            "kernel": case.label,
            "nloc": sum(p.nloc for p in parts), "ccn_max": max(p.ccn_max for p in parts),
            "llm_tokens": sum(p.llm_tokens for p in parts), "api_symbols": len({a for p in parts for a in p.api_names}),
        }  # fmt: skip
        for name, suffix in (("cuda_sm_90a", ".cu"), ("cpu", ".c")):
            target = TARGETS[name]
            if name == "cpu" and case.label == "block_sum":
                continue
            source = emit(capture(case.path, case.name, consts=case.consts, target=target), target).source
            nloc, _, ccn, _ = metrics.lizard_numbers(source, suffix)
            row |= {
                f"{name}_nloc": nloc,
                f"{name}_ccn": ccn,
                f"{name}_llm_tokens": len(metrics.ENCODING.encode(source)),
            }
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")


@app.command
def run(lib_dir: Path, out: Path = RESULTS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nvrtc = Nvrtc(lib_dir)
    log.info("NVRTC %s.%s, load %s", *nvrtc.nvrtc_version, os.getloadavg())
    work = out / "units"
    work.mkdir(exist_ok=True)
    frontend, compiled, clang = [], [], []
    for case in CASES:
        for name, target in TARGETS.items():
            if name == "cpu" and case.label == "block_sum":
                continue  # needs warp shuffles: capture rejects it on the CPU target, by design
            frontend.append(asdict(frontend_row(case, name, target)))
        compiled += nvrtc_rows(case, nvrtc, lib_dir, work)
        if case.label != "block_sum":
            clang.append(clang_row(case))
        log.info("%s done, load %.2f", case.label, load())
    meta = {
        "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        "nvrtc": nvrtc.nvrtc_version,
    }
    for name, rows in (("frontend", frontend), ("nvrtc", compiled), ("clang", clang)):
        (out / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
