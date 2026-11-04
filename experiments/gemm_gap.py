"""Resident CUDA GEMM over the unchanged Tilus example's CONFIGS.

CUDA events time queued calls, with three trials after warm-up. Every result is checked against float64 numpy on
32 rows. BF16 specializes a temporary copy of the example by replacing its f16 type token; the user kernel stays
unchanged. BF16 storage is uint16 until the runtime provides a public BF16 tensor constructor.

cuBLASLt uses the same resident pointers. Its current adapter caches planning per thread (the parallel library track
changed that during this task); an older adapter includes setup in each call, making that figure a lower bound.
Run: mb run --on silo env PYTHONPATH=src:.:experiments/sketch python -m experiments.gemm_gap --host silo
"""

import importlib.util
import json
import logging
import os
import statistics
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import cyclopts
import numpy as np
from cuda.core import ObjectCode
from nvmath.bindings.cublasLt import cuBLASLtError

from experiments.conformance import facts
from experiments.paths import RESULTS
from krn import Tiles
from krn.compiler.api import CaptureError, capture
from krn.compiler.debug import Json
from krn.compiler.dtypes import BF16
from krn.frontend.types import bf16, f16
from krn.hw import system as system
from krn.lib.lib_cublaslt import multiply
from krn.runtime.cuda_backend import BACKEND, CUDAError, CUDAProgram, compile_cubin

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDAStream

log = logging.getLogger("gemm_gap")
app = cyclopts.App()
KERNEL = Path(__file__).resolve().parents[1] / "examples" / "04_libraries" / "01_gemm_tiles.py"


def configs() -> list[tuple[int, int, int]]:
    """Read the baseline space from its single source."""
    spec = importlib.util.spec_from_file_location("tilus_v0", KERNEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIGS


def tflops(size: int, seconds: float) -> float:
    return 2 * size**3 / seconds / 1e12


def measure(stream: CUDAStream, enqueue: Callable[[], None], repeats: int) -> float:
    enqueue()
    stream.sync()
    trials = []
    for _ in range(3):
        start = stream.record()
        for _ in range(repeats):
            enqueue()
        stop = stream.record()
        stop.wait(timeout=60)
        trials.append(stop.elapsed_since(start) / 1000 / repeats)
    return statistics.median(trials)


def words(array: np.ndarray) -> np.ndarray:
    bits = array.astype(np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def values(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.uint32) << 16).view(np.float32) if array.dtype == np.uint16 else array.astype(np.float32)


def resources(program: CUDAProgram) -> dict[str, int]:
    # cuda.core's public attributes exist on the host; our optional CUDA stub currently declares only handle.
    attributes = getattr(program.function, "attributes", None)
    if attributes is None:
        raise RuntimeError("cuda.core kernel resource attributes are unavailable")
    return {
        "registers_per_thread": int(attributes.num_regs),
        "local_bytes_per_thread": int(attributes.local_size_bytes),
        "static_shared_bytes": int(attributes.shared_size_bytes),
    }


def sweep(gpu: CUDADevice, size: int, dtype: str, out: Path, repeats: int, only: str) -> None:
    matrix_repeat = int(os.environ.get("KRN_GEMM_MATRIX_REPEAT", "1"))
    if matrix_repeat not in (1, 2):
        raise ValueError("KRN_GEMM_MATRIX_REPEAT must be 1 or 2")
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal((size, size)).astype(np.float32) for _ in range(2))
    a, b = (words(x) if dtype == "bf16" else x.astype(np.float16) for x in (a, b))
    c = np.zeros((size, size), a.dtype)
    tensors = [gpu.to_device(x) for x in (a, b, c)]
    if dtype == "bf16":
        for tensor in tensors:
            tensor.element = BF16
    stream = gpu.default_stream
    av, bv = values(a[:32]).astype(np.float64), values(b).astype(np.float64)
    want, magnitude = av @ bv, np.abs(av) @ np.abs(bv)
    rows: list[Json] = []
    with TemporaryDirectory(prefix="krn-gemm-") as temporary:
        file = KERNEL
        if dtype == "bf16":
            file = Path(temporary) / KERNEL.name
            file.write_text(KERNEL.read_text().replace("f16", "bf16"))
        for bm, bn, bk in configs():
            label = f"{bm}x{bn}x{bk}"
            if only and only != label:
                continue
            try:
                kernel = capture(file, "matmul", consts={"t": {"bm": bm, "bn": bn, "bk": bk}}, target=gpu.nvidia)
                program = gpu.program(kernel)
                if matrix_repeat == 2:
                    # An experiment-only unit copy doubles MMA work with identical staging and fragment loads.
                    # Its output is twice GEMM, with a deliberately longer accumulator dependency chain.
                    source = (
                        "\n".join(
                            "\n".join([line] * (2 if 'asm volatile("mma.sync.' in line else 1))
                            for line in program.unit.source.splitlines()
                        )
                        + "\n"
                    )
                    unit = replace(program.unit, source=source)
                    function = ObjectCode.from_cubin(
                        compile_cubin(source, gpu.compile_options.flags()), name=unit.entry
                    ).get_kernel(unit.entry)
                    program = CUDAProgram(
                        kernel, unit, function, cooperative=False, digest="experiment-double-mma",
                        decisions=program.decisions,
                    )  # fmt: skip
                launch = replace(Tiles((size, size), (bm, bn)).blocks, threads=max(program.unit.threads, 128))
                call = program.call(program.geometry(launch), [{}, *tensors])
                call.enqueue(stream)
                actual = values(tensors[2].to_host()[:32]).astype(np.float64)
                reference = matrix_repeat * want
                rounding = 2 ** (-7 if dtype == "bf16" else -10) * np.abs(reference)
                bound = 8 * 2**-24 * (matrix_repeat * size + 8) * matrix_repeat * magnitude + rounding + 2**-24
                error = float(np.max(np.abs(actual - reference)))
                if not np.all(np.abs(actual - reference) <= bound):
                    raise AssertionError(
                        f"{label} exceeds the f32 accumulation and {dtype} output rounding bound: {error}"
                    )
                seconds = measure(stream, partial(call.enqueue, stream), repeats)
            except (CaptureError, CUDAError, RuntimeError) as refusal:
                log.info("%-12s left out: %s", label, str(refusal).splitlines()[0])
                rows.append({"config": label, "left_out": str(refusal).splitlines()[0]})
                continue
            log.info(
                "%-12s %8.3f ms %8.3f TFLOPS error %.3g shared %d threads %d",
                label,
                seconds * 1e3,
                tflops(size, seconds),
                error,
                program.unit.shared_bytes,
                launch.threads,
            )
            rows.append(
                {
                    "config": label,
                    "seconds": seconds,
                    "tflops": tflops(size, seconds),
                    "error": error,
                    "shared_bytes": program.unit.shared_bytes,
                    "threads": launch.threads,
                    **resources(program),
                }
            )
    baseline: float | None = None
    try:
        pointers = (tensors[0].pointer, tensors[1].pointer, tensors[2].pointer, None)
        kind = bf16 if dtype == "bf16" else f16
        baseline = measure(stream, lambda: multiply(pointers, (size, size, size), kind, stream=stream.handle), repeats)
        log.info(
            "cuBLASLt %8.3f ms %8.3f TFLOPS (adapter planning policy recorded in JSON)",
            baseline * 1e3,
            tflops(size, baseline),
        )
    except (cuBLASLtError, OSError, RuntimeError) as refusal:
        log.info("cuBLASLt unavailable: %s", refusal)
    successful = [row for row in rows if isinstance(row, dict) and isinstance(row.get("tflops"), float)]
    best = max(successful, key=lambda row: float(str(row["tflops"])), default=None)
    log.info("%d %s best: %s", size, dtype, best)
    result = {
        "machine": facts(),
        "size": size,
        "dtype": dtype,
        "matrix_repeat": matrix_repeat,
        "rows": rows,
        "best": best,
        "cublaslt_seconds": baseline,
        "cublaslt_tflops": tflops(size, baseline) if baseline else None,
        "cublaslt_policy": "warm-up before CUDA event timing; current adapter caches planning per thread; "
        "older uncached adapters give a lower bound including per-call setup",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


@app.default
def main(
    size: int = 0, dtype: str = "all", host: str = "", out: Path | None = None, repeats: int = 20, only: str = ""
) -> None:
    if dtype not in ("all", "f16", "bf16") or repeats < 1:
        raise ValueError("dtype must be all, f16 or bf16, and repeats must be positive")
    target = next((t for t in system.System.detect().targets if t.vendor == "nvidia"), None)
    if target is None:
        raise RuntimeError("a CUDA card is required")
    gpu = BACKEND.device(target)
    for n in (size,) if size else (2048, 4096):
        for kind in (dtype,) if dtype != "all" else ("f16", "bf16"):
            destination = (
                Path(str(out).format(host=host or "cuda", size=n, dtype=kind))
                if out
                else (RESULTS / f"gemm_gap_{host or 'cuda'}_{n}_{kind}.json")
            )
            sweep(gpu, n, kind, destination, repeats, only)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
