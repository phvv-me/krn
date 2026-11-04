"""cuBLASLt enqueue overhead and throughput on resident row-major matrices.

Run on the named host through mainboard, with sketch, prototype and experiments on PYTHONPATH:
`python -m experiments.vendor_bench cublaslt --host silo --out evidence/results/vendor_cublaslt_silo.json`.
The host option labels the machine; mainboard owns remote dispatch. Both f16 and f32 inputs accumulate in f32.
Each path gets five warm-ups and five trials: 20 calls at 4096 cubed, 2000 at 256 cubed. CUDA events measure
the queued interval; a wall clock around just the enqueue loop measures host cost, excluding synchronization.
The direct baseline constructs its own handle, descriptors and top heuristic once through nvmath.bindings.
"""

import ctypes
import json
import logging
import platform
import statistics
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import cyclopts
import numpy as np
from cuda.bindings.runtime import cudaDataType as DataType
from nvmath.bindings import cublasLt as lt
from nvmath.bindings.cublas import ComputeType

from experiments.paths import RESULTS
from krn import T, configure
from krn.compiler.debug import Json
from krn.frontend.numerics import FloatingPoint, name_of
from krn.hw import system as system
from krn.lib import lib as lib
from krn.lib import lib_cublaslt as lib_cublaslt
from krn.lib import lib_nvidia as lib_nvidia
from krn.runtime import host as runtime_host

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDAStream

log = logging.getLogger("vendor_bench")
app = cyclopts.App()


def direct(
    cleanup: ExitStack,
    size: int,
    dtype: type[FloatingPoint],
    pointers: tuple[int, int, int],
    workspace: tuple[int, int],
    stream: int,
) -> Callable[[], None]:
    """Plan the independent baseline; the returned call only enqueues cublasLtMatmul."""
    handle = lt.create()
    cleanup.callback(lt.destroy, handle)
    desc = lt.matmul_desc_create(ComputeType.COMPUTE_32F, DataType.CUDA_R_32F)
    cleanup.callback(lt.matmul_desc_destroy, desc)
    layouts = []
    for _ in range(3):
        code = DataType.CUDA_R_16F if dtype == T.f16 else DataType.CUDA_R_32F
        layout = lt.matrix_layout_create(code, size, size, size)
        cleanup.callback(lt.matrix_layout_destroy, layout)
        layouts.append(layout)
    preference = lt.matmul_preference_create()
    cleanup.callback(lt.matmul_preference_destroy, preference)
    limit = ctypes.c_uint64(workspace[1])
    lt.matmul_preference_set_attribute(
        preference, lt.MatmulPreferenceAttribute.MAX_WORKSPACE_BYTES, ctypes.addressof(limit), ctypes.sizeof(limit)
    )
    for attribute, pointer in (
        (lt.MatmulPreferenceAttribute.MIN_ALIGNMENT_A_BYTES, pointers[1]),
        (lt.MatmulPreferenceAttribute.MIN_ALIGNMENT_B_BYTES, pointers[0]),
        (lt.MatmulPreferenceAttribute.MIN_ALIGNMENT_C_BYTES, pointers[2]),
        (lt.MatmulPreferenceAttribute.MIN_ALIGNMENT_D_BYTES, pointers[2]),
    ):
        alignment = ctypes.c_uint32(min(256, pointer & -pointer))
        lt.matmul_preference_set_attribute(preference, attribute, ctypes.addressof(alignment), ctypes.sizeof(alignment))
    left, right, out = layouts
    result, found = lt.MatmulHeuristicResult(), ctypes.c_int()
    lt.matmul_algo_get_heuristic(
        handle, desc, left, right, out, out, preference, 1, result.ptr, ctypes.addressof(found)
    )
    if not found.value:
        raise RuntimeError(f"direct cuBLASLt found no algorithm for {name_of(dtype)} {size} cubed")
    if result.state or result.workspace_size > workspace[1] or workspace[0] % 256:
        raise RuntimeError("direct cuBLASLt returned an unusable algorithm or workspace")
    alpha, beta = ctypes.c_float(1), ctypes.c_float(0)

    # The heuristic struct starts with its algorithm; keep result alive in the closure.
    def enqueue() -> None:
        lt.matmul(
            handle,
            desc,
            ctypes.addressof(alpha),
            pointers[1],
            left,
            pointers[0],
            right,
            ctypes.addressof(beta),
            pointers[2],
            out,
            pointers[2],
            out,
            result.ptr,
            workspace[0],
            workspace[1],
            stream,
        )

    return enqueue


@dataclass(frozen=True)
class Measure:
    gpu_us: float
    host_us: float
    tflops: float
    throughput_ratio: float = 1.0
    gpu_time_ratio: float = 1.0
    host_time_ratio: float = 1.0


def measure(step: Callable[[], None], stream: CUDAStream, size: int, repeats: int) -> Measure:
    for _ in range(5):
        step()
    stream.record().wait(60)
    gpu_times, host_times = [], []
    for _ in range(5):
        begin = stream.record()
        started = time.perf_counter_ns()
        for _ in range(repeats):
            step()
        host_times.append((time.perf_counter_ns() - started) / repeats / 1e3)
        end = stream.record()
        end.wait(60)
        gpu_times.append(end.elapsed_since(begin) * 1e3 / repeats)
    gpu_us = statistics.median(gpu_times)
    return Measure(gpu_us, statistics.median(host_times), 2 * size**3 / gpu_us / 1e6)


def benchmark(device: CUDADevice, size: int, dtype: type[FloatingPoint]) -> dict[str, Json]:
    stream, rng = device.default_stream, np.random.default_rng(0)
    numpy_type = np.float16 if dtype == T.f16 else np.float32
    a_host = (rng.standard_normal((size, size)) / np.sqrt(size)).astype(numpy_type)
    b_host = rng.standard_normal((size, size)).astype(numpy_type)
    a, b, out = device.to_device(a_host), device.to_device(b_host), device.empty(dtype, size, size)
    options = lib.LibOptions.of(device.target)
    workspace = device.empty(T.u8, options.workspace_bytes)
    stream.record().wait(60)
    pointers = (lib_nvidia.address(a), lib_nvidia.address(b), lib_nvidia.address(out))
    problem = lib.Problem(lib.Op.MATMUL, (dtype,) * 3, (size,) * 3, options)
    operands, candidate = lib.Operands(a, b, out), lib_cublaslt.CublasLt()
    repeats = 20 if size == 4096 else 2000
    with ExitStack() as cleanup, stream:
        baseline = direct(
            cleanup,
            size,
            dtype,
            pointers,
            (lib_nvidia.address(workspace), options.workspace_bytes),
            lib_nvidia.handle(stream),
        )

        def adapter() -> None:
            candidate.run(problem, operands, stream)

        def dispatched() -> None:
            lib.matmul(a, b, out=out, options=options)

        baseline()
        expected = out.to_host()
        paths = {"direct": baseline, "candidate": adapter, "matmul": dispatched}
        measurements = {}
        for name, step in paths.items():
            measurements[name] = measure(step, stream, size, repeats)
            tolerance = 3e-3 if dtype == T.f16 else 1e-5
            np.testing.assert_allclose(out.to_host(), expected, rtol=tolerance, atol=tolerance)
    base = measurements["direct"]
    figures: dict[str, Json] = {}
    for name, result in measurements.items():
        result = Measure(
            result.gpu_us,
            result.host_us,
            result.tflops,
            result.tflops / base.tflops,
            result.gpu_us / base.gpu_us,
            result.host_us / base.host_us,
        )
        figures[name] = {
            "gpu_us": result.gpu_us,
            "host_us": result.host_us,
            "tflops": result.tflops,
            "throughput_ratio": result.throughput_ratio,
            "gpu_time_ratio": result.gpu_time_ratio,
            "host_time_ratio": result.host_time_ratio,
        }
        log.info(
            "%s %4d %-9s %8.3f TFLOP/s  GPU %9.3f us  host %8.3f us  throughput/direct %.4f",
            name_of(dtype),
            size,
            name,
            result.tflops,
            result.gpu_us,
            result.host_us,
            result.throughput_ratio,
        )
    return {
        "dtype": name_of(dtype),
        "compute": "f32",
        "shape": [size] * 3,
        "calls_per_trial": repeats,
        "trials": 5,
        "warmups": 5,
        "figures": figures,
    }


@app.command
def cublaslt(host: str = platform.node(), out: Path | None = None) -> None:
    """Measure the current CUDA device; host labels the JSON and its default output filename."""
    configure(**{"lib.mode": "vendor", "lib.tune": "off", "lib.matmul": None})
    for cached in (system.System.detect, lib._dispatcher):
        cached.cache_clear()
    backend = import_module("krn.runtime.cuda_backend")
    device = runtime_host.Device.current()
    if not isinstance(device, backend.CUDADevice):
        raise TypeError("vendor_bench cublaslt requires a CUDA device")
    if reason := lib_cublaslt.CublasLt().unavailable(device.target):
        raise RuntimeError(reason)
    status, driver = lib_nvidia.driver().cuDriverGetVersion()
    if status:
        raise RuntimeError(f"cuDriverGetVersion returned {status}")
    result: dict[str, Json] = {
        "machine": {
            "host": host,
            "gpu": device.name,
            "sm": device.nvidia.sm,
            "cublaslt": lib_cublaslt.CublasLt().library_version(),
            "cuda_driver": driver,
        },
        "results": [benchmark(device, size, dtype) for dtype in (T.f16, T.f32) for size in (4096, 256)],
    }
    destination = out or (RESULTS / f"vendor_cublaslt_{host}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    log.info("wrote %s", destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
