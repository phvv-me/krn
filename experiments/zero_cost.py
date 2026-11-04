"""Zero-cost check on NVIDIA: krn kernels against hand-written CUDA and cuBLASLt, timed by CUPTI device time.

Each time is the median over `--runs` captured windows of the summed CUPTI kernel durations (the device clock, never
host time). saxpy and softmax are compared with hand-written CUDA of the same algorithm, compiled with the NVRTC
flags krn uses and launched with the geometry krn chose. The GEMM baseline is cuBLASLt through `krn.lib`, because a fair
hand-written tensor-core GEMM is larger than this check. Hardware counters are not read: the driver restricts them
(ERR_NVGPUCTRPERM), so occupancy and throughput percentages are absent. Overhead is host time per enqueued launch.

Run on a CUDA host: `mb run --on silo python -m experiments.zero_cost`.
"""

import ctypes
import logging
import statistics
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import cyclopts
import numpy as np
from cuda.bindings import driver as cu
from cuda.core import Kernel as CoreKernel
from cuda.core import ObjectCode

from krn import Blocks, Const, Items, Matrix, T, Vector, block, f16, f32, grid, kernel, pipelined, struct, zeros
from krn.lib import lib_cublaslt
from krn.profile import Profiler
from krn.runtime import host
from krn.runtime.cuda_backend import CUDADevice, CUDATensor, compile_result

log = logging.getLogger("krn.experiments.zero_cost")
app = cyclopts.App(name="zero_cost", help=__doc__.splitlines()[0])

_SAXPY = """
extern "C" __global__ void hand_saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"""

# f16 storage and f32 arithmetic, as krn's softmax: the row maximum, exp(x - max), the sum, then a divide.
# Each block takes four rows, and the threads of a row reduce through shared memory.
_SOFTMAX = """
__device__ float load(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ unsigned short store(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
extern "C" __global__ void hand_softmax(const unsigned short* x, unsigned short* y, int rows, int cols) {
    __shared__ float red[1024];
    for (int r = blockIdx.x * 4; r < min(rows, blockIdx.x * 4 + 4); ++r) {
        const unsigned short* xr = x + (size_t)r * cols;
        unsigned short* yr = y + (size_t)r * cols;
        float m = __int_as_float(0xff800000);
        for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, load(xr[c]));
        red[threadIdx.x] = m;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        m = red[0];
        __syncthreads();
        float sum = 0.0f;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(load(xr[c]) - m);
        red[threadIdx.x] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        sum = red[0];
        __syncthreads();
        for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = store(expf(load(xr[c]) - m) / sum);
    }
}
"""

type Argument = ctypes.c_float | ctypes.c_int32 | ctypes.c_uint64


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@struct
class RowTiling:
    rows: Const[int] = 4
    cols: Const[int] = 4096


@kernel
def softmax(x: Matrix[T.f16], y: Matrix[T.f16], t: RowTiling) -> None:
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    y[rows, cols] = x[rows, cols].to(T.f32).softmax(axis=1).to(T.f16)


@struct
class Tiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32


@kernel
def matmul(t: Tiling, a: Matrix[f16], b: Matrix[f16], c: Matrix[f16]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc.to(f16)


@dataclass(frozen=True, kw_only=True)
class Measure:
    """The median device time of one variant, with the launch facts of its first window."""

    name: str
    device_us: float
    kernels: int
    registers: int
    shared_bytes: int
    geometry: tuple[int, int, int, int] | None


def _device_time(name: str, run: Callable[[], object], runs: int) -> Measure:
    """Time `run` over `runs` synchronized windows, each holding exactly that run's kernels."""
    windows = [Profiler.capture(run)[1] for _ in range(runs)]
    first = windows[0]
    if not first.kernels:
        raise RuntimeError(f"{name}: no CUPTI kernel records in the window")
    trace = first.kernels[0]
    durations = [sum(kernel.duration_ns for kernel in window.kernels) / 1000 for window in windows]
    return Measure(
        name=name,
        device_us=statistics.median(durations),
        kernels=len(first.kernels),
        registers=trace.registers,
        shared_bytes=trace.static_shared_mem + trace.dynamic_shared_mem,
        geometry=first.launches[0].geometry if first.launches else None,
    )


def _hand_kernel(gpu: CUDADevice, source: str, name: str) -> CoreKernel:
    """The hand-written kernel compiled with the device's own NVRTC flags, the flags krn compiles its kernels with."""
    cubin = compile_result(source, gpu.compile_options.flags()).cubin
    return ObjectCode.from_cubin(cubin, name=name).get_kernel(name)


def _launch(gpu: CUDADevice, function: CoreKernel, grid_x: int, threads: int, values: list[Argument]) -> None:
    """Enqueue `function` on the launch stream, one driver call with its arguments encoded as ctypes values."""
    stream = gpu.launch_stream()
    stream.gpu.activate()
    pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(value) for value in values))
    result = cu.cuLaunchKernel(
        function.handle, grid_x, 1, 1, threads, 1, 1, 0, stream.cu, ctypes.addressof(pointers), 0
    )
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed with {result[0]}")


def _geometry(measure: Measure) -> tuple[int, int, int, int]:
    """The launch geometry krn recorded for a variant that ran through the runtime hook."""
    if measure.geometry is None:
        raise RuntimeError(f"{measure.name}: no launch record")
    return measure.geometry


def _pointer(tensor: CUDATensor) -> ctypes.c_uint64:
    return ctypes.c_uint64(tensor.pointer)


def _saxpy_variants(gpu: CUDADevice, n: int, runs: int) -> list[Measure]:
    x, y = gpu.to_device(np.ones(n, np.float32)), gpu.to_device(np.ones(n, np.float32))
    launch = gpu.launcher(saxpy, Items(n)).bind(T.f32(2.0), x, y)
    krn = _device_time("krn saxpy", launch.submit, runs)
    grid_x, _, _, threads = _geometry(krn)
    hand = _hand_kernel(gpu, _SAXPY, "hand_saxpy")
    values: list[Argument] = [ctypes.c_float(2.0), _pointer(x), _pointer(y), ctypes.c_int32(n)]
    baseline = _device_time("hand saxpy", lambda: _launch(gpu, hand, grid_x, threads, values), runs)
    return [krn, baseline]


def _softmax_variants(gpu: CUDADevice, rows: int, cols: int, runs: int) -> list[Measure]:
    x = gpu.to_device(np.random.default_rng(0).standard_normal((rows, cols)).astype(np.float16))
    y = gpu.empty(f16, rows, cols)
    launch = gpu.launcher(softmax, Blocks(rows // 4)).bind(x, y, RowTiling(rows=4, cols=cols))
    krn = _device_time("krn softmax", launch.submit, runs)
    grid_x, _, _, threads = _geometry(krn)
    hand = _hand_kernel(gpu, _SOFTMAX, "hand_softmax")
    out = gpu.empty(f16, rows, cols)
    values: list[Argument] = [_pointer(x), _pointer(out), ctypes.c_int32(rows), ctypes.c_int32(cols)]
    baseline = _device_time("hand softmax", lambda: _launch(gpu, hand, grid_x, threads, values), runs)
    return [krn, baseline]


def _gemm_variants(gpu: CUDADevice, size: int, runs: int) -> list[Measure]:
    rng = np.random.default_rng(1)
    a = gpu.to_device(rng.standard_normal((size, size)).astype(np.float16))
    b = gpu.to_device(rng.standard_normal((size, size)).astype(np.float16))
    c = gpu.empty(f16, size, size)
    launch = gpu.launcher(matmul, Blocks(size // 128, size // 128)).bind(Tiling(), a, b, c)
    krn = _device_time(f"krn gemm {size}", launch.submit, runs)
    pointers = (a.pointer, b.pointer, c.pointer, None)
    stream = gpu.launch_stream().handle
    baseline = _device_time(
        f"cuBLASLt gemm {size}", lambda: lib_cublaslt.multiply(pointers, (size, size, size), f16, stream=stream), runs
    )
    return [krn, baseline]


def _overhead(gpu: CUDADevice, launches: int, repeats: int) -> dict[str, float]:
    """Host nanoseconds per enqueued launch, bare and under each profiler, the wait for the last launch included."""
    x, y = gpu.to_device(np.ones(1, np.float32)), gpu.to_device(np.ones(1, np.float32))
    launch = gpu.launcher(saxpy, Items(1)).bind(T.f32(2.0), x, y)
    modes: dict[str, Callable[[], AbstractContextManager[object]]] = {
        "bare": nullcontext,
        "runtime events": Profiler,
        "CUPTI native": lambda: Profiler(native=True),
    }
    timings: dict[str, float] = {}
    for mode, profiler in modes.items():
        samples = []
        for _ in range(repeats):
            with profiler():
                started = time.perf_counter_ns()
                for _ in range(launches):
                    launch.submit()
                launch.submit().wait()
                samples.append((time.perf_counter_ns() - started) / launches)
        timings[mode] = statistics.median(samples)
    return timings


def _table(title: str, krn: Measure, baseline: Measure) -> str:
    rows = [
        f"{title}",
        f"{'variant':<20}{'device us':>12}{'krn/base':>10}{'regs':>6}{'shared B':>10}{'kernels':>9}  geometry",
    ]
    for row in (krn, baseline):
        ratio = f"{krn.device_us / baseline.device_us:.3f}" if row is krn else ""
        rows.append(
            f"{row.name:<20}{row.device_us:>12.2f}{ratio:>10}{row.registers:>6}{row.shared_bytes:>10}"
            f"{row.kernels:>9}  {row.geometry}"
        )
    return "\n".join(rows)


@app.default
def main(
    *,
    runs: int = 50,
    launches: int = 1000,
    repeats: int = 7,
    gemm_sizes: tuple[int, ...] = (1024, 4096),
    saxpy_elements: int = 1 << 26,
) -> None:
    """Run every comparison on the first CUDA device and log one table per kernel and one overhead line per mode."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gpu = host.Device.current()
    if not isinstance(gpu, CUDADevice):
        raise SystemExit("zero_cost needs a CUDA device")
    log.info("device %s, median of %d windows per variant", gpu.name, runs)
    log.info("%s", _table("saxpy f32", *_saxpy_variants(gpu, saxpy_elements, runs)))
    log.info("%s", _table("softmax 4096 rows x 4096 cols f16", *_softmax_variants(gpu, 4096, 4096, runs)))
    for size in gemm_sizes:
        log.info("%s", _table(f"gemm {size}^3 f16 with f32 accumulate", *_gemm_variants(gpu, size, runs)))
    for mode, nanoseconds in _overhead(gpu, launches, repeats).items():
        log.info("overhead %-14s %10.0f ns per launch", mode, nanoseconds)


if __name__ == "__main__":
    app()
