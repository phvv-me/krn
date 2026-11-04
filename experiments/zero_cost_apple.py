"""Zero-cost check on the Apple GPU: krn's generated Metal against hand-written Metal and MLX

Three kernels, each run as krn's own generated MSL, as a hand-written Metal compute kernel of the same algorithm and
launch geometry (krn's grid and block, compiled with krn's options: `MTLMathModeSafe`, precise functions, the same
language version and `fp contract(off)`), and, for matmul and softmax, as MLX's op. Times are steady-state: each
measured launch is one of a queue of back-to-back launches after a discarded queue, because a GPU that idles between
submissions runs its clock lower and doubles some launches. krn and the hand-written kernels report GPU time, the
command buffer's GPUStartTime to GPUEndTime, with the timestamp counter at the encoder's stage boundaries read beside it
as a cross-check (`counter_gap_ns`). MLX does not expose its command buffers, so its time is the wall time of a queue of
ops evaluated together, divided by the number of ops. Each figure is the median over the measured queue.

Bandwidth is the traffic the algorithm asks for (saxpy reads x and y and writes y, softmax reads x twice and writes y),
against the M4 Pro's 273 GB/s peak. FLOP/s counts 2mnk for the matmuls. Ratios are krn over the reference, so above 1
krn is slower. A hand-written ratio over 1.05 needs a diagnosis from the generated lines, which the JSON keeps as
`krn_source`. Run on the Mac: `mb proc timeout 900 mb run python -m experiments.zero_cost_apple`. The JSON goes to
`evidence/results/`, which stays local.
"""

import json
import logging
import statistics
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cyclopts
import Metal
import mlx.core as mx
import numpy as np

from experiments.paths import RESULTS
from krn import Const, Items, Kernel, Matrix, T, Vector, block, f16, f32, grid, kernel, maximum, pipelined, zeros
from krn import struct as krn_struct
from krn.compiler.errors import TargetError
from krn.compiler.msl import LANGUAGE_VERSION
from krn.facts.targets import Apple
from krn.frontend.numerics import Scalar
from krn.profile.providers.apple.capabilities import Capabilities
from krn.profile.providers.apple.signals import Signal, require
from krn.profile.providers.apple.timing import StageTimer, clock_offset_ns, command_span
from krn.runtime.metal import MetalDevice, MetalLauncher
from krn.runtime.runtime import Argument

log = logging.getLogger("zero_cost_apple")
app = cyclopts.App(name="zero_cost_apple", help=__doc__.splitlines()[0])
PEAK_GBPS = 273.0  # Apple's memory bandwidth specification for the M4 Pro
WARMUP, MEASURED = 20, 50  # launches per queue: one discarded queue, then the measured one
MLX_REPEATS = 6
TILE, BK = 64, 32
SOFTMAX_ROWS, SOFTMAX_COLS = 4, 1024
DEVICE_ROWS, DEVICE_COLS = 1024, 4096
MATMUL_SIZE = 2048
SAXPY_SIZE = 1 << 24


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@krn_struct
class Tiling:
    bm: Const[int] = TILE
    bn: Const[int] = TILE
    bk: Const[int] = BK


@kernel
def matmul_f16(t: Tiling, a: Matrix[f16], b: Matrix[f16], c: Matrix[f16]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc.to(f16)


@kernel
def matmul_f32(t: Tiling, a: Matrix[f32], b: Matrix[f32], c: Matrix[f32]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@krn_struct
class RowTiling:
    rows: Const[int] = SOFTMAX_ROWS
    cols: Const[int] = SOFTMAX_COLS


@kernel
def softmax(t: RowTiling, x: Matrix[f32], y: Matrix[f32]) -> None:
    rows = block.tile(t.rows)
    m = zeros(T.f32, t.rows, 1).sub(3.0e38)
    s = zeros(T.f32, t.rows, 1)
    for cs in pipelined(x.shape[1], t.cols):
        v = x[rows, cs]
        m_new = maximum(m, v.max(keepdims=True))
        s = s * (m - m_new).exp() + (v - m_new).exp().sum(keepdims=True)
        m = m_new
    for cs in pipelined(x.shape[1], t.cols):
        y[rows, cs] = (x[rows, cs] - m).exp().div(s)


# Hand-written references, bound in the order their buffers are named. Each keeps krn's grid and block.
_SAXPY = """
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void saxpy(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& a [[buffer(2)]],
                  constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = a * x[i] + y[i];
}
"""

# A 64x64 block tile over K steps of 32, four simdgroups in a 2x2 arrangement, each owning a 32x32 corner of the tile
# as 4x4 simdgroup matrices. Staging copies without the generated code's bounds predicates, which the multiple-of-tile
# sizes here never need. Padded strides keep the half-precision tiles within the 32 KB threadgroup budget.
_MATMUL = """
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void matmul(const device {T}* a [[buffer(0)]], const device {T}* b [[buffer(1)]], device {T}* c [[buffer(2)]],
                   constant uint2& nk [[buffer(3)]], uint2 block [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {{
    threadgroup {T} ta[64 * {PA}];
    threadgroup {T} tb[32 * {PB}];
    threadgroup float acc[64 * 64];
    const uint n = nk.x, k = nk.y;
    const uint row0 = block.y * 64, col0 = block.x * 64;
    const uint wr = sg / 2 * 32, wc = sg % 2 * 32;
    simdgroup_float8x8 f[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) f[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint ks = 0; ks < k; ks += 32) {{
        for (uint e = tid; e < 64 * 32; e += 128) ta[(e / 32) * {PA} + e % 32] = a[(row0 + e / 32) * k + ks + e % 32];
        for (uint e = tid; e < 32 * 64; e += 128) tb[(e / 64) * {PB} + e % 64] = b[(ks + e / 64) * n + col0 + e % 64];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {{
            simdgroup_{S}8x8 am[4], bm[4];
            for (int i = 0; i < 4; ++i) simdgroup_load(am[i], ta + (wr + i * 8) * {PA} + kk, {PA});
            for (int j = 0; j < 4; ++j) simdgroup_load(bm[j], tb + kk * {PB} + wc + j * 8, {PB});
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(f[i][j], am[i], bm[j], f[i][j]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_store(f[i][j], acc + (wr + i * 8) * 64 + wc + j * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 64 * 64; e += 128) c[(row0 + e / 64) * n + col0 + e % 64] = ({T})acc[e];
}}
"""

# Four rows per threadgroup and 128 threads. Each thread folds its strided columns into an online max and sum, the
# first four threads combine the 128 partials, and every thread then writes its columns: the same two passes as above.
_SOFTMAX = """
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void softmax(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    constant uint& cols [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float pm[4][128], ps[4][128], fm[4], fs[4];
    const uint base = group * 4;
    for (uint r = 0; r < 4; ++r) {
        const device float* row = x + (base + r) * cols;
        float m = -3.0e38f, s = 0.0f;
        for (uint c = tid; c < cols; c += 128) {
            float v = row[c];
            float mn = max(m, v);
            s = s * exp(m - mn) + exp(v - mn);
            m = mn;
        }
        pm[r][tid] = m;
        ps[r][tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        float m = -3.0e38f, s = 0.0f;
        for (uint t = 0; t < 128; ++t) {
            float mn = max(m, pm[tid][t]);
            s = s * exp(m - mn) + ps[tid][t] * exp(pm[tid][t] - mn);
            m = mn;
        }
        fm[tid] = m;
        fs[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < 4; ++r) {
        const device float* row = x + (base + r) * cols;
        device float* out = y + (base + r) * cols;
        for (uint c = tid; c < cols; c += 128) out[c] = exp(row[c] - fm[r]) / fs[r];
    }
}
"""

type Record = float | int | str | bool | list[int] | None


@dataclass(frozen=True, kw_only=True)
class Case:
    """One kernel's medians in nanoseconds, with the traffic and arithmetic its algorithm needs."""

    name: str
    geometry: tuple[int, int, int, int]
    krn_ns: float
    krn_p10_ns: float
    krn_p90_ns: float
    hand_ns: float
    mlx_ns: float | None
    counter_gap_ns: float
    traffic_bytes: int
    flops: int
    correct: bool
    krn_source: str

    def record(self) -> dict[str, Record]:
        return {
            "case": self.name,
            "geometry": list(self.geometry),
            "krn_ns": self.krn_ns,
            "krn_p10_ns": self.krn_p10_ns,
            "krn_p90_ns": self.krn_p90_ns,
            "hand_ns": self.hand_ns,
            "mlx_ns": self.mlx_ns,
            "ratio_hand": self.krn_ns / self.hand_ns,
            "ratio_mlx": None if self.mlx_ns is None else self.krn_ns / self.mlx_ns,
            "krn_gbps": self.traffic_bytes / self.krn_ns,
            "krn_pct_peak": 100 * self.traffic_bytes / self.krn_ns / PEAK_GBPS,
            "krn_gflops": self.flops / self.krn_ns,
            "hand_gflops": self.flops / self.hand_ns,
            "counter_gap_ns": self.counter_gap_ns,
            "correct": self.correct,
            "krn_source": self.krn_source,
        }


def krn_queue(device: MetalDevice, launch: Callable[[], None], count: int) -> list[int]:
    """Queue `count` launches back to back and return each command buffer's GPU nanoseconds."""
    stream = device.launch_stream()
    for _ in range(count):
        launch()
    commands = list(stream.commands)
    stream.sync()
    return [end - start for start, end in (command_span(command) for command in commands)]


class HandWritten:
    """A hand-written compute kernel compiled through PyObjC with the options krn's Metal program uses."""

    def __init__(self, device: MetalDevice, source: str, entry: str) -> None:
        options = Metal.MTLCompileOptions.new()
        options.setLanguageVersion_(getattr(Metal, f"MTLLanguageVersion{LANGUAGE_VERSION[0]}_{LANGUAGE_VERSION[1]}"))
        options.setMathMode_(Metal.MTLMathModeSafe)
        options.setMathFloatingPointFunctions_(Metal.MTLMathFloatingPointFunctionsPrecise)
        library, error = device.core.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            raise RuntimeError(f"{error}\n{source}")
        pipeline, error = device.core.newComputePipelineStateWithFunction_error_(
            library.newFunctionWithName_(entry), None
        )
        if pipeline is None:
            raise RuntimeError(str(error))
        self.pipeline, self.device = pipeline, device

    def constant(self, fmt: str, *values: int | float) -> Metal.MTLBuffer:
        data = struct.pack(fmt, *values)
        return self.device.core.newBufferWithBytes_length_options_(data, len(data), Metal.MTLResourceStorageModeShared)

    def queue(
        self,
        timer: StageTimer,
        buffers: Sequence[Metal.MTLBuffer],
        geometry: tuple[int, int, int, int],
        count: int,
    ) -> tuple[list[int], list[int]]:
        """Queue `count` dispatches with krn's grid and block, then return each command buffer's GPU nanoseconds
        and the gap between its command-buffer span and its counter span."""
        x, y, z, threads = geometry
        commands = []
        for index in range(count):
            command = self.device.default_stream.queue.commandBuffer()
            encoder = command.computeCommandEncoderWithDescriptor_(timer.descriptor(index))
            encoder.setComputePipelineState_(self.pipeline)
            for slot, buffer in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(x, y, z), Metal.MTLSizeMake(threads, 1, 1)
            )
            encoder.endEncoding()
            command.commit()
            commands.append(command)
        times, gaps = [], []
        for index, command in enumerate(commands):
            command.waitUntilCompleted()
            if command.error() is not None:
                raise RuntimeError(str(command.error()))
            start, end = command_span(command)
            counter_start, counter_end = timer.read(index)
            times.append(end - start)
            gaps.append(abs(counter_start - start) + abs(counter_end - end))
        return times, gaps


def launch_geometry(launcher: MetalLauncher, args: tuple[Argument, ...]) -> tuple[int, int, int, int]:
    """The grid and block krn's own launch of `args` uses, so the hand-written kernel matches it."""
    ordered = launcher.ordered(*args)
    return launcher.geometry(launcher.plan(ordered), ordered)[0]


def mlx_queue[Out: mx.array](op: Callable[[], Out]) -> float:
    """MLX's wall time per op over a queue of ops evaluated together, median over repeats."""
    samples = []
    for trial in range(MLX_REPEATS + 1):
        started = time.perf_counter_ns()
        mx.eval(*[op() for _ in range(MEASURED)])
        mx.synchronize()
        if trial:
            samples.append((time.perf_counter_ns() - started) / MEASURED)
    return statistics.median(samples)


def report_capabilities(device: MetalDevice) -> None:
    caps = Capabilities.probe(device.core)
    log.info(
        "device %s; counter sets %s; sampling %s; capture %s",
        caps.chip,
        dict(caps.counter_sets),
        sorted(caps.sampling),
        caps.capture,
    )
    log.info("clock offset (GPU minus CPU) %d ns", clock_offset_ns(device.core))
    for signal in Signal:
        try:
            require(signal, caps)
        except TargetError as error:
            log.info("%s: %s", signal, error)
        else:
            log.info("%s: available", signal)


def summarize(
    name: str,
    geometry: tuple[int, int, int, int],
    krn: list[int],
    hand: list[int],
    gaps: list[int],
    *,
    traffic_bytes: int,
    flops: int,
    mlx_ns: float | None,
    correct: bool,
    krn_source: str,
) -> Case:
    deciles = statistics.quantiles(krn, n=10)
    return Case(
        name=name,
        geometry=geometry,
        krn_ns=statistics.median(krn),
        krn_p10_ns=deciles[0],
        krn_p90_ns=deciles[-1],
        hand_ns=statistics.median(hand),
        mlx_ns=mlx_ns,
        counter_gap_ns=statistics.median(gaps),
        traffic_bytes=traffic_bytes,
        flops=flops,
        correct=correct,
        krn_source=krn_source,
    )


def saxpy_case(device: MetalDevice, timer: StageTimer, rng: np.random.Generator) -> Case:
    n = SAXPY_SIZE
    x, y = device.empty(T.f32, n), device.empty(T.f32, n)
    x.array[:], y.array[:] = rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)
    y0 = y.array.copy()
    launcher = device.launcher(saxpy, Items(n))
    args = (T.f32(2.0), x, y)
    geometry = launch_geometry(launcher, args)
    hand = HandWritten(device, _SAXPY, "saxpy")
    buffers = (x.buffer, y.buffer, hand.constant("f", 2.0), hand.constant("I", n))
    krn_queue(device, lambda: launcher(*args), WARMUP)
    krn = krn_queue(device, lambda: launcher(*args), MEASURED)
    hand.queue(timer, buffers, geometry, WARMUP)
    hand_times, gaps = hand.queue(timer, buffers, geometry, MEASURED)
    y.array[:] = y0
    launcher(*args)
    device.default_stream.sync()
    correct = bool(np.allclose(y.array, np.float32(2.0) * x.array + y0, rtol=1e-6))
    return summarize(
        "saxpy f32",
        geometry,
        krn,
        hand_times,
        gaps,
        traffic_bytes=12 * n,
        flops=2 * n,
        mlx_ns=None,
        correct=correct,
        krn_source=launcher.plan(launcher.ordered(*args)).program.unit.source,
    )


def matmul_case(
    device: MetalDevice, timer: StageTimer, rng: np.random.Generator, dtype: type[Scalar], kernel_fn: Kernel, name: str
) -> Case:
    n = MATMUL_SIZE
    numpy_type = np.float16 if dtype is f16 else np.float32
    a, b, c, out = (device.empty(dtype, n, n) for _ in range(4))
    a.array[:] = rng.standard_normal((n, n)).astype(numpy_type)
    b.array[:] = rng.standard_normal((n, n)).astype(numpy_type)
    launcher = device.launcher(kernel_fn)
    args = (Tiling(), a, b, c)
    geometry = launch_geometry(launcher, args)
    if dtype is f16:
        source = _MATMUL.format(T="half", S="half", PA=40, PB=72)
    else:
        source = _MATMUL.format(T="float", S="float", PA=32, PB=64)
    hand = HandWritten(device, source, "matmul")
    buffers = (a.buffer, b.buffer, out.buffer, hand.constant("II", n, n))
    krn_queue(device, lambda: launcher(*args), WARMUP)
    krn = krn_queue(device, lambda: launcher(*args), MEASURED)
    hand.queue(timer, buffers, geometry, WARMUP)
    hand_times, gaps = hand.queue(timer, buffers, geometry, MEASURED)
    launcher(*args)
    device.default_stream.sync()
    want = a.array[:TILE].astype(np.float32) @ b.array.astype(np.float32)
    correct = bool(
        np.allclose(c.array[:TILE].astype(np.float32), want, rtol=2e-2, atol=2e-2)
        and np.allclose(out.array.astype(np.float32), c.array.astype(np.float32), rtol=2e-2, atol=2e-2)
    )
    lhs, rhs = mx.array(a.array), mx.array(b.array)
    return summarize(
        name,
        geometry,
        krn,
        hand_times,
        gaps,
        traffic_bytes=0,
        flops=2 * n * n * n,
        mlx_ns=mlx_queue(lambda: mx.matmul(lhs, rhs)),
        correct=correct,
        krn_source=launcher.plan(launcher.ordered(*args)).program.unit.source,
    )


def softmax_case(device: MetalDevice, timer: StageTimer, rng: np.random.Generator) -> Case:
    rows, cols = DEVICE_ROWS, DEVICE_COLS
    x, y = device.empty(T.f32, rows, cols), device.empty(T.f32, rows, cols)
    x.array[:] = rng.standard_normal((rows, cols)).astype(np.float32) * 4
    launcher = device.launcher(softmax)
    args = (RowTiling(), x, y)
    geometry = launch_geometry(launcher, args)
    hand = HandWritten(device, _SOFTMAX, "softmax")
    buffers = (x.buffer, y.buffer, hand.constant("I", cols))
    krn_queue(device, lambda: launcher(*args), WARMUP)
    krn = krn_queue(device, lambda: launcher(*args), MEASURED)
    hand.queue(timer, buffers, geometry, WARMUP)
    hand_times, gaps = hand.queue(timer, buffers, geometry, MEASURED)
    launcher(*args)
    device.default_stream.sync()
    e = np.exp(x.array - x.array.max(axis=1, keepdims=True))
    correct = bool(np.allclose(y.array, e / e.sum(axis=1, keepdims=True), atol=1e-5))
    lhs = mx.array(x.array)
    return summarize(
        "softmax f32",
        geometry,
        krn,
        hand_times,
        gaps,
        traffic_bytes=12 * rows * cols,
        flops=0,
        mlx_ns=mlx_queue(lambda: mx.softmax(lhs, axis=-1)),
        correct=correct,
        krn_source=launcher.plan(launcher.ordered(*args)).program.unit.source,
    )


@app.default
def main(*, out: Path = RESULTS / "zero_cost_apple.json") -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = MetalDevice(Apple.of("apple_m4"))
    report_capabilities(device)
    timer = StageTimer(device.core, capacity=2 * (WARMUP + MEASURED) + 8)
    rng = np.random.default_rng(0)
    cases = [
        saxpy_case(device, timer, rng),
        matmul_case(device, timer, rng, f16, matmul_f16, "matmul f16"),
        matmul_case(device, timer, rng, f32, matmul_f32, "matmul f32"),
        softmax_case(device, timer, rng),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([case.record() for case in cases], indent=2))
    log.info(
        "%-12s %9s %15s %9s %9s %8s %8s %9s %9s %7s",
        "case",
        "krn us",
        "krn p10-p90",
        "hand us",
        "mlx us",
        "k/hand",
        "k/mlx",
        "krn GB/s",
        "krn GF/s",
        "correct",
    )
    for case in cases:
        log.info(
            "%-12s %9.1f %15s %9.1f %9s %8.3f %8s %9.1f %9.1f %7s",
            case.name,
            case.krn_ns / 1e3,
            f"{case.krn_p10_ns / 1e3:.0f}-{case.krn_p90_ns / 1e3:.0f}",
            case.hand_ns / 1e3,
            "-" if case.mlx_ns is None else f"{case.mlx_ns / 1e3:.1f}",
            case.krn_ns / case.hand_ns,
            "-" if case.mlx_ns is None else f"{case.krn_ns / case.mlx_ns:.3f}",
            case.traffic_bytes / case.krn_ns,
            case.flops / case.krn_ns,
            case.correct,
        )
    log.info("wrote %s", out)


if __name__ == "__main__":
    app()
