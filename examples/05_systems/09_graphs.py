"""Record a sequence of launches once and replay it with one call: a graph on CUDA, HIP, Metal and CPUs

A launch costs the host a few microseconds, and for a pipeline of many small kernels that overhead is the run time.
Each vendor can record a sequence and replay it as a unit, and krn puts one typed interface over them: a CUDA or HIP
graph from stream capture, a Metal indirect command buffer, and on a CPU a prebuilt task DAG whose dependencies come
from the tensors each node reads and writes. `Graph[P]` is typed by the function it records, so a replay with a
wrong argument is a static error; replaying returns at once and `submit` gives an awaitable event.
`run.graphs=False` records nothing and replays plain launches, which bisects a suspected graph bug.

Level: 5
Topics: graphs, replay, typed recording, CPU task DAG
Needs: 05_systems/07_cpu_backend
"""

import asyncio
import logging
from collections.abc import Sequence

import numpy as np

from krn import Const, Matrix, T, Vector, block, host, kernel, pipelined, struct, zeros
from krn.compiler.api import CPUDevice, CPUTensor

log = logging.getLogger("graphs")


@struct
class Tiling:
    bm: Const[int] = 64
    bn: Const[int] = 64
    bk: Const[int] = 32


@kernel
def gemm(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def bias_gelu(t: Tiling, h: Matrix[T.f32], bias: Vector[T.f32], y: Matrix[T.f16]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    v = h[rows, cols].add(bias[cols])
    y[rows, cols] = v.div(v.mul(-1.702).exp().add(1.0)).to(T.f16)


@kernel
def row_softmax(t: Tiling, y: Matrix[T.f16], p: Matrix[T.f16]) -> None:
    rows, cols = block.tile(t.bm), slice(0, t.bn)
    p[rows, cols] = y[rows, cols].to(T.f32).softmax().to(T.f16)


def layer(
    x: Matrix[T.f16], w: Matrix[T.f16], h: Matrix[T.f32], y: Matrix[T.f16], p: Matrix[T.f16], bias: Vector[T.f32]
) -> None:
    """The recorded program: three launches that a plain loop would pay for separately."""
    t = Tiling()
    gemm[t](x, w, h)
    bias_gelu[t](h, bias, y)
    row_softmax[t](y, p)


async def serve(
    batches: Sequence[CPUTensor[T.f16, int, int]], w: CPUTensor[T.f16, int, int], bias: CPUTensor[T.f32, int]
) -> CPUTensor[T.f16, int, int]:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    h = device.empty(T.f32, batches[0].shape[0], w.shape[1])
    y = device.empty(T.f16, batches[0].shape[0], w.shape[1])
    p = device.empty(T.f16, batches[0].shape[0], w.shape[1])
    graph = device.graph(layer, batches[0], w, h, y, p, bias)  # records once, here
    for x in batches[1:]:
        graph(x, w, h, y, p, bias)  # replays: only the pointer of x changes
    await graph.submit(batches[0], w, h, y, p, bias)  # one awaitable for the whole sequence
    return p


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    batches = [device.empty(T.f16, 70, 96) for _ in range(4)]
    w, bias = device.empty(T.f16, 96, 60), device.empty(T.f32, 60)
    for tensor in (*batches, w, bias):
        tensor.array[:] = rng.standard_normal(tensor.array.shape)
    probabilities = await serve(batches, w, bias)  # the last replay is the submitted one: batches[0]
    h = batches[0].array.astype(np.float32) @ w.array.astype(np.float32) + bias.array
    gelu = h / (1 + np.exp(-1.702 * h))
    want = np.exp(gelu - gelu.max(axis=1, keepdims=True))
    np.testing.assert_allclose(
        probabilities.array.astype(np.float32), want / want.sum(axis=1, keepdims=True), atol=2e-2
    )
    log.info("one recording and %d replays ran; the last output agrees with numpy", len(batches))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
