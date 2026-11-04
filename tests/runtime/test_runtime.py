"""The CPU runtime against numpy: launches, streams, events, bound launches, graphs and concurrent builds.

Every wait carries a timeout, so a lost wakeup fails a test instead of hanging the run.
"""

import asyncio
import itertools
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from krn import Blocks, Items, Run, T, Vector, grid, host, kernel, targets
from krn.arithmetic import ceildiv
from krn.compiler.api import CPUDevice, CPUEvent, CPUProgram, CPUStream, CPUTensor, Kernel
from krn.compiler.emit import Unit
from krn.profile import Profiler
from krn.runtime import cpu, runtime
from krn.runtime.compile_options import CompileOptions
from krn.runtime.runtime import Argument, Task
from tests.helpers.example_gemm import GemmTiling, gemm
from tests.helpers.example_kernels import Tiling, add, saxpy
from tests.helpers.example_softmax import RowTiling, softmax

TIMEOUT = 60


@kernel
def fill(value: T.f32, y: Vector[T.f32]) -> None:
    """A store-only kernel: the last launch in stream order decides what `y` holds."""
    i = grid.thread_rank()
    if i < y.shape[0]:
        y[i] = value


@kernel
def twice(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = x[i] + x[i]


@pytest.fixture(scope="session")
def device() -> CPUDevice:
    current = host.Device.current()
    assert isinstance(current, CPUDevice)
    return current


def vector(device: CPUDevice, n: int, seed: int = 0) -> CPUTensor[T.f32, int]:
    tensor = device.empty(T.f32, n)
    tensor.array[:] = np.random.default_rng(seed).standard_normal(n, dtype=np.float32)
    return tensor


def matrix[D: (T.f16, T.f32)](
    device: CPUDevice, dtype: type[D], rows: int, cols: int, seed: int = 0
) -> CPUTensor[D, int, int]:
    tensor = device.empty(dtype, rows, cols)
    tensor.array[:] = np.random.default_rng(seed).standard_normal((rows, cols))
    return tensor


def settle(*streams: CPUStream) -> None:
    for stream in streams:
        stream.record().wait(TIMEOUT)


def test_the_cpu_backend_is_discovered_and_is_the_default_device(device: CPUDevice) -> None:
    assert runtime.BACKEND.device(device.target) is device
    assert host.Device.current() is device
    assert host.default_device() is host.default_device()  # one device per target, not one per call
    assert device.launcher(saxpy, Items(8)) is host.Device.current().launcher(saxpy, Items(8))


@pytest.mark.parametrize("n", [1, 255, 1000, 100_003])
def test_saxpy_matches_numpy(device: CPUDevice, n: int) -> None:
    x, y = vector(device, n, 1), vector(device, n, 2)
    expected = np.float32(2.5) * x.array + y.array
    saxpy(T.f32(2.5), x, y, launch=Items(n))
    settle(device.default_stream)
    np.testing.assert_allclose(y.array, expected, rtol=1e-5, atol=1e-6)


def test_to_device_copies_what_the_caller_keeps_changing(device: CPUDevice) -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = device.to_device(source)
    source[:] = -1
    np.testing.assert_array_equal(tensor.array, np.arange(6, dtype=np.float32).reshape(2, 3))
    saxpy(
        T.f32(2.0), device.to_device(np.ones(6, np.float32)), vector(device, 6), launch=Items(6)
    )  # a copy is a tensor


@pytest.mark.parametrize(("shape", "tile"), [((100, 70), 32), ((5, 3), 128), ((64, 64), 32)])
def test_tile_add_with_struct_consts_matches_numpy(device: CPUDevice, shape: tuple[int, int], tile: int) -> None:
    a, b = matrix(device, T.f32, *shape, seed=1), matrix(device, T.f32, *shape, seed=2)
    c = matrix(device, T.f32, *shape, seed=3)
    add(
        a,
        b,
        c,
        Tiling(rows=tile, cols=tile),
        launch=Blocks(ceildiv(shape[1], tile), ceildiv(shape[0], tile)),
    )
    settle(device.default_stream)
    np.testing.assert_array_equal(c.array, a.array + b.array)


@pytest.mark.parametrize(
    ("m", "n", "k", "tiles"),
    [(70, 50, 45, (32, 32, 16)), (33, 17, 5, (16, 16, 8)), (64, 64, 64, (128, 128, 32))],
)
def test_fp16_gemm_matches_numpy(device: CPUDevice, m: int, n: int, k: int, tiles: tuple[int, int, int]) -> None:
    a, b, c = (
        matrix(device, T.f16, m, k, 1),
        matrix(device, T.f16, k, n, 2),
        matrix(device, T.f32, m, n, 3),
    )
    bm, bn, bk = tiles
    gemm(a, b, c, GemmTiling(bm=bm, bn=bn, bk=bk), launch=Blocks(ceildiv(n, bn), ceildiv(m, bm)))
    settle(device.default_stream)
    np.testing.assert_allclose(c.array, a.array.astype(np.float32) @ b.array.astype(np.float32), rtol=1e-4, atol=1e-4)


def test_softmax_matches_numpy(device: CPUDevice) -> None:
    rows, cols = 7, 64
    x, y = matrix(device, T.f16, rows, cols, 1), matrix(device, T.f16, rows, cols, 2)
    softmax(x, y, RowTiling(rows=4, cols=cols), launch=Blocks(ceildiv(rows, 4)))
    settle(device.default_stream)
    v = x.array.astype(np.float32)
    e = np.exp(v - v.max(axis=1, keepdims=True))
    np.testing.assert_allclose(y.array.astype(np.float32), (e / e.sum(axis=1, keepdims=True)), rtol=2e-3, atol=1e-4)


def test_struct_consts_pick_the_program_and_equal_signatures_share_it(device: CPUDevice) -> None:
    a, b, c = (matrix(device, T.f32, 64, 64, seed) for seed in range(3))
    launcher = device.launcher(add, Blocks(2, 2))
    small, large, again = (
        Tiling(rows=32, cols=32),
        Tiling(rows=64, cols=64),
        Tiling(rows=32, cols=32),
    )
    assert launcher.program(a, b, c, small) is launcher.program(a, b, c, again)
    assert launcher.program(a, b, c, small) is not launcher.program(a, b, c, large)


def call_untyped(launcher: Callable[..., None], /, *args: Argument, **kwargs: Argument) -> None:
    """Call a launcher the way the checkers would reject: wrong counts and dtypes must still fail at runtime."""
    launcher(*args, **kwargs)


def test_wrong_arguments_fail_on_the_calling_thread(device: CPUDevice) -> None:
    x, y = vector(device, 4), vector(device, 4)
    launcher = host.Device.current().launcher(saxpy, Items(4))
    with pytest.raises(TypeError, match="takes 3 arguments"):
        call_untyped(launcher, 1.0, x, y, y)
    with pytest.raises(TypeError, match="is missing the argument 'y'"):
        call_untyped(launcher, 1.0, x)
    with pytest.raises(TypeError, match="is missing the argument 'a'"):
        call_untyped(launcher, x=x, y=y)
    wrong = CPUTensor[T.f32, int](np.zeros(4, dtype=np.float64))
    with pytest.raises(TypeError, match="C-contiguous f32 array of rank 1"):
        launcher(T.f32(1.0), wrong, y)


def test_keyword_arguments_are_placed_by_parameter_name(device: CPUDevice) -> None:
    x, y = vector(device, 100, 1), vector(device, 100, 2)
    expected = np.float32(3.0) * x.array + y.array
    saxpy(y=y, a=T.f32(3.0), x=x, launch=Items(100))
    settle(device.default_stream)
    np.testing.assert_allclose(y.array, expected, rtol=1e-5)


def test_a_tensor_knows_its_dtype(device: CPUDevice) -> None:
    for dtype in (T.f16, T.f32, T.f64, T.u8, T.i32):
        assert device.empty(dtype, 3, 2).dtype == dtype
    with pytest.raises(TypeError, match="not complex64"):
        CPUTensor[T.f32, int](np.zeros(3, dtype=np.complex64))


def test_dtype_and_contiguity_of_tensors() -> None:
    assert runtime.numpy_dtype(T.f16) == np.float16
    assert runtime.numpy_dtype(T.u8) == np.uint8
    with pytest.raises(TypeError, match="no host representation"):
        runtime.numpy_dtype(T.bf16)
    with pytest.raises(ValueError, match="C-contiguous"):
        CPUTensor[T.f32, int, int](np.zeros((4, 4), dtype=np.float32)[:, ::2])
    wrapped = np.arange(6, dtype=np.float32)
    assert CPUTensor[T.f32, int](wrapped).array is wrapped
    assert np.asarray(CPUTensor[T.f32, int](wrapped)) is wrapped


def test_launches_on_one_stream_run_in_order(device: CPUDevice) -> None:
    n, rounds = 4096, 50
    y = device.empty(T.f32, n)
    seen = [device.empty(T.f32, n) for _ in range(rounds)]
    stream = device.stream()
    with stream:
        for k in range(rounds):
            fill(T.f32(float(k)), y, launch=Items(n))
            twice(y, seen[k], launch=Items(n))
    settle(stream)
    for k, tensor in enumerate(seen):
        assert (tensor.array == 2 * k).all(), f"round {k} ran out of order"


class Meet(Task):
    """Occupies a worker until another lane's task arrives too, which only parallel lanes can do."""

    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier

    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        self.barrier.wait(TIMEOUT)
        return True


class Boom(Task):
    def run(self, lane: CPUStream, session: Profiler | None) -> bool:
        raise RuntimeError("boom")


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs two workers")
def test_different_streams_run_in_parallel() -> None:
    device = CPUDevice(targets.CPU.of("avx512"), host.Settings(run=Run(workers=2)))
    try:
        barrier = threading.Barrier(2)
        first, second = device.stream(), device.stream()
        first.enqueue(Meet(barrier))
        second.enqueue(Meet(barrier))
        settle(first, second)  # a broken barrier would surface as an error on the marker
    finally:
        device.close()


def test_one_worker_serves_streams_that_wait_on_each_other_and_a_graph() -> None:
    """A waiting lane parks without holding a worker, so a single worker is enough for any chain of waits."""
    device = CPUDevice(targets.CPU.of("avx512"), host.Settings(run=Run(workers=1)))
    try:
        n = 1000
        x, y, z = (device.empty(T.f32, n) for _ in range(3))
        x.array[:], y.array[:], z.array[:] = 1, 0, 0
        launch = device.launcher(saxpy, Items(n))
        gate, first, second = CPUEvent(), device.stream(), device.stream()
        first.wait(gate)
        with first:
            launch(T.f32(2.0), x, y)
            produced = first.record()
        second.wait(produced)
        with second:
            launch(T.f32(1.0), y, z)
            consumed = second.record()
        gate.resolve()
        consumed.wait(TIMEOUT)
        assert (y.array == 2).all()
        assert (z.array == 2).all()

        def five(x: CPUTensor[T.f32, int], y: CPUTensor[T.f32, int]) -> None:
            for _ in range(5):
                launch(T.f32(1.0), x, y)

        graph = device.graph(five, x, y)
        asyncio.run(graph.submit(x, y))
        assert (y.array == 7).all()
    finally:
        device.close()


def test_a_failed_task_is_reported_by_the_next_event_and_the_lane_goes_on(
    device: CPUDevice,
) -> None:
    stream = device.stream()
    stream.enqueue(Boom())
    with pytest.raises(RuntimeError, match="boom"):
        stream.record().wait(TIMEOUT)
    stream.record().wait(TIMEOUT)

    async def awaited() -> None:
        stream.enqueue(Boom())
        await stream.record()

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(awaited())


def test_events_complete_in_a_taskgroup_of_tasks_with_separate_streams(device: CPUDevice) -> None:
    tasks, n = 6, 5000
    outputs = [device.empty(T.f32, n) for _ in range(tasks)]
    used: list[CPUStream] = []

    async def work(k: int) -> None:
        async with device.stream() as stream:
            await asyncio.sleep(0)  # the other tasks run now, and this task's current stream must survive them
            assert host.current_stream.get() is stream
            used.append(stream)
            await host.Device.current().launcher(fill, Items(n)).submit(T.f32(float(k)), outputs[k])
            await asyncio.sleep(0)
            assert host.current_stream.get() is stream

    async def main() -> None:
        async with asyncio.TaskGroup() as group:
            for k in range(tasks):
                group.create_task(work(k))
        assert host.current_stream.get() is None

    asyncio.run(main())
    assert len({id(stream) for stream in used}) == tasks
    for k, tensor in enumerate(outputs):
        assert (tensor.array == k).all()


def test_an_event_that_is_already_done_can_be_awaited_and_done_never_blocks(
    device: CPUDevice,
) -> None:
    stream = device.stream()
    event = stream.record()
    event.wait(TIMEOUT)
    assert event.done()

    async def main() -> None:
        await event
        await event

    asyncio.run(main())


def test_synchronize_waits_for_every_stream(device: CPUDevice) -> None:
    n = 1 << 20
    outputs, streams = (
        [device.empty(T.f32, n) for _ in range(3)],
        [device.stream() for _ in range(3)],
    )
    for k, (stream, tensor) in enumerate(zip(streams, outputs, strict=True)):
        with stream:
            fill(T.f32(float(k + 1)), tensor, launch=Items(n))
    asyncio.run(device.synchronize())
    for k, tensor in enumerate(outputs):
        assert (tensor.array == k + 1).all()


def test_elapsed_since_measures_the_work_between_two_markers(device: CPUDevice) -> None:
    n = 1 << 22
    x, y = vector(device, n, 1), vector(device, n, 2)
    stream = device.stream()
    with stream:
        start = stream.record()
        for _ in range(4):
            saxpy(T.f32(1.0), x, y, launch=Items(n))
        end = stream.record()
    started = time.perf_counter()
    end.wait(TIMEOUT)
    waited_ms = 1e3 * (time.perf_counter() - started)
    elapsed = end.elapsed_since(start)
    assert 0 < elapsed <= waited_ms + 50
    with pytest.raises(RuntimeError, match="completed"):
        CPUEvent().elapsed_since(start)


def test_a_stream_waits_for_another_without_blocking_the_host(device: CPUDevice) -> None:
    n = 1000
    gate = CPUEvent()
    first, second = device.stream(), device.stream()
    x, y, z = vector(device, n, 1), vector(device, n, 2), vector(device, n, 3)
    expected_y = np.float32(2.0) * x.array + y.array
    expected_z = np.float32(1.0) * expected_y + z.array
    first.wait(gate)  # returns at once although nothing can pass this point yet
    with first:
        saxpy(T.f32(2.0), x, y, launch=Items(n))
        produced = first.record()
    second.wait(produced)
    with second:
        saxpy(T.f32(1.0), y, z, launch=Items(n))
        consumed = second.record()
    time.sleep(0.05)
    assert not produced.done()
    assert not consumed.done()
    gate.resolve()
    consumed.wait(TIMEOUT)
    np.testing.assert_allclose(z.array, expected_z, rtol=1e-5)


def test_concurrent_producers_on_one_stream_lose_no_launch(device: CPUDevice) -> None:
    n, producers, launches = 64, 8, 300
    ones, total = device.empty(T.f32, n), device.empty(T.f32, n)
    ones.array[:] = 1
    total.array[:] = 0
    stream = device.stream()
    barrier = threading.Barrier(producers)

    def produce() -> None:
        barrier.wait(TIMEOUT)
        with stream:
            for _ in range(launches):
                saxpy(T.f32(1.0), ones, total, launch=Items(n))

    with ThreadPoolExecutor(producers) as pool:
        for future in [pool.submit(produce) for _ in range(producers)]:
            future.result(TIMEOUT)
    settle(stream)
    assert (total.array == producers * launches).all()


def test_bound_launches_marshal_once_and_rebind_leaves_queued_launches_alone(
    device: CPUDevice,
) -> None:
    n = 500
    x, y, x2, y2 = (vector(device, n, seed) for seed in range(4))
    expected, expected2 = (
        np.float32(2.0) * x.array + y.array,
        np.float32(3.0) * x2.array + y2.array,
    )
    bound = host.Device.current().launcher(saxpy, Items(n)).bind(T.f32(2.0), x, y)
    stream, gate = device.stream(), CPUEvent()
    stream.wait(gate)  # holds the lane, so the first call is still queued when the arguments change
    with stream:
        bound()
        bound.rebind(T.f32(3.0), x2, y2)
        bound.submit()
    gate.resolve()
    settle(stream)
    np.testing.assert_allclose(y.array, expected, rtol=1e-5)
    np.testing.assert_allclose(y2.array, expected2, rtol=1e-5)


def test_a_bound_launch_equals_the_launcher_call_repeated(device: CPUDevice) -> None:
    n, repeats = 2000, 5
    x, y, y_bound = vector(device, n, 1), vector(device, n, 2), vector(device, n, 2)
    bound = host.Device.current().launcher(saxpy, Items(n)).bind(T.f32(0.5), x, y_bound)
    for _ in range(repeats):
        saxpy(T.f32(0.5), x, y, launch=Items(n))
        bound()
    settle(device.default_stream)
    expected = vector(device, n, 2).array.copy()
    for _ in range(repeats):
        expected = np.float32(0.5) * x.array + expected
    np.testing.assert_allclose(y.array, expected, rtol=1e-5)
    np.testing.assert_array_equal(y_bound.array, y.array)


class Operands(NamedTuple):
    """The seven tensors of `pipeline`: gemm reads a and b into c, add reads c and d into e, softmax reads x into y."""

    a: CPUTensor[T.f16, int, int]
    b: CPUTensor[T.f16, int, int]
    c: CPUTensor[T.f32, int, int]
    d: CPUTensor[T.f32, int, int]
    e: CPUTensor[T.f32, int, int]
    x: CPUTensor[T.f16, int, int]
    y: CPUTensor[T.f16, int, int]

    @classmethod
    def random(cls, device: CPUDevice, seed: int) -> Operands:
        m, n, k = 70, 50, 45
        made = cls(
            matrix(device, T.f16, m, k, seed),
            matrix(device, T.f16, k, n, seed + 1),
            matrix(device, T.f32, m, n, seed + 2),
            matrix(device, T.f32, m, n, seed + 3),
            matrix(device, T.f32, m, n, seed + 4),
            matrix(device, T.f16, 8, 64, seed + 5),
            matrix(device, T.f16, 8, 64, seed + 6),
        )
        for output in (made.c, made.e, made.y):
            output.array[:] = -7
        return made

    @property
    def outputs(self) -> list[np.ndarray]:
        return [self.c.array, self.e.array, self.y.array]


def pipeline(
    a: CPUTensor[T.f16, int, int],
    b: CPUTensor[T.f16, int, int],
    c: CPUTensor[T.f32, int, int],
    d: CPUTensor[T.f32, int, int],
    e: CPUTensor[T.f32, int, int],
    x: CPUTensor[T.f16, int, int],
    y: CPUTensor[T.f16, int, int],
) -> None:
    """Three kernels: gemm into c, then add reads c (a dependency), and softmax shares nothing with them."""
    gemm(a, b, c, GemmTiling(bm=32, bn=32, bk=16), launch=Blocks(2, 3))
    add(c, d, e, Tiling(rows=32, cols=32), launch=Blocks(2, 3))
    softmax(x, y, RowTiling(rows=4, cols=64), launch=Blocks(2))


def test_graph_replay_equals_eager_and_survives_new_arguments(device: CPUDevice) -> None:
    first, eager = Operands.random(device, 10), Operands.random(device, 10)
    pipeline(*eager)
    settle(device.default_stream)

    graph = device.graph(pipeline, *first)
    settle(device.default_stream)
    assert all((output == -7).all() for output in first.outputs), "recording must not launch anything"
    assert graph.edges == {(0, 1)}
    assert graph.roots == (0, 2)
    graph(*first)
    settle(device.default_stream)
    for got, want in zip(first.outputs, eager.outputs, strict=True):
        np.testing.assert_array_equal(got, want)

    second, eager_second = Operands.random(device, 20), Operands.random(device, 20)
    pipeline(*eager_second)
    settle(device.default_stream)
    snapshot = [output.copy() for output in first.outputs]
    graph(*second)
    settle(device.default_stream)
    for got, want in zip(second.outputs, eager_second.outputs, strict=True):
        np.testing.assert_array_equal(got, want)
    for kept, output in zip(snapshot, first.outputs, strict=True):
        np.testing.assert_array_equal(output, kept)  # the second replay did not touch the first set

    first.e.array[:] = -7
    graph(*first)  # back to the first set, re-encoded once more
    settle(device.default_stream)
    np.testing.assert_array_equal(first.e.array, eager.e.array)


def test_graph_dependencies_come_from_reads_and_writes(device: CPUDevice) -> None:
    n = 3000
    x, y, z, w = (vector(device, n, seed) for seed in range(4))
    y0, z0, w0 = y.array.copy(), z.array.copy(), w.array.copy()

    def chain(
        x: CPUTensor[T.f32, int],
        y: CPUTensor[T.f32, int],
        z: CPUTensor[T.f32, int],
        w: CPUTensor[T.f32, int],
    ) -> None:
        saxpy(T.f32(1.0), x, y, launch=Items(n))  # 0 writes y
        saxpy(T.f32(2.0), x, y, launch=Items(n))  # 1 writes y again: write after write on 0
        saxpy(T.f32(3.0), y, z, launch=Items(n))  # 2 reads y: read after write on 1, writes z
        saxpy(T.f32(4.0), x, y, launch=Items(n))  # 3 writes y that 2 reads: write after read on 2
        saxpy(T.f32(5.0), x, w, launch=Items(n))  # 4 touches y and z in no way: a second root

    graph = device.graph(chain, x, y, z, w)
    assert graph.edges == {(0, 1), (1, 2), (2, 3)}, "edges implied by another path are dropped"
    assert graph.roots == (0, 4)
    assert graph.dag.chains == ((0, 1, 2, 3), (4,)), "a call with one predecessor and no sibling joins its chain"
    graph(x, y, z, w)
    settle(device.default_stream)
    y_mid = y0 + np.float32(1.0) * x.array + np.float32(2.0) * x.array
    np.testing.assert_allclose(z.array, z0 + np.float32(3.0) * y_mid, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(y.array, y_mid + np.float32(4.0) * x.array, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(w.array, w0 + np.float32(5.0) * x.array, rtol=1e-4, atol=1e-5)


def test_a_diamond_graph_joins_exactly_once_per_replay(device: CPUDevice) -> None:
    shape, tiling = (40, 40), Tiling(rows=16, cols=16)
    a, b, y, z, w, v = (matrix(device, T.f32, *shape, seed) for seed in range(6))

    def diamond(
        a: CPUTensor[T.f32, int, int],
        b: CPUTensor[T.f32, int, int],
        y: CPUTensor[T.f32, int, int],
        z: CPUTensor[T.f32, int, int],
        w: CPUTensor[T.f32, int, int],
        v: CPUTensor[T.f32, int, int],
    ) -> None:
        add(a, b, y, tiling, launch=Blocks(3, 3))  # 0
        add(y, b, z, tiling, launch=Blocks(3, 3))  # 1 and 2 both read y
        add(y, a, w, tiling, launch=Blocks(3, 3))  # 2
        add(z, w, v, tiling, launch=Blocks(3, 3))  # 3 joins 1 and 2

    graph = device.graph(diamond, a, b, y, z, w, v)
    assert graph.edges == {(0, 1), (0, 2), (1, 3), (2, 3)}
    assert graph.dag.chains == ((0,), (1,), (2,), (3,))
    expected = (a.array + b.array + b.array) + (a.array + b.array + a.array)
    for _ in range(100):
        v.array[:] = 0
        graph(a, b, y, z, w, v)
        settle(device.default_stream)
        np.testing.assert_array_equal(v.array, expected)


def test_a_graph_swaps_struct_arguments_by_identity(device: CPUDevice) -> None:
    shape = (40, 40)
    a, b, c = (matrix(device, T.f32, *shape, seed) for seed in range(3))
    small, large = Tiling(rows=16, cols=16), Tiling(rows=32, cols=32)

    def one(
        a: CPUTensor[T.f32, int, int],
        b: CPUTensor[T.f32, int, int],
        c: CPUTensor[T.f32, int, int],
        tiling: Tiling,
    ) -> None:
        add(a, b, c, tiling, launch=Blocks(3, 3))

    graph = device.graph(one, a, b, c, small)
    programs = {id(call.program) for call in graph.calls}
    graph(a, b, c, large)  # the 32 by 32 program on the same grid still covers the 40 by 40 matrix
    settle(device.default_stream)
    np.testing.assert_array_equal(c.array, a.array + b.array)
    assert id(graph._binding.calls[0].program) not in programs, "the swapped struct selected another program"


def test_a_graph_can_be_replayed_on_two_streams_at_once(device: CPUDevice) -> None:
    first, second = Operands.random(device, 30), Operands.random(device, 40)
    first_eager, second_eager = Operands.random(device, 30), Operands.random(device, 40)
    pipeline(*first_eager)
    pipeline(*second_eager)
    settle(device.default_stream)
    graph = device.graph(pipeline, *first)
    left, right = device.stream(), device.stream()
    with left:
        graph(*first)
    with right:
        graph(*second)
    settle(left, right)
    for got, want in zip(first.outputs + second.outputs, first_eager.outputs + second_eager.outputs, strict=True):
        np.testing.assert_array_equal(got, want)


def test_a_graph_submit_is_awaitable(device: CPUDevice) -> None:
    operands = Operands.random(device, 50)
    eager = Operands.random(device, 50)
    pipeline(*eager)
    settle(device.default_stream)
    graph = device.graph(pipeline, *operands)

    async def main() -> None:
        await graph.submit(*operands)

    asyncio.run(main())
    for got, want in zip(operands.outputs, eager.outputs, strict=True):
        np.testing.assert_array_equal(got, want)


def test_a_launch_cannot_be_awaited_while_a_graph_records(device: CPUDevice) -> None:
    x, y = vector(device, 8), vector(device, 8)

    def awaits(x: CPUTensor[T.f32, int], y: CPUTensor[T.f32, int]) -> None:
        host.Device.current().launcher(saxpy, Items(8)).submit(T.f32(1.0), x, y)

    with pytest.raises(RuntimeError, match="cannot be awaited while a graph records"):
        device.graph(awaits, x, y)
    saxpy(T.f32(1.0), x, y, launch=Items(8))  # the failed recording left the thread in eager mode
    settle(device.default_stream)


def test_a_graph_replays_as_plain_launches_when_graphs_are_off() -> None:
    device = CPUDevice(targets.CPU.of("avx512"), host.Settings(run=Run(graphs=False, workers=2)))
    try:
        n = 100
        x, y = device.empty(T.f32, n), device.empty(T.f32, n)
        x.array[:] = 1
        y.array[:] = 0
        launch = device.launcher(saxpy, Items(n))

        def twice(x: CPUTensor[T.f32, int], y: CPUTensor[T.f32, int]) -> None:
            launch(T.f32(1.0), x, y)
            launch(T.f32(1.0), x, y)

        graph = device.graph(twice, x, y)
        assert graph.calls == ()
        settle(device.default_stream)
        assert (y.array == 0).all()
        graph(x, y)
        graph.submit(x, y).wait(TIMEOUT)
        assert (y.array == 4).all()
    finally:
        device.close()


def test_eight_threads_build_and_launch_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eight threads race for eight cold programs: one clang build per program, and no result differs from numpy."""
    monkeypatch.setattr(cpu, "CACHE", tmp_path)  # a cold on-disk cache, so every program really compiles
    builds: list[str] = []
    real_program = runtime.CPUProgram

    def counting_program(
        ir: Kernel,
        unit: Unit,
        target: targets.CPU | None = None,
        cache_dir: Path | None = None,
        options: CompileOptions | None = None,
    ) -> CPUProgram:
        builds.append(ir.name)
        return real_program(ir, unit, target, cache_dir, options)

    monkeypatch.setattr(runtime, "CPUProgram", counting_program)
    device = CPUDevice(targets.CPU.of("avx512"), host.Settings(run=Run(workers=4)))
    tilings = [Tiling(rows=rows, cols=cols) for rows, cols in itertools.product((8, 16), (8, 16, 32, 64))]
    threads, rounds, shape = 8, 3, (40, 70)
    barrier = threading.Barrier(threads)

    def hammer(index: int) -> None:
        rng = np.random.default_rng(index)
        stream = device.stream()
        barrier.wait(TIMEOUT)
        for _ in range(rounds):
            for tiling in tilings[index:] + tilings[:index]:  # every thread meets every cold program at the same time
                a = CPUTensor[T.f32, int, int](rng.standard_normal(shape, dtype=np.float32))
                b = CPUTensor[T.f32, int, int](rng.standard_normal(shape, dtype=np.float32))
                c = CPUTensor[T.f32, int, int](np.zeros(shape, dtype=np.float32))
                with stream:
                    grid = Blocks(ceildiv(shape[1], tiling.cols), ceildiv(shape[0], tiling.rows))
                    device.launcher(add, grid)(a, b, c, tiling)  # `add[grid]` would go to the default device
                stream.record().wait(TIMEOUT)
                np.testing.assert_array_equal(c.array, a.array + b.array)

    try:
        with ThreadPoolExecutor(threads) as pool:
            for future in [pool.submit(hammer, index) for index in range(threads)]:
                future.result(TIMEOUT)
    finally:
        device.close()
    assert len(builds) == len(tilings)
    if os.environ.get("PYTHON_GIL") == "0":
        assert not sys._is_gil_enabled(), "something imported during the run switched the GIL back on"
