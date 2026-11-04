"""The MSL lowering run on this Mac's GPU, compiled by the system from source and compared with the emulator.

They skip where PyObjC or a Metal device is missing. The kernels are those the CUDA card runs in `test_gpu.py`, minus
the ones the hardware lacks (`block.count`), and the tiles are sized for 32 KB of threadgroup memory.
"""

import asyncio
import runpy

import numpy as np
import pytest

from krn import Items, T, Tiles, Vector, host, items, kernel, math, targets
from krn.compiler.api import capture
from krn.compiler.errors import CaptureError
from krn.frontend.geometry import Geometry
from krn.runtime.cpu import Value
from tests.compiler.test_denormals import flushed, measure
from tests.compiler.test_validity import CASES as RAGGED
from tests.conftest import KERNELS, ROOT, SKETCH
from tests.emulator.test_emulate import CASES
from tests.helpers.devices import FILE, MetalRun, agree, compare, cooperative

metal = pytest.importorskip("krn.runtime.metal", reason="PyObjC and the Metal framework are macOS only")
RNG = np.random.default_rng(17)


@pytest.fixture(scope="module")
def gpu() -> metal.MetalDevice:
    try:
        return metal.MetalDevice(targets.Apple.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))


def twice(
    gpu: metal.MetalDevice, name: str, launch: Geometry, arguments: list[Value], file=FILE
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`compare` on the Metal GPU."""
    return compare(gpu.target, lambda kernel: MetalRun(gpu, kernel), name, launch, arguments, file)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_the_gpu_agrees_with_the_emulator(gpu: metal.MetalDevice, case) -> None:
    arguments = list(case.make())
    count = len(arguments[0]) if case.name != "take_lock" else 64
    if case.name == "smallest":  # a signed 64-bit atomic min: Metal has only unsigned min and max on 64 bits
        with pytest.raises(CaptureError, match="no 64-bit atomics"):
            twice(gpu, case.name, Items(count), arguments)
        return
    oracle, actual = twice(gpu, case.name, Items(count), arguments)
    agree(oracle, actual, case.outputs)


def test_the_gpu_flushes_exactly_the_float32_subnormals_the_target_table_names(gpu: metal.MetalDevice) -> None:
    """Re-measures the M4 Pro row of `facts/targets.py`: each operation's run against the emulator, on subnormals."""
    found = measure(gpu.target, lambda kernel: MetalRun(gpu, kernel))
    assert flushed(found) == gpu.target.denormal_flushes
    assert found["copy"].mode == "preserved"


@pytest.mark.parametrize("limit", [0, 3, 7])
def test_a_wrapping_counter_ends_where_the_emulator_does(gpu: metal.MetalDevice, limit: int) -> None:
    oracle, actual = twice(
        gpu,
        "wrapping_counter",
        Items(20),
        [limit, np.zeros(1, np.uint32), np.zeros(20, np.uint32)],
    )
    assert actual[0][0] == oracle[0][0]
    assert sorted(actual[1]) == sorted(oracle[1])  # the order of arrival differs, the multiset of old values cannot


def test_a_simdgroup_shuffle_sums_each_simdgroup(gpu: metal.MetalDevice) -> None:
    x = RNG.standard_normal(256).astype(np.float32)
    oracle, actual = twice(gpu, "warp_sum", Items(256), [x, np.zeros(8, np.float32)])
    agree(oracle, actual, (1,))
    assert np.allclose(actual[1], x.reshape(8, 32).sum(axis=1), rtol=1e-5)


def test_ballot_and_vote_see_the_whole_simdgroup(gpu: metal.MetalDevice) -> None:
    x = np.where(np.arange(256) % 3 == 0, 1, -1).astype(np.int32)
    x[64:96] = 5
    oracle, actual = twice(gpu, "votes", Items(256), [x, np.zeros(8, np.uint64), np.zeros(8, np.int32)])
    agree(oracle, actual, (1, 2))
    assert actual[2].tolist() == [int(w == 2) for w in range(8)]


def test_a_fence_and_a_barrier_order_the_writes(gpu: metal.MetalDevice) -> None:
    oracle, actual = twice(gpu, "ordered", Items(256), [np.zeros(256, np.float32), np.zeros(1, np.int32)])
    agree(oracle, actual)
    assert actual[0].tolist() == [1.0] * 256 and actual[1].tolist() == [1]


def test_a_tile_kernel_covers_every_column_with_a_cooperative_block(gpu: metal.MetalDevice) -> None:
    x = RNG.standard_normal((20, 40)).astype(np.float32)
    oracle, actual = twice(
        gpu,
        "tiles_wide",
        cooperative(Tiles(x.shape, (8, 16))),
        [x, np.full((20, 40), 99.0, np.float32)],
    )
    agree(oracle, actual, (1,))
    assert np.allclose(actual[1], x + 1.0)


def test_the_fp16_gemm_runs_its_cooperative_loops_on_the_gpu(gpu: metal.MetalDevice) -> None:
    a = RNG.standard_normal((20, 24)).astype(np.float16)
    b = RNG.standard_normal((24, 17)).astype(np.float16)
    c = np.zeros((20, 17), np.float32)
    kernel = capture(
        SKETCH / "gemm.py",
        "gemm",
        consts={"t": {"bm": 16, "bn": 16, "bk": 8}},
        target=gpu.target,
    )
    MetalRun(gpu, kernel).run(cooperative(Tiles(c.shape, (16, 16))), [a, b, c, {}])
    assert np.allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("case", RAGGED, ids=lambda c: c.name)
def test_ragged_tiles_skip_their_invalid_lanes_on_the_gpu(gpu: metal.MetalDevice, case) -> None:
    arguments = case.arguments()
    oracle, actual = twice(gpu, case.name, cooperative(case.launch), arguments, KERNELS / "ragged.py")
    agree(oracle, actual)
    for index, want in case.expected(arguments):
        assert np.allclose(actual[index], want, rtol=2e-5, atol=1e-6), case.name


def test_host_launches_events_bound_graphs_and_shared_storage(gpu: metal.MetalDevice, monkeypatch) -> None:
    """Run the host API on the M4 Pro, including the README's array allocation and await pattern."""
    monkeypatch.setattr(host.Device, "current", classmethod(lambda cls: gpu))
    x = gpu.to_device(np.arange(513, dtype=np.float32))
    y = gpu.empty(T.f32, 513)
    launch = gpu.launcher(square_root)
    launch.bind(x, y).submit().wait(60)
    np.testing.assert_allclose(y.to_host(), np.sqrt(x.array))
    stream = gpu.stream()
    with stream:
        start = stream.record()
        graph = gpu.graph(square_root, x, y)
        finished = graph.submit(x, y)
    gpu.default_stream.wait(finished)
    gpu.launcher(square_root)(y, x)

    async def finish() -> None:
        await finished
        await gpu.synchronize()

    asyncio.run(finish())
    assert finished.elapsed_since(start) >= 0
    np.testing.assert_allclose(x.to_host(), np.sqrt(np.sqrt(np.arange(513, dtype=np.float32))), rtol=1e-5)


@kernel
def square_root(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        y[i] = math.sqrt(x[i])


def test_readme_on_metal(gpu: metal.MetalDevice, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(host.Device, "current", classmethod(lambda cls: gpu))
    source = (ROOT / "README.md").read_text().split("```python\n", 1)[1].split("```", 1)[0]
    path = tmp_path / "readme.py"
    path.write_text(source)
    runpy.run_path(str(path))
