"""The CUDA backend against numpy: launches, streams, events, bound launches, graphs, programs and the discovery of the
backend by `host.Device.current()`.

Every test skips without the cuda-python wheels and a card. Every wait carries a timeout, so a lost wakeup fails a test
instead of hanging the run.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, Cache, Items, T, Vector, grid, kernel, targets
from krn.arithmetic import ceildiv
from krn.compiler.api import capture
from krn.compiler.dtypes import I32, U32
from krn.compiler.errors import TargetError
from krn.data.sparse import Csr, spmv
from krn.data.sparse_build import csr_arrays
from krn.frontend.types import u32
from krn.runtime.compile_options import STRICT_CUDA, CompileOptions
from krn.runtime.host import Settings
from krn.runtime.runtime import Argument
from tests.conftest import KERNELS
from tests.helpers.example_gemm import GemmTiling, gemm
from tests.helpers.example_kernels import Tiling, add, saxpy
from tests.helpers.fake_nvrtc import ADVERSARIAL, CONFLICTS, SOURCE, record_bindings, recording_nvidia

if TYPE_CHECKING:
    from krn.compiler.performance import Compilation
    from krn.runtime.cuda_backend import CUDADevice, CUDATensor

type Compiles = tuple[Callable[[str, Sequence[str]], Compilation], list[tuple[str, ...]]]
TIMEOUT = 60
RNG = np.random.default_rng(3)


@kernel
def fill(value: T.f32, y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < y.shape[0]:
        y[i] = value


@kernel
def twice(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = x[i] + x[i]


def host_floats(n: int) -> np.ndarray:
    return RNG.standard_normal(n).astype(np.float32)


@pytest.mark.parametrize("n", [1, 255, 1000, 100_003])
def test_saxpy_launched_by_the_launcher_matches_numpy(gpu: CUDADevice, n: int) -> None:
    x, y = host_floats(n), host_floats(n)
    dx, dy = gpu.to_device(x), gpu.to_device(y)
    gpu.launcher(saxpy, Items(n))(T.f32(2.5), dx, dy)
    np.testing.assert_allclose(dy.to_host(), np.float32(2.5) * x + y, rtol=1e-5, atol=1e-6)


def test_struct_consts_pick_the_program_and_tile_kernels_get_cooperating_threads(
    gpu: CUDADevice,
) -> None:
    shape = (100, 70)
    a, b = (
        RNG.standard_normal(shape).astype(np.float32),
        RNG.standard_normal(shape).astype(np.float32),
    )
    c = gpu.to_device(np.zeros(shape, np.float32))
    for tile in (32, 64):
        gpu.launcher(add, Blocks(ceildiv(shape[1], tile), ceildiv(shape[0], tile)))(
            gpu.to_device(a), gpu.to_device(b), c, Tiling(rows=tile, cols=tile)
        )
        np.testing.assert_array_equal(c.to_host(), a + b)
        c = gpu.to_device(np.zeros(shape, np.float32))


@pytest.mark.parametrize("tiles", [(16, 16, 8), (64, 64, 16), (128, 64, 32)], ids=str)
def test_the_fp16_gemm_matches_numpy_and_reserves_shared_memory_over_48_kib(
    gpu: CUDADevice, tiles: tuple[int, int, int]
) -> None:
    (bm, bn, bk), (m, n, k) = tiles, (150, 90, 70)
    a, b = (
        RNG.standard_normal((m, k)).astype(np.float16),
        RNG.standard_normal((k, n)).astype(np.float16),
    )
    c = gpu.to_device(np.zeros((m, n), np.float32))
    gpu.launcher(gemm, Blocks(ceildiv(n, bn), ceildiv(m, bm)))(
        gpu.to_device(a), gpu.to_device(b), c, GemmTiling(bm=bm, bn=bn, bk=bk)
    )
    np.testing.assert_allclose(c.to_host(), a.astype(np.float32) @ b.astype(np.float32), rtol=1e-3, atol=1e-3)


def test_a_struct_with_tensor_fields_is_passed_by_value(gpu: CUDADevice) -> None:
    dense = np.where(RNG.random((37, 29)) < 0.2, RNG.standard_normal((37, 29)), 0).astype(np.float32)
    arrays, x = csr_arrays(dense), host_floats(29)
    matrix = Csr(
        values=gpu.to_device(arrays.values),
        rows=u32(arrays.rows),
        columns=u32(arrays.columns),
        indptr=gpu.to_device(arrays.indptr),
        indices=gpu.to_device(arrays.indices),
    )
    out = gpu.to_device(np.zeros(37, np.float32))
    gpu.launcher(spmv, Items(37))(matrix, gpu.to_device(x), out)
    np.testing.assert_allclose(out.to_host(), dense @ x, rtol=1e-5, atol=1e-5)


def test_bound_launches_encode_once_and_rebind_in_place(gpu: CUDADevice) -> None:
    x, y = host_floats(1000), host_floats(1000)
    dx, dy = gpu.to_device(x), gpu.to_device(y)
    bound = gpu.launcher(saxpy, Items(1000)).bind(T.f32(0.5), dx, dy)
    for _ in range(3):
        bound()
    np.testing.assert_allclose(dy.to_host(), y + 3 * np.float32(0.5) * x, rtol=1e-5, atol=1e-5)
    other = gpu.to_device(y)
    bound.rebind(T.f32(1.0), dx, other)
    bound.submit().wait(TIMEOUT)
    np.testing.assert_allclose(other.to_host(), y + x, rtol=1e-5, atol=1e-5)


def call_untyped(launcher: Callable[..., None], /, *args: Argument) -> None:
    """Call a launcher the way the checkers would reject: wrong counts and types must still fail at runtime."""
    launcher(*args)


def test_wrong_arguments_fail_on_the_calling_thread(gpu: CUDADevice) -> None:
    x = gpu.to_device(host_floats(4))
    launcher = gpu.launcher(saxpy, Items(4))
    with pytest.raises(TypeError, match="takes 3 arguments"):
        call_untyped(launcher, 1.0, x, x, x)
    with pytest.raises(TypeError, match="is missing the argument 'y'"):
        call_untyped(launcher, 1.0, x)
    with pytest.raises(TypeError, match="expected a CUDA tensor, got ndarray"):
        call_untyped(launcher, 1.0, x, np.zeros(4, np.float32))
    with pytest.raises(TypeError, match=r"must be a .* tensor of rank 1"):
        call_untyped(launcher, 1.0, x, gpu.to_device(np.zeros((2, 2), np.float32)))


def test_events_order_streams_and_measure_the_device_clock(gpu: CUDADevice) -> None:
    n = 1 << 22
    first, second = gpu.stream(), gpu.stream()
    with first:
        y = gpu.empty(T.f32, n)
        begin = first.record()
        gpu.launcher(fill, Items(n))(T.f32(3.0), y)
        done = first.record()
    second.wait(done)
    with second:
        z = gpu.empty(T.f32, n)
        gpu.launcher(twice, Items(n))(y, z)
    finished = second.record()
    finished.wait(TIMEOUT)
    assert done.done() and finished.done()
    np.testing.assert_array_equal(z.to_host(second), np.full(n, 6.0, np.float32))
    assert 0 < done.elapsed_since(begin) < finished.elapsed_since(begin)


def test_streams_and_tensors_hand_their_handles_to_libraries(gpu: CUDADevice) -> None:
    stream, array = gpu.stream(), host_floats(12).reshape(3, 4)
    assert stream.handle != 0 and stream.__cuda_stream__() == (0, stream.handle)
    tensor = gpu.to_device(array)
    interface = tensor.__cuda_array_interface__
    assert (
        interface["typestr"] == "<f4" and interface["shape"] == (3, 4) and interface["data"] == (tensor.pointer, False)
    )
    assert tensor.strides == (16, 4) and tensor.nbytes == 48
    np.testing.assert_array_equal(tensor.to_host(), array)


def test_awaiting_an_event_and_the_device_suspends_only_the_task(gpu: CUDADevice) -> None:
    n = 1 << 20

    async def main() -> np.ndarray:
        y = gpu.empty(T.f32, n)
        await gpu.launcher(fill, Items(n)).submit(T.f32(7.0), y)
        await gpu.synchronize()
        return y.to_host()

    result = asyncio.run(asyncio.wait_for(main(), TIMEOUT))
    np.testing.assert_array_equal(result, np.full(n, 7.0, np.float32))


def test_a_thread_that_did_not_build_the_device_launches_and_times(gpu: CUDADevice) -> None:
    """The tuner of `lib` does this: another thread, a private stream, two events, an await, `elapsed_since`."""
    n, found = 1 << 20, {}

    async def timed() -> float:
        stream = gpu.stream()
        with stream:
            y = gpu.empty(T.f32, n)
            launcher = gpu.launcher(fill, Items(n))
            begin = stream.record()
            for _ in range(10):
                launcher(T.f32(1.0), y)
            end = stream.record()
        await end
        return end.elapsed_since(begin)

    def work() -> None:
        try:
            found["ms"] = asyncio.run(timed())
        except Exception as error:  # the top of a thread: the test reads what happened
            found["error"] = error

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(TIMEOUT)
    assert "error" not in found, found
    assert found["ms"] > 0


def test_a_program_compiles_loads_and_launches_on_pool_and_fresh_threads(gpu: CUDADevice) -> None:
    """Each thread binds the device's context before its first driver call, whichever thread compiles the program."""
    n = 1 << 12
    x, pooled, awaited = gpu.to_device(host_floats(n)), gpu.empty(T.f32, n), gpu.empty(T.f32, n)
    launcher = gpu.launcher(twice, Items(n))

    async def in_a_task() -> None:
        bound = await asyncio.to_thread(launcher.bind, x, awaited)
        await asyncio.to_thread(bound)
        await gpu.synchronize()

    with ThreadPoolExecutor(max_workers=1) as pool:
        bound = pool.submit(launcher.bind, x, pooled).result(TIMEOUT)
        pool.submit(bound).result(TIMEOUT)
    asyncio.run(asyncio.wait_for(in_a_task(), TIMEOUT))
    expected = 2 * x.to_host()
    np.testing.assert_array_equal(pooled.to_host(), expected)
    np.testing.assert_array_equal(awaited.to_host(), expected)


def test_programs_are_cached_by_content_and_persist_in_the_cache_directory(
    gpu: CUDADevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ir = capture(
        KERNELS / "intrinsics.py",
        "histogram",
        dtypes={"values": I32, "bins": I32},
        target=gpu.nvidia,
    )
    assert gpu.program(ir) is gpu.program(replace(ir))
    cached = type(gpu)(gpu.nvidia, replace(Settings.from_environ(), cache=Cache(dir=tmp_path)))
    first = cached.program(ir)
    assert {path.suffix for path in (tmp_path / "cuda").iterdir()} == {".cubin", ".json"}
    compiled = []
    monkeypatch.setattr("krn.runtime.cuda_backend.compile_result", lambda *args: compiled.append(args))
    again = type(gpu)(gpu.nvidia, replace(Settings.from_environ(), cache=Cache(dir=tmp_path))).program(ir)
    assert not compiled and again.digest == first.digest


def test_a_grid_barrier_is_launched_cooperatively_and_a_grid_that_cannot_be_resident_fails(
    gpu: CUDADevice,
) -> None:
    ir = capture(KERNELS / "groups.py", "grid_barrier", dtypes={"x": U32, "out": U32}, target=gpu.nvidia)
    program = gpu.program(ir)
    assert program.cooperative
    count = 4 * 64
    x, out = gpu.to_device(np.zeros(count, np.uint32)), gpu.to_device(np.zeros(count, np.uint32))
    program.call(program.geometry(Blocks(4, threads=64)), [x, out]).enqueue(gpu.default_stream)
    np.testing.assert_array_equal(out.to_host(), ((np.arange(count) + 64) % count).astype(np.uint32))
    huge = 1 << 20
    x, out = gpu.empty(T.u32, huge * 64), gpu.empty(T.u32, huge * 64)
    with pytest.raises(RuntimeError, match="COOPERATIVE_LAUNCH_TOO_LARGE"):
        program.call(program.geometry(Blocks(huge, threads=64)), [x, out]).enqueue(gpu.default_stream)


def test_a_graph_replays_what_it_recorded_and_refuses_other_arguments(
    gpu: CUDADevice,
) -> None:
    n = 4096
    x, y, z = (gpu.empty(T.f32, n) for _ in range(3))

    def pipeline(a: CUDATensor, b: CUDATensor, c: CUDATensor) -> None:
        gpu.launcher(fill, Items(n))(T.f32(2.0), a)
        gpu.launcher(twice, Items(n))(a, b)
        gpu.launcher(twice, Items(n))(b, c)

    graph = gpu.graph(pipeline, x, y, z)  # cold: the programs are compiled and loaded while the stream captures
    assert not gpu.default_stream.capturing
    graph(x, y, z)
    gpu.default_stream.sync()
    np.testing.assert_array_equal(z.to_host(), np.full(n, 8.0, np.float32))
    other = gpu.empty(T.f32, n)
    with pytest.raises(TargetError, match="graph replay"):
        graph(x, y, other)
    graph.submit(x, y, z).wait(TIMEOUT)


def test_a_function_that_fails_while_recording_leaves_no_capture_behind(gpu: CUDADevice) -> None:
    def broken() -> None:
        raise ValueError("while recording")

    with pytest.raises(ValueError, match="while recording"):
        gpu.graph(broken)
    x = host_floats(8)
    np.testing.assert_array_equal(gpu.to_device(x).to_host(), x)  # a capture left open would refuse this thread


SCRIPT = """
import json
import numpy as np
from krn import configure, host, lib

configure(**{"lib.mode": "ours", "lib.tune": "off"})
device = host.Device.current()
rng = np.random.default_rng(1)
result = {"device": type(device).__name__, "arch": device.target.arch, "error": {}}
for name, dtype in (("f32", np.float32), ("f16", np.float16)):
    a, b = rng.standard_normal((50, 70)).astype(dtype), rng.standard_normal((70, 30)).astype(dtype)
    out = lib.matmul(device.to_device(a), device.to_device(b)).to_host()
    result["error"][name] = float(np.abs(out.astype(np.float32) - a.astype(np.float32) @ b.astype(np.float32)).max())
print(json.dumps(result))
"""


def test_the_backend_is_discovered_and_lib_multiplies_through_it(gpu: CUDADevice) -> None:
    """A process with nothing set finds the card: `host.Device.current()` imports the backend and `lib.matmul` runs our
    kernel on its stream. The test process pins a CPU target, so this one starts clean."""
    environment = {k: v for k, v in os.environ.items() if k != "KRN_TARGET"}
    environment |= {
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=environment,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    found = json.loads(run.stdout.splitlines()[-1])
    assert (found["device"], found["arch"]) == ("CUDADevice", gpu.target.arch)
    assert found["error"]["f32"] < 1e-3 and found["error"]["f16"] < 5e-2


@pytest.fixture
def compiles(monkeypatch: pytest.MonkeyPatch) -> Compiles:
    """The runtime's `compile_result` with NVRTC replaced by a recorder, and the list of each compile's options."""
    backend = pytest.importorskip("krn.runtime.cuda_backend", reason="cuda-core and cuda-bindings are not installed")
    return backend.compile_result, record_bindings(monkeypatch, backend)


def test_compile_result_with_no_flags_still_compiles_the_strict_set(
    compiles: Compiles,
) -> None:
    compile_result, received = compiles
    compile_result(SOURCE, ())
    assert received == [("-default-device", *STRICT_CUDA, "--ptxas-options=-v")]


@pytest.mark.parametrize("extra", [(), ("--fmad false", "-arch=sm_90a")], ids=["runtime", "spelled"])
def test_the_runtime_and_inspection_hand_the_compiler_the_same_options(
    compiles: Compiles,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
) -> None:
    compile_result, runtime = compiles
    nvidia, inspection = recording_nvidia(monkeypatch)
    flags = (*CompileOptions.resolve(targets.Target.of("sm_90a")).flags(), *extra)
    compile_result(SOURCE, flags)
    nvidia.compile(SOURCE, "sm_90a", "cubin", flags)
    assert runtime == [(*inspection[0], "--ptxas-options=-v")]


@pytest.mark.parametrize("option", CONFLICTS)
def test_compile_result_refuses_each_spelling_that_changes_the_strict_arithmetic(
    compiles: Compiles, option: str
) -> None:
    compile_result, received = compiles
    with pytest.raises(TargetError) as failure:
        compile_result(SOURCE, ("--gpu-architecture=sm_90a", option))
    assert failure.value.feature == option and failure.value.target == "sm_90a"
    assert received == []


def test_compile_result_names_the_compile_when_no_architecture_is_given(
    compiles: Compiles,
) -> None:
    compile_result, received = compiles
    with pytest.raises(TargetError) as failure:
        compile_result(SOURCE, ("--ftz=true",))
    assert failure.value.target == "CUDA" and received == []


def test_compile_result_writes_each_architecture_once(
    compiles: Compiles,
) -> None:
    compile_result, received = compiles
    compile_result(SOURCE, ("-arch=sm_90a", "--gpu-architecture sm_90a"))
    (flags,) = received
    assert [flag for flag in flags if "arch" in flag] == ["--gpu-architecture=sm_90a"]


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
def test_the_runtime_compile_keeps_the_multiply_and_the_add_separate(tmp_path: Path, arch: str) -> None:
    """The runtime emits a cubin, not PTX, so the check reads its SASS: a fused multiply-add would be FFMA."""
    backend = pytest.importorskip("krn.runtime.cuda_backend", reason="cuda-core and cuda-bindings are not installed")
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is not on PATH, so the runtime's SASS cannot be read")
    cubin = backend.compile_result(ADVERSARIAL, CompileOptions.resolve(targets.Target.of(arch)).flags()).cubin
    path = tmp_path / "adversarial.cubin"
    path.write_bytes(cubin)
    run = subprocess.run(
        [cuobjdump, "--dump-sass", str(path)], capture_output=True, text=True, check=True, timeout=TIMEOUT
    )
    assert "FFMA" not in run.stdout and "FMUL" in run.stdout and "FADD" in run.stdout
