"""The vendor adapters against numpy: Accelerate through the dispatcher on the CPU backend, and Metal Performance
Shaders on this Mac's GPU (`multiply` on Metal buffers: the Metal runtime backend does not exist yet).

They skip where the framework is missing; the cuBLASLt test runs on the NVIDIA host (`mb run --on silo test`)."""

import ctypes
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from enum import IntEnum
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import hypothesis
import numpy as np
import pytest
from hypothesis import settings
from hypothesis import strategies as st

from krn import T, configure, lib
from krn.compiler.api import CPUDevice
from krn.data import sparse_build
from krn.data.sparse import Bsr, Coo, Csr, Ell
from krn.facts.targets import Target
from krn.frontend.memory import Tensor
from krn.frontend.numerics import FloatingPoint, name_of
from krn.frontend.types import u32
from krn.hw import system
from krn.lib import lib_cublaslt, lib_cusparse, lib_host, lib_nvidia, lib_ours
from krn.lib.lib import Compute, Determinism, Epilogue, NoCandidate
from krn.runtime import api as runtime

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice


class Status(IntEnum):
    SUCCESS = 0
    INVALID_VALUE = 7
    NOT_SUPPORTED = 15


class VendorError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"vendor status {status}")


class DescAttr(IntEnum):
    EPILOGUE = 1
    BIAS_DATA_TYPE = 2
    BIAS_POINTER = 3
    SM_COUNT_TARGET = 4


class PrefAttr(IntEnum):
    MAX_WORKSPACE_BYTES = 1
    MIN_ALIGNMENT_A_BYTES = 2
    MIN_ALIGNMENT_B_BYTES = 3
    MIN_ALIGNMENT_C_BYTES = 4
    MIN_ALIGNMENT_D_BYTES = 5
    REDUCTION_SCHEME_MASK = 6


RNG = np.random.default_rng(31)
M, N, K = 33, 29, 37


def use(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    configure(**{"lib.mode": mode})
    configure(**{"lib.tune": "off"})
    lib._dispatcher.cache_clear()


def test_accelerate_multiplies_float32_when_a_vendor_is_asked_for(accelerate, device: CPUDevice, monkeypatch) -> None:
    use(monkeypatch, "vendor")
    a, b = device.empty(T.f32, M, K), device.empty(T.f32, K, N)
    a.array[:], b.array[:] = RNG.standard_normal((M, K)), RNG.standard_normal((K, N))
    problem = lib.Problem(lib.Op.MATMUL, (T.f32,) * 3, (M, N, K), lib.LibOptions.of(device.target))
    assert lib.dispatcher().choose(lib.MatmulCandidate, problem, capturing=False).name == "accelerate"
    out = lib.matmul(a, b)
    np.testing.assert_allclose(np.asarray(out), a.array @ b.array, rtol=1e-5, atol=1e-5)
    lib._dispatcher.cache_clear()


def test_a_vendor_that_cannot_run_the_problem_says_why_instead_of_falling_back(
    accelerate, device: CPUDevice, monkeypatch
) -> None:
    use(monkeypatch, "vendor")
    silu = lib.Problem(lib.Op.MATMUL, (T.f32,) * 3, (M, N, K), lib.LibOptions.of(device.target), Epilogue.SILU)
    with pytest.raises(NoCandidate, match=r"accelerate: accelerate has no silu epilogue"):
        lib.dispatcher().choose(lib.MatmulCandidate, silu, capturing=False)
    half = lib.Problem(lib.Op.MATMUL, (T.f16,) * 3, (M, N, K), lib.LibOptions.of(device.target))
    with pytest.raises(NoCandidate, match="cblas_sgemm takes f32"):
        lib.dispatcher().choose(lib.MatmulCandidate, half, capturing=False)
    lib._dispatcher.cache_clear()


@pytest.mark.parametrize(("dtype", "numpy_type", "tolerance"), [(T.f32, np.float32, 1e-4), (T.f16, np.float16, 3e-2)])
def test_mps_multiplies_metal_buffers(dtype, numpy_type, tolerance: float) -> None:
    mps = pytest.importorskip("krn.lib.lib_mps")
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        gpu = metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))
    a = RNG.standard_normal((M, K)).astype(numpy_type)
    b = RNG.standard_normal((K, N)).astype(numpy_type)
    shared = metal.Metal.MTLResourceStorageModeShared
    buffers = tuple(gpu.core.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, shared) for x in (a, b))
    out = gpu.core.newBufferWithBytes_length_options_(bytes(M * N * a.itemsize), M * N * a.itemsize, shared)
    command = gpu.default_stream.queue.commandBuffer()
    mps.multiply(gpu.core, command, (*buffers, out), (M, N, K), dtype)
    command.commit()
    command.waitUntilCompleted()
    got = np.frombuffer(out.contents().as_buffer(M * N * a.itemsize), numpy_type).reshape(M, N)
    np.testing.assert_allclose(
        got.astype(np.float32),
        a.astype(np.float32) @ b.astype(np.float32),
        rtol=tolerance,
        atol=tolerance,
    )


PROFILE = settings(database=None, deadline=None, max_examples=25, derandomize=True)
NVIDIA = Target.of("sm_120a")
TRIPLES = (
    (T.f16, T.f16, T.f16),
    (T.bf16, T.bf16, T.bf16),
    (T.f16, T.f16, T.f32),
    (T.bf16, T.bf16, T.f32),
    (T.f32, T.f32, T.f32),
    (T.f64, T.f64, T.f64),
)
NUMPY_TRIPLES = tuple(triple for triple in TRIPLES if T.bf16 not in triple)
EPILOGUES = tuple(epilogue for epilogue in Epilogue if epilogue != Epilogue.SILU)
NUMPY_TYPES: dict[type[FloatingPoint], type[np.floating]] = {
    T.f16: np.float16,
    T.f32: np.float32,
    T.f64: np.float64,
}


def problem(
    triple: tuple[type[FloatingPoint], ...] = (T.f32,) * 3,
    shape: tuple[int, ...] = (M, N, K),
    epilogue: Epilogue = Epilogue.NONE,
    options: lib.LibOptions | None = None,
) -> lib.Problem:
    return lib.Problem(lib.Op.MATMUL, triple, shape, options or lib.LibOptions.of(NVIDIA), epilogue)


@pytest.mark.parametrize("triple", TRIPLES)
@pytest.mark.parametrize("determinism", tuple(Determinism))
def test_cublaslt_documented_triples(triple, determinism: Determinism) -> None:
    assert (
        lib_cublaslt.CublasLt().refusal(problem(triple, options=lib.LibOptions(determinism=determinism)), NVIDIA)
        is None
    )


@pytest.mark.parametrize("triple", [(T.f16, T.f32, T.f32), (T.f32, T.f32, T.f16), (T.f64, T.f64, T.f32)])
def test_cublaslt_undocumented_triples(triple) -> None:
    reason = lib_cublaslt.CublasLt().refusal(problem(triple), NVIDIA)
    documented = ", ".join(f"({', '.join(name_of(dtype) for dtype in entry)})" for entry in TRIPLES)
    assert reason == f"cuBLASLt documents (A, B, D) = {documented} only; got ({', '.join(map(name_of, triple))})"


@pytest.mark.parametrize("dtype", [T.f8e4m3fn, T.f8e5m2])
def test_cublaslt_fp8_requires_transposed_b(dtype) -> None:
    reason = lib_cublaslt.CublasLt().refusal(problem((dtype, dtype, T.f16)), NVIDIA)
    assert reason is not None
    assert name_of(dtype) in reason and "requires TN" in reason and "transposed b stored as (n, k)" in reason


def test_cublaslt_refuses_silu_with_the_reason() -> None:
    assert lib_cublaslt.CublasLt().refusal(problem(epilogue=Epilogue.SILU), NVIDIA) == "silu has no cuBLASLt epilogue"


@pytest.mark.parametrize("arch", ["avx512", "gfx942"])
def test_cublaslt_requires_nvidia(arch: str) -> None:
    candidate = lib_cublaslt.CublasLt()
    reason = f"cuBLASLt multiplies NVIDIA device memory, and {arch} is not NVIDIA"
    assert candidate.unavailable(Target.of(arch)) == reason
    assert candidate.refusal(problem(), Target.of(arch)) == reason


@pytest.mark.parametrize(
    ("compute", "triple", "accepted"),
    [
        (Compute.TF32, (T.f16,) * 3, False),
        (Compute.TF32, (T.f32,) * 3, True),
        (Compute.EMULATED, (T.f16,) * 3, False),
        (Compute.EMULATED, (T.f32,) * 3, True),
        (Compute.EMULATED, (T.f64,) * 3, True),
        (Compute.TF32, (T.f64,) * 3, False),
    ],
)
def test_cublaslt_compute_options(compute: Compute, triple, accepted: bool) -> None:
    reason = lib_cublaslt.CublasLt().refusal(replace(problem(triple), precision=compute), NVIDIA)
    assert (reason is None) == accepted
    if not accepted:
        assert reason is not None and f"compute {compute}" in reason


def test_cublaslt_missing_wheel_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> int:
        raise ImportError("nvmath-python[cu13] is missing")

    monkeypatch.setattr(lib_cublaslt, "version", missing)
    lib_cublaslt._wheel_unavailable.cache_clear()
    try:
        candidate = lib_cublaslt.CublasLt()
        assert candidate.unavailable(NVIDIA) == "nvmath-python[cu13] is missing"
        assert candidate.library_version() == "unavailable"
    finally:
        lib_cublaslt._wheel_unavailable.cache_clear()


def test_cublaslt_cannot_capture_before_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib_cublaslt, "_sessions", lib_nvidia.PerThread(lib_cublaslt.Session))
    key = lib_cublaslt.plan_key(problem(), (256, 512, 1024, None))
    assert key.alignments == (256, 256, 256, 0)
    assert key == lib_cublaslt.plan_key(problem(), (512, 1024, 2048, None))
    assert key != lib_cublaslt.plan_key(problem(), (258, 512, 1024, None))
    assert key != lib_cublaslt.plan_key(problem(), (256, 512, 1024, 8))
    assert not lib_cublaslt.CublasLt().capturable(problem())


def test_cublaslt_dispatch_preference(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(lib_cublaslt.CublasLt, "unavailable", lambda self, target: None)
    ours, host = lib_ours.Ours(), lib_host.Host()
    dispatcher = lib.Dispatcher(
        NVIDIA,
        {lib.Op.MATMUL: [lib_cublaslt.CublasLt(), ours, host]},
        lib.TuningDb(tmp_path / "tuning.json"),
        lib.Mode.AUTO,
        {},
        "fake-nvidia",
        tune="off",
    )
    assert dispatcher.choose(lib.MatmulCandidate, problem(epilogue=Epilogue.SILU), capturing=False).name == "ours"
    mixed = problem((T.f16, T.f16, T.f32))
    assert dispatcher.choose(lib.MatmulCandidate, mixed, capturing=False).name == "cublaslt"
    assert ours.refusal(mixed, NVIDIA) is not None
    assert host.unavailable(NVIDIA) is not None


@pytest.fixture
def lt_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Mock]]:
    """Replace the nvmath boundary while exercising the adapter on the Mac."""
    bindings = MagicMock()
    bindings.MatmulDescAttribute = DescAttr
    bindings.MatmulPreferenceAttribute = PrefAttr
    bindings.Epilogue.__getitem__.return_value = 1
    bindings.ReductionScheme.NONE = 0
    bindings.cuBLASLtError = VendorError
    bindings.MatmulHeuristicResult.return_value = Mock(ptr=4096, state=Status.SUCCESS, workspace_size=256)
    calls = {
        name: getattr(bindings, name)
        for name in (
            "create",
            "destroy",
            "matmul_desc_create",
            "matmul_desc_destroy",
            "matrix_layout_create",
            "matrix_layout_destroy",
            "matmul_preference_create",
            "matmul_preference_destroy",
            "matmul_desc_set_attribute",
            "matmul_preference_set_attribute",
            "matmul_algo_get_heuristic",
            "matmul",
        )
    }
    for name in (
        "create",
        "matmul_desc_create",
        "matrix_layout_create",
        "matmul_preference_create",
    ):
        calls[name].return_value = 16

    def heuristic(*args: int) -> None:
        # nvmath takes an integer address of the host count allocated by the adapter.
        ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_int))[0] = 1

    def preference(handle: int, attribute: int, pointer: int, size: int) -> None:
        calls["preferences"](attribute, int.from_bytes(ctypes.string_at(pointer, size), "little"))

    calls["preferences"] = Mock()
    calls["matmul_preference_set_attribute"].side_effect = preference
    calls["matmul_algo_get_heuristic"].side_effect = heuristic
    monkeypatch.setattr(lib_cublaslt, "cublasLt", bindings)
    cublas = MagicMock(Status=Status)
    cublas.ComputeType.__getitem__.return_value = 68
    monkeypatch.setattr(lib_cublaslt, "cublas", cublas)
    monkeypatch.setattr(lib_nvidia, "data_type", lambda dtype: 0)
    monkeypatch.setattr(lib_nvidia, "cu", None)
    monkeypatch.setattr(lib_cublaslt, "_sessions", lib_nvidia.PerThread(lib_cublaslt.Session))
    yield calls
    session = lib_cublaslt._sessions()
    for plan in session.plans.values():
        plan._finalizer()
    if "handle" in session.__dict__:
        session._finalizer()


def test_cublaslt_plan_reuses_heuristic_and_swaps_layouts(lt_calls: dict[str, Mock]) -> None:
    options = lib.LibOptions(determinism=Determinism.STABLE, workspace_bytes=512, sm_count=12)
    spec, pointers = (
        problem((T.f16, T.f16, T.f32), epilogue=Epilogue.BIAS, options=options),
        (256, 512, 768, 1024),
    )
    session = lib_cublaslt._sessions()
    plan = session.plan(spec, pointers)
    assert session.plan(spec, pointers) is plan
    assert lib_cublaslt.CublasLt().capturable(spec)
    layouts = lt_calls["matrix_layout_create"].call_args_list
    assert [call.args[1:] for call in layouts] == [(N, K, N), (K, M, K), (N, M, N)]
    preferences = {call.args[0]: call.args[1] for call in lt_calls["preferences"].call_args_list}
    assert preferences[PrefAttr.MAX_WORKSPACE_BYTES] == 512
    assert preferences[PrefAttr.REDUCTION_SCHEME_MASK] == 0
    plan.enqueue(session.handle, pointers, (1280, 256), 7)
    plan.enqueue(session.handle, pointers, (1280, 256), 7)
    changed = (*pointers[:3], 2048)
    plan.enqueue(session.handle, changed, (1280, 256), 7)
    assert lt_calls["matmul_algo_get_heuristic"].call_count == 1
    bias_calls = [
        call for call in lt_calls["matmul_desc_set_attribute"].call_args_list if call.args[1] == DescAttr.BIAS_POINTER
    ]
    assert len(bias_calls) == 2
    launch = lt_calls["matmul"].call_args.args
    assert (launch[3], launch[5], launch[8], launch[10], launch[13:15]) == (
        512,
        256,
        768,
        768,
        (1280, 256),
    )
    assert launch[15] == 7
    with pytest.raises(lib.Declined, match="workspace bytes"):
        plan.enqueue(session.handle, pointers, (1280, 255), 7)
    with pytest.raises(lib.Declined, match="256-byte aligned"):
        plan.enqueue(session.handle, pointers, (1281, 256), 7)
    assert lt_calls["matmul"].call_count == 3


@pytest.mark.parametrize("code", [None, Status.NOT_SUPPORTED, Status.INVALID_VALUE])
def test_cublaslt_declines_and_releases_partial_plan(lt_calls: dict[str, Mock], code: Status | None) -> None:
    lt_calls["matmul_algo_get_heuristic"].side_effect = None if code is None else VendorError(code)
    with pytest.raises(lib.Declined, match="no cuBLASLt algorithm for"):
        lib_cublaslt._sessions().plan(problem(), (256, 512, 768, None))
    assert not lib_cublaslt._sessions().plans
    assert lt_calls["matmul_desc_destroy"].call_count == 1
    assert lt_calls["matrix_layout_destroy"].call_count == 3
    assert lt_calls["matmul_preference_destroy"].call_count == 1
    lt_calls["matmul"].assert_not_called()


def test_cublaslt_run_reuses_stream_workspace_and_checks_capture_alignment(
    lt_calls: dict[str, Mock],
    device: CPUDevice,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operands = lib.Operands(device.empty(T.f32, M, K), device.empty(T.f32, K, N), device.empty(T.f32, M, N))
    pointers = {id(operands.a): 256, id(operands.b): 512, id(operands.out): 768}
    monkeypatch.setattr(lib_nvidia, "address", lambda tensor: pointers[id(tensor)])
    monkeypatch.setattr(lib_nvidia, "handle", lambda stream: 7)
    scratch = Mock()
    scratch.at_least.return_value = 1280
    factory = Mock(return_value=scratch)
    monkeypatch.setattr(lib_nvidia, "Scratch", factory)
    candidate = lib_cublaslt.CublasLt()
    assert not candidate.capturable(problem(), operands)
    for _ in range(2):
        candidate.run(problem(), operands, device.default_stream)
    factory.assert_called_once_with(device)
    assert scratch.at_least.call_count == 2
    assert lt_calls["create"].call_count == 1
    assert lt_calls["matmul_algo_get_heuristic"].call_count == 1
    assert lt_calls["matmul"].call_count == 2
    assert candidate.capturable(problem(), operands)
    pointers[id(operands.a)] = 258
    assert not candidate.capturable(problem(), operands)


def test_cublaslt_direct_call_reuses_plan_and_caller_workspace(lt_calls: dict[str, Mock]) -> None:
    for _ in range(2):
        lib_cublaslt.multiply((256, 512, 768, None), (M, N, K), T.f32, workspace=(1280, 256), stream=7)
    assert lt_calls["create"].call_count == 1
    assert lt_calls["matmul_algo_get_heuristic"].call_count == 1
    assert lt_calls["matmul"].call_count == 2
    assert lt_calls["matmul"].call_args.args[13:15] == (1280, 256)
    assert not lib_cublaslt._sessions().scratch


@pytest.fixture
def cuda_library(gpu: CUDADevice, monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[CUDADevice]:
    configure(**{"run.target": gpu.target.arch})
    configure(**{"lib.tune": "off"})
    configure(cache_dir=tmp_path)
    configure(**{"lib.matmul": None})
    for cached in (system.System.detect, lib._dispatcher):
        cached.cache_clear()
    try:
        device = runtime.default_device()
        backend = pytest.importorskip("krn.runtime.cuda_backend")
        assert isinstance(device, backend.CUDADevice)
        reason = lib_cublaslt.CublasLt().unavailable(gpu.target)
        if reason:
            pytest.skip(reason)
        yield device
    finally:
        for cached in (system.System.detect, lib._dispatcher):
            cached.cache_clear()


def reference(a: np.ndarray, b: np.ndarray, bias: np.ndarray, epilogue: Epilogue) -> tuple[np.ndarray, np.ndarray]:
    x = a.astype(np.float64) @ b.astype(np.float64)
    magnitude = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    if "bias" in epilogue.value:
        x += bias.astype(np.float64)
        magnitude += np.abs(bias.astype(np.float64))
    if "relu" in epilogue.value:
        x = np.maximum(x, 0)
    if "gelu" in epilogue.value:
        x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    if epilogue == Epilogue.SILU:
        x = x / (1 + np.exp(-x))
    return x, magnitude


def assert_bound(got: np.ndarray, want: np.ndarray, magnitude: np.ndarray, epilogue: Epilogue) -> None:
    bound = 4 * (np.finfo(got.dtype).eps / 2) * magnitude
    if "gelu" in epilogue.value:
        bound += 2e-3
    error = np.abs(got.astype(np.float64) - want)
    assert np.all(error <= bound), f"max error {error.max()}, max bound {bound.max()}"


def check_cuda(
    device: CUDADevice,
    triple: tuple[type[FloatingPoint], ...],
    shape: tuple[int, ...],
    epilogue: Epilogue,
    *,
    compare_ours: bool = False,
) -> None:
    m, n, k = shape
    rng = np.random.default_rng(31)
    a = (rng.standard_normal((m, k)) / np.sqrt(k)).astype(NUMPY_TYPES[triple[0]])
    b = rng.standard_normal((k, n)).astype(NUMPY_TYPES[triple[1]])
    bias = rng.standard_normal(n).astype(NUMPY_TYPES[triple[2]])
    da, db, dc = device.to_device(a), device.to_device(b), device.to_device(bias)
    out = device.empty(triple[2], m, n)
    spec = problem(triple, shape, epilogue, lib.LibOptions.of(device.target))
    assert lib.dispatcher().choose(lib.MatmulCandidate, spec, capturing=False).name == "cublaslt"
    lib.matmul(da, db, out=out, bias=dc if "bias" in epilogue.value else None, epilogue=epilogue)
    got = out.to_host()
    want, magnitude = reference(a, b, bias, epilogue)
    assert_bound(got, want, magnitude, epilogue)
    if compare_ours and len(set(triple)) == 1 and triple[0] in (T.f16, T.f32):
        ours = device.empty(triple[2], m, n)
        lib_ours.Ours().run(
            spec,
            lib.Operands(da, db, ours, dc if "bias" in epilogue.value else None),
            device.default_stream,
        )
        ours_host = ours.to_host()
        assert_bound(ours_host, want, magnitude, epilogue)
        bound = 8 * np.finfo(got.dtype).eps / 2 * magnitude + (4e-3 if "gelu" in epilogue.value else 0)
        assert np.all(np.abs(got.astype(np.float64) - ours_host.astype(np.float64)) <= bound)


@pytest.mark.parametrize("epilogue", EPILOGUES)
@pytest.mark.parametrize("triple", NUMPY_TRIPLES)
@pytest.mark.parametrize("shape", [(33, 29, 37), (1, 5, 3)])
def test_cublaslt_matmul(cuda_library: CUDADevice, monkeypatch, triple, shape, epilogue: Epilogue) -> None:
    use(monkeypatch, "vendor")
    check_cuda(cuda_library, triple, shape, epilogue)


def test_cublaslt_shapes(cuda_library: CUDADevice) -> None:
    with pytest.MonkeyPatch.context() as patch:
        use(patch, "vendor")

        @PROFILE
        @hypothesis.given(
            m=st.integers(1, 96),
            n=st.integers(1, 96),
            k=st.integers(1, 96),
            triple=st.sampled_from(NUMPY_TRIPLES),
            epilogue=st.sampled_from(EPILOGUES),
        )
        def check(m: int, n: int, k: int, triple, epilogue: Epilogue) -> None:
            check_cuda(cuda_library, triple, (m, n, k), epilogue, compare_ours=True)

        check()


def test_cublaslt_stable_and_graph(cuda_library: CUDADevice, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, "vendor")
    device = cuda_library
    a_host = RNG.standard_normal((M, K)).astype(np.float32)
    b_host = RNG.standard_normal((K, N)).astype(np.float32)
    a, b = device.to_device(a_host), device.to_device(b_host)
    out = device.empty(T.f32, M, N)
    options = lib.LibOptions(determinism=Determinism.STABLE)

    def multiply() -> None:
        lib.matmul(a, b, out=out, options=options)

    multiply()
    eager = out.to_host().tobytes()
    multiply()
    assert out.to_host().tobytes() == eager
    recorded = device.to_device(np.full((M, N), np.nan, np.float32))

    def record() -> None:
        lib.matmul(a, b, out=recorded, options=options)

    graph = device.graph(record)
    for _ in range(2):
        graph()
        assert recorded.to_host().tobytes() == eager


@pytest.mark.parametrize(("mode", "epilogue"), [("ours", Epilogue.NONE), ("auto", Epilogue.SILU)])
def test_cublaslt_fallback(cuda_library: CUDADevice, monkeypatch, mode: str, epilogue: Epilogue) -> None:
    use(monkeypatch, mode)
    device = cuda_library
    a = RNG.standard_normal((M, K)).astype(np.float32)
    b = RNG.standard_normal((K, N)).astype(np.float32)
    bias = np.zeros(N, np.float32)
    da, db = device.to_device(a), device.to_device(b)
    spec = problem(epilogue=epilogue, options=lib.LibOptions.of(device.target))
    assert lib.dispatcher().choose(lib.MatmulCandidate, spec, capturing=False).name == "ours"
    out = device.empty(T.f32, M, N)
    lib.matmul(da, db, out=out, epilogue=epilogue)
    want, magnitude = reference(a, b, bias, epilogue)
    assert_bound(out.to_host(), want, magnitude, epilogue)


CARD_BLOCK = 2
UNIT: dict[type[FloatingPoint], float] = {T.f16: 2.0**-11, T.f32: 2.0**-24, T.f64: 2.0**-53}
SPMM_TRIPLES = ((T.f32,) * 3, (T.f64,) * 3, (T.f16, T.f16, T.f32), (T.f16,) * 3)
SPMV_TRIPLES = (*SPMM_TRIPLES, (T.f32, T.f64, T.f64))
ARRAYS = {
    "Csr": (sparse_build.csr_arrays, Csr),
    "Coo": (sparse_build.coo_arrays, Coo),
    "Bsr": (lambda dense, dtype: sparse_build.bsr_arrays(dense, CARD_BLOCK, dtype), Bsr),
    "Ell": (sparse_build.ell_arrays, Ell),
}
CARD_LAYOUTS = ("Csr", "Coo", "Bsr")


def test_nvidia_sessions_belong_to_a_context_and_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Mock()
    driver.cuInit.return_value = (0,)
    driver.cuCtxGetCurrent.return_value = (0, 11)
    monkeypatch.setattr(lib_nvidia, "cu", driver)
    make = Mock(side_effect=[[], [], []])
    sessions = lib_nvidia.PerThread(make)
    first = sessions()
    assert sessions() is first
    driver.cuCtxGetCurrent.return_value = (0, 12)
    assert sessions() is not first
    driver.cuCtxGetCurrent.return_value = (0, 11)
    assert sessions() is first
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(sessions).result(timeout=5) is not first
    assert make.call_count == 3


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
@pytest.mark.parametrize("layout", CARD_LAYOUTS)
def test_cusparse_nvmath_plan_repoints_and_owns_stream_buffers(device, monkeypatch, operation, layout) -> None:
    api = MagicMock()
    api.SpMVAlg.__getitem__.return_value = 1
    api.SpMMAlg.__getitem__.return_value = 1
    api.IndexType.INDEX_32I = 2
    api.IndexBase.ZERO = 0
    api.Order.ROW = 2
    api.Operation.NON_TRANSPOSE = 0
    api.PointerMode.HOST = 0
    api.create.return_value = 16
    for name in ("create_csr", "create_coo", "create_bsr"):
        getattr(api, name).return_value = 32
    api.create_dn_vec.side_effect = (48, 64)
    api.create_dn_mat.side_effect = (48, 64)
    api.sp_mv_buffer_size.return_value = api.sp_mm_buffer_size.return_value = 256
    monkeypatch.setattr(lib_cusparse, "cusparse", api)
    monkeypatch.setattr(lib_nvidia, "address", id)
    monkeypatch.setattr(lib_nvidia, "handle", id)
    monkeypatch.setattr(lib_nvidia, "data_type", lambda dtype: 0)
    monkeypatch.setattr(lib_nvidia, "relaxed_capture", nullcontext)
    monkeypatch.setattr(lib_nvidia, "synchronize", lambda stream: None)
    monkeypatch.setattr(lib_nvidia, "allocate", lambda device, size: device.empty(T.u8, max(1, size)))
    matrix = on_card(device, layout, np.eye(4, dtype=np.float32), T.f32)
    spmv = operation is lib.Op.SPMV
    x = device.empty(T.f32, 4, *(() if spmv else (3,)))
    out = device.empty(T.f32, *x.shape)
    kind = lib.SpmvOperands if spmv else lib.SpmmOperands
    candidate = lib_cusparse.CusparseSpmv() if spmv else lib_cusparse.CusparseSpmm()
    spec = sparse_problem(
        device,
        operation,
        matrix,
        (T.f32,) * 3,
        x.shape,
        lib.LibOptions(determinism=Determinism.STABLE),
    )
    algorithm = candidate.algorithm_for(spec)
    session = lib_cusparse.Session()
    try:
        session.run(spec, kind(matrix, x, out), device.default_stream, algorithm)
        matrix = replace(matrix, values=device.empty(T.f32, *matrix.values.shape))
        changed = kind(matrix, device.empty(T.f32, *x.shape), device.empty(T.f32, *out.shape))
        session.run(spec, changed, device.default_stream, algorithm)
        session.run(spec, changed, device.stream(), algorithm)
        assert len(session.plans) == 1
        plan = next(iter(session.plans.values()))
        assert plan.pattern == lib_cusparse.pattern_of(matrix)
        create = getattr(api, f"create_{layout.lower()}")
        create.assert_called_once()
        api.sp_mat_set_values.assert_called_once_with(32, id(matrix.values))
        prefix = "sp_mv" if spmv else "sp_mm"
        getattr(api, f"{prefix}_buffer_size").assert_called_once()
        assert getattr(api, f"{prefix}_preprocess").call_count == (1 if layout == "Csr" else 0)
        calls = getattr(api, prefix).call_args_list
        assert len(calls) == 3 and calls[0].args[-1] == calls[1].args[-1] != calls[2].args[-1]
        assert len(plan.spare) == 1
        setter = api.dn_vec_set_values if spmv else api.dn_mat_set_values
        assert [call.args for call in setter.call_args_list] == [
            (48, id(changed.x)),
            (64, id(changed.out)),
        ]
    finally:
        for plan in session.plans.values():
            plan._finalizer()
        session._finalizer()


def struct_of(layout: str, dense: np.ndarray, dtype, convert: Callable):
    """The struct of `layout` that holds `dense`, every array and count of it passed through `convert`."""
    build, kind = ARRAYS[layout]
    return kind(**{name: convert(value) for name, value in build(dense, dtype)._asdict().items()})


def on_card(device, layout: str, dense: np.ndarray, dtype):
    return struct_of(
        layout,
        dense,
        dtype,
        lambda v: device.to_device(v) if isinstance(v, np.ndarray) else u32(v),
    )


def on_host(layout: str, dense: np.ndarray, dtype):
    """The same matrix in host arrays, which `lib_host.product` reads as triples and sums in float64."""
    return struct_of(layout, dense, dtype, lambda v: v if isinstance(v, np.ndarray) else u32(v))


@st.composite
def sparse_cases(draw, layout: str) -> tuple[np.ndarray, np.random.Generator, int]:
    """A dense matrix with zeros in it (so some rows are empty), even dimensions for the blocked layout."""
    unit = CARD_BLOCK if layout == "Bsr" else 1
    rows, columns = (unit * draw(st.integers(1, 24)) for _ in range(2))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    dense = rng.standard_normal((rows, columns)) * (
        rng.random((rows, columns)) < draw(st.sampled_from([0.05, 0.3, 1.0]))
    )
    dense[0, 0] = 1.0  # at least one stored entry
    return dense, rng, draw(st.integers(1, 9))


def host(tensor: Tensor) -> np.ndarray:
    """A CUDA tensor read back; only tests that run on a card call it."""
    backend = pytest.importorskip("krn.runtime.cuda_backend")
    assert isinstance(tensor, backend.CUDATensor)
    return tensor.to_host()


def upload(device, rng: np.random.Generator, dtype, *shape: int):
    values = rng.standard_normal(shape).astype(NUMPY_TYPES[dtype])
    return device.to_device(values), values.astype(np.float64)


def assert_within_bound(got: np.ndarray, layout: str, dense: np.ndarray, triple, x: np.ndarray, mode: str) -> None:
    """`|y - y_ref| <= c * u * sum_j |a_ij x_j|` per row against the float64 product of the rounded operands: u is the
    rounding of the output plus the accumulation of the columns, which `ours` does in the type of the output."""
    a_type, _, out_type = triple
    want = lib_host.product(on_host(layout, dense, a_type), x)
    weight = lib_host.product(on_host(layout, np.abs(dense), a_type), np.abs(x))
    accumulator = UNIT[out_type] if mode == "ours" else UNIT[T.f64] if out_type == T.f64 else UNIT[T.f32]
    bound = 2 * (UNIT[out_type] + dense.shape[1] * accumulator) * weight + 1e-30
    assert np.all(np.abs(got.astype(np.float64) - want) <= bound), (layout, triple, mode)


def sparse_problem(device, kind: lib.Op, matrix, triple, x_shape: tuple[int, ...], options=None) -> lib.Problem:
    rows, columns = int(matrix.rows), int(matrix.columns)
    shape = (rows, columns, matrix.values.shape[0], *x_shape[1:])
    return lib.Problem(
        kind,
        triple,
        shape,
        options or lib.LibOptions.of(device.target),
        layout=type(matrix).__name__,
    )


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
@pytest.mark.parametrize("layout", CARD_LAYOUTS)
def test_cusparse_equals_ours_and_the_float64_reference(cuda_library, monkeypatch, operation, layout) -> None:
    device, spmv = cuda_library, operation is lib.Op.SPMV
    kind = lib.SpmvCandidate if spmv else lib.SpmmCandidate

    @PROFILE
    @hypothesis.given(
        case=sparse_cases(layout),
        triple=st.sampled_from(SPMV_TRIPLES if spmv else SPMM_TRIPLES),
    )
    def check(case, triple) -> None:
        dense, rng, n = case
        matrix = on_card(device, layout, dense, triple[0])
        x, x_host = upload(device, rng, triple[1], dense.shape[1], *(() if spmv else (n,)))
        for mode, name in (("vendor", "cusparse"), ("ours", "ours")):
            use(monkeypatch, mode)
            problem = sparse_problem(device, operation, matrix, triple, x.shape)
            try:
                assert lib.dispatcher().choose(kind, problem, capturing=False).name == name
            except NoCandidate:
                assert mode == "ours" and layout == "Coo"  # no row owns the entries of a Coo in SpMM, f16 atomics
                continue
            out = device.empty(triple[2], dense.shape[0], *(() if spmv else (n,)))
            (lib.spmv if spmv else lib.spmm)(matrix, x, out=out)
            assert_within_bound(out.to_host(), layout, dense, triple, x_host, mode)

    check()


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
@pytest.mark.parametrize("layout", CARD_LAYOUTS)
def test_cusparse_stable_runs_are_bitwise_equal(cuda_library, monkeypatch, operation, layout) -> None:
    use(monkeypatch, "vendor")
    device, spmv = cuda_library, operation is lib.Op.SPMV
    rng = np.random.default_rng(7)
    dense = rng.standard_normal((200, 200)) * (rng.random((200, 200)) < 0.3)
    matrix = on_card(device, layout, dense, T.f32)
    x, _ = upload(device, rng, T.f32, 200, *(() if spmv else (5,)))
    options = lib.LibOptions(determinism=Determinism.STABLE)
    problem = sparse_problem(device, operation, matrix, (T.f32,) * 3, x.shape, options)
    kind = lib.SpmvCandidate if spmv else lib.SpmmCandidate
    assert lib.dispatcher().choose(kind, problem, capturing=False).name == "cusparse"
    runs = set()
    for _ in range(5):
        out = device.empty(T.f32, 200, *(() if spmv else (5,)))
        (lib.spmv if spmv else lib.spmm)(matrix, x, out=out, options=options)
        runs.add(out.to_host().tobytes())
    assert len(runs) == 1


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
def test_cusparse_graph_replay_equals_the_eager_result(cuda_library, monkeypatch, operation) -> None:
    use(monkeypatch, "vendor")
    device, spmv = cuda_library, operation is lib.Op.SPMV
    rng = np.random.default_rng(11)
    dense = rng.standard_normal((120, 90)) * (rng.random((120, 90)) < 0.2)
    matrix = on_card(device, "Csr", dense, T.f32)
    x, _ = upload(device, rng, T.f32, 90, *(() if spmv else (4,)))
    shape = (120, *(() if spmv else (4,)))
    function = lib.spmv if spmv else lib.spmm
    eager = function(matrix, x, out=device.empty(T.f32, *shape))
    recorded = device.to_device(np.full(shape, np.nan, np.float32))
    problem = sparse_problem(device, operation, matrix, (T.f32,) * 3, x.shape)
    if spmv:
        planned = lib.dispatcher().choose(
            lib.SpmvCandidate,
            problem,
            capturing=True,
            operands=lib.SpmvOperands(matrix, x, recorded),
        )
    else:
        planned = lib.dispatcher().choose(
            lib.SpmmCandidate,
            problem,
            capturing=True,
            operands=lib.SpmmOperands(matrix, x, recorded),
        )
    assert planned.name == "cusparse"

    def record() -> None:
        function(matrix, x, out=recorded)

    graph = device.graph(record)
    for _ in range(2):
        graph()
        assert host(recorded).tobytes() == host(eager).tobytes()


def test_a_matrix_never_planned_is_not_captured_and_falls_to_ours(cuda_library, monkeypatch) -> None:
    use(monkeypatch, "auto")
    device = cuda_library
    rng = np.random.default_rng(13)
    dense = rng.standard_normal((50, 40)) * (rng.random((50, 40)) < 0.2)
    matrix = on_card(device, "Csr", dense, T.f32)
    x, x_host = upload(device, rng, T.f32, 40)
    out = device.to_device(np.full(50, np.nan, np.float32))

    def record() -> None:
        lib.spmv(matrix, x, out=out)

    graph = device.graph(record)
    graph()
    assert_within_bound(out.to_host(), "Csr", dense, (T.f32,) * 3, x_host, "vendor")


def test_two_streams_run_one_plan_on_buffers_of_their_own(cuda_library, monkeypatch) -> None:
    use(monkeypatch, "vendor")
    device = cuda_library
    rng = np.random.default_rng(17)
    dense = rng.standard_normal((300, 300)) * (rng.random((300, 300)) < 0.1)
    matrix = on_card(device, "Csr", dense, T.f32)
    x, x_host = upload(device, rng, T.f32, 300)
    outs = [device.empty(T.f32, 300) for _ in range(3)]
    for out, stream in zip(outs, (device.default_stream, device.stream(), device.stream()), strict=True):
        with stream:
            lib.spmv(matrix, x, out=out)
    for out in outs:
        assert_within_bound(out.to_host(), "Csr", dense, (T.f32,) * 3, x_host, "vendor")


def test_ell_is_refused_by_cusparse_and_runs_on_ours(cuda_library, monkeypatch) -> None:
    device = cuda_library
    rng = np.random.default_rng(19)
    dense = rng.standard_normal((30, 20)) * (rng.random((30, 20)) < 0.2)
    dense[0, 0] = 1.0
    matrix = on_card(device, "Ell", dense, T.f32)
    x, x_host = upload(device, rng, T.f32, 20)
    use(monkeypatch, "vendor")
    with pytest.raises(NoCandidate, match="cusparse: cuSPARSE reads Ell as sliced ELL"):
        lib.spmv(matrix, x)
    use(monkeypatch, "auto")
    assert_within_bound(host(lib.spmv(matrix, x)), "Ell", dense, (T.f32,) * 3, x_host, "ours")


def test_cusparse_without_its_wheel_refuses_and_the_call_falls_to_ours(cuda_library, monkeypatch) -> None:
    monkeypatch.setattr(lib_cusparse, "wheel_unavailable", lambda: "nvidia-cusparse wheel is missing")
    use(monkeypatch, "auto")
    device = cuda_library
    rng = np.random.default_rng(23)
    dense = rng.standard_normal((20, 20)) * (rng.random((20, 20)) < 0.3)
    dense[0, 0] = 1.0
    matrix = on_card(device, "Csr", dense, T.f32)
    x, x_host = upload(device, rng, T.f32, 20)
    problem = sparse_problem(device, lib.Op.SPMV, matrix, (T.f32,) * 3, x.shape)
    assert lib.dispatcher().choose(lib.SpmvCandidate, problem, capturing=False).name == "ours"
    assert_within_bound(host(lib.spmv(matrix, x)), "Csr", dense, (T.f32,) * 3, x_host, "ours")
    use(monkeypatch, "vendor")
    with pytest.raises(NoCandidate, match="nvidia-cusparse wheel is missing"):
        lib.spmv(matrix, x)


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
def test_the_tuner_times_every_contender_on_the_callers_operands(cuda_library, monkeypatch, operation) -> None:
    use(monkeypatch, "auto")
    configure(**{"lib.tune": "eager"})
    lib._dispatcher.cache_clear()
    device, spmv = cuda_library, operation is lib.Op.SPMV
    rng = np.random.default_rng(29)
    dense = rng.standard_normal((400, 400)) * (rng.random((400, 400)) < 0.05)
    matrix = on_card(device, "Csr", dense, T.f32)
    x, _ = upload(device, rng, T.f32, 400, *(() if spmv else (6,)))
    out = device.empty(T.f32, 400, *(() if spmv else (6,)))
    (lib.spmv if spmv else lib.spmm)(matrix, x, out=out)
    dispatcher = lib.dispatcher()
    for thread in dispatcher.measuring:
        thread.join(timeout=120)
    ((key, winner),) = dispatcher.tuning.winners.items()
    kind = lib.SpmvCandidate if spmv else lib.SpmmCandidate
    names = {c.name for c in dispatcher.candidates[operation] if isinstance(c, kind)}
    assert winner in names and winner.startswith(("ours", "cusparse")) and "Csr" in key
    problem = sparse_problem(device, operation, matrix, (T.f32,) * 3, x.shape)
    assert dispatcher.choose(kind, problem, capturing=False).name == winner
