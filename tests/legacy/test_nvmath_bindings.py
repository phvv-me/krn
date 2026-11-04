"""Official NVIDIA bindings: library lifetimes and small numerical calls on a CUDA card."""

import ctypes
from contextlib import ExitStack
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import T
from krn.lib import lib_nvidia

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

try:
    from nvmath.bindings import cublas, cublasLt, cusparse
except ImportError, OSError:
    cublas = cublasLt = cusparse = None


@pytest.fixture
def nvmath() -> None:
    if cublasLt is None or cusparse is None or cublas is None:
        pytest.skip("nvmath.bindings is not importable")


@pytest.mark.parametrize("library", ["cublasLt", "cusparse"])
def test_version(nvmath: None, gpu: CUDADevice, library: str) -> None:
    assert cublasLt is not None and cusparse is not None
    if library == "cublasLt":
        assert cublasLt.get_version() > 0
    else:
        handle = cusparse.create()
        try:
            assert cusparse.get_version(handle) > 0
        finally:
            cusparse.destroy(handle)


@pytest.mark.parametrize("library", ["cublasLt", "cusparse"])
def test_handle_round_trip(nvmath: None, gpu: CUDADevice, library: str) -> None:
    assert cublasLt is not None and cusparse is not None
    api = cublasLt if library == "cublasLt" else cusparse
    handle = api.create()
    try:
        assert isinstance(handle, int) and handle != 0
    finally:
        api.destroy(handle)


def test_cublaslt_f16_f32_compute(nvmath: None, gpu: CUDADevice) -> None:
    assert cublasLt is not None and cublas is not None
    api = cublasLt
    rng = np.random.default_rng(31)
    m, n, k = 7, 5, 9
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    da, db, out = gpu.to_device(a), gpu.to_device(b), gpu.empty(T.f16, m, n)
    gpu.default_stream.sync()
    with ExitStack() as cleanup:
        handle = api.create()
        cleanup.callback(api.destroy, handle)
        desc = api.matmul_desc_create(cublas.ComputeType.COMPUTE_32F, lib_nvidia.data_type(T.f32))
        cleanup.callback(api.matmul_desc_destroy, desc)
        layouts = []
        for rows, columns in ((n, k), (k, m), (n, m)):
            layout = api.matrix_layout_create(lib_nvidia.data_type(T.f16), rows, columns, rows)
            cleanup.callback(api.matrix_layout_destroy, layout)
            layouts.append(layout)
        preference = api.matmul_preference_create()
        cleanup.callback(api.matmul_preference_destroy, preference)
        workspace = lib_nvidia.allocate(gpu, 4 << 20)
        limit = ctypes.c_uint64(workspace.shape[0])
        api.matmul_preference_set_attribute(
            preference,
            api.MatmulPreferenceAttribute.MAX_WORKSPACE_BYTES,
            ctypes.addressof(limit),
            ctypes.sizeof(limit),
        )
        result, found = api.MatmulHeuristicResult(), ctypes.c_int()
        left, right, output = layouts
        api.matmul_algo_get_heuristic(
            handle,
            desc,
            left,
            right,
            output,
            output,
            preference,
            1,
            result.ptr,
            ctypes.addressof(found),
        )
        assert found.value == 1 and result.state == cublas.Status.SUCCESS
        assert result.workspace_size <= workspace.shape[0]
        alpha, beta = ctypes.c_float(1), ctypes.c_float(0)
        api.matmul(
            handle,
            desc,
            ctypes.addressof(alpha),
            lib_nvidia.address(db),
            left,
            lib_nvidia.address(da),
            right,
            ctypes.addressof(beta),
            lib_nvidia.address(out),
            output,
            lib_nvidia.address(out),
            output,
            result.ptr,
            lib_nvidia.address(workspace),
            workspace.shape[0],
            lib_nvidia.handle(gpu.default_stream),
        )
        np.testing.assert_allclose(out.to_host(), a.astype(np.float32) @ b.astype(np.float32), rtol=2e-3, atol=2e-3)


def test_cusparse_csr_spmv(nvmath: None, gpu: CUDADevice) -> None:
    assert cusparse is not None
    api = cusparse
    offsets = gpu.to_device(np.array([0, 2, 3, 5], np.int32))
    indices = gpu.to_device(np.array([0, 2, 1, 0, 2], np.int32))
    values = gpu.to_device(np.array([2, 3, 4, 5, 6], np.float32))
    x_host = np.array([1, 2, 3], np.float32)
    x, out = gpu.to_device(x_host), gpu.empty(T.f32, 3)
    gpu.default_stream.sync()
    with ExitStack() as cleanup:
        handle = api.create()
        cleanup.callback(api.destroy, handle)
        api.set_stream(handle, lib_nvidia.handle(gpu.default_stream))
        api.set_pointer_mode(handle, api.PointerMode.HOST)
        dtype = lib_nvidia.data_type(T.f32)
        matrix = api.create_csr(
            3,
            3,
            5,
            lib_nvidia.address(offsets),
            lib_nvidia.address(indices),
            lib_nvidia.address(values),
            api.IndexType.INDEX_32I,
            api.IndexType.INDEX_32I,
            api.IndexBase.ZERO,
            dtype,
        )
        cleanup.callback(api.destroy_sp_mat, matrix)
        vector = api.create_dn_vec(3, lib_nvidia.address(x), dtype)
        cleanup.callback(api.destroy_dn_vec, vector)
        output = api.create_dn_vec(3, lib_nvidia.address(out), dtype)
        cleanup.callback(api.destroy_dn_vec, output)
        alpha, beta = ctypes.c_float(1), ctypes.c_float(0)
        arguments = (
            handle,
            api.Operation.NON_TRANSPOSE,
            ctypes.addressof(alpha),
            matrix,
            vector,
            ctypes.addressof(beta),
            output,
            dtype,
            api.SpMVAlg.CSR_ALG2,
        )
        workspace = lib_nvidia.allocate(gpu, api.sp_mv_buffer_size(*arguments))
        api.sp_mv_preprocess(*arguments, lib_nvidia.address(workspace))
        api.sp_mv(*arguments, lib_nvidia.address(workspace))
        dense = np.array([[2, 0, 3], [0, 4, 0], [5, 0, 6]], np.float32)
        np.testing.assert_allclose(out.to_host(), dense @ x_host, rtol=1e-6)
