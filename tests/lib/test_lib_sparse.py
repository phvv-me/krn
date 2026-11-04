"""Sparse dispatch on the CPU, independent host products, and pure vendor eligibility rules."""

from collections.abc import Callable
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from krn import T, lib
from krn.compiler.api import CPUDevice
from krn.data import sparse_build
from krn.data.sparse import Bsr, Coo, Csr, Ell, Rows
from krn.facts.targets import Target
from krn.frontend.types import FloatingPoint
from krn.lib import lib_cusparse, lib_ours
from krn.lib.lib import NoCandidate
from krn.lib.lib_host import product

RNG = np.random.default_rng(41)
ROWS, COLUMNS, WIDTH, BLOCK = 40, 36, 9, 4
TIMEOUT = 60

type Builder = Callable[[CPUDevice, np.ndarray], Rows | Coo]
FORMATS: dict[str, Builder] = {
    "Csr": sparse_build.csr,
    "Ell": sparse_build.ell,
    "Bsr": lambda device, dense: sparse_build.bsr(device, dense, BLOCK),
    "Coo": sparse_build.coo,
    "Dense": sparse_build.dense,
}
DENSE = (RNG.standard_normal((ROWS, COLUMNS)) * (RNG.random((ROWS, COLUMNS)) < 0.15)).astype(np.float32)
DENSE[3] = 0.0  # an empty row
DENSE[17] = RNG.standard_normal(COLUMNS)  # a full one
X = RNG.standard_normal(COLUMNS).astype(np.float32)
B = RNG.standard_normal((COLUMNS, WIDTH)).astype(np.float32)


def operands(device: CPUDevice, name: str):
    tx, tb = device.empty(T.f32, COLUMNS), device.empty(T.f32, COLUMNS, WIDTH)
    tx.array[:], tb.array[:] = X, B
    return FORMATS[name](device, DENSE), tx, tb


@pytest.mark.parametrize("dtype", [T.f16, T.f32, T.f64])
def test_sparse_builders_upload_contiguous_typed_arrays(device: CPUDevice, dtype: type[FloatingPoint]) -> None:
    source = np.arange(12, dtype=np.float64)[::2]
    with patch.object(device, "to_device", wraps=device.to_device) as upload:
        values = sparse_build.floats(device, source, dtype)
        indices = sparse_build.indices(device, source)
    assert upload.call_count == 2
    for call, expected in zip(upload.call_args_list, (sparse_build.NUMPY[dtype], np.dtype(np.uint32)), strict=True):
        array = call.args[0]
        assert array.flags.c_contiguous and array.dtype == expected
        np.testing.assert_array_equal(array, source)
    np.testing.assert_array_equal(np.asarray(values), source.astype(sparse_build.NUMPY[dtype]))
    np.testing.assert_array_equal(np.asarray(indices), source.astype(np.uint32))


@pytest.mark.parametrize("name", FORMATS)
def test_the_host_reads_every_format_as_the_dense_product(device: CPUDevice, name: str) -> None:
    matrix, _, _ = operands(device, name)
    np.testing.assert_allclose(product(matrix, X), DENSE @ X, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(product(matrix, B), DENSE @ B, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", FORMATS)
def test_our_kernels_equal_the_dense_product_in_every_format(library, device: CPUDevice, name: str) -> None:
    library("ours")
    matrix, tx, tb = operands(device, name)
    y = lib.spmv(matrix, tx)
    device.default_stream.record().wait(TIMEOUT)
    np.testing.assert_allclose(np.asarray(y), DENSE @ X, rtol=1e-4, atol=1e-4)
    if name == "Coo":
        with pytest.raises(NoCandidate, match="Coo has no row to own its entries"):
            lib.spmm(matrix, tb)
        return
    c = lib.spmm(matrix, tb)
    device.default_stream.record().wait(TIMEOUT)
    np.testing.assert_allclose(np.asarray(c), DENSE @ B, rtol=1e-4, atol=1e-4)


def test_a_format_nobody_runs_falls_to_the_host_instead_of_failing(library, device: CPUDevice) -> None:
    library("auto")
    matrix, _, tb = operands(device, "Coo")
    c = lib.spmm(matrix, tb)  # accelerate takes only Csr, ours only the row formats
    np.testing.assert_allclose(np.asarray(c), DENSE @ B, rtol=1e-4, atol=1e-4)


def test_accelerate_takes_csr_and_says_why_it_refuses_the_rest(accelerate, library, device: CPUDevice) -> None:
    library("vendor")
    matrix, tx, tb = operands(device, "Csr")
    options = lib.LibOptions.of(device.target)
    problem = lib.Problem(lib.Op.SPMV, (T.f32,) * 3, (ROWS, COLUMNS, 1), options, layout="Csr")
    assert lib.dispatcher().choose(lib.SpmvCandidate, problem, capturing=False).name == "accelerate"
    y, c = lib.spmv(matrix, tx), lib.spmm(matrix, tb)
    np.testing.assert_allclose(np.asarray(y), DENSE @ X, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(c), DENSE @ B, rtol=1e-4, atol=1e-4)
    for name in ("Ell", "Bsr", "Coo", "Dense"):
        other, tx, _ = operands(device, name)
        with pytest.raises(NoCandidate, match=f"stores CSC, of which Csr is the transpose; {name} would need"):
            lib.spmv(other, tx)


def test_the_vendor_mode_names_what_it_excluded(library, device: CPUDevice) -> None:
    library("vendor")
    matrix, tx, _ = operands(device, "Ell")
    with pytest.raises(NoCandidate, match=r"lib.mode=vendor excludes"):
        lib.spmv(matrix, tx)


def test_every_format_class_is_a_struct_the_host_can_read() -> None:
    assert all(issubclass(kind, Rows) for kind in (Csr, Ell, Bsr)) and not issubclass(Coo, Rows)


def test_cusparse_refuses_what_it_has_no_descriptor_for_and_says_so() -> None:
    target = Target.of("sm_120a")
    for candidate in (lib_cusparse.CusparseSpmv(), lib_cusparse.CusparseSpmm()):
        shape = (ROWS, COLUMNS, 1, WIDTH) if candidate.op is lib.Op.SPMM else (ROWS, COLUMNS, 1)
        problem = lib.Problem(candidate.op, (T.f32,) * 3, shape, lib.LibOptions.of(target))
        for layout, reason in (("Ell", "sliced ELL"), ("Dense", "Dense is not a sparse format")):
            assert reason in str(candidate.refusal(replace(problem, layout=layout), target))
        for layout in ("Csr", "Bsr", "Coo"):
            assert candidate.refusal(replace(problem, layout=layout), target) is None
        assert candidate.unavailable(Target.of("avx512")) == (
            "cuSPARSE multiplies NVIDIA device memory, and avx512 is not NVIDIA"
        )


@pytest.mark.parametrize("candidate", lib_cusparse.CANDIDATES)
def test_cusparse_algorithm_contract(candidate) -> None:
    target = Target.of("sm_120a")
    options = lib.LibOptions(determinism=lib.Determinism.STABLE)
    expected = {
        lib.Op.SPMV: {
            "csr_alg1": ("Csr", False),
            "csr_alg2": ("Csr", True),
            "coo_alg1": ("Coo", False),
            "coo_alg2": ("Coo", True),
        },
        lib.Op.SPMM: {
            "csr_alg1": ("Csr", False),
            "csr_alg2": ("Csr", False),
            "csr_alg3": ("Csr", True),
            "coo_alg1": ("Coo", False),
            "coo_alg2": ("Coo", True),
            "coo_alg3": ("Coo", False),
            "coo_alg4": ("Coo", False),
        },
    }
    assert {c.algorithm for c in lib_cusparse.CANDIDATES if c.op is candidate.op and c.contender} == set(
        expected[candidate.op]
    )
    for layout in ("Csr", "Coo", "Bsr"):
        problem = lib.Problem(candidate.op, (T.f32,) * 3, (ROWS, COLUMNS, 1, WIDTH), options, layout=layout)
        algorithm = candidate.algorithm_for(problem)
        if candidate.contender:
            kind, deterministic = expected[candidate.op][candidate.algorithm]
            assert candidate.name == f"cusparse:{candidate.algorithm}"
            assert algorithm.layouts == (kind,) and algorithm.deterministic == deterministic
            reason = candidate.refusal(problem, target)
            assert (reason is None) == (kind == layout and deterministic)
            unrestricted = replace(problem, options=lib.LibOptions())
            assert (candidate.refusal(unrestricted, target) is None) == (kind == layout)
        else:
            name = {
                "Csr": "csr_alg2" if candidate.op is lib.Op.SPMV else "csr_alg3",
                "Coo": "coo_alg2",
                "Bsr": "bsr_alg1",
            }[layout]
            assert algorithm.name == name and algorithm.deterministic
            assert candidate.refusal(problem, target) is None
            unrestricted = replace(problem, options=lib.LibOptions())
            assert candidate.algorithm_for(unrestricted).name == ("bsr_alg1" if layout == "Bsr" else "default")


@pytest.mark.parametrize("candidate", [lib_cusparse.CusparseSpmv(), lib_cusparse.CusparseSpmm()])
@pytest.mark.parametrize("dimension", range(3))
def test_cusparse_refuses_unsigned_sizes_above_int_max(candidate, dimension: int) -> None:
    target = Target.of("sm_120a")
    shape = [4, 4, 7, WIDTH]
    shape[dimension] = 2**31 - 1
    problem = lib.Problem(candidate.op, (T.f32,) * 3, tuple(shape), lib.LibOptions.of(target), layout="Csr")
    assert candidate.refusal(problem, target) is None
    shape[dimension] += 1
    assert "signed 32-bit" in str(candidate.refusal(replace(problem, shape=tuple(shape)), target))


@pytest.mark.parametrize("candidate", [lib_ours.OursSpmv(), lib_ours.OursSpmvThread(), lib_ours.OursSpmvWarp()])
def test_ours_spmv_contenders_and_atomic_determinism(candidate) -> None:
    target = Target.of("sm_120a")
    problem = lib.Problem(lib.Op.SPMV, (T.f32,) * 3, (4, 4, 7), lib.LibOptions(), layout="Coo")
    assert candidate.refusal(problem, target) is None
    stable = replace(problem, options=lib.LibOptions(determinism=lib.Determinism.STABLE))
    assert "atomics add in any order" in str(candidate.refusal(stable, target))
    if candidate.name == "ours:warp":
        assert "WARP_SHUFFLE" in str(candidate.refusal(replace(problem, layout="Csr"), Target.of("avx512")))
