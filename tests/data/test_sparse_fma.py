"""The sparse row kernels fuse each product into the running sum, and agree with a float64 sum and with the emulator.

The row `[-(1 + 2**-11), 1 + 2**-12]` against `[1, 1 + 2**-12]` cancels to zero when each product rounds first and to
`2**-24` when the second product is fused. Every format stores that row, so each must give `2**-24`. The emulator runs
the captured IR, and the C backend runs one thread per block, which is what its empty barrier supports.
"""

from pathlib import Path

import numpy as np
import pytest

from krn import Items
from krn.compiler.api import capture, emit
from krn.data import sparse, sparse_build
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.frontend.geometry import Geometry
from krn.runtime.cpu import CPUProgram, Value
from tests.helpers.devices import bindings

SPARSE_KERNELS_FILE = Path(sparse.__file__)
SCALAR = Target.of("avx512")
VECTOR = Target.of("sm_90a")
ONE = 1 + 2.0**-12
ROWS, COLUMNS, WIDTH, BLOCK = 4, 4, 3, 2
FORMATS = ("Csr", "Ell", "Bsr", "Dense")
RNG = np.random.default_rng(7)


def build(name: str, dense: np.ndarray) -> dict[str, np.ndarray]:
    """The arrays of the struct `name` holds for the dense matrix."""
    if name == "Bsr":
        return sparse_build.bsr_arrays(dense, BLOCK)._asdict()
    return {
        "Csr": sparse_build.csr_arrays,
        "Ell": sparse_build.ell_arrays,
        "Dense": sparse_build.dense_arrays,
    }[name](dense)._asdict()


def run(
    kernel: str, arguments: list[Value], launch: Geometry, structs: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """The output, the last argument, after `kernel` ran in the emulator and in the C backend from the same start."""
    ir = capture(
        SPARSE_KERNELS_FILE,
        kernel,
        structs=structs,
        dtypes=bindings(SPARSE_KERNELS_FILE, kernel, arguments),
        target=SCALAR,
    )
    oracle = [duplicate(argument) for argument in arguments]
    emulate(ir, SCALAR).run(launch, oracle)
    actual = [duplicate(argument) for argument in arguments]
    CPUProgram(ir, emit(ir, SCALAR))(*actual, launch=launch)
    return np.asarray(oracle[-1]), np.asarray(actual[-1])


def duplicate(argument: Value) -> Value:
    """A run's own copy of its arrays; the scalars of a struct (`rows`, `width`) are values already."""
    if isinstance(argument, dict):
        return {name: duplicate(value) for name, value in argument.items()}
    return argument.copy() if isinstance(argument, np.ndarray) else argument


def cancelling_dense() -> tuple[np.ndarray, np.ndarray]:
    """Row 0 cancels; the other rows hold ordinary values, so the output is not all zero."""
    dense = np.zeros((ROWS, COLUMNS), np.float32)
    dense[0, :2] = [-(1 + 2.0**-11), ONE]
    dense[1] = RNG.standard_normal(COLUMNS)
    x = np.array([1.0, ONE, 0.0, 0.0], np.float32)
    return dense, x


@pytest.mark.parametrize("name", FORMATS)
def test_a_cancelling_row_keeps_its_fused_product_in_every_format(name: str) -> None:
    dense, x = cancelling_dense()
    oracle, actual = run("spmv", [build(name, dense), x, np.zeros(ROWS, np.float32)], Items(ROWS), {"a": name})
    assert oracle[0] == np.float32(2.0**-24)
    np.testing.assert_array_equal(actual, oracle)


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize(("columns", "expected"), [((0, 32), 2.0**-24), ((0, 1), 0.0)])
def test_a_lane_holding_a_cancelling_pair_keeps_the_fused_product(
    name: str, columns: tuple[int, int], expected: float
) -> None:
    """Lane 0 strides over positions 0 and 32 of the row, so it holds both products of the pair when the pair sits
    there and fuses them; the other lanes multiply by zero. The same pair in lanes 0 and 1 is rounded in its lanes and
    cancels to zero in the tree sum, which is the kernel's order. The row has 33 positions so both lanes exist in every
    format. `spmv_vector` needs warp shuffles, so only the emulator runs it."""
    dense = np.zeros((ROWS, 36), np.float32)
    dense[0, :33] = 1.0
    dense[0, list(columns)] = [-(1 + 2.0**-11), ONE]
    x = np.zeros(36, np.float32)
    x[list(columns)] = [1.0, ONE]
    matrix, y = build(name, dense), np.zeros(ROWS, np.float32)
    ir = capture(
        SPARSE_KERNELS_FILE,
        "spmv_vector",
        structs={"a": name},
        dtypes=bindings(SPARSE_KERNELS_FILE, "spmv_vector", [matrix, x, y]),
        target=VECTOR,
    )
    emulate(ir, VECTOR).run(Items(ROWS * VECTOR.warp_size), [matrix, x, y])
    assert y[0] == np.float32(expected)


@pytest.mark.parametrize("name", FORMATS)
def test_a_cancelling_row_keeps_its_fused_product_in_spmm(name: str) -> None:
    dense, x = cancelling_dense()
    b = np.stack([x, np.ones(COLUMNS, np.float32), np.zeros(COLUMNS, np.float32)], axis=1)
    oracle, actual = run(
        "spmm", [build(name, dense), b, np.zeros((ROWS, WIDTH), np.float32)], Items(ROWS * WIDTH), {"a": name}
    )
    assert oracle[0, 0] == np.float32(2.0**-24)
    np.testing.assert_array_equal(actual, oracle)


def test_a_cancelling_dot_product_keeps_its_fused_product_in_sddmm() -> None:
    dense, _ = cancelling_dense()
    pattern = sparse_build.csr_arrays(np.where(dense != 0, 1.0, 0.0).astype(np.float32))._asdict()
    left = np.zeros((ROWS, 2), np.float32)
    left[0] = [-(1 + 2.0**-11), ONE]
    right = np.zeros((2, COLUMNS), np.float32)
    right[:, 0] = [1.0, ONE]
    out = np.zeros(pattern["values"].shape[0], np.float32)
    oracle, actual = run("sddmm", [pattern, left, right, out], Items(ROWS), {"pattern": "Csr"})
    assert oracle[0] == np.float32(2.0**-24)
    np.testing.assert_array_equal(actual, oracle)


@pytest.mark.parametrize("name", FORMATS)
def test_random_products_agree_with_float64_and_the_emulator(name: str) -> None:
    dense = (RNG.standard_normal((40, 36)) * (RNG.random((40, 36)) < 0.15)).astype(np.float32)
    x = RNG.standard_normal(36).astype(np.float32)
    b = RNG.standard_normal((36, 9)).astype(np.float32)
    matrix = build(name, dense)
    cases: dict[str, tuple[list[Value], Items, np.ndarray]] = {
        "spmv": ([matrix, x, np.zeros(40, np.float32)], Items(40), dense.astype(np.float64) @ x.astype(np.float64)),
        "spmm": (
            [matrix, b, np.zeros((40, 9), np.float32)],
            Items(40 * 9),
            dense.astype(np.float64) @ b.astype(np.float64),
        ),
    }
    for kernel, (arguments, launch, reference) in cases.items():
        oracle, actual = run(kernel, arguments, launch, {"a": name})
        np.testing.assert_allclose(oracle, reference, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(actual, oracle)


def test_random_sddmm_agrees_with_float64_and_the_emulator() -> None:
    mask = (RNG.random((40, 36)) < 0.15).astype(np.float32)
    left = RNG.standard_normal((40, 5)).astype(np.float32)
    right = RNG.standard_normal((5, 36)).astype(np.float32)
    pattern = sparse_build.csr_arrays(mask)._asdict()
    out = np.zeros(pattern["values"].shape[0], np.float32)
    oracle, actual = run("sddmm", [pattern, left, right, out], Items(40), {"pattern": "Csr"})
    rows = np.repeat(np.arange(40), np.diff(pattern["indptr"]))
    columns = pattern["indices"]
    reference = np.einsum("kd,dk->k", left[rows].astype(np.float64), right[:, columns].astype(np.float64))
    np.testing.assert_allclose(oracle, reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual, oracle)
