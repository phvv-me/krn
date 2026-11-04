"""Sparse vectors and matrices as kernels of a few lines: CSR, Ellpack, blocked CSR, COO and 2:4 sparsity

`y[i] = sum(v * x[c] for c, v in a.row(i))` is SpMV whatever the layout: a matrix is a struct of dense vectors with
one method, `row(i)`, that yields the (column, value) of the entries stored for row `i`, and the class of the
argument picks the layout at launch, the way a C++ template is instantiated. `lib.spmv` and `lib.spmm` put a vendor
library in front where it runs the layout. The rest is written in the language with `items`, `sum`, gathers, warp
reductions and atomics: `spmv_vector` (a warp to a row), `sddmm`, `gather`, `scatter`, `axpy`, `spvv` and a 2:4
structured-sparse product, each a cuSPARSE or cuSPARSELt routine. Nothing sparse is built into the language.

Level: 4
Topics: sparse formats, SpMV, atomics, warp reductions
Needs: 03_memory_and_sync/01_histogram_atomics, 03_memory_and_sync/05_warp_shuffles
"""

import asyncio
import logging

import numpy as np

from krn import Items, Matrix, Vector, f32, host, items, kernel, lib, struct, sum, u8, u32
from krn.compiler.api import CPUDevice
from krn.data.sparse import sddmm, spmv_vector
from krn.data.sparse_build import bsr, coo, csr, ell, floats, indices
from krn.data.sparse_build import dense as dense_layout
from krn.emulate import emulate
from krn.frontend.atomic import atomic

log = logging.getLogger("sparse")


@struct
class SpVec:
    """A sparse vector: the positions of its entries in the dense vector, and their values."""

    indices: Vector[u32]
    values: Vector[f32]


@struct
class Packed24:
    """2:4 structured sparsity: two values kept of every four, and where in the group of four each one sat."""

    values: Matrix[f32]
    where: Matrix[u8]


@kernel
def gather(y: Vector[f32], x: SpVec) -> None:
    for k in items(x.values.shape[0]):
        x.values[k] = y[x.indices[k]]


@kernel
def scatter(x: SpVec, y: Vector[f32]) -> None:
    for k in items(x.values.shape[0]):
        y[x.indices[k]] = x.values[k]


@kernel
def scale(beta: f32, y: Vector[f32]) -> None:
    for i in items(y.shape[0]):
        y[i] = beta * y[i]


@kernel
def axpy(alpha: f32, x: SpVec, y: Vector[f32]) -> None:
    """y += alpha x. The indices of a sparse vector are distinct, so no two threads meet in a cell."""
    for k in items(x.values.shape[0]):
        y[x.indices[k]] += alpha * x.values[k]


@kernel
def spvv(x: SpVec, y: Vector[f32], result: Vector[f32]) -> None:
    for k in items(x.values.shape[0]):
        atomic.add(result, 0, x.values[k] * y[x.indices[k]])


@kernel
def beats(a: f32, b: f32, tie: bool) -> u32:
    """1 when `a` outranks `b`: larger, or equal when `tie` says the earlier position wins."""
    return u32(a > b or (a == b and tie))


@kernel
def prune_24(weights: Matrix[f32], packed: Packed24) -> None:
    """Keep the two largest magnitudes of every group of four along a row, and say where they were."""
    groups = weights.shape[1] // 4
    for t in items(weights.shape[0] * groups):
        i, group = divmod(t, groups)
        base = group * 4
        a0 = abs(weights[i, base])
        a1 = abs(weights[i, base + 1])
        a2 = abs(weights[i, base + 2])
        a3 = abs(weights[i, base + 3])
        keep0 = beats(a0, a1, True) + beats(a0, a2, True) + beats(a0, a3, True) > 1
        keep1 = beats(a1, a0, False) + beats(a1, a2, True) + beats(a1, a3, True) > 1
        keep2 = beats(a2, a0, False) + beats(a2, a1, False) + beats(a2, a3, True) > 1
        keep3 = beats(a3, a0, False) + beats(a3, a1, False) + beats(a3, a2, False) > 1
        first = 0 if keep0 else (1 if keep1 else 2)
        second = 3 if keep3 else (2 if keep2 else 1)
        packed.values[i, group * 2] = weights[i, base + first]
        packed.values[i, group * 2 + 1] = weights[i, base + second]
        packed.where[i, group * 2] = u8(first)
        packed.where[i, group * 2 + 1] = u8(second)


@kernel
def spmv_24(packed: Packed24, x: Vector[f32], y: Vector[f32]) -> None:
    for i in items(y.shape[0]):
        y[i] = sum(packed.values[i, k] * x[k // 2 * 4 + u32(packed.where[i, k])] for k in range(packed.values.shape[1]))


def random_dense(rng: np.random.Generator, rows: int, columns: int, density: float) -> np.ndarray:
    """A random sparse matrix as a dense one; some rows are empty and one is full."""
    matrix = rng.standard_normal((rows, columns)).astype(np.float32)
    matrix *= rng.random((rows, columns)) < density
    matrix[rng.integers(0, rows)] = 0.0
    matrix[rng.integers(0, rows), :] = rng.standard_normal(columns).astype(np.float32)
    return matrix


def prune_reference(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 2:4 pruning by its definition: per group of four keep the two largest magnitudes, in position order."""
    groups = matrix.reshape(matrix.shape[0], -1, 4)
    keep = np.sort(np.argsort(-np.abs(groups), axis=2, kind="stable")[:, :, :2], axis=2)
    values = np.take_along_axis(groups, keep, axis=2).reshape(matrix.shape[0], -1)
    pruned = np.zeros_like(groups)
    np.put_along_axis(pruned, keep, np.take_along_axis(groups, keep, axis=2), axis=2)
    return values, keep.reshape(matrix.shape[0], -1).astype(np.uint8), pruned.reshape(matrix.shape)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(5)
    rows, columns, width, block = 40, 36, 9, 4
    dense = random_dense(rng, rows, columns, 0.15)
    x = rng.standard_normal(columns).astype(np.float32)
    b = rng.standard_normal((columns, width)).astype(np.float32)

    def check(name: str, got: np.ndarray, expected: np.ndarray) -> None:
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4, err_msg=name)
        log.info("%-18s agrees with numpy", name)

    xs, tb = floats(device, x), device.empty(f32, columns, width)
    np.asarray(tb)[:] = b
    layouts = {
        "csr": csr(device, dense),
        "ell": ell(device, dense),
        "bsr": bsr(device, dense, block),
        "dense": dense_layout(device, dense),
    }

    for name, matrix in (*layouts.items(), ("coo", coo(device, dense))):
        y = lib.spmv(matrix, xs)  # whichever candidate runs it: the vendor's library, or the kernels of `sparse.py`
        await device.synchronize()
        check(f"lib.spmv {name}", np.asarray(y), dense @ x)
        if name != "coo":
            c = lib.spmm(matrix, tb)
            await device.synchronize()
            check(f"lib.spmm {name}", np.asarray(c), dense @ b)

    # the lanes of a warp talk to each other, which a C loop over threads cannot do, so the emulator runs this one
    for name, matrix in layouts.items():
        y = device.empty(f32, rows)
        emulate(spmv_vector, Items(rows * 32), matrix, np.asarray(xs), y)  # a warp of 32 lanes: its default target
        check(f"spmv_vector {name}", np.asarray(y), dense @ x)

    left = rng.standard_normal((rows, 5)).astype(np.float32)
    right = rng.standard_normal((5, columns)).astype(np.float32)
    ta, tr = device.empty(f32, rows, 5), device.empty(f32, 5, columns)
    np.asarray(ta)[:], np.asarray(tr)[:] = left, right
    pattern = csr(device, dense)
    out = device.empty(f32, len(np.asarray(pattern.values)))
    sddmm(pattern, ta, tr, out)
    await device.synchronize()
    check("sddmm csr", np.asarray(out), (left @ right)[np.nonzero(dense)])

    # sparse vectors
    positions = np.sort(rng.choice(columns, 11, replace=False)).astype(np.uint32)
    sparse = SpVec(indices(device, positions), device.empty(f32, 11))
    gather(xs, sparse)
    await device.synchronize()
    check("gather", np.asarray(sparse.values), x[positions])
    dense_out = floats(device, np.zeros(columns, np.float32))
    scatter(sparse, dense_out)
    await device.synchronize()
    expected = np.zeros(columns, np.float32)
    expected[positions] = x[positions]
    check("scatter", np.asarray(dense_out), expected)
    scale(f32(0.5), xs)
    axpy(f32(2.0), sparse, xs)
    await device.synchronize()
    check("axpy", np.asarray(xs), 0.5 * x + 2.0 * expected)
    result = floats(device, np.zeros(1, np.float32))
    spvv(sparse, xs, result)
    await device.synchronize()
    check("spvv", np.asarray(result), np.array([np.dot(x[positions], (0.5 * x + 2.0 * expected)[positions])]))

    # 2:4 structured sparsity
    weights = rng.standard_normal((16, 32)).astype(np.float32)
    packed_values, packed_where, pruned = prune_reference(weights)
    tw = device.empty(f32, 16, 32)
    np.asarray(tw)[:] = weights
    packed = Packed24(device.empty(f32, 16, 16), device.empty(u8, 16, 16))
    prune_24(tw, packed)
    await device.synchronize()
    check("prune_24", np.asarray(packed.values), packed_values)
    np.testing.assert_array_equal(np.asarray(packed.where), packed_where)
    v, out24 = rng.standard_normal(32).astype(np.float32), device.empty(f32, 16)
    spmv_24(packed, floats(device, v), out24)
    await device.synchronize()
    check("spmv_24", np.asarray(out24), pruned @ v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
