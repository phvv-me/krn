"""Sparse matrices from dense ones: the host arrays of each format, and the structs of device tensors built from them.

`csr_arrays(dense)` is the host form, which is also what a struct becomes when it crosses to a card (`._asdict()`).
`csr(device, dense)` uploads it into a `Csr`. A zero is a missing entry; a matrix with no zeros makes a (full) sparse
matrix, which is legal and slow.
"""

from typing import NamedTuple, overload

import numpy as np

from krn.frontend.memory import Vector
from krn.frontend.types import FloatingPoint, f16, f32, f64, u32
from krn.runtime.api import Device

from .sparse import Bsr, Coo, Csr, Dense, Ell

NUMPY: dict[type[FloatingPoint], np.dtype] = {
    f16: np.dtype("float16"),
    f32: np.dtype("float32"),
    f64: np.dtype("float64"),
}


class CsrArrays(NamedTuple):
    values: np.ndarray
    rows: int
    columns: int
    indptr: np.ndarray
    indices: np.ndarray


class EllArrays(NamedTuple):
    values: np.ndarray
    rows: int
    columns: int
    indices: np.ndarray
    width: int


class BsrArrays(NamedTuple):
    values: np.ndarray
    rows: int
    columns: int
    indptr: np.ndarray
    indices: np.ndarray
    block: int


class DenseArrays(NamedTuple):
    values: np.ndarray
    rows: int
    columns: int


class CooArrays(NamedTuple):
    values: np.ndarray
    rows: int
    columns: int
    row_indices: np.ndarray
    column_indices: np.ndarray


def csr_from_triples(
    rows: int,
    columns: int,
    row: np.ndarray,
    column: np.ndarray,
    value: np.ndarray,
    dtype: type[FloatingPoint] = f32,
) -> CsrArrays:
    """CSR of the entries (row[k], column[k], value[k]) in any order, each position at most once."""
    order = np.lexsort((column, row))
    pointers = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=rows))])
    return CsrArrays(
        value[order].astype(NUMPY[dtype]),
        rows,
        columns,
        pointers.astype(np.uint32),
        column[order].astype(np.uint32),
    )


def csr_arrays(dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> CsrArrays:
    rows, columns = dense.shape
    row, column = np.nonzero(dense)
    return csr_from_triples(rows, columns, row, column, dense[row, column], dtype)


def ell_arrays(dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> EllArrays:
    """Column-major, so neighboring rows are neighbors in memory; padding entries have value 0 and column 0."""
    rows, columns = dense.shape
    width = max(1, int((dense != 0).sum(axis=1).max()))
    indices, values = np.zeros((width, rows), np.uint32), np.zeros((width, rows), NUMPY[dtype])
    for row in range(rows):
        kept = np.nonzero(dense[row])[0]
        indices[: len(kept), row], values[: len(kept), row] = kept, dense[row, kept]
    return EllArrays(values.reshape(-1), rows, columns, indices.reshape(-1), width)


def bsr_arrays(dense: np.ndarray, block: int, dtype: type[FloatingPoint] = f32) -> BsrArrays:
    """Tiles of `block` by `block`, row-major inside, kept when any entry is nonzero. Both sides must divide."""
    rows, columns = dense.shape
    if rows % block or columns % block:
        raise ValueError(f"a {rows} by {columns} matrix does not divide into tiles of {block}")
    tiles = dense.reshape(rows // block, block, columns // block, block).transpose(0, 2, 1, 3)
    kept = [np.flatnonzero(tiles[row].any(axis=(1, 2))) for row in range(rows // block)]
    return BsrArrays(
        np.concatenate([tiles[row, k].reshape(-1) for row, k in enumerate(kept)]).astype(NUMPY[dtype]),
        rows,
        columns,
        np.concatenate([[0], np.cumsum([len(k) for k in kept])]).astype(np.uint32),
        np.concatenate(kept).astype(np.uint32),
        block,
    )


def coo_arrays(dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> CooArrays:
    rows, columns = dense.shape
    row_indices, column_indices = np.nonzero(dense)
    return CooArrays(
        dense[row_indices, column_indices].astype(NUMPY[dtype]),
        rows,
        columns,
        row_indices.astype(np.uint32),
        column_indices.astype(np.uint32),
    )


def dense_arrays(dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> DenseArrays:
    """Every entry, row-major."""
    rows, columns = dense.shape
    return DenseArrays(dense.astype(NUMPY[dtype]).reshape(-1), rows, columns)


@overload
def floats(device: Device, array: np.ndarray) -> Vector[f32]: ...
@overload
def floats[F: FloatingPoint](device: Device, array: np.ndarray, dtype: type[F]) -> Vector[F]: ...
def floats(device: Device, array: np.ndarray, dtype: type[FloatingPoint] = f32) -> Vector:
    return device.to_device(np.ascontiguousarray(array, dtype=NUMPY[dtype]))


def indices(device: Device, array: np.ndarray) -> Vector[u32]:
    return device.to_device(np.ascontiguousarray(array, dtype=np.uint32))


@overload
def csr(device: Device, dense: np.ndarray) -> Csr[f32]: ...
@overload
def csr[F: FloatingPoint](device: Device, dense: np.ndarray, dtype: type[F]) -> Csr[F]: ...
def csr(device: Device, dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> Csr:
    return csr_on(device, csr_arrays(dense, dtype), dtype)


@overload
def csr_on(device: Device, a: CsrArrays) -> Csr[f32]: ...
@overload
def csr_on[F: FloatingPoint](device: Device, a: CsrArrays, dtype: type[F]) -> Csr[F]: ...
def csr_on(device: Device, a: CsrArrays, dtype: type[FloatingPoint] = f32) -> Csr:
    """The arrays of a matrix that was never dense, uploaded."""
    return Csr(
        values=floats(device, a.values, dtype),
        rows=u32(a.rows),
        columns=u32(a.columns),
        indptr=indices(device, a.indptr),
        indices=indices(device, a.indices),
    )


@overload
def ell(device: Device, dense: np.ndarray) -> Ell[f32]: ...
@overload
def ell[F: FloatingPoint](device: Device, dense: np.ndarray, dtype: type[F]) -> Ell[F]: ...
def ell(device: Device, dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> Ell:
    a = ell_arrays(dense, dtype)
    return Ell(
        values=floats(device, a.values, dtype),
        rows=u32(a.rows),
        columns=u32(a.columns),
        indices=indices(device, a.indices),
        width=u32(a.width),
    )


@overload
def bsr(device: Device, dense: np.ndarray, block: int) -> Bsr[f32]: ...
@overload
def bsr[F: FloatingPoint](device: Device, dense: np.ndarray, block: int, dtype: type[F]) -> Bsr[F]: ...
def bsr(device: Device, dense: np.ndarray, block: int, dtype: type[FloatingPoint] = f32) -> Bsr:
    a = bsr_arrays(dense, block, dtype)
    return Bsr(
        values=floats(device, a.values, dtype),
        rows=u32(a.rows),
        columns=u32(a.columns),
        indptr=indices(device, a.indptr),
        indices=indices(device, a.indices),
        block=u32(a.block),
    )


@overload
def coo(device: Device, dense: np.ndarray) -> Coo[f32]: ...
@overload
def coo[F: FloatingPoint](device: Device, dense: np.ndarray, dtype: type[F]) -> Coo[F]: ...
def coo(device: Device, dense: np.ndarray, dtype: type[FloatingPoint] = f32) -> Coo:
    a = coo_arrays(dense, dtype)
    return Coo(
        values=floats(device, a.values, dtype),
        rows=u32(a.rows),
        columns=u32(a.columns),
        row_indices=indices(device, a.row_indices),
        column_indices=indices(device, a.column_indices),
    )


@overload
def dense(device: Device, matrix: np.ndarray) -> Dense[f32]: ...
@overload
def dense[F: FloatingPoint](device: Device, matrix: np.ndarray, dtype: type[F]) -> Dense[F]: ...
def dense(device: Device, matrix: np.ndarray, dtype: type[FloatingPoint] = f32) -> Dense:
    a = dense_arrays(matrix, dtype)
    return Dense(values=floats(device, a.values, dtype), rows=u32(a.rows), columns=u32(a.columns))
