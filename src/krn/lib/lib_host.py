"""The products on the host, for tensors that live in host memory: numpy does them, and the call does not return before
it is done, so `run` never enqueues. It is the last resort of the preference lists.

A sparse matrix is read as its (row, column, value) triples, which every format has, and summed in float64 by `add.at`,
so this is also the reference the other candidates are measured against, written independently of the format methods
that the kernels use.
"""

from math import erf

import numpy as np

from krn.data.sparse import Bsr, Coo, Csr, Dense, Ell, Rows
from krn.facts.targets import CPU, Target
from krn.frontend.numerics import FloatingPoint
from krn.runtime.api import Stream

from .lib import (
    Epilogue,
    MatmulCandidate,
    Op,
    Operands,
    Origin,
    Problem,
    SpmmCandidate,
    SpmmOperands,
    SpmvCandidate,
    SpmvOperands,
)

ACTIVATIONS = {
    Epilogue.RELU: lambda x: np.maximum(x, 0),
    Epilogue.BIAS_RELU: lambda x: np.maximum(x, 0),
    Epilogue.GELU: lambda x: 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2))),
    Epilogue.BIAS_GELU: lambda x: 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2))),
    Epilogue.SILU: lambda x: x / (1 + np.exp(-x)),
}


class _Host:
    def library_version(self) -> str:
        return np.__version__

    def unavailable(self, target: Target) -> str | None:
        return None if isinstance(target, CPU) else f"host memory is not reachable from {target.arch} tensors"

    def refusal(self, problem: Problem, target: Target) -> str | None:
        return None


class Host(_Host, MatmulCandidate):
    name, op, origin = "host", Op.MATMUL, Origin.HOST

    def run(self, problem: Problem, operands: Operands, stream: Stream) -> None:
        a, b, out = np.asarray(operands.a), np.asarray(operands.b), np.asarray(operands.out)
        product = a.astype(np.float64) @ b.astype(np.float64)  # the oracle for any pair of types sums in float64
        if operands.bias is not None and problem.epilogue in (
            Epilogue.BIAS,
            Epilogue.BIAS_RELU,
            Epilogue.BIAS_GELU,
        ):
            product += np.asarray(operands.bias).astype(np.float64)
        out[...] = ACTIVATIONS.get(problem.epilogue, lambda x: x)(product).astype(out.dtype)


def triples[F: FloatingPoint](
    matrix: Rows[F] | Coo[F],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The row, column and value of every stored entry, padding included (its value is zero)."""
    values = np.asarray(matrix.values)
    rows = int(matrix.rows)
    match matrix:
        case Csr():
            indptr = np.asarray(matrix.indptr)
            return np.repeat(np.arange(rows), np.diff(indptr)), np.asarray(matrix.indices), values
        case Ell():
            return np.arange(len(values)) % rows, np.asarray(matrix.indices), values
        case Bsr():
            block, indptr, tiles = (
                int(matrix.block),
                np.asarray(matrix.indptr),
                np.asarray(matrix.indices),
            )
            slot = np.arange(len(values))
            tile, inside = slot // (block * block), slot % (block * block)
            tile_row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
            return (
                tile_row[tile] * block + inside // block,
                tiles[tile] * block + inside % block,
                values,
            )
        case Coo():
            return np.asarray(matrix.row_indices), np.asarray(matrix.column_indices), values
        case Dense():
            columns = int(matrix.columns)
            return np.repeat(np.arange(rows), columns), np.tile(np.arange(columns), rows), values
        case _:
            raise TypeError(f"the host reads Csr, Ell, Bsr, Dense and Coo as triples, not {type(matrix).__name__}")


def product[F: FloatingPoint](matrix: Rows[F] | Coo[F], x: np.ndarray) -> np.ndarray:
    """matrix @ x for a vector or matrix, accumulated in float64; assignment rounds to the output dtype."""
    rows, columns, values = triples(matrix)
    terms = values.astype(np.float64).reshape(-1, *[1] * (x.ndim - 1)) * x[columns].astype(np.float64)
    result = np.zeros((int(matrix.rows), *x.shape[1:]), np.float64)
    np.add.at(result, rows, terms)
    return result


class HostSpmv(_Host, SpmvCandidate):
    name, op, origin = "host", Op.SPMV, Origin.HOST

    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        np.asarray(operands.out)[...] = product(operands.matrix, np.asarray(operands.x))


class HostSpmm(_Host, SpmmCandidate):
    name, op, origin = "host", Op.SPMM, Origin.HOST

    def run(self, problem: Problem, operands: SpmmOperands, stream: Stream) -> None:
        np.asarray(operands.out)[...] = product(operands.matrix, np.asarray(operands.x))


CANDIDATES = [Host(), HostSpmv(), HostSpmm()]
