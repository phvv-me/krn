"""Apple's Accelerate: the BLAS of the CPU cores and the matrix unit of the chip, and its Sparse Solvers library.

Sparse Solvers stores a matrix by columns (CSC), with 64-bit column starts and 32-bit row indices, and multiplies
column-major dense matrices. A CSR matrix is the CSC of its transpose, so `Csr` goes in with the transpose attribute
set and costs only the widening of its row pointers. The other formats would need a conversion and are refused.

The multiply is the exported `_SparseSpMV_Float` that the inline `SparseMultiply` of the header calls, with the structs
of `Sparse/Solve.h`. Their sizes were checked against the header with a C compiler: 32, 40 and 24 bytes.
"""

import ctypes
import platform
from collections.abc import Callable
from functools import cache

import numpy as np

from krn.data.sparse import Csr
from krn.facts.targets import CPU, Target
from krn.frontend.numerics import FloatingPoint, f32, name_of
from krn.hw.libraries import Libraries
from krn.runtime.api import Stream

from .lib import Op, Origin, Problem, SpmmCandidate, SpmmOperands, SpmvCandidate, SpmvOperands
from .lib_blas import Cblas

TRANSPOSE = 1  # the first bit of SparseAttributes_t


class Structure(ctypes.Structure):
    """SparseMatrixStructure: the pattern of a block CSC matrix."""

    _fields_ = (
        ("rowCount", ctypes.c_int), ("columnCount", ctypes.c_int), ("columnStarts", ctypes.c_void_p),
        ("rowIndices", ctypes.c_void_p), ("attributes", ctypes.c_uint32), ("blockSize", ctypes.c_uint8),
    )  # fmt: skip


class SparseMatrix(ctypes.Structure):
    """SparseMatrix_Float."""

    _fields_ = (("structure", Structure), ("data", ctypes.c_void_p))


class DenseMatrix(ctypes.Structure):
    """DenseMatrix_Float, column-major."""

    _fields_ = (
        ("rowCount", ctypes.c_int), ("columnCount", ctypes.c_int), ("columnStride", ctypes.c_int),
        ("attributes", ctypes.c_uint32), ("data", ctypes.c_void_p),
    )  # fmt: skip

    @classmethod
    def of(cls, array: np.ndarray) -> DenseMatrix:
        """A float32 matrix in Fortran order."""
        return cls(array.shape[0], array.shape[1], array.shape[0], 0, array.ctypes.data)


class Accelerate(Cblas):
    name = "accelerate"
    path = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    version = platform.mac_ver()[0]


@cache
def spmv_float() -> Callable[[float, SparseMatrix, DenseMatrix, bool, DenseMatrix], None]:
    function = Libraries.load("accelerate").handle._SparseSpMV_Float
    function.argtypes = (ctypes.c_float, SparseMatrix, DenseMatrix, ctypes.c_bool, DenseMatrix)
    function.restype = None
    return function


def multiply[F: FloatingPoint](matrix: Csr[F], x: np.ndarray, out: np.ndarray) -> None:
    """`out = matrix @ x` for float32 `x` of shape (columns,) or (columns, n), copied to column-major and back."""
    pointers = np.asarray(matrix.indptr).astype(np.int64)
    indices = np.ascontiguousarray(np.asarray(matrix.indices)).view(np.int32)
    values = np.asarray(matrix.values)
    dense = np.asfortranarray(x.reshape(len(x), -1))
    result = np.empty((int(matrix.rows), dense.shape[1]), np.float32, order="F")
    structure = Structure(
        int(matrix.columns),
        int(matrix.rows),
        pointers.ctypes.data,
        indices.ctypes.data,
        TRANSPOSE,
        1,
    )
    spmv_float()(
        1.0,
        SparseMatrix(structure, values.ctypes.data),
        DenseMatrix.of(dense),
        False,
        DenseMatrix.of(result),
    )
    out[...] = result.reshape(out.shape)


def refusal(problem: Problem, target: Target) -> str | None:
    if not isinstance(target, CPU):
        return f"accelerate computes in host memory, and {target.arch} tensors are not there"
    if problem.dtypes != (f32, f32, f32):
        return "Accelerate Sparse here takes f32 for the matrix, x and out, got " + ", ".join(
            map(name_of, problem.dtypes)
        )
    if problem.layout != "Csr":
        return f"Accelerate Sparse stores CSC, of which Csr is the transpose; {problem.layout} would need a conversion"
    return None


class AccelerateSpmv(SpmvCandidate):
    name, op, origin, version = "accelerate", Op.SPMV, Origin.VENDOR, platform.mac_ver()[0]

    def refusal(self, problem: Problem, target: Target) -> str | None:
        return refusal(problem, target)

    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        if not isinstance(operands.matrix, Csr):
            raise TypeError(f"{problem.layout} was refused above")
        multiply(operands.matrix, np.asarray(operands.x), np.asarray(operands.out))


class AccelerateSpmm(SpmmCandidate):
    name, op, origin, version = "accelerate", Op.SPMM, Origin.VENDOR, platform.mac_ver()[0]

    def refusal(self, problem: Problem, target: Target) -> str | None:
        return refusal(problem, target)

    def run(self, problem: Problem, operands: SpmmOperands, stream: Stream) -> None:
        if not isinstance(operands.matrix, Csr):
            raise TypeError(f"{problem.layout} was refused above")
        multiply(operands.matrix, np.asarray(operands.x), np.asarray(operands.out))


CANDIDATES = [Accelerate(), AccelerateSpmv(), AccelerateSpmm()]
