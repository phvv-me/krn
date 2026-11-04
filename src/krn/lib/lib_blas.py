"""A BLAS with the CBLAS interface as a `lib.matmul` candidate: Accelerate on a Mac, MKL on an Intel machine.

`cblas_sgemm` multiplies float32 matrices that sit in host memory and has no epilogue, so a problem with one, or in
another type, is refused with that reason and a candidate that can run it takes over.
"""

import ctypes
from typing import ClassVar

import numpy as np

from krn.facts.targets import CPU, Target
from krn.frontend.numerics import f32
from krn.hw.libraries import Libraries, LibraryHandle
from krn.runtime.api import Stream

from .lib import Epilogue, MatmulCandidate, Op, Operands, Origin, Problem

ROW_MAJOR, NO_TRANSPOSE = 101, 111


class Cblas(MatmulCandidate):
    op, origin = Op.MATMUL, Origin.VENDOR
    path: ClassVar[str]  # where the loader finds the library

    @property
    def library(self) -> LibraryHandle:
        return Libraries.load(self.path).handle

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if not isinstance(target, CPU):
            return f"{self.name} computes in host memory, and {target.arch} tensors are not there"
        if problem.dtypes != (f32, f32, f32):
            return f"cblas_sgemm takes f32, not {', '.join(d.__name__ for d in problem.dtypes)}"
        if problem.epilogue is not Epilogue.NONE:
            return f"{self.name} has no {problem.epilogue} epilogue"
        return None

    def run(self, problem: Problem, operands: Operands, stream: Stream) -> None:
        a, b, out = np.asarray(operands.a), np.asarray(operands.b), np.asarray(operands.out)
        (m, n, k), pointer = problem.shape, ctypes.c_void_p
        sgemm = self.library.cblas_sgemm
        sgemm.argtypes = (
            [ctypes.c_int] * 3
            + [ctypes.c_int] * 3
            + [
                ctypes.c_float,
                pointer,
                ctypes.c_int,
                pointer,
                ctypes.c_int,
                ctypes.c_float,
                pointer,
                ctypes.c_int,
            ]
        )
        sgemm.restype = None
        sgemm(
            ROW_MAJOR,
            NO_TRANSPOSE,
            NO_TRANSPOSE,
            m,
            n,
            k,
            1.0,
            a.ctypes.data,
            k,
            b.ctypes.data,
            n,
            0.0,
            out.ctypes.data,
            n,
        )
