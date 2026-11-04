"""cuSPARSE generic SpMV/SpMM (NVIDIA reference 13.2, section 8.6, and release note CUSPARSE-1897).

Each thread owns a handle and 32 plans, keyed by the shape, the algorithm and the identity of the index tensors. A plan
keeps those tensors alive, so an address in a key cannot be recycled, and owns the buffer that `preprocess` made
active: the library lets only values, x and y change after it, so values and dense operands are re-pointed per call
and a new matrix is a new plan. A second stream runs the plan on a buffer of its own, which is inactive (always
allowed, possibly slower), so two streams never share scratch. The first use of a key plans on the default stream and
waits for it, and a plan must exist before a graph capture (`capturable`): measured on an RTX 5080 with cuSPARSE 12.8,
a plan made while another stream captures leaves the library unable to enqueue into that capture.
CSR, COO and square row-major BSR blocks use signed 32-bit indices, so dimensions and stored counts above INT_MAX
are refused. Ell lacks sliced-ELL offsets and uses zero rather than -1 padding; Dense is not a sparse format.
Only documented storage/compute pairs and bitwise deterministic algorithms under STABLE are accepted.
Source: https://docs.nvidia.com/cuda/cusparse/index.html (cusparseSpMV, cusparseSpMM).
"""

import ctypes
import weakref
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

try:
    from nvmath.bindings import cusparse
except ImportError, OSError:
    cusparse = None

from krn.data.sparse import Bsr, Coo, Csr, Rows
from krn.facts.targets import Nvidia, Target
from krn.frontend.memory import Vector
from krn.frontend.numerics import FloatingPoint, bf16, f16, f32, f64, name_of, u8
from krn.runtime.api import Device, Stream

from . import lib_nvidia as nv
from .lib import (
    Declined,
    Determinism,
    Op,
    Origin,
    Problem,
    SpmmCandidate,
    SpmmOperands,
    SpmvCandidate,
    SpmvOperands,
)

PAIRS: dict[tuple[type[FloatingPoint], ...], type[FloatingPoint]] = {
    (f32, f32, f32): f32,
    (f64, f64, f64): f64,
    (f16, f16, f16): f32,
    (f16, f16, f32): f32,
    (bf16, bf16, bf16): f32,
    (bf16, bf16, f32): f32,
}
SPMV_PAIRS = {**PAIRS, (f32, f64, f64): f64}


def bindings():
    if cusparse is None:
        raise ImportError("nvmath-python[cu13] is unavailable")
    return cusparse


def version() -> int:
    """The version as cusparseGetVersion scales it, read without a handle so that a graph capture may ask."""
    api = bindings()
    try:
        # libraryPropertyType 0, 1 and 2 are MAJOR_VERSION, MINOR_VERSION and PATCH_LEVEL.
        return sum(scale * api.get_property(kind) for kind, scale in enumerate((1000, 100, 1)))
    except api.cuSPARSEError as error:
        raise RuntimeError(str(error)) from error


@dataclass(frozen=True, slots=True)
class Algorithm:
    name: str
    layouts: tuple[str, ...]
    deterministic: bool


SPMV_ALGORITHMS = {
    a.name: a
    for a in (
        Algorithm("default", ("Csr", "Coo", "Bsr"), False),
        Algorithm("csr_alg1", ("Csr",), False),
        Algorithm("csr_alg2", ("Csr",), True),
        Algorithm("coo_alg1", ("Coo",), False),
        Algorithm("coo_alg2", ("Coo",), True),
        Algorithm("sell_alg1", ("Ell",), False),
        Algorithm("bsr_alg1", ("Bsr",), True),
    )
}
SPMM_ALGORITHMS = {
    a.name: a
    for a in (
        Algorithm("default", ("Csr", "Coo", "Bsr"), False),
        Algorithm("csr_alg1", ("Csr",), False),
        Algorithm("csr_alg2", ("Csr",), False),
        Algorithm("csr_alg3", ("Csr",), True),
        Algorithm("coo_alg1", ("Coo",), False),
        Algorithm("coo_alg2", ("Coo",), True),
        Algorithm("coo_alg3", ("Coo",), False),
        Algorithm("coo_alg4", ("Coo",), False),
        Algorithm("bsr_alg1", ("Bsr",), True),
    )
}


@dataclass(frozen=True, slots=True)
class _Key:
    op: Op
    layout: str
    dtypes: tuple[type[FloatingPoint], ...]
    shape: tuple[int, ...]
    algorithm: Algorithm
    pattern: tuple[int, ...]


def pattern_of(matrix: Rows | Coo) -> tuple[Vector, ...]:
    """The index tensors that preprocessing is tied to."""
    match matrix:
        case Csr() | Bsr():
            return matrix.indptr, matrix.indices
        case Coo():
            return matrix.row_indices, matrix.column_indices
        case _:
            raise TypeError(f"cuSPARSE has no descriptor for {type(matrix).__name__}")


def plan_key(problem: Problem, operands: SpmvOperands | SpmmOperands, algorithm: Algorithm) -> _Key:
    matrix = operands.matrix
    pattern = tuple(nv.address(index) for index in pattern_of(matrix))
    block = (int(matrix.block),) if isinstance(matrix, Bsr) else ()
    return _Key(problem.op, problem.layout, problem.dtypes, problem.shape, algorithm, (*pattern, *block))


class Plan:
    """Descriptors and the active preprocessing buffer for one pattern and algorithm."""

    def __init__(
        self,
        session: Session,
        device: Device,
        problem: Problem,
        operands: SpmvOperands | SpmmOperands,
        algorithm: Algorithm,
    ) -> None:
        self.api = bindings()
        self.spmv = problem.op is Op.SPMV
        self.algorithm = algorithm
        self.algorithm_id = (self.api.SpMVAlg if self.spmv else self.api.SpMMAlg)[algorithm.name.upper()]
        self.xv = self.yv = self.xm = self.ym = 0
        self.addresses = (
            nv.address(operands.matrix.values),
            nv.address(operands.x),
            nv.address(operands.out),
        )
        compute = (SPMV_PAIRS if problem.op is Op.SPMV else PAIRS)[problem.dtypes]
        self.compute = nv.data_type(compute)
        scalar = ctypes.c_double if compute == f64 else ctypes.c_float
        self.alpha, self.beta = scalar(1), scalar(0)
        self.pattern = pattern_of(operands.matrix)
        self.owner: int | None = None  # the stream that runs on the active buffer
        self.spare: dict[int, Vector[u8]] = {}
        with ExitStack() as resources, nv.relaxed_capture():
            session.on(device.default_stream)
            self._matrix(problem, operands)
            resources.callback(self.api.destroy_sp_mat, self.matrix)
            rows, columns, _ = problem.shape[:3]
            if problem.op is Op.SPMV:
                for name, size, address, dtype in (
                    ("xv", columns, self.addresses[1], problem.dtypes[1]),
                    ("yv", rows, self.addresses[2], problem.dtypes[2]),
                ):
                    descriptor = self.api.create_dn_vec(size, address, nv.data_type(dtype))
                    setattr(self, name, descriptor)
                    resources.callback(self.api.destroy_dn_vec, descriptor)
            else:
                n = problem.shape[3]
                for name, size, address, dtype in (
                    ("xm", columns, self.addresses[1], problem.dtypes[1]),
                    ("ym", rows, self.addresses[2], problem.dtypes[2]),
                ):
                    descriptor = self.api.create_dn_mat(size, n, n, address, nv.data_type(dtype), self.api.Order.ROW)
                    setattr(self, name, descriptor)
                    resources.callback(self.api.destroy_dn_mat, descriptor)
            self.size = self._size(session.handle)
            self.active = nv.allocate(device, self.size)
            self.active_address = nv.address(self.active)
            if problem.layout == "Csr" and (problem.op is Op.SPMV or algorithm.name in ("csr_alg1", "csr_alg3")):
                self._call(session.handle, self.active_address, preprocess=True)
            nv.synchronize(device.default_stream)
            self._finalizer = weakref.finalize(self, resources.pop_all().close)

    def _matrix(self, problem: Problem, operands: SpmvOperands | SpmmOperands) -> None:
        rows, columns, entries = problem.shape[:3]
        matrix, values, dtype = operands.matrix, self.addresses[0], nv.data_type(problem.dtypes[0])
        match matrix:
            case Csr():
                self.matrix = self.api.create_csr(
                    rows,
                    columns,
                    entries,
                    nv.address(matrix.indptr),
                    nv.address(matrix.indices),
                    values,
                    self.api.IndexType.INDEX_32I,
                    self.api.IndexType.INDEX_32I,
                    self.api.IndexBase.ZERO,
                    dtype,
                )
            case Coo():
                self.matrix = self.api.create_coo(
                    rows,
                    columns,
                    entries,
                    nv.address(matrix.row_indices),
                    nv.address(matrix.column_indices),
                    values,
                    self.api.IndexType.INDEX_32I,
                    self.api.IndexBase.ZERO,
                    dtype,
                )
            case Bsr():
                block = int(matrix.block)
                self.matrix = self.api.create_bsr(
                    rows // block,
                    columns // block,
                    matrix.indices.shape[0],
                    block,
                    block,
                    nv.address(matrix.indptr),
                    nv.address(matrix.indices),
                    values,
                    self.api.IndexType.INDEX_32I,
                    self.api.IndexType.INDEX_32I,
                    self.api.IndexBase.ZERO,
                    dtype,
                    self.api.Order.ROW,
                )
            case _:
                raise TypeError(f"cuSPARSE has no descriptor for {type(matrix).__name__}")

    def _mv_arguments(self, handle: int) -> tuple[int, int, int, int, int, int, int, int, int]:
        alpha, beta = ctypes.addressof(self.alpha), ctypes.addressof(self.beta)
        op = self.api.Operation.NON_TRANSPOSE
        return (
            handle,
            op,
            alpha,
            self.matrix,
            self.xv,
            beta,
            self.yv,
            self.compute,
            self.algorithm_id,
        )

    def _mm_arguments(self, handle: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        alpha, beta = ctypes.addressof(self.alpha), ctypes.addressof(self.beta)
        op = self.api.Operation.NON_TRANSPOSE
        return (
            handle,
            op,
            op,
            alpha,
            self.matrix,
            self.xm,
            beta,
            self.ym,
            self.compute,
            self.algorithm_id,
        )

    def _size(self, handle: int) -> int:
        if self.spmv:
            return self.api.sp_mv_buffer_size(*self._mv_arguments(handle))
        return self.api.sp_mm_buffer_size(*self._mm_arguments(handle))

    def _call(self, handle: int, buffer: int, *, preprocess: bool = False) -> None:
        if self.spmv:
            mv = self.api.sp_mv_preprocess if preprocess else self.api.sp_mv
            mv(*self._mv_arguments(handle), buffer)
        else:
            mm = self.api.sp_mm_preprocess if preprocess else self.api.sp_mm
            mm(*self._mm_arguments(handle), buffer)

    def buffer(self, stream: Stream, cuda_stream: int) -> int:
        """The scratch for a stream: the active buffer for the first one, a buffer of its own for any other."""
        if self.owner in (None, cuda_stream):
            self.owner = cuda_stream
            return self.active_address
        try:
            return nv.address(self.spare[cuda_stream])
        except KeyError:
            spare = self.spare[cuda_stream] = nv.allocate(stream.device, self.size)
            return nv.address(spare)

    def run(self, handle: int, operands: SpmvOperands | SpmmOperands, buffer: int) -> None:
        addresses = (
            nv.address(operands.matrix.values),
            nv.address(operands.x),
            nv.address(operands.out),
        )
        if addresses[0] != self.addresses[0]:
            self.api.sp_mat_set_values(self.matrix, addresses[0])
        if self.spmv:
            if addresses[1] != self.addresses[1]:
                self.api.dn_vec_set_values(self.xv, addresses[1])
            if addresses[2] != self.addresses[2]:
                self.api.dn_vec_set_values(self.yv, addresses[2])
        else:
            if addresses[1] != self.addresses[1]:
                self.api.dn_mat_set_values(self.xm, addresses[1])
            if addresses[2] != self.addresses[2]:
                self.api.dn_mat_set_values(self.ym, addresses[2])
        self.addresses = addresses
        self._call(handle, buffer)


class Session:
    """A thread's handle and LRU; no handle or preprocessing buffer crosses threads."""

    def __init__(self) -> None:
        self.api = bindings()
        self.stream: int | None = None
        self.plans: OrderedDict[_Key, Plan] = OrderedDict()
        with nv.relaxed_capture():
            self.handle = self.api.create()
            self._finalizer = weakref.finalize(self, self.api.destroy, self.handle)
            self.api.set_pointer_mode(self.handle, self.api.PointerMode.HOST)

    def on(self, stream: Stream) -> None:
        address = nv.handle(stream)
        if address != self.stream:
            self.api.set_stream(self.handle, address)
            self.stream = address

    def planned(self, problem: Problem, operands: SpmvOperands | SpmmOperands | None, algorithm: Algorithm) -> bool:
        """Whether a plan exists for these operands, or for any pattern of the problem without operands."""
        if operands is not None:
            return plan_key(problem, operands, algorithm) in self.plans
        return any(
            key.layout == problem.layout and key.dtypes == problem.dtypes and key.shape == problem.shape
            for key in self.plans
            if key.algorithm == algorithm
        )

    def run(
        self,
        problem: Problem,
        operands: SpmvOperands | SpmmOperands,
        stream: Stream,
        algorithm: Algorithm,
    ) -> None:
        key = plan_key(problem, operands, algorithm)
        try:
            plan = self.plans[key]
        except KeyError:
            plan = self.plans[key] = self._plan(problem, operands, stream, algorithm)
            if len(self.plans) > 32:
                self.plans.popitem(last=False)
        self.plans.move_to_end(key)
        self.on(stream)
        plan.run(self.handle, operands, plan.buffer(stream, nv.handle(stream)))

    def _plan(
        self,
        problem: Problem,
        operands: SpmvOperands | SpmmOperands,
        stream: Stream,
        algorithm: Algorithm,
    ) -> Plan:
        try:
            return Plan(self, stream.device, problem, operands, algorithm)
        except self.api.cuSPARSEError as error:
            if error.status in (
                self.api.Status.NOT_SUPPORTED,
                self.api.Status.INVALID_VALUE,
                self.api.Status.ARCH_MISMATCH,
                self.api.Status.INSUFFICIENT_RESOURCES,
            ):
                raise Declined(str(error)) from error
            raise


_session = nv.PerThread(Session)


@cache
def wheel_unavailable() -> str | None:
    return nv.unavailable(version)


class _Cusparse:
    origin = Origin.VENDOR
    algorithm: ClassVar[str] = "default"
    op: ClassVar[Op]

    def unavailable(self, target: Target) -> str | None:
        if not isinstance(target, Nvidia):
            return f"cuSPARSE multiplies NVIDIA device memory, and {target.arch} is not NVIDIA"
        return wheel_unavailable()

    def capturable(self, problem: Problem, operands: SpmvOperands | SpmmOperands | None) -> bool:
        return _session().planned(problem, operands, self.algorithm_for(problem))

    def library_version(self) -> str:
        try:
            return str(version())
        except ImportError, RuntimeError, OSError:
            return "unavailable"

    def algorithm_for(self, problem: Problem) -> Algorithm:
        table = SPMV_ALGORITHMS if self.op is Op.SPMV else SPMM_ALGORITHMS
        name = self.algorithm
        if name == "default":
            if problem.layout == "Bsr":
                name = "bsr_alg1"
            elif problem.options.determinism is Determinism.STABLE:
                name = "coo_alg2" if problem.layout == "Coo" else "csr_alg2" if self.op is Op.SPMV else "csr_alg3"
        return table[name]

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if problem.layout == "Ell":
            return "cuSPARSE reads Ell as sliced ELL, but Ell lacks slice offsets and pads with 0 rather than -1"
        if problem.layout == "Dense":
            return "Dense is not a sparse format"
        pairs = SPMV_PAIRS if self.op is Op.SPMV else PAIRS
        if problem.dtypes not in pairs:
            return (
                "cuSPARSE documents (A, X, Y) = "
                + ", ".join(f"({', '.join(map(name_of, pair))})" for pair in pairs)
                + " pairs only; A and X must share a type except f32 with f64; "
                + f"got ({', '.join(map(name_of, problem.dtypes))})"
            )
        if any(size > 2**31 - 1 for size in problem.shape[:3]):
            return "cuSPARSE signed 32-bit indices require rows, columns and stored entries <= 2**31 - 1"
        algorithm = self.algorithm_for(problem)
        if problem.layout not in algorithm.layouts:
            return f"cuSPARSE {algorithm.name} takes {', '.join(algorithm.layouts)}, not {problem.layout}"
        if problem.options.determinism is Determinism.STABLE and not algorithm.deterministic:
            return f"cuSPARSE {algorithm.name} is not bitwise deterministic under STABLE"
        return None


class CusparseSpmv(_Cusparse, SpmvCandidate):
    name, op = "cusparse", Op.SPMV

    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        _session().run(problem, operands, stream, self.algorithm_for(problem))


class CusparseSpmm(_Cusparse, SpmmCandidate):
    name, op = "cusparse", Op.SPMM

    def run(self, problem: Problem, operands: SpmmOperands, stream: Stream) -> None:
        _session().run(problem, operands, stream, self.algorithm_for(problem))


class SpmvCsrAlg1(CusparseSpmv):
    name, algorithm, contender = "cusparse:csr_alg1", "csr_alg1", True


class SpmvCsrAlg2(CusparseSpmv):
    name, algorithm, contender = "cusparse:csr_alg2", "csr_alg2", True


class SpmvCooAlg1(CusparseSpmv):
    name, algorithm, contender = "cusparse:coo_alg1", "coo_alg1", True


class SpmvCooAlg2(CusparseSpmv):
    name, algorithm, contender = "cusparse:coo_alg2", "coo_alg2", True


class SpmmCsrAlg1(CusparseSpmm):
    name, algorithm, contender = "cusparse:csr_alg1", "csr_alg1", True


class SpmmCsrAlg2(CusparseSpmm):
    name, algorithm, contender = "cusparse:csr_alg2", "csr_alg2", True


class SpmmCsrAlg3(CusparseSpmm):
    name, algorithm, contender = "cusparse:csr_alg3", "csr_alg3", True


class SpmmCooAlg1(CusparseSpmm):
    name, algorithm, contender = "cusparse:coo_alg1", "coo_alg1", True


class SpmmCooAlg2(CusparseSpmm):
    name, algorithm, contender = "cusparse:coo_alg2", "coo_alg2", True


class SpmmCooAlg3(CusparseSpmm):
    name, algorithm, contender = "cusparse:coo_alg3", "coo_alg3", True


class SpmmCooAlg4(CusparseSpmm):
    name, algorithm, contender = "cusparse:coo_alg4", "coo_alg4", True


CANDIDATES = [
    CusparseSpmv(),
    CusparseSpmm(),
    SpmvCsrAlg1(),
    SpmvCsrAlg2(),
    SpmvCooAlg1(),
    SpmvCooAlg2(),
    SpmmCsrAlg1(),
    SpmmCsrAlg2(),
    SpmmCsrAlg3(),
    SpmmCooAlg1(),
    SpmmCooAlg2(),
    SpmmCooAlg3(),
    SpmmCooAlg4(),
]
