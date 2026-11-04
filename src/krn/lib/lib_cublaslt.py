"""Row-major a @ b is column-major b^T @ a^T, so the operands and their layouts are swapped.

Each thread caches descriptors and the top heuristic by dtypes, shape, epilogue, pointer alignments and numerical
options, with workspace per stream. Capture requires an eagerly warmed plan. SILU has no cuBLASLt epilogue; FP8
requires TN and a transposed b, which this row-major API cannot express. Other undocumented triples are refused.
Source: NVIDIA cuBLAS 13.2 reference, section 3.4.17 (R), read November 1, 2026.
"""

import ctypes
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache, cached_property

try:
    from nvmath.bindings import cublas, cublasLt
except ImportError, OSError:
    cublas = cublasLt = None

from krn.facts.targets import Nvidia, Target
from krn.frontend.numerics import (
    FloatingPoint,
    bf16,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f16,
    f32,
    f64,
    name_of,
)
from krn.runtime.api import Stream

from . import lib_nvidia
from .lib import (
    Compute,
    Declined,
    Determinism,
    Epilogue,
    LibOptions,
    MatmulCandidate,
    Op,
    Operands,
    Origin,
    Problem,
)

type Pointers = tuple[int, int, int, int | None]
_FORMATS: dict[tuple[type[FloatingPoint], ...], tuple[str, type[ctypes.c_float | ctypes.c_double]]] = {
    (f16, f16, f16): ("COMPUTE_32F", ctypes.c_float),
    (bf16, bf16, bf16): ("COMPUTE_32F", ctypes.c_float),
    (f16, f16, f32): ("COMPUTE_32F", ctypes.c_float),
    (bf16, bf16, f32): ("COMPUTE_32F", ctypes.c_float),
    (f32, f32, f32): ("COMPUTE_32F_PEDANTIC", ctypes.c_float),
    (f64, f64, f64): ("COMPUTE_64F", ctypes.c_double),
}
_EPILOGUES = {
    Epilogue.NONE: "DEFAULT",
    Epilogue.BIAS: "BIAS",
    Epilogue.RELU: "RELU",
    Epilogue.GELU: "GELU",
    Epilogue.BIAS_RELU: "RELU_BIAS",
    Epilogue.BIAS_GELU: "GELU_BIAS",
}
_FP8 = {f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz}
_COMPUTES: dict[tuple[Compute, tuple[type[FloatingPoint], ...]], str] = {
    (Compute.TF32, (f32, f32, f32)): "COMPUTE_32F_FAST_TF32",
    (Compute.EMULATED, (f32, f32, f32)): "COMPUTE_32F_EMULATED_16BFX9",
    (Compute.EMULATED, (f64, f64, f64)): "COMPUTE_64F_EMULATED_FIXEDPOINT",
}


def bindings():
    if cublasLt is None:
        raise ImportError("nvmath-python[cu13] is unavailable")
    return cublasLt


def version() -> int:
    api = bindings()
    try:
        return api.get_version()
    except api.cuBLASLtError as error:
        raise RuntimeError(str(error)) from error


def _names(dtypes: tuple[type[FloatingPoint], ...]) -> str:
    return f"({', '.join(name_of(dtype) for dtype in dtypes)})"


def _refusal(problem: Problem) -> str | None:
    if any(dtype in _FP8 for dtype in problem.dtypes):
        return (
            f"cuBLASLt FP8 {_names(problem.dtypes)} requires TN "
            "(A transposed, B non transposed in column-major terms); "
            "row-major lib.matmul is NN and needs a transposed b stored as (n, k)"
        )
    if problem.dtypes not in _FORMATS:
        documented = ", ".join(_names(triple) for triple in _FORMATS)
        return f"cuBLASLt documents (A, B, D) = {documented} only; got {_names(problem.dtypes)}"
    if problem.epilogue not in _EPILOGUES:
        return f"{problem.epilogue} has no cuBLASLt epilogue"
    if problem.precision != Compute.STRICT and (problem.precision, problem.dtypes) not in _COMPUTES:
        return f"cuBLASLt compute {problem.precision} is not documented for {_names(problem.dtypes)}"
    return None


@dataclass(frozen=True, slots=True)
class PlanKey:
    dtypes: tuple[type[FloatingPoint], ...]
    compute: Compute
    shape: tuple[int, ...]
    epilogue: Epilogue
    has_bias: bool
    alignments: tuple[int, ...]
    numerics: tuple[str, int, int]

    def matches(self, problem: Problem) -> bool:
        return (
            self.dtypes == problem.dtypes
            and self.compute == problem.precision
            and self.shape == problem.shape
            and self.epilogue == problem.epilogue
            and self.numerics == problem.options.numerics_key
        )


def plan_key(problem: Problem, pointers: Pointers) -> PlanKey:
    """Pure key: changing addresses only matters when their guaranteed alignment changes."""
    return PlanKey(
        problem.dtypes,
        problem.precision,
        problem.shape,
        problem.epilogue,
        pointers[3] is not None,
        tuple(min(256, pointer & -pointer) if pointer else 0 for pointer in pointers),
        problem.options.numerics_key,
    )


def _pointers(operands: Operands) -> Pointers:
    return (
        lib_nvidia.address(operands.a),
        lib_nvidia.address(operands.b),
        lib_nvidia.address(operands.out),
        lib_nvidia.address(operands.bias) if operands.bias is not None else None,
    )


class Plan:
    def __init__(self, handle: int, problem: Problem, pointers: Pointers) -> None:
        if reason := _refusal(problem):
            raise Declined(reason)
        self.api = api = bindings()
        assert cublas is not None
        self.bias: int | None = None
        self.layouts: list[int] = []
        compute, scalar = _FORMATS[problem.dtypes]
        compute = _COMPUTES.get((problem.precision, problem.dtypes), compute)
        self.alpha, self.beta = scalar(1), scalar(0)
        try:
            with ExitStack() as cleanup:
                self.desc = api.matmul_desc_create(
                    cublas.ComputeType[compute],
                    lib_nvidia.data_type(f64 if scalar is ctypes.c_double else f32),
                )
                cleanup.callback(bindings().matmul_desc_destroy, self.desc)
                self._attribute(
                    bindings().MatmulDescAttribute.EPILOGUE,
                    ctypes.c_int(api.Epilogue[_EPILOGUES[problem.epilogue]]),
                )
                self._attribute(
                    bindings().MatmulDescAttribute.BIAS_DATA_TYPE,
                    ctypes.c_int(lib_nvidia.data_type(problem.dtypes[2])),
                )
                if problem.options.sm_count > 0:
                    self._attribute(
                        bindings().MatmulDescAttribute.SM_COUNT_TARGET,
                        ctypes.c_int(problem.options.sm_count),
                    )
                self.set_bias(pointers[3])
                m, n, k = problem.shape
                for dtype, rows, columns in (
                    (problem.dtypes[1], n, k),
                    (problem.dtypes[0], k, m),
                    (problem.dtypes[2], n, m),
                ):
                    layout = api.matrix_layout_create(lib_nvidia.data_type(dtype), rows, columns, rows)
                    cleanup.callback(bindings().matrix_layout_destroy, layout)
                    self.layouts.append(layout)
                self.preference = api.matmul_preference_create()
                cleanup.callback(bindings().matmul_preference_destroy, self.preference)
                self._prefer(
                    bindings().MatmulPreferenceAttribute.MAX_WORKSPACE_BYTES,
                    ctypes.c_uint64(problem.options.workspace_bytes),
                )
                alignments = plan_key(problem, pointers).alignments
                for attribute, alignment in (
                    (bindings().MatmulPreferenceAttribute.MIN_ALIGNMENT_A_BYTES, alignments[1]),
                    (bindings().MatmulPreferenceAttribute.MIN_ALIGNMENT_B_BYTES, alignments[0]),
                    (bindings().MatmulPreferenceAttribute.MIN_ALIGNMENT_C_BYTES, alignments[2]),
                    (bindings().MatmulPreferenceAttribute.MIN_ALIGNMENT_D_BYTES, alignments[2]),
                ):
                    self._prefer(attribute, ctypes.c_uint32(alignment))
                if problem.options.determinism == Determinism.STABLE:
                    self._prefer(
                        bindings().MatmulPreferenceAttribute.REDUCTION_SCHEME_MASK,
                        ctypes.c_uint32(bindings().ReductionScheme.NONE),
                    )
                results, found = api.MatmulHeuristicResult(), ctypes.c_int()
                left, right, out = self.layouts
                bindings().matmul_algo_get_heuristic(
                    handle,
                    self.desc,
                    left,
                    right,
                    out,
                    out,
                    self.preference,
                    1,
                    results.ptr,
                    ctypes.addressof(found),
                )
                if found.value == 0:
                    raise Declined(
                        f"no cuBLASLt algorithm for {_names(problem.dtypes)} {problem.shape} {problem.epilogue}"
                    )
                self.result = results
                if self.result.state != cublas.Status.SUCCESS:
                    raise Declined(f"no cuBLASLt algorithm for {problem}: heuristic status {self.result.state}")
                self._finalizer = weakref.finalize(self, cleanup.pop_all().close)
        except api.cuBLASLtError as error:
            if error.status in (cublas.Status.NOT_SUPPORTED, cublas.Status.INVALID_VALUE):
                raise Declined(f"no cuBLASLt algorithm for {problem}: {error}") from error
            raise

    def _attribute(self, attribute: int, value: ctypes.c_int | ctypes.c_void_p) -> None:
        self.api.matmul_desc_set_attribute(self.desc, attribute, ctypes.addressof(value), ctypes.sizeof(value))

    def _prefer(self, attribute: int, value: ctypes.c_uint32 | ctypes.c_uint64) -> None:
        self.api.matmul_preference_set_attribute(
            self.preference, attribute, ctypes.addressof(value), ctypes.sizeof(value)
        )

    def set_bias(self, pointer: int | None) -> None:
        if pointer != self.bias:
            self._attribute(self.api.MatmulDescAttribute.BIAS_POINTER, ctypes.c_void_p(pointer))
            self.bias = pointer

    def enqueue(self, handle: int, pointers: Pointers, workspace: tuple[int, int], stream: int) -> None:
        if workspace[1] < self.result.workspace_size:
            raise Declined(f"cuBLASLt needs {self.result.workspace_size} workspace bytes, got {workspace[1]}")
        if workspace[0] % 256 or (workspace[1] and not workspace[0]):
            raise Declined("cuBLASLt workspace must have a nonzero, 256-byte aligned address")
        self.set_bias(pointers[3])
        left, right, out = self.layouts
        # cublasLtMatmulHeuristicResult_t begins with its cublasLtMatmulAlgo_t.
        self.api.matmul(
            handle,
            self.desc,
            ctypes.addressof(self.alpha),
            pointers[1],
            left,
            pointers[0],
            right,
            ctypes.addressof(self.beta),
            pointers[2],
            out,
            pointers[2],
            out,
            self.result.ptr,
            workspace[0],
            workspace[1],
            stream,
        )


class Session:
    def __init__(self) -> None:
        self.plans: dict[PlanKey, Plan] = {}
        self.scratch: dict[int, lib_nvidia.Scratch] = {}

    @cached_property
    def handle(self) -> int:
        handle = bindings().create()
        self._finalizer = weakref.finalize(self, bindings().destroy, handle)
        return handle

    def plan(self, problem: Problem, pointers: Pointers) -> Plan:
        key = plan_key(problem, pointers)
        try:
            return self.plans[key]
        except KeyError:
            plan = self.plans[key] = Plan(self.handle, problem, pointers)
            return plan


_sessions = lib_nvidia.PerThread(Session)


@cache
def _wheel_unavailable() -> str | None:
    return lib_nvidia.unavailable(version)


class CublasLt(MatmulCandidate):
    precisions = frozenset(Compute)
    name, op, origin = "cublaslt", Op.MATMUL, Origin.VENDOR

    def unavailable(self, target: Target) -> str | None:
        if not isinstance(target, Nvidia):
            return f"cuBLASLt multiplies NVIDIA device memory, and {target.arch} is not NVIDIA"
        return _wheel_unavailable()

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if not isinstance(target, Nvidia):
            return f"cuBLASLt multiplies NVIDIA device memory, and {target.arch} is not NVIDIA"
        return _refusal(problem)

    def capturable(self, problem: Problem, operands: Operands | None = None) -> bool:
        if operands is not None:
            return plan_key(problem, _pointers(operands)) in _sessions().plans
        return any(key.matches(problem) for key in _sessions().plans)

    def library_version(self) -> str:
        try:
            return str(version())
        except ImportError, OSError, RuntimeError, AttributeError:
            return "unavailable"

    def run(self, problem: Problem, operands: Operands, stream: Stream) -> None:
        pointers = _pointers(operands)
        session, cuda_stream = _sessions(), lib_nvidia.handle(stream)
        plan = session.plan(problem, pointers)
        workspace = (0, 0)
        if size := plan.result.workspace_size:
            try:
                scratch = session.scratch[cuda_stream]
            except KeyError:
                scratch = session.scratch[cuda_stream] = lib_nvidia.Scratch(stream.device)
            workspace = (scratch.at_least(size), size)
        plan.enqueue(session.handle, pointers, workspace, cuda_stream)


def multiply(
    pointers: Pointers,
    shape: tuple[int, ...],
    dtype: type[FloatingPoint],
    epilogue: Epilogue = Epilogue.NONE,
    *,
    workspace: tuple[int, int] = (0, 0),
    stream: int = 0,
) -> None:
    """Direct row-major multiply with caller-owned workspace and a CUstream; planning is cached per thread."""
    problem = Problem(Op.MATMUL, (dtype,) * 3, shape, LibOptions(workspace_bytes=workspace[1]), epilogue)
    session = _sessions()
    session.plan(problem, pointers).enqueue(session.handle, pointers, workspace, stream)


CANDIDATES = [CublasLt()]
