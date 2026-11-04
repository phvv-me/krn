"""Our own kernels behind `lib.matmul`, `lib.spmv` and `lib.spmm`, launched on the stream the call was given.

The multiply is the kernel of `lib_kernels`: it takes every epilogue (SILU included, which cuBLASLt lacks) and
accumulates in float32. The sparse products are the kernels of `sparse`, which run every format of that module. They
run wherever the runtime has a backend and are the candidates `krn.configure(lib="ours")` leaves. Vendor adapters
are checked against these references.
"""

from functools import cache
from hashlib import sha256
from pathlib import Path

from krn.data.sparse import Coo, clear, spmm, spmv, spmv_coo, spmv_vector
from krn.facts.targets import Feature, Target
from krn.frontend.geometry import Items
from krn.frontend.numerics import bf16, f16, f32, f64, name_of, widens_to
from krn.runtime.api import Stream, default_device

from .lib import (
    Determinism,
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
from .lib_kernels import MatmulTiling, matmul_epilogue

FLAGS = {  # which constants an epilogue sets
    Epilogue.NONE: {},
    Epilogue.BIAS: {"bias": True},
    Epilogue.RELU: {"relu": True},
    Epilogue.GELU: {"gelu": True},
    Epilogue.BIAS_RELU: {"bias": True, "relu": True},
    Epilogue.BIAS_GELU: {"bias": True, "gelu": True},
    Epilogue.SILU: {"silu": True},
}


@cache
def _kernel_version() -> str:
    root = Path(__file__).parent
    return sha256(b"".join((root / name).read_bytes() for name in ("lib_kernels.py", "../data/sparse.py"))).hexdigest()[
        :16
    ]


class _KernelVersion:
    def library_version(self) -> str:
        return _kernel_version()


class Ours(_KernelVersion, MatmulCandidate):
    name, op, origin = "ours", Op.MATMUL, Origin.OURS

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if len(set(problem.dtypes)) != 1 or problem.dtypes[0] not in (f16, f32):
            return (
                "our matmul_epilogue kernel takes one F for a, b, bias and out: "
                "all dtypes must be the same f16 or f32; "
                f"got {', '.join(name_of(d) for d in problem.dtypes)}"
            )
        return None

    def run(self, problem: Problem, operands: Operands, stream: Stream) -> None:
        a, b, out = operands.a, operands.b, operands.out
        bias = operands.bias or default_device().empty(a.dtype, 1)  # read only when the epilogue has a bias
        tiling = MatmulTiling(**FLAGS[problem.epilogue])
        with stream:
            matmul_epilogue[tiling](a, b, bias, out)


def sparse_refusal(problem: Problem, target: Target) -> str | None:
    names = ", ".join(name_of(d) for d in problem.dtypes)
    if bf16 in problem.dtypes:
        return f"our device backends have no bf16 yet, got {names}"
    if not set(problem.dtypes) <= {f16, f32, f64}:
        return f"our sparse kernels take f16, f32 and f64, got {names}"
    if f64 in problem.dtypes and not target.has(Feature.FLOAT64):
        return f"our sparse kernels need FLOAT64 for {names}; {target.arch} lacks it"
    if not all(widens_to(dtype, problem.dtypes[2]) for dtype in problem.dtypes[:2]):
        return f"our sparse kernels need both inputs to convert losslessly to the output dtype, got {names}"
    return None


class OursSpmv(_KernelVersion, SpmvCandidate):
    """A thread to a row, or a warp to a row when the average row is at least a warp long, and one atomic add per
    entry for `Coo`. On the ten matrices of `experiments/sparse_bench.py` the rule picked the faster of the two
    kernels every time."""

    name, op, origin = "ours", Op.SPMV, Origin.OURS

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if reason := sparse_refusal(problem, target):
            return reason
        if problem.layout == "Coo" and problem.options.determinism is Determinism.STABLE:
            return "our COO float atomics add in any order and are not bitwise deterministic under STABLE"
        if problem.layout == "Coo" and problem.dtypes[2] == f16:
            return "our COO kernel needs atomic.add, which has no f16 output support yet"
        return None

    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        rows, _, entries = problem.shape
        target = default_device().target
        if target.has(Feature.WARP_SHUFFLE) and entries >= rows * target.warp_size:
            self._warp(problem, operands, stream)
        else:
            self._thread(problem, operands, stream)

    def _thread(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        matrix, x, out = operands.matrix, operands.x, operands.out
        with stream:
            if isinstance(matrix, Coo):
                clear(out)
                spmv_coo(matrix, x, out)
            else:
                spmv(matrix, x, out)

    def _warp(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        if isinstance(operands.matrix, Coo):
            self._thread(problem, operands, stream)
            return
        with stream:
            spmv_vector(
                operands.matrix,
                operands.x,
                operands.out,
                launch=Items(problem.shape[0] * default_device().target.warp_size),
            )


class OursSpmvThread(OursSpmv):
    name, contender = "ours:thread", True

    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        self._thread(problem, operands, stream)


class OursSpmvWarp(OursSpmv):
    name, contender = "ours:warp", True

    def refusal(self, problem: Problem, target: Target) -> str | None:
        return super().refusal(problem, target) or (
            None if target.has(Feature.WARP_SHUFFLE) else "our warp SpMV needs WARP_SHUFFLE"
        )

    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None:
        self._warp(problem, operands, stream)


class OursSpmm(_KernelVersion, SpmmCandidate):
    name, op, origin = "ours", Op.SPMM, Origin.OURS

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if reason := sparse_refusal(problem, target):
            return reason
        return "our spmm takes the row formats: Coo has no row to own its entries" if problem.layout == "Coo" else None

    def run(self, problem: Problem, operands: SpmmOperands, stream: Stream) -> None:
        if isinstance(operands.matrix, Coo):
            raise TypeError("Coo was refused above")
        with stream:
            spmm(operands.matrix, operands.x, operands.out)


CANDIDATES = [Ours(), OursSpmv(), OursSpmm(), OursSpmvThread(), OursSpmvWarp()]
