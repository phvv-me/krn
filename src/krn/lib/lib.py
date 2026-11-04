"""One call reaches the best implementation on the machine: a vendor library, one of our kernels, or the host.

    lib.matmul(a, b, epilogue=Epilogue.BIAS_GELU, bias=bias)
    lib.spmv(csr, x)

A sparse matrix is a struct of dense tensors (`krn.data.sparse`), and a vendor sparse library is one more candidate
next to our kernels, which run every format.

Candidates are classes in vendor adapter modules (`lib_cublaslt`, `lib_cusparse`, `lib_hipblaslt`, `lib_mps`,
`lib_cpu`, `lib_ours`), imported by name when the machine lists their library, or when it is NVIDIA, whose wheels the
loader cannot see: a candidate that cannot run says why in `unavailable` or `refusal`, and one that finds out only
while planning raises `Declined`. A `contender` is a variant (one cuSPARSE algorithm, one of our two SpMV kernels) that
no static order picks and the tuner times against the leaders on the caller's own operands.

Use krn.configure(lib=krn.Lib(mode="ours", matmul="vendor", tune="off")) or krn.config.lib.matmul.
Each dispatch reads the current immutable snapshot; library filters have no environment form.
"""

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from functools import cache
from importlib import import_module
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, ClassVar, Self, overload

from krn.cache.tuning import TuningDb as TuningDb
from krn.compiler.errors import PrecisionWarning, Span, TargetError
from krn.compiler.performance import warn_once
from krn.facts.targets import Nvidia, Target
from krn.frontend.memory import Matrix, Tensor, Vector
from krn.frontend.numerics import FloatingPoint, name_of, product_type
from krn.hw.system import System
from krn.runtime.api import Device, Stream, current_stream, default_device
from krn.runtime.settings import Settings, current_settings, retained

if TYPE_CHECKING:
    from krn.data.sparse import Coo, Rows

log = logging.getLogger(__name__)
TUNE_MILLISECONDS = 20.0  # the length of one timed window of the tuner
TUNE_WINDOWS = 4


class Op(StrEnum):
    MATMUL = auto()
    CHOLESKY = auto()
    SOLVE_TRIANGULAR = auto()
    FFT = auto()
    SPMV = auto()
    SPMM = auto()


class Mode(StrEnum):
    """The library filter. VENDOR and OURS filter the candidates by origin, AUTO leaves it to the policy."""

    AUTO = auto()
    VENDOR = auto()
    OURS = auto()
    CUBLASLT = auto()
    CUBLAS = auto()
    CUSPARSE = auto()
    HIPBLASLT = auto()
    ACCELERATE = auto()
    MPS = auto()
    MKL = auto()
    ONEDNN = auto()
    HOST = auto()


class Origin(StrEnum):
    VENDOR = auto()
    OURS = auto()
    HOST = auto()


class Epilogue(StrEnum):
    """Fused after the multiply. cuBLASLt has no SILU epilogue (nvmath 1.0 header): there only ours fuses it."""

    NONE = auto()
    BIAS = auto()
    RELU = auto()
    GELU = auto()
    BIAS_RELU = auto()
    BIAS_GELU = auto()
    SILU = auto()


class Determinism(StrEnum):
    ANY = auto()
    STABLE = auto()  # bitwise equal run to run on one machine, one library version


class Compute(StrEnum):
    STRICT = auto()  # the accumulator the dtypes name
    TF32 = auto()  # cuBLAS 32F_FAST_TF32, oneDNN fpmath tf32, MPS relaxed precision
    EMULATED = auto()  # cuBLAS BF16x9 (fp32) and fixed-point (fp64), Ozaki schemes


@dataclass(frozen=True, slots=True, kw_only=True)
class LibOptions:
    """Plan-time options. The target fills the defaults; `numerics_key` is the part that changes results or algorithms.

    workspace_bytes: cuBLASLt maximum workspace for the heuristic; hipBLASLt defaults to 0, so it must be set.
    sm_count: carve the multiply to this many SMs or CUs (nvmath `sm_count_target`), 0 for all.
    candidates: heuristic results kept for autotuning, not keyed.
    threads: CPU threads, 0 for all cores; keyed for tuning only.
    """

    determinism: Determinism = Determinism.ANY
    workspace_bytes: int = 32 << 20
    sm_count: int = 0
    candidates: int = 8
    threads: int = 0

    @property
    def numerics_key(self) -> tuple[str, int, int]:
        return (
            self.determinism,
            self.workspace_bytes,
            self.sm_count,
        )

    @classmethod
    def of(cls, target: Target) -> Self:
        small = isinstance(target, Nvidia) and target.sm < 90  # cuBLAS docs: 4 MiB below sm_90, 32 MiB from sm_90 on
        return cls(workspace_bytes=(4 << 20) if small else 32 << 20)


@dataclass(frozen=True, slots=True)
class OperationOperands(ABC):
    """The operands of one operation: its inputs and the `out` it writes.

    A subclass declares `out = field()`, which replaces the abstract property when the class is created (the dataclass
    then removes the assignment, so the field has no default and keeps its place in the positional order).
    """

    @property
    @abstractmethod
    def out(self) -> Tensor: ...

    def private(self, device: Device) -> Self:
        """Share inputs with the caller, but isolate writes made by the measuring thread."""
        return replace(self, out=device.empty(self.out.dtype, *self.out.shape))


@dataclass(frozen=True, slots=True)
class Operands[A: FloatingPoint, B: FloatingPoint = A, Y: FloatingPoint = A](OperationOperands):
    """The tensors of a multiply by role: `a` (m, k), `b` (k, n), the result `out` (m, n), the optional `bias` (n),
    which has the type of the result. `Operands[S]` is the uniform case."""

    a: Matrix[A]
    b: Matrix[B]
    out: Matrix[Y] = field()
    bias: Vector[Y] | None = None


@dataclass(frozen=True, slots=True)
class SpmvOperands[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](OperationOperands):
    """`out = matrix @ x`: a sparse matrix (rows, columns) times a dense vector (columns) into a vector (rows)."""

    matrix: Rows[A] | Coo[A]
    x: Vector[X]
    out: Vector[Y] = field()


@dataclass(frozen=True, slots=True)
class SpmmOperands[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](OperationOperands):
    """`out = matrix @ x`: a sparse matrix (rows, columns) times a dense matrix (columns, n) into a matrix (rows, n)."""

    matrix: Rows[A] | Coo[A]
    x: Matrix[X]
    out: Matrix[Y] = field()


@dataclass(frozen=True, slots=True)
class Problem:
    """shape is (m, n, k) for a multiply, (rows, columns, stored entries[, n]) for a sparse one, whose `layout` names
    the format (`Csr`, `Ell`, `Bsr`, `Coo`)."""

    op: Op
    dtypes: tuple[type[FloatingPoint], ...]
    shape: tuple[int, ...]
    options: LibOptions
    epilogue: Epilogue = Epilogue.NONE
    batch: int = 1
    layout: str = ""
    precision: Compute = Compute.STRICT

    @property
    def bucket(self) -> tuple[int, ...]:
        """Powers of two, but dims up to 64 stay exact: decode-sized rows select other kernels."""
        return tuple(d if d <= 64 else 1 << (d - 1).bit_length() for d in self.shape)


class Candidate[O: OperationOperands](ABC):
    """One implementation of one operation: who it is and what it refuses. The class of its operation adds `run`,
    which only enqueues and never synchronizes. Machine-wide inability to run belongs in `unavailable`, while
    `refusal` describes unsupported problems."""

    name: ClassVar[str]
    op: ClassVar[Op]
    origin: ClassVar[Origin]
    version: ClassVar[str] = ""  # fallback for adapters without a runtime library version query
    precisions: ClassVar[frozenset[Compute]] = frozenset({Compute.STRICT})
    contender: ClassVar[bool] = False  # a variant that only the tuner and a stored winner pick, never the static order

    def unavailable(self, target: Target) -> str | None:
        """Why this implementation cannot run on the machine at all, or None if available."""
        return None

    def capturable(self, problem: Problem, operands: O | None) -> bool:
        """Whether `run` may enqueue these operands while a graph is being captured, which allows no planning of
        its own: a vendor adapter answers whether it has planned them already. Without operands, whether it ever can."""
        return True

    def library_version(self) -> str:
        return self.version

    @abstractmethod
    def refusal(self, problem: Problem, target: Target) -> str | None:
        """Why this candidate cannot run the problem (dtype, layout, epilogue, architecture), or None if it can."""

    @abstractmethod
    def run(self, problem: Problem, operands: O, stream: Stream) -> None: ...


class MatmulCandidate(Candidate[Operands]):
    @abstractmethod
    def run(self, problem: Problem, operands: Operands, stream: Stream) -> None: ...


class SpmvCandidate(Candidate[SpmvOperands]):
    @abstractmethod
    def run(self, problem: Problem, operands: SpmvOperands, stream: Stream) -> None: ...


class SpmmCandidate(Candidate[SpmmOperands]):
    @abstractmethod
    def run(self, problem: Problem, operands: SpmmOperands, stream: Stream) -> None: ...


class NoCandidate(LookupError):
    """Every candidate refused. The message lists each refusal, so the fix (install a wheel, relax a flag) is in it."""


class LibraryTargetError(TargetError, NoCandidate):
    """An explicit library choice cannot run on this target."""


class Declined(RuntimeError):
    """Raised by `Candidate.run` before it enqueues anything when planning shows that it cannot take the problem after
    all (a vendor library answers NOT_SUPPORTED). The call falls to the next candidate, and the reason joins the list
    that `NoCandidate` prints."""


PREFERENCE: Mapping[tuple[Op, str], tuple[str, ...]] = {
    (Op.MATMUL, "nvidia"): ("cublaslt", "ours", "host"),
    (Op.MATMUL, "amd"): ("hipblaslt", "ours", "host"),
    (Op.MATMUL, "apple"): (
        "ours",
        "mps",
        "accelerate",
    ),  # MLX also keeps its own steel kernels over MPS on the GPU
    (Op.MATMUL, "cpu"): ("onednn", "mkl", "ours"),
    (Op.CHOLESKY, "nvidia"): ("cusolver", "ours", "host"),
    (Op.CHOLESKY, "apple"): (
        "accelerate",
        "ours",
    ),  # MPSMatrixDecompositionCholesky is the GPU option, not the default
    # Sparse, from `experiments/sparse_bench.py` on an RTX 5080 (`notes/round9/vendor-candidates-status.md`).
    # The rule: the main candidate with the lower geometric mean time over the ten matrices goes first, and a difference
    # under the 10 percent of the tuner is a tie that the vendor wins. SpMV: cuSPARSE default over ours is 1.09, a tie.
    # SpMM with 16 columns: 2.05 (ours is faster on 9 of 10, not on long rows), so ours goes first.
    # Either way the tuner then times every cuSPARSE algorithm and both our SpMV kernels on the caller's own matrix.
    # Accelerate is 1 to 14 times faster than our one thread per launch on the CPU.
    (Op.SPMV, "nvidia"): ("cusparse", "ours", "host"),
    (Op.SPMM, "nvidia"): ("ours", "cusparse", "host"),
    (Op.SPMV, "cpu"): ("accelerate", "ours", "host"),
    (Op.SPMM, "cpu"): ("accelerate", "ours", "host"),
}


@dataclass(frozen=True, slots=True)
class Choice:
    """What accepts a problem, worked out once: `eligible` is every candidate that takes it (variants included), `pool`
    those that count for the default, in the static order of the target, and `key` is where its winner is stored."""

    candidates: Mapping[Op, Sequence[Candidate]]
    order: tuple[str, ...]
    eligible: tuple[Candidate, ...]
    pool: tuple[Candidate, ...]
    key: str


@dataclass(slots=True)
class Dispatcher:
    target: Target
    candidates: Mapping[Op, Sequence[Candidate]]
    tuning: TuningDb
    mode: Mode
    per_op: Mapping[Op, Mode]
    fingerprint: str
    tune: str = "lazy"
    seen: dict[str, int] = field(default_factory=dict)
    measuring: list[Thread] = field(default_factory=list)
    choices: dict[tuple[type[Candidate], Problem], Choice] = field(default_factory=dict)
    declined: dict[tuple[type[Candidate], Problem], dict[str, str]] = field(default_factory=dict)

    def launch[O: OperationOperands](
        self, kind: type[Candidate[O]], problem: Problem, operands: O, stream: Stream
    ) -> None:
        """Choose, then enqueue on `stream`; a candidate that declines while planning is skipped from then on."""
        while True:
            candidate = self.choose(kind, problem, capturing=stream.capturing, operands=operands)
            try:
                candidate.run(problem, operands, stream)
            except Declined as reason:
                self.declined.setdefault((kind, problem), {})[candidate.name] = str(reason)
                self.choices.pop((kind, problem), None)
                log.info("lib: %s declined %s: %s", candidate.name, problem.op, reason)
            else:
                return

    def choose[O: OperationOperands](
        self,
        kind: type[Candidate[O]],
        problem: Problem,
        *,
        capturing: bool,
        operands: O | None = None,
    ) -> Candidate[O]:
        """Capability check, the user's filter, the measured winner, then the static preference of the target.

        kind: the class of candidates that run the operation (`MatmulCandidate`, `SpmvCandidate`, `SpmmCandidate`).
        """
        order = PREFERENCE.get((problem.op, self.target.vendor), ())
        choice = self.choices.get((kind, problem))
        if choice is None or choice.candidates is not self.candidates or choice.order != order:
            choice = self.choices[kind, problem] = self._accepted(kind, problem, order)
        eligible, pool = choice.eligible, choice.pool
        if capturing:
            eligible = tuple(c for c in eligible if c.capturable(problem, operands))
            pool = tuple(c for c in pool if c.capturable(problem, operands))
            if not pool:
                names = ", ".join(c.name for c in choice.pool)
                raise NoCandidate(f"{problem.op}: {names} cannot enqueue while a graph is captured; run once before")
        winner = self.tuning.winners.get(choice.key)
        if winner and (match := next((c for c in eligible if c.name == winner), None)):
            return match
        self.seen[choice.key] = seen = self.seen.get(choice.key, 0) + 1
        if (
            operands is not None
            and len(eligible) > 1
            and not capturing
            and (self.tune == "eager" or (self.tune == "lazy" and seen == 8))
        ):
            contest = [*pool, *(c for c in eligible if c.contender)]
            thread = Thread(target=self._measure, args=(choice.key, problem, contest, operands), daemon=True)
            self.measuring.append(thread)
            thread.start()  # the caller never waits
        return pool[0]

    def _accepted[O: OperationOperands](
        self, kind: type[Candidate[O]], problem: Problem, order: tuple[str, ...]
    ) -> Choice:
        candidates = [c for c in self.candidates[problem.op] if isinstance(c, kind)]
        declined = self.declined.get((kind, problem), {})
        refused = {
            c.name: declined.get(c.name)
            or c.unavailable(self.target)
            or (
                f"{c.name} has no {problem.precision} precision contract"
                if problem.precision not in c.precisions
                else None
            )
            or c.refusal(problem, self.target)
            for c in candidates
        }
        mode = self.per_op.get(problem.op, self.mode)
        library = "cublaslt" if mode is Mode.CUBLAS else mode.value
        vendors = {
            "cublaslt": ("nvidia",),
            "cusparse": ("nvidia",),
            "hipblaslt": ("amd",),
            "accelerate": ("apple", "cpu"),
            "mps": ("apple",),
            "mkl": ("cpu",),
            "onednn": ("cpu",),
        }
        if library in vendors and self.target.vendor not in vendors[library]:
            alternatives = ", ".join(PREFERENCE.get((problem.op, self.target.vendor), ("ours", "host")))
            raise LibraryTargetError(
                f"lib.{problem.op}={mode}",
                self.target.arch,
                f"{library} supports {', '.join(vendors[library])} targets, not {self.target.vendor}",
                f"use lib.mode=auto or one of {alternatives}",
            )
        wanted = [c for c in candidates if refused[c.name] is None]
        if not wanted and problem.precision is not Compute.STRICT:
            raise TargetError(
                f"matmul precision={problem.precision}",
                self.target.arch,
                "; ".join(f"{name}: {reason}" for name, reason in refused.items())
                or "no library implements this precision",
                "use precision=Compute.STRICT or select a target and library that implement this precision",
            )
        eligible = [c for c in wanted if mode is Mode.AUTO or c.origin.value == mode.value or c.name == library]
        pool = sorted(
            (c for c in eligible if not c.contender),
            key=lambda c: order.index(c.name) if c.name in order else len(order),
        )
        if not pool:
            reasons = [f"{name}: {why}" for name, why in refused.items() if why]
            excluded = [c.name for c in wanted if c not in eligible]
            message = f"{problem.op}: " + "; ".join(
                [
                    *reasons,
                    *([f"lib.mode={mode} excludes {', '.join(excluded)}"] if excluded else []),
                ]
            )
            if mode is not Mode.AUTO:
                alternatives = ", ".join(c.name for c in wanted) or "ours with a supported dtype and layout"
                raise LibraryTargetError(
                    f"lib.{problem.op}={mode}",
                    self.target.arch,
                    message or f"the {mode} library has no adapter for this target",
                    f"use lib.mode=auto; available alternatives: {alternatives}",
                )
            raise NoCandidate(message)
        names = ",".join(name_of(d) for d in problem.dtypes)
        shape = f"{problem.bucket}|{problem.precision}|{problem.options.numerics_key}|{problem.options.threads}"
        key = f"{self.fingerprint}|{self.target.arch}|{problem.op}|{names}|{shape}|{problem.layout}|{problem.epilogue}"
        return Choice(self.candidates, order, tuple(eligible), tuple(pool), key)

    def _measure[O: OperationOperands](
        self, key: str, problem: Problem, contest: Sequence[Candidate[O]], operands: O
    ) -> None:
        """Time the contest, the static choice first, on a private stream outside any capture; persist the winner."""
        times = asyncio.run(self._time_all(problem, contest, operands))
        if not (contest := [c for c in contest if c.name in times]):
            return  # every candidate declined; the call itself raises when it runs
        best = min(contest, key=lambda c: times[c.name])
        if times[best.name] > 0.9 * times[contest[0].name]:
            best = contest[0]  # within 10 percent is noise: keep the static preference of the target
        self.tuning.put(key, best.name)
        log.info("lib: %s %s -> %s %s", problem.op, problem.bucket, best.name, times)

    async def _time_all[O: OperationOperands](
        self, problem: Problem, contest: Sequence[Candidate[O]], operands: O
    ) -> dict[str, float]:
        device = default_device()
        stream = device.stream()
        with stream:  # the private output is allocated in this stream's order, which a device with ordered memory needs
            operands = operands.private(device)
        times: dict[str, float] = {}
        # The candidate timed first reads slow on tiny products (measured: 42 us for a 12 us kernel), so a second
        # sweep in the opposite order gives every candidate a turn that is not the first.
        for sweep in (list(contest), list(reversed(contest))):
            for candidate in sweep:
                if (milliseconds := await self._window(candidate, problem, operands, stream)) is not None:
                    times[candidate.name] = min(milliseconds, times.get(candidate.name, milliseconds))
        return times

    async def _window[O: OperationOperands](
        self, candidate: Candidate[O], problem: Problem, operands: O, stream: Stream
    ) -> float | None:
        """Milliseconds per call of the best timed window of a candidate, or None if it declines the problem."""
        try:
            candidate.run(problem, operands, stream)  # warm-up: plans and workspaces stay out of the timing
        except Declined as reason:
            log.info("lib: %s declined %s while timing: %s", candidate.name, problem.op, reason)
            return None
        begin = stream.record()
        candidate.run(problem, operands, stream)
        probe = stream.record()
        await probe
        repeats = min(max(int(TUNE_MILLISECONDS / max(probe.elapsed_since(begin), 1e-3)), 10), 2000)
        windows = []
        for _ in range(TUNE_WINDOWS):
            begin = stream.record()
            for _ in range(repeats):
                candidate.run(problem, operands, stream)
            end = stream.record()
            await end
            windows.append(end.elapsed_since(begin) / repeats)
        return min(windows[1:])  # the first window lets the clocks ramp up; the host adds noise to the others


def fingerprint(candidates: Mapping[Op, Sequence[Candidate]]) -> str:
    return ";".join(sorted({f"{c.name}={c.library_version()}" for cs in candidates.values() for c in cs}))


_snapshots: dict[str, Settings] = {}


def dispatcher() -> Dispatcher:
    snapshot = current_settings()
    _snapshots[snapshot.key] = snapshot
    return _dispatcher(snapshot.key)


@cache
def _dispatcher(key: str) -> Dispatcher:
    """Built on the first call from the facts of the machine; adapters load only for libraries the loader finds, and
    the NVIDIA ones on every NVIDIA machine, whose wheels the loader cannot see: each says in `unavailable` if its is
    missing."""
    snapshot = _snapshots[key]
    machine = System.detect()
    adapters = {
        "hipblaslt": "lib_hipblaslt", "mkl_rt": "lib_mkl", "dnnl": "lib_onednn",
        "Accelerate": "lib_accelerate", "MetalPerformanceShaders": "lib_mps",
    }  # fmt: skip
    modules = ["lib_ours", "lib_host"] + [m for lib, m in adapters.items() if lib in machine.libraries]
    if machine.target.vendor == "nvidia":
        modules += ["lib_cublaslt", "lib_cusparse"]
    found: dict[Op, list[Candidate]] = {op: [] for op in Op}
    for name in modules:
        for candidate in import_module(f".{name}", __package__).CANDIDATES:  # each adapter lists its instances
            found[candidate.op].append(candidate)
    mode = Mode(snapshot.lib.mode)
    per_op = {op: Mode(value) for op in Op if (value := getattr(snapshot.lib, op.value)) is not None}
    root = snapshot.cache.dir or Path.home() / ".cache" / "krn"
    tuning = TuningDb(root / "lib_tuning.json")
    tune = snapshot.lib.tune
    return Dispatcher(machine.target, found, tuning, mode, per_op, fingerprint(found), tune)


def _warn_reduced_accumulation(precision: Compute, arch: str, caller: Span) -> None:
    """Once per call site, operation (with its precision) and target; the caller's span is the call site."""
    warn_once(
        f"matmul precision={precision}",
        arch,
        "reduced-accumulation",
        "explicit library precision; TF32 truncates input mantissas; emulated schemes change intermediate arithmetic; "
        "no uniform ulp bound; use precision=Compute.STRICT for full f32 inputs and accumulation",
        PrecisionWarning,
        caller,
    )


@overload
def matmul[S: FloatingPoint](
    a: Matrix[S],
    b: Matrix[S],
    *,
    precision: Compute = Compute.STRICT,
    epilogue: Epilogue = Epilogue.NONE,
    bias: Vector[S] | None = None,
    out: None = None,
    options: LibOptions | None = None,
) -> Matrix[S]: ...
@overload
def matmul[A: FloatingPoint, B: FloatingPoint](
    a: Matrix[A],
    b: Matrix[B],
    *,
    precision: Compute = Compute.STRICT,
    epilogue: Epilogue = Epilogue.NONE,
    bias: Vector[FloatingPoint] | None = None,
    out: None = None,
    options: LibOptions | None = None,
) -> Matrix[FloatingPoint]: ...
@overload
def matmul[A: FloatingPoint, B: FloatingPoint, Y: FloatingPoint](
    a: Matrix[A],
    b: Matrix[B],
    *,
    precision: Compute = Compute.STRICT,
    epilogue: Epilogue = Epilogue.NONE,
    bias: Vector[Y] | None = None,
    out: Matrix[Y],
    options: LibOptions | None = None,
) -> Matrix[Y]: ...
@retained
def matmul[A: FloatingPoint, B: FloatingPoint, Y: FloatingPoint](
    a: Matrix[A],
    b: Matrix[B],
    *,
    precision: Compute = Compute.STRICT,
    epilogue: Epilogue = Epilogue.NONE,
    bias: Vector[Y] | None = None,
    out: Matrix[Y] | None = None,
    options: LibOptions | None = None,
) -> Matrix[Y] | Matrix[FloatingPoint]:
    """a @ b with an optional fused epilogue, enqueued on the current stream. Allocates `out` when it is not given,
    in the type that holds both inputs (the type itself for two equal ones).

    precision=Compute.STRICT uses full input precision and accumulation. TF32 and
    EMULATED are explicit operation choices, never library defaults; an unsupported
    choice raises TargetError instead of falling through to a different computation.
    """
    device = default_device()
    stream = current_stream.get() or device.default_stream
    if precision is not Compute.STRICT:
        caller = sys._getframe(2)  # frame 1 is the retained wrapper; the user's call is frame 2
        _warn_reduced_accumulation(precision, device.target.arch, Span(caller.f_code.co_filename, caller.f_lineno, 1))
    result = out if out is not None else device.empty(product_type(a.dtype, b.dtype), a.shape[0], b.shape[1])
    shape = (a.shape[0], b.shape[1], a.shape[1])
    problem = Problem(
        Op.MATMUL,
        (a.dtype, b.dtype, result.dtype),
        shape,
        options or LibOptions.of(device.target),
        epilogue,
        precision=precision,
    )
    operands: Operands = Operands(a, b, result, bias)
    dispatcher().launch(MatmulCandidate, problem, operands, stream)
    return result


@overload
def spmv[A: FloatingPoint, X: FloatingPoint](
    matrix: Rows[A] | Coo[A], x: Vector[X], *, out: None = None, options: LibOptions | None = None
) -> Vector[FloatingPoint]: ...
@overload
def spmv[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](
    matrix: Rows[A] | Coo[A], x: Vector[X], *, out: Vector[Y], options: LibOptions | None = None
) -> Vector[Y]: ...
@retained
def spmv[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](
    matrix: Rows[A] | Coo[A],
    x: Vector[X],
    *,
    out: Vector[Y] | None = None,
    options: LibOptions | None = None,
) -> Vector[Y] | Vector[FloatingPoint]:
    """matrix @ x for a sparse matrix of any format, enqueued on the current stream. Allocates `out` when not given."""
    device = default_device()
    stream = current_stream.get() or device.default_stream
    result = out if out is not None else device.empty(product_type(matrix.values.dtype, x.dtype), int(matrix.rows))
    shape = (int(matrix.rows), int(matrix.columns), matrix.values.shape[0])
    problem = Problem(
        Op.SPMV,
        (matrix.values.dtype, x.dtype, result.dtype),
        shape,
        options or LibOptions.of(device.target),
        layout=type(matrix).__name__,
    )
    dispatcher().launch(SpmvCandidate, problem, SpmvOperands(matrix, x, result), stream)
    return result


@overload
def spmm[A: FloatingPoint, X: FloatingPoint](
    matrix: Rows[A] | Coo[A], x: Matrix[X], *, out: None = None, options: LibOptions | None = None
) -> Matrix[FloatingPoint]: ...
@overload
def spmm[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](
    matrix: Rows[A] | Coo[A], x: Matrix[X], *, out: Matrix[Y], options: LibOptions | None = None
) -> Matrix[Y]: ...
@retained
def spmm[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](
    matrix: Rows[A] | Coo[A],
    x: Matrix[X],
    *,
    out: Matrix[Y] | None = None,
    options: LibOptions | None = None,
) -> Matrix[Y] | Matrix[FloatingPoint]:
    """matrix @ x for a sparse matrix of any format and a dense matrix, enqueued on the current stream."""
    device = default_device()
    stream = current_stream.get() or device.default_stream
    result = (
        out
        if out is not None
        else device.empty(product_type(matrix.values.dtype, x.dtype), int(matrix.rows), x.shape[1])
    )
    shape = (int(matrix.rows), int(matrix.columns), matrix.values.shape[0], x.shape[1])
    problem = Problem(
        Op.SPMM,
        (matrix.values.dtype, x.dtype, result.dtype),
        shape,
        options or LibOptions.of(device.target),
        layout=type(matrix).__name__,
    )
    dispatcher().launch(SpmmCandidate, problem, SpmmOperands(matrix, x, result), stream)
    return result
