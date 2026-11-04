"""Lowering of the IR to C-family source. One `Emitter` walks the IR for every target and asks a `Renderer` (one
class per target, in the style of tinygrad) for each spelling: types, intrinsics, qualifiers and launch geometry.

Tiles use buffers unless a matrix unit gives them a register home. Elementwise work follows that home's ownership;
arbitrary accesses first spill a mirror. The renderer spells the matrix instructions and asynchronous copies.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import ClassVar

from krn import catalog
from krn.arithmetic import align_up, ceildiv
from krn.facts.targets import Target
from krn.frontend.atomic import AtomicOp
from krn.frontend.geometry import Blocks, Geometry, Items, Tiles
from krn.runtime.settings import current_settings

from . import narrow
from .dtypes import BOOL, F32, I64, PY_FLOAT, PY_INT, U32, U64, Dtype, Kind, uint
from .errors import CaptureError, TargetError
from .heuristics import DEFAULTS, Decisions, Inlining, decide
from .heuristics import threads as default_threads
from .ir import (
    Accumulate,
    Arange,
    Assert,
    Assign,
    Atomic,
    Binary,
    Breakpoint,
    Cast,
    Const,
    Dim,
    Effect,
    Expr,
    Fn,
    For,
    If,
    Intrinsic,
    Kernel,
    Load,
    Math,
    Op,
    Param,
    Print,
    Reduce,
    Reduction,
    Select,
    Slot,
    Stmt,
    Store,
    StructType,
    Sync,
    TensorRef,
    TensorType,
    TileAssign,
    TileLoad,
    TileStore,
    Transpose,
    Unary,
    Var,
    While,
    Zeros,
    slots,
    tile_axes,
)
from .matrix import (
    FP8,
    FP8_AVAILABILITY,
    MatrixPlan,
    MatrixUnit,
    fallback,
    operand,
    pipeline_stages,
)
from .narrow import check as check_narrow
from .numerics import check as check_numerics
from .performance import Facts
from .performance import check as check_performance
from .validity import can_be_invalid, children, masked_tiles, walk

log = logging.getLogger(__name__)

type Interval = tuple[str, str]  # the C expressions of the lower and the upper bound of a range of lanes
type Box = list[Interval]  # one interval per axis of a tile
BOX_HELPERS = {  # the largest lower bound and the smallest upper bound, as functions so no operand is written twice
    "_lmax": "long long _lmax(long long a, long long b) { return a > b ? a : b; }",
    "_lmin": "long long _lmin(long long a, long long b) { return a < b ? a : b; }",
}
COMPARES = {Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE, Op.LAND, Op.LOR}
C_BYTES = {"_Bool": 1, "bool": 1, "unsigned char": 1, "signed char": 1, "unsigned short": 2, "short": 2, "_Float16": 2,
           "unsigned": 4, "int": 4, "float": 4, "unsigned long long": 8, "long long": 8, "double": 8}  # fmt: skip
INT_SUFFIX = {U32: "u", U64: "ull", I64: "ll", PY_INT: "ll"}
LIBM = {
    Fn.EXP: "exp", Fn.EXP2: "exp2", Fn.EXPM1: "expm1", Fn.LOG: "log", Fn.LOG2: "log2", Fn.LOG1P: "log1p",
    Fn.SQRT: "sqrt", Fn.SIN: "sin", Fn.COS: "cos", Fn.TANH: "tanh", Fn.ERF: "erf", Fn.FLOOR: "floor",
    Fn.CEIL: "ceil", Fn.ROUND: "rint", Fn.ABS: "fabs", Fn.POW: "pow", Fn.LOG10: "log10", Fn.EXP10: "exp10",
    Fn.TRUNC: "trunc", Fn.CBRT: "cbrt", Fn.FMA: "fma",
}  # fmt: skip
INLINE = {  # operations that name each operand once, over `{0}`, with the libm names of the type as `{exp}`
    Fn.RECIPROCAL: "(1 / {0})",
    Fn.RSQRT: "(1 / {sqrt}({0}))",
    Fn.SIGMOID: "(1 / (1 + {exp}(-{0})))",
}
HELPERS = {  # functions of the type for the operations that name an operand twice, `{f}` is the literal suffix
    Fn.RELU: (("x",), "x < 0 ? 0 : x"),
    Fn.GELU: (("x",), "0.5{f} * x * (1 + {erf}(x * 0.7071067811865476{f}))"),
    Fn.SILU: (("x",), "x / (1 + {exp}(-x))"),
    Fn.SOFTPLUS: (("x",), "(x > 0 ? x : 0) + {log1p}({exp}(-{fabs}(x)))"),
    Fn.CLIP: (("x", "lo", "hi"), "x < lo ? lo : (x > hi ? hi : x)"),
    Fn.MAXIMUM: (
        ("a", "b"),
        "a > b || a != a ? a : b",
    ),  # a NaN in either operand is the result, as in numpy and torch
    Fn.MINIMUM: (("a", "b"), "a < b || a != a ? a : b"),
}
ATOMIC_UPDATE = {  # the new value of an element `o` given the operand `v`, for a target that updates it in one step
    AtomicOp.ADD: "o + v", AtomicOp.SUB: "o - v", AtomicOp.AND: "o & v", AtomicOp.OR: "o | v", AtomicOp.XOR: "o ^ v",
    AtomicOp.EXCH: "v", AtomicOp.INC: "o >= v ? 0 : o + 1", AtomicOp.DEC: "o == 0 || o > v ? v : o - 1",
    AtomicOp.MAX: "o > v ? o : v", AtomicOp.MIN: "o < v ? o : v",
}  # fmt: skip
PRINTF = {  # the conversion and the C type a printed value is cast to
    "sint": ("%d", "int"), "uint": ("%u", "unsigned"), "float": ("%g", "double"), "bool": ("%d", "int"),
}  # fmt: skip
REDUCERS = {
    Reduction.MAX: "_acc = _acc > _x || _acc != _acc ? _acc : _x;",
    Reduction.MIN: "_acc = _acc < _x || _acc != _acc ? _acc : _x;",
    Reduction.SUM: "_acc = _acc + _x;",
    Reduction.PROD: "_acc = _acc * _x;",
    Reduction.MEAN: "_acc = _acc + _x;",
}


@dataclass(frozen=True, slots=True)
class Unit:
    """One translation unit: the source, its entry point and the dynamic shared memory a launch must reserve."""

    source: str
    entry: str
    shared_bytes: int = 0
    threads: int = 0
    helpers: tuple[tuple[str, int, str], ...] = ()


THREADS = DEFAULTS.element_threads


def strided(loop: While) -> bool:
    """The grid-stride advance and condition emitted by Builder.items, allowing its integer cast."""
    match loop:
        case While(
            cond=Binary(op=Op.LT, lhs=Var() as index),
            body=(*_, Assign(var=var, value=Binary(op=Op.ADD, lhs=lhs, rhs=stride))),
        ):
            if var != index or lhs != index:
                return False
            while isinstance(stride, Cast):
                stride = stride.operand
            return isinstance(stride, Intrinsic) and stride.name == "grid.num_threads"
    return False


def dimensions(kernel: Kernel, launch: Geometry, decisions: Decisions | None = None) -> tuple[int, int, int, int]:
    """The grid and the threads per block of `launch`. A tiled kernel runs one thread per block on a CPU. Items per
    thread shrink the grid only for a kernel with an `items` loop, which covers any grid; a kernel that indexes by
    thread rank needs one thread per item."""
    if decisions is None:
        settings = current_settings()
        threads = 1 if kernel.tiled else default_threads(kernel, settings)
        per_thread = DEFAULTS.unroll
    else:
        threads, per_thread = decisions.block.value, decisions.unroll.value
    if not any(axis.strided for axis in kernel.space):
        per_thread = 1
    match launch:
        case Items(count=count):
            return ceildiv(count, threads * per_thread), 1, 1, threads
        case Blocks(x=x, y=y, z=z, threads=explicit):
            return x, y, z, explicit or threads
        case Tiles():
            return dimensions(kernel, launch.blocks, decisions)


class Renderer[T: Target](ABC):
    """The spellings of one target. Subclasses fill the class variables and override what differs."""

    types: ClassVar[dict[Dtype, str]]
    functions: ClassVar[dict[Fn, str]]  # the libm name of each function the target has natively
    libm: ClassVar[str]  # what precedes a libm name: `__builtin_` for clang, nothing for NVRTC
    inline: ClassVar[str] = "@inline "
    qualifiers: ClassVar[Mapping[Inlining, str]]
    unroll_pragma: ClassVar[str] = "#pragma unroll {count}"
    tid: ClassVar[str]
    bdim: ClassVar[str]
    sync: ClassVar[str]
    leaf_join: ClassVar[str]
    restrict: ClassVar[str]
    struct_param: ClassVar[str]
    depth: ClassVar[int]
    printf_decl: ClassVar[str] = ""  # the declaration of printf where the compiler does not know it without a header
    debuggable: ClassVar[bool] = True  # targets without these tools raise TargetError
    renamed: ClassVar[dict[str, str]] = {}  # libm names the target spells differently, because its library lacks them
    libm_suffix: ClassVar[bool] = True  # libm names take an `f` for float32 (`expf`); Metal overloads `exp` instead
    matrix_unit: ClassVar[MatrixUnit | None] = None
    # The spellings of a float's bits and of a float from its bits, with `{0}` for the operand.
    bitcast: ClassVar[tuple[str, str]] = ("__builtin_bit_cast(unsigned, {0})", "__builtin_bit_cast(float, {0})")

    def __init__(self, target: T, decisions: Decisions | None = None) -> None:
        self.target = target
        self.decisions = decisions
        self.noncoherent: set[str] = set()
        self.helpers: dict[str, str] = {}
        self.helper_rows: list[tuple[str, int, str]] = []
        self.matrix_buffers: dict[str, Dtype] = {}
        self.threads = 0

    def helper_text(self) -> list[str]:
        """Resolve each marked helper independently, including bundles of conversion helpers."""
        assert self.decisions is not None
        rendered = []
        self.helper_rows.clear()
        for text in self.helpers.values():
            # Older catalog entries spell CUDA functions directly, alongside device globals such as _grid_arrived.
            prefix, *device = text.split("__device__ ")
            text = prefix + "".join(
                (self.inline if "(" in part.partition(";")[0] else "__device__ ") + part for part in device
            )
            prefix, *helpers = text.split(self.inline)
            qualified = []
            for helper in helpers:
                size = helper.count(";")
                name = helper.partition("(")[0].split()[-1]
                qualifier = self.qualifiers[self.decisions.helper(size)]
                self.helper_rows.append((name, size, qualifier.strip() or "unqualified"))
                qualified.append(qualifier + helper)
            rendered.append(prefix + "".join(qualified))
        return rendered

    def ctype(self, dtype: Dtype) -> str:
        """Storage spelling: the element type of a tensor. A narrow float is its code, one byte."""
        if narrow.is_narrow(dtype):
            return "unsigned char"
        if dtype not in self.types:
            raise NotImplementedError(f"{dtype} is not lowered for target {self.target.name} in this slice")
        return self.types[dtype]

    def vtype(self, dtype: Dtype) -> str:
        """Spelling of a computed value; differs from storage where the target has no native type (f16 on CUDA)."""
        return self.ctype(dtype)

    def leaf(self, path: tuple[str, ...]) -> str:
        return self.leaf_join.join(path)

    def literal(self, dtype: Dtype, value: float) -> str:
        if dtype.is_float or dtype == PY_FLOAT:
            if not math.isfinite(value):
                raise NotImplementedError("infinite and NaN literals are not lowered in this slice")
            return f"{float(value)!r}" + ("f" if self.vtype(dtype) == "float" else "")
        return f"{int(value)}{INT_SUFFIX.get(dtype, '')}"

    def zero(self, dtype: Dtype) -> str:
        return f"({self.vtype(dtype)})0"

    def math(self, fn: Fn, dtype: Dtype, args: list[str]) -> str:
        """A call of an elementwise function: libm where the target has it, else the formula (as a helper function
        when it names an operand twice, so a nested operand is not repeated in the text)."""
        ctype = self.vtype(dtype)
        suffix = "f" if ctype == "float" else ""
        named = suffix if self.libm_suffix else ""
        spell = {name: f"{self.libm}{self.renamed.get(name, name)}{named}" for name in self.functions.values()}
        spell |= {"f": suffix}
        if fn in self.functions:
            return f"{spell[self.functions[fn]]}({', '.join(args)})"
        if fn in INLINE:
            return INLINE[fn].format(*args, **spell)
        params, body = HELPERS[fn]
        name = f"_p_{fn}_{ctype}"
        signature = ", ".join(f"{ctype} {p}" for p in params)
        self.helpers.setdefault(name, f"{self.inline}{ctype} {name}({signature}) {{ return {body.format(**spell)}; }}")
        return f"{name}({', '.join(args)})"

    def load(self, dtype: Dtype, pointer: str, index: str) -> str:
        return f"{pointer}[{index}]"

    def store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        return f"{pointer}[{index}] = {value};"

    def matrix_bits(self, dtype: Dtype, value: str) -> str:
        return value

    def matrix_store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        return self.store(dtype, pointer, index, value)

    def cast(self, src: Dtype, dst: Dtype, text: str) -> str:
        """A conversion. A narrow float decodes to float32 first and encodes last; the rest is `value_cast`."""
        if narrow.is_narrow(src):
            text, src = self.decode(src, text), F32
        if narrow.is_narrow(dst):
            return f"({self.ctype(dst)}){self.encode(dst, text)}"
        return self.value_cast(src, dst, text)

    def value_cast(self, src: Dtype, dst: Dtype, text: str) -> str:
        return f"({self.vtype(dst)})({text})"

    def decode(self, dtype: Dtype, text: str) -> str:
        """The float32 of a narrow code: composed from integer operations, once per unit."""
        fmt = narrow.FORMATS[dtype]
        self.helpers.setdefault(f"narrow {narrow.NAMES[fmt]}", narrow.bundle(fmt, *self.bitcast))
        return f"{narrow.decode_name(fmt)}({text})"

    def encode(self, dtype: Dtype, text: str) -> str:
        """The narrow code of a float32, rounded to nearest even and saturated."""
        fmt = narrow.FORMATS[dtype]
        self.helpers.setdefault(f"narrow {narrow.NAMES[fmt]}", narrow.bundle(fmt, *self.bitcast))
        return f"{narrow.encode_name(fmt)}({text})"

    def field(self, slot: Slot, written: bool, qualifier: str = "") -> str:
        """A parameter or struct field: pointers are `const` unless the kernel writes through them."""
        if slot.role != "ptr":
            return f"{self.ctype(slot.dtype)} {slot.name}"
        assert self.decisions is not None
        const = "const " if self.decisions.const.value and not written else ""
        qualifier = qualifier if self.decisions.restrict.value else ""
        return f"{const}{self.ctype(slot.dtype)}* {qualifier}{slot.name}"

    def parameters(self, kernel: Kernel) -> list[str]:
        """The C parameters: scalars, `pointer, n0, n1` for a tensor, a whole struct for a struct."""
        assert self.decisions is not None
        struct_param = self.struct_param if self.decisions.const.value else self.struct_param.removeprefix("const ")
        return [
            struct_param.format(type=type.name, name=name)
            if isinstance(type, StructType)
            else self.field(slot, (name,) in kernel.written, self.restrict)
            for name, type in kernel.params
            if not (isinstance(type, StructType) and not type.fields)
            for slot in ([Slot(name, I64, "scalar")] if isinstance(type, StructType) else slots(name, type))
        ]

    def struct_decl(self, kernel: Kernel, param: str, struct: StructType) -> str:
        fields = " ".join(
            f"{self.field(slot, (param, slot.name) in kernel.written)};"
            for name, type in struct.fields
            for slot in slots(name, type)
        )
        return f"typedef struct {{ {fields} }} {struct.name};"

    def struct_decls(self, kernel: Kernel) -> list[str]:
        return [
            self.struct_decl(kernel, name, type)
            for name, type in kernel.params
            if isinstance(type, StructType) and type.fields
        ]

    def atomic(
        self,
        op: AtomicOp,
        dtype: Dtype,
        pointer: str,
        value: str,
        compare: str | None,
        *,
        shared: bool = False,
        scope: str = "device",
    ) -> str:
        """A call of a helper that applies `op` to `*pointer` and returns the old value; `shared` says the element is in
        block-shared memory, which Metal types differently. The default updates in one step, which is atomic where the
        threads of a block run one after the other, as on the C backend."""
        ctype, name = self.vtype(dtype), f"_atomic_{op}_{dtype}"
        if op is AtomicOp.CAS:
            body = f"{ctype} o = *p; if (o == c) *p = v; return o;"
            self.helpers.setdefault(name, f"{self.inline}{ctype} {name}({ctype}* p, {ctype} c, {ctype} v) {{ {body} }}")
            return f"{name}({pointer}, {compare}, {value})"
        body = f"{ctype} o = *p; *p = ({ctype})({ATOMIC_UPDATE[op]}); return o;"
        self.helpers.setdefault(name, f"{self.inline}{ctype} {name}({ctype}* p, {ctype} v) {{ {body} }}")
        return f"{name}({pointer}, {value})"

    @abstractmethod
    def unit(self, kernel: Kernel, body: list[str], buffers: dict[str, tuple[Dtype, int]]) -> Unit: ...


class Emitter[T: Target]:
    """Walks one kernel and produces the body lines; the renderer wraps them into a unit."""

    def __init__(self, kernel: Kernel, renderer: Renderer[T], *, lineinfo: bool = False) -> None:
        self.kernel, self.r, self.lineinfo = kernel, renderer, lineinfo
        self.decisions = renderer.decisions or decide(kernel, renderer.target, current_settings())
        check_performance(
            kernel,
            renderer.target.arch,
            self.decisions.settings.warn,
            Facts(unrolled_instructions=sum(1 for _ in walk(kernel.body))),
        )
        check_numerics(kernel, renderer.target)
        check_narrow(kernel, renderer.target)
        renderer.decisions = self.decisions
        if self.decisions.const.value and self.decisions.restrict.value:
            for name, kind in kernel.params:
                fields = (
                    [((name, field), value) for field, value in kind.fields]
                    if isinstance(kind, StructType)
                    else [((name,), kind)]
                )
                renderer.noncoherent.update(
                    renderer.leaf(path)
                    for path, value in fields
                    if isinstance(value, TensorType) and value.dtype != BOOL and path not in kernel.written
                )
        self.lines: list[str] = []
        self.depth = renderer.depth
        self.buffers = {v.name: (v.dtype, math.prod(v.shape)) for v in kernel.tiles}
        self.ready: dict[Expr, str] = {}
        self.masked = masked_tiles(kernel)
        self.plans: dict[Accumulate, MatrixPlan] = {}
        self.homes: dict[str, MatrixPlan] = {}
        self.owned: tuple[MatrixPlan, tuple[str, ...]] | None = None
        self.spilled: set[str] = set()
        self.mirrors: set[str] = set()
        self.substitutions: dict[str, str] = {}
        for s in walk(kernel.body):
            if isinstance(s, Accumulate) and renderer.matrix_unit is None and operand(s.lhs).dtype in FP8:
                fallback(
                    s, renderer.target, f"{renderer.target.arch} has no implemented matrix unit; {FP8_AVAILABILITY}"
                )
        if unit := renderer.matrix_unit:
            reasons: set[str] = set()
            for s in walk(kernel.body):
                if isinstance(s, Accumulate):
                    plan = unit.plan(s, renderer.target)
                    if isinstance(plan, str):
                        reasons.add(plan)
                    else:
                        self.plans[s] = plan
                        self.homes.setdefault(s.acc.name, plan)
            for reason in sorted(reasons):
                log.info("%s: %s", kernel.name, reason)
            # Every product of one accumulator must agree on fragment ownership.
            for s in walk(kernel.body):
                if isinstance(s, Accumulate) and s.acc.name in self.homes:
                    home = self.homes[s.acc.name]
                    plan = self.plans.get(s)
                    if plan is None or (plan.wm, plan.wn) != (home.wm, home.wn):
                        self.homes.pop(s.acc.name)
            self.plans = {s: p for s, p in self.plans.items() if s.acc.name in self.homes}
            for s in walk(kernel.body):
                if isinstance(s, Accumulate) and operand(s.lhs).dtype in FP8 and s not in self.plans:
                    raise TargetError(
                        "fp8 matrix product",
                        renderer.target.arch,
                        "fp8 products require consistent matrix fragment ownership",
                        "use consistent fragment ownership or use f16 or bf16 operands",
                        span=s.span,
                    )
            if not unit.owned_fragments:
                stages = sum(
                    2 * (p.bm * (p.bk + 8) + p.bk * (p.bn + 8))
                    for s in walk(kernel.body)
                    if isinstance(s, Accumulate) and (p := self.plans.get(s)) is not None
                )
                if self.homes and self._shared_bytes() + stages > renderer.target.shared_bytes:
                    log.info(
                        "%s: %s operand stages, mirrors and other shared buffers need %d bytes, exceeding %d; "
                        "run on the FMA loop",
                        kernel.name,
                        renderer.target.arch,
                        self._shared_bytes() + stages,
                        renderer.target.shared_bytes,
                    )
                    self.homes.clear()
                    self.plans.clear()
                self.mirrors.update(self.homes)
            renderer.threads = max((p.threads for p in self.homes.values()), default=0)

    def invalid(self, e: Expr) -> bool:
        return can_be_invalid(e, self.masked)

    def emit(self) -> Unit:
        if unit := self.r.matrix_unit:
            for name, plan in self.homes.items():
                self.line(unit.declare(name, plan))
        self.statements(self.kernel.body)
        for name in self.homes.keys() - self.mirrors:
            del self.buffers[name]
        names = [
            f"_b_{v.name}_{end}{axis}"
            for v in self.kernel.tiles
            if v.name in self.masked
            for axis in range(len(v.shape))
            for end in ("lo", "hi")
        ]
        if names:  # the valid interval of each masked tile variable along each axis, declared where the body begins
            self.lines.insert(0, f"{'    ' * self.r.depth}long long {', '.join(names)};")
        unit = self.r.unit(self.kernel, self.lines, self.buffers)
        return replace(unit, helpers=tuple(self.r.helper_rows))

    def line(self, text: str) -> None:
        if text:
            self.lines.append("    " * self.depth + text)

    @contextmanager
    def scope(self, header: str) -> Generator[None]:
        """`header {` ... `}` around the lines emitted inside a `with`."""
        self.line(f"{header} {{")
        self.depth += 1
        yield
        self.depth -= 1
        self.line("}")

    def statements(self, body: tuple[Stmt, ...]) -> None:
        for statement in body:
            self.ready.clear()
            self.spilled.clear()  # a mirror made in another branch need not exist on this path
            if self.lineinfo:  # a directive in column 0: the lines that follow belong to this line of the Python source
                file = statement.span.file.replace("\\", "\\\\").replace('"', '\\"')
                self.lines.append(f'#line {statement.span.line} "{file}"')
            getattr(self, f"emit_{type(statement).__name__}")(statement)

    def emit_Assign(self, s: Assign) -> None:
        self.line(f"{s.var.name} = {self.value(s.value)};")

    def emit_Store(self, s: Store) -> None:
        index = self.address(s.tensor, [self.index(i) for i in s.index])
        self.line(self.r.store(s.tensor.type.dtype, self.r.leaf(s.tensor.path), index, self.value(s.value)))

    def emit_If(self, s: If) -> None:
        with self.scope(f"if ({self.value(s.cond)})"):
            self.statements(s.then)
        if s.orelse:
            with self.scope("else"):
                self.statements(s.orelse)

    def emit_While(self, s: While) -> None:
        if strided(s) and (self.decisions.unroll.value != 1 or self.decisions.unroll.explicit):
            self.line(self.r.unroll_pragma.format(count=self.decisions.unroll.value))
        with self.scope(f"while ({self.value(s.cond)})"):
            self.statements(s.body)

    def emit_For(self, s: For) -> None:
        if s.pipelined:
            reason = self.pipeline(s)
            if reason is None:
                return
            if s.stages:
                raise TargetError(
                    "pipelined(stages)",
                    self.r.target.arch,
                    reason,
                    "use stages=AUTO for the target default, or a supported CUDA tile and stage count",
                    span=s.span,
                )
        name = s.var.name
        if isinstance(s.lo, Const) and isinstance(s.hi, Const) and self.decisions.unroll.explicit:
            self.line(self.r.unroll_pragma.format(count=self.decisions.unroll.value))
        with self.scope(
            f"for ({self.r.vtype(s.var.dtype)} {name} = {self.value(s.lo)}; "
            f"{name} < {self.value(s.hi)}; {name} += {s.step})"
        ):
            self.statements(s.body)

    def pipeline(self, loop: For) -> str | None:
        """Overlap immutable global loads with matrix work using a ring of shared stages."""
        unit = self.r.matrix_unit
        if unit is None or not unit.async_copy or not loop.body:
            return "this backend has no asynchronous operand-copy pipeline for this loop"
        products: list[tuple[Accumulate, MatrixPlan, TileLoad, TileLoad]] = []
        for s in loop.body:
            if not isinstance(s, Accumulate) or (plan := self.plans.get(s)) is None:
                return "the loop body is not a sequence of supported matrix products"
            a, b = operand(s.lhs), operand(s.rhs)
            if plan.fp8 or plan.tf32:
                return "raw 16-bit asynchronous stages cannot transpose fp8 B or widen mixed operands"
            if not isinstance(a, TileLoad) or not isinstance(b, TileLoad) or loop.step != plan.bk:
                return "asynchronous stages require direct tile loads and a loop step matching BK"
            if any(sl.squeeze for sl in (*a.slices, *b.slices)):
                return "asynchronous stages need every tensor axis to be a tile axis, not a scalar index"
            if (
                a.tensor.extent
                or b.tensor.extent
                or a.tensor.path in self.kernel.written
                or b.tensor.path in self.kernel.written
            ):
                return "asynchronous stages require immutable global operands, not shared or written tensors"
            if a.slices[1].start != loop.var or b.slices[0].start != loop.var:
                return "asynchronous stages require the loop index along the operands' reduction axes"
            if self.uses(a.slices[0].start, loop.var) or self.uses(b.slices[1].start, loop.var):
                return "asynchronous stages require output tile coordinates independent of the loop index"
            products.append((s, plan, a, b))
        stage_bytes = sum(2 * (p.bm * (p.bk + 8) + p.bk * (p.bn + 8)) for _, p, _, _ in products)
        reserved = sum(
            count * C_BYTES[self.r.vtype(dtype)]
            for name, (dtype, count) in self.buffers.items()
            if name not in self.homes
        )
        reserved += sum(ref.extent * C_BYTES[self.r.vtype(ref.type.dtype)] for ref in self.kernel.shared)
        stages = pipeline_stages(stage_bytes, loop.stages, self.r.target.shared_bytes, reserved)
        if not stages:
            return f"the requested stages must be 2 through 8 and fit {self.r.target.shared_bytes} shared-memory bytes"
        loads: list[tuple[TileLoad, str, int]] = []
        for _, _, a, b in products:
            for e in (a, b):
                rows, cols = e.shape
                loads.append((e, self.buffer(uint(16), (stages * rows, cols + 8)), rows * (cols + 8)))
        with self.scope(""):
            self.substitutions[loop.var.name] = self.value(loop.lo)
            for i, (e, _, _) in enumerate(loads):
                pointer = self.r.leaf(e.tensor.path)
                column = self.index(e.slices[1].start)
                self.line(
                    f"const bool _aligned{i} = ((unsigned long long){pointer} % 16 == 0) && "
                    f"({self.dim(e.tensor, 1)} % 8 == 0) && ({column} % 8 == 0);"
                )
            del self.substitutions[loop.var.name]
            with self.scope(f"for (int _pre = 0; _pre < {stages - 1}; ++_pre)"):
                self.line(f"long long _future = {self.value(loop.lo)} + _pre * {loop.step};")
                self.issue_stage(loop, loads, "_pre")
            name = loop.var.name
            with self.scope(
                f"for ({self.r.vtype(loop.var.dtype)} {name} = {self.value(loop.lo)}; "
                f"{name} < {self.value(loop.hi)}; {name} += {loop.step})"
            ):
                self.line(f"long long _iteration = ({name} - {self.value(loop.lo)}) / {loop.step};")
                self.line(unit.wait(stages - 2))
                self.line(self.r.sync)
                self.line(f"long long _future = {name} + {stages - 1} * {loop.step};")
                self.issue_stage(loop, loads, f"((_iteration + {stages - 1}) % {stages})")
                for i, (s, plan, _, _) in enumerate(products):
                    self.accumulate_box(s)
                    a, b = loads[2 * i], loads[2 * i + 1]
                    ap = f"({a[1]} + (_iteration % {stages}) * {a[2]})"
                    bp = f"({b[1]} + (_iteration % {stages}) * {b[2]})"
                    for line in unit.product(s.acc.name, plan, ap, bp):
                        self.line(line)
                    self.spilled.discard(s.acc.name)
            self.line(unit.wait(0))
            self.line(self.r.sync)
        return None

    @staticmethod
    def uses(e: Expr, var: Var) -> bool:
        return e == var or any(Emitter.uses(c, var) for c in children(e))

    def issue_stage(self, loop: For, loads: list[tuple[TileLoad, str, int]], stage: str) -> None:
        unit = self.r.matrix_unit
        assert unit is not None
        self.substitutions[loop.var.name] = "_future"
        for i, (e, name, stride) in enumerate(loads):
            rows, cols = e.shape
            with self.scope(
                f"for (long long _chunk = {self.r.tid}; _chunk < {rows * cols // 8}; _chunk += {self.r.bdim})"
            ):
                row, col = f"(_chunk / {cols // 8})", f"(_chunk % {cols // 8} * 8)"
                cells = [f"({self.index(sl.start)} + {v})" for sl, v in zip(e.slices, (row, col), strict=True)]
                dest = f"({stage}) * {stride} + {row} * {cols + 8} + {col}"
                full = self.inside(e.tensor, [cells[0], f"({cells[1]} + 7)"])
                live = f"(_future < {self.value(loop.hi)})"
                with self.scope(f"if (_aligned{i} && (({full}) || !{live}))"):
                    size = f"({live} ? 16 : 0)"
                    ptr = self.r.leaf(e.tensor.path)
                    src = f"({live} ? &{ptr}[{self.address(e.tensor, cells)}] : {ptr})"
                    self.line(unit.copy_async(f"&{name}[{dest}]", src, size))
                with self.scope("else"):
                    self.line("#pragma unroll")
                    with self.scope("for (int _word = 0; _word < 8; ++_word)"):
                        scalar = [cells[0], f"({cells[1]} + _word)"]
                        raw = f"{self.r.leaf(e.tensor.path)}[{self.address(e.tensor, scalar)}]"
                        self.line(f"{name}[{dest} + _word] = {live} && {self.inside(e.tensor, scalar)} ? {raw} : 0;")
        del self.substitutions[loop.var.name]
        self.line(unit.commit())

    def emit_Sync(self, s: Sync) -> None:
        self.line(self.r.sync)

    def emit_Effect(self, s: Effect) -> None:
        self.line(f"{self.value(s.value)};")

    def emit_Breakpoint(self, s: Breakpoint) -> None:
        self.check_debug("breakpoint", s)
        self.line(f"{self.spell('breakpoint', s)};")

    def emit_Print(self, s: Print) -> None:
        self.check_debug("print", s)
        values = [p for p in s.parts if not isinstance(p, str)]
        text = " ".join(self.escaped(p) if isinstance(p, str) else self.conversion(p)[0] for p in s.parts)
        self.printf(text, [self.conversion(v)[1] for v in values], values)

    def emit_Assert(self, s: Assert) -> None:
        self.check_debug("assert", s)
        block, thread = (self.spell(f"{scope}.index.x", s) for scope in ("block", "thread"))
        with self.scope(f"if (!({self.value(s.cond)}))"):
            message = f"{self.escaped(s.text)} (block %u, thread %u)"
            self.line(f'printf("{message}\\n", (unsigned){block}, (unsigned){thread});')
            self.line(f"{self.spell('trap', s)};")

    def check_debug(self, feature: str, s: Stmt) -> None:
        if not self.r.debuggable:
            raise TargetError(
                feature,
                self.r.target.arch,
                "Metal Shading Language has no device printf, assertion reporting or debugger trap lowering",
                "inspect an output buffer on the host or debug with the CPU backend or emulator",
                span=s.span,
            )

    def printf(self, text: str, casts: list[str], values: list[Expr]) -> None:
        if self.r.printf_decl:
            self.r.helpers.setdefault("printf", self.r.printf_decl)
        arguments = "".join(f", ({cast})({self.value(v)})" for cast, v in zip(casts, values, strict=True))
        self.line(f'printf("{text}\\n"{arguments});')

    @staticmethod
    def conversion(e: Expr) -> tuple[str, str]:
        """The printf conversion of a value and the C type it is cast to, as the varargs promotion would have it."""
        dtype = e.dtype
        if (dtype.is_int and dtype.bits == 64) or dtype == PY_INT:
            return ("%llu", "unsigned long long") if dtype.kind is Kind.UINT else ("%lld", "long long")
        return PRINTF.get(dtype.kind.value, ("%lld", "long long"))

    def spell(self, name: str, at: Stmt | Expr, *operands: str, bits: int = 32, sign: str = "u") -> str:
        """The target's text for intrinsic `name`; a vendor without one fails the capture at the user's line."""
        try:
            entry = catalog.BY_NAME[name]
            if helper := entry.helper_for(self.r.target.vendor):
                self.r.helpers.setdefault(helper, helper.replace("@inline ", self.r.inline))
            return catalog.spell(name, self.r.target, *operands, bits=bits, sign=sign)
        except TargetError as error:
            raise TargetError(error.feature, error.target, error.reason, error.instead, span=at.span) from None
        except LookupError as error:
            raise CaptureError(at.span, str(error)) from None

    @staticmethod
    def escaped(text: str) -> str:
        """`text` as the inside of a C string literal that printf prints unchanged."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("\n", "\\n")

    def emit_TileAssign(self, s: TileAssign) -> None:
        home = self.homes.get(s.var.name)
        unit = self.r.matrix_unit
        opaque = home is not None and unit is not None and not unit.owned_fragments
        if opaque:
            home = None
        owner = home or self.owner(s.value)
        self.prepare(s.value, owner)
        shape, name = s.var.shape, s.var.name
        if name in self.masked:
            self.set_box(s.var, self.box(s.value))

        def write(idx: tuple[str, ...]) -> None:
            dest = self.value(s.var, idx) if home else f"{name}[{self.linear(shape, idx)}]"
            self.line(f"{dest} = {self.value(s.value, idx)};")

        self.iterate(shape, write, owner)
        if opaque:
            assert unit is not None
            self.mirrors.add(name)
            for line in unit.fill(name, self.homes[name], name):
                self.line(line)
        self.spilled.discard(name)

    def emit_TileStore(self, s: TileStore) -> None:
        owner = self.owner(s.value)
        self.prepare(s.value, owner)

        def write(idx: tuple[str, ...]) -> None:
            cells = [f"({self.index(sl.start)} + {i})" for sl, i in tile_axes(s.slices, idx)]
            address = self.address(s.tensor, cells)
            store = self.r.store(s.tensor.type.dtype, self.r.leaf(s.tensor.path), address, self.value(s.value, idx))
            guard = " && ".join(filter(None, (self.inside(s.tensor, cells), self.valid(s.value, idx))))
            self.line(f"if ({guard}) {store}")

        self.iterate(s.value.shape, write, owner)

    def fused(self, s: Accumulate, a: str, b: str, acc: str) -> str:
        """`fma(a, b, acc)` in the renderer's `krn.math.fma` spelling: one rounding per product, as the instructions."""
        if Fn.FMA not in self.r.functions:
            raise TargetError(
                "matrix product fallback",
                self.r.target.arch,
                f"{self.r.target.arch} has no fused multiply-add for f32, so each product cannot round once",
                "use a target with a fused f32 multiply-add, or operands with a matrix unit on this target",
                span=s.span,
            )
        return self.r.math(Fn.FMA, F32, [a, b, acc])

    def emit_Accumulate(self, s: Accumulate) -> None:
        k, n, acc, shape = s.lhs.shape[1], s.rhs.shape[1], s.acc.name, s.acc.shape

        def fma(idx: tuple[str, ...]) -> None:
            self.line(f"float _s = {acc}[{self.linear(shape, idx)}];")
            with self.scope(f"for (long long _k = 0; _k < {k}; ++_k)"):
                product = self.fused(s, f"{a}[{idx[0]} * {k} + _k]", f"{b}[_k * {n} + {idx[1]}]", "_s")
                self.line(f"_s = {product};")
            self.line(f"{acc}[{self.linear(shape, idx)}] = _s;")

        self.accumulate_box(s)
        if (plan := self.plans.get(s)) is not None:
            assert self.r.matrix_unit is not None
            a, b = self.matrix_stage(s.lhs, plan), self.matrix_stage(s.rhs, plan, rhs=True)
            for line in self.r.matrix_unit.product(acc, plan, a, b):
                self.line(line)
            self.line(self.r.sync)
            self.spilled.discard(acc)
        else:
            reserved = self._shared_bytes()
            unit = self.r.matrix_unit
            if (
                unit is not None
                and not unit.owned_fragments
                and not self.uses(s.lhs, s.acc)
                and not self.uses(s.rhs, s.acc)
                and reserved
                <= self.r.target.shared_bytes
                < reserved + 4 * (math.prod(s.lhs.shape) + math.prod(s.rhs.shape))
            ):
                self._direct_product(s)
                return
            a, b = self.stage(s.lhs), self.stage(s.rhs)
            self.elements(shape, fma)

    def _shared_bytes(self) -> int:
        return sum(
            align_up(count * C_BYTES.get(self.r.vtype(dtype), dtype.bits // 8), 16)
            for dtype, count in self.buffers.values()
        ) + sum(
            ref.extent * C_BYTES.get(self.r.vtype(ref.type.dtype), ref.type.dtype.bits // 8)
            for ref in self.kernel.shared
        )

    def _direct_product(self, s: Accumulate) -> None:
        """The memory-limited FMA fallback reads operands directly instead of allocating larger f32 stages."""
        self.prepare(s.lhs, None)
        self.prepare(s.rhs, None)

        def fma(idx: tuple[str, ...]) -> None:
            dest = f"{s.acc.name}[{self.linear(s.acc.shape, idx)}]"
            self.line(f"float _s = {dest};")
            with self.scope(f"for (long long _k = 0; _k < {s.lhs.shape[1]}; ++_k)"):
                operands = []
                for e, cells in ((s.lhs, (idx[0], "_k")), (s.rhs, ("_k", idx[1]))):
                    value, ok = self.value(e, cells), self.valid(e, cells)
                    operands.append(value if ok is None else f"({ok} ? {value} : 0.0f)")
                product = self.fused(s, operands[0], operands[1], "_s")
                self.line(f"_s = {product};")
            self.line(f"{dest} = _s;")

        self.elements(s.acc.shape, fma)

    def accumulate_box(self, s: Accumulate) -> None:
        if s.acc.name in self.masked:  # a row or column without a valid operand lane has none in the product
            (rows, inner), (reduced, cols) = (
                self.box(s.lhs) or self.full(s.lhs),
                self.box(s.rhs) or self.full(s.rhs),
            )
            product = [
                (rows[0], f"({inner[0]} < {inner[1]} ? {rows[1]} : 0)"),
                (cols[0], f"({reduced[0]} < {reduced[1]} ? {cols[1]} : 0)"),
            ]
            self.set_box(s.acc, self.meet([self.box(s.acc) or self.full(s.acc), product]))

    def owner(self, e: Expr) -> MatrixPlan | None:
        """An elementwise expression may reuse one compatible fragment's ownership."""
        if self.r.matrix_unit is not None and not self.r.matrix_unit.owned_fragments:
            return None
        if isinstance(e, Reduce):
            return None
        if isinstance(e, Var):
            return self.homes.get(e.name)
        owners = [self.owner(c) for c in children(e)]
        return next((p for p in owners if p is not None and (p.bm, p.bn) == e.shape), None)

    def prepare(self, e: Expr, owner: MatrixPlan | None) -> None:
        if isinstance(e, Var) and (plan := self.homes.get(e.name)) is not None and plan != owner:
            self.spill(e)
        for child in children(e):
            self.prepare(child, None if isinstance(e, Reduce) else owner)
        self.materialize(e)

    def spill(self, e: Var) -> None:
        if e.name in self.spilled:
            return
        plan = self.homes[e.name]
        self.mirrors.add(e.name)
        unit = self.r.matrix_unit
        assert unit is not None
        if not unit.owned_fragments:
            for line in unit.spill(e.name, plan, e.name):
                self.line(line)
            self.spilled.add(e.name)
            return
        self.iterate(
            e.shape,
            lambda idx: self.line(f"{e.name}[{self.linear(e.shape, idx)}] = {self.value(e, idx)};"),
            plan,
        )
        self.spilled.add(e.name)

    def iterate(
        self,
        shape: tuple[int, ...],
        body: Callable[[tuple[str, ...]], None],
        plan: MatrixPlan | None,
    ) -> None:
        if plan is None:
            self.elements(shape, body)
            return
        unit = self.r.matrix_unit
        assert unit is not None
        with self.scope(f"if ({unit.active(plan)})"):
            self.line("#pragma unroll")
            with self.scope(f"for (int _mt = 0; _mt < {plan.mt}; ++_mt)"):
                self.line("#pragma unroll")
                with self.scope(f"for (int _nt = 0; _nt < {plan.nt}; ++_nt)"):
                    self.line("#pragma unroll")
                    with self.scope("for (int _fr = 0; _fr < 4; ++_fr)"):
                        idx = unit.coordinates(plan)
                        previous, self.owned = self.owned, (plan, idx)
                        body(idx)
                        self.owned = previous
        self.line(self.r.sync)

    def matrix_stage(self, e: Expr, plan: MatrixPlan, *, rhs: bool = False) -> str:
        """Stage raw instruction operands, or exactly widened tf32 words, zero-filling invalid lanes."""
        e = operand(e)
        owner = self.owner(e)
        self.prepare(e, owner)
        rows, cols = e.shape
        transpose = plan.fp8 and rhs
        pad = 16 if plan.fp8 else 4 if plan.tf32 and not rhs else (8 - cols) % 32 if plan.tf32 else 8
        storage = F32 if plan.tf32 else uint(8) if plan.fp8 else uint(16)
        stride = (rows if transpose else cols) + pad
        name = self.buffer(storage, (cols if transpose else rows, stride))
        self.r.matrix_buffers[name] = e.dtype

        def copy(idx: tuple[str, ...]) -> None:
            dest = f"{idx[1]} * {stride} + {idx[0]}" if transpose else f"{idx[0]} * {stride} + {idx[1]}"
            if isinstance(e, TileLoad) and not plan.tf32:
                cells = [f"({self.index(sl.start)} + {i})" for sl, i in tile_axes(e.slices, idx)]
                raw = f"{self.r.leaf(e.tensor.path)}[{self.address(e.tensor, cells)}]"
                self.line(f"{name}[{dest}] = {self.inside(e.tensor, cells)} ? {self.r.matrix_bits(e.dtype, raw)} : 0;")
            else:
                ok = self.valid(e, idx)
                value = self.value(e, idx)
                if ok is not None:
                    value = f"({ok} ? {value} : 0.0f)"
                self.line(
                    self.r.store(F32, name, dest, value)
                    if plan.tf32
                    else self.r.matrix_store(e.dtype, name, dest, value)
                )

        self.iterate(e.shape, copy, owner)
        return name

    def stage(self, e: Expr) -> str:
        """A buffer with the f32 elements of a matrix-product operand, filled once per block.

        An invalid lane stages as zero, which adds nothing to a product.
        """
        if isinstance(e, Var) and e.dtype == F32 and e.name not in self.masked and e.name not in self.homes:
            return e.name
        owner = self.owner(e)
        self.prepare(e, owner)
        name = self.buffer(F32, e.shape)

        def copy(idx: tuple[str, ...]) -> None:
            value = self.convert(self.value(e, idx), e.dtype, F32)
            ok = self.valid(e, idx)
            self.line(f"{name}[{self.linear(e.shape, idx)}] = {value if ok is None else f'({ok} ? {value} : 0.0f)'};")

        self.iterate(e.shape, copy, owner)
        return name

    def buffer(self, dtype: Dtype, shape: tuple[int, ...]) -> str:
        name = f"_t{len(self.buffers)}"
        self.buffers[name] = (dtype, math.prod(shape))
        return name

    def elements(self, shape: tuple[int, ...], body: Callable[[tuple[str, ...]], None]) -> None:
        """A block-cooperative loop over the elements of a tile, then a barrier."""
        with self.scope(f"for (long long _e = {self.r.tid}; _e < {math.prod(shape)}; _e += {self.r.bdim})"):
            names = tuple(f"_i{a}" for a in range(len(shape)))
            for a, name in enumerate(names):
                self.line(f"long long {name} = _e / {math.prod(shape[a + 1 :])} % {shape[a]};")
            body(names)
        self.line(self.r.sync)

    def materialize(self, e: Expr) -> None:
        """Emit the loops of every reduction under `e`; each leaves its result in a buffer."""
        for child in children(e):
            self.materialize(child)
        if isinstance(e, Reduce) and e not in self.ready:
            self.reduce(e)

    def reduce(self, e: Reduce) -> None:
        """One thread per output element walks the reduced axis, over the valid lanes only. A mean divides the sum by
        the number of valid lanes, so an element with none divides zero by zero: NaN, as numpy gives."""
        name, dtype = self.buffer(e.dtype, e.shape), self.r.vtype(e.dtype)
        count, skip = e.operand.shape[e.axis], 1 if e.keepdims else 0
        box = self.box(e.operand)

        def run(idx: tuple[str, ...]) -> None:
            def along(k: str) -> tuple[str, ...]:
                return (*idx[: e.axis], k, *idx[e.axis + skip :])

            if box is None:
                self.line(f"{dtype} _acc = {self.value(e.operand, along('0'))};")
                with self.scope(f"for (long long _r = 1; _r < {count}; ++_r)"):
                    self.line(f"{dtype} _x = {self.value(e.operand, along('_r'))};")
                    self.line(REDUCERS[e.op])
                number = self.r.literal(e.dtype, count)
            else:  # the first valid lane starts the reduction; with none the result is zero
                low, high = box[e.axis]
                self.line(f"long long _lo = {low}, _hi = {high};")
                if e.op is Reduction.MEAN:  # an element whose other axes hold no valid lane counts none
                    rows = [
                        f"({i} >= {lo} && {i} < {hi})"
                        for a, (i, (lo, hi)) in enumerate(zip(self.align(e.operand, along("0")), box, strict=True))
                        if a != e.axis
                    ]
                    self.line(f"if (!({' && '.join(rows) or '1'})) _hi = _lo;")
                self.line(f"{dtype} _acc = {self.r.zero(e.dtype)};")
                with self.scope("for (long long _r = _lo; _r < _hi; ++_r)"):
                    self.line(f"{dtype} _x = {self.value(e.operand, along('_r'))};")
                    self.line(f"if (_r > _lo) {{ {REDUCERS[e.op]} }} else {{ _acc = _x; }}")
                number = self.convert("(_hi > _lo ? _hi - _lo : 0)", I64, e.dtype)
            if e.op is Reduction.MEAN:
                self.line(f"_acc = _acc / {number};")
            self.line(f"{name}[{self.linear(e.shape, idx)}] = _acc;")

        self.elements(e.shape, run)
        self.ready[e] = name

    def linear(self, shape: tuple[int, ...], idx: tuple[str, ...]) -> str:
        return " + ".join(f"{i} * {math.prod(shape[a + 1 :])}" for a, i in enumerate(idx)) or "0"

    def convert(self, text: str, src: Dtype, dst: Dtype) -> str:
        return text if src == dst else self.r.cast(src, dst, text)

    def index(self, e: Expr) -> str:
        return f"(long long)({self.value(e)})"

    def dim(self, ref: TensorRef, axis: int) -> str:
        return self.r.leaf(ref.dim(axis))

    def inside(self, ref: TensorRef, cells: list[str]) -> str:
        """The bounds test of a tile access: one unsigned comparison per axis also rejects negative cells."""
        return " && ".join(
            f"(unsigned long long){c} < (unsigned long long){self.dim(ref, a)}" for a, c in enumerate(cells)
        )

    def address(self, ref: TensorRef, cells: list[str]) -> str:
        """Row-major offset: ((c0) * n1 + c1) * n2 + c2."""
        offset = cells[0]
        for axis, cell in enumerate(cells[1:], start=1):
            offset = f"({offset}) * {self.dim(ref, axis)} + {cell}"
        return offset

    def align(self, e: Expr, idx: tuple[str, ...]) -> tuple[str, ...]:
        """The element index of an operand: aligned from the right, and 0 along every axis it broadcasts."""
        tail = idx[len(idx) - len(e.shape) :] if e.shape else ()
        return tuple("0" if n == 1 else i for n, i in zip(e.shape, tail, strict=True))

    def value(self, e: Expr, idx: tuple[str, ...] = ()) -> str:
        """The expression of `e` at tile element `idx`; scalars ignore it."""
        return getattr(self, f"value_{type(e).__name__}")(e, self.align(e, idx))

    def valid(self, e: Expr, idx: tuple[str, ...]) -> str | None:
        """A C condition that is true where `e` has a valid lane at tile element `idx`; None when every lane is."""
        box = self.box(e)
        if box is None:
            return None
        return " && ".join(f"({i} >= {lo} && {i} < {hi})" for i, (lo, hi) in zip(self.align(e, idx), box, strict=True))

    def box(self, e: Expr) -> Box | None:
        """The valid lanes of the tile `e` as one half-open interval per axis; None when every lane is valid.

        Every mask a tile can have is such a box: a load is a rectangle cut by the edges of its tensor, an elementwise
        operation keeps the common part of its operands, and a reduction keeps the other axes. Each bound is a C
        expression, so a box costs no memory and the lanes outside it are never tested one by one.
        """
        if not e.shape or not self.invalid(e):
            return None
        return getattr(self, f"box_{type(e).__name__}", self.box_elementwise)(e)

    def full(self, e: Expr) -> Box:
        """The box of a tile with no invalid lane."""
        return [("0", str(n)) for n in e.shape]

    def box_TileLoad(self, e: TileLoad) -> Box:
        box: Box = []
        for axis, sl in enumerate(e.slices):
            if sl.squeeze:
                continue
            start, room = (
                self.index(sl.start),
                f"{self.dim(e.tensor, axis)} - {self.index(sl.start)}",
            )
            box.append((f"({start} < 0 ? -{start} : 0)", f"({room} < {sl.extent} ? {room} : {sl.extent})"))
        return box

    def box_Transpose(self, e: Transpose) -> Box:
        rows, cols = self.box(e.operand) or self.full(e.operand)
        return [cols, rows]

    def box_Var(self, e: Var) -> Box:
        return [(f"_b_{e.name}_lo{axis}", f"_b_{e.name}_hi{axis}") for axis in range(len(e.shape))]

    def box_Reduce(self, e: Reduce) -> Box:
        box = self.box(e.operand) or self.full(e.operand)
        low, high = box[e.axis]
        reduced = (
            "0",
            f"({low} < {high} ? 1 : 0)",
        )  # a reduction keeps a lane when any lane under it was valid
        if e.keepdims:
            return [*box[: e.axis], reduced, *box[e.axis + 1 :]]
        return [*box[: e.axis], *box[e.axis + 1 :]]

    def box_elementwise(self, e: Expr) -> Box:
        """The part of the result every operand is valid on; an operand with one lane on an axis it broadcasts over
        is valid on all of that axis or on none of it."""
        boxes = []
        for child in children(e):
            if not child.shape:
                continue
            shift = len(e.shape) - len(child.shape)
            aligned: Box = [("0", str(n)) for n in e.shape[:shift]]  # an axis the operand lacks does not limit it
            for axis, (low, high) in enumerate(self.box(child) or self.full(child)):
                extent = e.shape[shift + axis]
                broadcast = child.shape[axis] == 1 and extent != 1
                aligned.append(("0", f"({low} < {high} ? {extent} : 0)") if broadcast else (low, high))
            boxes.append(aligned)
        return self.meet(boxes)

    def meet(self, boxes: list[Box]) -> Box:
        """The intersection of boxes: the largest lower bound and the smallest upper bound of each axis."""
        for name, body in BOX_HELPERS.items():
            self.r.helpers.setdefault(name, f"{self.r.inline}{body}")
        meet: Box = []
        for axis in range(len(boxes[0])):
            low, high = boxes[0][axis]
            for other in boxes[1:]:
                low, high = f"_lmax({low}, {other[axis][0]})", f"_lmin({high}, {other[axis][1]})"
            meet.append((low, high))
        return meet

    def set_box(self, var: Var, box: Box | None) -> None:
        """Record in the interval variables of `var` which of its lanes are valid after this statement."""
        for axis, (low, high) in enumerate(box or self.full(var)):
            self.line(f"_b_{var.name}_lo{axis} = {low};")
            self.line(f"_b_{var.name}_hi{axis} = {high};")

    def value_Const(self, e: Const, idx: tuple[str, ...]) -> str:
        return str(int(e.value)) if e.dtype == BOOL else self.r.literal(e.dtype, e.value)

    def value_Var(self, e: Var, idx: tuple[str, ...]) -> str:
        if e.name in self.substitutions:
            return self.substitutions[e.name]
        if e.name in self.homes and self.owned is not None:
            plan, own_idx = self.owned
            if plan == self.homes[e.name] and idx == own_idx:
                assert self.r.matrix_unit is not None
                return self.r.matrix_unit.slot(e.name)
        return f"{e.name}[{self.linear(e.shape, idx)}]" if e.shape else e.name

    def value_Param(self, e: Param, idx: tuple[str, ...]) -> str:
        return self.r.leaf(e.path)

    def value_Dim(self, e: Dim, idx: tuple[str, ...]) -> str:
        return self.dim(e.tensor, e.axis)

    def value_Load(self, e: Load, idx: tuple[str, ...]) -> str:
        address = self.address(e.tensor, [self.index(i) for i in e.index])
        return self.r.load(e.tensor.type.dtype, self.r.leaf(e.tensor.path), address)

    def value_Intrinsic(self, e: Intrinsic, idx: tuple[str, ...]) -> str:
        if (atomic := catalog.BY_NAME[e.name].atomic) is not None:
            op, scope = atomic
            target = e.args[0]
            assert isinstance(target, Load)
            compare = self.value(e.args[1]) if op is AtomicOp.CAS else None
            return self.r.atomic(
                op,
                e.dtype,
                "&" + self.value(target),
                self.value(e.args[-1]),
                compare,
                shared=target.tensor.extent > 0,
                scope=scope,
            )
        bits = e.args[0].dtype.bits if e.args and e.args[0].dtype.bits else 32
        sign = {Kind.UINT: "u", Kind.SINT: "s", Kind.FLOAT: "f"}.get(e.args[0].dtype.kind, "u") if e.args else "u"
        name = f"fma._{e.dtype}" if e.name == "fma" and bits == 16 else e.name
        return self.spell(name, e, *(self.value(a, idx) for a in e.args), bits=bits, sign=sign)

    def value_Atomic(self, e: Atomic, idx: tuple[str, ...]) -> str:
        address = self.address(e.tensor, [self.index(i) for i in e.index])
        pointer = f"&{self.r.leaf(e.tensor.path)}[{address}]"
        compare = self.value(e.compare) if e.compare is not None else None
        return self.r.atomic(e.op, e.dtype, pointer, self.value(e.value), compare, shared=e.tensor.extent > 0)

    def value_Unary(self, e: Unary, idx: tuple[str, ...]) -> str:
        operand = self.value(e.operand, idx)
        text = f"({'-' if e.op is Op.NEG else e.op.value}{operand})"
        return text if e.op is Op.NOT else self.wrap(text, e.dtype)

    def value_Math(self, e: Math, idx: tuple[str, ...]) -> str:
        return self.r.math(e.fn, e.dtype, [self.value(a, idx) for a in e.args])

    def value_Binary(self, e: Binary, idx: tuple[str, ...]) -> str:
        lhs, rhs, dtype = self.value(e.lhs, idx), self.value(e.rhs, idx), e.lhs.dtype
        if e.op in COMPARES:
            return f"({lhs} {e.op.value} {rhs})"
        if (dtype.is_int or dtype == PY_INT) and e.op in (Op.DIV, Op.MOD, Op.SHL, Op.SHR):
            # Function parameters evaluate each operand once, including an atomic divisor.
            name = f"_integer_{e.op.name.lower()}_{dtype}_{e.rhs.dtype}"
            ctype, count_type = self.r.vtype(dtype), self.r.vtype(e.rhs.dtype)
            if e.op in (Op.DIV, Op.MOD):
                expression = f"(b == 0 ? 0 : a {e.op.value} b)"
            else:
                bits = dtype.bits or 64
                word = self.r.ctype(uint(max(32, bits)))
                value = f"(({word})a)" if e.op is Op.SHL else "a"
                expression = f"(b < 0 || b >= {bits} ? 0 : ({value} {e.op.value} b))"
            self.r.helpers.setdefault(
                name,
                f"{self.r.inline}{ctype} {name}({ctype} a, {count_type} b) {{ return {expression}; }}",
            )
            return self.wrap(f"{name}({lhs}, {rhs})", dtype)
        narrow = dtype.kind is Kind.UINT and dtype.bits < 32
        if narrow:  # C promotes u8 and u16 to signed int, where overflow is undefined
            lhs, rhs = f"(unsigned){lhs}", rhs if e.op in (Op.SHL, Op.SHR) else f"(unsigned){rhs}"
        return self.wrap(f"({lhs} {e.op.value} {rhs})", dtype)

    def wrap(self, text: str, dtype: Dtype) -> str:
        """Integers narrower than 32 bits wrap in their own width, not in C's promoted int."""
        return f"({self.r.ctype(dtype)}){text}" if dtype.is_int and dtype.bits < 32 else text

    def value_Cast(self, e: Cast, idx: tuple[str, ...]) -> str:
        return self.convert(self.value(e.operand, idx), e.operand.dtype, e.dtype)

    def value_Select(self, e: Select, idx: tuple[str, ...]) -> str:
        return f"({self.value(e.cond, idx)} ? {self.value(e.then, idx)} : {self.value(e.orelse, idx)})"

    def value_Transpose(self, e: Transpose, idx: tuple[str, ...]) -> str:
        return self.value(e.operand, (idx[1], idx[0]))

    def value_Arange(self, e: Arange, idx: tuple[str, ...]) -> str:
        return f"({self.value(e.start)} + {idx[e.axis]})"

    def value_Zeros(self, e: Zeros, idx: tuple[str, ...]) -> str:
        return self.r.zero(e.dtype)

    def value_Reduce(self, e: Reduce, idx: tuple[str, ...]) -> str:
        return f"{self.ready[e]}[{self.linear(e.shape, idx)}]"

    def value_TileLoad(self, e: TileLoad, idx: tuple[str, ...]) -> str:
        """Elements outside the tensor read as zero; `box` says they are invalid, so nothing consumes the zero."""
        cells = [f"({self.index(sl.start)} + {i})" for sl, i in tile_axes(e.slices, idx)]
        load = self.r.load(e.dtype, self.r.leaf(e.tensor.path), self.address(e.tensor, cells))
        return f"({self.inside(e.tensor, cells)} ? {load} : {self.r.zero(e.dtype)})"


def check_shared(kernel: Kernel, target: Target, total: int) -> None:
    if total > target.shared_bytes:
        message = (
            f"tile buffers and shared arrays need {total} bytes of shared memory, "
            f"{target.name} has {target.shared_bytes}"
        )
        raise TargetError(
            "shared memory",
            target.arch,
            message,
            "reduce tile sizes or shared arrays, or use a target with more shared memory",
            span=kernel.span,
        )
