"""Capture: read a kernel from its source, type check it with the design's arithmetic and build the IR.

The builder walks the AST once. Each `visit_*` returns a typed IR expression (or a capture-time binding such as a
tensor, a struct, a slice or a number type) and each `stmt_*` appends IR statements, so the check and the lowering to
IR are one pass and every node keeps the span of the user's text.
"""

import ast
import inspect
import logging
import math
import operator
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from krn import catalog, dispatch, host
from krn.facts.targets import CPU, Feature, Target, all_targets
from krn.frontend.atomic import AtomicOp, Strategy, plan
from krn.frontend.numerics import Accumulator, Selector, widens_to
from krn.frontend.records import AUTO
from krn.runtime.settings import current_settings, retained

from .dtypes import (
    BF16,
    BOOL,
    F16,
    F32,
    F64,
    I32,
    PY_FLOAT,
    PY_INT,
    U32,
    U64,
    Arithmetic,
    Dtype,
    Kind,
    Sketch,
    from_sketch,
    is_sketch_type,
    packed,
    to_sketch,
)
from .errors import CaptureError, Span, TargetError
from .hints import check_fusable
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
    ParamType,
    Print,
    Reduce,
    Reduction,
    Select,
    SliceExpr,
    SliceType,
    Stmt,
    Store,
    StructType,
    Sync,
    TensorRef,
    TensorType,
    TileAssign,
    TileLoad,
    TileStore,
    TileType,
    Transpose,
    Unary,
    ValueType,
    Var,
    While,
    Zeros,
    tile_shape,
)
from .matrix import FP8
from .narrow import is_narrow, refuse_product
from .numerics import float64_refusal, refuse_float64
from .performance import PerformanceWarning, warn_once
from .source import (
    KERNEL_DECORATORS,
    FieldDef,
    FuncOrClass,
    Module,
    Role,
    StructDef,
    decorators,
    foldable,
)
from .space import iteration_space, synchronizes
from .validity import children

log = logging.getLogger(__name__)

type Number = int | float
type ConstValue = Sketch  # a Const field: a number, AUTO, a type or a selector
type ConstValues = Mapping[str, Mapping[str, ConstValue]]
GENERAL = frozenset({"add_sat", "sub_sat", "avg", "abs_diff", "maximum"})  # krn.math functions resolved by dispatch
PACKED_ROWS = {Op.ADD: "vadd", Op.SUB: "vsub", Op.EQ: "vcmpeq", Op.NE: "vcmpne"}
ORDERED_ROWS = {Op.LT: "lt", Op.LE: "le", Op.GT: "gt", Op.GE: "ge"}


def packed_row(op: Op, dtype: Dtype) -> str | None:
    """The SIMD row of a lane-wise operation on a packed word, or None when packed words do not have the operation."""
    if op in ORDERED_ROWS:
        signed = dtype.lane is not None and dtype.lane.kind is Kind.SINT
        stem = f"vcmp{ORDERED_ROWS[op]}{'s' if signed else 'u'}"
    elif op in PACKED_ROWS:
        stem = PACKED_ROWS[op]
    else:
        return None
    return f"simd.{stem}{dtype.lanes}"


@dataclass(eq=False, slots=True)
class TensorVal:
    ref: TensorRef


@dataclass(eq=False, slots=True)
class StructVal:
    definition: StructDef
    fields: dict[str, Binding]
    types: dict[str, Dtype]


@dataclass(frozen=True, slots=True)
class FunctionVal:
    module: Module
    definition: ast.FunctionDef


@dataclass(frozen=True, slots=True)
class CallSpan(Span):
    """The original source position plus the inline expansion that reached it."""

    callers: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class TypeVal:
    """A number type known at capture: `T.f16`, a selector resolved for the target, `a.dtype`, a `Const` field."""

    dtype: Dtype


@dataclass(frozen=True, slots=True)
class Packed:
    """Several values from one call, for unpacking: `rows, cols = block.tile(bm, bn)`."""

    items: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Seq:
    """A list or tuple of a fixed length: each item a constant or a register that holds a snapshot, never storage.

    name: "list" or "tuple", which only decides what `+` may join.
    """

    items: tuple[Binding, ...]
    name: str


@dataclass(frozen=True, slots=True)
class Stride:
    """The elements of a `range`: `count` values from `first` in signed constant steps (count is 0 when empty)."""

    first: Expr
    step: int
    count: Expr


@dataclass(frozen=True, slots=True)
class TargetVal:
    """`target` as a `match` subject: the hardware of the capture, tested by family with class patterns."""

    target: Target


@dataclass(frozen=True, slots=True)
class Auto:
    """`AUTO`: the compiler chooses, which on a target without the mechanism means nothing."""


AUTO_VALUE = Auto()


@dataclass(slots=True)
class Weak:
    """A local whose every value so far is a bare literal: it takes the type of the first typed value it meets.

    `sites` are the statements that set it, so that they can be rewritten when the type is known, and `read` says
    whether it was used before that, in which case its type was already needed and cannot change."""

    sites: list[tuple[list[Stmt], int]]
    read: bool = False


@dataclass(frozen=True, slots=True)
class Consumer:
    """The `for` loop that iterates an inlined generator: the scope its body runs in, and what it binds and does."""

    env: dict[str, Binding]
    module: Module
    types: dict[str, Dtype]
    target: ast.expr
    body: list[ast.stmt]
    outer: Consumer | None


type Binding = Expr | TensorVal | StructVal | FunctionVal | TypeVal | Packed | Auto | Seq | TargetVal
WHAT: dict[type, str] = {
    TensorVal: "a tensor",
    StructVal: "a struct",
    TypeVal: "a type",
    Packed: "several values",
    Auto: "AUTO",
    Seq: "a sequence",
    TargetVal: "a target",
    FunctionVal: "a device function",
}


UNROLL_LIMIT = 256
SEQUENTIAL_BARRIER = (
    "the C backend runs the threads of a block one after the other, so this barrier would not wait; "
    "run the kernel in the emulator (krn.emulator.emulate) or capture it for a GPU target"
)
FIXED_SEQUENCE = "a register sequence is fixed at capture: build a new one (`xs = [*xs, v]`)"

CELLS = (
    ("thread", "index"),
    ("block", "index"),
    ("block", "dim"),
    ("grid", "dim"),
    ("cluster", "dim_blocks"),
    ("cluster", "block_index"),
)


@dataclass(frozen=True, slots=True)
class IntrinsicSpec:
    dtype: Dtype | None  # None: the type of the first argument
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    types: tuple[Dtype, ...] = ()

    @property
    def parameters(self) -> tuple[str, ...]:
        return (*self.required, *self.optional)

    @property
    def mask(self) -> int | None:
        return self.parameters.index("mask") if "mask" in self.parameters else None


INTRINSICS = {
    "cluster.block_rank": IntrinsicSpec(U32),
    "cluster.num_blocks": IntrinsicSpec(U32),
    "cluster.sync": IntrinsicSpec(U32),
    "cluster.arrive": IntrinsicSpec(U32),
    "cluster.wait": IntrinsicSpec(U32),
    "grid.thread_rank": IntrinsicSpec(U64),
    "grid.num_threads": IntrinsicSpec(U64),
    "grid.block_rank": IntrinsicSpec(U32),
    "grid.num_blocks": IntrinsicSpec(U32),
    "block.thread_rank": IntrinsicSpec(U32),
    "block.num_threads": IntrinsicSpec(U32),
    **{f"{scope}.{field}.{axis}": IntrinsicSpec(U32) for scope, field in CELLS for axis in "xyz"},
    "warp.size": IntrinsicSpec(PY_INT),
    "warp.lane": IntrinsicSpec(U32),  # where a thread is needs no hardware; talking to the other lanes does
    "warp.index": IntrinsicSpec(U32),
    **{
        f"warp.{name}": IntrinsicSpec(None, ("x", second), ("width", "mask"))
        for name, second in (
            ("shfl", "src"),
            ("shfl_up", "delta"),
            ("shfl_down", "delta"),
            ("shfl_xor", "lane_mask"),
        )
    },
    "warp.ballot": IntrinsicSpec(U64, ("predicate",), ("mask",)),
    **{
        f"warp.reduce_{name}": IntrinsicSpec(None, ("x",), ("mask",), types=(I32, U32))
        for name in ("add", "min", "max", "and", "or", "xor")
    },
    **{f"warp.{name}": IntrinsicSpec(BOOL, ("predicate",), ("mask",)) for name in ("all", "any", "eq")},
    **{f"warp.{name}": IntrinsicSpec(U64, ("x",), ("mask",)) for name in ("match_any", "match_all")},
    "warp.active_mask": IntrinsicSpec(U64),
    **{f"warp.lanemask_{suffix}": IntrinsicSpec(U64) for suffix in ("lt", "le", "eq", "ge", "gt")},
    "warp.sync": IntrinsicSpec(U32, (), ("mask",)),
    "block.sync": IntrinsicSpec(U32),
    "block.count": IntrinsicSpec(U32, ("predicate",)),
    "block.all": IntrinsicSpec(BOOL, ("predicate",)),
    "block.any": IntrinsicSpec(BOOL, ("predicate",)),
    "grid.sync": IntrinsicSpec(U32),
    **{f"thread.fence_{scope}": IntrinsicSpec(U32) for scope in ("block", "device", "system")},
    **{f"thread.fence_acq_rel_{scope}": IntrinsicSpec(U32) for scope in ("block", "device", "system")},
    "thread.fence_proxy_async": IntrinsicSpec(U32),
    "thread.clock": IntrinsicSpec(U32),
    "thread.clock64": IntrinsicSpec(U64),
    "thread.globaltimer": IntrinsicSpec(U64),
}
BINOPS: dict[type[ast.operator], Op] = {
    ast.Add: Op.ADD, ast.Sub: Op.SUB, ast.Mult: Op.MUL, ast.Div: Op.DIV, ast.FloorDiv: Op.DIV, ast.Mod: Op.MOD,
    ast.BitAnd: Op.AND, ast.BitOr: Op.OR, ast.BitXor: Op.XOR, ast.LShift: Op.SHL, ast.RShift: Op.SHR,
}  # fmt: skip
COMPARES: dict[type[ast.cmpop], Op] = {
    ast.Lt: Op.LT, ast.LtE: Op.LE, ast.Gt: Op.GT, ast.GtE: Op.GE, ast.Eq: Op.EQ, ast.NotEq: Op.NE,
}  # fmt: skip
FOLDS: dict[Op, Callable[[float, float], float | bool]] = {
    Op.ADD: operator.add, Op.SUB: operator.sub, Op.MUL: operator.mul, Op.MOD: operator.mod, Op.AND: operator.and_,
    Op.DIV: lambda a, b: (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
    if isinstance(a, int) and isinstance(b, int) else a / b,
    Op.OR: operator.or_, Op.XOR: operator.xor, Op.SHL: operator.lshift, Op.SHR: operator.rshift, Op.LT: operator.lt,
    Op.LE: operator.le, Op.GT: operator.gt, Op.GE: operator.ge, Op.EQ: operator.eq, Op.NE: operator.ne,
    Op.LAND: lambda a, b: bool(a and b), Op.LOR: lambda a, b: bool(a or b),
}  # fmt: skip
INTEGER_ONLY = {Op.AND, Op.OR, Op.XOR, Op.SHL, Op.SHR, Op.MOD}
TARGETS = tuple(Target.of(arch) for arch in ("sm_90a", "gfx942", "apple_m5", "avx512"))
RESERVED = frozenset({
    "auto", "bool", "break", "case", "char", "class", "const", "continue", "default", "delete", "do", "double", "else",
    "enum", "extern", "false", "float", "for", "goto", "if", "inline", "int", "long", "new", "register", "restrict",
    "return", "short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "true", "typedef",
    "typename", "union", "unsigned", "void", "volatile", "while",
    "bx", "by", "bz", "tx", "gx", "gy", "gz", "bdim", "threadIdx", "blockIdx", "blockDim",
})  # fmt: skip
# C and C++ keywords and the names the renderers use for launch geometry: a user's `tx` or `float` must not collide
COMPTIME_BUILTINS = {"range": range, "len": len, "min": min, "max": max, "abs": abs, "int": int}
UNARY = frozenset(Fn) - {
    Fn.MAXIMUM,
    Fn.MINIMUM,
    Fn.POW,
    Fn.CLIP,
    Fn.FMA,
}  # the elementwise methods of one operand
ARITHMETIC = {"add": Op.ADD, "sub": Op.SUB, "mul": Op.MUL, "div": Op.DIV}
BINARY = {"maximum": Fn.MAXIMUM, "minimum": Fn.MINIMUM, "pow": Fn.POW}
REDUCTIONS = (
    "sum",
    "prod",
    "mean",
    "max",
    "min",
    "logsumexp",
    "norm",
)  # take `axis` and `keepdims`
NORMALIZING = ("softmax", "log_softmax")  # take `axis` and keep the shape
FLOATS = (F32, F64)
MATH = {
    "sqrt": Fn.SQRT, "exp": Fn.EXP, "exp2": Fn.EXP2, "log": Fn.LOG, "log2": Fn.LOG2, "log10": Fn.LOG10,
    "sin": Fn.SIN, "cos": Fn.COS, "tanh": Fn.TANH, "floor": Fn.FLOOR, "ceil": Fn.CEIL, "trunc": Fn.TRUNC,
    "cbrt": Fn.CBRT, "exp10": Fn.EXP10,
}  # fmt: skip
FLUENT = {fn.value for fn in UNARY} | set(ARITHMETIC) | set(BINARY) | {"clip", "fma", *REDUCTIONS, *NORMALIZING}


def element_dtype(type: ValueType) -> Dtype:
    return type.dtype if isinstance(type, TileType) else type if isinstance(type, Dtype) else PY_INT


def retype(type: ValueType, dtype: Dtype) -> ValueType:
    return TileType(dtype, type.shape) if isinstance(type, TileType) else dtype


def wrap(dtype: Dtype, value: Number) -> Number:
    """The value of an explicit conversion: integers keep their low bits, floats stay floats."""
    if dtype.is_float:
        return float(value)
    bits = int(value) & ((1 << dtype.bits) - 1)
    return bits - (1 << dtype.bits) if dtype.kind is Kind.SINT and bits >> (dtype.bits - 1) else bits


def element_name(annotation: ast.expr | None) -> str | None:
    """`F` in the annotation `Matrix[F, M, K]`, `Vector[F]` or `F`."""
    match annotation:
        case (
            ast.Subscript(slice=ast.Tuple(elts=[ast.Name(id=name), *_]))
            | ast.Subscript(slice=ast.Name(id=name))
            | ast.Name(id=name)
        ):
            return name
    return None


def names_a_variable(node: ast.expr) -> bool:
    """Whether an expression is a variable, or an in-place call on one: `v` and `v.exp_()` both stand for `v`."""
    match node:
        case ast.Name():
            return True
        case ast.Call(func=ast.Attribute(value=inner, attr=attr)):
            return attr.endswith("_") and attr.removesuffix("_") in FLUENT and names_a_variable(inner)
    return False


def dotted(node: ast.expr) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=base, attr=attr) if head := dotted(base):
            return f"{head}.{attr}"
    return None


@retained
def capture(
    path: Path | str,
    name: str,
    *,
    consts: ConstValues | None = None,
    dtypes: Mapping[str, Dtype] | None = None,
    structs: Mapping[str, str] | None = None,
    target: Target | None = None,
    arithmetic: Arithmetic = Arithmetic.STRICT,
) -> Kernel:
    """The IR of the `@kernel` function `name` in the file at `path`, for `target`.

    consts: the `Const` fields of struct arguments by argument name (the configuration of a kernel is its first
        struct); a field without a value falls back to its default.
    dtypes: element types by parameter path (`x`, `a.values`), binding kernel and struct type parameters.
    structs: the class of each struct argument by parameter name, which stands in for the annotation, so a kernel
        written against a base class (`a: Rows`) is captured once for each format it is launched with.
    """
    module = Module.read(Path(path))
    builder = Builder(
        module,
        target or Target.of("avx512"),
        arithmetic,
        consts or {},
        dtypes or {},
        structs or {},
    )
    kernel = builder.kernel(module.kernel(name))
    refuse_float64(kernel, builder.target)
    check_fusable(kernel, builder.target, current_settings().warn)
    return kernel


def roles(
    path: Path | str,
    name: str,
    *,
    consts: ConstValues | None = None,
    dtypes: Mapping[str, Dtype] | None = None,
    structs: Mapping[str, str] | None = None,
    target: Target | None = None,
) -> dict[str, frozenset[Role]]:
    """The roles each function played in the capture of `name`, with the arguments of `capture`.

    `name` is an entry; a function it calls is a callee, a folded one when it was evaluated on constants; a function
    whose first parameter is a `@struct` with `Const` fields is configured. Methods are named `Struct.method`.
    """
    module = Module.read(Path(path))
    builder = Builder(
        module,
        target or Target.of("avx512"),
        Arithmetic.STRICT,
        consts or {},
        dtypes or {},
        structs or {},
    )
    builder.kernel(module.kernel(name))
    return {function: frozenset(found) for function, found in builder.roles.items()}


class Builder:
    def __init__(
        self,
        module: Module,
        target: Target,
        arithmetic: Arithmetic,
        consts: ConstValues,
        dtypes: Mapping[str, Dtype],
        structs: Mapping[str, str],
    ) -> None:
        self.module, self.target, self.arith, self.consts, self.dtypes = (
            module,
            target,
            arithmetic,
            consts,
            dtypes,
        )
        self.structs = structs
        self.entry: tuple[str, Span] | None = None  # the entry kernel's name and span, read before any inlining
        self.types: dict[str, Dtype] = {}
        self.fold_types: dict[str, Dtype] = {}
        self.env: dict[str, Binding] = {}
        self.out: list[Stmt] = []
        self.locals: dict[str, Var] = {}
        self.tiles: dict[str, Var] = {}
        self.shared: list[TensorRef] = []
        self.written: set[tuple[str, ...]] = set()
        self.names = set(RESERVED)
        self.assigned: Counter[str] = Counter()
        self.returned: Binding | None = None
        self.weak: dict[str, Weak] = {}
        self.augmenting: str | None = None  # the local whose `+=` is being read, which is not a use of its type
        self.consumer: Consumer | None = None
        self.closed: list[list[Stmt]] = []  # statement lists that became tuples, which can no longer be rewritten
        self.tiled = False
        self.barrier: tuple[Span, tuple[Span, ...]] | None = None
        self.call_sites: list[Span] = []
        self.roles: defaultdict[str, set[Role]] = defaultdict(set)
        self.item_counts: list[Expr] = []  # the `n` of every `for i in items(n)`, which the launch is inferred from

    def kernel(self, fn: ast.FunctionDef) -> Kernel:
        self.entry = (fn.name, self.module.span(fn))
        self.assigned = Counter(n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        self.types = self.bind_types(fn)
        self.require(fn)
        params = tuple((arg.arg, self.bind(arg)) for arg in fn.args.args)
        self.classify(fn, Role.ENTRY, self.env.get(fn.args.args[0].arg) if fn.args.args else None)
        body = self.block(fn.body)
        if self.barrier is not None and not self.tiled and isinstance(self.target, CPU):
            site, calls = self.barrier
            raise CaptureError(
                site,
                SEQUENTIAL_BARRIER
                + "; targets that have it: "
                + ", ".join(t.arch for t in all_targets() if not isinstance(t, CPU))
                + "".join(f", called from {call}" for call in reversed(calls)),
            )
        return Kernel(
            fn.name, params, tuple(self.locals.values()), tuple(self.tiles.values()), tuple(self.shared), body,
            frozenset(self.written), self.tiled, self.module.span(fn),
            iteration_space(body, self.item_counts), synchronizes(body),
        )  # fmt: skip

    def error(self, node: ast.AST, message: str) -> CaptureError:
        return CaptureError(self.module.span(node), message)

    def classify(
        self,
        fn: ast.FunctionDef,
        role: Role,
        first: Binding | None,
        owner: StructVal | None = None,
    ) -> None:
        """Record how `fn` is used, and that it is configured when its first parameter is a struct with `Const`s."""
        found = self.roles[f"{owner.definition.name}.{fn.name}" if owner else fn.name]
        found.add(role)
        if isinstance(first, StructVal) and any(field.is_const for field in first.definition.fields):
            found.add(Role.CONFIGURED)

    def require(self, fn: ast.FunctionDef) -> None:
        """Fail the capture when the target lacks a feature named by `@requires(...)`."""
        for decorator in fn.decorator_list:
            if isinstance(decorator, ast.Call) and dotted(decorator.func) == "requires":
                needed = {Feature[a.attr] for a in decorator.args if isinstance(a, ast.Attribute)}
                if missing := needed - self.target.features:
                    capable = [t.name for t in TARGETS if needed <= t.features]
                    names = ", ".join(sorted(f.name for f in missing))
                    raise self.error(
                        decorator,
                        f"target {self.target.name} lacks {names}; targets that have all: {capable}",
                    )

    def bind_types(self, fn: ast.FunctionDef) -> dict[str, Dtype]:
        """Bind tensor element parameters and positional struct parameters from the launched operands."""
        wanted, bound = {p.name for p in fn.type_params if isinstance(p, ast.TypeVar)}, {}
        for arg in fn.args.args:
            annotation = arg.annotation
            if annotation is None:
                continue
            head = ast.Name(id=self.structs[arg.arg]) if arg.arg in self.structs else annotation
            head = head.value if isinstance(head, ast.Subscript) else head
            found = self.module.lookup(head.id) if isinstance(head, ast.Name) else None
            pairs: list[tuple[str | None, Dtype | None]]
            if found and isinstance(found[1], ast.ClassDef):
                definition = found[0].struct(found[1])
                types = self.struct_types(arg, definition)
                arguments = (
                    (annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice])
                    if isinstance(annotation, ast.Subscript)
                    else []
                )
                if arguments and len(arguments) != len(definition.type_params):
                    raise self.error(
                        arg,
                        f"{definition.name} needs {len(definition.type_params)} type arguments",
                    )
                pairs = [
                    (a.id if isinstance(a, ast.Name) else None, types[p])
                    for a, p in zip(arguments, definition.type_params, strict=False)
                ]
            else:
                pairs = [(element_name(annotation), self.dtypes.get(arg.arg))]
            for name, dtype in pairs:
                if name in wanted and dtype is not None and bound.setdefault(name, dtype) != dtype:
                    raise self.error(arg, f"type parameter `{name}` is {bound[name]} and `{arg.arg}` is {dtype}")
        if unbound := wanted - bound.keys():
            raise self.error(
                fn,
                f"type parameter `{min(unbound)}` is unbound; pass dtypes={{...}} for the tensors it types",
            )
        return bound

    def struct_types(self, arg: ast.arg, definition: StructDef) -> dict[str, Dtype]:
        """Each struct parameter takes the dtype of its first supplied field; all other supplied fields agree."""
        bound: dict[str, Dtype] = {}
        fields: dict[str, str] = {}
        for field in definition.fields:
            if isinstance(field.annotation, ast.Subscript) and getattr(
                definition.module.sketch(field.annotation.value), "__name__", None
            ) not in ("Vector", "Matrix", "Tensor3"):
                continue
            name = element_name(field.annotation)
            if name not in definition.type_params:
                continue
            assert name is not None
            fields.setdefault(name, field.name)
            if (dtype := self.dtypes.get(f"{arg.arg}.{field.name}")) is not None:
                if name in bound and bound[name] != dtype:
                    raise self.error(
                        arg,
                        f"struct parameter `{name}` disagrees: {arg.arg}.{fields[name]} is {bound[name]} "
                        f"and {arg.arg}.{field.name} is {dtype}",
                    )
                if name not in bound:
                    fields[name] = field.name
                bound[name] = dtype
        for name in definition.type_params:
            if name not in bound:
                field = fields.get(name, name)
                raise self.error(
                    arg,
                    f"struct parameter `{name}` is unbound; pass dtypes={{'{arg.arg}.{field}': ...}}",
                )
        return bound

    def bind(self, arg: ast.arg) -> ParamType:
        if arg.annotation is None:
            raise self.error(arg, f"parameter `{arg.arg}` needs a type annotation")
        span = self.module.span(arg)
        self.names.add(arg.arg)  # a local of an inlined function may not shadow the parameter in the generated C
        annotation = arg.annotation
        if arg.arg in self.structs:
            annotation = ast.copy_location(ast.Name(id=self.structs[arg.arg], ctx=ast.Load()), annotation)
        match self.module.resolve(annotation, self.types):
            case Dtype() as dtype:
                self.env[arg.arg] = Param(dtype, (arg.arg,), span=span)
                return dtype
            case TensorType() as tensor:
                self.env[arg.arg] = TensorVal(TensorRef((arg.arg,), tensor))
                return tensor
            case StructDef() as definition:
                return self.bind_struct(arg, definition)

    def bind_struct(self, arg: ast.arg, definition: StructDef) -> StructType:
        types = self.struct_types(arg, definition)
        if isinstance(annotation := arg.annotation, ast.Subscript):
            arguments = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            declared = tuple(self.module.dtype(a, self.types) for a in arguments)
            actual = tuple(types[p] for p in definition.type_params)
            if declared != actual:
                have, want = (", ".join(map(str, values)) for values in (actual, declared))
                raise self.error(
                    arg,
                    f"parameter {arg.arg} is {definition.name}[{have}], the kernel declares {definition.name}[{want}]",
                )
        values: dict[str, Binding] = {}
        runtime: list[tuple[str, Dtype | TensorType]] = []
        for field in definition.fields:
            path = (arg.arg, field.name)
            if field.is_const:
                values[field.name] = self.const_field(arg, definition, field)
                continue
            resolved = definition.module.resolve(field.annotation, types)
            if isinstance(resolved, Dtype):
                values[field.name] = Param(resolved, path, span=field.span)
                runtime.append((field.name, resolved))
            elif isinstance(resolved, TensorType):
                values[field.name] = TensorVal(TensorRef(path, resolved))
                runtime.append((field.name, resolved))
            else:
                raise CaptureError(field.span, f"nested struct field `{field.name}` is not supported")
        self.env[arg.arg] = StructVal(definition, values, types)
        name = "_".join((definition.name, *(str(types[p]) for p in definition.type_params)))
        # Renderers qualify field pointers per argument, so even equal specializations need distinct declarations.
        return StructType(self.fresh(name), tuple(runtime))

    def const_field(self, arg: ast.arg, definition: StructDef, field: FieldDef) -> Binding:
        """A compile-time field: its value from the caller, else its default. A number folds into a `Const`, a type or
        a selector becomes a `TypeVal` for the target, and `AUTO` stays `AUTO`."""
        supplied = self.consts.get(arg.arg, {})
        if field.name in supplied:
            value = supplied[field.name]
        elif field.default is not None:
            value = definition.module.evaluate(field.default)
        else:
            hint = f"consts={{{arg.arg!r}: {{{field.name!r}: ...}}}}"
            raise self.error(
                arg,
                f"missing Const `{definition.name}.{field.name}` (declared at {field.span}, no default); pass {hint}",
            )
        if value is AUTO:
            return AUTO_VALUE
        if typed := self.typed(value, arg):
            return typed
        if isinstance(value, bool):
            return Const(BOOL, value, span=field.span)
        if isinstance(value, int | float):
            return (
                Const(PY_FLOAT, float(value), span=field.span)
                if field.is_float
                else Const(PY_INT, int(value), span=field.span)
            )
        raise self.error(arg, f"Const `{definition.name}.{field.name}` is {value!r}, which a kernel cannot use")

    def block(self, nodes: Sequence[ast.stmt]) -> tuple[Stmt, ...]:
        saved, self.out = self.out, []
        for node in nodes:
            self.stmt(node)
        done, self.out = self.out, saved
        self.closed.append(done)
        return tuple(done)

    def stmt(self, node: ast.stmt) -> None:
        getattr(self, f"stmt_{type(node).__name__}", self.unsupported)(node)

    def unsupported(self, node: ast.AST) -> None:
        raise self.error(node, f"unsupported construct `{type(node).__name__}`: {ast.unparse(node)[:60]}")

    def binding(self, node: ast.expr) -> Binding | None:
        return getattr(self, f"visit_{type(node).__name__}", self.unsupported)(node)

    def expr(self, node: ast.expr) -> Expr:
        return self.require_expr(self.binding(node), node)

    def require_expr(self, value: Binding | None, node: ast.expr) -> Expr:
        if not isinstance(value, Expr):
            what = "nothing" if value is None else WHAT[type(value)]
            raise self.error(node, f"expected a value, got {what}: `{ast.unparse(node)}`")
        return value

    def const_value(self, node: ast.expr, what: str) -> Number | bool:
        value = self.expr(node)
        if not isinstance(value, Const):
            raise self.capture_constant(node, what)
        return value.value

    def capture_constant(self, node: ast.AST, what: str) -> CaptureError:
        return self.error(
            node,
            f"{what} must be a capture-time constant (known at capture), got the runtime value `{ast.unparse(node)}`; "
            "a literal, a target fact or a `Const` field is known at capture",
        )

    def condition(self, node: ast.expr) -> Expr:
        value = self.expr(node)
        if value.type != BOOL:
            raise self.error(node, f"condition must be bool, got {value.type}: `{ast.unparse(node)}`")
        return value

    def fresh(self, name: str) -> str:
        unique, count = name, 1
        while unique in self.names:
            count += 1
            unique = f"{name}_{count}"
        self.names.add(unique)
        return unique

    def declare(self, name: str, type: ValueType, node: ast.AST, *, hoist: bool = True) -> Var:
        var = Var(type, self.fresh(name), span=self.module.span(node))
        if hoist:
            (self.tiles if isinstance(type, TileType) else self.locals)[var.name] = var
        self.env[name] = var
        return var

    def coerce(self, e: Expr, dtype: Dtype, node: ast.AST) -> Expr:
        """Make `e` the element type `dtype`: a literal adopts it, anything else the join allowed is a cast."""
        have = element_dtype(e.type)
        if have == dtype:
            return e
        if isinstance(e, Const) and have.is_py:
            if not dtype.fits(e.value):
                raise self.error(node, f"literal {e.value} does not fit {dtype}")
            return Const(dtype, float(e.value) if dtype.is_float else int(e.value), span=e.span)
        return Cast(retype(e.type, dtype), e, span=e.span)

    def exact(self, e: Expr, type: ValueType, node: ast.AST, what: str) -> Expr:
        """Implicit conversion for assignments, stores and arguments: only a literal or a Python int adopts a type."""
        have = element_dtype(e.type)
        if e.type == type:
            return e
        if have.is_py and isinstance(type, Dtype) and self.arith.common(have, type) == type:
            return self.coerce(e, type, node)
        raise self.error(
            node,
            f"cannot use {e.type if isinstance(e.type, Dtype) else f'{have} tile'} as {what} of type {type}",
        )

    def visit_Constant(self, node: ast.Constant) -> Expr:
        span = self.module.span(node)
        match node.value:
            case bool():
                return Const(BOOL, node.value, span=span)
            case int():
                return Const(PY_INT, node.value, span=span)
            case float():
                return Const(PY_FLOAT, node.value, span=span)
        raise self.error(node, f"unsupported literal {node.value!r}")

    def visit_Name(self, node: ast.Name) -> Binding:
        if node.id in self.weak and node.id != self.augmenting:
            self.weak[node.id].read = True
        if node.id in self.env:
            return self.env[node.id]
        if node.id in self.types:
            return TypeVal(self.types[node.id])
        if (found := self.module.lookup(node.id)) and isinstance(found[1], ast.FunctionDef):
            return FunctionVal(found[0], found[1])
        return self.package_value(node)

    def visit_Attribute(self, node: ast.Attribute) -> Binding:
        if (
            node.attr == "lanes"
            and isinstance(base := self.binding(node.value), Expr)
            and base.dtype.kind is Kind.PACKED
        ):
            return self.unpacked(base, node)
        if isinstance(node.value, ast.Call) and (cell := dotted(node.value.func)) in {f"{s}.{f}" for s, f in CELLS}:
            self.arguments(node.value, ())
            if node.attr not in "xyz" or len(node.attr) != 1:
                raise self.error(node, f"`{cell}()` has axes x, y and z")
            return self.intrinsic(f"{cell}.{node.attr}", [], node)
        path = dotted(node)
        root = (path or "").split(".")[0]
        if root not in self.env:
            cells = {f"{s}.{f}" for s, f in CELLS}
            if path and (path in INTRINSICS or path in cells):
                base, _, axis = path.rpartition(".")
                call = f"{base}().{axis}" if base in cells else f"{path}()"
                raise self.error(node, f"`{path}` is a function; write `{call}`")
            if root == "target" and isinstance(node.value, ast.Name):
                return self.target_fact(node)
            return self.package_value(node)
        base = self.binding(node.value)
        match base:
            case StructVal() if node.attr in base.fields:
                return base.fields[node.attr]
            case TensorVal(ref=ref) if node.attr == "dtype":
                return TypeVal(ref.type.dtype)
            case Expr() if node.attr == "dtype":
                return TypeVal(base.dtype)
            case Expr() if node.attr == "lanes" and base.dtype.kind is Kind.PACKED:
                return self.unpacked(base, node)
        raise self.error(node, f"`{ast.unparse(node.value)}` has no attribute `{node.attr}`")

    def target_fact(self, node: ast.Attribute) -> Const:
        """`target.warp_size`, `target.shared_bytes`: a constant of the capture."""
        value = getattr(self.target, node.attr, None)
        if not isinstance(value, bool | int):
            raise self.error(node, f"target {self.target.name} has no number or flag `{node.attr}`")
        return Const(BOOL if isinstance(value, bool) else PY_INT, value, span=self.module.span(node))

    def typed(self, found: Sketch | None, node: ast.AST) -> TypeVal | None:
        """The number type of a `krn` value: a type itself, or a selector resolved for the target."""
        if isinstance(found, Selector):
            try:
                found = found.resolve(self.target)
            except LookupError as error:
                raise self.error(node, str(error)) from None
        return TypeVal(from_sketch(found)) if is_sketch_type(found) else None

    def package_value(self, node: ast.Name | ast.Attribute) -> Binding:
        """A name from `krn` used as a value: a number type, a selector or `AUTO`. `host` is never in a kernel."""
        self.reject_host(node)
        path, found = dotted(node) or ast.unparse(node), self.module.sketch(node)
        if found is AUTO:
            return AUTO_VALUE
        if typed := self.typed(found, node):
            return typed
        if (function := self.source_function(found)) is not None and isinstance(function[1], ast.FunctionDef):
            return FunctionVal(function[0], function[1])
        if isinstance(found, float) and math.isfinite(found):
            return Const(PY_FLOAT, found, span=self.module.span(node))
        if found is None:
            raise self.error(node, f"undefined name `{path}`{self.stdlib_hint(path)}")
        raise self.error(node, f"`{path}` cannot be used as a value in a kernel")

    @staticmethod
    def stdlib_hint(path: str) -> str:
        """Why a name of the standard library is undefined in a kernel, and what stands in for it."""
        module = path.split(".")[0]
        if module == "math":
            return (
                "; the standard library's math returns Python floats, which would silently become f32: "
                "write `from krn import math`"
            )
        return "; the standard library is not available in a kernel" if module in sys.stdlib_module_names else ""

    def reject_host(self, node: ast.expr) -> None:
        path = dotted(node) or ast.unparse(node)
        if self.module.names.get(path.split(".")[0]) is host:
            raise self.error(node, f"`{path}` is host-only and cannot be used in a kernel")

    def tensor(self, node: ast.expr) -> TensorRef:
        value = self.binding(node)
        if not isinstance(value, TensorVal):
            raise self.error(node, f"`{ast.unparse(node)}` is not a tensor")
        return value.ref

    def visit_Subscript(self, node: ast.Subscript) -> Binding:
        span = self.module.span(node)
        if isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            if isinstance(base := self.binding(node.value.value), Expr) and isinstance(base.type, TileType):
                axis = self.const_value(node.slice, "a shape index")
                if not isinstance(axis, int) or not 0 <= axis < len(base.shape):
                    name = ast.unparse(node.value.value)
                    raise self.error(node, f"`{name}` has {len(base.shape)} axes, no axis {axis}")
                return Const(PY_INT, base.shape[axis], span=span)
            ref, axis = (
                self.tensor(node.value.value),
                self.const_value(node.slice, "a shape index"),
            )
            if not isinstance(axis, int) or not 0 <= axis < ref.type.rank:
                raise self.error(
                    node,
                    f"`{ast.unparse(node.value.value)}` has {ref.type.rank} axes, no axis {axis}",
                )
            return Const(PY_INT, ref.extent, span=span) if ref.extent else Dim(PY_INT, ref, axis, span=span)
        base = self.binding(node.value)
        if isinstance(base, Seq):
            return self.sequence_index(base, node)
        if not isinstance(base, TensorVal):
            raise self.error(node.value, f"`{ast.unparse(node.value)}` is not a tensor")
        ref = base.ref
        items = self.indices(node.slice, ref, node)
        if any(isinstance(i, SliceExpr) for i in items):
            slices = self.squeezed(items)
            self.tiled = True
            return TileLoad(TileType(ref.type.dtype, tile_shape(slices)), ref, slices, span=span)
        return Load(ref.type.dtype, ref, tuple(items), span=span)

    def indices(self, node: ast.expr, ref: TensorRef, whole: ast.expr) -> list[Expr]:
        """The index list of a subscript: all scalars (one element) or all slices (a tile)."""
        elements = node.elts if isinstance(node, ast.Tuple) else [node]
        if len(elements) != ref.type.rank:
            name = ".".join(ref.path)
            raise self.error(
                whole,
                f"tensor `{name}` has {ref.type.rank} axes but {len(elements)} indices were given",
            )
        items = [self.slice_expr(e) if isinstance(e, ast.Slice) else self.expr(e) for e in elements]
        for item, element in zip(items, elements, strict=True):
            dtype = item.type
            if not isinstance(dtype, SliceType) and not (
                isinstance(dtype, Dtype) and (dtype.is_int or dtype == PY_INT)
            ):
                raise self.error(element, f"index must be an integer or a slice, got {dtype}")
        return items

    def squeezed(self, items: list[Expr]) -> tuple[SliceExpr, ...]:
        """Every axis of a subscript as a slice: a scalar index becomes a squeezed slice of extent 1."""
        return tuple(
            i if isinstance(i, SliceExpr) else SliceExpr(SliceType(1), i, span=i.span, squeeze=True) for i in items
        )

    def slice_expr(self, node: ast.Slice | ast.Call) -> SliceExpr:
        args = (node.lower, node.upper) if isinstance(node, ast.Slice) else tuple(node.args)
        lo, hi = args if len(args) == 2 and args[0] else (None, args[-1])
        if hi is None:
            raise self.error(node, "a slice needs an upper bound")
        start = self.expr(lo) if lo else Const(PY_INT, 0, span=self.module.span(node))
        stop = self.expr(hi)
        match stop:
            case Const() if isinstance(start, Const):
                extent = int(stop.value - start.value)
            case Binary(op=Op.ADD, lhs=lhs, rhs=Const(value=int() as value)) if lhs == start:
                extent = value
            case _:
                raise self.capture_constant(node, "slice extent")
        if extent <= 0:
            raise self.error(node, f"slice extent must be positive, got {extent}")
        return SliceExpr(SliceType(extent), start, span=self.module.span(node))

    def element(self, node: ast.expr) -> Binding:
        value = self.binding(node)
        if value is None:
            raise self.error(node, "a sequence element must have a value")
        return value

    def snapshot(self, node: ast.expr) -> Binding:
        """An element of a sequence, evaluated once: a variable or an expression is copied into a fresh local."""
        value = self.element(node)
        if isinstance(value, Expr) and not isinstance(value, Const):
            name = self.fresh("_element")
            self.assign_name(name, value, node, self.module.span(node))
            return self.env[name]
        return value

    def sequence_elements(self, elements: Sequence[ast.expr]) -> tuple[Binding, ...]:
        values: list[Binding] = []
        for element in elements:
            if isinstance(element, ast.Starred):
                value = self.binding(element.value)
                if not isinstance(value, Seq):
                    raise self.error(element, "a starred register sequence element must be a register sequence")
                values.extend(value.items)
            else:
                values.append(self.snapshot(element))
        return tuple(values)

    def visit_List(self, node: ast.List) -> Seq:
        return Seq(self.sequence_elements(node.elts), "list")

    def visit_Tuple(self, node: ast.Tuple) -> Seq:
        return Seq(self.sequence_elements(node.elts), "tuple")

    def sequence_length(self, node: ast.AST, name: str = "list") -> CaptureError:
        return self.error(
            node,
            f"a {name} needs a length known at capture, got the runtime length from `{ast.unparse(node)}`: "
            "use a Vector, a generator (`sum(... for ...)`) or a `Const` length",
        )

    def fixed_layout(self, node: ast.AST, name: str) -> None:
        raise self.error(
            node,
            f"a {name} has no fixed layout in registers: "
            "use a Vector, a generator (`sum(... for ...)`) or a `Const` length",
        )

    def visit_Set(self, node: ast.Set) -> None:
        self.fixed_layout(node, "set")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.fixed_layout(node, "set")

    def visit_Dict(self, node: ast.Dict) -> None:
        self.fixed_layout(node, "dict")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.fixed_layout(node, "dict")

    def call_set(self, node: ast.Call) -> None:
        self.fixed_layout(node, "set")

    call_frozenset = call_set

    def call_dict(self, node: ast.Call) -> None:
        self.fixed_layout(node, "dict")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise self.error(
            node,
            "a generator expression must be consumed by sum, min, max, any, all, tuple or list",
        )

    def sequence_index(self, sequence: Seq, node: ast.Subscript) -> Binding:
        index = node.slice
        if isinstance(index, ast.Slice):
            parts = [
                None if part is None else self.const_value(part, "a sequence slice")
                for part in (index.lower, index.upper, index.step)
            ]
            if any(part is not None and not isinstance(part, int) for part in parts):
                raise self.error(index, "a register sequence slice needs integer constants")
            lo, hi, step = (None if part is None else int(part) for part in parts)
            if step == 0:
                raise self.error(index, "a register sequence slice step must not be zero")
            return Seq(sequence.items[slice(lo, hi, step)], sequence.name)
        value = self.expr(index)
        if not isinstance(value, Const):
            raise self.error(
                index,
                "a register sequence is indexed by a capture-time constant, "
                f"got the runtime value `{ast.unparse(index)}`: "
                "iterate it (`for i, x in enumerate(xs)`) or use a Vector",
            )
        if not isinstance(value.value, int):
            raise self.error(index, "a register sequence index must be an integer")
        try:
            return sequence.items[value.value]
        except IndexError:
            raise self.error(
                index,
                f"register sequence index {value.value} is out of range for length {len(sequence.items)}",
            ) from None

    def known_constant(self, node: ast.expr) -> bool:
        """A side-effect-free eligibility check; evaluating an ineligible iterable must emit nothing."""
        match node:
            case ast.Constant():
                return True
            case ast.Name(id=name):
                return isinstance(self.env.get(name), Const)
            case ast.UnaryOp(operand=operand):
                return self.known_constant(operand)
            case ast.BinOp(left=left, right=right):
                return self.known_constant(left) and self.known_constant(right)
            case ast.Attribute(value=ast.Name(id="target")):
                return True
            case ast.Attribute(value=ast.Name(id=name), attr=attr):
                owner = self.env.get(name)
                return isinstance(owner, StructVal) and isinstance(owner.fields.get(attr), Const)
            case ast.Call(func=func, args=args, keywords=[]):
                if dotted(func) == "len" and len(args) == 1:
                    return self.sequence_candidate(args[0])
                if isinstance(func, ast.Name) and (found := self.module.lookup(func.id)):
                    return (
                        isinstance(found[1], ast.FunctionDef)
                        and foldable(found[1])
                        and all(self.known_constant(a) for a in args)
                    )
                return is_sketch_type(self.module.sketch(func)) and all(self.known_constant(a) for a in args)
        return False

    def sequence_candidate(self, node: ast.expr, *, ranges: bool = False) -> bool:
        if self.flatten_call(node):
            return True
        match node:
            case ast.Name(id=name):
                return isinstance(self.env.get(name), Seq)
            case ast.List() | ast.Tuple() | ast.ListComp():
                return True
            case ast.Subscript(value=value):
                return self.sequence_candidate(value, ranges=ranges)
            case ast.BinOp(left=left, right=right):
                return self.sequence_candidate(left, ranges=ranges) or self.sequence_candidate(right, ranges=ranges)
            case ast.Call(func=ast.Name(id="range"), args=args, keywords=[]):
                return ranges and all(self.known_constant(a) for a in args)
            case ast.Call(func=ast.Name(id=name), args=args) if name in ("list", "tuple"):
                return True
            case ast.Call(func=ast.Name(id=name), args=args) if name in (
                "zip",
                "enumerate",
                "reversed",
            ):
                count = 1 if name == "enumerate" else len(args)
                iterables = args[:count]
                return (
                    all(self.sequence_candidate(a, ranges=True) for a in iterables)
                    and (ranges or any(self.sequence_candidate(a) for a in iterables))
                    and (
                        name != "enumerate"
                        or (
                            all(self.known_constant(a) for a in args[1:])
                            and all(k.value is not None and self.known_constant(k.value) for k in node.keywords)
                        )
                    )
                )
        return False

    def compile_sequence(self, node: ast.expr, name: str = "list") -> Seq:
        if not self.sequence_candidate(node, ranges=True):
            raise self.sequence_length(node, name)
        if isinstance(node, ast.Call) and dotted(node.func) == "range":
            stride = self.range_stride(node)
            if not isinstance(stride.first, Const) or not isinstance(stride.count, Const):
                raise self.sequence_length(node, name)
            count = int(stride.count.value)
            if count > UNROLL_LIMIT:
                raise self.error(
                    node,
                    f"unrolling `{ast.unparse(node)}` would emit {count} copies; "
                    f"use a loop or a generator (the cap is {UNROLL_LIMIT})",
                )
            return Seq(
                tuple(
                    Const(
                        stride.first.type,
                        int(stride.first.value) + i * stride.step,
                        span=self.module.span(node),
                    )
                    for i in range(count)
                ),
                name,
            )
        value = self.binding(node)
        if not isinstance(value, Seq):
            raise self.sequence_length(node, name)
        return value

    def call_len(self, node: ast.Call) -> Const:
        args = self.arguments(node, ("sequence",))
        value = self.binding(args["sequence"])
        if not isinstance(value, Seq):
            raise self.error(node, "`len` needs a register sequence; use a tensor's shape for its extent")
        return Const(PY_INT, len(value.items), span=self.module.span(node))

    def call_list(self, node: ast.Call) -> Seq:
        return self.sequence_call(node, "list")

    def call_tuple(self, node: ast.Call) -> Seq:
        return self.sequence_call(node, "tuple")

    def flatten_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and (dotted(node.func) or "").split(".")[0] not in self.env
            and self.module.sketch(node.func) == chain.from_iterable
        )

    def flatten_sequence(self, node: ast.Call) -> Seq:
        """Flatten fixed register sequences in row order without a device iterator."""
        source = self.arguments(node, ("iterable",))["iterable"]
        rows = (
            self.comprehension(source, "tuple")
            if isinstance(source, ast.GeneratorExp)
            else self.compile_sequence(source)
        )
        items: list[Binding] = []
        for row in rows.items:
            if not isinstance(row, Seq):
                raise self.error(node, "`chain.from_iterable` needs register sequences for its rows")
            items.extend(row.items)
        return Seq(tuple(items), "tuple")

    def sequence_call(self, node: ast.Call, name: str) -> Seq:
        if not node.args and not node.keywords:
            return Seq((), name)
        args = self.arguments(node, ("iterable",))
        source = args["iterable"]
        if isinstance(source, ast.GeneratorExp):
            return self.comprehension(source, name)
        return Seq(self.compile_sequence(source, name).items, name)

    def visit_ListComp(self, node: ast.ListComp) -> Seq:
        return self.comprehension(node, "list")

    def comprehension(self, node: ast.ListComp | ast.GeneratorExp, name: str) -> Seq:
        outer = self.compile_sequence(node.generators[0].iter, name)
        names = {n.id for clause in node.generators for n in ast.walk(clause.target) if isinstance(n, ast.Name)}
        saved = {key: self.env.get(key) for key in names}
        weak = {key: self.weak.get(key) for key in names}
        for key in names:
            self.env.pop(key, None)
            self.weak.pop(key, None)
        try:
            values = self.comprehension_items(node, name, 0, outer)
        finally:
            for key, value in saved.items():
                if value is None:
                    self.env.pop(key, None)
                else:
                    self.env[key] = value
                self.weak.pop(key, None)
                if (previous := weak[key]) is not None:
                    self.weak[key] = previous
        return Seq(values, name)

    def comprehension_items(
        self, node: ast.ListComp | ast.GeneratorExp, name: str, level: int, outer: Seq
    ) -> tuple[Binding, ...]:
        if level == len(node.generators):
            return self.sequence_elements([node.elt])
        clause = node.generators[level]
        if clause.is_async:
            raise self.error(clause, "async comprehensions are not supported")
        sequence = outer if level == 0 else self.compile_sequence(clause.iter, name)
        values: list[Binding] = []
        for item in sequence.items:
            self.unpack(clause.target, item, clause.target, unrolled=True)
            accepted = True
            for condition in clause.ifs:
                value = self.condition(condition)
                if not isinstance(value, Const):
                    raise self.sequence_length(condition, name)
                if not value.value:
                    accepted = False
                    break
            if accepted:
                values.extend(self.comprehension_items(node, name, level + 1, outer))
        return tuple(values)

    def call_enumerate(self, node: ast.Call) -> Seq:
        args = self.arguments(node, ("iterable",), ("start",))
        sequence = self.compile_sequence(args["iterable"])
        start = self.expr(args["start"]) if "start" in args else Const(PY_INT, 0, span=self.module.span(node))
        if not isinstance(start, Const):
            raise self.capture_constant(args["start"], "an unrolled enumerate start")
        if not isinstance(start.value, int):
            raise self.error(node, "an enumerate start must be an integer")
        return Seq(
            tuple(
                Seq((Const(start.type, i, span=self.module.span(node)), value), "tuple")
                for i, value in enumerate(sequence.items, start.value)
            ),
            "tuple",
        )

    def call_reversed(self, node: ast.Call) -> Seq:
        args = self.arguments(node, ("sequence",))
        value = self.compile_sequence(args["sequence"])
        return Seq(value.items[::-1], value.name)

    def call_zip(self, node: ast.Call) -> Seq:
        self.zip_keywords(node)
        sequences = [self.compile_sequence(a).items for a in node.args]
        return Seq(tuple(Seq(tuple(items), "tuple") for items in zip(*sequences, strict=False)), "tuple")

    def zip_keywords(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg != "strict" or self.const_value(keyword.value, "zip strict") is not False:
                raise self.error(
                    node,
                    "kernel `zip` stops at the shortest iterable; only `strict=False` is supported",
                )

    def visit_BinOp(self, node: ast.BinOp) -> Binding:
        if isinstance(node.op, ast.MatMult):
            return self.product(node)
        lhs, rhs = self.binding(node.left), self.binding(node.right)
        if isinstance(lhs, Seq) or isinstance(rhs, Seq):
            if isinstance(node.op, ast.Add) and isinstance(lhs, Seq) and isinstance(rhs, Seq):
                if lhs.name != rhs.name:
                    raise self.error(node, f"cannot concatenate {lhs.name} and {rhs.name} register sequences")
                return Seq(lhs.items + rhs.items, lhs.name)
            sequence, count = (lhs, rhs) if isinstance(lhs, Seq) else (rhs, lhs)
            if isinstance(node.op, ast.Mult) and isinstance(sequence, Seq):
                if isinstance(count, Const) and isinstance(count.value, int):
                    return Seq(sequence.items * count.value, sequence.name)
                raise self.sequence_length(node, sequence.name)
            raise self.error(node, f"operator `{ast.unparse(node.op)}` is not defined for a register sequence")
        return self.operate(
            type(node.op),
            self.require_expr(lhs, node.left),
            self.require_expr(rhs, node.right),
            node,
        )

    def operate(self, op: type[ast.operator], lhs: Expr, rhs: Expr, node: ast.AST) -> Expr:
        if op is ast.Pow:
            return self.power(lhs, rhs, node)
        result = self.binary(BINOPS[op], lhs, rhs, node)
        dtype = element_dtype(result.type)
        if op is ast.Div and dtype.kind in (Kind.UINT, Kind.SINT, Kind.PY_INT):
            raise self.error(node, "`/` is for floats; use `//` for integer division")
        if op is ast.FloorDiv and not dtype.is_int and dtype != PY_INT:
            raise self.error(node, "`//` is for integers; use `/` for floats")
        if dtype.kind is Kind.SINT and op in (ast.FloorDiv, ast.Mod):
            raise self.error(
                node,
                "signed `//` and `%` are not defined by the sketch's `sint` (C truncates, Python floors); "
                "convert both operands to an unsigned type when they are nonnegative",
            )
        return result

    def power(self, base: Expr, exponent: Expr, node: ast.AST) -> Expr:
        """Integer powers wrap by repeated squaring; large or nonintegral float powers use libm."""
        dtype, span = element_dtype(base.type), self.module.span(node)
        if isinstance(base, Const) and isinstance(exponent, Const) and dtype.is_py and exponent.dtype.is_py:
            value = base.value**exponent.value
            if isinstance(value, complex):
                raise self.error(node, "a constant power must be real; convert the base to T.f32 for IEEE NaN")
            folded = PY_INT if dtype == PY_INT and isinstance(exponent.value, int) and exponent.value >= 0 else PY_FLOAT
            return Const(folded, value, span=span)
        if isinstance(exponent, Const) and isinstance(exponent.value, int):
            n = exponent.value
            if n < 0 and (dtype.is_int or dtype == PY_INT):
                raise self.error(
                    node,
                    "negative integer powers return a Python float; convert the base to T.f32",
                )
            if n >= 0 and (not dtype.is_float or n <= 8):
                if n == 0:
                    one = Const(dtype, 1.0 if dtype.is_float else 1, span=span)
                    return (
                        self.binary(Op.ADD, Zeros(base.type, span=span), one, node)
                        if isinstance(base.type, TileType)
                        else one
                    )
                result: Expr | None = None
                while n:
                    if n & 1:
                        result = base if result is None else self.binary(Op.MUL, result, base, node)
                    n >>= 1
                    if n:
                        base = self.binary(Op.MUL, base, base, node)
                assert result is not None
                return result
        if dtype.is_int or dtype == PY_INT:
            match node:
                case ast.BinOp(right=source) | ast.AugAssign(value=source) | ast.Call(args=[_, source]):
                    pass
                case _:
                    source = node
            raise self.capture_constant(source, "an integer exponent")
        return self.math(Fn.POW, [base, exponent], node)

    def visit_Compare(self, node: ast.Compare) -> Expr:
        if len(node.ops) != 1:
            raise self.error(node, "chained comparisons are not supported; use `and`")
        return self.binary(COMPARES[type(node.ops[0])], self.expr(node.left), self.expr(node.comparators[0]), node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Expr:
        op = Op.LAND if isinstance(node.op, ast.And) else Op.LOR
        value = self.condition(node.values[0])
        for operand in node.values[1:]:
            if isinstance(value, Const) and bool(value.value) == (op is Op.LOR):
                break
            saved, self.out = self.out, []
            right = self.condition(operand)
            body, self.out = self.out, saved
            if not body:
                value = self.binary(op, value, right, node)
                continue
            flag = self.declare(self.fresh("short_circuit"), BOOL, node)
            self.out.append(Assign(flag, value, span=self.module.span(node)))
            body.append(Assign(flag, right, span=self.module.span(operand)))
            self.closed.append(body)
            self.out.append(
                If(
                    flag,
                    tuple(body) if op is Op.LAND else (),
                    tuple(body) if op is Op.LOR else (),
                    span=self.module.span(node),
                )
            )
            value = flag
        return value

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expr:
        operand, span = self.expr(node.operand), self.module.span(node)
        dtype = element_dtype(operand.type)
        match node.op:
            case ast.Not() if dtype == BOOL:
                return (
                    Const(BOOL, not operand.value, span=span)
                    if isinstance(operand, Const)
                    else Unary(BOOL, Op.NOT, operand, span=span)
                )
            case ast.USub() if dtype.is_float or dtype.is_py or dtype.kind is Kind.SINT:
                return (
                    Const(dtype, -operand.value, span=span)
                    if isinstance(operand, Const)
                    else Unary(operand.type, Op.NEG, operand, span=span)
                )
            case ast.Invert() if dtype.is_int:
                return Unary(operand.type, Op.INVERT, operand, span=span)
            case ast.USub() if dtype.kind is Kind.PACKED:
                row = f"simd.vneg{dtype.lanes}"
                self.available(catalog.BY_NAME[row], node)
                return Intrinsic(operand.type, row, (operand,), span=span)
        raise self.error(node, f"unary `{type(node.op).__name__}` is not defined for {dtype}")

    def visit_IfExp(self, node: ast.IfExp) -> Expr:
        cond = self.condition(node.test)
        if isinstance(cond, Const):  # a capture-time test keeps one branch, and the other is never evaluated
            return self.expr(node.body if cond.value else node.orelse)
        saved, self.out = self.out, []
        then = self.expr(node.body)
        then_body, self.out = self.out, []
        orelse = self.expr(node.orelse)
        else_body, self.out = self.out, saved
        dtype = self.arith.common(element_dtype(then.type), element_dtype(orelse.type))
        if dtype is None or isinstance(then.type, TileType):
            raise self.error(node, f"branches of a conditional have types {then.type} and {orelse.type}")
        then, orelse = self.coerce(then, dtype, node), self.coerce(orelse, dtype, node)
        span = self.module.span(node)
        if not then_body and not else_body:
            return Select(dtype, cond, then, orelse, span=span)
        result = self.declare(self.fresh("conditional"), dtype, node)
        then_body.append(Assign(result, then, span=span))
        else_body.append(Assign(result, orelse, span=span))
        self.closed.extend((then_body, else_body))
        self.out.append(If(cond, tuple(then_body), tuple(else_body), span=span))
        return result

    def check_broadcast_effects(self, operands: Sequence[Expr], shape: tuple[int, ...] | None, node: ast.AST) -> None:
        def atomic(expression: Expr) -> bool:
            return (
                isinstance(expression, Atomic)
                or (isinstance(expression, Intrinsic) and catalog.BY_NAME[expression.name].atomic is not None)
                or any(atomic(child) for child in children(expression))
            )

        if shape is not None and any(not operand.shape and atomic(operand) for operand in operands):
            raise self.error(
                node,
                "bind scalar atomic results to local variables before broadcasting over a tile; "
                "the scalar emitter would otherwise repeat the atomic for each lane",
            )

    def binary(self, op: Op, lhs: Expr, rhs: Expr, node: ast.AST) -> Expr:
        """Type check `lhs op rhs` with the design's arithmetic and build the node, folding literals."""
        if Kind.PACKED in (element_dtype(lhs.type).kind, element_dtype(rhs.type).kind):
            return self.packed_binary(op, lhs, rhs, node)
        left, right = element_dtype(lhs.type), element_dtype(rhs.type)
        shift = op in (Op.SHL, Op.SHR)
        dtype = left if shift and (right.is_int or right == PY_INT) else self.arith.common(left, right)
        if dtype is None:
            hint = "int to float is never implicit; " if left.is_float != right.is_float else ""
            raise self.error(
                node,
                f"dtype mismatch in `{op}`: {left} and {right}; {hint}convert one side: {left}(x) or x.to({left})",
            )
        if op in INTEGER_ONLY and not (dtype.is_int or dtype == PY_INT):
            raise self.error(node, f"`{op}` is only defined for integers, got {dtype}")
        shape = self.broadcast(lhs.type, rhs.type, node)
        self.check_broadcast_effects((lhs, rhs), shape, node)
        compare = op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE, Op.LAND, Op.LOR)
        if op in (Op.LAND, Op.LOR) and dtype != BOOL:
            raise self.error(node, f"`{op}` needs bool operands, got {dtype}")
        lhs, rhs = self.coerce(lhs, dtype, node), (rhs if shift else self.coerce(rhs, dtype, node))
        result: Dtype = BOOL if compare else dtype
        span = self.module.span(node)
        if isinstance(lhs, Const) and isinstance(rhs, Const) and (dtype.is_py or compare) and op in FOLDS:
            if dtype == PY_INT:
                if (op in (Op.DIV, Op.MOD) and rhs.value == 0) or (op in (Op.SHL, Op.SHR) and not 0 <= rhs.value < 64):
                    return Const(result, 0, span=span)
                if op is Op.MOD:
                    quotient = FOLDS[Op.DIV](lhs.value, rhs.value)
                    return Const(result, lhs.value - quotient * rhs.value, span=span)
            return Const(result, FOLDS[op](lhs.value, rhs.value), span=span)
        return Binary(TileType(result, shape) if shape is not None else result, op, lhs, rhs, span=span)

    def broadcast(self, a: ValueType, b: ValueType, node: ast.AST) -> tuple[int, ...] | None:
        """numpy broadcasting of tile shapes, aligned from the right; None when both are scalars."""
        shapes = [t.shape for t in (a, b) if isinstance(t, TileType)]
        if not shapes:
            return None
        width = max(len(s) for s in shapes)
        padded = [(1,) * (width - len(s)) + s for s in shapes]
        merged = tuple(max(dims) for dims in zip(*padded, strict=True))
        if any(d not in (1, m) for dims, m in zip(zip(*padded, strict=True), merged, strict=True) for d in dims):
            raise self.error(node, f"tile shapes {shapes[0]} and {shapes[-1]} do not broadcast")
        return merged

    def convert(self, dtype: Dtype, e: Expr, node: ast.AST) -> Expr:
        """An explicit conversion: u64(x), f32(x), x.to(T). Literals fold, integers keep their low bits."""
        if dtype == F64 and not self.target.has(Feature.FLOAT64):
            raise float64_refusal(self.target, self.module.span(node))
        if element_dtype(e.type) == dtype:
            return e
        source = element_dtype(e.type)
        if Kind.PACKED in (dtype.kind, source.kind) and not all(
            d.kind is Kind.PACKED or (d.is_int and d.bits == 32) for d in (dtype, source)
        ):
            raise self.error(
                node,
                f"a packed word converts only to or from a 32-bit integer; from {source} to {dtype}; use pack(...)",
            )
        if isinstance(e, Const):
            return Const(dtype, wrap(dtype, e.value), span=e.span)
        return Cast(retype(e.type, dtype), e, span=self.module.span(node))

    def dtype_arg(self, node: ast.expr) -> Dtype:
        """A number type argument: `T.f32`, `T.fp.min`, `T.acc(a.dtype)`, `c.dtype`, a `Const` field holding a type."""
        value = self.binding(node)
        if not isinstance(value, TypeVal):
            raise self.error(node, f"expected a number type like T.f32, got `{ast.unparse(node)}`")
        return value.dtype

    def intrinsic(
        self,
        path: str,
        args: Sequence[ast.expr],
        node: ast.AST,
        keywords: Sequence[ast.keyword] = (),
    ) -> Intrinsic:
        spec, entry = INTRINSICS[path], catalog.BY_NAME[path]
        self.available(entry, node)
        call = ast.copy_location(
            ast.Call(func=ast.parse(path, mode="eval").body, args=list(args), keywords=list(keywords)),
            node,
        )
        given = self.arguments(call, spec.required, spec.optional)
        values: list[Expr] = []
        for parameter in spec.parameters:
            value = given.get(parameter)
            if parameter in spec.optional and (
                value is None or (isinstance(value, ast.Constant) and value.value is None)
            ):
                default = self.target.warp_size if parameter == "width" else (1 << self.target.warp_size) - 1
                values.append(
                    Const(
                        PY_INT if parameter == "width" else U64,
                        default,
                        span=self.module.span(node),
                    )
                )
            else:
                assert value is not None
                values.append(self.expr(value))
        if spec.types and values[0].dtype not in spec.types:
            accepted = ", ".join(map(str, spec.types))
            raise self.error(node, f"`{path}` accepts {accepted}, got {values[0].dtype}")
        if "width" in spec.parameters:
            width = values[spec.parameters.index("width")]
            if isinstance(width, Const) and (
                not isinstance(width.value, int)
                or width.value <= 0
                or width.value > self.target.warp_size
                or width.value & (width.value - 1)
            ):
                raise self.error(node, f"`{path}` width must be a power of two up to {self.target.warp_size}")
        span = self.module.span(node)
        if path == "warp.active_mask" and self.call_sites:
            span = CallSpan(span.file, span.line, span.col, tuple(self.call_sites))
        return Intrinsic(spec.dtype or values[0].type, path, tuple(values), span=span)

    def family(self, entry: catalog.Entry, node: ast.Call) -> Intrinsic | Packed:
        """Bind a catalog contract without widening typed kernel operands."""
        signature = entry.signature
        assert signature is not None
        given = self.arguments(node, tuple(p.name for p in signature.params))
        types = {str(t): t for t in (U32, I32, U64, Dtype(Kind.SINT, 64), F32, F64, BOOL)}
        operands: list[Expr] = []
        generic: Dtype | None = None
        for parameter in signature.params:
            argument, kind = given[parameter.name], parameter.kind
            if kind == "const":
                constant = self.const_value(argument, f"argument `{parameter.name}`")
                if not isinstance(constant, int):
                    raise self.error(node, f"argument `{parameter.name}` must be a capture-time integer")
                if entry.name == "async_copy.copy" and constant not in (1, 2, 4):
                    raise self.error(node, "argument `count` must be 1, 2, or 4 elements")
                if entry.name == "async_copy.wait" and not 0 <= constant <= 7:
                    raise self.error(node, "argument `pending` must be between 0 and 7")
                operands.append(Const(PY_INT, constant, span=self.module.span(argument)))
                continue
            words = kind.split()
            reference = words[0] in ("ref", "out")
            access = words[0] if reference else ""
            space = words[1] if len(words) == 3 else ""
            kind = words[-1]
            if reference and not isinstance(argument, ast.Subscript):
                raise self.error(node, f"argument `{parameter.name}` must be a tensor element written t[i]")
            value = self.expr(argument)
            if reference and (not isinstance(value, Load) or str(value.dtype) == "f16"):
                raise self.error(node, f"argument `{parameter.name}` must be a supported tensor element")
            if reference and isinstance(value, Load):
                shared = value.tensor.extent > 0
                if space and shared != (space == "shared"):
                    raise self.error(node, f"argument `{parameter.name}` must be a {space} tensor element")
                if access == "out":
                    self.written.add(value.tensor.path)
            if kind == "T":
                if generic is None:
                    generic = value.dtype
                    if generic.is_py and signature.types:
                        generic = types[signature.types[0]]
                    if str(generic) not in signature.types:
                        raise self.error(
                            node,
                            f"`{entry.name}` accepts {', '.join(signature.types)}, got {value.dtype}",
                        )
                dtype = generic
            else:
                dtype = types[kind]
            operands.append(self.exact(value, dtype, node, f"argument `{parameter.name}`"))
        span = self.module.span(node)
        result = generic if signature.result == "T" else types.get(signature.result, U32)
        assert result is not None
        if entry.atomic is not None:
            choice = plan(entry.atomic[0], str(result), self.target)
            if choice.strategy is Strategy.REFUSED:
                raise TargetError(
                    entry.name,
                    self.target.arch,
                    choice.reason,
                    "use a supported atomic operand type or another target",
                    span=span,
                )
        if entry.parts:
            parts: list[Expr] = []
            for index, name in enumerate(entry.parts):
                self.available(catalog.BY_NAME[name], node)
                predicate = name.rpartition(".")[2].startswith("pred")
                dtype = BOOL if predicate else I32 if entry.name == "libm.frexp" and index == 1 else result
                parts.append(Intrinsic(dtype, name, tuple(operands), span=span))
            return Packed(tuple(parts))
        self.available(entry, node)
        return Intrinsic(result, entry.name, tuple(operands), span=span)

    def available(self, entry: catalog.Entry, node: ast.AST) -> None:
        path = entry.name
        if (entry.feature is not None and not self.target.has(entry.feature)) or not entry.lowering(self.target):
            if path.startswith("fast.") and self.target.vendor == "amd":
                raise TargetError(
                    f"fast.math.{path.removeprefix('fast.')}",
                    self.target.arch,
                    "no HIP lowering of this approximate function has been verified against the compiler yet",
                    f"the strict `krn.math.{path.removeprefix('fast.')}`",
                    span=self.module.span(node),
                )
            supported = ", ".join(
                t.arch for t in all_targets() if (entry.feature is None or t.has(entry.feature)) and entry.lowering(t)
            )
            reason = (
                f"needs {entry.feature.name}, which target {self.target.arch} lacks"
                if entry.feature is not None and not self.target.has(entry.feature)
                else f"has no lowering for target {self.target.arch}: {entry.note or 'no instruction'}"
            )
            if path.startswith("warp.reduce_") and entry.feature is not None and not self.target.has(entry.feature):
                reason += f"; {entry.note}"
            if path == "grid.sync" and isinstance(self.target, CPU):
                reason += f"; {SEQUENTIAL_BARRIER}"
            raise TargetError(
                f"fast.math.{path.removeprefix('fast.')}" if path.startswith("fast.") else path,
                self.target.arch,
                f"`{path}` {reason}; targets that have it: {supported or 'none'}",
                f"use math.{path.removeprefix('fast.')}"
                if path.startswith("fast.")
                else f"targets that have it: {supported or 'none; use a portable expression'}",
                span=self.module.span(node),
            )

    def visit_Call(self, node: ast.Call) -> Binding | None:
        path, func = dotted(node.func), node.func
        if isinstance(func, ast.Name) and func.id in self.fold_types:
            return self.fold_term(node.args[0], self.fold_types[func.id])
        if (
            isinstance(func, ast.Subscript)
            and isinstance(func.value, ast.Name)
            and (found := self.module.lookup(func.value.id))
            and isinstance(found[1], ast.ClassDef)
            and found[0].struct(found[1]).type_params
        ):
            raise self.error(node, f"construct generic struct {found[1].name} on the host")
        if isinstance(func, ast.Call):
            value = self.binding(func)
            if isinstance(value, TypeVal):
                args = self.arguments(node, ("value",))
                return self.convert(value.dtype, self.expr(args["value"]), node)
        if isinstance(func, ast.Name) and isinstance(value := self.env.get(func.id), FunctionVal):
            return self.inline(value.module, value.definition, node, None)
        if isinstance(func, ast.Name) and func.id in self.types:
            args = self.arguments(node, ("value",))
            return self.convert(self.types[func.id], self.expr(args["value"]), node)
        if isinstance(func, ast.Name) and isinstance(dtype := self.env.get(func.id), TypeVal):
            args = self.arguments(node, ("value",))
            return self.convert(dtype.dtype, self.expr(args["value"]), node)
        if (path or "").split(".")[0] not in self.env:
            if path in {f"{s}.{f}" for s, f in CELLS}:
                raise self.error(node, f"`{path}()` needs an axis; write `{path}().x`, `.y` or `.z`")
            if path == "warp.size":
                self.arguments(node, ())
                return Const(PY_INT, self.target.warp_size, span=self.module.span(node))
            if path in INTRINSICS:
                value = self.intrinsic(path, node.args, node, node.keywords)
                if path == "warp.match_all":
                    mask = self.declare(self.fresh("match_mask"), U64, node)
                    self.out.append(Assign(mask, value, span=self.module.span(node)))
                    return Packed((mask, self.binary(Op.NE, mask, Const(U64, 0, span=mask.span), node)))
                return value
            if path in catalog.FUNCTIONS:
                return self.family(catalog.FUNCTIONS[path], node)
            if path == "block.tile":
                return self.block_tile(node)
            if path and path.startswith("atomic."):
                return self.atomic(path.removeprefix("atomic."), node)
            if path == "target.has":
                return self.target_has(node)
            if isinstance(func, ast.Name) and (found := self.module.lookup(func.id)):
                return self.call_function(found, node)
            callee = self.module.sketch(func)
            if self.flatten_call(node):
                return self.flatten_sequence(node)
            if path and path.startswith(("thread.", "block.", "grid.", "warp.")) and callee is not None:
                raise self.error(
                    node,
                    f"`{path}` has no lowering for target {self.target.arch}; targets that have it: none",
                )
            if isinstance(func, ast.Name) or callee is not None:
                return self.call_package(callee, node)
        if isinstance(func, ast.Attribute):
            base = self.binding(func.value)
            if isinstance(base, Seq) and func.attr in (
                "append",
                "extend",
                "insert",
                "pop",
                "clear",
                "remove",
            ):
                raise self.error(node, FIXED_SEQUENCE)
            if isinstance(base, StructVal) and func.attr in base.definition.methods:
                return self.inline(base.definition.module, base.definition.methods[func.attr], node, base)
            if isinstance(base, Expr):
                if method := getattr(self, f"method_{func.attr}", None):
                    return method(base, node)
                if func.attr.removesuffix("_") in FLUENT:
                    return self.fluent(func.attr, base, node)
        raise self.error(node, f"unsupported call `{ast.unparse(node.func)}(...)`")

    def call_package(self, callee: Sketch | None, node: ast.Call) -> Binding | None:
        """A call of a `krn` function, a number type (a conversion), or a Python builtin such as `abs`."""
        self.reject_host(node.func)
        if dtype := self.typed(callee, node.func):
            return self.convert(dtype.dtype, self.expr(node.args[0]), node)
        name = getattr(callee, "__name__", None) or dotted(node.func)
        module = getattr(callee, "__module__", None)
        if module == "krn.math" and name in GENERAL:
            return self.general(name, node)
        if module == "krn.math" and name in ("pack", "unpack"):
            return getattr(self, f"call_{name}")(node)
        if module == "krn.intrinsics" and name is not None:
            return self.raw_intrinsic(name, node)
        if module == "krn.fast.math":
            return self.family(catalog.FUNCTIONS[f"fast.{name}"], node)
        if getattr(callee, "__module__", None) == "krn.math" and name == "pow":
            args = self.arguments(node, ("x", "y"))
            return self.math(Fn.POW, [self.expr(args["x"]), self.expr(args["y"])], node)
        if name in MATH:
            args = self.arguments(node, ("x",))
            return self.math(MATH[name], [self.expr(args["x"])], node)
        if method := getattr(self, f"call_{name}", None):
            return method(node)
        if (found := self.source_function(callee)) is not None:
            return self.call_function(found, node)
        path = ast.unparse(node.func)
        raise self.error(node, f"unsupported call `{path}(...)`{self.stdlib_hint(path)}")

    @staticmethod
    def source_function(callee: Sketch | None) -> tuple[Module, FuncOrClass] | None:
        """The source of a `@kernel` function or a `@struct` that `krn` or a file it imports defines."""
        if not (inspect.isfunction(callee) or inspect.isclass(callee)):
            return None
        try:
            file = inspect.getsourcefile(callee)
        except TypeError:
            return None  # builtin classes have no Python source
        if file is None:
            return None
        module = Module.read(Path(file))
        found = module.lookup(callee.__name__)
        if found and (KERNEL_DECORATORS | {"struct"}) & set(decorators(found[1])):
            return found
        return None

    def call_function(self, found: tuple[Module, FuncOrClass], node: ast.Call) -> Binding | None:
        module, fn = found
        if isinstance(fn, ast.FunctionDef):
            module.check_decorators(fn)
        if isinstance(fn, ast.ClassDef):
            if "struct" not in decorators(fn):
                raise self.error(node, f"`{fn.name}` is not a @struct")
            definition = module.struct(fn)
            if definition.type_params:
                raise self.error(node, f"construct generic struct {fn.name} on the host")
            fields = definition.fields
            given = self.arguments(node, (), tuple(f.name for f in fields))
            values: dict[str, Binding] = {}
            for field in fields:
                expression = given.get(field.name)
                if expression is None and field.default is None:
                    raise self.error(node, f"`{fn.name}` is missing {field.name}")
                if expression is not None:
                    value = self.binding(expression)
                else:
                    saved = self.module
                    self.module = module
                    try:
                        assert field.default is not None
                        value = self.binding(field.default)
                    finally:
                        self.module = saved
                if value is None:
                    raise self.error(node, f"field `{field.name}` needs a value")
                if field.is_const and not isinstance(value, Const | TypeVal | Auto):
                    raise self.capture_constant(node, f"Const field `{field.name}`")
                if not field.is_const and isinstance(value, Expr):
                    dtype = module.resolve(field.annotation, self.types)
                    if isinstance(dtype, Dtype) and dtype not in (PY_INT, PY_FLOAT):
                        value = self.exact(value, dtype, node, f"field `{field.name}`")
                if isinstance(value, Expr) and not isinstance(value, Const | Var):
                    value = self.expr(self.hold(value, node, field.name))
                values[field.name] = value
            return StructVal(definition, values, {})
        if isinstance(fn, ast.AsyncFunctionDef):
            raise self.error(node, f"`{fn.name}` is async, and async functions are not lowered in this slice")
        if (
            foldable(fn)
            and not node.keywords
            and len(node.args) == len(fn.args.args)
            and all(self.known_constant(a) for a in node.args)
        ):
            self.roles[fn.name] |= {Role.CALLEE, Role.FOLDED}
            return self.folded(module, fn, node)
        return self.inline(module, fn, node, None)

    def target_has(self, node: ast.Call) -> Const:
        """`target.has(Feature.TENSOR_CORE)`: a flag of the capture, so a branch on it folds."""
        feature = self.module.evaluate(node.args[0])
        if not isinstance(feature, Feature):
            raise self.error(node, f"`target.has` takes a Feature, got `{ast.unparse(node.args[0])}`")
        return Const(BOOL, self.target.has(feature), span=self.module.span(node))

    def call_isinstance(self, node: ast.Call) -> Const:
        """`isinstance(target, targets.CPU)`: the family of the target, a constant of the capture."""
        subject, family = node.args
        kind = self.module.evaluate(family)
        if not (isinstance(subject, ast.Name) and subject.id == "target" and isinstance(kind, type)):
            raise self.error(node, "only `isinstance(target, <target class>)` is a capture-time test")
        return Const(BOOL, isinstance(self.target, kind), span=self.module.span(node))

    def folded(self, module: Module, fn: ast.FunctionDef, node: ast.Call) -> Const:
        """Evaluate a foldable function (one `return <expression>` over `int`s and `bool`s) on constant arguments."""
        (expression,) = (s.value for s in fn.body if isinstance(s, ast.Return) and s.value is not None)
        args = [self.const_value(a, f"an argument of the folded `{fn.name}`") for a in node.args]
        scope = dict(zip((a.arg for a in fn.args.args), args, strict=True))
        value = eval(
            compile(ast.Expression(expression), module.file, "eval"),
            {"__builtins__": COMPTIME_BUILTINS},
            scope,
        )
        return Const(BOOL if isinstance(value, bool) else PY_INT, value, span=self.module.span(node))

    def inline(
        self,
        module: Module,
        fn: ast.FunctionDef,
        node: ast.Call,
        owner: StructVal | None,
        consumer: Consumer | None = None,
    ) -> Binding | None:
        """Expand a kernel function or a struct method at the call site: its parameters become locals or aliases.

        A generator (a function with `yield`) is expanded inside the `for` loop that consumes it, whose body runs at
        each `yield`. Parameters with defaults may be left out."""
        module.check_decorators(fn)
        params = fn.args.args[1:] if owner else fn.args.args
        defaults = fn.args.defaults
        split = len(params) - len(defaults)
        given = self.arguments(node, tuple(p.arg for p in params[:split]), tuple(p.arg for p in params[split:]))
        args = [self.binding(given[p.arg]) if p.arg in given else None for p in params]
        site = self.module.span(node)
        spans = [self.module.span(given[p.arg]) if p.arg in given else site for p in params]
        types = {name: dtype for name, dtype in owner.types.items()} if owner else {}
        wanted = {p.name for p in fn.type_params if isinstance(p, ast.TypeVar)}
        types = {name: dtype for name, dtype in types.items() if name not in wanted}
        for param, value in zip(params, args, strict=True):
            name = element_name(param.annotation)
            dtype = (
                value.dtype
                if isinstance(value, Expr)
                else value.ref.type.dtype
                if isinstance(value, TensorVal)
                else None
            )
            if name in wanted and dtype is not None and types.setdefault(name, dtype) != dtype:
                raise self.error(node, f"type parameter `{name}` is {types[name]} and `{param.arg}` is {dtype}")
        if unbound := wanted - types.keys():
            raise self.error(node, f"type parameter `{min(unbound)}` is unbound")
        generator = any(isinstance(n, ast.Yield) for n in ast.walk(fn))
        if generator and consumer is None:
            raise self.error(
                node,
                f"`{fn.name}` has `yield`: iterate it with `for ... in {fn.name}(...)` or `sum(...)`",
            )
        saved = (self.env, self.module, self.returned, self.consumer, self.types, self.assigned)
        scope: dict[str, Binding] = {"self": owner} if owner else {}
        self.env, self.module, self.returned = scope, module, None
        self.types = types
        self.assigned = Counter(n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        self.consumer = consumer if generator else self.consumer
        self.call_sites.append(site)
        try:
            for index, (param, arg, span) in enumerate(zip(params, args, spans, strict=True)):
                if arg is None and index >= split:
                    arg = self.binding(defaults[index - split])
                self.bind_argument(param, arg, span)
            self.classify(fn, Role.CALLEE, scope.get(params[0].arg) if params else None, owner)
            for statement in fn.body:
                if self.returned is not None:
                    raise self.error(
                        statement,
                        "code after `return` is not supported when a function is inlined",
                    )
                self.stmt(statement)
            result = self.returned
            if fn.returns is not None and isinstance(result, Expr):
                returns = module.dtype(fn.returns, self.types)
                if returns not in (
                    PY_INT,
                    PY_FLOAT,
                ):  # `-> int` and `-> Idx` return whatever type the body made
                    result = self.exact(result, returns, fn.returns, f"the result of `{fn.name}`")
        except CaptureError as error:
            message = error.message
            if (
                isinstance(self.target, CPU)
                and "barrier would not wait" not in message
                and any(isinstance(call, ast.Call) and dotted(call.func) == "block.sync" for call in ast.walk(fn))
            ):
                message += f"; {SEQUENTIAL_BARRIER}"
            if module.file != saved[1].file:
                message += f", called from {site}"
            if isinstance(error, TargetError):
                context = message.removeprefix(error.message)
                raise TargetError(
                    error.feature, error.target, error.reason + context, error.instead, span=error.span
                ) from None
            raise CaptureError(error.span, message) from None
        finally:
            self.call_sites.pop()
            self.env, self.module, self.returned, self.consumer, self.types, self.assigned = saved
        return result

    def bind_argument(self, param: ast.arg, arg: Binding | None, span: Span) -> None:
        if arg is None:
            raise self.error(param, f"parameter `{param.arg}` needs a value")
        if param.annotation is None:
            if isinstance(arg, Expr):
                self.assign_name(param.arg, arg, param, span)
            else:
                self.env[param.arg] = arg
            return
        if isinstance(arg, FunctionVal):
            self.env[param.arg] = arg
            return
        want = self.module.resolve(param.annotation, self.types)
        if isinstance(arg, Expr) and isinstance(want, Dtype):
            value = arg if self.generic(arg, want) else self.exact(arg, want, param, f"argument `{param.arg}`")
            if isinstance(value, Const):
                self.env[param.arg] = value
            else:
                self.assign_name(param.arg, value, param, span)
        else:
            self.env[param.arg] = arg

    def generic(self, arg: Expr, want: Dtype) -> bool:
        """`int`, `Idx` and `float` parameters take a number of any width: the callee is typed by the caller."""
        have = element_dtype(arg.type)
        scalar = not isinstance(arg.type, TileType)
        return scalar and (
            (want == PY_INT and (have.is_int or have == PY_INT))
            or (want == PY_FLOAT and (have.is_float or have == PY_FLOAT))
        )

    def call_where(self, node: ast.Call) -> Expr:
        """`where(condition, x, y)`: x where the condition holds, else y, elementwise over broadcast tiles."""
        args = self.arguments(node, ("condition", "x", "y"))
        condition, x, y = (self.expr(args[name]) for name in ("condition", "x", "y"))
        if element_dtype(condition.type) != BOOL:
            raise self.error(node, f"`where` needs a bool condition, got {condition.type}; compare with <, > or ==")
        dtype = self.arith.common(element_dtype(x.type), element_dtype(y.type))
        if dtype is None:
            raise self.error(node, f"`where` branches have types {x.type} and {y.type}; convert one with .to(...)")
        shape = self.broadcast(condition.type, x.type, node)
        shape = self.broadcast(TileType(BOOL, shape) if shape is not None else BOOL, y.type, node)
        self.check_broadcast_effects((condition, x, y), shape, node)
        x, y = self.coerce(x, dtype, node), self.coerce(y, dtype, node)
        return Select(
            TileType(dtype, shape) if shape is not None else dtype,
            condition,
            x,
            y,
            span=self.module.span(node),
        )

    def call_arange(self, node: ast.Call) -> Expr:
        """`arange(n, axis, rank=2, start=s)`: an int32 tile of extent n along `axis`, holding s + its index there."""
        args = self.arguments(node, ("extent",), ("axis", "rank", "start"))
        extent = int(self.const_value(args["extent"], "a tile extent"))
        axis = int(self.const_value(args["axis"], "an axis")) if "axis" in args else 0
        rank = int(self.const_value(args["rank"], "a rank")) if "rank" in args else 1
        if not 0 <= axis < rank:
            raise self.error(node, f"`arange` axis {axis} is out of range for rank {rank}")
        shape = tuple(extent if a == axis else 1 for a in range(rank))
        span = self.module.span(node)
        start = self.convert(I32, self.expr(args["start"]), node) if "start" in args else Const(I32, 0, span=span)
        self.tiled = True
        return Arange(TileType(I32, shape), axis, start, span=span)

    def method_transpose(self, base: Expr, node: ast.Call) -> Expr:
        """`x.transpose()` swaps the two axes of a matrix tile; the compiler reads the swapped element directly."""
        if node.args or node.keywords:
            raise self.error(node, "`transpose()` takes no arguments; a tile has two axes")
        if not (isinstance(base.type, TileType) and len(base.type.shape) == 2):
            raise self.error(node, f"`transpose()` needs a rank-2 tile, got {base.type}")
        rows, cols = base.type.shape
        return Transpose(TileType(base.dtype, (cols, rows)), base, span=self.module.span(node))

    def call_zeros(self, node: ast.Call) -> Expr:
        if node.keywords:
            raise TargetError(
                "zeros(space)",
                self.target.arch,
                "`space=` is not lowered in this slice: every tile lives where the target puts it",
                "omit space to use the target's default tile storage",
                span=self.module.span(node),
            )
        shape = tuple(int(self.const_value(a, "a tile extent")) for a in node.args[1:])
        self.tiled = True
        return Zeros(TileType(self.dtype_arg(node.args[0]), shape), span=self.module.span(node))

    def call_acc(self, node: ast.Call) -> TypeVal:
        """`T.acc(dtype)`: the type a sum of products of `dtype` is kept in, by the rule of the package."""
        dtype = self.dtype_arg(node.args[0])
        try:
            accumulator = Accumulator(to_sketch(dtype)).resolve(self.target)
        except TypeError as error:
            raise self.error(node, str(error)) from None
        return TypeVal(from_sketch(accumulator))

    def call_selector(self, node: ast.Call) -> TypeVal:
        """`T.fp.emax(8)` and `T.fp.mmax(8)`: evaluated by the package, then resolved for the target."""
        if (selected := self.typed(self.module.evaluate(node), node)) is None:
            raise self.error(node, f"`{ast.unparse(node)}` is not a number type")
        return selected

    call_emax = call_mmax = call_selector

    def call_sum(self, node: ast.Call) -> Expr:
        return self.fold(node, "sum")

    def call_any(self, node: ast.Call) -> Expr:
        if self.is_lane_comparison(node):
            return self.lane_test(node, "any")
        return self.fold(node, "any")

    def call_all(self, node: ast.Call) -> Expr:
        if self.is_lane_comparison(node):
            return self.lane_test(node, "all")
        return self.fold(node, "all")

    @staticmethod
    def is_lane_comparison(node: ast.Call) -> bool:
        return len(node.args) == 1 and isinstance(node.args[0], ast.Compare)

    def lane_test(self, node: ast.Call, name: str) -> Expr:
        """`all(a OP b)` and `any(a OP b)` over the lanes of two packed words. The comparison's SIMD set row holds 1 in
        each true lane: all lanes are true when the word equals the ones of every lane, any when it is nonzero."""
        compare = node.args[0]
        assert isinstance(compare, ast.Compare)
        if len(compare.ops) != 1:
            raise self.error(node, f"`{name}` takes one comparison of two packed words, such as {name}(a == b)")
        lhs, rhs = self.expr(compare.left), self.expr(compare.comparators[0])
        if lhs.dtype.kind is not Kind.PACKED or rhs.dtype != lhs.dtype:
            raise self.error(
                node,
                f"`{name}` takes a comparison of two packed words of one type, such as `{name}(a == b)`",
            )
        self.scalar_only(f"`{name}`", (lhs, rhs), node)
        span = self.module.span(node)
        row = packed_row(COMPARES[type(compare.ops[0])], lhs.dtype)
        assert row is not None
        lanes, row = lhs.dtype.lanes, row.replace("simd.vcmp", "simd.vset", 1)
        self.available(catalog.BY_NAME[row], node)
        test = Intrinsic(U32, row, (lhs, rhs), span=span)
        if name == "any":
            return self.binary(Op.NE, test, Const(U32, 0, span=span), node)
        ones = sum(1 << (lane * (32 // lanes)) for lane in range(lanes))
        return self.binary(Op.EQ, test, Const(U32, ones, span=span), node)

    def synthetic(self, statements: Sequence[ast.stmt], node: ast.AST) -> None:
        for statement in statements:
            ast.copy_location(statement, node)
            ast.fix_missing_locations(statement)
            self.stmt(statement)

    def generator_loops(self, generator: ast.GeneratorExp, body: list[ast.stmt]) -> list[ast.stmt]:
        for clause in reversed(generator.generators):
            if clause.is_async:
                raise self.error(generator, "async generators are not supported")
            for condition in reversed(clause.ifs):
                body = [ast.copy_location(ast.If(condition, body, []), condition)]
            body = [ast.copy_location(ast.For(clause.target, clause.iter, body, []), clause.iter)]
        return body

    def fold(self, node: ast.Call, name: str) -> Expr:
        """Fold terms in their own type, or sum losslessly in `acc`, widening product operands before multiplying."""
        if name != "sum" and any(k.arg == "acc" for k in node.keywords):
            raise self.error(node, "only sum takes acc=")
        args = self.arguments(node, ("terms",), ("acc",))
        accumulator = self.dtype_arg(args["acc"]) if "acc" in args else None
        source = args["terms"]
        if accumulator is not None and isinstance(source, ast.List | ast.Tuple):
            source = ast.copy_location(
                ast.List(
                    [
                        item if isinstance(item, ast.Starred) else self.fold_cast(item, accumulator)
                        for item in source.elts
                    ],
                    ast.Load(),
                ),
                source,
            )
        if isinstance(source, ast.GeneratorExp):
            generator = source
        elif source is not None and (
            self.sequence_candidate(source) or (isinstance(source, ast.Call) and dotted(source.func) == "range")
        ):
            item = self.fresh("_fold_item")
            generator = ast.GeneratorExp(
                ast.Name(item, ast.Load()),
                [ast.comprehension(ast.Name(item, ast.Store()), source, [], 0)],
            )
        else:
            message = (
                "min/max lower only the two-operand form"
                if name in ("min", "max")
                else f"in a kernel `{name}` takes one generator expression, a range or a register sequence"
            )
            raise self.error(node, message)
        outer = generator.generators[0]
        empty = False
        if self.sequence_candidate(outer.iter):
            sequence = self.compile_sequence(outer.iter)
            empty = not sequence.items
            held = self.fresh("_fold_sequence")
            self.env[held] = sequence
            outer = ast.comprehension(outer.target, ast.Name(held, ast.Load()), outer.ifs, outer.is_async)
        elif (
            isinstance(outer.iter, ast.Call)
            and dotted(outer.iter.func) == "range"
            and all(self.known_constant(a) for a in outer.iter.args)
        ):
            lo, hi, step = self.range_bounds(outer.iter)
            empty = isinstance(lo, Const) and isinstance(hi, Const) and not range(int(lo.value), int(hi.value), step)
            iterable = ast.copy_location(
                ast.Call(
                    ast.Name("range", ast.Load()),
                    [
                        self.hold(lo, outer.iter, "_fold_lo"),
                        self.hold(hi, outer.iter, "_fold_hi"),
                        ast.Constant(step),
                    ],
                    [],
                ),
                outer.iter,
            )
            outer = ast.comprehension(outer.target, iterable, outer.ifs, outer.is_async)
        generator = ast.GeneratorExp(generator.elt, [outer, *generator.generators[1:]])
        if empty and name in ("min", "max"):
            raise self.error(node, f"`{name}()` of an empty sequence: nothing to fold")
        result = self.fresh(f"_{name}")
        self.assigned[result] = 2
        load = ast.Name(result, ast.Load())
        target = ast.Name(result, ast.Store())
        prefix: list[ast.stmt]
        body: list[ast.stmt]
        suffix: list[ast.stmt] = []
        if name == "sum":
            term = generator.elt
            if accumulator is not None:
                zero = self.fresh("_fold_zero")
                self.env[zero] = Const(accumulator, 0, span=self.module.span(node))
                initial: ast.expr = ast.Name(zero, ast.Load())
                term = self.fold_cast(term, accumulator)
            else:
                initial = ast.Constant(0)
            prefix = [ast.Assign([target], initial)]
            body = [ast.AugAssign(target, ast.Add(), term)]
        elif name in ("any", "all"):
            prefix = [ast.Assign([target], ast.Constant(name == "all"))]
            update: list[ast.stmt] = [ast.Assign([target], ast.Constant(name == "any"))]
            body = [ast.If(generator.elt, update if name == "any" else [], update if name == "all" else [])]
        else:
            first, term = self.fresh("_first"), self.fresh("_term")
            self.assigned[first] = self.assigned[term] = 2
            prefix = [ast.Assign([ast.Name(first, ast.Store())], ast.Constant(True))]
            body = [
                ast.Assign([ast.Name(term, ast.Store())], generator.elt),
                ast.If(
                    ast.Name(first, ast.Load()),
                    [
                        ast.Assign([target], ast.Name(term, ast.Load())),
                        ast.Assign([ast.Name(first, ast.Store())], ast.Constant(False)),
                    ],
                    [
                        ast.Assign(
                            [target],
                            ast.Call(ast.Name(name, ast.Load()), [load, ast.Name(term, ast.Load())], []),
                        )
                    ],
                ),
            ]
            if not isinstance(self.target, CPU):  # the C renderer's assert calls printf without declaring it
                suffix = [
                    ast.Assert(
                        ast.UnaryOp(ast.Not(), ast.Name(first, ast.Load())),
                        ast.Constant(f"{name}() iterable argument is empty"),
                    )
                ]
        self.synthetic([*prefix, *self.generator_loops(generator, body), *suffix], node)
        return self.require_expr(self.env[result], node)

    def fold_cast(self, node: ast.expr, dtype: Dtype) -> ast.Call:
        """Delay widening until the synthetic fold visits a term, before sequence snapshots or scalar multiplication."""
        name = self.fresh("_fold_widen")
        self.fold_types[name] = dtype
        return ast.copy_location(ast.Call(ast.Name(name, ast.Load()), [node], []), node)

    def fold_term(self, node: ast.expr, dtype: Dtype) -> Expr:
        """A widening multiply-add evaluates each product operand in the accumulator type."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self.binary(Op.MUL, self.fold_term(node.left, dtype), self.fold_term(node.right, dtype), node)
        term = self.expr(node)
        source = element_dtype(term.type)
        if source.is_py:
            return self.coerce(term, dtype, node)
        if not widens_to(to_sketch(source), to_sketch(dtype)):
            raise self.error(
                node,
                f"term `{ast.unparse(node)}` has dtype {source}, which does not convert losslessly to "
                f"accumulator {dtype}; convert explicitly",
            )
        return self.convert(dtype, term, node)

    def call_items(self, node: ast.Call) -> None:
        raise self.error(node, "`items(...)` is the iterable of a for loop: `for i in items(n)`")

    def call_pipelined(self, node: ast.Call) -> None:
        raise self.error(
            node,
            "`pipelined(...)` is the iterable of a for loop: `for ks in pipelined(extent, tile)`",
        )

    def call_copy(self, node: ast.Call) -> None:
        raise self.error(node, "`copy` belongs to async kernels, which are not lowered in this slice")

    def call_breakpoint(self, node: ast.Call) -> None:
        self.out.append(Breakpoint(span=self.module.span(node)))

    def call_print(self, node: ast.Call) -> None:
        if node.keywords:
            raise self.error(node, "`print` in a kernel takes values only: no `sep`, `end` or `file`")
        parts = [
            a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else self.expr(a) for a in node.args
        ]
        if any(isinstance(p, Expr) and isinstance(p.type, TileType | SliceType) for p in parts):
            raise self.error(node, "`print` takes scalars; index a tile or reduce it first")
        self.out.append(Print(tuple(parts), span=self.module.span(node)))

    def call_popcount(self, node: ast.Call) -> Intrinsic:
        return self.bits("popcount", U32, self.expr(node.args[0]), node)

    def call_leading_zeros(self, node: ast.Call) -> Intrinsic:
        return self.bits("leading_zeros", U32, self.expr(node.args[0]), node)

    def call_first_set(self, node: ast.Call) -> Intrinsic:
        return self.bits("first_set", U32, self.expr(node.args[0]), node)

    def call_reverse_bits(self, node: ast.Call) -> Intrinsic:
        return self.bits("reverse_bits", None, self.expr(node.args[0]), node)

    def bits(self, name: str, result: Dtype | None, x: Expr, node: ast.Call) -> Intrinsic:
        """A bit-twiddling intrinsic over one unsigned or signed integer operand."""
        if not (element_dtype(x.type).is_int and isinstance(x.type, Dtype)):
            raise self.error(node, f"`{name}` takes a fixed-width integer, got {x.type}")
        self.available(catalog.BY_NAME[name], node)
        return Intrinsic(result or x.type, name, (x,), span=self.module.span(node))

    def call_fma(self, node: ast.Call) -> Expr:
        args = self.arguments(node, ("a", "b", "c"))
        operands = [self.expr(args[name]) for name in ("a", "b", "c")]
        dtype, span = element_dtype(operands[0].type), self.module.span(node)
        if dtype == F64 and not self.target.has(Feature.FLOAT64):
            raise TargetError(
                "math.fma",
                self.target.arch,
                "this target has no float64 arithmetic",
                "convert to f32 or select a target with float64 arithmetic",
                span=span,
            )
        if dtype not in (F16, BF16):
            return self.math(Fn.FMA, operands, node)
        native = catalog.BY_NAME[f"fma._{dtype}"]
        if not native.lowering(self.target):
            raise TargetError(
                "math.fma",
                self.target.arch,
                native.note,
                "convert to f32 and call math.fma, or select a target with native half FMA",
                span=span,
            )
        return Intrinsic(dtype, "fma", tuple(self.exact(o, dtype, node, "operand") for o in operands), span=span)

    def call_div(self, node: ast.Call) -> Expr:
        args = self.arguments(node, ("x", "y"))
        x = self.expr(args["x"])
        if element_dtype(x.type) not in FLOATS:
            raise self.error(node, f"div needs f32 or f64, got {x.type}; `x / y` also divides integers")
        return self.binary(Op.DIV, x, self.expr(args["y"]), node)

    def call_reciprocal(self, node: ast.Call) -> Expr:
        return self.math(Fn.RECIPROCAL, [self.expr(self.arguments(node, ("x",))["x"])], node)

    def call_rsqrt(self, node: ast.Call) -> Expr:
        return self.math(Fn.RSQRT, [self.expr(self.arguments(node, ("x",))["x"])], node)

    def call_tan(self, node: ast.Call) -> Binding:
        return self.family(catalog.FUNCTIONS["libm.tan"], node)

    def method_bit_count(self, base: Expr, node: ast.Call) -> Expr:
        return self.bit_method("bit_count", base, node)

    def method_bit_length(self, base: Expr, node: ast.Call) -> Expr:
        return self.bit_method("bit_length", base, node)

    def bit_method(self, name: str, base: Expr, node: ast.Call) -> Expr:
        self.arguments(node, ())
        if isinstance(base, Const) and base.dtype == PY_INT and isinstance(base.value, int):
            value = base.value.bit_count() if name == "bit_count" else base.value.bit_length()
            return Const(PY_INT, value, span=self.module.span(node))
        dtype = base.type
        if not isinstance(dtype, Dtype) or dtype.kind is not Kind.UINT:
            raise self.error(
                node,
                f"`{name}` needs an unsigned integer scalar: Python counts the magnitude where the kernel "
                "counts the bit pattern; use popcount or leading_zeros for fixed-width signed bits",
            )
        if isinstance(base, Const) and isinstance(base.value, int):
            value = base.value.bit_count() if name == "bit_count" else base.value.bit_length()
            return Const(U32, value, span=self.module.span(node))
        if name == "bit_count":
            return self.bits("popcount", U32, base, node)
        wide = self.convert(U32 if dtype.bits <= 32 else U64, base, node)
        zeros = self.bits("leading_zeros", U32, wide, node)
        return self.binary(Op.SUB, Const(U32, 32 if dtype.bits <= 32 else 64, span=base.span), zeros, node)

    def call_isnan(self, node: ast.Call) -> Expr:
        x = self.float_scalar(self.expr(self.arguments(node, ("x",))["x"]), node)
        return self.binary(Op.NE, x, x, node)

    def call_isfinite(self, node: ast.Call) -> Expr:
        x = self.float_scalar(self.expr(self.arguments(node, ("x",))["x"]), node)
        difference = self.binary(Op.SUB, x, x, node)
        return self.binary(Op.EQ, difference, Const(PY_FLOAT, 0.0, span=x.span), node)

    def float_scalar(self, x: Expr, node: ast.AST) -> Expr:
        if x.type not in FLOATS:
            raise self.error(node, f"math predicate or intrinsic needs an f32 or f64 scalar, got {x.type}")
        return x

    def call_copysign(self, node: ast.Call) -> Expr:
        args = self.arguments(node, ("x", "y"))
        x, y = self.expr(args["x"]), self.expr(args["y"])
        dtype = self.arith.common(x.dtype, y.dtype)
        if dtype not in FLOATS or isinstance(x.type, TileType) or isinstance(y.type, TileType):
            raise self.error(node, f"copysign needs matching f32 or f64 scalars, got {x.type} and {y.type}")
        assert dtype is not None
        return self.scalar_math_intrinsic(
            "copysign", dtype, (self.coerce(x, dtype, node), self.coerce(y, dtype, node)), node
        )

    def call_ldexp(self, node: ast.Call) -> Expr:
        args = self.arguments(node, ("x", "exponent"))
        x = self.float_scalar(self.expr(args["x"]), node)
        exponent = self.expr(args["exponent"])
        if not isinstance(exponent.type, Dtype) or not (exponent.dtype.is_int or exponent.dtype == PY_INT):
            raise self.error(node, "ldexp exponent must be an integer scalar")
        return self.scalar_math_intrinsic("ldexp", x.type, (x, self.convert(I32, exponent, node)), node)

    def scalar_math_intrinsic(self, name: str, dtype: ValueType, operands: tuple[Expr, ...], node: ast.Call) -> Expr:
        self.available(catalog.BY_NAME[name], node)
        return Intrinsic(dtype, name, operands, span=self.module.span(node))

    def call_sleep(self, node: ast.Call) -> None:
        ns = self.coerce(self.expr(node.args[0]), U32, node)
        self.out.append(
            Effect(
                Intrinsic(U32, "sleep", (ns,), span=self.module.span(node)),
                span=self.module.span(node),
            )
        )

    def cell(self, cell: ast.expr, node: ast.AST) -> Expr:
        """One index of an element access: an integer expression, where a Python int literal adopts the index type."""
        value = self.expr(cell)
        return self.coerce(value, PY_INT, node) if element_dtype(value.type).is_py else value

    def atomic(self, name: str, node: ast.Call) -> Atomic:
        """`atomic.add(tensor, index, value)` and the other ten: the old value of the element."""
        op = AtomicOp(name.removesuffix("_"))
        names = ("tensor", "index", "expected", "value") if op is AtomicOp.CAS else ("tensor", "index", "value")
        args = self.arguments(node, names)
        ref = self.tensor(args["tensor"])
        cells = args["index"].elts if isinstance(args["index"], ast.Tuple) else [args["index"]]
        if len(cells) != ref.type.rank:
            raise self.error(
                node,
                f"`{ast.unparse(args['tensor'])}` has {ref.type.rank} axes, the index has {len(cells)}",
            )
        dtype, span = ref.type.dtype, self.module.span(node)
        if (chosen := plan(op, str(dtype), self.target)).strategy is Strategy.REFUSED:
            raise TargetError(
                f"atomic.{op}",
                self.target.arch,
                chosen.reason,
                "use a supported atomic operand type or another target",
                span=span,
            )
        index = tuple(self.cell(c, node) for c in cells)
        value = self.exact(self.expr(args["value"]), dtype, node, f"the operand of atomic.{name}")
        compare = (
            self.exact(self.expr(args["expected"]), dtype, node, "the expected value") if op is AtomicOp.CAS else None
        )
        self.written.add(ref.path)
        return Atomic(dtype, op, ref, index, value, compare, span=span)

    def call_slice(self, node: ast.Call) -> Expr:
        return self.slice_expr(node)

    def call_abs(self, node: ast.Call) -> Expr:
        """Signed magnitude wraps, including abs(minimum) == minimum."""
        x = self.expr(self.arguments(node, ("x",))["x"])
        if isinstance(x, Const):
            value = abs(x.value)
            return Const(x.type, wrap(x.dtype, value) if x.dtype.is_int else value, span=x.span)
        if x.dtype.is_float:
            return self.math(Fn.ABS, [x], node)
        if isinstance(x.type, TileType):
            raise self.error(node, "abs on tiles is float-only; convert with .to(T.f32)")
        if x.dtype.kind is Kind.UINT:
            return x
        if x.dtype.kind is Kind.SINT:
            negative = self.binary(Op.LT, x, Const(PY_INT, 0, span=x.span), node)
            return Select(x.type, negative, Unary(x.type, Op.NEG, x, span=x.span), x, span=x.span)
        raise self.error(node, f"abs needs a fixed-width scalar, got {x.type}")

    def call_pow(self, node: ast.Call) -> Expr:
        if len(node.args) != 2 or node.keywords:
            raise self.error(
                node,
                "pow lowers only the two-operand form; the third modular argument is unsupported",
            )
        return self.power(self.expr(node.args[0]), self.expr(node.args[1]), node)

    def call_divmod(self, node: ast.Call) -> Packed:
        args = self.arguments(node, ("a", "b"))
        lhs, rhs = self.expr(args["a"]), self.expr(args["b"])
        return Packed((self.operate(ast.FloorDiv, lhs, rhs, node), self.operate(ast.Mod, lhs, rhs, node)))

    def call_min(self, node: ast.Call) -> Expr:
        if any(k.arg == "acc" for k in node.keywords):
            raise self.error(node, "only sum takes acc=")
        if len(node.args) == 1:
            return self.fold(node, "min")
        return self.extremum(Fn.MINIMUM, Op.LT, node)

    def call_max(self, node: ast.Call) -> Expr:
        if any(k.arg == "acc" for k in node.keywords):
            raise self.error(node, "only sum takes acc=")
        if len(node.args) == 1:
            return self.fold(node, "max")
        return self.extremum(Fn.MAXIMUM, Op.GT, node)

    def extremum(self, fn: Fn, op: Op, node: ast.Call) -> Expr:
        """Float extrema propagate either NaN; unlike Python, the result is independent of argument order."""
        if len(node.args) != 2 or node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise self.error(node, "min/max lower only the two-operand form")
        lhs, rhs = (self.expr(a) for a in node.args)
        if isinstance(lhs, Const) and isinstance(rhs, Const):
            value = min(lhs.value, rhs.value) if op is Op.LT else max(lhs.value, rhs.value)
            dtype = self.arith.common(lhs.dtype, rhs.dtype)
            if dtype is not None and dtype.is_py:
                return Const(dtype, value, span=self.module.span(node))
        if lhs.dtype.is_float or rhs.dtype.is_float:
            return self.math(fn, [lhs, rhs], node)
        comparison = self.binary(op, lhs, rhs, node)
        if isinstance(comparison, Const):
            dtype = self.arith.common(lhs.dtype, rhs.dtype)
            assert dtype is not None  # binary already checked the operand types
            return self.coerce(lhs if comparison.value else rhs, dtype, node)
        if not isinstance(comparison, Binary) or not (comparison.lhs.dtype.is_int or comparison.lhs.dtype == PY_INT):
            raise self.error(node, "min/max need integer scalars or f32/f64 operands")
        return Select(
            comparison.lhs.type,
            comparison,
            comparison.lhs,
            comparison.rhs,
            span=self.module.span(node),
        )

    def call_maximum(self, node: ast.Call) -> Expr:
        return self.pair(Fn.MAXIMUM, node)

    def call_minimum(self, node: ast.Call) -> Expr:
        return self.pair(Fn.MINIMUM, node)

    def pair(self, fn: Fn, node: ast.Call) -> Expr:
        args = self.arguments(node, ("a", "b"))
        return self.math(fn, [self.expr(args["a"]), self.expr(args["b"])], node)

    def arguments(
        self, node: ast.Call, required: tuple[str, ...], optional: tuple[str, ...] = ()
    ) -> dict[str, ast.expr]:
        """The arguments of a call by parameter name: positional ones first, then keywords."""
        names, callee = (*required, *optional), ast.unparse(node.func)
        if len(node.args) > len(names):
            raise self.error(node, f"`{callee}` takes at most {len(names)} arguments ({', '.join(names)})")
        given = dict(zip(names, node.args, strict=False))
        for keyword in node.keywords:
            if keyword.arg not in names or keyword.arg in given:
                raise self.error(keyword.value, f"unexpected argument `{keyword.arg}` to `{callee}`")
            given[keyword.arg] = keyword.value
        if missing := [name for name in required if name not in given]:
            raise self.error(node, f"`{callee}` is missing {', '.join(missing)}")
        return given

    def general(self, name: str, node: ast.Call) -> Expr:
        """`krn.math.add_sat`, `abs_diff` and `maximum`: resolved by operand type and target, folded left for n-ary."""
        arity = (2, 3) if name == "maximum" else (2,)
        values = [self.expr(argument) for argument in node.args]
        if node.keywords or len(values) not in arity:
            raise self.error(node, f"`{name}` takes {' or '.join(map(str, arity))} operands")
        self.scalar_only(f"`{name}`", values, node)
        result = values[0]
        for operand in values[1:]:
            result = self.general_pair(name, result, operand, node)
        return result

    def general_pair(self, name: str, lhs: Expr, rhs: Expr, node: ast.Call) -> Expr:
        dtype = self.arith.common(lhs.dtype, rhs.dtype)
        if dtype is None:
            raise self.error(node, f"`{name}` needs operands of one type, got {lhs.dtype} and {rhs.dtype}")
        span = self.module.span(node)
        overload = dispatch.resolve(name, dtype, self.target, span)
        lhs = self.expr(self.hold(self.coerce(lhs, dtype, node), node, "_lhs"))
        rhs = self.expr(self.hold(self.coerce(rhs, dtype, node), node, "_rhs"))
        built = dispatch.lower(overload, (lhs, rhs), span, self.target)
        settings = current_settings().warn
        if (
            not built.native
            and self.target.vendor != "cpu"
            and settings.warn_software_fallback
            and built.instructions >= settings.software_fallback_min_instructions
        ):
            assert self.entry is not None
            kernel, anchor = self.entry
            warn_once(
                kernel,
                self.target.arch,
                f"software-fallback:{name}",
                dispatch.fallback_message(overload, built, dtype, self.target),
                PerformanceWarning,
                span,
                anchor,
            )
        return built.expr

    def scalar_only(self, feature: str, operands: Sequence[Expr], node: ast.AST) -> None:
        """Packed words and the general functions compute one value per thread, so a tile operand is refused."""
        if any(isinstance(operand.type, TileType) for operand in operands):
            raise TargetError(
                f"{feature} on a tile",
                self.target.arch,
                "packed vector operations are scalar-per-thread for now",
                "index the tile element and call per element",
                span=self.module.span(node),
            )

    def packed_binary(self, op: Op, lhs: Expr, rhs: Expr, node: ast.AST) -> Expr:
        """Lane-wise `+`, `-` and the comparisons of two packed words of one type; a comparison is a lane mask."""
        dtype = lhs.dtype
        if (row := packed_row(op, dtype)) is None:
            raise self.error(
                node,
                f"`{op}` is not defined for {dtype}: packed words add, subtract and compare lane by lane; "
                "use unpack(x) for other arithmetic",
            )
        if rhs.dtype != dtype:
            raise self.error(
                node, f"dtype mismatch in `{op}`: {dtype} and {rhs.dtype}; packed words combine with their own type"
            )
        self.scalar_only(f"`{op}` on {dtype}", (lhs, rhs), node)
        self.available(catalog.BY_NAME[row], node)
        return Intrinsic(dtype, row, (lhs, rhs), span=self.module.span(node))

    def raw_intrinsic(self, name: str, node: ast.Call) -> Intrinsic | Packed:
        """A `krn.intrinsics` call is exactly the NVIDIA instruction; any other target refuses, never emulates."""
        if self.target.vendor != "nvidia":
            raise TargetError(
                f"krn.intrinsics.{name}",
                self.target.arch,
                f"`{name}` is one NVIDIA SIMD instruction and {self.target.arch} has no such instruction",
                f"use krn.math for the general function, which computes the same lanes on {self.target.arch}",
                span=self.module.span(node),
            )
        return self.family(catalog.FUNCTIONS[f"simd.{name}"], node)

    def call_pack(self, node: ast.Call) -> Expr:
        """`pack(a, b, c, d)` builds a word of four 8-bit lanes and `pack(a, b)` one of two 16-bit lanes."""
        lanes = [self.expr(argument) for argument in node.args]
        if node.keywords or not lanes or not lanes[0].dtype.is_int or lanes[0].dtype.bits not in (8, 16):
            raise self.error(node, "pack takes 8-bit lanes (four) or 16-bit lanes (two) of one integer type")
        lane = lanes[0].dtype
        if len(lanes) != 32 // lane.bits:
            raise self.error(node, f"pack takes {32 // lane.bits} {lane} lanes, got {len(lanes)}")
        span = self.module.span(node)
        word: Expr | None = None
        for index, value in enumerate(lanes):
            part = self.binary(
                Op.AND,
                self.convert(U32, self.coerce(value, lane, node), node),
                Const(U32, (1 << lane.bits) - 1, span=span),
                node,
            )
            if index:
                part = self.binary(Op.SHL, part, Const(U32, index * lane.bits, span=span), node)
            word = part if word is None else self.binary(Op.OR, word, part, node)
        assert word is not None
        return self.convert(packed(lane), word, node)

    def call_unpack(self, node: ast.Call) -> Packed:
        return self.unpacked(self.expr(self.arguments(node, ("x",))["x"]), node)

    def unpacked(self, word: Expr, node: ast.AST) -> Packed:
        """The lanes of a packed word, low lane first, each as its lane type."""
        if word.dtype.kind is not Kind.PACKED:
            raise self.error(node, f"unpack needs a packed word such as u8x4, got {word.dtype}")
        lane = word.dtype.lane
        assert lane is not None
        span = self.module.span(node)
        bits = self.convert(U32, word, node)
        lanes: list[Expr] = []
        for index in range(word.dtype.lanes):
            part = bits if index == 0 else self.binary(Op.SHR, bits, Const(U32, index * lane.bits, span=span), node)
            masked = self.binary(Op.AND, part, Const(U32, (1 << lane.bits) - 1, span=span), node)
            lanes.append(self.convert(lane, masked, node))
        return Packed(tuple(lanes))

    def call_static_assert(self, node: ast.Call) -> None:
        message = f"static_assert failed: {ast.literal_eval(node.args[1])}; target warp size {self.target.warp_size}"
        try:
            condition = self.const_value(node.args[0], "a static_assert condition")
        except CaptureError as error:
            raise self.error(node, f"{error.message}; {message}") from None
        if not condition:
            raise self.error(
                node,
                message,
            )

    def call_invoke_one(self, node: ast.Call) -> None:
        if len(node.args) < 2 or node.keywords:
            raise self.error(node, "invoke_one(group, fn, *args) needs a group and a device function")
        group, function = self.binding(node.args[0]), self.binding(node.args[1])
        if not isinstance(group, StructVal) or not isinstance(function, FunctionVal):
            raise self.error(node, "invoke_one needs a group and a device function")
        group_name, fn_name = self.fresh("_group"), self.fresh("_function")
        self.env[group_name], self.env[fn_name] = group, function
        self.synthetic(
            [
                ast.If(
                    ast.Compare(
                        ast.Call(
                            ast.Attribute(ast.Name(group_name, ast.Load()), "thread_rank", ast.Load()),
                            [],
                            [],
                        ),
                        [ast.Eq()],
                        [ast.Constant(0)],
                    ),
                    [ast.Expr(ast.Call(ast.Name(fn_name, ast.Load()), node.args[2:], []))],
                    [],
                )
            ],
            node,
        )

    def call_static_warn(self, node: ast.Call) -> None:
        if not self.const_value(node.args[0], "a static_warn condition"):
            log.warning("%s: %s", self.module.span(node), ast.literal_eval(node.args[1]))

    def call_shared_array(self, node: ast.Call) -> TensorVal:
        size = int(self.const_value(node.args[1], "a shared array size"))
        ref = TensorRef((f"shared{len(self.shared)}",), TensorType(self.dtype_arg(node.args[0]), 1), size)
        self.shared.append(ref)
        return TensorVal(ref)

    def call_type(self, node: ast.Call) -> TypeVal:
        args = self.arguments(node, ("value",))
        return TypeVal(self.expr(args["value"]).dtype)

    def call_int(self, node: ast.Call) -> Expr:
        args = self.arguments(node, ("value",))
        return self.convert(PY_INT, self.expr(args["value"]), node)

    def method_to(self, base: Expr, node: ast.Call) -> Expr:
        """`x.to(dtype)` converts every element; a kernel takes no device or copy argument."""
        if node.keywords or len(node.args) != 1:
            raise self.error(node, "krn's `.to` takes a dtype; devices are chosen with the launch")
        return self.convert(self.dtype_arg(node.args[0]), base, node)

    def method_tile(self, base: Expr, node: ast.Call) -> SliceExpr:
        """`id.tile(n)` is the slice [id * n, id * n + n)."""
        return self.tile_slice(base, int(self.const_value(node.args[0], "a tile size")), node)

    def tile_slice(self, index: Expr, size: int, node: ast.AST) -> SliceExpr:
        if size <= 0:
            raise self.error(node, f"a tile size must be positive, got {size}")
        span = self.module.span(node)
        start = self.binary(Op.MUL, Cast(PY_INT, index, span=span), Const(PY_INT, size, span=span), node)
        return SliceExpr(SliceType(size), start, span=span)

    def block_tile(self, node: ast.Call) -> SliceExpr | Packed:
        """`block.tile(bn)` is the tile along x, `block.tile(bm, bn)` the rows along y and the columns along x."""
        sizes = [int(self.const_value(a, "a tile size")) for a in node.args]
        if node.keywords or not 1 <= len(sizes) <= 2:
            raise self.error(
                node,
                "`block.tile` takes one size (along x) or two (rows along y, columns along x)",
            )
        axes = "yx"[-len(sizes) :]
        tiles = tuple(
            self.tile_slice(self.intrinsic(f"block.index.{axis}", [], node), size, node)
            for axis, size in zip(axes, sizes, strict=True)
        )
        return tiles[0] if len(tiles) == 1 else Packed(tiles)

    def fluent(self, name: str, base: Expr, node: ast.Call) -> Expr:
        """A fluent tile operation, `x.exp()`, or its in-place form `x.exp_()` that updates the tile variable `x`."""
        pure = name.removesuffix("_")
        if pure != name and pure in (*REDUCTIONS, *NORMALIZING):
            raise self.error(node, f"`{pure}` has no in-place form")
        value = self.operation(pure, base, node)
        if pure == name:
            return value
        target = node.func.value if isinstance(node.func, ast.Attribute) else node.func
        if not (names_a_variable(target) and isinstance(base, Var) and isinstance(base.type, TileType)):
            raise self.error(
                node,
                f"`{name}` updates a tile variable in place, but `{ast.unparse(target)}` is not one; use `{pure}`",
            )
        if value.type != base.type:
            raise self.error(
                node,
                f"`{name}` would make tile `{ast.unparse(target)}` {value.type}, it is {base.type}",
            )
        self.out.append(TileAssign(base, value, span=self.module.span(node)))
        return base

    def operation(self, name: str, base: Expr, node: ast.Call) -> Expr:
        if name in ARITHMETIC:
            return self.binary(ARITHMETIC[name], base, self.expr(self.arguments(node, ("other",))["other"]), node)
        if name in BINARY:
            return self.math(BINARY[name], [base, self.expr(self.arguments(node, ("other",))["other"])], node)
        if name == "fma":
            args = self.arguments(node, ("b", "c"))
            return self.math(Fn.FMA, [base, self.expr(args["b"]), self.expr(args["c"])], node)
        if name == "clip":
            args = self.arguments(node, ("low", "high"))
            return self.math(Fn.CLIP, [base, self.expr(args["low"]), self.expr(args["high"])], node)
        if name in REDUCTIONS or name in NORMALIZING:
            return self.reduction(name, base, node)
        return self.math(Fn(name), [base], node)

    def math(self, fn: Fn, operands: Sequence[Expr], node: ast.AST) -> Expr:
        """An elementwise function over floats: operands broadcast, and a literal adopts the type of the others."""
        dtype = element_dtype(operands[0].type)
        for operand in operands[1:]:
            common = self.arith.common(dtype, element_dtype(operand.type))
            if common is None:
                got = ", ".join(str(element_dtype(o.type)) for o in operands)
                raise self.error(node, f"dtype mismatch in `{fn}`: {got}; convert one side with .to(...)")
            dtype = common
        if dtype not in FLOATS:
            raise self.error(node, f"{fn} needs f32 or f64, got {dtype}; convert with .to(T.f32)")
        shape, kind = None, operands[0].type
        for operand in operands[1:]:
            shape = self.broadcast(kind, operand.type, node)
            kind = TileType(dtype, shape) if shape is not None else dtype
        if isinstance(operands[0].type, TileType) and shape is None:
            shape = operands[0].type.shape
        self.check_broadcast_effects(operands, shape, node)
        args = tuple(self.coerce(operand, dtype, node) for operand in operands)
        return Math(
            TileType(dtype, shape) if shape is not None else dtype,
            fn,
            args,
            span=self.module.span(node),
        )

    def reduction(self, name: str, base: Expr, node: ast.Call) -> Expr:
        """`x.sum(axis=1, keepdims=True)`, `x.softmax(axis=1)` and the like; a negative axis counts from the end."""
        args = self.arguments(node, (), ("axis",) if name in NORMALIZING else ("axis", "keepdims"))
        axis = int(self.const_value(args["axis"], "axis")) if "axis" in args else -1
        keepdims = bool(self.const_value(args["keepdims"], "keepdims")) if "keepdims" in args else False
        if not isinstance(base.type, TileType):
            raise self.error(node, f".{name}() reduces a tile, got {base.type}")
        rank = len(base.type.shape)
        if not -rank <= axis < rank:
            raise self.error(node, f"axis {axis} is out of range for a rank-{rank} tile")
        axis %= rank
        if name == "mean" and element_dtype(base.type).kind is not Kind.FLOAT:
            raise self.error(node, f"mean needs a float tile, got {element_dtype(base.type)}")

        def total(x: Expr, keep: bool) -> Expr:
            return self.reduce(Reduction.SUM, x, axis, keep, node)

        def shifted() -> Expr:
            """e^(x - max), the stable exponent shared by softmax and logsumexp."""
            top = self.reduce(Reduction.MAX, base, axis, True, node)
            return self.math(Fn.EXP, [self.binary(Op.SUB, base, top, node)], node)

        match name:
            case "norm":
                return self.math(Fn.SQRT, [total(self.binary(Op.MUL, base, base, node), keepdims)], node)
            case "softmax":
                return self.binary(Op.DIV, shifted(), total(shifted(), True), node)
            case "log_softmax":
                top = self.reduce(Reduction.MAX, base, axis, True, node)
                logsum = self.math(Fn.LOG, [total(shifted(), True)], node)
                return self.binary(Op.SUB, base, self.binary(Op.ADD, logsum, top, node), node)
            case "logsumexp":
                top = self.reduce(Reduction.MAX, base, axis, keepdims, node)
                return self.binary(Op.ADD, self.math(Fn.LOG, [total(shifted(), keepdims)], node), top, node)
        return self.reduce(Reduction[name.upper()], base, axis, keepdims, node)

    def reduce(self, op: Reduction, base: Expr, axis: int, keepdims: bool, node: ast.AST) -> Reduce:
        shape = base.shape
        out = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
        if not out:
            raise self.error(
                node,
                "reducing a rank-1 tile to a scalar is not supported in this slice; pass keepdims=True",
            )
        return Reduce(TileType(base.dtype, out), op, base, axis, keepdims, span=self.module.span(node))

    def stmt_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant):
            return
        if isinstance(node.value, ast.Yield):
            self.emit_yield(node.value)
            return
        if isinstance(node.value, ast.Call) and dotted(node.value.func) == "block.sync":
            self.arguments(node.value, ())
            self.barrier = self.barrier or (self.module.span(node), tuple(self.call_sites))
            self.out.append(Sync(span=self.module.span(node)))
            return
        value = self.binding(node.value)
        if isinstance(value, Intrinsic | Atomic):
            self.out.append(Effect(value, span=self.module.span(node)))

    def stmt_Assert(self, node: ast.Assert) -> None:
        span = self.module.span(node)
        text = node.msg if isinstance(node.msg, ast.Constant) and isinstance(node.msg.value, str) else None
        message = text.value if text else ast.unparse(node.msg or node.test)
        self.out.append(Assert(self.condition(node.test), f"{span}: assertion failed: {message}", span=span))

    def stmt_Pass(self, node: ast.Pass) -> None:
        pass

    def stmt_Return(self, node: ast.Return) -> None:
        if node.value is None:
            raise self.error(node, "a kernel or inlined function must return a value or nothing at its end")
        self.returned = self.binding(node.value)

    def stmt_Assign(self, node: ast.Assign) -> None:
        self.assign(node.targets[0], node.value, node)

    def stmt_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            raise self.error(node, "an annotated variable needs a value")
        try:
            declared = self.module.resolve(node.annotation, self.types)
        except CaptureError:
            declared = None  # an annotation that is not a number type says nothing here
        if isinstance(node.target, ast.Name) and isinstance(declared, Dtype) and declared not in (PY_INT, PY_FLOAT):
            value = self.exact(self.expr(node.value), declared, node, f"variable `{node.target.id}`")
            self.assign_name(node.target.id, value, node, self.module.span(node))
            return
        self.assign(node.target, node.value, node)

    def assign(self, target: ast.expr, value: ast.expr, node: ast.stmt) -> None:
        span = self.module.span(node)
        match target:
            case ast.Name(id=name):
                self.assign_name(name, self.binding(value), node, span)
            case ast.Tuple(elts=names) | ast.List(elts=names):
                packed = self.binding(value)
                if isinstance(packed, Seq):
                    self.unpack(target, packed, node)
                    return
                if not (isinstance(packed, Packed) and len(packed.items) == len(names)):
                    raise self.error(target, f"cannot unpack `{ast.unparse(value)}` into {len(names)} names")
                for one, item in zip(names, packed.items, strict=True):
                    if not isinstance(one, ast.Name):
                        raise self.error(one, f"cannot assign to `{ast.unparse(one)}`")
                    self.assign_name(one.id, item, node, span)
            case ast.Subscript():
                if self.sequence_candidate(target.value):
                    raise self.error(target, FIXED_SEQUENCE)
                self.store(target, self.expr(value), node)
            case _:
                raise self.error(target, f"cannot assign to `{ast.unparse(target)}`")

    def unpack(self, target: ast.expr, value: Binding, node: ast.AST, *, unrolled: bool = False) -> None:
        if isinstance(target, ast.Name):
            if target.id == "_":
                return
            if unrolled and isinstance(value, Const):
                self.env[target.id] = value
                self.weak.pop(target.id, None)
            else:
                self.assign_name(target.id, value, node, self.module.span(node))
            return
        if not isinstance(target, ast.Tuple | ast.List) or not isinstance(value, Seq):
            raise self.error(
                target,
                f"cannot unpack {WHAT.get(type(value), 'a value')} into `{ast.unparse(target)}`",
            )
        stars = [i for i, element in enumerate(target.elts) if isinstance(element, ast.Starred)]
        size, wanted = len(value.items), len(target.elts)
        if len(stars) > 1 or (not stars and size != wanted) or (stars and size < wanted - 1):
            raise self.error(target, f"cannot unpack sequence of length {size} into target of length {wanted}")
        values = list(value.items)
        if stars:
            index = stars[0]
            stop = size - (wanted - index - 1)
            values[index:stop] = [Seq(tuple(values[index:stop]), "list")]
        for element, item in zip(target.elts, values, strict=True):
            self.unpack(
                element.value if isinstance(element, ast.Starred) else element,
                item,
                node,
                unrolled=unrolled,
            )

    def stmt_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if self.sequence_candidate(target):
                raise self.error(target, FIXED_SEQUENCE)
        raise self.error(node, "deletion is not supported in a kernel")

    def assign_name(self, name: str, value: Binding | None, node: ast.AST, span: Span) -> None:
        existing = self.env.get(name)
        match value:
            case None:
                raise self.error(node, f"`{name}` is assigned a call that returns nothing")
            case (
                TensorVal()
                | StructVal()
                | FunctionVal()
                | TypeVal()
                | Auto()
                | Seq()
                | TargetVal()
                | Expr(type=SliceType())
            ):
                self.env[name] = value
            case Packed():
                raise self.error(node, f"`{name}` cannot hold several values; unpack them: `a, b = ...`")
            case Const() if self.assigned[name] == 1 and not isinstance(existing, Var):
                self.env[name] = value
            case Expr(type=TileType() as tile):
                self.tiled = True
                if isinstance(existing, Var) and existing.type != tile:
                    raise self.error(
                        node,
                        f"tile `{name}` is {existing.dtype} {existing.shape}, cannot become {tile.dtype} {tile.shape}",
                    )
                var = existing if isinstance(existing, Var) else self.declare(name, tile, node)
                self.out.append(TileAssign(var, value, span=span))
            case Expr():
                if isinstance(existing, Var) and isinstance(existing.type, Dtype):
                    existing = self.retype(name, existing, value, node) or existing
                    var, value = (
                        existing,
                        self.exact(value, existing.type, node, f"variable `{name}`"),
                    )
                else:
                    var = self.declare(name, value.type, node)
                    if isinstance(value, Const) and element_dtype(value.type).is_py:
                        self.weak[name] = Weak([])
                self.out.append(Assign(var, value, span=span))
                if name in self.weak:
                    if isinstance(value, Const):
                        self.weak[name].sites.append((self.out, len(self.out) - 1))
                    else:
                        del self.weak[name]  # a computed value: this local keeps the type it has

    def retype(self, name: str, current: Var, value: Expr, node: ast.AST) -> Var | None:
        """A local set from literals only meets its first typed value: give it that type, and rewrite where it was set.

        `total = 0.0` and then `total += x[i] * y[i]` is a float32 accumulator without saying so."""
        weak, dtype = self.weak.get(name), element_dtype(value.type)
        if weak is None or dtype.is_py or isinstance(value.type, TileType):
            return None
        if self.arith.common(element_dtype(current.type), dtype) != dtype:
            return None
        if weak.read or any(out is done for out, _ in weak.sites for done in self.closed):
            used = f"`{name}` was used as a plain {current.type} before it met {dtype}"
            raise self.error(node, f"{used}; give it a type: `{name}: {dtype} = ...`")
        typed = Var(dtype, current.name, span=current.span)
        for out, index in weak.sites:
            old = out[index]
            assert isinstance(old, Assign)
            out[index] = Assign(typed, self.coerce(old.value, dtype, node), span=old.span)
        self.locals[current.name] = typed
        self.env[name] = typed
        del self.weak[name]
        return typed

    def store(self, target: ast.Subscript, value: Expr, node: ast.AST) -> None:
        ref = self.tensor(target.value)
        items = self.indices(target.slice, ref, target)
        name, span, dtype = ast.unparse(target.value), self.module.span(node), ref.type.dtype
        self.written.add(ref.path)
        if any(isinstance(i, SliceExpr) for i in items):
            slices = self.squeezed(items)
            if not isinstance(value.type, TileType):
                raise self.error(node, f"cannot store a scalar {value.type} into a tile slice of `{name}`")
            if value.type != TileType(dtype, tile_shape(slices)):
                want = TileType(dtype, tile_shape(slices))
                raise self.error(
                    node,
                    f"cannot store {value.dtype} tile {value.shape} into {dtype} slice {want.shape} of `{name}`",
                )
            self.tiled = True
            self.out.append(TileStore(ref, slices, value, span=span))
            return
        if isinstance(value.type, TileType | SliceType):
            raise self.error(node, f"cannot store a tile into a single element of `{name}`")
        if element_dtype(value.type) != dtype and not (
            element_dtype(value.type).is_py and self.arith.common(element_dtype(value.type), dtype) == dtype
        ):
            raise self.error(node, f"cannot store {value.type} into {dtype} tensor `{name}`")
        self.out.append(Store(ref, tuple(items), self.coerce(value, dtype, node), span=span))

    def stmt_AugAssign(self, node: ast.AugAssign) -> None:
        span = self.module.span(node)
        if isinstance(node.target, ast.Subscript) and self.sequence_candidate(node.target.value):
            raise self.error(node.target, FIXED_SEQUENCE)
        if isinstance(node.op, ast.MatMult):
            raise self.error(node, "`@=` is not supported; write `acc += a @ b`")
        if (
            isinstance(node.op, ast.Add)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.MatMult)
        ):
            product = node.value
            acc = self.expr(node.target)
            self.accumulate(node, acc, self.expr(product.left), self.expr(product.right), product)
            return
        if isinstance(node.target, ast.Name):
            self.augmenting = node.target.id
        try:
            current = self.expr(node.target)
        finally:
            self.augmenting = None
        new = self.operate(type(node.op), current, self.expr(node.value), node)
        if isinstance(node.target, ast.Subscript):
            self.store(node.target, new, node)
        elif isinstance(node.target, ast.Name):
            self.assign_name(node.target.id, new, node, span)

    def product(self, node: ast.BinOp) -> Var:
        """`a @ b` outside `acc += a @ b`: the product in a fresh float32 tile, so `s = q @ kt` reads as algebra."""
        lhs, rhs = self.expr(node.left), self.expr(node.right)
        if not all(isinstance(o.type, TileType) and len(o.type.shape) == 2 for o in (lhs, rhs)):
            raise self.error(node, f"`@` needs two rank-2 tiles, got {lhs.type} and {rhs.type}")
        span = self.module.span(node)
        shape = (lhs.shape[0], rhs.shape[1])
        acc = Var(TileType(F32, shape), self.fresh("product"), span=span)
        self.tiles[acc.name], self.tiled = acc, True
        self.out.append(TileAssign(acc, Zeros(TileType(F32, shape), span=span), span=span))
        self.accumulate(node, acc, lhs, rhs, node)
        return acc

    def accumulate(self, node: ast.AST, acc: Expr, lhs: Expr, rhs: Expr, product: ast.BinOp) -> None:
        """Keep fp8 loads raw; widen mixed f16/bf16/f32 operands losslessly before instruction selection."""
        if not (isinstance(acc, Var) and isinstance(acc.type, TileType)):
            raise self.error(node, "the target of `+= a @ b` must be a tile variable")
        if not (
            isinstance(lhs.type, TileType)
            and isinstance(rhs.type, TileType)
            and len(lhs.type.shape) == len(rhs.type.shape) == 2
        ):
            raise self.error(product, f"`@` needs two rank-2 tiles, got {lhs.type} and {rhs.type}")
        (m, k), (k2, n) = lhs.type.shape, rhs.type.shape
        operands = (lhs.type.dtype, rhs.type.dtype)
        fp8 = all(d in FP8 for d in operands)
        if narrow_operand := next((d for d in operands if is_narrow(d) and d not in FP8), None):
            refuse_product(narrow_operand, self.target, self.module.span(product))
        if fp8 and not (isinstance(lhs, TileLoad) and isinstance(rhs, TileLoad)):
            raise TargetError(
                "fp8 matrix product",
                self.target.arch,
                "fp8 matrix operands must be direct TileLoad operands; no fp8 expression decode exists",
                "load fp8 tiles directly or use f16 or bf16 operands",
                span=self.module.span(product),
            )
        if not fp8 and (
            any(d.kind is not Kind.FLOAT or d.bits not in (16, 32) for d in operands)
            or not all(widens_to(to_sketch(d), to_sketch(F32)) for d in operands)
        ):
            raise self.error(
                product,
                f"`@` f32 accumulator needs lossless operands (f16, bf16 or f32), got "
                f"{operands[0]} and {operands[1]}; fp8 and block-scaled formats keep their own rule (QTile / mma)",
            )
        if k != k2:
            raise self.error(product, f"inner dimensions of `@` differ: {lhs.type.shape} @ {rhs.type.shape}")
        if acc.type != TileType(F32, (m, n)):
            raise self.error(
                node,
                f"the accumulator is {acc.type.dtype} {acc.type.shape}, the product is f32 ({m}, {n})",
            )
        if not fp8 and operands[0] != operands[1]:
            lhs, rhs = self.convert(F32, lhs, product.left), self.convert(F32, rhs, product.right)
        self.out.append(Accumulate(acc, lhs, rhs, span=self.module.span(node)))

    def stmt_If(self, node: ast.If) -> None:
        cond = self.condition(node.test)
        if isinstance(cond, Const):
            for statement in node.body if cond.value else node.orelse:
                self.stmt(statement)
            return
        self.out.append(If(cond, self.block(node.body), self.block(node.orelse), span=self.module.span(node)))

    def stmt_Match(self, node: ast.Match) -> None:
        subject = (
            TargetVal(self.target)
            if isinstance(node.subject, ast.Name) and node.subject.id == "target"
            else self.binding(node.subject)
        )
        if not isinstance(subject, Const | TypeVal | Auto | TargetVal):
            raise self.error(
                node.subject,
                "match needs a capture-time subject "
                "(a target fact, a dtype, a `Const` field or a literal), "
                f"got the runtime value `{ast.unparse(node.subject)}`; use `if` for a runtime value",
            )
        for case in node.cases:
            captures: dict[str, Binding] = {}
            if not self.match_pattern(case.pattern, subject, captures):
                continue
            saved = {name: self.env.get(name) for name in captures}
            self.env.update(captures)
            if case.guard is not None:
                guard = self.condition(case.guard)
                if not isinstance(guard, Const):
                    raise self.capture_constant(case.guard, "a match guard")
                if not guard.value:
                    for name, value in saved.items():
                        if value is None:
                            self.env.pop(name, None)
                        else:
                            self.env[name] = value
                    continue
            for statement in case.body:
                self.stmt(statement)
            return

    def match_pattern(self, pattern: ast.pattern, subject: Binding, captures: dict[str, Binding]) -> bool:
        match pattern:
            case ast.MatchValue(value=value):
                expected = self.binding(value)
                if isinstance(subject, Const) and isinstance(expected, Const):
                    return subject.value == expected.value
                if isinstance(subject, TypeVal) and isinstance(expected, TypeVal):
                    return subject.dtype == expected.dtype
                return isinstance(subject, Auto) and isinstance(expected, Auto)
            case ast.MatchSingleton(value=value):
                return isinstance(subject, Const) and subject.value is value
            case ast.MatchAs(pattern=inner, name=name):
                if inner is not None and not self.match_pattern(inner, subject, captures):
                    return False
                if name is not None:
                    captures[name] = subject
                return True
            case ast.MatchOr(patterns=patterns):
                for alternative in patterns:
                    candidate: dict[str, Binding] = {}
                    if self.match_pattern(alternative, subject, candidate):
                        captures.update(candidate)
                        return True
                return False
            case ast.MatchClass(cls=cls, patterns=[], kwd_attrs=[], kwd_patterns=[]):
                family = self.module.evaluate(cls)
                if isinstance(subject, TargetVal) and isinstance(family, type) and issubclass(family, Target):
                    return isinstance(subject.target, family)
        raise self.error(pattern, f"unsupported pattern `{ast.unparse(pattern)}` in a capture-time match")

    def stmt_While(self, node: ast.While) -> None:
        """The test runs on every pass. A call in it expands into statements, which would run once before the loop, so
        they run again at the end of each pass and leave their result in a flag that the loop tests."""
        span, before = self.module.span(node), len(self.out)
        test = self.condition(node.test)
        if len(self.out) == before:
            self.out.append(While(test, self.block(node.body), span=span))
            return
        flag = self.declare(self.fresh("while_test"), BOOL, node)
        self.out.append(Assign(flag, test, span=span))
        saved, self.out = self.out, []
        for statement in node.body:
            self.stmt(statement)
        self.out.append(Assign(flag, self.condition(node.test), span=span))
        body, self.out = self.out, saved
        self.closed.append(body)
        self.out.append(While(flag, tuple(body), span=span))

    def emit_yield(self, node: ast.Yield) -> None:
        """At a `yield` of an inlined generator: bind the loop's names to the values and run the loop's body here."""
        consumer = self.consumer
        if consumer is None or node.value is None:
            raise self.error(
                node,
                "`yield` needs a value, and a `for` loop or `sum(...)` that iterates the generator",
            )
        parts = node.value.elts if isinstance(node.value, ast.Tuple) else None
        value = Seq(tuple(self.element(part) for part in parts), "tuple") if parts else self.binding(node.value)
        if value is None:
            raise self.error(node, "a generator must yield a value")
        saved = (self.env, self.module, self.consumer, self.types)
        self.env, self.module, self.consumer, self.types = (
            consumer.env,
            consumer.module,
            consumer.outer,
            consumer.types,
        )
        try:
            self.unpack(consumer.target, value, consumer.target)
            for statement in consumer.body:
                self.stmt(statement)
        finally:
            self.env, self.module, self.consumer, self.types = saved

    def generator_of(self, call: ast.expr) -> tuple[Module, ast.FunctionDef, StructVal | None] | None:
        """The generator function (or struct method) that `call` calls, if it is one."""
        if not isinstance(call, ast.Call):
            return None
        func = call.func
        if isinstance(func, ast.Name) and func.id not in self.env and (found := self.module.lookup(func.id)):
            module, fn = found
            if isinstance(fn, ast.FunctionDef) and any(isinstance(n, ast.Yield) for n in ast.walk(fn)):
                return module, fn, None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in self.env:
            owner = self.env[func.value.id]
            if isinstance(owner, StructVal) and func.attr in owner.definition.methods:
                method = owner.definition.methods[func.attr]
                if any(isinstance(n, ast.Yield) for n in ast.walk(method)):
                    return owner.definition.module, method, owner
        return None

    def items(self, node: ast.For, call: ast.Call) -> None:
        """`for i in items(n)`: the items of this thread, a grid-stride loop whose index is typed like `n`."""
        if len(call.args) != 1 or call.keywords or node.orelse or not isinstance(node.target, ast.Name):
            raise self.error(node, "a loop over the items of a thread is `for i in items(n):`")
        count, span = self.expr(call.args[0]), self.module.span(node)
        self.item_counts.append(count)
        dtype = element_dtype(count.type)
        dtype = U32 if dtype.is_py else dtype
        if not dtype.is_int:
            raise self.error(call, f"`items(n)` needs an integer count, got {count.type}")

        def geometry(path: str) -> Expr:
            return self.convert(dtype, self.intrinsic(path, [], call), call)

        stride = geometry("grid.num_threads")
        var = self.declare(node.target.id, dtype, node)
        self.out.append(Assign(var, geometry("grid.thread_rank"), span=span))
        advance = Assign(var, Binary(dtype, Op.ADD, var, stride, span=span), span=span)
        condition = Binary(BOOL, Op.LT, var, self.coerce(count, dtype, node), span=span)
        self.out.append(While(condition, (*self.block(node.body), advance), span=span))

    def stmt_For(self, node: ast.For) -> None:
        loop = node.iter
        if isinstance(loop, ast.Call) and getattr(self.module.sketch(loop.func), "__name__", None) == "pipelined":
            self.pipelined(node, loop)
            return
        if isinstance(loop, ast.Call) and getattr(self.module.sketch(loop.func), "__name__", None) == "items":
            self.items(node, loop)
            return
        if node.orelse:
            raise self.error(node, "for ... else is not supported in a kernel")
        if self.sequence_candidate(loop):
            for item in self.compile_sequence(loop).items:
                self.unpack(node.target, item, node.target, unrolled=True)
                for statement in node.body:
                    self.stmt(statement)
            return
        if (generator := self.generator_of(loop)) is not None and not node.orelse:
            module, fn, owner = generator
            assert isinstance(loop, ast.Call)
            self.inline(
                module,
                fn,
                loop,
                owner,
                Consumer(self.env, self.module, self.types, node.target, node.body, self.consumer),
            )
            return
        if isinstance(loop, ast.Call):
            name = dotted(loop.func)
            if name == "enumerate":
                args = self.arguments(loop, ("iterable",), ("start",))
                if not isinstance(node.target, ast.Tuple) or len(node.target.elts) != 2:
                    raise self.error(
                        node.target,
                        "a runtime enumerate loop needs a two-tuple target `(index, value)`",
                    )
                index, inner = node.target.elts
                counter = self.fresh("_enumerate")
                self.assigned[counter] = 2
                self.synthetic(
                    [
                        ast.Assign([ast.Name(counter, ast.Store())], args.get("start", ast.Constant(0))),
                        ast.For(
                            inner,
                            args["iterable"],
                            [
                                ast.Assign([index], ast.Name(counter, ast.Load())),
                                *node.body,
                                ast.AugAssign(ast.Name(counter, ast.Store()), ast.Add(), ast.Constant(1)),
                            ],
                            [],
                        ),
                    ],
                    node,
                )
                return
            if name in ("reversed", "zip"):
                if name == "reversed":
                    args = self.arguments(loop, ("sequence",))
                    self.stride_loop(node, [self.reverse_stride(self.iteration_stride(args["sequence"]), node)])
                else:
                    self.zip_keywords(loop)
                    if any(self.generator_of(a) is not None for a in loop.args):
                        raise self.error(
                            loop,
                            "`zip` cannot step a generator method in lockstep "
                            "(its body is inlined where it yields): use `enumerate(generator)`, "
                            "or zip ranges and register sequences",
                        )
                    if any(self.sequence_candidate(a) for a in loop.args):
                        raise self.error(
                            loop,
                            "`zip` mixing a runtime range with a register sequence cannot "
                            "index the sequence: use `enumerate` over the sequence",
                        )
                    self.stride_loop(node, [self.iteration_stride(a) for a in loop.args], zipped=True)
                return
        if (
            not (isinstance(loop, ast.Call) and dotted(loop.func) == "range" and isinstance(node.target, ast.Name))
            or node.orelse
        ):
            raise self.error(
                node,
                "only `for name in range(...)` and `for tile in pipelined(...)` loops are supported",
            )
        lo, hi, step = self.range_bounds(loop)
        if step < 0:
            self.stride_loop(node, [self.range_stride(loop, (lo, hi, step))])
            return
        span = self.module.span(node)
        dtype = lo.dtype
        var = self.declare(node.target.id, dtype, node, hoist=False)
        self.out.append(
            For(
                var,
                self.coerce(lo, dtype, node),
                self.coerce(hi, dtype, node),
                step,
                self.block(node.body),
                span=span,
            )
        )

    def range_bounds(self, node: ast.Call) -> tuple[Expr, Expr, int]:
        if not 1 <= len(node.args) <= 3 or node.keywords:
            raise self.error(node, "`range` takes one to three positional integer arguments")
        values = [self.expr(a) for a in node.args]
        span = self.module.span(node)
        lo, hi = (Const(PY_INT, 0, span=span), values[0]) if len(values) == 1 else (values[0], values[1])
        step = values[2] if len(values) == 3 else Const(PY_INT, 1, span=span)
        if not isinstance(step, Const):
            raise self.capture_constant(node.args[2], "the step of a loop")
        if not isinstance(step.value, int):
            raise self.error(node, "the step of a loop must be an integer constant")
        if step.value == 0:
            raise self.error(node, "the step of a loop must not be zero")
        dtype = self.arith.common(lo.dtype, hi.dtype)
        if dtype is None or not (dtype.is_int or dtype == PY_INT):
            raise self.error(node, f"loop bounds have types {lo.type} and {hi.type}; convert one side")
        return self.coerce(lo, dtype, node), self.coerce(hi, dtype, node), step.value

    def hold(self, value: Expr, node: ast.AST, name: str) -> ast.Name:
        key = self.fresh(name)
        if isinstance(value, Const):
            self.env[key] = value
        else:
            self.assign_name(key, value, node, self.module.span(node))
        return ast.copy_location(ast.Name(key, ast.Load()), node)

    def range_stride(self, node: ast.Call, bounds: tuple[Expr, Expr, int] | None = None) -> Stride:
        lo, hi, step = bounds if bounds is not None else self.range_bounds(node)
        span, dtype = self.module.span(node), lo.dtype
        if isinstance(lo, Const) and isinstance(hi, Const):
            return Stride(lo, step, Const(dtype, len(range(int(lo.value), int(hi.value), step)), span=span))
        lo = self.expr(self.hold(lo, node, "_lo"))
        hi = self.expr(self.hold(hi, node, "_hi"))
        lower, upper = (lo, hi) if step > 0 else (hi, lo)
        condition = self.binary(Op.LT, lower, upper, node)
        count = self.binary(Op.SUB, upper, lower, node)
        if abs(step) != 1:
            count = self.binary(Op.ADD, count, Const(PY_INT, abs(step) - 1, span=span), node)
            count = self.binary(Op.DIV, count, Const(PY_INT, abs(step), span=span), node)
        count = Select(dtype, condition, count, Const(dtype, 0, span=span), span=span)
        return Stride(lo, step, self.expr(self.hold(count, node, "_range_count")))

    def iteration_stride(self, node: ast.expr) -> Stride:
        if isinstance(node, ast.Call) and dotted(node.func) == "range":
            return self.range_stride(node)
        if isinstance(node, ast.Call) and dotted(node.func) == "reversed":
            args = self.arguments(node, ("sequence",))
            return self.reverse_stride(self.iteration_stride(args["sequence"]), node)
        if self.generator_of(node) is not None:
            raise self.error(
                node,
                "`reversed` needs a range or a register sequence: a generator method yields in its own order",
            )
        raise self.error(node, "this runtime iterator needs a range or a register sequence")

    def reverse_stride(self, stride: Stride, node: ast.AST) -> Stride:
        count = self.expr(self.hold(stride.count, node, "_reverse_count"))
        return Stride(
            self.stride_offset(stride.first, count, stride.step, node, previous=True),
            -stride.step,
            count,
        )

    def stride_offset(self, first: Expr, index: Expr, step: int, node: ast.AST, *, previous: bool = False) -> Expr:
        if previous:
            index = self.binary(Op.SUB, index, Const(PY_INT, 1, span=self.module.span(node)), node)
        if abs(step) != 1:
            index = self.binary(Op.MUL, index, Const(PY_INT, abs(step), span=self.module.span(node)), node)
        return self.binary(Op.ADD if step > 0 else Op.SUB, first, index, node)

    def stride_loop(self, node: ast.For, strides: list[Stride], *, zipped: bool = False) -> None:
        if zipped and (not isinstance(node.target, ast.Tuple) or len(node.target.elts) != len(strides)):
            raise self.error(node.target, "a runtime zip loop needs a tuple target naming each range")
        if not strides:
            return
        dtype, count = strides[0].first.dtype, strides[0].count
        for stride in strides[1:]:
            common = self.arith.common(dtype, stride.first.dtype)
            if common is None or common != dtype or common != stride.first.dtype:
                raise self.error(
                    node,
                    f"loop bounds have types {dtype} and {stride.first.dtype}; convert one side",
                )
            comparison = self.binary(Op.LT, count, stride.count, node)
            count = Select(dtype, comparison, count, stride.count, span=self.module.span(node))
        stop = self.hold(count, node, "_count")
        starts = [self.hold(stride.first, node, "_first") for stride in strides]
        counter = self.fresh("_iteration")
        targets = node.target.elts if zipped and isinstance(node.target, ast.Tuple) else [node.target]
        body: list[ast.stmt] = []
        for target, start, stride in zip(targets, starts, strides, strict=True):
            for binding in ast.walk(target):
                if isinstance(binding, ast.Name):
                    existing = self.env.get(binding.id)
                    if isinstance(existing, Var) and existing.name not in self.locals:
                        del self.env[binding.id]
            offset: ast.expr = ast.Name(counter, ast.Load())
            if abs(stride.step) != 1:
                offset = ast.BinOp(offset, ast.Mult(), ast.Constant(abs(stride.step)))
            value = ast.BinOp(start, ast.Add() if stride.step > 0 else ast.Sub(), offset)
            body.append(ast.Assign([target], value))
        self.synthetic(
            [
                ast.For(
                    ast.Name(counter, ast.Store()),
                    ast.Call(ast.Name("range", ast.Load()), [ast.Constant(0), stop], []),
                    [*body, *node.body],
                    [],
                )
            ],
            node,
        )

    def pipelined(self, node: ast.For, call: ast.Call) -> None:
        """`for ks in pipelined(extent, tile)`: a loop over the tiles of an axis, `ks` being the slice of each.

        The loop steps by `tile` and `ks` is `start : start + tile`. The last tile may run past `extent`, and the slice
        reads and writes of a ragged tile follow the usual validity rules. Overlapping loads with the multiply is the
        job of a GPU target; explicit stages on a target without a pipeline fail during emission.
        """
        args = self.arguments(call, ("extent",), ("tile", "stages", "schedule"))
        if not isinstance(node.target, ast.Name) or node.orelse:
            raise self.error(node, "a `pipelined` loop binds one name and has no else branch")
        if isinstance(args["extent"], ast.Call) and dotted(args["extent"].func) == "range":
            raise self.error(
                args["extent"],
                "`pipelined(extent, tile)` takes the extent of the axis, not a `range`",
            )
        if "tile" not in args:
            raise self.error(call, "`pipelined(extent, tile)` is missing the tile size")
        extent, size, span = (
            self.expr(args["extent"]),
            int(self.const_value(args["tile"], "a tile size")),
            self.module.span(node),
        )
        dtype = self.arith.common(PY_INT, element_dtype(extent.type))
        if dtype is None or not (dtype.is_int or dtype == PY_INT) or size <= 0:
            raise self.error(
                call,
                f"`pipelined` needs an integer extent and a positive tile, got {extent.type} and {size}",
            )
        self.check_pipeline(args.get("stages"), args.get("schedule"))
        depth = self.binding(args["stages"]) if "stages" in args else None
        stages = int(depth.value) if isinstance(depth, Const) else 0
        start = Var(dtype, self.fresh(f"{node.target.id}_start"), span=span)
        self.env[node.target.id] = SliceExpr(SliceType(size), start, span=span)
        body = self.block(node.body)
        del self.env[node.target.id]
        self.out.append(
            For(
                start,
                self.coerce(Const(PY_INT, 0, span=span), dtype, node),
                self.coerce(extent, dtype, node),
                size,
                body,
                pipelined=True,
                stages=stages,
                span=span,
            )
        )

    def check_pipeline(self, stages: ast.expr | None, schedule: ast.expr | None) -> None:
        """`stages` is AUTO or a positive constant. A schedule the target cannot run is an error, not a hint."""
        if stages is not None:
            depth = self.binding(stages)
            positive = isinstance(depth, Const) and isinstance(depth.value, int) and depth.value > 0
            if not (isinstance(depth, Auto) or positive):
                raise self.error(
                    stages,
                    f"stages must be AUTO or a positive constant, got `{ast.unparse(stages)}`",
                )
        if schedule is None:
            return
        named = self.module.sketch(schedule.func if isinstance(schedule, ast.Call) else schedule)
        name = getattr(named, "__name__", None)
        if name not in ("AUTO", "Uniform", "WarpSpecialized", "PingPong"):
            raise self.error(
                schedule,
                f"schedule must be AUTO or a schedule such as Uniform(), got `{ast.unparse(schedule)}`",
            )
        if name == "WarpSpecialized" and not self.target.has(Feature.BULK_COPY):
            raise TargetError(
                "WarpSpecialized",
                self.target.arch,
                f"WarpSpecialized needs BULK_COPY, which target {self.target.name} lacks",
                "use schedule=AUTO or Uniform()",
                span=self.module.span(schedule),
            )
        if name in ("WarpSpecialized", "PingPong"):
            raise TargetError(
                name,
                self.target.arch,
                f"krn has no implemented {name} schedule lowering for this target",
                "use schedule=AUTO or Uniform()",
                span=self.module.span(schedule),
            )
