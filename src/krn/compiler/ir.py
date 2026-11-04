"""The typed IR: structured statements over expressions that carry a type and the user's source span.

Scalars are SIMT values. A tile is a value of fixed shape, block-cooperative, and every extent is a capture-time
constant, so an emitter never needs a shape at run time.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto

from krn.frontend.atomic import AtomicOp

from .dtypes import I64, PY_INT, Dtype
from .errors import Span


class Op(StrEnum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    AND = "&"
    OR = "|"
    XOR = "^"
    SHL = "<<"
    SHR = ">>"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    LAND = "&&"
    LOR = "||"
    NEG = "neg"
    NOT = "!"
    INVERT = "~"


class Fn(StrEnum):
    """The elementwise functions of the fluent tile operations, one per name of the table in `tools/gen_ops.py`."""

    EXP = auto()
    EXP2 = auto()
    EXP10 = auto()
    EXPM1 = auto()
    LOG = auto()
    LOG2 = auto()
    LOG10 = auto()
    LOG1P = auto()
    SQRT = auto()
    CBRT = auto()
    RSQRT = auto()
    RECIPROCAL = auto()
    SIN = auto()
    COS = auto()
    TANH = auto()
    SIGMOID = auto()
    ERF = auto()
    FLOOR = auto()
    CEIL = auto()
    ROUND = auto()
    TRUNC = auto()
    RELU = auto()
    GELU = auto()
    SILU = auto()
    SOFTPLUS = auto()
    ABS = auto()
    MAXIMUM = auto()
    MINIMUM = auto()
    POW = auto()
    CLIP = auto()
    FMA = auto()


class Reduction(StrEnum):
    MAX = auto()
    MIN = auto()
    SUM = auto()
    PROD = auto()
    MEAN = auto()  # the sum over the valid lanes divided by their number, NaN where there are none


@dataclass(frozen=True, slots=True)
class TileType:
    dtype: Dtype
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SliceType:
    extent: int


@dataclass(frozen=True, slots=True)
class TensorType:
    """A row-major view of global memory. Its shape travels with the pointer."""

    dtype: Dtype
    rank: int


@dataclass(frozen=True, slots=True)
class StructType:
    """The runtime fields of a struct. Fields typed `Const[...]` are folded at capture and are not here."""

    name: str
    fields: tuple[tuple[str, Dtype | TensorType], ...]


type ValueType = Dtype | TileType | SliceType
type ParamType = Dtype | TensorType | StructType


@dataclass(frozen=True, slots=True)
class TensorRef:
    path: tuple[str, ...]
    type: TensorType
    extent: int = 0  # elements of a block-shared array, 0 for global memory

    def dim(self, axis: int) -> tuple[str, ...]:
        return (*self.path[:-1], f"{self.path[-1]}_n{axis}")


@dataclass(frozen=True, slots=True)
class Node:
    span: Span = field(kw_only=True, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Expr(Node):
    type: ValueType

    @property
    def dtype(self) -> Dtype:
        """The element type: the type of a scalar, the element of a tile."""
        kind = self.type
        match kind:
            case TileType():
                return kind.dtype
            case Dtype():
                return kind
            case SliceType():
                return PY_INT

    @property
    def shape(self) -> tuple[int, ...]:
        return self.type.shape if isinstance(self.type, TileType) else ()


@dataclass(frozen=True, slots=True)
class Const(Expr):
    value: int | float | bool


@dataclass(frozen=True, slots=True)
class Var(Expr):
    """A scalar local, or a tile variable (its type says which)."""

    name: str


@dataclass(frozen=True, slots=True)
class Param(Expr):
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Dim(Expr):
    tensor: TensorRef
    axis: int


@dataclass(frozen=True, slots=True)
class Load(Expr):
    tensor: TensorRef
    index: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Intrinsic(Expr):
    name: str
    args: tuple[Expr, ...] = ()


@dataclass(frozen=True, slots=True)
class Atomic(Expr):
    """`atomic.<op>(tensor, index, value)`: the value of the element before the update. `compare` is the expected value
    of a compare-and-swap."""

    op: AtomicOp
    tensor: TensorRef
    index: tuple[Expr, ...]
    value: Expr
    compare: Expr | None = None


@dataclass(frozen=True, slots=True)
class Unary(Expr):
    op: Op
    operand: Expr


@dataclass(frozen=True, slots=True)
class Math(Expr):
    """An elementwise function of one to three operands, over floats unless the function says otherwise."""

    fn: Fn
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    op: Op
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Cast(Expr):
    operand: Expr


@dataclass(frozen=True, slots=True)
class Select(Expr):
    cond: Expr
    then: Expr
    orelse: Expr


@dataclass(frozen=True, slots=True)
class Transpose(Expr):
    """A matrix tile with its two axes swapped: element (i, j) of the result is element (j, i) of the operand."""

    operand: Expr


@dataclass(frozen=True, slots=True)
class Arange(Expr):
    """The integers along one axis of a tile: an element is `start` plus its index on `axis`, whatever the others."""

    axis: int
    start: Expr


@dataclass(frozen=True, slots=True)
class SliceExpr(Expr):
    """The half-open range [start, start + extent) of one axis. A squeezed axis is one scalar index: it addresses the
    tensor but is not an axis of the tile, so its extent is 1 and the tile type leaves it out."""

    start: Expr
    squeeze: bool = False

    @property
    def extent(self) -> int:
        return self.type.extent if isinstance(self.type, SliceType) else 0


@dataclass(frozen=True, slots=True)
class TileLoad(Expr):
    """A tile read. Elements outside the tensor read as zero and are invalid lanes (see `validity`)."""

    tensor: TensorRef
    slices: tuple[SliceExpr, ...]


@dataclass(frozen=True, slots=True)
class Zeros(Expr):
    pass


@dataclass(frozen=True, slots=True)
class Reduce(Expr):
    op: Reduction
    operand: Expr
    axis: int
    keepdims: bool


@dataclass(frozen=True, slots=True)
class Assign(Node):
    var: Var
    value: Expr


@dataclass(frozen=True, slots=True)
class Store(Node):
    tensor: TensorRef
    index: tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True, slots=True)
class TileAssign(Node):
    var: Var
    value: Expr


@dataclass(frozen=True, slots=True)
class TileStore(Node):
    """A tile write. Elements outside the tensor are skipped."""

    tensor: TensorRef
    slices: tuple[SliceExpr, ...]
    value: Expr


@dataclass(frozen=True, slots=True)
class Accumulate(Node):
    """acc += lhs @ rhs, with the products and the sum in float32."""

    acc: Var
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class For(Node):
    var: Var
    lo: Expr
    hi: Expr
    step: int
    body: tuple[Stmt, ...]
    pipelined: bool = False
    stages: int = 0


@dataclass(frozen=True, slots=True)
class While(Node):
    cond: Expr
    body: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class If(Node):
    cond: Expr
    then: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class Sync(Node):
    pass


@dataclass(frozen=True, slots=True)
class Effect(Node):
    """An intrinsic or an atomic called for its effect: a fence, a warp barrier, an update whose old value is unused."""

    value: Expr


@dataclass(frozen=True, slots=True)
class Breakpoint(Node):
    """`breakpoint()` in the kernel body."""


@dataclass(frozen=True, slots=True)
class Print(Node):
    """`print(...)` in the kernel body: literal text and values, separated by spaces, ended by a newline."""

    parts: tuple[str | Expr, ...]


@dataclass(frozen=True, slots=True)
class Assert(Node):
    """`assert condition, message`: when false, say where and stop the kernel."""

    cond: Expr
    text: str


type Stmt = (
    Assign
    | Store
    | TileAssign
    | TileStore
    | Accumulate
    | For
    | While
    | If
    | Sync
    | Effect
    | Breakpoint
    | Print
    | Assert
)


@dataclass(frozen=True, slots=True)
class Slot:
    """One C-level field of a parameter: a pointer, a dimension of a pointer, or a scalar."""

    name: str
    dtype: Dtype
    role: str


@dataclass(frozen=True, slots=True)
class Axis:
    """One axis of the work a kernel does, read from its body: `items(n)` loops and `block.tile(...)` slices.

    A grid-stride `items` loop (`strided`) is covered by any number of threads. A block tile is covered by one block
    for each `tile` elements along grid axis `grid`. `extents` are integer expressions over parameters and tensor
    dimensions, closed so a launcher can evaluate them from the arguments: the tensors the kernel stores with this
    tile, else the ones it loads. They must agree at launch.
    """

    grid: str
    tile: int
    extents: tuple[Expr, ...]
    strided: bool = False


@dataclass(frozen=True, slots=True)
class Kernel:
    name: str
    params: tuple[tuple[str, ParamType], ...]
    locals: tuple[Var, ...]
    tiles: tuple[Var, ...]
    shared: tuple[TensorRef, ...]
    body: tuple[Stmt, ...]
    written: frozenset[tuple[str, ...]]
    tiled: bool  # the body needs a block to cooperate, so a CPU runs it as one thread per block
    span: Span
    space: tuple[Axis, ...] = ()  # what the launch geometry is read from; empty when the body does not say
    cooperative: bool = False  # the body calls `grid.sync()`, so the launch must be cooperative


def tile_shape(slices: tuple[SliceExpr, ...]) -> tuple[int, ...]:
    """The extents of a tile read or written at `slices`: the squeezed axes are not in it."""
    return tuple(s.extent for s in slices if not s.squeeze)


def tile_axes(slices: tuple[SliceExpr, ...], idx: tuple[str, ...]) -> list[tuple[SliceExpr, str]]:
    """Pair each tensor axis with its element index in the tile: a squeezed axis has none, so it pairs with 0."""
    tail = iter(idx)
    return [(sl, "0" if sl.squeeze else next(tail)) for sl in slices]


def slots(name: str, type: ParamType) -> list[Slot]:
    """The flat C fields of one parameter, in declaration order."""
    match type:
        case Dtype():
            return [Slot(name, type, "scalar")]
        case TensorType():
            return [
                Slot(name, type.dtype, "ptr"),
                *(Slot(f"{name}_n{a}", I64, "dim") for a in range(type.rank)),
            ]
        case StructType():
            return [slot for field_name, field_type in type.fields for slot in slots(field_name, field_type)]
