"""What the body of a kernel says about its launch: the iteration space and the cooperative barrier.

Capture records the counts of the `items(n)` loops. The block tiles are found here, in the slices that load and store
tensors: a slice that starts at `block.index.<axis> * size` makes grid axis `axis` cover `size` elements of the
dimension it slices. A launcher evaluates the extents on the arguments and gets a geometry, so `k(a, b, c)` needs no
geometry in brackets. A body that says nothing needs `launch=`.
"""

import dataclasses
import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from krn import Tensor
from krn.arithmetic import ceildiv
from krn.frontend.geometry import Blocks, Items
from krn.frontend.numerics import Scalar

from .dtypes import PY_INT
from .ir import (
    Assign,
    Axis,
    Binary,
    Cast,
    Const,
    Dim,
    Expr,
    Intrinsic,
    Kernel,
    Op,
    Param,
    Select,
    SliceExpr,
    Stmt,
    TensorRef,
    TileLoad,
    TileStore,
    Unary,
    Var,
)
from .validity import children, walk

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type Operand = Tensor | np.ndarray | np.generic | Scalar | int | float | Mapping[str, Operand] | DataclassInstance

BINARY: dict[Op, Callable[[int, int], int | bool]] = {
    Op.ADD: operator.add,
    Op.SUB: operator.sub,
    Op.MUL: operator.mul,
    Op.DIV: operator.floordiv,
    Op.MOD: operator.mod,
    Op.AND: operator.and_,
    Op.OR: operator.or_,
    Op.XOR: operator.xor,
    Op.SHL: operator.lshift,
    Op.SHR: operator.rshift,
    Op.LT: operator.lt,
    Op.LE: operator.le,
    Op.GT: operator.gt,
    Op.GE: operator.ge,
    Op.EQ: operator.eq,
    Op.NE: operator.ne,
    Op.LAND: lambda a, b: a and b,
    Op.LOR: lambda a, b: a or b,
}
UNARY: dict[Op, Callable[[int], int | bool]] = {
    Op.NEG: operator.neg,
    Op.NOT: operator.not_,
    Op.INVERT: operator.invert,
}
FIXES = (
    "write the loop as `for i in items(n)`, or slice the tensors with `block.tile`, so the body says how big the work "
    "is; or pass `launch=Items(n)` or `launch=Blocks(...)` to the call"
)


class LaunchError(ValueError):
    """The launch geometry of a kernel cannot be read from its body and its arguments."""

    def __init__(self, kernel: Kernel, reason: str) -> None:
        super().__init__(f"{kernel.name}: the launch geometry cannot be inferred, {reason}; {FIXES}")


class Unreadable(Exception):
    """An extent reads something only the device has: a local variable, a tensor element, an intrinsic."""


def iteration_space(body: tuple[Stmt, ...], counts: Sequence[Expr]) -> tuple[Axis, ...]:
    """The axes of a captured body: one strided axis for each distinct `items` count, one tile axis for each grid axis
    and tile size that slices a tensor. A tile axis is as long as the tensors the body stores with it, else loads."""
    definitions = single_assignments(body)
    items = dict.fromkeys(Axis("x", 1, (closed(count, definitions),), strided=True) for count in counts)
    stored: defaultdict[tuple[str, int], dict[Expr, None]] = defaultdict(dict)
    loaded: defaultdict[tuple[str, int], dict[Expr, None]] = defaultdict(dict)
    for tensor, slices, writes in tile_uses(body):
        for dimension, tile in enumerate(slices):
            if (grid := block_axis(tile)) is not None:
                (stored if writes else loaded)[grid, tile.extent][Dim(PY_INT, tensor, dimension, span=tile.span)] = None
    tiles = (Axis(*key, tuple(stored.get(key) or loaded[key])) for key in dict.fromkeys((*stored, *loaded)))
    return (*items, *tiles)


def synchronizes(body: tuple[Stmt, ...]) -> bool:
    """Whether the body calls `grid.sync()`, which needs a cooperative launch."""
    return any(
        isinstance(found, Intrinsic) and found.name == "grid.sync"
        for statement in walk(body)
        for expression in expressions(statement)
        for found in nested(expression)
    )


def geometry(kernel: Kernel, values: Sequence[Operand]) -> Items | Blocks:
    """The launch the body of `kernel` asks for, from `values`, the arguments in parameter order.

    A grid-stride `items` loop is covered by `Items`, as long as the longest loop. Tiles give one block for each tile
    along the grid axis they follow, rounded up. A launch always has one block at least.
    """
    strided = [axis for axis in kernel.space if axis.strided]
    tiled = [axis for axis in kernel.space if not axis.strided]
    if not kernel.space:
        raise LaunchError(kernel, "its body has no `items(n)` loop and slices no tensor with `block.tile`")
    if strided and tiled:
        raise LaunchError(kernel, "its body mixes `items(n)` loops with `block.tile` slices")
    arguments = dict(zip((name for name, _ in kernel.params), values, strict=True))
    if strided:
        return Items(max(1, *(extent(kernel, axis, arguments) for axis in strided)))
    blocks: dict[str, int] = {}
    for axis in tiled:
        if axis.grid in blocks:
            raise LaunchError(kernel, f"its body tiles grid axis {axis.grid} by more than one size")
        blocks[axis.grid] = max(1, ceildiv(extent(kernel, axis, arguments), axis.tile))
    return Blocks(blocks.get("x", 1), blocks.get("y", 1), blocks.get("z", 1))


def extent(kernel: Kernel, axis: Axis, arguments: Mapping[str, Operand]) -> int:
    """How long `axis` is on these arguments. Every tensor the axis follows must have the same length."""
    try:
        lengths = {evaluate(each, arguments) for each in axis.extents}
    except Unreadable as unreadable:
        raise LaunchError(kernel, f"its extent reads {unreadable}, which the host cannot evaluate") from None
    if len(lengths) > 1:
        raise LaunchError(
            kernel,
            f"the tensors it tiles along grid axis {axis.grid} have lengths {sorted(lengths)}",
        )
    return lengths.pop()


def evaluate(expression: Expr, arguments: Mapping[str, Operand]) -> int:
    """An extent on the arguments: constants, scalar parameters, tensor dimensions and integer arithmetic."""
    match expression:
        case Const(value=value):
            return int(value)
        case Param(path=path):
            return whole(operand(arguments, path))
        case Dim(tensor=tensor, axis=axis):
            return length(operand(arguments, tensor.path), axis)
        case Cast(operand=inner):
            return evaluate(inner, arguments)
        case Unary(op=op, operand=inner) if op in UNARY:
            return int(UNARY[op](evaluate(inner, arguments)))
        case Binary(op=op, lhs=lhs, rhs=rhs) if op in BINARY:
            return int(BINARY[op](evaluate(lhs, arguments), evaluate(rhs, arguments)))
        case Select(cond=cond, then=then, orelse=orelse):
            return evaluate(then if evaluate(cond, arguments) else orelse, arguments)
        case Var(name=name):
            raise Unreadable(f"the local variable `{name}`")
    raise Unreadable(f"a {type(expression).__name__}")


def operand(arguments: Mapping[str, Operand], path: tuple[str, ...]) -> Operand:
    """The argument at `path`: a parameter, then the fields of a struct, whether it is a mapping or an object."""
    value = arguments[path[0]]
    for name in path[1:]:
        value = value[name] if isinstance(value, Mapping) else getattr(value, name)
    return value


def whole(value: Operand) -> int:
    if isinstance(value, Scalar):
        return int(value.value)
    if isinstance(value, int | float | np.integer | np.floating | np.bool_):
        return int(value)
    raise Unreadable(f"a {type(value).__name__} scalar")


def length(value: Operand, axis: int) -> int:
    match value:
        case Tensor() | np.ndarray():
            return value.shape[axis]
    raise TypeError(f"expected a tensor, got {type(value).__name__}")


def block_axis(tile: SliceExpr) -> str | None:
    """The grid axis a slice follows when it is the tile of a block, `[block.index.<axis> * size, + size)`."""
    match tile.start:
        case Binary(
            op=Op.MUL,
            lhs=Cast(operand=Intrinsic(name=name)) | Intrinsic(name=name),
            rhs=Const(value=size),
        ):
            follows = size == tile.extent
        case Cast(operand=Intrinsic(name=name)) | Intrinsic(name=name):
            follows = tile.extent == 1
        case _:
            return None
    return name.removeprefix("block.index.") if follows and name.startswith("block.index.") else None


def tile_uses(body: tuple[Stmt, ...]) -> Iterator[tuple[TensorRef, tuple[SliceExpr, ...], bool]]:
    """Every tile of a global tensor the body loads or stores: the tensor, its slices and whether it is written."""
    for statement in walk(body):
        if isinstance(statement, TileStore) and not statement.tensor.extent:
            yield statement.tensor, statement.slices, True
        for expression in expressions(statement):
            for found in nested(expression):
                if isinstance(found, TileLoad) and not found.tensor.extent:
                    yield found.tensor, found.slices, False


def single_assignments(body: tuple[Stmt, ...]) -> dict[str, Expr]:
    """The locals the body assigns once, at its top level: what they stand for in any expression that reads them."""
    assigned = Counter(statement.var.name for statement in walk(body) if isinstance(statement, Assign))
    return {
        statement.var.name: statement.value
        for statement in body
        if isinstance(statement, Assign) and assigned[statement.var.name] == 1
    }


def closed(expression: Expr, definitions: Mapping[str, Expr]) -> Expr:
    """`expression` with the locals it reads replaced by what they stand for, as far as `definitions` says."""
    if isinstance(expression, Var):
        return closed(definitions[expression.name], definitions) if expression.name in definitions else expression
    match expression:
        case Binary(lhs=lhs, rhs=rhs):
            return dataclasses.replace(expression, lhs=closed(lhs, definitions), rhs=closed(rhs, definitions))
        case Unary(operand=inner) | Cast(operand=inner):
            return dataclasses.replace(expression, operand=closed(inner, definitions))
        case Select(cond=cond, then=then, orelse=orelse):
            return dataclasses.replace(
                expression,
                cond=closed(cond, definitions),
                then=closed(then, definitions),
                orelse=closed(orelse, definitions),
            )
    return expression


def expressions(statement: Stmt) -> Iterator[Expr]:
    """The expressions a statement holds directly, not those of the statements nested in it."""
    for field in dataclasses.fields(statement):
        value = getattr(statement, field.name)
        if isinstance(value, Expr):
            yield value
        elif isinstance(value, tuple):
            yield from (item for item in value if isinstance(item, Expr))


def nested(expression: Expr) -> Iterator[Expr]:
    """The expression and every expression inside it."""
    yield expression
    for child in children(expression):
        yield from nested(child)
