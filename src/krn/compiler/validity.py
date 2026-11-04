"""Which lanes of a tile exist.

A tile is read at a position that may run past the edge of its tensor. Those lanes read as zero, which is the visible
value, but they are *invalid*: an elementwise operation is invalid where any operand is, a reduction skips them (an
invalid lane is neutral for `max`, `min`, `sum` and `prod` alike, which zero is for the last two only), a matrix product
counts them as zero, and a store leaves them alone. This module finds the tile variables that can hold such lanes, so
the C emitter and the emulator keep a validity byte for those and for nothing else.
"""

import dataclasses
from collections.abc import Iterator

from .ir import Accumulate, Expr, For, If, Kernel, Reduce, Reduction, Stmt, TileAssign, TileLoad, Var, While


def walk(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    """Every statement of `body`, the nested ones too."""
    for statement in body:
        yield statement
        match statement:
            case For(body=inner) | While(body=inner):
                yield from walk(inner)
            case If(then=then, orelse=orelse):
                yield from walk(then)
                yield from walk(orelse)


def children(e: Expr | Stmt) -> Iterator[Expr]:
    for field in dataclasses.fields(e):
        value = getattr(e, field.name)
        if isinstance(value, Expr):
            yield value
        elif isinstance(value, tuple):
            yield from (v for v in value if isinstance(v, Expr))


def can_be_invalid(e: Expr, masked: frozenset[str] | set[str]) -> bool:
    """Whether `e` can have an invalid lane, given the tile variables known to. A mean is valid everywhere: a lane
    with no valid operand lane is NaN, as numpy's mean of nothing is, and that NaN is stored."""
    match e:
        case TileLoad():
            return True
        case Var(name=name):
            return name in masked
        case Reduce(op=Reduction.MEAN):
            return False
    return any(can_be_invalid(child, masked) for child in children(e))


def masked_tiles(kernel: Kernel) -> frozenset[str]:
    """The tile variables that can hold invalid lanes: those assigned a value that can, to a fixed point."""
    masked: set[str] = set()
    grown = True
    while grown:
        grown = False
        for statement in walk(kernel.body):
            match statement:
                case TileAssign(var=var, value=value) if can_be_invalid(value, masked):
                    grown |= var.name not in masked
                    masked.add(var.name)
                case Accumulate(acc=acc, lhs=lhs, rhs=rhs) if can_be_invalid(lhs, masked) or can_be_invalid(
                    rhs, masked
                ):
                    grown |= acc.name not in masked
                    masked.add(acc.name)
    return frozenset(masked)
