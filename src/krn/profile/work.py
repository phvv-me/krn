"""Conservative logical work counts from scalar IR, before backend optimization."""

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import cache

from ..compiler.dtypes import Dtype, Kind
from ..compiler.ir import (
    Assign,
    Binary,
    Cast,
    Const,
    Dim,
    Expr,
    Fn,
    For,
    Intrinsic,
    Kernel,
    Load,
    Math,
    Op,
    Param,
    Stmt,
    Store,
    Sync,
    Unary,
    Var,
)


@dataclass(frozen=True, kw_only=True)
class Work:
    """Logical counts; None where a count is unknown, and flops_by_dtype names the dtype of each arithmetic count."""

    flops: int | None = 0
    bytes_moved: int | None = 0
    flops_by_dtype: dict[str, int] = field(default_factory=dict, hash=False)

    def plus(self, other: Work) -> Work:
        return Work(
            flops=None if self.flops is None or other.flops is None else self.flops + other.flops,
            bytes_moved=None
            if self.bytes_moved is None or other.bytes_moved is None
            else self.bytes_moved + other.bytes_moved,
            flops_by_dtype=dict(Counter(self.flops_by_dtype) + Counter(other.flops_by_dtype)),
        )

    def times(self, count: int) -> Work:
        return Work(
            flops=None if self.flops is None else self.flops * count,
            bytes_moved=None if self.bytes_moved is None else self.bytes_moved * count,
            flops_by_dtype={dtype: ops * count for dtype, ops in self.flops_by_dtype.items()},
        )


UNKNOWN = Work(flops=None, bytes_moved=None)


def payload_width(dtype: Dtype) -> int | None:
    if dtype.kind is Kind.BOOL:
        return 1
    return dtype.bits // 8 if dtype.bits and dtype.bits % 8 == 0 else None


def expression(value: Expr) -> Work:
    match value:
        case Const() | Var() | Param() | Dim():
            return Work()
        case Intrinsic(name=name, args=()) if name in {
            "grid.thread_rank",
            "grid.num_threads",
            "grid.block_rank",
            "grid.num_blocks",
            "block.thread_rank",
            "block.num_threads",
            "warp.lane",
            "warp.size",
        }:
            return Work()
        case Load(tensor=tensor, index=index):
            return expressions(index).plus(Work(bytes_moved=0 if tensor.extent else payload_width(tensor.type.dtype)))
        case Binary(op=op, lhs=lhs, rhs=rhs, type=Dtype() as dtype):
            cost = int(dtype.is_float and op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV))
            return expression(lhs).plus(expression(rhs)).plus(Work(flops=cost, flops_by_dtype={str(dtype): cost}))
        case Unary(operand=operand) | Cast(operand=operand):
            return expression(operand)
        case Math(fn=fn, args=args) as math_value:
            if fn is not Fn.FMA:
                return expressions(args).plus(Work(flops=None))
            return expressions(args).plus(Work(flops=2, flops_by_dtype={str(math_value.dtype): 2}))
        case _:
            return UNKNOWN


def expressions(values: tuple[Expr, ...]) -> Work:
    result = Work()
    for value in values:
        result = result.plus(expression(value))
    return result


def statements(body: tuple[Stmt, ...]) -> Work:
    result = Work()
    for node in body:
        match node:
            case Assign(value=value):
                cost = expression(value)
            case Store(tensor=tensor, index=index, value=value):
                cost = (
                    expression(value)
                    .plus(expressions(index))
                    .plus(Work(bytes_moved=0 if tensor.extent else payload_width(tensor.type.dtype)))
                )
            case For(lo=Const(value=int() as lo), hi=Const(value=int() as hi), step=step, body=inner):
                cost = statements(inner).times(len(range(lo, hi, step)))
            case Sync():
                cost = Work()
            case _:
                return UNKNOWN
        result = result.plus(cost)
    return result


@cache
def _body(body: tuple[Stmt, ...]) -> Work:
    """One body's per-thread counts, computed once per kernel rather than per launch."""
    return statements(body)


def counts(ir: Kernel, geometry: tuple[int, int, int, int]) -> Work:
    """Count scalar operations and global payload bytes; dynamic/tiled work is unknown."""
    if ir.tiled:
        return UNKNOWN
    return _body(ir.body).times(math.prod(geometry))
