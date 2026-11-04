"""Multiply-adds that strict arithmetic leaves unfused, found in the captured IR.

krn never contracts `a * b + c` into one fused operation on its own, because strict arithmetic rounds the product
first. A multiply whose only use is an add or subtract is where `math.fma` is one instruction and more accurate. The
warning spells that replacement from the IR; the source is read only to quote the line.
"""

import linecache
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from krn.facts import catalog
from krn.facts.targets import Feature, Target
from krn.runtime.settings import Warn

from .dtypes import BF16, F16, F32, F64, Dtype
from .ir import (
    Assign,
    Binary,
    Cast,
    Const,
    Dim,
    Expr,
    For,
    Intrinsic,
    Kernel,
    Load,
    Math,
    Op,
    Param,
    Select,
    Stmt,
    Store,
    TensorRef,
    TileAssign,
    Unary,
    Var,
)
from .performance import PerformanceWarning, warn_once
from .validity import children, walk

KIND = "fusable-multiply-add"


def _fusable(dtype: Dtype, target: Target) -> bool:
    """Whether `math.fma` on `dtype` is one instruction on `target`: half types need a native lowering and f64 needs
    float64 arithmetic. No target table records a fused f32, so f32 counts as fusable on every target."""
    if dtype == F64:
        return target.has(Feature.FLOAT64)
    if dtype in (F16, BF16):
        return bool(catalog.BY_NAME[f"fma._{dtype}"].lowering(target))
    return dtype == F32


@dataclass(frozen=True, kw_only=True)
class _Pattern:
    """`consumer` adds or subtracts `product` (its left operand when `product_left`), a multiply used nowhere else.
    `statement` evaluates the consumer."""

    statement: Stmt
    consumer: Binary
    product: Binary
    product_left: bool


def _subexpressions(expr: Expr) -> Iterator[Expr]:
    yield expr
    for child in children(expr):
        yield from _subexpressions(child)


def _evaluated(statement: Stmt) -> Iterator[Expr]:
    """The expressions a statement evaluates, which excludes the variables it assigns."""
    match statement:
        case Assign(value=value) | TileAssign(value=value):
            yield value
        case For(lo=lo, hi=hi):
            yield from (lo, hi)
        case _:
            yield from children(statement)


def _temporaries(statements: list[Stmt]) -> dict[str, Binary]:
    """The multiplies held by a local that is assigned once and read once, so its one use can take the product."""
    assigned = Counter(s.var.name for s in statements if isinstance(s, Assign | TileAssign | For))
    reads = Counter(
        node.name
        for statement in statements
        for root in _evaluated(statement)
        for node in _subexpressions(root)
        if isinstance(node, Var)
    )
    values = {s.var.name: s.value for s in statements if isinstance(s, Assign | TileAssign)}
    return {
        name: value
        for name, value in values.items()
        if assigned[name] == 1 and reads[name] == 1 and isinstance(value, Binary) and value.op is Op.MUL
    }


def _product(operand: Expr, temporaries: dict[str, Binary]) -> Binary | None:
    candidate = temporaries.get(operand.name) if isinstance(operand, Var) else operand
    return candidate if isinstance(candidate, Binary) and candidate.op is Op.MUL else None


def _patterns(kernel: Kernel, target: Target) -> list[_Pattern]:
    """Every add or subtract of a fusable float multiply, in statement order; a consumer of two takes the left one."""
    statements = list(walk(kernel.body))
    temporaries = _temporaries(statements)
    found: list[_Pattern] = []
    for statement in statements:
        for root in _evaluated(statement):
            for expr in _subexpressions(root):
                if not (isinstance(expr, Binary) and expr.op in (Op.ADD, Op.SUB) and _fusable(expr.dtype, target)):
                    continue
                if (left := _product(expr.lhs, temporaries)) is not None:
                    found.append(_Pattern(statement=statement, consumer=expr, product=left, product_left=True))
                elif (right := _product(expr.rhs, temporaries)) is not None:
                    found.append(_Pattern(statement=statement, consumer=expr, product=right, product_left=False))
    return found


_PRECEDENCE = {
    Op.OR: 1,
    Op.XOR: 2,
    Op.AND: 3,
    Op.SHL: 4,
    Op.SHR: 4,
    Op.ADD: 5,
    Op.SUB: 5,
    Op.MUL: 6,
    Op.DIV: 6,
    Op.MOD: 6,
}
_TIGHTEST = 7  # above every binary operator: a unary operand is parenthesized only when it is a binary operation


@dataclass(frozen=True, kw_only=True)
class _Spelling:
    """Writes IR as krn source, with the consumer of one pattern spelled as its `math.fma` call."""

    pattern: _Pattern

    def statement(self, statement: Stmt) -> str:
        match statement:
            case Assign(var=var, value=value):
                return f"{var.name} = {self.expr(value)}"
            case Store(tensor=tensor, index=index, value=value):
                return f"{self.access(tensor, index)} = {self.expr(value)}"
            case _:
                return self.expr(self.pattern.consumer)

    def expr(self, expr: Expr, parent: int = 0) -> str:
        """`expr` as source, parenthesized when a binary operation of lower precedence than `parent` holds it."""
        if expr is self.pattern.consumer:
            return self.fused()
        match expr:
            case Var(name=name):
                return name
            case Const(value=value):
                return repr(value)
            case Param(path=path):
                return ".".join(path)
            case Dim(tensor=tensor, axis=axis):
                return f"{'.'.join(tensor.path)}.shape[{axis}]"
            case Load(tensor=tensor, index=index):
                return self.access(tensor, index)
            case Binary(op=op, lhs=lhs, rhs=rhs):
                precedence = _PRECEDENCE.get(op, 0)
                symbol = "//" if op is Op.DIV and not expr.dtype.is_float else op
                text = f"{self.expr(lhs, precedence)} {symbol} {self.expr(rhs, precedence + 1)}"
                return f"({text})" if precedence < parent else text
            case Unary(op=op, operand=operand):
                return f"{'-' if op is Op.NEG else op}{self.expr(operand, _TIGHTEST)}"
            case Math(fn=fn, args=args):
                return f"math.{fn}({self.arguments(args)})"
            case Intrinsic(name=name, args=args):
                return f"{name}({self.arguments(args)})"
            case Cast(type=dtype, operand=operand):
                return f"T.{dtype}({self.expr(operand)})"
            case Select(cond=cond, then=then, orelse=orelse):
                return f"({self.expr(then)} if {self.expr(cond)} else {self.expr(orelse)})"
            case _:
                return "..."

    def arguments(self, args: tuple[Expr, ...]) -> str:
        return ", ".join(self.expr(arg) for arg in args)

    def access(self, tensor: TensorRef, index: tuple[Expr, ...]) -> str:
        return f"{'.'.join(tensor.path)}[{self.arguments(index)}]"

    def fused(self) -> str:
        """`math.fma` of the product's factors and the other operand, negated where a subtraction takes the product."""
        product, consumer = self.pattern.product, self.pattern.consumer
        other = consumer.rhs if self.pattern.product_left else consumer.lhs
        factors, addend = [self.expr(product.lhs), self.expr(product.rhs)], self.expr(other)
        if consumer.op is Op.SUB:
            if self.pattern.product_left:
                addend = f"-{self.expr(other, _TIGHTEST)}"
            else:
                factors[0] = f"-{self.expr(product.lhs, _TIGHTEST)}"
        return f"math.fma({factors[0]}, {factors[1]}, {addend})"


def check_fusable(kernel: Kernel, target: Target, limits: Warn) -> None:
    """Warn once per kernel and target, quoting the first pattern's line and its fused spelling, when the kernel has at
    least `min_fusable_multiply_adds` patterns. The target table has no fused-f32 flag, so f32 warns everywhere."""
    if not limits.hint_fusable_multiply_add:
        return
    found = sorted(_patterns(kernel, target), key=lambda p: (p.consumer.span.line, p.consumer.span.col))
    if not found or len(found) < limits.min_fusable_multiply_adds:
        return
    first = found[0]
    span = first.consumer.span
    source = linecache.getline(span.file, span.line).strip()
    noun = "pattern" if len(found) == 1 else "patterns"
    spelled = _Spelling(pattern=first).statement(first.statement)
    message = (
        f"has {len(found)} fusable multiply-add {noun}; the first, at line {span.line}, is `{source}`: "
        "two roundings; math.fma(a, b, c) is a single fused instruction and more accurate; "
        f"spell that line `{spelled}`"
    )
    warn_once(kernel.name, target.arch, KIND, message, PerformanceWarning, span, kernel.span)
