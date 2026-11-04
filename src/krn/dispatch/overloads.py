"""The overloads of the general functions: packed lanes to SIMD rows, integers and floats to ordinary IR.

The operands arrive as constants or variables, each evaluated once by capture, so an expansion may read one twice.
"""

from collections.abc import Sequence

from krn.compiler.dtypes import BOOL, I8X4, I16X2, U8X4, U16X2, Dtype, Kind, uint
from krn.compiler.errors import Span
from krn.compiler.ir import Binary, Cast, Const, Expr, Fn, Math, Op, Select

from .registry import Expansion, Family, Instruction, register

NVIDIA = frozenset({"nvidia"})
APPLE = frozenset({"apple"})
PACKED = (I8X4, U8X4, I16X2, U16X2)
UNSIGNED = (U8X4, U16X2)
UNSIGNED_LANES = {I8X4: U8X4, I16X2: U16X2}


def _compare(op: Op, lhs: Expr, rhs: Expr, span: Span) -> Expr:
    return Binary(BOOL, op, lhs, rhs, span=span)


def _saturating_add(args: Sequence[Expr], span: Span) -> Expr:
    """`a + b` clamped to the type's range. The test compares b with the room left in a, so the sum is only computed
    when it fits: no operation overflows, which matters because signed overflow is undefined in MSL and CUDA."""
    a, b = args
    dtype = a.dtype
    if dtype.kind is Kind.UINT:
        top = Const(dtype, (1 << dtype.bits) - 1, span=span)
        room = Binary(dtype, Op.SUB, top, b, span=span)
        return Select(dtype, _compare(Op.GT, a, room, span), top, Binary(dtype, Op.ADD, a, b, span=span), span=span)
    high = (1 << (dtype.bits - 1)) - 1
    top = Const(dtype, high, span=span)
    bottom = Binary(dtype, Op.SUB, Const(dtype, -high, span=span), Const(dtype, 1, span=span), span=span)
    zero = Const(dtype, 0, span=span)
    upward = Binary(
        BOOL,
        Op.LAND,
        _compare(Op.GT, b, zero, span),
        _compare(Op.GT, a, Binary(dtype, Op.SUB, top, b, span=span), span),
        span=span,
    )
    downward = Binary(
        BOOL,
        Op.LAND,
        _compare(Op.LT, b, zero, span),
        _compare(Op.LT, a, Binary(dtype, Op.SUB, bottom, b, span=span), span),
        span=span,
    )
    total = Binary(dtype, Op.ADD, a, b, span=span)
    return Select(dtype, upward, top, Select(dtype, downward, bottom, total, span=span), span=span)


def _saturating_sub(args: Sequence[Expr], span: Span) -> Expr:
    """`a - b` clamped to the type's range. The subtraction is only computed when it fits, as in `_saturating_add`."""
    a, b = args
    dtype = a.dtype
    zero = Const(dtype, 0, span=span)
    if dtype.kind is Kind.UINT:
        return Select(dtype, _compare(Op.LT, a, b, span), zero, Binary(dtype, Op.SUB, a, b, span=span), span=span)
    high = (1 << (dtype.bits - 1)) - 1
    top = Const(dtype, high, span=span)
    bottom = Const(dtype, -high - 1, span=span)
    upward = Binary(
        BOOL,
        Op.LAND,
        _compare(Op.LT, b, zero, span),
        _compare(Op.GT, a, Binary(dtype, Op.ADD, top, b, span=span), span),
        span=span,
    )
    downward = Binary(
        BOOL,
        Op.LAND,
        _compare(Op.GT, b, zero, span),
        _compare(Op.LT, a, Binary(dtype, Op.ADD, bottom, b, span=span), span),
        span=span,
    )
    total = Binary(dtype, Op.SUB, a, b, span=span)
    return Select(dtype, upward, top, Select(dtype, downward, bottom, total, span=span), span=span)


def _average(args: Sequence[Expr], span: Span) -> Expr:
    """The average of a and b rounded half up, floor((a + b + 1) / 2), as (a >> 1) + (b >> 1) + the odd bit of a | b:
    no intermediate sum exists, so it cannot overflow for any integer type."""
    a, b = args
    dtype = a.dtype
    one = Const(dtype, 1, span=span)
    halves = Binary(
        dtype,
        Op.ADD,
        Binary(dtype, Op.SHR, a, one, span=span),
        Binary(dtype, Op.SHR, b, one, span=span),
        span=span,
    )
    rounding = Binary(dtype, Op.AND, Binary(dtype, Op.OR, a, b, span=span), one, span=span)
    return Binary(dtype, Op.ADD, halves, rounding, span=span)


def _maximum(args: Sequence[Expr], span: Span) -> Expr:
    a, b = args
    return Select(a.type, _compare(Op.GT, a, b, span), a, b, span=span)


def _float_maximum(args: Sequence[Expr], span: Span) -> Expr:
    a, b = args
    return Math(a.type, Fn.MAXIMUM, (a, b), span=span)


def _absolute_difference(args: Sequence[Expr], span: Span) -> Expr:
    """The magnitude of a - b. A signed pair is ordered by value and subtracted as unsigned, so the result is exact."""
    a, b = args
    dtype = a.dtype
    if dtype.is_float:
        return Math(dtype, Fn.ABS, (Binary(dtype, Op.SUB, a, b, span=span),), span=span)
    ordered = _compare(Op.GT, a, b, span)
    if dtype.kind is Kind.SINT:
        result = uint(dtype.bits)
        a, b = Cast(result, a, span=span), Cast(result, b, span=span)
    else:
        result = dtype
    return Select(
        result,
        ordered,
        Binary(result, Op.SUB, a, b, span=span),
        Binary(result, Op.SUB, b, a, span=span),
        span=span,
    )


def _abs_diff_result(dtype: Dtype) -> Dtype:
    return UNSIGNED_LANES.get(dtype, dtype)


def _register_packed() -> None:
    """Metal's vector integer functions are one native call on every packed type, so only NVIDIA's SIMD instructions
    and Metal are native; the signed average is native on Metal alone, since NVIDIA's signed SIMD average rounds away
    from zero."""
    for packed in PACKED:
        both = NVIDIA | APPLE
        register("add_sat", packed, Instruction(f"general.add_sat.{packed}", both))
        register("sub_sat", packed, Instruction(f"general.sub_sat.{packed}", both))
        register("maximum", packed, Instruction(f"general.maximum.{packed}", both))
        register("abs_diff", packed, Instruction(f"general.abs_diff.{packed}", both, _abs_diff_result(packed)))
        register("avg", packed, Instruction(f"general.avg.{packed}", both if packed in UNSIGNED else APPLE))


def _register_scalars() -> None:
    register("add_sat", Family.INT, Expansion(_saturating_add))
    register("sub_sat", Family.INT, Expansion(_saturating_sub))
    register("avg", Family.INT, Expansion(_average))
    register("maximum", Family.INT, Expansion(_maximum))
    register("maximum", Family.FLOAT, Expansion(_float_maximum))
    register("abs_diff", Family.INT, Expansion(_absolute_difference))
    register("abs_diff", Family.FLOAT, Expansion(_absolute_difference))


_register_packed()
_register_scalars()
