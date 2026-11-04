from dataclasses import replace

from krn.compiler.dtypes import BOOL, F32, I64, PY_INT, uint
from krn.compiler.errors import Span
from krn.compiler.ir import (
    Assign,
    Binary,
    Cast,
    Const,
    Dim,
    Fn,
    For,
    If,
    Intrinsic,
    Kernel,
    Load,
    Math,
    Op,
    Param,
    Store,
    Sync,
    TensorRef,
    TensorType,
    Unary,
    Var,
)
from krn.profile.work import UNKNOWN, Work, counts, expression, payload_width, statements

SPAN = Span(__file__, 1, 1)


def test_logical_work_static_loops_and_unknown_control_flow() -> None:
    assert payload_width(BOOL) == 1
    assert payload_width(uint(3)) is None
    assert payload_width(F32) == 4
    assert payload_width(PY_INT) is None
    c = Const(F32, 2.0, span=SPAN)
    index = Const(I64, 0, span=SPAN)
    v = Var(F32, "x", span=SPAN)
    tensor = TensorRef(("x",), TensorType(F32, 1))
    load = Load(F32, tensor, (index,), span=SPAN)
    add = Binary(F32, Op.ADD, load, c, span=SPAN)
    store = Store(tensor, (index,), add, span=SPAN)
    ir = Kernel("test", (), (), (), (), (store,), frozenset(), False, SPAN)
    assert counts(ir, (2, 1, 1, 4)) == Work(flops=8, bytes_moved=64, flops_by_dtype={"f32": 8})
    assert counts(replace(ir, tiled=True), (1, 1, 1, 1)) == UNKNOWN
    assert counts(replace(ir, body=()), (1, 1, 1, 1)) == Work()
    assert expression(Unary(F32, Op.NEG, c, span=SPAN)) == Work()
    assert expression(Cast(F32, index, span=SPAN)) == Work()
    assert expression(Math(F32, Fn.FMA, (c, c, c), span=SPAN)).flops == 2
    assert expression(Math(F32, Fn.EXP, (c,), span=SPAN)).flops is None
    assert expression(Binary(I64, Op.ADD, index, index, span=SPAN)).flops == 0
    assert expression(Binary(F32, Op.LT, c, c, span=SPAN)).flops == 0
    assert expression(Param(F32, ("a",), span=SPAN)) == Work()
    assert expression(Dim(I64, tensor, 0, span=SPAN)) == Work()
    assert expression(Intrinsic(I64, "grid.thread_rank", span=SPAN)) == Work()
    assert expression(Intrinsic(F32, "unknown", (c,), span=SPAN)) == UNKNOWN
    shared = replace(tensor, extent=8)
    assert expression(Load(F32, shared, (index,), span=SPAN)).bytes_moved == 0
    assert statements((Store(shared, (index,), c, span=SPAN),)).bytes_moved == 0
    loop = For(Var(I64, "i", span=SPAN), index, Const(I64, 3, span=SPAN), 1, (store,), span=SPAN)
    assert statements((loop,)).flops == 3
    assert statements((Sync(span=SPAN), Assign(v, c, span=SPAN))) == Work()
    assert statements((If(index, (store,), (), span=SPAN),)) == UNKNOWN
    assert statements((replace(loop, hi=v),)) == UNKNOWN
    assert UNKNOWN.plus(Work()).times(3) == UNKNOWN
    assert Work().plus(UNKNOWN) == UNKNOWN
