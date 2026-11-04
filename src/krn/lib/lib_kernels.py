"""The multiply that `lib_ours` launches: one tiled kernel, with the bias and the activation chosen by constants.

A `Const[bool]` folds its branch at capture, so each combination is its own program with no test left in it. The
accumulator is float32 whatever the operands are, and the result is rounded once, to the type of `out`.
"""

from krn import (
    Const,
    Matrix,
    T,
    Vector,
    block,
    kernel,
    pipelined,
    struct,
    zeros,  # capture reads this line
)


@struct
class MatmulTiling:
    bm: Const[int] = 16
    bn: Const[int] = 16
    bk: Const[int] = 8
    bias: Const[bool] = False
    relu: Const[bool] = False
    gelu: Const[bool] = False
    silu: Const[bool] = False


@kernel
def matmul_epilogue[F: T.FloatingPoint](
    t: MatmulTiling, a: Matrix[F], b: Matrix[F], bias: Vector[F], out: Matrix[F]
) -> None:
    """`out = activation(a @ b + bias)`; `bias` is read only when `t.bias` is set, so any vector will do otherwise."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    if t.bias:
        acc += bias[cols].to(T.f32)
    if t.relu:
        acc = acc.relu()
    if t.gelu:
        acc = acc.gelu()
    if t.silu:
        acc = acc.silu()
    out[rows, cols] = acc.to(out.dtype)
