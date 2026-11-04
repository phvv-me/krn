"""Dropping to hardware instructions for one expression, keeping the rest of the kernel portable.

A kernel body stays portable and an `@intrinsic` names one operation with a typed contract. Each target registers its
own lowering, and the registration with the base class `Target` is the fallback every other target gets, a CPU too.
The inline assembly is a *template string* (PEP 750): the interpolations are operands, so the compiler knows their
dtypes and picks register constraints, instead of formatting text by hand.

    NVIDIA   ptx(t"ex2.approx.ftz.f32 {OUT}, {x};", f32)     one SFU instruction (2 ulp), verified with ptxas
    AMD      amdgcn(t"v_exp_f32 {OUT}, {x}", f32)             the same transcendental unit
    Apple    msl(t"fast::exp2({x})", f32)                      Metal has no inline assembly: a raw MSL expression
    others   exp(x * ln2)                                     the portable fallback

Use an escape hatch when the measured profile asks for it; the portable body is the reference every escape hatch is
tested against. The prototype's assembler probes (experiments/ptx_asm.py) already show which PTX each architecture
accepts, so a lowering that would not assemble on `sm_120` is rejected in CI, not on the GPU.

Python features used: PEP 750 template strings with typed interpolations, decorator registration of lowerings,
`when=` predicates over a typed target (`t.sm >= 80`), and a fallback that is an ordinary function.
"""

import math

from krn import OUT, Const, Matrix, T, amdgcn, block, exp, intrinsic, kernel, msl, ptx, struct
from krn.facts import targets as targets

_LN2 = math.log(2.0)


@intrinsic
def fast_exp2(x: T.f32) -> T.f32:
    """2**x. This signature is the contract every lowering must meet."""
    raise NotImplementedError


@fast_exp2.lower(targets.Nvidia, when=lambda t: t.sm >= 75)
def fast_exp2_nvidia(x: T.f32) -> T.f32:
    return ptx(t"ex2.approx.ftz.f32 {OUT}, {x};", T.f32)


@fast_exp2.lower(targets.AMD)
def fast_exp2_amd(x: T.f32) -> T.f32:
    return amdgcn(t"v_exp_f32 {OUT}, {x}", T.f32)


@fast_exp2.lower(targets.Apple)
def fast_exp2_apple(x: T.f32) -> T.f32:
    return msl(t"fast::exp2({x})", T.f32)


@fast_exp2.lower(targets.Target)
def fast_exp2_portable(x: T.f32) -> T.f32:
    return exp(x * _LN2)


@struct
class Tiling:
    rows: Const[int] = 4
    cols: Const[int] = 2048


@kernel
def softmax_exp2(x: Matrix[T.f16], y: Matrix[T.f16], t: Tiling) -> None:
    """Row softmax through base-2 exponentials: the log2(e) factor folds into one multiply."""
    rows, cols = block.tile(t.rows), slice(0, t.cols)
    v = x[rows, cols].to(T.f32).mul_(1.4426950408889634)
    e = v.sub(v.max(keepdims=True)).map(fast_exp2)
    y[rows, cols] = e.div(e.sum(keepdims=True)).to(T.f16)
