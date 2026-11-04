"""Narrow floats (fp8, fp6 and fp4) converted in software where the target has no conversion instruction.

A narrow float is one code per byte, with the format's sign in the top bit of the code. A conversion decodes a code to
float32 with integer bit operations and encodes a float32 with round to nearest even and the PTX `.satfinite` rule:
an overflow or an infinity becomes the largest finite value, a NaN becomes the canonical NaN (the largest positive NaN
code), and a format without NaN maps a NaN to its largest positive value. `reference_encode` and `reference_decode`
are the table oracle that the composed text and the native instructions match bit for bit.
"""

import math
from functools import cache
from typing import Final, NoReturn

import numpy as np
from numpy.typing import ArrayLike

from krn.data.formats import E2M1, E2M3, E3M2, E4M3FN, E5M2, FloatFormat, Special
from krn.facts.targets import Dtype as Token
from krn.facts.targets import Nvidia, Target
from krn.runtime.settings import current_settings

from .dtypes import Dtype, fp
from .errors import Span, TargetError
from .ir import Binary, Cast, Expr, Intrinsic, Kernel, Math, Reduce, Unary
from .numerics import expressions
from .performance import PerformanceWarning, warn_once
from .validity import children

FORMATS: Final[dict[Dtype, FloatFormat]] = {
    fp(4, 3): E4M3FN, fp(5, 2): E5M2, fp(2, 3): E2M3, fp(3, 2): E3M2, fp(2, 1): E2M1,
}  # fmt: skip
# The FNUZ fp8 floats: a dtype of their own with no composition, so every conversion of one is refused.
FNUZ_DTYPES: Final = (fp(4, 3, fnuz=True), fp(5, 2, fnuz=True))
NAMES: Final[dict[FloatFormat, str]] = {fmt: str(dtype) for dtype, fmt in FORMATS.items()}
# Where NVIDIA has the conversion instructions: fp8 from sm_89 (PTX 8.1), fp4 and fp6 on these three (PTX 9.4 cvt
# section, `.rn.satfinite` forms). sm_101a was renamed sm_110a in PTX 9.0. sm_103a has only the `.rs` form and sm_121a
# none, so neither is listed.
FP4_ARCHES: Final = frozenset({"sm_100a", "sm_101a", "sm_120a"})
# gfx942 stores fp8 as FNUZ (no negative zero, bias one higher, one NaN code); krn models only the OCP layout.
FNUZ_FP8: Final = frozenset({Token.F8E4M3FNUZ, Token.F8E5M2FNUZ})
NATIVE_WHERE: Final[dict[int, str]] = {
    8: "sm_89 and later",
    6: "sm_100a, sm_101a and sm_120a",
    4: "sm_100a, sm_101a and sm_120a",
}  # by bit width; the text names the cards that have the instruction


def is_narrow(dtype: Dtype) -> bool:
    return dtype.is_float and dtype.bits <= 8


def label(dtype: Dtype) -> str:
    """The name of a float as the frontend spells it: `f8e4m3fn`, `f4e2m1`, `f32`."""
    return str(dtype)


def native(target: Target, dtype: Dtype) -> bool:
    """Whether the target converts this narrow float with one instruction; only NVIDIA has them, see `NATIVE_WHERE`."""
    if not isinstance(target, Nvidia) or (fmt := FORMATS.get(dtype)) is None:
        return False
    return target.sm >= 89 if fmt.bits == 8 else target.arch in FP4_ARCHES


def decode_name(fmt: FloatFormat) -> str:
    return f"_decode_{NAMES[fmt]}"


def encode_name(fmt: FloatFormat) -> str:
    return f"_encode_{NAMES[fmt]}"


@cache
def _levels(fmt: FloatFormat) -> tuple[np.ndarray, np.ndarray]:
    """The codes of the nonnegative finite values, ascending, and their values as float64."""
    codes = np.array([c for c in range(1 << (fmt.bits - 1)) if math.isfinite(fmt.table[c])], np.uint8)
    return codes, np.asarray(fmt.table, np.float64)[codes]


def maximum_code(fmt: FloatFormat) -> int:
    """The largest positive finite code: the saturation value of overflow and infinity."""
    return int(_levels(fmt)[0][-1])


def nan_code(fmt: FloatFormat) -> int:
    """The canonical NaN: the largest positive NaN code, FNUZ's only NaN code, or the maximum where there is no NaN."""
    half = 1 << (fmt.bits - 1)
    nans = [c for c in range(1 << fmt.bits) if math.isnan(fmt.table[c])]
    positive = [c for c in nans if c < half]
    if positive:
        return max(positive)
    return nans[0] if nans else maximum_code(fmt)


def reference_encode(fmt: FloatFormat, values: ArrayLike) -> np.ndarray:
    """The codes of float32 `values`: the nearest finite value, ties to the even code, saturating; NaN is `nan_code`."""
    x = np.asarray(values, np.float32)
    codes, levels = _levels(fmt)
    mids = (levels[:-1] + levels[1:]) / 2
    with np.errstate(invalid="ignore"):
        magnitude = np.abs(x).astype(np.float64)
    index = np.searchsorted(mids, magnitude, side="left")
    tie = magnitude == mids[np.minimum(index, len(mids) - 1)]
    index = index + (tie & (codes[index] % 2 == 1))
    signed = codes[index] | (np.signbit(x).astype(np.uint8) << (fmt.bits - 1))
    return np.where(np.isnan(x), nan_code(fmt), signed).astype(np.uint8)


def reference_decode(fmt: FloatFormat, codes: ArrayLike) -> np.ndarray:
    """The float32 value of each code: a table lookup, so every code has one meaning."""
    return np.asarray(fmt.table, np.float32)[np.asarray(codes, np.intp)]


def _decoder(fmt: FloatFormat, bits: str, floats: str) -> str:
    """`_decode_<name>(c)`: the float32 of code `c` from its fields, with the NaN codes tested first."""
    sign, top, man = fmt.bits - 1, (1 << fmt.exp_bits) - 1, fmt.man_bits
    lines = [
        f"@inline float {decode_name(fmt)}(unsigned c) {{",
        f"    unsigned s = (c >> {sign}) << 31, mag = c & {(1 << sign) - 1}u, ex = mag >> {man}, "
        f"m = mag & {(1 << man) - 1}u;",
    ]
    if fmt.special is Special.IEEE:
        lines.append(f"    if (ex == {top}u) return {floats.format('m ? 0x7fc00000u : s | 0x7f800000u')};")
    if fmt.special is Special.FN:
        lines.append(f"    if (mag == {(1 << sign) - 1}u) return {floats.format('0x7fc00000u')};")
    # An FNUZ format needs its one NaN code (`c == 1 << sign`) tested here before `mag == 0`; none is in FORMATS yet.
    lines.append(f"    if (mag == 0u) return {floats.format('s')};")
    subnormal = (1 - fmt.exp_bias - man + 127) << 23
    scale = floats.format(f"0x{subnormal:x}u")
    lines.append(f"    if (ex == 0u) {{ float v = (float)m * {scale}; return s ? -v : v; }}")
    normal = f"s | ((ex + {127 - fmt.exp_bias}u) << 23) | (m << {23 - man})"
    lines += [f"    return {floats.format(normal)};", "}"]
    return "\n".join(lines)


def _encoder(fmt: FloatFormat, bits: str) -> str:
    """`_encode_<name>(x)`: the code of float32 `x`, rounded in integer arithmetic on its bits."""
    sign, man, bias = fmt.bits - 1, fmt.man_bits, fmt.exp_bias
    shift, maximum = 23 - man, maximum_code(fmt)
    lines = [
        f"@inline unsigned {encode_name(fmt)}(float x) {{",
        f"    unsigned u = {bits.format('x')}, a = u & 0x7fffffffu, s = (u >> 31) << {sign};",
        f"    if (a > 0x7f800000u) return {nan_code(fmt)}u;",
        f"    if (a == 0x7f800000u) return s | {maximum}u;",
        "    if (a < 0x00800000u) return s;",
        "    unsigned q, mag, m = (a & 0x7fffffu) | 0x800000u;",
        "    int e = (int)(a >> 23) - 127;",
        f"    if (e >= {1 - bias}) {{",
        f"        q = m >> {shift};",
        f"        q += (m & {(1 << shift) - 1}u) > {1 << (shift - 1)}u"
        f" || ((m & {(1 << shift) - 1}u) == {1 << (shift - 1)}u && (q & 1u));",
        f"        mag = ((unsigned)(e + {bias - 1}) << {man}) + q;",
        "    } else {",
        f"        int sh = {24 - bias - man} - e;",
        "        if (sh > 24) return s;",
        "        unsigned rem = m & ((1u << sh) - 1u);",
        "        q = m >> sh;",
        "        q += rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (q & 1u));",
        "        mag = q;",
        "    }",
        f"    if (mag > {maximum}u) mag = {maximum}u;",
        "    return s | mag;",
        "}",
    ]
    return "\n".join(lines)


def bundle(fmt: FloatFormat, bits: str, floats: str) -> str:
    """The decoder and encoder of `fmt` as C text; `bits` and `floats` spell the bit casts with `{0}` as the operand."""
    return "\n".join([_decoder(fmt, bits, floats), _encoder(fmt, bits)])


def statements(fmt: FloatFormat, decoding: bool) -> int:
    """The statements of the decoder or the encoder: the instruction count a composed conversion warns with."""
    text = bundle(fmt, "{0}", "{0}").split("@inline")[1 if decoding else 2]
    return text.count(";")


def refuse(kernel: Kernel, target: Target) -> None:
    """Refuse what krn does not compute on a narrow float: its arithmetic, comparisons and reductions, and the
    conversions of FNUZ, of a format with no composition, and of fp8 on a target with FNUZ storage.

    The emulator calls this too, so every path that would compute or convert refuses the same way.
    """
    for found in expressions(kernel):
        match found:
            case Cast():
                _refuse_conversion(found, target)
            case Binary() | Unary() | Math() | Reduce() | Intrinsic():
                if operand := next((o for o in children(found) if is_narrow(o.dtype)), None):
                    _refuse_arithmetic(found, operand.dtype, target)


def refuse_product(dtype: Dtype, target: Target, span: Span) -> NoReturn:
    """Refuse a plain `@` on a narrow float with no unscaled matrix unit in krn: fp6 and fp4 take a block scale, and the
    FNUZ fp8 convention is not modeled in a product."""
    if dtype in FNUZ_DTYPES:
        reason = "krn does not model the FNUZ convention in a matrix product yet"
        instead = "convert the operands to f16 with .to and multiply"
    else:
        reason = "fp6 and fp4 matrix units take a block scale, which a plain product does not carry"
        instead = "convert the operands to f16 with .to and multiply, or use a block-scaled QMatrix on sm_120a"
    raise TargetError(f"`@` on {dtype}", target.arch, reason, instead, span=span)


def _refuse_arithmetic(found: Expr, dtype: Dtype, target: Target) -> NoReturn:
    raise TargetError(
        f"arithmetic on {dtype}",
        target.arch,
        "narrow floats are storage formats: the operation would add the byte codes, not the values",
        "convert with .to(T.f16) or T.f32 first, compute, then convert back",
        span=found.span,
    )


def _refuse_conversion(found: Cast, target: Target) -> None:
    for dtype in (found.operand.dtype, found.dtype):
        if not is_narrow(dtype):
            continue
        if dtype in FNUZ_DTYPES:
            raise TargetError(
                f"the fp8 conversion of {dtype}",
                target.arch,
                f"{dtype} is the FNUZ convention, which krn composes on no target yet",
                "convert through f32 to f8e4m3fn or f8e5m2, the OCP formats",
                span=found.span,
            )
        if dtype not in FORMATS:
            raise TargetError(
                f"the software conversion of {dtype}",
                target.arch,
                f"{dtype} is a narrow float with no software composition",
                f"convert through f32 to one of {', '.join(NAMES.values())}",
                span=found.span,
            )
        if dtype.bits == 8 and FNUZ_FP8 & target.storage:
            raise TargetError(
                f"the fp8 conversion of {dtype}",
                target.arch,
                f"{target.arch} stores fp8 as FNUZ, which krn does not model yet",
                "use gfx950 or compute in f16",
                span=found.span,
            )


def check(kernel: Kernel, target: Target) -> None:
    """Refuse what `refuse` refuses; warn once per kernel, kind and target where a GPU composes a conversion."""
    refuse(kernel, target)
    limits = current_settings().warn
    if target.vendor == "cpu" or not limits.warn_software_fallback:
        return
    for found in expressions(kernel):
        if not isinstance(found, Cast):
            continue
        source, dest = found.operand.dtype, found.dtype
        composed = [d for d in (source, dest) if is_narrow(d) and not native(target, d)]
        if not composed:
            continue
        size = sum(statements(FORMATS[d], decoding=d == source) for d in composed)
        if size < limits.software_fallback_min_instructions:
            continue
        where = " and ".join(sorted({NATIVE_WHERE[FORMATS[d].bits] for d in composed}))
        message = (
            f"convert {source} to {dest} runs as a {size}-instruction software sequence on "
            f"{target.arch}; native on NVIDIA GPUs at {where}"
        )
        warn_once(
            kernel.name,
            target.arch,
            f"software-fallback:{source}-to-{dest}",
            message,
            PerformanceWarning,
            found.span,
            kernel.span,
        )
