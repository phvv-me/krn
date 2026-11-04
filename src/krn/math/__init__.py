"""Typed kernel math: IEEE results and NaNs for domain errors, without Python exceptions."""

import math as _math
from typing import overload

from krn.frontend.iteration import sum
from krn.frontend.kernel import kernel
from krn.frontend.memory import Vector
from krn.frontend.numerics import (
    FloatingPoint,
    Integer,
    Packed,
    Scalar,
    i8,
    i8x4,
    i16,
    i16x2,
    todo,
    u8,
    u8x4,
    u16,
    u16x2,
)
from krn.frontend.scopes import cbrt, fma, trunc
from krn.frontend.types import f32, u32

pi: float = _math.pi
e: float = _math.e
tau: float = _math.tau

__all__ = [
    "abs_diff",
    "add_sat",
    "avg",
    "cbrt",
    "ceil",
    "clip",
    "copysign",
    "cos",
    "div",
    "e",
    "exp",
    "exp2",
    "exp10",
    "floor",
    "fma",
    "isfinite",
    "isnan",
    "ldexp",
    "log",
    "log2",
    "log10",
    "maximum",
    "pack",
    "pi",
    "pow",
    "reciprocal",
    "rsqrt",
    "sin",
    "sqrt",
    "sub_sat",
    "tan",
    "tanh",
    "tau",
    "trunc",
    "unpack",
]


@overload
def add_sat(a: i8x4, b: i8x4) -> i8x4:
    raise NotImplementedError


@overload
def add_sat(a: u8x4, b: u8x4) -> u8x4:
    raise NotImplementedError


@overload
def add_sat(a: i16x2, b: i16x2) -> i16x2:
    raise NotImplementedError


@overload
def add_sat(a: u16x2, b: u16x2) -> u16x2:
    raise NotImplementedError


@overload
def add_sat(a: Integer, b: Integer) -> Integer:
    raise NotImplementedError


def add_sat(a: Scalar, b: Scalar) -> Scalar:
    """Sum clamped to the range of the type, per lane for packed words; signedness comes from the element type.

    .. versionadded:: 0.0.1

    Supported on (generated):
        NVIDIA: sm_80+ composed (13 instructions), 8 of 16 operand types native or refused
        Apple: apple_m1+ composed (13 instructions), 8 of 16 operand types native or refused
        AMD: gfx942+ composed (136 instructions), 4 of 16 operand types refused
        CPU: avx2+ composed (136 instructions), 4 of 16 operand types refused
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


@overload
def abs_diff(a: i8x4, b: i8x4) -> u8x4:
    raise NotImplementedError


@overload
def abs_diff(a: u8x4, b: u8x4) -> u8x4:
    raise NotImplementedError


@overload
def abs_diff(a: i16x2, b: i16x2) -> u16x2:
    raise NotImplementedError


@overload
def abs_diff(a: u16x2, b: u16x2) -> u16x2:
    raise NotImplementedError


@overload
def abs_diff(a: Integer, b: Integer) -> Integer:
    raise NotImplementedError


@overload
def abs_diff[F: FloatingPoint](a: F, b: F) -> F:
    raise NotImplementedError


def abs_diff(a: Scalar, b: Scalar) -> Scalar:
    """Magnitude of a - b, exact: a signed pair gives an unsigned result of the same width, and packed signed lanes
    give unsigned lanes.

    .. versionadded:: 0.0.1

    Supported on (generated):
        NVIDIA: sm_80+ composed (8 instructions), 4 of 16 operand types native
        Apple: apple_m1+ composed (8 instructions), 5 of 16 operand types native or refused
        AMD: gfx942+ composed (136 instructions)
        CPU: avx2+ composed (136 instructions)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


@overload
def sub_sat(a: i8x4, b: i8x4) -> i8x4:
    raise NotImplementedError


@overload
def sub_sat(a: u8x4, b: u8x4) -> u8x4:
    raise NotImplementedError


@overload
def sub_sat(a: i16x2, b: i16x2) -> i16x2:
    raise NotImplementedError


@overload
def sub_sat(a: u16x2, b: u16x2) -> u16x2:
    raise NotImplementedError


@overload
def sub_sat(a: Integer, b: Integer) -> Integer:
    raise NotImplementedError


def sub_sat(a: Scalar, b: Scalar) -> Scalar:
    """Difference clamped to the type's range, per lane for packed words; signedness comes from the element type.

    .. versionadded:: 0.0.1

    Supported on (generated):
        NVIDIA: sm_80+ composed (11 instructions), 8 of 16 operand types native or refused
        Apple: apple_m1+ composed (11 instructions), 8 of 16 operand types native or refused
        AMD: gfx942+ composed (136 instructions), 4 of 16 operand types refused
        CPU: avx2+ composed (136 instructions), 4 of 16 operand types refused
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


@overload
def avg(a: i8x4, b: i8x4) -> i8x4:
    raise NotImplementedError


@overload
def avg(a: u8x4, b: u8x4) -> u8x4:
    raise NotImplementedError


@overload
def avg(a: i16x2, b: i16x2) -> i16x2:
    raise NotImplementedError


@overload
def avg(a: u16x2, b: u16x2) -> u16x2:
    raise NotImplementedError


@overload
def avg(a: Integer, b: Integer) -> Integer:
    raise NotImplementedError


def avg(a: Scalar, b: Scalar) -> Scalar:
    """Mean of a and b rounded half up, floor((a + b + 1) / 2), so no intermediate sum can overflow. Per lane for packed
    words; a signed lane rounds half up too, where NVIDIA's signed SIMD average rounds ties away from zero.

    .. versionadded:: 0.0.1

    Supported on (generated):
        NVIDIA: sm_80+ composed (8 instructions), 6 of 16 operand types native or refused
        Apple: apple_m1+ composed (6 instructions), 8 of 16 operand types native or refused
        AMD: gfx942+ composed (136 instructions), 4 of 16 operand types refused
        CPU: avx2+ composed (136 instructions), 4 of 16 operand types refused
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


@overload
def maximum[S: Scalar](a: S, b: S) -> S:
    raise NotImplementedError


@overload
def maximum[S: Scalar](a: S, b: S, c: S) -> S:
    raise NotImplementedError


def maximum[S: Scalar](a: S, b: S, c: S | None = None) -> S:
    """Largest of two or three operands: per lane for packed words, NaN-propagating for floats.

    Supported on (generated):
        NVIDIA: sm_80+ native, 8 of 16 operand types composed
        Apple: apple_m1+ composed (2 instructions), 8 of 16 operand types native or refused
        AMD: gfx942+ composed (136 instructions), 4 of 16 operand types native
        CPU: avx2+ composed (136 instructions), 4 of 16 operand types native
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


@overload
def pack(a: i8, b: i8, c: i8, d: i8) -> i8x4:
    raise NotImplementedError


@overload
def pack(a: u8, b: u8, c: u8, d: u8) -> u8x4:
    raise NotImplementedError


@overload
def pack(a: i16, b: i16) -> i16x2:
    raise NotImplementedError


@overload
def pack(a: u16, b: u16) -> u16x2:
    raise NotImplementedError


def pack(a: Integer, b: Integer, c: Integer | None = None, d: Integer | None = None) -> Packed:
    """Four 8-bit lanes, or two 16-bit lanes, into one word, the first lane lowest.

    Supported on (generated):
        Native on every target
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


@overload
def unpack(x: i8x4) -> tuple[i8, i8, i8, i8]:
    raise NotImplementedError


@overload
def unpack(x: u8x4) -> tuple[u8, u8, u8, u8]:
    raise NotImplementedError


@overload
def unpack(x: i16x2) -> tuple[i16, i16]:
    raise NotImplementedError


@overload
def unpack(x: u16x2) -> tuple[u16, u16]:
    raise NotImplementedError


def unpack(x: Packed) -> tuple[Integer, ...]:
    """The lanes of a packed word, the first lane lowest; `x.lanes` reads the same lanes.

    Supported on (generated):
        Native on every target
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def sqrt[F: FloatingPoint](x: F) -> F:
    """IEEE square root; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (sqrt needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def exp[F: FloatingPoint](x: F) -> F:
    """IEEE exponential; overflow returns infinity, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (exp needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def exp2[F: FloatingPoint](x: F) -> F:
    """IEEE base-two exponential; overflow returns infinity, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (exp2 needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def log[F: FloatingPoint](x: F) -> F:
    """IEEE natural logarithm; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (log needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def log2[F: FloatingPoint](x: F) -> F:
    """IEEE base-two logarithm; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (log2 needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def log10[F: FloatingPoint](x: F) -> F:
    """IEEE base-ten logarithm; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (log10 needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def sin[F: FloatingPoint](x: F) -> F:
    """IEEE sine; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (sin needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def cos[F: FloatingPoint](x: F) -> F:
    """IEEE cosine; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (cos needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def tanh[F: FloatingPoint](x: F) -> F:
    """IEEE hyperbolic tangent; NaNs propagate without an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (tanh needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def tan[F: FloatingPoint](x: F) -> F:
    """IEEE tangent of x in radians, correctly rounded where libm is; NaNs propagate without an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (`libm.tan` accepts f32, f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def exp10[F: FloatingPoint](x: F) -> F:
    """IEEE base-ten exponential; overflow returns infinity, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (exp10 needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def pow[F: FloatingPoint](x: F, y: F | float) -> F:
    """IEEE power x to the y through libm; a domain error returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (pow needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def reciprocal[F: FloatingPoint](x: F) -> F:
    """IEEE 1 / x, correctly rounded; 1 / 0 returns infinity, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (reciprocal needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def rsqrt[F: FloatingPoint](x: F) -> F:
    """1 / sqrt(x) as two correctly rounded operations: one rounding of the square root, one of the division.

    A negative x returns NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (rsqrt needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def div[F: FloatingPoint](x: F, y: F | float) -> F:
    """IEEE division x / y, correctly rounded; a zero divisor returns infinity or NaN, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (div needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def floor[F: FloatingPoint](x: F) -> F:
    """Round toward negative infinity, returning the float type, never an int or an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (floor needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def ceil[F: FloatingPoint](x: F) -> F:
    """Round toward positive infinity, returning the float type, never an int or an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (ceil needs f32 or f64, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def copysign[F: FloatingPoint](x: F, y: F | float) -> F:
    """IEEE magnitude of x with the sign of y, including signed zero; never raises.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (copysign needs matching f32 or f64 scalars, got bf16 and bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def isnan[F: FloatingPoint](x: F) -> bool:
    """Test IEEE NaN without an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (math predicate or intrinsic needs an f32 or f64 scalar, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def isfinite[F: FloatingPoint](x: F) -> bool:
    """Test IEEE finiteness without an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (math predicate or intrinsic needs an f32 or f64 scalar, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


def ldexp[F: FloatingPoint](x: F, exponent: Integer | int) -> F:
    """IEEE scaling by a power of two; overflow returns infinity, never an exception.

    Supported on (generated):
        NVIDIA: sm_80+ native (bf16 refused; f16 refused)
        Apple: apple_m1+ refused (math predicate or intrinsic needs an f32 or f64 scalar, got bf16) (f32 native)
        AMD: gfx942+ native (bf16 refused; f16 refused)
        CPU: avx2+ native (bf16 refused; f16 refused)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7; AMD from
        gfx942: gfx942; CPU from avx2: AVX2.

    """
    return todo()


@kernel
def clip(value: f32, minimum: f32, maximum: f32) -> f32:
    """Clip to ordered bounds; kernel min/max propagate NaN in either operand. clip(x, 0.0, 1.0) is saturation."""
    return min(max(value, minimum), maximum)


@kernel
def in_range(value: f32, minimum: f32, maximum: f32) -> bool:
    """Include the lower bound and exclude the upper bound."""
    return minimum <= value and value < maximum


@kernel
def lerp(start: f32, stop: f32, fraction: f32) -> f32:
    """Linear interpolation, permitting extrapolation when the fraction is outside [0, 1]."""
    return start + (stop - start) * fraction


@kernel
def inverse_lerp(start: f32, stop: f32, value: f32) -> f32:
    """The interpolation fraction, or zero for coincident endpoints."""
    return (value - start) / (stop - start) if start != stop else f32(0.0)


@kernel
def mean(x: Vector[f32], start: u32 | int, stop: u32 | int) -> f32:
    """An empty row gives NaN (0/0), matching es-toolkit."""
    return sum(x[i] for i in range(start, stop)) / f32(stop - start)
