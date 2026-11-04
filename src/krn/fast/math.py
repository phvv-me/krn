"""Precision-losing math, each name paired with the strict `krn.math` function of the same name and signature.

Contracts follow CUDA's intrinsic accuracy table and PTX ISA, Metal Shading Language section Numerical Compliance,
and the HIP math API. Bounds apply only in the stated ranges. The emulator and C backend use precise math. A fused
multiply-add gives up no precision, so it is `krn.math.fma`. See language/numerics in the docs.
"""

from krn.frontend.numerics import FloatingPoint, todo


def sqrt[F: FloatingPoint](x: F) -> F:
    """CUDA: <= 3 ulp for positive finite x (relative error <= 2**-23).
    Metal: x * fast::rsqrt(x), conservative <= 5 ulp for positive normal x.
    HIP: __fsqrt_rn, <= 1 ulp on [0, 100].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.sqrt`.

    Supported on (generated):
        NVIDIA: sm_80+ composed (4 instructions)
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: Apple from apple_m1: Apple7.

    """
    return todo()


def rsqrt[F: FloatingPoint](x: F) -> F:
    """CUDA: <= 3 ulp for positive finite x (relative error <= 2**-22.9).
    Metal: <= 2 ulp for positive normal x.
    HIP: __frsqrt_rn, <= 1 ulp on [0.01, 100].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.rsqrt`.

    Supported on (generated):
        NVIDIA: sm_80+ composed (4 instructions)
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: Apple from apple_m1: Apple7.

    """
    return todo()


def reciprocal[F: FloatingPoint](x: F) -> F:
    """CUDA: <= 1 ulp across the input range.
    Metal: fast::divide(1, x), <= 2.5 ulp for |x| in [2**-126, 2**126] and normal results.
    HIP: __frcp_rn, correctly rounded (<= 0.5 ulp) on [-100, 100].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.reciprocal`.

    Supported on (generated):
        NVIDIA: sm_80+ composed (4 instructions)
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: Apple from apple_m1: Apple7.

    """
    return todo()


def div[F: FloatingPoint](x: F, y: F | float) -> F:
    """CUDA: <= 2 ulp for |y| in [2**-126, 2**126].
    Metal: <= 2.5 ulp for |y| in [2**-126, 2**126] and normal results.
    HIP: __fdiv_rn, correctly rounded (<= 0.5 ulp) on [-100, 100] for nonzero y.
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.div`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def exp[F: FloatingPoint](x: F) -> F:
    """CUDA: <= 2 + floor(abs(1.173*x)) ulp for finite normal results.
    Metal: <= 3 + floor(abs(2*x)) ulp for finite normal results.
    HIP: <= 6 ulp on [-10, 10].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.exp`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def exp2[F: FloatingPoint](x: F) -> F:
    """CUDA: <= 2 ulp across the input range.
    Metal: <= 3 + floor(abs(2*x)) ulp for finite normal results.
    HIP: precise exp2f, <= 1 ulp on [-10, 10].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.exp2`.

    Supported on (generated):
        NVIDIA: sm_80+ composed (4 instructions)
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: Apple from apple_m1: Apple7.

    """
    return todo()


def exp10[F: FloatingPoint](x: F) -> F:
    """CUDA: <= 2 + floor(abs(2.97*x)) ulp for finite normal results.
    Metal: exp2(x*log2(10)); composed error, no fixed ulp bound.
    HIP: <= 18 ulp on [-4, 4].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.exp10`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def log[F: FloatingPoint](x: F) -> F:
    """CUDA: absolute error <= 2**-21.41 on [0.5, 2], <= 3 ulp otherwise for x > 0.
    Metal: absolute error <= 2**-21 on [0.5, 2], <= 3 ulp otherwise for x > 0.
    HIP: <= 2 ulp on [1e-6, 1e6].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.log`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def log2[F: FloatingPoint](x: F) -> F:
    """CUDA: absolute error <= 2**-22 on [0.5, 2], <= 2 ulp otherwise for x > 0.
    Metal: absolute error <= 2**-22 on [0.5, 2], <= 2 ulp otherwise for x > 0.
    HIP: <= 1 ulp on [1e-6, 1e6].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.log2`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def log10[F: FloatingPoint](x: F) -> F:
    """CUDA: absolute error <= 2**-24 on [0.5, 2], <= 3 ulp otherwise for x > 0.
    Metal: log2(x)*log10(2); composed error, no fixed ulp bound.
    HIP: <= 2 ulp on [1e-6, 1e6].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.log10`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def sin[F: FloatingPoint](x: F) -> F:
    """CUDA: absolute error <= 2**-21.41 on [-pi, pi]; no fixed ulp bound near zeros.
    Metal: absolute error <= 2**-13 on [-pi, pi]; no fixed ulp bound near zeros.
    HIP: <= 18 ulp on [-pi, pi].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.sin`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def cos[F: FloatingPoint](x: F) -> F:
    """CUDA: absolute error <= 2**-21.41 on [-pi, pi]; no fixed ulp bound near zeros.
    Metal: absolute error <= 2**-13 on [-pi, pi]; no fixed ulp bound near zeros.
    HIP: <= 4 ulp on [-pi, pi].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.cos`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def tan[F: FloatingPoint](x: F) -> F:
    """CUDA: sin(x)/cos(x); composed error, no fixed ulp bound near poles.
    Metal: sin(x)/cos(x); composed error, no fixed ulp bound near poles.
    HIP: <= 1 ulp on [-1.47*pi, 1.47*pi].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.tan`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


def tanh[F: FloatingPoint](x: F) -> F:
    """CUDA: relative error <= 2**-11 (<= 8193 ulp for normal results), all finite x.
    Metal: (exp(2*x)-1)/(exp(2*x)+1); composed error, no fixed ulp bound.
    HIP: precise tanhf, <= 2 ulp on [-5, 5].
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.tanh`.

    Supported on (generated):
        NVIDIA: sm_80+ composed (4 instructions)
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: Apple from apple_m1: Apple7.

    """
    return todo()


def pow[F: FloatingPoint](x: F, y: F | float) -> F:
    """CUDA: exp2(y*log2(x)); composed error, no fixed ulp bound; x > 0.
    Metal: exp2(y*log2(x)); composed error, no fixed ulp bound; x > 0.
    HIP: <= 1 ulp for x in [-4, 4], y in [-2, 2] in the real domain.
    C and emulator: precise math; libm accuracy is platform dependent.
    No surrounding operation is relaxed; GPU fast forms may have restricted special values.
    Strict counterpart: `krn.math.pow`.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ native
        AMD: gfx942+ refused (no HIP lowering of this approximate function has been verified...)
        CPU: avx2+ composed (1 instruction)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

    """
    return todo()


__all__ = [
    "cos",
    "div",
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "pow",
    "reciprocal",
    "rsqrt",
    "sin",
    "sqrt",
    "tan",
    "tanh",
]
