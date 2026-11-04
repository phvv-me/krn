"""Vendor accuracy contracts for explicit fast calls; no translation-unit math modes.

Sources: CUDA Programming Guide intrinsic functions, PTX ISA 9.7.3,
Metal Shading Language Numerical Compliance, HIP math API intrinsic tables.
Absolute and relative bounds are retained when the vendor specifies no fixed ULP bound.
"""

from .targets import Target

ACCURACY: dict[str, dict[str, str]] = {
    "sqrt": {
        "nvidia": "CUDA: <= 3 ulp for positive finite x (relative error <= 2**-23).",
        "apple": "Metal: x * fast::rsqrt(x), conservative <= 5 ulp for positive normal x.",
        "amd": "not yet verified on HIP; HIP docs: __fsqrt_rn, <= 1 ulp on [0, 100].",
    },
    "rsqrt": {
        "nvidia": "CUDA: <= 3 ulp for positive finite x (relative error <= 2**-22.9).",
        "apple": "Metal: <= 2 ulp for positive normal x.",
        "amd": "not yet verified on HIP; HIP docs: __frsqrt_rn, <= 1 ulp on [0.01, 100].",
    },
    "reciprocal": {
        "nvidia": "CUDA: <= 1 ulp across the input range.",
        "apple": "Metal: fast::divide(1, x), <= 2.5 ulp for |x| in [2**-126, 2**126] and normal results.",
        "amd": "not yet verified on HIP; HIP docs: __frcp_rn, correctly rounded (<= 0.5 ulp) on [-100, 100].",
    },
    "div": {
        "nvidia": "CUDA: <= 2 ulp for |y| in [2**-126, 2**126].",
        "apple": "Metal: <= 2.5 ulp for |y| in [2**-126, 2**126] and normal results.",
        "amd": "not yet verified on HIP; HIP docs: __fdiv_rn, correctly rounded (<= 0.5 ulp) on [-100, 100] "
        "for nonzero y.",
    },
    "exp": {
        "nvidia": "CUDA: <= 2 + floor(abs(1.173*x)) ulp for finite normal results.",
        "apple": "Metal: <= 3 + floor(abs(2*x)) ulp for finite normal results.",
        "amd": "not yet verified on HIP; HIP docs: <= 6 ulp on [-10, 10].",
    },
    "exp2": {
        "nvidia": "CUDA: <= 2 ulp across the input range.",
        "apple": "Metal: <= 3 + floor(abs(2*x)) ulp for finite normal results.",
        "amd": "not yet verified on HIP; HIP docs: precise exp2f, <= 1 ulp on [-10, 10].",
    },
    "exp10": {
        "nvidia": "CUDA: <= 2 + floor(abs(2.97*x)) ulp for finite normal results.",
        "apple": "Metal: exp2(x*log2(10)); composed error, no fixed ulp bound.",
        "amd": "not yet verified on HIP; HIP docs: <= 18 ulp on [-4, 4].",
    },
    "log": {
        "nvidia": "CUDA: absolute error <= 2**-21.41 on [0.5, 2], <= 3 ulp otherwise for x > 0.",
        "apple": "Metal: absolute error <= 2**-21 on [0.5, 2], <= 3 ulp otherwise for x > 0.",
        "amd": "not yet verified on HIP; HIP docs: <= 2 ulp on [1e-6, 1e6].",
    },
    "log2": {
        "nvidia": "CUDA: absolute error <= 2**-22 on [0.5, 2], <= 2 ulp otherwise for x > 0.",
        "apple": "Metal: absolute error <= 2**-22 on [0.5, 2], <= 2 ulp otherwise for x > 0.",
        "amd": "not yet verified on HIP; HIP docs: <= 1 ulp on [1e-6, 1e6].",
    },
    "log10": {
        "nvidia": "CUDA: absolute error <= 2**-24 on [0.5, 2], <= 3 ulp otherwise for x > 0.",
        "apple": "Metal: log2(x)*log10(2); composed error, no fixed ulp bound.",
        "amd": "not yet verified on HIP; HIP docs: <= 2 ulp on [1e-6, 1e6].",
    },
    "sin": {
        "nvidia": "CUDA: absolute error <= 2**-21.41 on [-pi, pi]; no fixed ulp bound near zeros.",
        "apple": "Metal: absolute error <= 2**-13 on [-pi, pi]; no fixed ulp bound near zeros.",
        "amd": "not yet verified on HIP; HIP docs: <= 18 ulp on [-pi, pi].",
    },
    "cos": {
        "nvidia": "CUDA: absolute error <= 2**-21.41 on [-pi, pi]; no fixed ulp bound near zeros.",
        "apple": "Metal: absolute error <= 2**-13 on [-pi, pi]; no fixed ulp bound near zeros.",
        "amd": "not yet verified on HIP; HIP docs: <= 4 ulp on [-pi, pi].",
    },
    "tan": {
        "nvidia": "CUDA: sin(x)/cos(x); composed error, no fixed ulp bound near poles.",
        "apple": "Metal: sin(x)/cos(x); composed error, no fixed ulp bound near poles.",
        "amd": "not yet verified on HIP; HIP docs: <= 1 ulp on [-1.47*pi, 1.47*pi].",
    },
    "tanh": {
        "nvidia": "CUDA: relative error <= 2**-11 (<= 8193 ulp for normal results), all finite x.",
        "apple": "Metal: (exp(2*x)-1)/(exp(2*x)+1); composed error, no fixed ulp bound.",
        "amd": "not yet verified on HIP; HIP docs: precise tanhf, <= 2 ulp on [-5, 5].",
    },
    "pow": {
        "nvidia": "CUDA: exp2(y*log2(x)); composed error, no fixed ulp bound; x > 0.",
        "apple": "Metal: exp2(y*log2(x)); composed error, no fixed ulp bound; x > 0.",
        "amd": "not yet verified on HIP; HIP docs: <= 1 ulp for x in [-4, 4], y in [-2, 2] in the real domain.",
    },
}


def accuracy(name: str, target: Target) -> str:
    if target.vendor == "cpu":
        return "precise C libm (platform accuracy), IEEE division and sqrt <= 0.5 ulp in their domains"
    return ACCURACY[name][target.vendor]


def relaxation(name: str, target: Target) -> str:
    if target.vendor == "nvidia" and name == "div":
        return "for 2**126 < |y| < 2**128, finite x gives zero and infinite x gives NaN"
    if target.vendor == "apple":
        return "fast functions may flush denormals and assume finite inputs; no surrounding arithmetic is relaxed"
    if target.vendor == "nvidia" and name in {"exp", "exp10", "log", "log2", "log10", "sin", "cos", "tan", "pow"}:
        return "CUDA __ intrinsics may flush float32 denormal inputs/results; no surrounding arithmetic is relaxed"
    return "no additional relaxation of surrounding arithmetic or compiler denormal mode"
