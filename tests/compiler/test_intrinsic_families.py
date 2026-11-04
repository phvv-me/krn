"""Independent scalar oracles shared by execution and optional vendor compilation tests."""

import inspect
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, catalog
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.capture import INTRINSICS, IntrinsicSpec
from krn.compiler.dtypes import U32
from krn.compiler.errors import TargetError
from krn.compiler.ir import Kernel
from krn.emulator.emulate import build as emulate
from krn.facts.intrinsics import (
    async_copy,
    atomic_block,
    atomic_system,
    bfloat162,
    bits,
    bulk,
    cache,
    dsmem,
    fast,
    half2,
    libm,
    mbarrier,
    rounded,
    simd,
)
from krn.facts.targets import Feature, Target
from krn.frontend.atomic import AtomicOp, Strategy, plan
from krn.runtime.cpu import CLANG, CPUProgram, Value
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.helpers.devices import CUDARun, MetalRun
from tests.helpers.text import descriptor, raises_literal
from tests.helpers.text import words as source_words

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.metal import MetalDevice

GPU = Target.of("sm_90a")
DTYPES = {
    "u32": "uint32",
    "i32": "int32",
    "u64": "uint64",
    "i64": "int64",
    "f32": "float32",
    "f64": "float64",
}
SIZE = 128
BOUNDARIES = (0, 1, 31, 32, 63, 64, 255, 256)
CLANG_HAS_AMDGPU = (
    "amdgcn"
    in subprocess.run([CLANG[0], "--print-targets"], capture_output=True, text=True, check=False, timeout=10).stdout
)
ERFCX_NODES, ERFCX_WEIGHTS = np.polynomial.laguerre.laggauss(32)


def integer_oracle(name: str, values: tuple[int, ...], width: int, signed: bool) -> int:
    """Build selected bits or unpack lanes rather than using the catalog's formulas."""
    a = values[0] % (1 << width)
    if name in ("bfe", "bfi"):
        p, n = values[-2] % 256, values[-1] % 256
        if name == "bfi":
            output = list(f"{values[1]:0{width}b}"[::-1])
            source = f"{a:0{width}b}"[::-1]
            for i in range(p, min(width, p + n)):
                output[i] = source[i - p]
            return int("".join(output[::-1]), 2)
        source = [(a >> i) & 1 for i in range(width)]
        fill = source[min(p + n - 1, width - 1)] if n and signed else 0
        return sum((source[p + i] if i < n and p + i < width else fill) << i for i in range(width))
    if name == "byte_perm":
        source = a.to_bytes(4, "little") + values[1].to_bytes(4, "little")
        controls = [(values[2] >> (4 * i)) & 15 for i in range(4)]
        return int.from_bytes(
            bytes((255 if source[s % 8] >= 128 else 0) if s >= 8 else source[s] for s in controls),
            "little",
        )
    if name.startswith("funnelshift"):
        word = f"{values[1]:032b}{a:032b}"
        shift = min(values[2], 32) if name.endswith("c") else values[2] % 32
        word = (word + "0" * shift)[shift:] if "_l" in name else ("0" * shift + word)[:64]
        return int(word[:32] if "_l" in name else word[-32:], 2)
    if name == "lop3":
        result = 0
        for pattern in range(8):
            if values[3] & (1 << pattern):
                term = (1 << 32) - 1
                for i, operand in enumerate(values[:3]):
                    term &= operand if pattern & (4 >> i) else ~operand
                result |= term
        return result
    left = int(a).to_bytes(4, "little")
    right = int(values[1] % (1 << 32)).to_bytes(4, "little")
    if name == "dp4a":
        products = [
            int.from_bytes(left[i : i + 1], "little", signed=signed)
            * int.from_bytes(right[i : i + 1], "little", signed=signed)
            for i in range(4)
        ]
    else:
        offset = 2 if name.endswith("hi") else 0
        products = [
            int.from_bytes(left[2 * i : 2 * i + 2], "little", signed=signed)
            * int.from_bytes(right[offset + i : offset + i + 1], "little", signed=signed)
            for i in range(2)
        ]
    return (sum(products) + values[2]) % (1 << 32)


def simd_oracle(name: str, a: int, b: int = 0) -> int:
    lanes = int(name[-1])
    op, width = name[:-1], 32 // lanes
    signed = op in ("vneg", "vabs") or (op.endswith("s") and op not in ("vaddus", "vsubus"))
    dtype = np.dtype(f"{'int' if signed else 'uint'}{width}")
    x = np.frombuffer(a.to_bytes(4, "little"), dtype=dtype).astype(np.int64)
    y = np.frombuffer(b.to_bytes(4, "little"), dtype=dtype).astype(np.int64)
    if op.startswith(("vcmp", "vset")):
        relation = op[4:].rstrip("su")
        comparisons = {
            "eq": np.equal,
            "ne": np.not_equal,
            "ge": np.greater_equal,
            "gt": np.greater,
            "le": np.less_equal,
            "lt": np.less,
        }
        z = comparisons[relation](x, y).astype(np.int64) * ((1 << width) - 1 if op.startswith("vcmp") else 1)
    elif op.startswith("vadd"):
        z = x + y
    elif op.startswith("vsub"):
        z = x - y
    elif op.startswith("vabsdiff"):
        z = np.abs(x - y)
    elif op.startswith("vmax"):
        z = np.maximum(x, y)
    elif op.startswith("vmin"):
        z = np.minimum(x, y)
    elif op.startswith("vavg"):
        means = (x + y) / 2
        z = (np.copysign(np.ceil(np.abs(means)), means) if signed else np.ceil(means)).astype(np.int64)
    elif op == "vhaddu":
        z = (x + y) // 2
    elif op.startswith("vneg"):
        z = -x
    else:
        z = np.abs(x)
    if op.endswith("ss"):
        z = np.clip(z, -(1 << (width - 1)), (1 << (width - 1)) - 1)
    elif op.endswith("us"):
        z = np.clip(z, 0, (1 << width) - 1)
    return int.from_bytes((z % (1 << width)).astype(f"uint{width}").tobytes(), "little")


def decode(word: int, bf: bool) -> float:
    raw = np.array([word & 65535], np.uint32)
    return float((raw << 16).view(np.float32)[0]) if bf else float(raw.astype(np.uint16).view(np.float16)[0])


def encode(value: float, bf: bool) -> int:
    if not bf:
        with np.errstate(over="ignore", invalid="ignore"):
            return int(np.array([value], np.float16).view(np.uint16)[0])
    if not math.isfinite(value):
        return (32768 if math.copysign(1, value) < 0 else 0) | (32704 if math.isnan(value) else 32640)
    # Quantize binary64 to a bfloat16 lattice before using a float32 carrier.
    magnitude = abs(value)
    quantum = max(float(np.spacing(np.float64(magnitude))) * 2**45, 2.0**-133)
    rounded = float(np.rint(magnitude / quantum)) * quantum if magnitude else 0.0
    with np.errstate(over="ignore"):
        raw = int(np.array([math.copysign(rounded, value)], np.float32).view(np.uint32)[0])
    return raw >> 16


def packed_oracle(name: str, values: tuple[int | float, ...], bf: bool) -> int | float:
    if name == "pack":
        return encode(float(values[0]), bf) | encode(float(values[1]), bf) << 16
    if name in ("low", "high"):
        return decode(int(values[0]) >> (16 if name == "high" else 0), bf)
    output = []
    for lane in range(2):
        x = [decode(int(v) >> (16 * lane), bf) for v in values]
        operations: dict[str, Callable[[list[float]], float]] = {
            "add": lambda v: v[0] + v[1],
            "sub": lambda v: v[0] - v[1],
            "mul": lambda v: v[0] * v[1],
            "fma": lambda v: math.fma(v[0], v[1], v[2]),
            "min": lambda v: float(np.fmin(v[0], v[1])),
            "max": lambda v: float(np.fmax(v[0], v[1])),
            "neg": lambda v: -v[0],
            "abs": lambda v: abs(v[0]),
        }
        value = operations[name](x)
        if name in ("min", "max") and x[0] == x[1] == 0:
            signs = np.signbit(x)
            value = -0.0 if (np.any(signs) if name == "min" else np.all(signs)) else 0.0
        output.append(encode(value, bf))
    return output[0] | output[1] << 16


def math_oracle(name: str, values: tuple[int | float, ...], dtype: str) -> tuple[int | float, ...]:
    x = float(values[0])
    if name in ("erfinv", "erfcinv"):
        lo, hi = -8.0, 8.0
        for _ in range(80):
            middle = (lo + hi) / 2
            below = math.erf(middle) < x if name == "erfinv" else math.erfc(middle) > x
            lo, hi = (middle, hi) if below else (lo, middle)
        return ((lo + hi) / 2,)
    if name == "erfcx":
        if x >= 25:
            integral = np.dot(ERFCX_WEIGHTS, np.exp(-((ERFCX_NODES / (2 * x)) ** 2)))
            return (float(integral) / (math.sqrt(math.pi) * x),)
        return (math.exp(x * x) * math.erfc(x),)
    if name == "reciprocal":
        return (1 / x,)
    if name == "rsqrt":
        return (float(1 / np.sqrt(x)),)
    if name == "sincos":
        return (float(np.sin(x)), float(np.cos(x)))
    if name == "frexp":
        mantissa, exponent = np.frexp(x)
        return float(mantissa), int(exponent)
    if name == "nextafter":
        return (float(np.nextafter(np.array(x, DTYPES[dtype]), np.array(values[1], DTYPES[dtype]))),)
    if name == "exp10":
        return (10.0**x,)
    if name == "div":
        return (x / float(values[1]),)
    if name == "ldexp":
        return (float(np.ldexp(x, int(values[1]))),)
    if name in ("erf", "erfc"):
        return ((math.erf(x) if name == "erf" else math.erfc(x)),)
    functions: dict[str, Callable[..., np.floating | np.ndarray]] = {
        "sqrt": np.sqrt,
        "exp2": np.exp2,
        "tanh": np.tanh,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "atan2": np.arctan2,
        "hypot": np.hypot,
        "fmod": np.fmod,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "pow": np.power,
        "copysign": np.copysign,
    }
    return (float(functions[name](*values)),)


def check_fast_accuracy(
    name: str, actual: np.ndarray, expected: np.ndarray, inputs: list[np.ndarray], target: Target
) -> None:
    """Vendor bounds against the precise emulator, including forward error for composed functions."""
    x = inputs[0].astype(np.float64)
    spacing = np.abs(np.spacing(expected)).astype(np.float64)
    ulps = np.full(x.shape, 2.0)  # precise host libm can differ from binary64-then-f32 by an ulp
    absolute = np.zeros_like(x)
    if target.vendor == "nvidia":
        ulps.fill({"reciprocal": 1, "sqrt": 3, "rsqrt": 3, "div": 2, "exp2": 2, "tanh": 8193}.get(name, 3))
        if name in {"exp", "exp10"}:
            ulps = 2 + np.floor(np.abs(x * (1.173 if name == "exp" else 2.97)))
        if name in {"sin", "cos"}:
            absolute.fill(2**-21.41)
        if name in {"log", "log2", "log10"}:
            small = (x >= 0.5) & (x <= 2)
            absolute[small] = {"log": 2**-21.41, "log2": 2**-22, "log10": 2**-24}[name]
    elif target.vendor == "amd":
        ulps.fill(
            {
                "exp": 6,
                "exp10": 18,
                "log": 2,
                "log2": 1,
                "log10": 2,
                "sin": 18,
                "cos": 4,
                "tan": 1,
                "tanh": 2,
                "pow": 1,
                "exp2": 1,
                "sqrt": 1,
                "rsqrt": 1,
                "reciprocal": 0,
                "div": 0,
            }[name]
        )
    elif target.vendor == "apple":
        ulps.fill({"sqrt": 5, "rsqrt": 2, "reciprocal": 1, "div": 3}.get(name, 3))
        if name in {"exp", "exp2"}:
            ulps = 3 + np.floor(np.abs(2 * x))
        if name in {"sin", "cos"}:
            absolute.fill(2**-13)
        if name in {"log", "log2"}:
            small = (x >= 0.5) & (x <= 2)
            absolute[small] = 2 ** (-21 if name == "log" else -22)
        if name == "log10":
            absolute = 2**-22 * math.log10(2) + 3 * spacing
        if name == "exp10":
            p = x * math.log2(10)
            absolute = np.abs(expected) * math.log(2) * np.abs(np.spacing(p.astype(np.float32))) / 2
            ulps = 4 + np.floor(np.abs(2 * p))
        if name == "tanh":
            t = np.exp(2 * x)
            delta = (3 + np.floor(np.abs(4 * x))) * np.abs(np.spacing(t.astype(np.float32)))
            absolute = 2 * delta / (t + 1 - delta) ** 2 + 4 * spacing
    # pow and tan have composed contracts. Bound their primitive errors before comparing in result ULPs.
    if name in {"pow", "tan"} and target.vendor in {"nvidia", "apple"}:
        if name == "tan":
            error = 2 ** (-21.41 if target.vendor == "nvidia" else -13)
            sine, cosine = np.sin(x), np.cos(x)
            absolute = error * (1 + np.abs(sine / cosine)) / (np.abs(cosine) - error) + 4 * spacing
        else:
            y = inputs[1].astype(np.float64)
            logarithm = np.log2(x)
            log_error = np.maximum(2**-22, 2 * np.abs(np.spacing(logarithm.astype(np.float32))))
            p = y * logarithm
            delta = np.abs(y) * log_error + np.abs(np.spacing(p.astype(np.float32))) / 2
            absolute = np.abs(expected) * np.expm1(math.log(2) * delta)
            ulps = 3 + np.floor(np.abs(2 * p)) if target.vendor == "apple" else np.full(x.shape, 3.0)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    assert np.all(difference <= absolute + ulps * spacing), (name, target.arch, np.max(difference / spacing))


def rounded_oracle(name: str, values: tuple[int | float, ...]) -> float:
    """Bracket an exact result with numpy neighbors of a binary64 estimate, then select by mode."""
    op, mode = name.split("_")
    x = [float(v) for v in values]
    signs = [bool(np.signbit(v)) for v in x]
    q = [Fraction(v) for v in x]
    exact = (
        q[0] + q[1]
        if op == "add"
        else q[0] * q[1]
        if op == "mul"
        else q[0] / q[1]
        if op == "div"
        else q[0] * q[1] + q[2]
        if op == "fma"
        else 1 / q[0]
        if op == "rcp"
        else q[0]
    )
    if exact == 0:
        if op in ("mul", "div"):
            negative = signs[0] ^ signs[1]
        elif op == "sqrt":
            negative = signs[0]
        else:
            first_sign = signs[0] ^ signs[1] if op == "fma" else signs[0]
            second_sign = signs[-1]
            both_zero = q[0] * q[1] == q[-1] == 0 if op == "fma" else q[0] == q[1] == 0
            negative = first_sign if both_zero and first_sign == second_sign else mode == "rd"
        return -0.0 if negative else 0.0
    negative = exact < 0
    magnitude = abs(exact)
    estimate = math.sqrt(float(magnitude)) if op == "sqrt" else float(magnitude)
    with np.errstate(over="ignore", under="ignore"):
        middle = np.float32(estimate)
        if np.isinf(middle):
            low, high = np.float32(np.finfo(np.float32).max), middle
        else:
            candidate = Fraction(float(middle))
            comparison = candidate * candidate if op == "sqrt" else candidate
            if comparison == magnitude:
                return -float(middle) if negative else float(middle)
            low, high = (
                (middle, np.nextafter(middle, np.float32(np.inf)))
                if comparison < magnitude
                else (np.nextafter(middle, np.float32(-np.inf)), middle)
            )
    if mode == "rn":
        midpoint = (Fraction(float(low)) + (Fraction(1 << 128) if np.isinf(high) else Fraction(float(high)))) / 2
        boundary = midpoint * midpoint if op == "sqrt" else midpoint
        chosen = high if magnitude > boundary or (magnitude == boundary and int(low.view(np.uint32)) & 1) else low
    else:
        chosen = high if (mode == "ru" and not negative) or (mode == "rd" and negative) else low
    return -float(chosen) if negative else float(chosen)


def rounded_inputs(name: str) -> list[np.ndarray]:
    op = name.split("_")[0]
    count = 3 if op == "fma" else 1 if op in ("sqrt", "rcp") else 2
    rng = np.random.default_rng(2026)
    # Full exponent range, finite, with nonzero divisors; no binary64 operation can overflow these inputs.
    arrays = [rng.integers(1, 0x7F800000, SIZE, dtype=np.uint32).view(np.float32) for _ in range(count)]
    for array in arrays:
        array[::2] *= -1
    small, normal, large = 2.0**-149, 2.0**-126, float(np.finfo(np.float32).max)
    rows = {
        "add": [
            (0, -0.0),
            (-0.0, -0.0),
            (1, 2**-24),
            (-1, -(2**-24)),
            (normal, -small),
            (-normal, small),
            (large, large),
            (-large, -large),
            (small, -small),
            (1, -1),
        ],
        "mul": [
            (0, -1),
            (-0.0, -1),
            (normal, 2**-24),
            (-normal, 2**-24),
            (large, 2),
            (-large, 2),
            (small, 0.5),
            (-small, 0.5),
            (small, 1.5),
            (1 + 2**-23, 1 + 2**-23),
        ],
        "div": [
            (0, -1),
            (-0.0, -1),
            (1, 3),
            (-1, 3),
            (small, 2),
            (-small, 2),
            (large, small),
            (-large, small),
        ],
        "fma": [
            (0, -1, -0.0),
            (-0.0, -1, 0),
            (1 + 2**-23, 1 - 2**-23, -1),
            (large, large, -large),
            (-large, large, large),
            (normal, 2**-24, 0),
            (small, 1.5, -small),
        ],
        "sqrt": [(0,), (-0.0,), (small,), (normal,), (large,), (2,), (1 + 2**-23,)],
        "rcp": [(1,), (-1,), (3,), (-3,), (small,), (-small,), (large,), (-large,)],
    }[op]
    if op == "sqrt":
        arrays[0] = np.abs(arrays[0])
    for i, array in enumerate(arrays):
        array[: len(rows)] = [row[i] for row in rows]
    return arrays


@dataclass(frozen=True)
class Case:
    name: str
    dtypes: tuple[str, ...]
    table: int = 0x96

    @property
    def entry(self) -> catalog.Entry:
        return catalog.BY_NAME[self.name]

    @property
    def signature(self) -> catalog.Signature:
        signature = self.entry.signature
        if signature is None:
            return catalog.sig("x: f32, exponent: i32 -> f32" if self.name == "ldexp" else "x: f32, y: f32 -> f32")
        return signature

    @property
    def results(self) -> tuple[str, ...]:
        if self.name in ("copysign", "ldexp"):
            return (self.dtypes[0],)
        result = self.dtypes[0] if self.signature.result == "T" else self.signature.result
        return (result, "i32" if self.name == "libm.frexp" else result) if self.entry.parts else (result,)

    def file(self, directory: Path, *, expression: str = "", dtypes: tuple[str, ...] = ()) -> Path:
        types = dtypes or self.dtypes
        params = [f"a{i}: Vector[T.{dtype}]" for i, dtype in enumerate(types)]
        params += [f"out{i}: Vector[T.{dtype}]" for i, dtype in enumerate(self.results)]
        args = [str(self.table) if p.kind == "const" else f"a{i}[i]" for i, p in enumerate(self.signature.params)]
        call = expression or f"{self.name}({', '.join(args)})"
        assign = (
            f"out0[i] = {call}"
            if len(self.results) == 1
            else f"v0, v1 = {call}\n        out0[i] = v0\n        out1[i] = v1"
        )
        file = directory / "family.py"
        file.write_text(
            "from krn import T, Vector, grid, kernel, copysign, ldexp\n"
            "from krn.facts.intrinsics import libm, fast, bits, simd, half2, bfloat162, cache, rounded\n"
            f"@kernel\ndef family_case({', '.join(params)}) -> None:\n"
            f"    i = grid.thread_rank()\n    if i < out0.shape[0]:\n        {assign}\n"
        )
        return file

    def supported(self, target: Target) -> bool:
        entries = [catalog.BY_NAME[name] for name in self.entry.parts] if self.entry.parts else [self.entry]
        return all(entry.lowering(target) for entry in entries)

    def inputs(self) -> list[np.ndarray]:
        rng = np.random.default_rng(1234)
        module, _, name = self.name.rpartition(".")
        if module == "rounded":
            return rounded_inputs(name)
        values: list[np.ndarray] = []
        for i, dtype in enumerate(self.dtypes):
            dt = np.dtype(DTYPES[dtype])
            if dt.kind == "f":
                low, high = (
                    (-0.9, 0.9)
                    if name in ("asin", "acos", "erfinv")
                    else (0.02, 1.98)
                    if name == "erfcinv"
                    else (0.1, 2.0)
                    if name in ("sqrt", "rsqrt", "reciprocal", "log", "log2", "log10", "pow", "div", "fmod")
                    else (-2.0, 2.0)
                )
                a = rng.uniform(low, high, SIZE).astype(dt)
                if module in ("half2", "bfloat162"):
                    a[:12] = [
                        0.0,
                        -0.0,
                        2**-24,
                        -(2**-24),
                        1.00048828125,
                        1.00390625,
                        65504,
                        65520,
                        -65520,
                        2**-133,
                        3.39e38,
                        -3.39e38,
                    ]
                elif name in ("copysign", "ldexp", "nextafter", "frexp", "sincos"):
                    a[:6] = [0, -0.0, 2**-126, -(2**-126), 1, -1]
                elif name == "erfcx":
                    a[:6] = [-5, 0, 5, 25, 26, 80]
                elif name == "erfinv":
                    a[:4] = [-0.989, -0.001, 0.001, 0.989]
            elif module in ("half2", "bfloat162"):
                bf = module == "bfloat162"
                samples = [
                    0.0,
                    -0.0,
                    2 ** (-133 if bf else -24),
                    -(2 ** (-133 if bf else -24)),
                    1,
                    -1,
                    127,
                    -128,
                    65504,
                    -65504,
                ]
                if bf:
                    samples += [decode(0x7F7F, True), decode(0xFF7F, True)]
                if i:
                    samples = samples[2:] + samples[:2]
                words = [encode(x, bf) | encode(y, bf) << 16 for x, y in zip(samples, reversed(samples), strict=True)]
                words += [
                    encode(float(x), bf) | encode(float(y), bf) << 16
                    for x, y in rng.uniform(-5, 5, (SIZE - len(words), 2))
                ]
                if name in ("min", "max"):
                    nan = 0x7FC0 if bf else 0x7E00
                    words[:6] = (
                        [
                            0,
                            0x80008000,
                            nan | 0x3C00 << 16,
                            0x3C00 | nan << 16,
                            0x7C007C00,
                            0xFC00FC00,
                        ]
                        if i == 0
                        else [
                            0x80008000,
                            0,
                            0x3C00 | nan << 16,
                            nan | 0x3C00 << 16,
                            0xFC00FC00,
                            0x7C007C00,
                        ]
                    )
                if bf and name == "fma":
                    word = (0x7F7F, 0x3FC0, 0xFF49)[i]
                    words[0] = word | word << 16
                a = np.array(words, dt)
            else:
                info = np.iinfo(dt)
                a = rng.integers(info.min, info.max, SIZE, dtype=dt)
                a[:8] = np.array([0, 1, info.max, info.min, 127, 128, 255, 256], dtype=dt)
                if module == "bits" and self.signature.params[i].name in ("pos", "length"):
                    a[:64] = (
                        np.repeat(BOUNDARIES, 8) if self.signature.params[i].name == "pos" else np.tile(BOUNDARIES, 8)
                    )
                if name.startswith("funnelshift") and i == 2:
                    a[:6] = [0, 1, 31, 32, 33, 63]
                if name == "byte_perm" and i == 2:
                    a[:8] = [0x0123, 0x4567, 0x89AB, 0xCDEF, 0x8888, 0xFFFF, 0x7654, 0x3210]
                if module == "simd":
                    a[:12] = np.array(
                        [
                            0,
                            0xFFFFFFFF,
                            0x80808080,
                            0x7F7F7F7F,
                            0x80008000,
                            0x7FFF7FFF,
                            0x01010101,
                            0xFF00FF00,
                            0xFFFF0000,
                            0x00800080,
                            0x007F007F,
                            0x80000000,
                        ],
                        np.uint32,
                    ).view(dt)
                    if i == 1:
                        a[:12] = np.roll(a[:12], 3)
                if self.name == "ldexp":
                    a = rng.integers(-20, 20, SIZE, dtype=dt)
                    a[:8] = [-1075, -150, -149, -126, 0, 128, 200, 1024]
            values.append(a)
        return values

    def expected(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, ...]:
        module, _, name = self.name.rpartition(".")
        rows: list[tuple[int | float, ...]] = []
        for index in range(SIZE):
            args = tuple(float(a[index]) if a.dtype.kind == "f" else int(a[index]) for a in inputs)
            if module == "rounded":
                row = (rounded_oracle(name, args),)
            elif module == "cache":
                row = args
            elif module in ("libm", "fast", ""):
                row = math_oracle(name, args, self.dtypes[0])
            elif module in ("half2", "bfloat162"):
                row = (packed_oracle(name, args, module == "bfloat162"),)
            elif module == "simd" and not name.startswith("dp"):
                row = (simd_oracle(name, *(int(x) for x in args)),)
            else:
                integers = tuple(int(x) for x in args) + ((self.table,) if name == "lop3" else ())
                result = integer_oracle(name, integers, int(self.dtypes[0][1:]), self.dtypes[0].startswith("i"))
                if self.results[0].startswith("i") and result >= 1 << (int(self.results[0][1:]) - 1):
                    result -= 1 << int(self.results[0][1:])
                row = (result,)
            rows.append(row)
        return tuple(np.array([row[i] for row in rows], DTYPES[dtype]) for i, dtype in enumerate(self.results))

    def flushed_rows(self, inputs: list[np.ndarray], expected: tuple[np.ndarray, ...]) -> np.ndarray:
        """Rows where a bfloat16 subnormal is an operand or the result of float arithmetic. Those are float32
        denormals, which an Apple GPU flushes to zero."""
        rows = np.zeros(SIZE, bool)
        if not self.name.startswith("bfloat162.") or self.name.rpartition(".")[2] not in (
            "add",
            "sub",
            "mul",
            "fma",
        ):
            return rows
        for words in (*inputs, *expected):
            for shift in (0, 16):
                lane = (words >> shift) & 0xFFFF
                rows |= ((lane & 0x7F80) == 0) & ((lane & 0x7F) != 0)
        return rows

    def check(
        self,
        actual: list[np.ndarray],
        expected: tuple[np.ndarray, ...],
        skip: np.ndarray | None = None,
    ) -> None:
        keep = np.ones(SIZE, bool) if skip is None else ~skip
        for a, b in ((x[keep], y[keep]) for x, y in zip(actual, expected, strict=True)):
            if self.name.startswith(("cache.", "rounded.")):
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                continue
            if b.dtype.kind == "f":
                np.testing.assert_allclose(
                    a,
                    b,
                    rtol=1e-4 if self.name.startswith("fast.") else 1e-5,
                    atol=1e-6,
                    equal_nan=True,
                )
                np.testing.assert_array_equal(np.signbit(a[b == 0]), np.signbit(b[b == 0]))
            else:
                np.testing.assert_array_equal(a, b)

    def run(
        self,
        directory: Path,
        target: Target,
        backend: str,
        launcher: Callable[[Kernel], CUDARun | MetalRun] | None = None,
    ) -> None:
        if target.vendor == "apple" and "f64" in self.dtypes:
            pytest.skip("Metal has no double")
        file = self.file(directory)
        if not self.supported(target):
            with pytest.raises(CaptureError, match=rf"{target.arch}.*targets that have it:"):
                capture(file, "family_case", target=target)
            return
        ir = capture(file, "family_case", target=target)
        inputs = self.inputs()
        with np.errstate(all="ignore"):
            expected = self.expected(inputs)
            outputs = [np.zeros_like(a) for a in expected]
            emulate(ir, target).run(Blocks(SIZE // 32, threads=32), [*inputs, *outputs])
            self.check(outputs, expected)
            if backend == "emulator":
                return
            actual = [np.zeros_like(a) for a in expected]
            arguments: list[Value] = [*inputs, *actual]
            if backend == "c":
                CPUProgram(ir, emit(ir, target))(*arguments, launch=Blocks(SIZE // 32, threads=32))
            else:
                assert launcher is not None
                launcher(ir).run(Blocks(SIZE // 32, threads=32), arguments)
            skip = self.flushed_rows(inputs, expected) if target.vendor == "apple" else None
            if self.name.startswith("fast."):
                check_fast_accuracy(self.name.removeprefix("fast."), actual[0], outputs[0], inputs, target)
            else:
                self.check(actual, expected, skip)
                self.check(actual, tuple(outputs), skip)


@dataclass(frozen=True)
class ScopeCase:
    name: str
    dtype: str = "u32"
    masked: bool = False

    @property
    def entry(self) -> catalog.Entry:
        return catalog.BY_NAME[self.name]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype,)

    def supported(self, target: Target) -> bool:
        return bool(self.entry.lowering(target)) and (self.entry.feature is None or target.has(self.entry.feature))

    def file(self, directory: Path) -> Path:
        file = directory / "scope.py"
        if self.name.startswith("warp.reduce_"):
            call = f"{self.name}(x[i]" + (", mask=T.u64(65535))" if self.masked else ")")
            body = f"out[i] = {call}"
            if self.masked:
                body = "if warp.lane() < 16:\n        " + body
        elif self.name.startswith("thread.clock") or self.name == "thread.globaltimer":
            body = f"out[i] = {self.name}()"
        elif self.name.startswith("cluster.") and self.name.endswith(tuple("xyz")):
            base, axis = self.name.rsplit(".", 1)
            body = f"out[i] = {base}().{axis}"
        else:
            body = f"{self.name}()\n    out[i] = x[i]"
        file.write_text(
            "from krn import T, Vector, grid, kernel, thread, warp, cluster\n"
            f"@kernel\ndef family_case(x: Vector[T.{self.dtype}], out: Vector[T.{self.dtype}]):\n"
            f"    i = grid.thread_rank()\n    {body}\n"
        )
        return file

    def expected(self, x: np.ndarray, warp: int) -> np.ndarray:
        if not self.name.startswith("warp.reduce_"):
            return x.copy()
        result = np.zeros_like(x)
        operation = self.name.removeprefix("warp.reduce_")
        functions = {
            "add": np.add,
            "min": np.minimum,
            "max": np.maximum,
            "and": np.bitwise_and,
            "or": np.bitwise_or,
            "xor": np.bitwise_xor,
        }
        count = 16 if self.masked else warp
        for start in range(0, len(x), warp):
            result[start : start + count] = functions[operation].reduce(x[start : start + count], dtype=x.dtype)
        return result

    def run(
        self,
        directory: Path,
        target: Target,
        backend: str,
        launcher: Callable[[Kernel], CUDARun | MetalRun] | None = None,
    ) -> None:
        if not self.supported(target):
            with pytest.raises(CaptureError, match="targets that have it:"):
                capture(self.file(directory), "family_case", target=target)
            return
        ir = capture(self.file(directory), "family_case", target=target)
        if self.name.startswith("cluster."):
            with pytest.raises(NotImplementedError, match=r"one block at a time.*cluster launch"):
                emulate(ir, target)
            return
        dtype = np.dtype(DTYPES[self.dtype])
        rng = np.random.default_rng(811)
        x = rng.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, 64, dtype=dtype)
        out = np.zeros_like(x)
        launch = Blocks(1, threads=64)
        if backend == "emulator":
            emulate(ir, target).run(launch, [x, out])
        elif backend == "c":
            CPUProgram(ir, emit(ir, target))(x, out, launch=launch)
        else:
            assert launcher is not None
            launcher(ir).run(launch, [x, out])
        if self.name.startswith("thread.clock") or self.name == "thread.globaltimer":
            assert np.any(out)
        else:
            np.testing.assert_array_equal(out, self.expected(x, target.warp_size))


SCOPE_CASES = (
    [
        ScopeCase(f"warp.reduce_{op}", dtype, masked)
        for op in ("add", "min", "max", "and", "or", "xor")
        for dtype in ("i32", "u32")
        for masked in (False, True)
    ]
    + [ScopeCase(f"thread.{name}", "u32" if name == "clock" else "u64") for name in ("clock", "clock64", "globaltimer")]
    + [ScopeCase(f"thread.fence_acq_rel_{scope}") for scope in ("block", "device", "system")]
    + [
        ScopeCase("thread.fence_proxy_async"),
        ScopeCase("cluster.arrive"),
        ScopeCase("cluster.wait"),
    ]
    + [ScopeCase(f"cluster.{field}.{axis}") for field in ("dim_blocks", "block_index") for axis in "xyz"]
)


@dataclass(frozen=True)
class AtomicCase:
    name: str
    dtype: str
    shared: bool = False

    @property
    def entry(self) -> catalog.Entry:
        return catalog.FUNCTIONS[self.name]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype,)

    @property
    def op(self) -> AtomicOp:
        assert self.entry.atomic is not None
        return self.entry.atomic[0]

    def supported(self, target: Target) -> bool:
        return plan(self.op, self.dtype, target).strategy is not Strategy.REFUSED

    def file(self, directory: Path) -> Path:
        target = "tile[i]" if self.shared else "x[i]"
        operands = "T." + self.dtype
        args = f"{operands}(3), {operands}(11)" if self.op is AtomicOp.CAS else f"{operands}(7)"
        local = f"    tile = shared_array(T.{self.dtype}, 64)\n    tile[i] = x[i]\n" if self.shared else ""
        finish = "    x[i] = tile[i]\n" if self.shared else ""
        file = directory / "scoped_atomic.py"
        file.write_text(
            "from krn import T, Vector, grid, kernel, shared_array\n"
            "from krn.facts.intrinsics import atomic_block, atomic_system\n"
            f"@kernel\ndef family_case(x: Vector[T.{self.dtype}], out: Vector[T.{self.dtype}]):\n"
            f"    i = grid.thread_rank()\n{local}    out[i] = {self.name}({target}, {args})\n{finish}"
        )
        return file

    def run(
        self,
        directory: Path,
        target: Target,
        backend: str,
        launcher: Callable[[Kernel], CUDARun | MetalRun] | None = None,
    ) -> None:
        file = self.file(directory)
        if not self.supported(target):
            with pytest.raises(CaptureError):
                capture(file, "family_case", target=target)
            return
        ir = capture(file, "family_case", target=target)
        dtype = np.dtype(DTYPES[self.dtype])
        x = (np.arange(64) % 13).astype(dtype)
        old = x.copy()
        out = np.zeros_like(x)
        formulas: dict[AtomicOp, Callable[[np.ndarray], np.ndarray]] = {
            AtomicOp.ADD: lambda a: a + 7,
            AtomicOp.SUB: lambda a: a - 7,
            AtomicOp.AND: lambda a: a & 7,
            AtomicOp.OR: lambda a: a | 7,
            AtomicOp.XOR: lambda a: a ^ 7,
            AtomicOp.EXCH: lambda a: np.full_like(a, 7),
            AtomicOp.INC: lambda a: np.where(a >= 7, 0, a + 1),
            AtomicOp.DEC: lambda a: np.where((a == 0) | (a > 7), 7, a - 1),
            AtomicOp.MAX: lambda a: np.maximum(a, 7),
            AtomicOp.MIN: lambda a: np.minimum(a, 7),
            AtomicOp.CAS: lambda a: np.where(a == 3, 11, a),
        }
        expected = formulas[self.op](old).astype(dtype)
        launch = Blocks(1, threads=64)
        if backend == "emulator":
            emulate(ir, target).run(launch, [x, out])
        elif backend == "c":
            CPUProgram(ir, emit(ir, target))(x, out, launch=launch)
        else:
            assert launcher is not None
            launcher(ir).run(launch, [x, out])
        np.testing.assert_array_equal(out, old)
        np.testing.assert_array_equal(x, expected)


ATOMIC_CASES = [
    AtomicCase(entry.name, dtype)
    for entry in catalog.FUNCTIONS.values()
    if entry.atomic is not None
    for signature in (entry.signature,)
    if signature is not None
    for dtype in signature.types
]


@dataclass(frozen=True)
class CacheEffectCase:
    name: str
    dtype: str

    @property
    def entry(self) -> catalog.Entry:
        return catalog.FUNCTIONS[self.name]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype,)

    def supported(self, target: Target) -> bool:
        return bool(self.entry.lowering(target))

    def file(self, directory: Path) -> Path:
        call = f"{self.name}(out[i], x[i])" if ".st" in self.name else f"{self.name}(x[i])\n    out[i] = x[i]"
        file = directory / "cache.py"
        file.write_text(
            "from krn import T, Vector, grid, kernel\nfrom krn.facts.intrinsics import cache\n"
            f"@kernel\ndef family_case(x: Vector[T.{self.dtype}], out: Vector[T.{self.dtype}]):\n"
            f"    i = grid.thread_rank()\n    {call}\n"
        )
        return file

    def run(
        self,
        directory: Path,
        target: Target,
        backend: str,
        launcher: Callable[[Kernel], CUDARun | MetalRun] | None = None,
    ) -> None:
        if self.dtype == "f64" and target.vendor == "apple":
            pytest.skip("Metal has no double")
        ir = capture(self.file(directory), "family_case", target=target)
        dtype = np.dtype(DTYPES[self.dtype])
        x = np.arange(64, dtype=dtype)
        if dtype.kind == "f":
            x[:6] = [0, -0.0, np.inf, -np.inf, np.nan, np.finfo(dtype).tiny]
        else:
            x[:2] = [np.iinfo(dtype).min, np.iinfo(dtype).max]
        out = np.zeros_like(x)
        launch = Blocks(1, threads=64)
        if backend == "emulator":
            emulate(ir, target).run(launch, [x, out])
        elif backend == "c":
            CPUProgram(ir, emit(ir, target))(x, out, launch=launch)
        else:
            assert launcher is not None
            launcher(ir).run(launch, [x, out])
        np.testing.assert_array_equal(x.view(np.uint8), out.view(np.uint8))


CACHE_EFFECT_CASES = [
    CacheEffectCase(e.name, dtype)
    for e in catalog.FUNCTIONS.values()
    if e.name.startswith("cache.") and e.reference is None
    for signature in (e.signature,)
    if signature is not None
    for dtype in signature.types
]


@dataclass(frozen=True)
class AsyncCase:
    name: str = "async_copy.copy"
    dtype: str = "u32"
    count: int = 1
    pending: int = 0

    @property
    def entry(self) -> catalog.Entry:
        return catalog.FUNCTIONS[self.name]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype,)

    def supported(self, target: Target) -> bool:
        return bool(self.entry.lowering(target))

    def file(self, directory: Path) -> Path:
        file = directory / "async.py"
        file.write_text(
            "from krn import T, Vector, grid, block, kernel, shared_array, targets\n"
            "from krn.facts.intrinsics import async_copy\n"
            f"@kernel\ndef family_case(x: Vector[T.{self.dtype}], out: Vector[T.{self.dtype}]):\n"
            "    i = grid.thread_rank()\n    lane = block.thread_rank()\n"
            f"    tile = shared_array(T.{self.dtype}, 256)\n"
            f"    async_copy.copy(tile[lane * {self.count}], x[i * {self.count}], {self.count})\n"
            f"    async_copy.commit()\n    async_copy.wait({self.pending})\n    async_copy.wait(0)\n"
            "    match target:\n        case targets.CPU():\n            pass\n"
            "        case _:\n            block.sync()\n"
            "    neighbor = (lane + 1) % block.num_threads()\n"
            f"    for k in range({self.count}):\n"
            f"        out[(i - T.u64(lane) + T.u64(neighbor)) * {self.count} + k] = tile[neighbor * {self.count} + k]\n"
        )
        return file

    def run(
        self,
        directory: Path,
        target: Target,
        backend: str,
        launcher: Callable[[Kernel], CUDARun | MetalRun] | None = None,
    ) -> None:
        ir = capture(self.file(directory), "family_case", target=target)
        x = (np.arange(64 * self.count) * 19).astype(DTYPES[self.dtype])
        out = np.zeros_like(x)
        # Sequential C has one thread per block; GPU/emulator lanes consume a neighbor's tile after the barrier.
        launch = Blocks(64, threads=1) if backend == "c" else Blocks(1, threads=64)
        if backend == "emulator":
            emulate(ir, target).run(launch, [x, out])
        elif backend == "c":
            CPUProgram(ir, emit(ir, target))(x, out, launch=launch)
        else:
            assert launcher is not None
            launcher(ir).run(launch, [x, out])
        np.testing.assert_array_equal(out, x)
        assert ("x",) not in ir.written and ir.shared[0].path in ir.written


ASYNC_CASES = [AsyncCase(dtype=dtype, count=count) for dtype in ("i32", "u32", "f32") for count in (1, 2, 4)] + [
    AsyncCase("async_copy.commit"),
    *[AsyncCase("async_copy.wait", pending=p) for p in range(8)],
]


@dataclass(frozen=True)
class CUDAMemoryCase:
    name: str
    dtype: str = "u32"

    @property
    def entry(self) -> catalog.Entry:
        return catalog.FUNCTIONS[self.name]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype,)

    def supported(self, target: Target) -> bool:
        return bool(self.entry.lowering(target)) and (self.entry.feature is None or target.has(self.entry.feature))

    def file(self, directory: Path) -> Path:
        operations = {
            "mbarrier.init": "mbarrier.init(bar[0], 1)\n    out[0] = x[0]",
            "mbarrier.arrive": "out[0] = T.u32(mbarrier.arrive(bar[0]))",
            "mbarrier.arrive_expect_tx": "out[0] = T.u32(mbarrier.arrive_expect_tx(bar[0], 16))",
            "mbarrier.wait_parity": "mbarrier.wait_parity(bar[0], 0)\n    out[0] = x[0]",
            "bulk.copy": "bulk.copy(tile[0], x[0], 16, bar[0])\n    out[0] = x[0]",
            "dsmem.load": "out[0] = dsmem.load(tile[0], 0)",
            "dsmem.store": "dsmem.store(tile[0], 0, x[0])\n    out[0] = tile[0]",
        }
        file = directory / "cuda_memory.py"
        file.write_text(
            "from krn import T, Vector, kernel, shared_array\n"
            "from krn.facts.intrinsics import mbarrier, bulk, dsmem\n"
            f"@kernel\ndef family_case(x: Vector[T.{self.dtype}], out: Vector[T.{self.dtype}]):\n"
            f"    tile = shared_array(T.{self.dtype}, 64)\n    bar = shared_array(T.u64, 1)\n"
            f"    {operations[self.name]}\n"
        )
        return file


CUDA_MEMORY_CASES = [
    CUDAMemoryCase(e.name, dtype)
    for e in catalog.FUNCTIONS.values()
    if e.name.startswith(("mbarrier.", "bulk.", "dsmem."))
    for signature in (e.signature,)
    if signature is not None
    for dtype in signature.types or ("u32",)
]
type CompileCase = Case | ScopeCase | AtomicCase | CacheEffectCase | AsyncCase | CUDAMemoryCase
VIDEO = frozenset(name for name in catalog.FUNCTIONS if name.startswith(("simd.vi", "simd.vsad")))  # tests/intrinsics


CASES = (
    [
        Case(
            entry.name,
            tuple(dtype if p.kind.split()[-1] == "T" else p.kind for p in entry.signature.params if p.kind != "const"),
            table,
        )
        for entry in catalog.FUNCTIONS.values()
        if entry.signature is not None and entry.reference is not None and entry.name not in VIDEO
        for dtype in entry.signature.types or ("u32",)
        for table in ((0x00, 0xFF, 0xF0, 0xCA, 0x96, 0xE8) if entry.name == "bits.lop3" else (0x96,))
    ]
    + [Case("copysign", (dtype, dtype)) for dtype in ("f32", "f64")]
    + [Case("ldexp", (dtype, "i32")) for dtype in ("f32", "f64")]
)


COMPILE_CASES: list[CompileCase] = [
    *CASES,
    *SCOPE_CASES,
    *ATOMIC_CASES,
    *CACHE_EFFECT_CASES,
    *ASYNC_CASES,
    *CUDA_MEMORY_CASES,
]


def test_sketch_contracts_and_cases_are_complete() -> None:
    functions: set[str] = set()
    for module in (
        libm,
        fast,
        bits,
        simd,
        half2,
        bfloat162,
        atomic_block,
        atomic_system,
        cache,
        async_copy,
        mbarrier,
        bulk,
        dsmem,
        rounded,
    ):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn.__module__ != module.__name__ and not (module is fast and fn.__module__ == "krn.fast.math"):
                continue
            path = f"{module.__name__.rsplit('.', 1)[1]}.{name}"
            functions.add(path)
            signature = catalog.FUNCTIONS[path].signature
            assert signature is not None
            assert tuple(inspect.signature(fn).parameters) == tuple(p.name for p in signature.params)
    assert functions == set(catalog.FUNCTIONS)
    exclusions = {c.name: "effectful atomic: independent element and contention kernels" for c in ATOMIC_CASES}
    exclusions |= {c.name: "effectful cache hint: exact bitwise copy kernels" for c in CACHE_EFFECT_CASES}
    exclusions |= {c.name: "effectful copy: tile copy, commit, and wait kernels" for c in ASYNC_CASES}
    exclusions |= {name: "video SIMD and SAD: own lane oracles in tests/intrinsics" for name in VIDEO}
    exclusions |= {
        c.name: "NVIDIA-only memory: compile cases; transaction copy runs on sm_120a only" for c in CUDA_MEMORY_CASES
    }
    assert functions <= {case.name for case in CASES} | exclusions.keys()


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.name}-{'-'.join(c.dtypes)}-{c.table:x}")
@pytest.mark.parametrize("backend", ["emulator", "c"])
def test_family_local(tmp_path: Path, case: Case, backend: str) -> None:
    case.run(tmp_path, GPU if backend == "emulator" else Target.of("avx512"), backend)


@pytest.fixture(scope="module")
def family_metal() -> MetalDevice:
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        return metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def family_card(gpu: CUDADevice) -> CUDADevice:
    return gpu


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_family_metal(tmp_path: Path, family_metal: MetalDevice, case: Case) -> None:
    case.run(tmp_path, family_metal.target, "metal", lambda ir: MetalRun(family_metal, ir))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_family_cuda(tmp_path: Path, family_card: CUDADevice, case: Case) -> None:
    case.run(tmp_path, family_card.target, "cuda", lambda ir: CUDARun(family_card, ir))


@pytest.mark.parametrize(
    "case",
    COMPILE_CASES,
    ids=lambda c: c.name,
)
@pytest.mark.parametrize("arch", ["sm_80", "sm_90a", "sm_120a", "sm_100a", "sm_103a"])
def test_family_nvrtc(
    tmp_path: Path,
    case: CompileCase,
    arch: str,
    nvrtc: Nvrtc,
) -> None:
    if case.entry.sm > int(arch[3:].rstrip("a")):
        pytest.skip("instruction needs a newer SM")
    target = Target.of(arch)
    if not case.supported(target):
        with pytest.raises(CaptureError, match="targets that have it:"):
            capture(case.file(tmp_path), "family_case", target=target)
        return
    ir = capture(case.file(tmp_path), "family_case", target=target)
    assert nvrtc.compile(emit(ir, target).source, arch, "cubin").data[:4] == b"\x7fELF"


HIP_HEADER = (
    "\n".join(
        f'extern "C" __device__ float {name}f(float{", float" if name == "atan2" else ""});\n'
        f'extern "C" __device__ double {name}(double{", double" if name == "atan2" else ""});'
        for name in ("tan", "asin", "acos", "atan", "atan2")
    )
    + '\nextern "C" __device__ double sin(double);\nextern "C" __device__ double cos(double);\n'
    + "\n".join(
        f"#define {name}{suffix} __builtin_{name}{suffix}"
        for name in (
            "erf",
            "erfc",
            "hypot",
            "fmod",
            "nextafter",
            "copysign",
            "ldexp",
        )
        for suffix in ("", "f")
    )
    + "\n"
    + "\n".join(f"#define {name}f _family_{name}" for name in ("erfinv", "erfcinv", "erfcx"))
    + "\n__device__ inline double rsqrt(double x) { return 1.0/__builtin_sqrt(x); }\n"
)


def check_fast_amd_refusal(file: Path, name: str, arch: str) -> None:
    """A `fast.*` row has no verified HIP lowering: capture refuses it and names the strict twin."""
    with pytest.raises(TargetError) as refusal:
        capture(file, "family_case", target=Target.of(arch))
    error, short = refusal.value, name.removeprefix("fast.")
    assert (error.feature, error.target, error.reason, error.instead) == (
        f"fast.math.{short}",
        arch,
        "no HIP lowering of this approximate function has been verified against the compiler yet",
        f"the strict `krn.math.{short}`",
    )


@pytest.mark.skipif(not CLANG_HAS_AMDGPU, reason="this clang has no AMDGPU backend")
@pytest.mark.parametrize(
    "case",
    COMPILE_CASES,
    ids=lambda c: c.name,
)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_family_hip(tmp_path: Path, case: CompileCase, arch: str) -> None:
    target = Target.of(arch)
    if case.name.startswith("fast."):
        check_fast_amd_refusal(case.file(tmp_path), case.name, arch)
        return
    if not case.supported(target):
        with pytest.raises(CaptureError, match="targets that have it:"):
            capture(case.file(tmp_path), "family_case", target=target)
        return
    source = emit(capture(case.file(tmp_path), "family_case", target=target), target).source
    header = tmp_path / "extra.h"
    header.write_text(
        "#include <hip/hip_runtime.h>\n"
        + HIP_HEADER
        + catalog.BY_NAME["libm.erfinv"].helper_for("cpu").replace("@inline", "__device__ inline ")
    )
    file, output = tmp_path / "family.hip", tmp_path / "family.s"
    file.write_text(source)
    done = subprocess.run(
        [
            CLANG[0],
            "-x",
            "hip",
            f"--offload-arch={arch}",
            "--offload-device-only",
            "-nogpulib",
            "-nogpuinc",
            "-I",
            str(Path(__file__).resolve().parents[1] / "hip_shim"),
            "-include",
            str(header),
            "-S",
            "-o",
            str(output),
            str(file),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-6000:]
    assert descriptor(output.read_text(), "family_case")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("arch", ["avx512", "apple_m4", "gfx942", "sm_90a"])
def test_family_sources_and_refusals(tmp_path: Path, case: Case, arch: str) -> None:
    target = Target.of(arch)
    if arch == "apple_m4" and "f64" in case.dtypes:
        return
    file = case.file(tmp_path)
    if case.supported(target):
        assert emit(capture(file, "family_case", target=target), target).entry == "family_case"
    elif case.name.startswith("fast.") and target.vendor == "amd":
        check_fast_amd_refusal(file, case.name, arch)
    else:
        with pytest.raises(CaptureError, match=rf"{arch}.*targets that have it:"):
            capture(file, "family_case", target=target)


@pytest.mark.parametrize("case", [c for c in CASES if c.name in catalog.FUNCTIONS], ids=lambda c: c.name)
def test_family_wrong_dtype_is_at_user_call(tmp_path: Path, case: Case) -> None:
    wrong = "f64" if case.dtypes[0] == "f32" else "i32" if case.dtypes[0] == "u32" else "f32"
    if case.signature.params[0].kind.split()[-1] == "T" and wrong in case.signature.types:
        wrong = "u32" if wrong.startswith("f") else "f32"
        if wrong in case.signature.types:
            wrong = "f16"
    types = (wrong, *case.dtypes[1:])
    file = case.file(tmp_path, dtypes=types)
    with pytest.raises(CaptureError) as error:
        capture(file, "family_case", target=GPU)
    assert error.value.span is not None and error.value.span.file == str(file)
    assert error.value.span is not None and error.value.span.line == 7


def test_intrinsic_spec_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(INTRINSICS, "warp.typed", IntrinsicSpec(None, ("x",), types=(U32,)))
    monkeypatch.setitem(catalog.BY_NAME, "warp.typed", replace(catalog.BY_NAME["warp.lane"], name="warp.typed"))
    file = tmp_path / "typed.py"
    file.write_text("from krn import T, kernel, warp\n@kernel\ndef bad(x: T.f32):\n    warp.typed(x)\n")
    with pytest.raises(CaptureError, match="accepts u32, got f32") as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.line == 4


@pytest.mark.parametrize("case", SCOPE_CASES, ids=lambda c: f"{c.name}-{c.dtype}-{c.masked}")
@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a", "apple_m4", "gfx942", "avx512"])
def test_scope_local(tmp_path: Path, case: ScopeCase, arch: str) -> None:
    target = Target.of(arch)
    case.run(tmp_path, target, "c" if target.vendor == "cpu" else "emulator")


@pytest.mark.parametrize("case", SCOPE_CASES, ids=lambda c: f"{c.name}-{c.dtype}-{c.masked}")
def test_scope_metal(tmp_path: Path, family_metal: MetalDevice, case: ScopeCase) -> None:
    case.run(tmp_path, family_metal.target, "metal", lambda ir: MetalRun(family_metal, ir))


@pytest.mark.parametrize("case", SCOPE_CASES, ids=lambda c: f"{c.name}-{c.dtype}-{c.masked}")
def test_scope_cuda(tmp_path: Path, family_card: CUDADevice, case: ScopeCase) -> None:
    case.run(tmp_path, family_card.target, "cuda", lambda ir: CUDARun(family_card, ir))


@pytest.mark.parametrize("op", ["add", "min", "max", "and", "or", "xor"])
def test_reduce_dtype_and_c_refusal(tmp_path: Path, op: str) -> None:
    case = ScopeCase(f"warp.reduce_{op}", "f32")
    with pytest.raises(CaptureError, match="accepts i32, u32, got f32") as error:
        capture(case.file(tmp_path), "family_case", target=GPU)
    assert error.value.span is not None and error.value.span.line == 5
    with pytest.raises(CaptureError, match="threads of a block one after the other"):
        capture(ScopeCase(case.name).file(tmp_path), "family_case", target=Target.of("avx512"))


@pytest.mark.parametrize("op", ["add", "min", "max", "and", "or", "xor"])
@pytest.mark.parametrize("backend", ["emulator", "metal", "cuda"])
def test_signed_reduction_comparison(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    op: str,
    backend: str,
) -> None:
    card = request.getfixturevalue("family_card") if backend == "cuda" else None
    metal = pytest.importorskip("krn.runtime.metal") if backend == "metal" else None
    gpu = request.getfixturevalue("family_metal") if metal is not None else None
    target = card.target if card is not None else gpu.target if gpu is not None else GPU
    file = tmp_path / "comparison.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel, warp\n@kernel\n"
        "def run(x: Vector[T.i32], out: Vector[T.u32]):\n    i = grid.thread_rank()\n"
        f"    out[i] = T.u32(warp.reduce_{op}(x[i]) < T.i32(0))\n"
    )
    x = np.full(64, -1, np.int32)
    # Keep XOR negative for each warp as well as the other reductions.
    x[::32] = 0
    out = np.zeros(64, np.uint32)
    expected = (ScopeCase(f"warp.reduce_{op}", "i32").expected(x, target.warp_size) < 0).astype(np.uint32)
    ir, launch = capture(file, "run", target=target), Blocks(1, threads=64)
    if card is not None:
        CUDARun(card, ir).run(launch, [x, out])
    elif metal is not None and gpu is not None:
        MetalRun(gpu, ir).run(launch, [x, out])
    else:
        emulate(ir, target).run(launch, [x, out])
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("backend", ["emulator", "cuda"])
def test_clock_intervals(tmp_path: Path, backend: str, request: pytest.FixtureRequest) -> None:
    card = request.getfixturevalue("family_card") if backend == "cuda" else None
    target = card.target if card is not None else GPU
    file = tmp_path / "clocks.py"
    file.write_text(
        "from krn import T, Vector, kernel, thread, sleep\n@kernel\ndef clocks(out: Vector[T.u64]):\n"
        "    c = thread.clock64()\n    t = thread.globaltimer()\n    x = T.u64(0)\n"
        "    for i in range(100):\n        x += T.u64(i)\n    for k in range(20):\n        sleep(100000)\n"
        "    out[0] = thread.clock64() - c\n    out[1] = thread.globaltimer() - t\n    out[2] = x\n"
    )
    ir = capture(file, "clocks", target=target)
    out = np.zeros(3, np.uint64)
    launch = Blocks(1, threads=1)
    if card is None:
        emulate(ir, target).run(launch, [out])
        assert out[0] >= 0 and out[1] >= 0
    else:
        CUDARun(card, ir).run(launch, [out])
        assert out[0] > 0
        # __nanosleep(t) sleeps somewhere in [0, 2t] (CUDA programming guide), so one call proves nothing: 20 calls of
        # 100 us must still add up to more than 100 us on the timer, and the whole kernel stays under 50 ms
        assert 100000 < out[1] < 50000000
    assert out[2] == 4950


def test_cluster_cells_require_axes(tmp_path: Path) -> None:
    file = tmp_path / "cluster.py"
    file.write_text("from krn import kernel, cluster\n@kernel\ndef bad():\n    cluster.dim_blocks()\n")
    with pytest.raises(CaptureError, match="needs an axis"):
        capture(file, "bad", target=GPU)


def test_scope_catalog_specs_cover_every_added_instruction() -> None:
    assert {c.name for c in SCOPE_CASES} <= INTRINSICS.keys()
    for case in SCOPE_CASES:
        assert (case.entry.reference is not None) == case.name.startswith("warp.reduce_")
        if case.entry.feature is Feature.CLUSTERS:
            assert not case.supported(Target.of("sm_120a"))


@pytest.mark.parametrize("case", ATOMIC_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
@pytest.mark.parametrize("arch", ["sm_90a", "gfx942", "apple_m4", "avx512"])
def test_scoped_atomic_local(tmp_path: Path, case: AtomicCase, arch: str) -> None:
    target = Target.of(arch)
    case.run(tmp_path, target, "c" if target.vendor == "cpu" else "emulator")


@pytest.mark.parametrize("case", ATOMIC_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
def test_scoped_atomic_metal(tmp_path: Path, family_metal: MetalDevice, case: AtomicCase) -> None:
    if case.dtype == "f32":
        pytest.skip("Metal execution coverage here uses 32-bit integer atomics")
    case.run(tmp_path, family_metal.target, "metal", lambda ir: MetalRun(family_metal, ir))


@pytest.mark.parametrize("case", ATOMIC_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
def test_scoped_atomic_cuda(tmp_path: Path, family_card: CUDADevice, case: AtomicCase) -> None:
    case.run(tmp_path, family_card.target, "cuda", lambda ir: CUDARun(family_card, ir))


@pytest.mark.parametrize(
    "case",
    [replace(c, shared=True) for c in ATOMIC_CASES if c.dtype == "u32"],
    ids=lambda c: c.name,
)
@pytest.mark.parametrize("arch", ["sm_90a", "gfx942", "apple_m4", "avx512"])
def test_scoped_atomic_shared(tmp_path: Path, case: AtomicCase, arch: str) -> None:
    target = Target.of(arch)
    case.run(tmp_path, target, "c" if target.vendor == "cpu" else "emulator")
    assert (
        "threadgroup"
        in emit(
            capture(case.file(tmp_path), "family_case", target=Target.of("apple_m4")),
            Target.of("apple_m4"),
        ).source
    )


@pytest.mark.parametrize("scope", ["block", "system"])
@pytest.mark.parametrize("scenario", ["histogram", "largest", "smallest", "signed_largest", "counter", "lock"])
@pytest.mark.parametrize("backend", ["emulator", "c", "metal", "cuda"])
def test_scoped_atomic_contention(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    scope: str,
    scenario: str,
    backend: str,
) -> None:
    card = request.getfixturevalue("family_card") if backend == "cuda" else None
    metal = pytest.importorskip("krn.runtime.metal") if backend == "metal" else None
    gpu = request.getfixturevalue("family_metal") if metal is not None else None
    target = (
        card.target
        if card is not None
        else gpu.target
        if gpu is not None
        else Target.of("avx512")
        if backend == "c"
        else GPU
    )
    dtype = {
        "histogram": "i32",
        "largest": "f32",
        "smallest": "i64",
        "signed_largest": "i64",
        "counter": "u32",
        "lock": "i32",
    }[scenario]
    if backend == "metal" and dtype in {"i64", "u64"}:
        pytest.skip("Metal has no 64-bit atomics beyond unsigned min/max")
    rng = np.random.default_rng(321)
    x = rng.integers(0, 8, 64).astype(DTYPES[dtype])
    if scenario == "largest":
        x = rng.standard_normal(64).astype(np.float32)
    elif scenario in {"smallest", "signed_largest"}:
        x = rng.integers(-(1 << 62), 1 << 62, 64, dtype=np.int64)
        x[:4] = [-(1 << 63), (1 << 63) - 1, -1, 0]
    op = {
        "histogram": "add",
        "largest": "max",
        "smallest": "min",
        "signed_largest": "max",
        "counter": "inc",
        "lock": "cas",
    }[scenario]
    operand = (
        "state[x[i]], 1"
        if scenario == "histogram"
        else "state[0], 7"
        if scenario == "counter"
        else "state[0], 0, 1"
        if scenario == "lock"
        else "state[0], x[i]"
    )
    initial = -np.inf if scenario == "largest" else 1 << 50 if scenario == "smallest" else 0
    if scenario == "signed_largest":
        initial = -(1 << 63)
    state = np.full(8 if scenario == "histogram" else 1, initial, dtype=DTYPES[dtype])
    out = np.zeros_like(x)
    file = tmp_path / "contention.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\nfrom krn.facts.intrinsics import atomic_block, atomic_system\n"
        f"@kernel\ndef run(x: Vector[T.{dtype}], state: Vector[T.{dtype}], out: Vector[T.{dtype}]):\n"
        f"    i = grid.thread_rank()\n    out[i] = atomic_{scope}.{op}({operand})\n"
    )
    if gpu is not None and dtype == "f32":
        pytest.skip("Metal contention coverage here uses integer atomics")
    ir = capture(file, "run", target=target)
    launch = Blocks(1 if scope == "block" else 2, threads=64 if scope == "block" else 32)
    if card is not None:
        CUDARun(card, ir).run(launch, [x, state, out])
    elif metal is not None and gpu is not None:
        MetalRun(gpu, ir).run(launch, [x, state, out])
    elif backend == "c":
        CPUProgram(ir, emit(ir, target))(x, state, out, launch=launch)
    else:
        emulate(ir, target).run(launch, [x, state, out])
    expected = (
        np.bincount(x.astype(np.int64), minlength=8)
        if scenario == "histogram"
        else [x.max()]
        if scenario in {"largest", "signed_largest"}
        else [x.min()]
        if scenario == "smallest"
        else [0]
        if scenario == "counter"
        else [1]
    )
    np.testing.assert_array_equal(state, expected)
    if scenario == "counter":
        np.testing.assert_array_equal(np.sort(out), np.repeat(np.arange(8), 8))
    elif scenario == "lock":
        assert np.count_nonzero(out == 0) == 1


@pytest.mark.parametrize("scope", ["block", "system"])
@pytest.mark.parametrize("dtype", ["u32", "f32", "u64"])
def test_scoped_atomic_codegen_uses_scope(tmp_path: Path, scope: str, dtype: str) -> None:
    op = "max" if dtype == "f32" else "inc" if dtype == "u64" else "add"
    case = AtomicCase(f"atomic_{scope}.{op}", dtype)
    cuda = emit(capture(case.file(tmp_path), "family_case", target=GPU), GPU).source
    assert (f"atomicCAS_{scope}" if dtype != "u32" else f"atomicAdd_{scope}") in cuda
    target = Target.of("gfx942")
    hip = emit(capture(case.file(tmp_path), "family_case", target=target), target).source
    assert ("__HIP_MEMORY_SCOPE_WORKGROUP" if scope == "block" else "__HIP_MEMORY_SCOPE_SYSTEM") in hip
    assert "__hip_atomic_" in hip


@pytest.mark.parametrize("case", CACHE_EFFECT_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
@pytest.mark.parametrize("arch", ["sm_90a", "gfx942", "apple_m4", "avx512"])
def test_cache_effect_local(tmp_path: Path, case: CacheEffectCase, arch: str) -> None:
    target = Target.of(arch)
    case.run(tmp_path, target, "c" if target.vendor == "cpu" else "emulator")


@pytest.mark.parametrize("case", CACHE_EFFECT_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
def test_cache_effect_metal(tmp_path: Path, case: CacheEffectCase, family_metal: MetalDevice) -> None:
    case.run(tmp_path, family_metal.target, "metal", lambda ir: MetalRun(family_metal, ir))


@pytest.mark.parametrize("case", CACHE_EFFECT_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
def test_cache_effect_cuda(tmp_path: Path, case: CacheEffectCase, family_card: CUDADevice) -> None:
    case.run(tmp_path, family_card.target, "cuda", lambda ir: CUDARun(family_card, ir))


@pytest.mark.parametrize("kind", ["ref", "out"])
@pytest.mark.parametrize("space", ["", "global", "shared"])
@pytest.mark.parametrize("shared", [False, True])
def test_family_element_access_and_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    space: str,
    shared: bool,
) -> None:
    entry = catalog.Entry(
        "cache.access_test",
        "test access",
        "({0})",
        "({0})",
        "({0})",
        "({0})",
        signature=catalog.sig(f"x: {kind} {space} u32 -> u32"),
    )
    monkeypatch.setitem(catalog.FUNCTIONS, entry.name, entry)
    monkeypatch.setitem(catalog.BY_NAME, entry.name, entry)
    local = "    tile = shared_array(T.u32, 1)\n" if shared else ""
    value = "tile[0]" if shared else "x[0]"
    file = tmp_path / "access.py"
    file.write_text(
        "from krn import T, Vector, kernel, shared_array\nfrom krn.facts.intrinsics import cache\n"
        f"@kernel\ndef access(x: Vector[T.u32], out: Vector[T.u32]):\n{local}"
        f"    out[0] = cache.access_test({value})\n"
    )
    if space and shared != (space == "shared"):
        with pytest.raises(CaptureError, match=f"argument `x` must be a {space} tensor element"):
            capture(file, "access", target=GPU)
    else:
        ir = capture(file, "access", target=GPU)
        path = ir.shared[0].path if shared else ("x",)
        assert (path in ir.written) == (kind == "out")


def test_all_function_entries_have_vendor_compile_cases() -> None:
    assert set(catalog.FUNCTIONS) - VIDEO <= {c.name for c in COMPILE_CASES}


@pytest.mark.parametrize(
    "entry",
    [
        e
        for e in catalog.FUNCTIONS.values()
        if e.name.split(".")[0]
        in {
            "atomic_block",
            "atomic_system",
            "cache",
            "async_copy",
            "mbarrier",
            "bulk",
            "dsmem",
            "rounded",
        }
    ],
    ids=lambda e: e.name,
)
def test_new_family_contract_error_is_at_user_call(tmp_path: Path, entry: catalog.Entry) -> None:
    signature = entry.signature
    assert signature is not None
    arguments = [
        "half[0]" if p.kind.startswith(("ref ", "out ")) or p.kind == "const" else "T.u32(1)" for p in signature.params
    ]
    if not arguments:
        arguments = ["T.u32(1)"]
    file = tmp_path / "contract.py"
    file.write_text(
        "from krn import T, Vector, kernel\n"
        "from krn.facts.intrinsics import atomic_block, atomic_system, cache, "
        "async_copy, mbarrier, bulk, dsmem, rounded\n"
        "@kernel\ndef bad(half: Vector[T.f16]):\n"
        f"    {entry.name}({', '.join(arguments)})\n"
    )
    with pytest.raises(CaptureError) as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.file == str(file) and error.value.span.line == 5


@pytest.mark.parametrize("case", ASYNC_CASES, ids=lambda c: f"{c.name}-{c.dtype}-{c.count}-{c.pending}")
@pytest.mark.parametrize("arch", ["sm_90a", "gfx942", "apple_m4", "avx512"])
def test_async_copy_local(tmp_path: Path, case: AsyncCase, arch: str) -> None:
    target = Target.of(arch)
    case.run(tmp_path, target, "c" if target.vendor == "cpu" else "emulator")


@pytest.mark.parametrize("case", ASYNC_CASES, ids=lambda c: f"{c.name}-{c.dtype}-{c.count}-{c.pending}")
def test_async_copy_metal(tmp_path: Path, case: AsyncCase, family_metal: MetalDevice) -> None:
    case.run(tmp_path, family_metal.target, "metal", lambda ir: MetalRun(family_metal, ir))


@pytest.mark.parametrize("case", ASYNC_CASES, ids=lambda c: f"{c.name}-{c.dtype}-{c.count}-{c.pending}")
def test_async_copy_cuda(tmp_path: Path, case: AsyncCase, family_card: CUDADevice) -> None:
    case.run(tmp_path, family_card.target, "cuda", lambda ir: CUDARun(family_card, ir))


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("async_copy.copy(x[0], x[0], 1)", "argument `dst` must be a shared"),
        ("async_copy.copy(tile[0], tile[0], 1)", "argument `src` must be a global"),
        ("async_copy.copy(tile[0], x[0], 3)", "argument `count` must be 1, 2, or 4"),
        ("async_copy.copy(tile[0], x[0], x[0])", "capture-time constant"),
        ("async_copy.wait(8)", "argument `pending` must be between 0 and 7"),
        ("async_copy.wait(-1)", "argument `pending` must be between 0 and 7"),
    ],
)
def test_async_contract_errors(tmp_path: Path, expression: str, reason: str) -> None:
    file = tmp_path / "async_bad.py"
    file.write_text(
        "from krn import T, Vector, shared_array, kernel\nfrom krn.facts.intrinsics import async_copy\n"
        "@kernel\ndef bad(x: Vector[T.u32]):\n    tile = shared_array(T.u32, 4)\n"
        f"    {expression}\n"
    )
    with raises_literal(CaptureError, reason) as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.line == 6


@pytest.mark.parametrize("backend", ["emulator", "c"])
def test_async_copy_contiguous_span_crosses_matrix_rows(tmp_path: Path, backend: str) -> None:
    file = tmp_path / "matrix_copy.py"
    file.write_text(
        "from krn import T, Matrix, Vector, shared_array, kernel\n"
        "from krn.facts.intrinsics import async_copy\n@kernel\n"
        "def run(x: Matrix[T.i32], out: Vector[T.i32]):\n    tile = shared_array(T.i32, 4)\n"
        "    async_copy.copy(tile[0], x[1, 1], 4)\n    async_copy.commit()\n    async_copy.wait(0)\n"
        "    for i in range(4):\n        out[i] = tile[i]\n"
    )
    target = GPU if backend == "emulator" else Target.of("avx512")
    ir, launch = capture(file, "run", target=target), Blocks(1, threads=1)
    x, out = np.arange(9, dtype=np.int32).reshape(3, 3), np.zeros(4, np.int32)
    if backend == "emulator":
        emulate(ir, target).run(launch, [x, out])
    else:
        CPUProgram(ir, emit(ir, target))(x, out, launch=launch)
    np.testing.assert_array_equal(out, x.ravel()[4:8])


@pytest.mark.parametrize("case", CUDA_MEMORY_CASES, ids=lambda c: f"{c.name}-{c.dtype}")
@pytest.mark.parametrize("arch", ["sm_80", "sm_90a", "sm_100a", "sm_103a", "sm_120a", "apple_m4", "gfx942", "avx512"])
def test_cuda_memory_sources_and_refusals(tmp_path: Path, case: CUDAMemoryCase, arch: str) -> None:
    """Vendor compile cases cover supported SMs; only the transaction-copy pipeline runs on sm_120a."""
    target = Target.of(arch)
    file = case.file(tmp_path)
    if case.supported(target):
        ir = capture(file, "family_case", target=target)
        source = emit(ir, target).source
        assert "__cvta_generic_to_shared" in source
        with raises_literal(NotImplementedError, case.name):
            emulate(ir, target)
        if case.name == "mbarrier.wait_parity":
            assert "(1u << 26)" in source and "__trap();" in source
    else:
        with pytest.raises(CaptureError, match="targets that have it:") as error:
            capture(file, "family_case", target=target)
        assert error.value.span is not None and error.value.span.line == 7


def bulk_tile_file(directory: Path) -> Path:
    file = directory / "bulk_tile.py"
    file.write_text(
        "from krn import T, Vector, block, thread, kernel, shared_array\n"
        "from krn.facts.intrinsics import mbarrier, bulk\n"
        "@kernel\ndef bulk_tile(x: Vector[T.f32], out: Vector[T.f32]):\n"
        "    tile = shared_array(T.f32, 64)\n    bar = shared_array(T.u64, 1)\n"
        "    i = block.thread_rank()\n    if i == 0:\n        mbarrier.init(bar[0], 1)\n"
        "        thread.fence_proxy_async()\n        mbarrier.arrive_expect_tx(bar[0], 256)\n"
        "        bulk.copy(tile[0], x[0], 256, bar[0])\n    block.sync()\n"
        "    mbarrier.wait_parity(bar[0], 0)\n    out[i] = tile[i]\n"
    )
    return file


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
def test_bulk_tile_nvrtc(tmp_path: Path, arch: str, nvrtc: Nvrtc) -> None:
    """Compile checked on sm_90a/sm_120a; hardware execution is restricted to sm_120a."""
    target = Target.of(arch)
    source = emit(capture(bulk_tile_file(tmp_path), "bulk_tile", target=target), target).source
    assert nvrtc.compile(source, arch, "cubin").data[:4] == b"\x7fELF"


def test_bulk_tile_sm120a(tmp_path: Path, family_card: CUDADevice) -> None:
    """Compile checked on supported SMs; this bounded transaction-copy test runs on sm_120a only."""
    if family_card.target.arch != "sm_120a":
        pytest.skip("the transaction-copy hardware test is validated on sm_120a only")
    ir = capture(bulk_tile_file(tmp_path), "bulk_tile", target=family_card.target)
    x = np.random.default_rng(5).standard_normal(64).astype(np.float32)
    out = np.zeros_like(x)
    CUDARun(family_card, ir).run(Blocks(1, threads=64), [x, out])
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize("mode", ["rn", "rz", "ru", "rd"])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("op", ["add", "mul", "div", "fma", "rcp"])
def test_rounded_overflow_bits(op: str, mode: str, negative: bool) -> None:
    maximum, smallest = float(np.finfo(np.float32).max), 2.0**-149
    signed_max = -maximum if negative else maximum
    values = {
        "add": (signed_max, signed_max),
        "mul": (signed_max, 2),
        "div": (signed_max, smallest),
        "fma": (signed_max, 2, 0),
        "rcp": (-smallest if negative else smallest,),
    }[op]
    reference = catalog.FUNCTIONS[f"rounded.{op}_{mode}"].reference
    assert reference is not None
    result = reference(*values, bits=32, kind="f")
    assert not isinstance(result, tuple)
    infinite = mode == "rn" or mode == ("rd" if negative else "ru")
    expected = (0x80000000 if negative else 0) | (0x7F800000 if infinite else 0x7F7FFFFF)
    assert int(np.float32(result).view(np.uint32)) == expected


@pytest.mark.parametrize("mode", ["rn", "rz", "ru", "rd"])
@pytest.mark.parametrize("negative", [False, True])
def test_rounded_half_subnormal(mode: str, negative: bool) -> None:
    reference = catalog.FUNCTIONS[f"rounded.mul_{mode}"].reference
    assert reference is not None
    result = reference(-(2.0**-149) if negative else 2.0**-149, 0.5, bits=32, kind="f")
    assert not isinstance(result, tuple)
    expected = (0x80000000 if negative else 0) | int(mode == ("rd" if negative else "ru"))
    assert int(np.float32(result).view(np.uint32)) == expected


def test_spell_errors_and_signature_metadata() -> None:
    with pytest.raises(catalog.SpellError, match="needs more operands"):
        catalog.spell("bits.bfi", GPU, "a", "b", "c")
    entry = replace(catalog.BY_NAME["bits.bfe"], sm=100)
    assert entry.lowering(GPU) == ""
    assert entry.lowering(Target.of("sm_120a"))
    assert catalog.sig("x: T, output: ref u32, table: const -> None", T="u32").params[1].kind == "ref u32"
    signature = catalog.sig("src: ref global T, dst: out shared u32 -> None", T="u32")
    assert [p.kind for p in signature.params] == ["ref global T", "out shared u32"]


@pytest.mark.parametrize("availability", ["lowering", "sm"])
def test_family_availability_keeps_target_and_user_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, availability: str
) -> None:
    original = catalog.FUNCTIONS["bits.bfe"]
    entry = (
        replace(original, nvidia="", note="test missing instruction")
        if availability == "lowering"
        else replace(original, sm=100)
    )
    monkeypatch.setitem(catalog.FUNCTIONS, entry.name, entry)
    monkeypatch.setitem(catalog.BY_NAME, entry.name, entry)
    file = Case(entry.name, ("u32", "u32", "u32")).file(tmp_path)
    with pytest.raises(
        CaptureError, match=r"`bits.bfe` has no lowering for target sm_90a:.*targets that have it:"
    ) as error:
        capture(file, "family_case", target=GPU)
    assert error.value.span is not None and error.value.span.line == 7


MSL_HEADER = """#include <metal_stdlib>
#define INFINITY __builtin_inff()
namespace metal {
template<class T> inline T select(T no, T yes, bool condition) { return condition ? yes : no; }
template<class T> inline T simd_sum(T x) { return x; }
template<class T> inline T simd_min(T x) { return x; }
template<class T> inline T simd_max(T x) { return x; }
template<class T> inline T simd_and(T x) { return x; }
template<class T> inline T simd_or(T x) { return x; }
template<class T> inline T simd_xor(T x) { return x; }
inline float fmin(float x, float y) { return __builtin_fminf(x,y); }
inline float fmax(float x, float y) { return __builtin_fmaxf(x,y); }
inline float min(float x, float y) { return fmin(x,y); }
inline float max(float x, float y) { return fmax(x,y); }
inline bool isnan(float x) { return __builtin_isnan(x); }
inline bool isinf(float x) { return __builtin_isinf(x); }
inline float tan(float x) { return __builtin_tanf(x); }
inline float asin(float x) { return __builtin_asinf(x); }
inline float acos(float x) { return __builtin_acosf(x); }
inline float atan(float x) { return __builtin_atanf(x); }
inline float atan2(float y, float x) { return __builtin_atan2f(y,x); }
inline float fmod(float x, float y) { return __builtin_fmodf(x,y); }
inline float rsqrt(float x) { return 1.0f/__builtin_sqrtf(x); }
inline float frexp(float x, int& e) { return __builtin_frexpf(x,&e); }
inline float ldexp(float x, int e) { return __builtin_ldexpf(x,e); }
}
namespace fast {
using metal::exp; using metal::exp10; using metal::log; using metal::log2; using metal::log10;
using metal::sin; using metal::cos; using metal::tan; using metal::pow;
using metal::sqrt; using metal::rsqrt; using metal::exp2; using metal::tanh;
inline float reciprocal(float x) { return 1.0f/x; }
inline float divide(float x, float y) { return x/y; }
}
"""


@pytest.mark.parametrize("case", [c for c in COMPILE_CASES if "f64" not in c.dtypes], ids=lambda c: c.name)
def test_family_msl_shapes(tmp_path: Path, case: CompileCase) -> None:
    target = Target.of("apple_m4")
    if not case.supported(target):
        with pytest.raises(CaptureError):
            capture(case.file(tmp_path), "family_case", target=target)
        return
    source = emit(capture(case.file(tmp_path), "family_case", target=target), target).source
    assert "double" not in source_words(source)
    header, file = tmp_path / "extra.h", tmp_path / "family.metal"
    header.write_text(MSL_HEADER)
    file.write_text(source)
    done = subprocess.run(
        [
            CLANG[0],
            "-x",
            "c++",
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(Path(__file__).resolve().parents[1] / "msl_shim"),
            "-include",
            str(header),
            str(file),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr[-6000:]


@pytest.mark.parametrize(
    ("signature", "expression", "error"),
    [
        ("x: ref T -> T", "a[0]", ""),
        ("x: ref u32 -> u32", "a[0]", ""),
        ("x: ref T -> T", "T.u32(1)", "tensor element"),
        ("x: ref T -> T", "a", "tensor element"),
        ("x: ref T -> T", "half[0]", "supported tensor element"),
        ("x: ref T -> T", "wrong[0]", "accepts u32"),
        ("x: ref u32 -> u32", "wrong[0]", "argument `x`"),
    ],
)
def test_family_reference_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signature: str, expression: str, error: str
) -> None:
    entry = catalog.Entry(
        "bits.ref_test",
        "test tensor lvalues",
        "({0})",
        "({0})",
        "({0})",
        "({0})",
        signature=catalog.sig(signature, T="u32"),
    )
    monkeypatch.setitem(catalog.FUNCTIONS, entry.name, entry)
    monkeypatch.setitem(catalog.BY_NAME, entry.name, entry)
    file = tmp_path / "ref.py"
    file.write_text(
        "from krn import T, Vector, kernel\nfrom krn.facts.intrinsics import bits\n"
        "@kernel\ndef ref_test(a: Vector[T.u32], wrong: Vector[T.i32], half: Vector[T.f16]):\n"
        f"    bits.ref_test({expression})\n"
    )
    if error:
        with pytest.raises(CaptureError, match=error) as failure:
            capture(file, "ref_test", target=GPU)
        assert failure.value.span is not None and failure.value.span.line == 5
    else:
        assert "a[" in emit(capture(file, "ref_test", target=GPU), GPU).source


@pytest.mark.parametrize(
    "expression",
    [
        "bits.lop3(a[0], a[0], a[0], a[0])",
        "bits.lop3(a[0], a[0], a[0], 1.5)",
        "bits.byte_perm(a[0], a[0])",
        "bits.byte_perm(a[0], a[0], selector=a[0], bad=0)",
        "libm.hypot(x[0], y[0])",
    ],
)
def test_family_binding_errors(tmp_path: Path, expression: str) -> None:
    file = tmp_path / "binding.py"
    file.write_text(
        "from krn import T, Vector, kernel\nfrom krn.facts.intrinsics import bits, libm\n"
        "@kernel\ndef bad(a: Vector[T.u32], x: Vector[T.f32], y: Vector[T.f64]):\n"
        f"    {expression}\n"
    )
    with pytest.raises(CaptureError) as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.line == 5


def test_family_statement_and_helper_deduplication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = "@inline unsigned _test_identity(unsigned x) { return x; }"
    for name in ("bits.effect_one", "bits.effect_two"):
        entry = catalog.Entry(
            name,
            "test statement helper",
            "_test_identity({0})",
            "_test_identity({0})",
            "_test_identity({0})",
            "_test_identity({0})",
            signature=catalog.sig("x: u32 -> None"),
            helper={"*": helper},
        )
        monkeypatch.setitem(catalog.FUNCTIONS, name, entry)
        monkeypatch.setitem(catalog.BY_NAME, name, entry)
    file = tmp_path / "effect.py"
    file.write_text(
        "from krn import T, kernel\nfrom krn.facts.intrinsics import bits\n"
        "@kernel\ndef effects(x: T.u32):\n    bits.effect_one(x)\n    bits.effect_two(x)\n"
    )
    for arch, qualifier in (
        ("avx512", "static inline"),
        ("sm_90a", "__device__ __forceinline__"),
        ("apple_m4", "inline"),
    ):
        target = Target.of(arch)
        source = emit(capture(file, "effects", target=target), target).source
        dtype = "uint" if arch == "apple_m4" else "unsigned"
        assert " ".join(source.split()).count(f"{qualifier} {dtype} _test_identity") == 1
        assert source.index("_test_identity") < source.index("void effects")


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.name.startswith(("half2.", "bfloat162."))], ids=lambda c: c.name
)
def test_metal_packed_algorithm_on_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: Case) -> None:
    helper = (
        case.entry.helper_for("apple")
        .replace("return as_type<unsigned>(x);", "union {float f; unsigned u;} v; v.f=x; return v.u;")
        .replace("return as_type<float>(x);", "union {float f; unsigned u;} v; v.u=x; return v.f;")
    )
    for name in ("fma", "fmin", "fmax"):
        helper = helper.replace(f"metal::{name}(", f"__builtin_{name}f(")
    entry = replace(case.entry, helper={"cpu": helper})
    monkeypatch.setitem(catalog.BY_NAME, case.name, entry)
    case.run(tmp_path, Target.of("avx512"), "c")
