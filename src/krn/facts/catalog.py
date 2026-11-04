"""Every device intrinsic with its spelling per vendor, and where numba-cuda-mlir's `cuda` namespace lands in it.

A lowering is a format string over the operands `{0} {1} {2}`: `{ll}` is `ll` for a 64-bit operand and empty for 32 bits,
`{bits}` the operand width, `{s}` is the operand kind (u, s, or f), `{f}` is `f` for float32, `{warp}` the lane count and `{full}` the mask of all
lanes. An empty string means the vendor has no spelling and the capture refuses the call, naming the target; the
`note` of the entry says what would be needed. The CUDA and C columns are compiled by the prototype (NVRTC and clang);
the HIP and MSL columns are written from the vendors' documentation and not yet compiled, so they are a plan.

The tables are data, not code paths: the prototype's renderers read their strings from here, and `COVERAGE` is what
the tests check against the reference page of numba-cuda-mlir 0.6.0 (the names of its `cuda` namespace, copied from
nvidia.github.io/numba-cuda-mlir/latest/reference/kernel.html on 2026-10-31).
"""

import math
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial, reduce
from operator import and_, or_, xor

from krn.compiler.errors import TargetError
from krn.frontend.atomic import OPERANDS, AtomicOp

from .targets import Feature, Nvidia, Target

type Number = int | float | bool
type Reference = Callable[..., Number | tuple[Number, ...]]


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class Signature:
    params: tuple[Param, ...]
    result: str
    types: tuple[str, ...]


def sig(text: str, *, T: str = "") -> Signature:
    """Parse scalar, constant, and ref/out element contracts with optional global/shared space."""
    parameters, result = text.split("->")
    return Signature(
        tuple(
            Param(name.strip(), " ".join(kind.split()))
            for item in parameters.split(",")
            if item.strip()
            for name, kind in (item.split(":", 1),)
        ),
        result.strip(),
        tuple(T.split()),
    )


class SpellError(LookupError):
    """A target or operand list cannot instantiate a catalog template."""


class IntrinsicTargetError(TargetError, SpellError):
    """A catalog intrinsic has no implementation for the selected target."""


@dataclass(frozen=True, slots=True)
class Entry:
    name: str
    summary: str
    nvidia: str
    amd: str
    apple: str
    cpu: str
    feature: Feature | None = None
    note: str = ""
    numba: tuple[str, ...] = field(default=())  # the names in numba-cuda-mlir this entry covers
    helper: str | Mapping[str, str] = ""
    signature: Signature | None = None
    reference: Reference | None = None
    sm: int = 0
    parts: tuple[str, ...] = ()
    atomic: tuple[AtomicOp, str] | None = None

    def lowering(self, target: Target) -> str:
        if isinstance(target, Nvidia) and target.sm < self.sm:
            return ""
        return getattr(self, target.vendor)

    def helper_for(self, vendor: str) -> str:
        return self.helper if isinstance(self.helper, str) else self.helper.get(vendor, self.helper.get("*", ""))


def _axes(name: str, summary: str, nvidia: str, amd: str, apple: str, cpu: dict[str, str], numba: str) -> list[Entry]:
    return [
        Entry(
            f"{name}.{a}",
            summary,
            nvidia.format(a=a),
            amd.format(a=a),
            apple.format(a=a),
            cpu[a],
            numba=(f"{numba}.{a}",),
        )
        for a in "xyz"
    ]


_LOCKSTEP = (
    "the C backend runs the threads of a block one after the other, so lanes cannot exchange values; the emulator can"
)
_THREAD_RANK = "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)"
_BLOCK_RANK = "(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)"
_BLOCK_THREADS = "(blockDim.x * blockDim.y * blockDim.z)"
_GRID_BLOCKS = "(gridDim.x * gridDim.y * gridDim.z)"
_MSL_THREAD_RANK = "(_thread.x + _thread.y * _blockdim.x + _thread.z * _blockdim.x * _blockdim.y)"
_MSL_BLOCK_RANK = "(_block.x + _block.y * _griddim.x + _block.z * _griddim.x * _griddim.y)"
_MSL_BLOCK_THREADS = "(_blockdim.x * _blockdim.y * _blockdim.z)"
_MSL_GRID_BLOCKS = "(_griddim.x * _griddim.y * _griddim.z)"
_C_BLOCK_RANK = "((unsigned)(bx + by * gx + bz * gx * gy))"
_GRID_SYNC = """__device__ unsigned long long _grid_arrived;
__device__ void _grid_sync() {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        __threadfence();
        unsigned long long blocks = (unsigned long long)gridDim.x * gridDim.y * gridDim.z;
        unsigned long long ticket = atomicAdd(&_grid_arrived, 1ull);
        unsigned long long goal = (ticket / blocks + 1) * blocks;
        while (*(volatile unsigned long long*)&_grid_arrived < goal) {}
        __threadfence();
    }
    __syncthreads();
}"""
_CLUSTER_NOTE = "clusters need NVIDIA cluster instructions and a cluster launch"
_WARP_REDUCE_AMD = """@inline unsigned _warp_reduce(unsigned x, unsigned long long mask, int op, int signed_word) {
    unsigned result = 0;
    int first = 1;
    for (int lane = 0; lane < 64; ++lane) {
        unsigned y = __shfl(x, lane, 64);
        if ((mask >> lane) & 1ull) {
            if (first) { result = y; first = 0; }
            else if (op == 0) result += y;
            else if (op == 1) result = (signed_word ? (int)y < (int)result : y < result) ? y : result;
            else if (op == 2) result = (signed_word ? (int)y > (int)result : y > result) ? y : result;
            else if (op == 3) result &= y;
            else if (op == 4) result |= y;
            else result ^= y;
        }
    }
    return result;
}"""


def _warp_reduce_reference(op: str, *values: int, bits: int, kind: str) -> int:
    mask = (1 << bits) - 1
    words = [v & mask for v in values]
    ordered = [_signed(v, bits) for v in words] if kind == "s" else words
    if op == "add":
        result = sum(words)
    elif op in ("min", "max"):
        result = (min if op == "min" else max)(ordered)
    else:
        result = reduce({"and": and_, "or": or_, "xor": xor}[op], words)
    return _signed(result & mask, bits) if kind == "s" else result & mask


# Standard-library reference semantics; vendor lowerings are independent below.
def _inverse_erf(x: float) -> float:
    if abs(x) == 1:
        return math.copysign(math.inf, x)
    if abs(x) > 1:
        return math.nan
    w = -math.log1p(-x * x)
    coefficients = (
        (
            2.81022636e-08,
            3.43273939e-07,
            -3.5233877e-06,
            -4.39150654e-06,
            0.00021858087,
            -0.00125372503,
            -0.00417768164,
            0.246640727,
            1.50140941,
        )
        if w < 5
        else (
            -0.000200214257,
            0.000100950558,
            0.00134934322,
            -0.00367342844,
            0.00573950773,
            -0.0076224613,
            0.00943887047,
            1.00167406,
            2.83297682,
        )
    )
    w = w - 2.5 if w < 5 else math.sqrt(w) - 3
    p = 0.0
    for coefficient in coefficients:
        p = coefficient + p * w
    value = p * x
    for _ in range(4):
        value -= (math.erf(value) - x) / (2 / math.sqrt(math.pi) * math.exp(-value * value))
    return value


def _float_bits(x: float) -> int:
    try:
        return int.from_bytes(struct.pack("<f", x), "little")
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def _bits_float(x: int) -> float:
    return struct.unpack("<f", x.to_bytes(4, "little"))[0]


def _small_fma(a: float, b: float, c: float, *, bf: bool) -> float:
    if not all(map(math.isfinite, (a, b, c))):
        try:
            return math.fma(a, b, c)
        except ValueError:
            return math.nan
    value = Fraction(a) * Fraction(b) + Fraction(c)
    if not value:
        return math.fma(a, b, c)
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < Fraction(2) ** exponent:
        exponent -= 1
    mantissa, low, high = (7, -126, 127) if bf else (10, -14, 15)
    quantum = Fraction(2) ** (max(low, min(high, exponent)) - mantissa)
    rounded = round(magnitude / quantum) * quantum
    if rounded >= Fraction(2) ** (high + 1):
        return math.copysign(math.inf, float(value))
    return math.copysign(float(rounded), float(value))


def _math_reference(name: str, *values: Number, bits: int, kind: str) -> Number | tuple[Number, ...]:
    """Standard-library semantics, independent of vendor approximations."""
    x = float(values[0])
    match name:
        case "sincos":
            return math.sin(x), math.cos(x)
        case "frexp":
            return math.frexp(x)
        case "erfinv":
            return _inverse_erf(x)
        case "erfcinv":
            value = _inverse_erf(1 - x)
            if math.isfinite(value):
                for _ in range(4):
                    value += (math.erfc(value) - x) / (2 / math.sqrt(math.pi) * math.exp(-value * value))
            return value
        case "erfcx":
            if x < 25:
                return math.exp(x * x) * math.erfc(x)
            term, total = 1.0, 1.0
            for n in range(1, 20):
                term *= -(2 * n - 1) / (2 * x * x)
                total += term
            return total / (math.sqrt(math.pi) * x)
        case "fma":
            if bits == 16:
                return _small_fma(*map(float, values), bf=kind == "bf")
            if bits == 64:
                try:
                    return math.fma(*map(float, values))
                except ValueError:
                    return math.nan
                except OverflowError:
                    return math.copysign(math.inf, x * float(values[1]))
            return _rounded_reference("fma", "rn", *values, bits=bits, kind=kind)
        case "reciprocal":
            return math.copysign(math.inf, x) if x == 0 else 1 / x
        case "rsqrt":
            return math.copysign(math.inf, x) if x == 0 else math.nan if x < 0 else 1 / math.sqrt(x)
        case "nextafter":
            y = float(values[1])
            if bits == 64:
                return math.nextafter(x, y)
            if math.isnan(x) or math.isnan(y):
                return math.nan
            if x == y:
                return y
            word = _float_bits(x)
            if x == 0:
                return _bits_float((0x80000000 if y < 0 else 0) | 1)
            return _bits_float(word + (1 if (y > x) == (x > 0) else -1))
        case "frexp_mantissa":
            return math.frexp(x)[0]
        case "frexp_exponent":
            return math.frexp(x)[1]
        case "exp10":
            try:
                return 10.0**x
            except OverflowError:
                return math.inf
        case "div":
            y = float(values[1])
            if y == 0:
                return (
                    math.nan
                    if x == 0 or math.isnan(x)
                    else math.copysign(math.inf, math.copysign(1, x) * math.copysign(1, y))
                )
            return x / y
        case "ldexp":
            try:
                return math.ldexp(x, int(values[1]))
            except OverflowError:
                return math.copysign(math.inf, x)
    functions: dict[str, Callable[..., float]] = {
        "erf": math.erf,
        "erfc": math.erfc,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "hypot": math.hypot,
        "fmod": math.fmod,
        "sin": math.sin,
        "cos": math.cos,
        "sqrt": math.sqrt,
        "tanh": math.tanh,
        "exp2": math.exp2,
        "exp": math.exp,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "pow": math.pow,
        "copysign": math.copysign,
    }
    if name in {"log", "log2", "log10"} and x == 0:
        return -math.inf
    try:
        return functions[name](*values)
    except ValueError:
        return math.nan
    except OverflowError:
        negative = name == "pow" and x < 0 and float(values[1]).is_integer() and int(values[1]) % 2 == 1
        return -math.inf if negative else math.inf


def _bit_reference(name: str, *values: int, bits: int, kind: str) -> int:
    mask = (1 << bits) - 1
    x = values[0] & mask
    match name:
        case "bfe":
            position, length = (v & 255 for v in values[1:])
            if not length:
                return 0
            sign = (x >> min(position + length - 1, bits - 1)) & 1 if kind == "s" else 0
            result = (x >> position) & ((1 << length) - 1)
            if sign:
                result |= mask ^ ((1 << min(length, bits)) - 1)
                if position + length > bits:
                    result |= mask ^ ((1 << max(bits - position, 0)) - 1)
            return result & mask
        case "bfi":
            base, position, length = values[1:]
            position, length = position & 255, length & 255
            field = (((1 << length) - 1) << position) & mask
            return (base & ~field | x << position & field) & mask
        case "byte_perm":
            y, selector = values[1:]
            source = x | y << 32
            result = 0
            for n in range(4):
                control = selector >> (4 * n) & 15
                byte = source >> (8 * (control & 7)) & 255
                result |= ((255 if byte & 128 else 0) if control & 8 else byte) << (8 * n)
            return result
        case "lop3":
            b, c, table = values[1:]
            return sum(
                ((table >> (((x >> i & 1) << 2) | ((b >> i & 1) << 1) | (c >> i & 1))) & 1) << i for i in range(32)
            )
    hi, shift = values[1:]
    shift = min(shift, 32) if name.endswith("c") else shift & 31
    pair = hi << 32 | x
    return ((pair << shift >> 32) if name.startswith("funnelshift_l") else pair >> shift) & 0xFFFFFFFF


def _signed(x: int, width: int) -> int:
    return (x ^ (1 << (width - 1))) - (1 << (width - 1))


def _dot_reference(name: str, a: int, b: int, c: int, *, bits: int, kind: str) -> int:
    width, count, start = (8, 4, 0) if name == "dp4a" else (16, 2, 2 if name == "dp2a_hi" else 0)
    for i in range(count):
        left, right = a >> (width * i) & ((1 << width) - 1), b >> (8 * (i + start)) & 255
        c += (_signed(left, width) if kind == "s" else left) * (_signed(right, 8) if kind == "s" else right)
    return c & 0xFFFFFFFF


_SIMD_ROWS = (
    ("vadd", "x+y"), ("vsub", "x-y"), ("vaddss", "x+y"), ("vaddus", "x+y"),
    ("vsubss", "x-y"), ("vsubus", "x-y"), ("vabsdiffs", "x>y ? x-y : y-x"), ("vabsdiffu", "x>y ? x-y : y-x"),
    ("vmaxs", "x>y ? x : y"), ("vmaxu", "x>y ? x : y"), ("vmins", "x<y ? x : y"), ("vminu", "x<y ? x : y"),
    ("vavgs", "(x+y+(x+y>=0))>>1"), ("vavgu", "(x+y+1)>>1"), ("vhaddu", "(x+y)>>1"),
    ("vneg", "-x"), ("vnegss", "-x"), ("vabs", "x<0 ? -x : x"), ("vabsss", "x<0 ? -x : x"),
    *((prefix+name, f"x{relation}y") for prefix in ("vcmp", "vset") for name, relation in (
        ("eq", "=="), ("ne", "!="), ("ges", ">="), ("geu", ">="), ("gts", ">"), ("gtu", ">"),
        ("les", "<="), ("leu", "<="), ("lts", "<"), ("ltu", "<"),
    )),
)  # fmt: skip
SIMD_UNARY = ("vneg", "vnegss", "vabs", "vabsss")


def _simd_reference(op: str, lanes: int, a: int, b: int = 0, *, bits: int, kind: str) -> int:
    width, result = 32 // lanes, 0
    mask = (1 << width) - 1
    for lane in range(lanes):
        x, y = (a >> (lane * width)) & mask, (b >> (lane * width)) & mask
        if (op.endswith("s") and not op.endswith("us")) or op in ("vneg", "vabs"):
            x, y = _signed(x, width), _signed(y, width)
        if op.startswith(("vcmp", "vset")):
            relation = op[4:].rstrip("su")
            yes = {
                "eq": x == y,
                "ne": x != y,
                "ge": x >= y,
                "gt": x > y,
                "le": x <= y,
                "lt": x < y,
            }[relation]
            value = (mask if op.startswith("vcmp") else 1) if yes else 0
        elif op.startswith("vadd"):
            value = x + y
        elif op.startswith("vsub"):
            value = x - y
        elif op.startswith("vabsdiff"):
            value = abs(x - y)
        elif op.startswith("vmax"):
            value = max(x, y)
        elif op.startswith("vmin"):
            value = min(x, y)
        elif op.startswith("vavg"):
            value = (x + y + (1 if op == "vavgu" or x + y >= 0 else 0)) >> 1
        elif op == "vhaddu":
            value = (x + y) >> 1
        elif op.startswith("vneg"):
            value = -x
        else:
            value = abs(x)
        if op.endswith("ss"):
            value = max(-(1 << (width - 1)), min((1 << (width - 1)) - 1, value))
        elif op.endswith("us"):
            value = max(0, min(mask, value))
        result |= (value & mask) << (lane * width)
    return result


def _packed_decode(word: int, bf: bool) -> float:
    return _bits_float((word & 65535) << 16) if bf else struct.unpack("<e", (word & 65535).to_bytes(2, "little"))[0]


def _packed_encode(value: float, bf: bool) -> int:
    if not bf:
        try:
            return int.from_bytes(struct.pack("<e", value), "little")
        except OverflowError:
            return 0xFC00 if value < 0 else 0x7C00
    # Round the binary64 significand directly, avoiding a binary32 double rounding.
    word = int.from_bytes(struct.pack("<d", value), "little")
    sign = (word >> 48) & 0x8000
    if not math.isfinite(value):
        return sign | (0x7FC0 if math.isnan(value) else 0x7F80)
    if value == 0:
        return sign
    exponent = math.frexp(abs(value))[1] - 1
    quantum = math.ldexp(1.0, max(exponent - 7, -133))
    rounded = round(abs(value) / quantum)
    if exponent < -126:
        return sign | rounded
    if rounded == 256:
        rounded, exponent = 128, exponent + 1
    return sign | (0x7F80 if exponent > 127 else ((exponent + 127) << 7) | (rounded - 128))


def _packed_reference(op: str, bf: bool, *values: Number, bits: int, kind: str) -> Number:
    if op == "pack":
        return _packed_encode(float(values[0]), bf) | _packed_encode(float(values[1]), bf) << 16
    if op in ("low", "high"):
        return _packed_decode(int(values[0]) >> (16 if op == "high" else 0), bf)
    if op == "neg":
        return int(values[0]) ^ 0x80008000
    if op == "abs":
        return int(values[0]) & 0x7FFF7FFF
    result = 0
    for lane in range(2):
        args = [_packed_decode(int(v) >> (16 * lane), bf) for v in values]
        a = args[0]
        match op:
            case "add":
                value = a + args[1]
            case "sub":
                value = a - args[1]
            case "mul":
                value = a * args[1]
            case "fma":
                value = math.fma(a, args[1], args[2])
            case "min":
                value = args[1] if math.isnan(a) else a if math.isnan(args[1]) else min(a, args[1])
            case "max":
                value = args[1] if math.isnan(a) else a if math.isnan(args[1]) else max(a, args[1])
            case "neg":
                value = -a
            case _:
                value = abs(a)
        if op in ("min", "max") and a == args[1] == 0:
            negative = math.copysign(1, a) < 0
            other_negative = math.copysign(1, args[1]) < 0
            value = -0.0 if (negative or other_negative if op == "min" else negative and other_negative) else 0.0
        result |= _packed_encode(value, bf) << (lane * 16)
    return result


ENTRIES: tuple[Entry, ...] = (
    *_axes("thread.index", "thread inside its block", "threadIdx.{a}", "threadIdx.{a}", "_thread.{a}",
           {"x": "((unsigned)tx)", "y": "0u", "z": "0u"}, "threadIdx"),
    *_axes("block.index", "block inside the grid", "blockIdx.{a}", "blockIdx.{a}", "_block.{a}",
           {a: f"((unsigned)b{a})" for a in "xyz"}, "blockIdx"),
    *_axes("block.dim", "threads per block", "blockDim.{a}", "blockDim.{a}", "_blockdim.{a}",
           {"x": "((unsigned)bdim)", "y": "1u", "z": "1u"}, "blockDim"),
    *_axes("grid.dim", "blocks per axis", "gridDim.{a}", "gridDim.{a}", "_griddim.{a}",
           {a: f"((unsigned)g{a})" for a in "xyz"}, "gridDim"),
    Entry("block.thread_rank", "linear thread rank inside the block", _THREAD_RANK, _THREAD_RANK,
          _MSL_THREAD_RANK, "((unsigned)tx)"),
    Entry("block.num_threads", "threads in the block", _BLOCK_THREADS, _BLOCK_THREADS,
          _MSL_BLOCK_THREADS, "((unsigned)bdim)"),
    Entry("grid.block_rank", "linear block rank inside the grid", _BLOCK_RANK, _BLOCK_RANK,
          _MSL_BLOCK_RANK, _C_BLOCK_RANK),
    Entry("grid.num_blocks", "blocks in the grid", _GRID_BLOCKS, _GRID_BLOCKS,
          _MSL_GRID_BLOCKS, "((unsigned)(gx * gy * gz))"),
    Entry("grid.thread_rank", "global linear thread rank",
          f"((unsigned long long){_BLOCK_RANK} * {_BLOCK_THREADS} + {_THREAD_RANK})",
          f"((unsigned long long){_BLOCK_RANK} * {_BLOCK_THREADS} + {_THREAD_RANK})",
          f"((ulong){_MSL_BLOCK_RANK} * {_MSL_BLOCK_THREADS} + {_MSL_THREAD_RANK})",
          f"((unsigned long long){_C_BLOCK_RANK} * bdim + tx)", numba=("grid",)),
    Entry("grid.num_threads", "threads in the grid",
          f"((unsigned long long)gridDim.x * gridDim.y * gridDim.z * {_BLOCK_THREADS})",
          f"((unsigned long long)gridDim.x * gridDim.y * gridDim.z * {_BLOCK_THREADS})",
          f"((ulong)_griddim.x * _griddim.y * _griddim.z * {_MSL_BLOCK_THREADS})",
          "((unsigned long long)gx * gy * gz * bdim)", numba=("gridsize",)),
    Entry("warp.size", "lanes per warp", "{warp}", "{warp}", "{warp}", "{warp}", numba=("warpsize",)),
    Entry("warp.lane", "lane inside the warp", f"({_THREAD_RANK} % {{warp}})", f"({_THREAD_RANK} % {{warp}})", "_lane",
          "((unsigned)tx % {warp})", numba=("laneid",)),
    Entry("warp.index", "warp inside the block", f"({_THREAD_RANK} / {{warp}})", f"({_THREAD_RANK} / {{warp}})", "_simdgroup",
          "((unsigned)tx / {warp})"),
    Entry("block.sync", "barrier for the block", "__syncthreads()", "__syncthreads()",
          "threadgroup_barrier(mem_flags::mem_threadgroup)", "(void)0", numba=("syncthreads",)),
    Entry("block.count", "barrier that counts the threads with a true predicate", "__syncthreads_count({0})",
          "__syncthreads_count({0})", "", "", note="Metal needs a simd_sum plus threadgroup memory; the C backend is sequential",
          numba=("syncthreads_count",)),
    Entry("block.all", "barrier that ands a predicate over the block", "__syncthreads_and({0})", "__syncthreads_and({0})",
          "", "", note="as block.count", numba=("syncthreads_and",)),
    Entry("block.any", "barrier that ors a predicate over the block", "__syncthreads_or({0})", "__syncthreads_or({0})",
          "", "", note="as block.count", numba=("syncthreads_or",)),
    Entry("grid.sync", "barrier for the whole grid", "_grid_sync()", "_grid_sync()", "", "",
          note="needs a cooperative launch with every block resident; Metal and C have no grid barrier",
          numba=("cg.this_grid", "cg.GridGroup.sync"), helper=_GRID_SYNC),
    *(Entry(f"cluster.{field}.{axis}", f"cluster {field} along {axis}", f"_cluster_{field}_{axis}()", "", "", "",
            Feature.CLUSTERS, _CLUSTER_NOTE, sm=90,
            helper={"nvidia": f'__device__ unsigned _cluster_{field}_{axis}() {{ unsigned v; asm volatile("mov.u32 %0, %%{register}.{axis};" : "=r"(v)); return v; }}'})
      for field, register in (("dim_blocks", "cluster_nctaid"), ("block_index", "cluster_ctaid")) for axis in "xyz"),
    *(Entry(f"cluster.{name}", f"{name} at the cluster barrier", f"_cluster_{name}()", "", "", "",
            Feature.CLUSTERS, _CLUSTER_NOTE, sm=90,
            helper={"nvidia": f'__device__ void _cluster_{name}() {{ asm volatile("barrier.cluster.{name}.aligned;" ::: "memory"); }}'})
      for name in ("arrive", "wait")),
    Entry("cluster.block_rank", "block rank inside the cluster", "_cluster_block_rank()", "", "", "",
          Feature.CLUSTERS, note=_CLUSTER_NOTE,
          helper='__device__ unsigned _cluster_block_rank() { unsigned v; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(v)); return v; }'),
    Entry("cluster.num_blocks", "blocks inside the cluster", "_cluster_num_blocks()", "", "", "",
          Feature.CLUSTERS, note=_CLUSTER_NOTE,
          helper='__device__ unsigned _cluster_num_blocks() { unsigned v; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(v)); return v; }'),
    Entry("cluster.sync", "barrier for the cluster", "_cluster_sync()", "", "", "",
          Feature.CLUSTERS, note=_CLUSTER_NOTE,
          helper='__device__ void _cluster_sync() { asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }'),
    Entry("thread.fence_block", "order memory for the block", "__threadfence_block()", "__threadfence_block()",
          "atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup)",
          "__atomic_thread_fence(__ATOMIC_SEQ_CST)", numba=("threadfence_block",)),
    Entry("thread.fence_device", "order memory for the device", "__threadfence()", "__threadfence()",
          "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)",
          "__atomic_thread_fence(__ATOMIC_SEQ_CST)", numba=("threadfence",)),
    Entry("thread.fence_system", "order memory across devices and the host", "__threadfence_system()", "__threadfence_system()",
          "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)",
          "__atomic_thread_fence(__ATOMIC_SEQ_CST)", note="Metal has no system scope; the device fence is the strongest",
          numba=("threadfence_system",)),
    *(Entry(f"thread.{name}", summary, nvidia, amd, "", "",
            note="Metal exposes no cycle counter; the host reads time with its own clock",
            helper={"nvidia": '__device__ unsigned long long _globaltimer() { unsigned long long v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)); return v; }'} if name == "globaltimer" else "")
      for name, summary, nvidia, amd in (
          ("clock", "low 32 bits of the cycle counter", "(unsigned)clock()", "(unsigned)__builtin_readcyclecounter()"),
          ("clock64", "cycle counter", "(unsigned long long)clock64()", "__builtin_readcyclecounter()"),
          ("globaltimer", "device timer in nanoseconds", "_globaltimer()", "__builtin_readsteadycounter()"),
      )),
    *(Entry(f"thread.fence_acq_rel_{scope}", f"acquire-release memory fence for {scope}",
            f'asm volatile("fence.acq_rel.{ptx};" ::: "memory")',
            f'__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "{amd}")',
            f"atomic_thread_fence(mem_flags::mem_{space}, memory_order_seq_cst, thread_scope_{metal})",
            "__atomic_thread_fence(__ATOMIC_ACQ_REL)",
            note="Metal lowers acq_rel to seq_cst; system uses the device fence")
      for scope, ptx, amd, space, metal in (
          ("block", "cta", "workgroup", "threadgroup", "threadgroup"),
          ("device", "gpu", "agent", "device", "device"),
          ("system", "sys", "", "device", "device"),
      )),
    Entry("thread.fence_proxy_async", "order generic and async proxy memory accesses",
          'asm volatile("fence.proxy.async;" ::: "memory")', "", "", "", sm=90,
          note="the async proxy fence is an NVIDIA instruction"),
    *(Entry(f"warp.reduce_{op}", f"{op} reduction over the member lanes of a mask",
            "({word})" + f"__reduce_{op}_sync((unsigned)({{1}}), " + ("(unsigned)({0})" if op in ("and", "or", "xor") else "{0}") + ")",
            f"({{word}})_warp_reduce((unsigned)({{0}}), {{1}}, {code}, {{signed}})",
            f"simd_{metal}(select(({{word}})({identity}), {{0}}, (({{1}} >> _lane) & 1ul) != 0))",
            "", Feature.WARP_VOTE, _LOCKSTEP, sm=80, helper={"amd": _WARP_REDUCE_AMD},
            reference=partial(_warp_reduce_reference, op))
      for code, (op, metal, identity) in enumerate((
          ("add", "sum", "0"), ("min", "min", "{imax}"), ("max", "max", "{imin}"),
          ("and", "and", "-1"), ("or", "or", "0"), ("xor", "xor", "0"),
      ))),
    Entry("warp.sync", "barrier for the lanes of a mask", "__syncwarp((unsigned)({0}))", "__builtin_amdgcn_wave_barrier()",
          "simdgroup_barrier(mem_flags::mem_threadgroup)", "", Feature.WARP_VOTE, _LOCKSTEP, numba=("syncwarp",)),
    Entry("warp.active_mask", "lanes executing now", "(unsigned long long)__activemask()", "__builtin_amdgcn_read_exec()",
          "(ulong)simd_active_threads_mask()", "", Feature.WARP_VOTE, _LOCKSTEP, numba=("activemask",)),
    Entry("warp.ballot", "mask of lanes with a true predicate", "(unsigned long long)__ballot_sync((unsigned)({1}), {0})",
          "(__ballot({0}) & ({1}))", "((ulong)simd_ballot({0}) & ({1}))", "", Feature.WARP_VOTE, _LOCKSTEP, numba=("ballot_sync",)),
    Entry("warp.all", "predicate true in every lane", "__all_sync((unsigned)({1}), {0})",
          "((__ballot({0}) & ({1})) == (({1}) & __builtin_amdgcn_read_exec()))",
          "(((ulong)simd_ballot({0}) & ({1})) == (({1}) & (ulong)simd_active_threads_mask()))",
          "", Feature.WARP_VOTE, _LOCKSTEP, numba=("all_sync",)),
    Entry("warp.any", "predicate true in some lane", "__any_sync((unsigned)({1}), {0})",
          "((__ballot({0}) & ({1})) != 0ull)", "(((ulong)simd_ballot({0}) & ({1})) != 0ul)",
          "", Feature.WARP_VOTE, _LOCKSTEP, numba=("any_sync",)),
    Entry("warp.eq", "predicate equal in every lane",
          "(__ballot_sync((unsigned)({1}), {0}) == 0u || __ballot_sync((unsigned)({1}), {0}) == (unsigned)({1}))",
          "((__ballot({0}) & ({1})) == 0ull || (__ballot({0}) & ({1})) == (({1}) & __builtin_amdgcn_read_exec()))",
          "(((ulong)simd_ballot({0}) & ({1})) == 0ul || ((ulong)simd_ballot({0}) & ({1})) == (({1}) & (ulong)simd_active_threads_mask()))", "", Feature.WARP_VOTE, _LOCKSTEP, numba=("eq_sync",)),
    Entry("warp.shfl", "value of another lane", "__shfl_sync((unsigned)({3}), {0}, {1}, {2})", "__shfl({0}, {1}, {2})",
          "simd_shuffle({0}, ushort((_lane / ({2})) * ({2}) + (({1}) % ({2}))))", "", Feature.WARP_SHUFFLE, _LOCKSTEP,
          numba=("shfl_sync",)),
    Entry("warp.shfl_up", "value of lane - delta", "__shfl_up_sync((unsigned)({3}), {0}, {1}, {2})",
          "__shfl_up({0}, {1}, {2})", "simd_shuffle({0}, ushort((_lane % ({2})) >= ({1}) ? _lane - ({1}) : _lane))",
          "", Feature.WARP_SHUFFLE, _LOCKSTEP, numba=("shfl_up_sync",)),
    Entry("warp.shfl_down", "value of lane + delta", "__shfl_down_sync((unsigned)({3}), {0}, {1}, {2})",
          "__shfl_down({0}, {1}, {2})", "simd_shuffle({0}, ushort((_lane % ({2})) + ({1}) < ({2}) ? _lane + ({1}) : _lane))",
          "", Feature.WARP_SHUFFLE, _LOCKSTEP,
          numba=("shfl_down_sync",)),
    Entry("warp.shfl_xor", "value of lane ^ mask", "__shfl_xor_sync((unsigned)({3}), {0}, {1}, {2})",
          "__shfl_xor({0}, {1}, {2})",
          "simd_shuffle({0}, ushort(((_lane ^ ({1})) / ({2})) == (_lane / ({2})) ? (_lane ^ ({1})) : _lane))",
          "", Feature.WARP_SHUFFLE, _LOCKSTEP,
          numba=("shfl_xor_sync",)),
    Entry("warp.match_any", "lanes holding the same value", "(unsigned long long)__match_any_sync((unsigned)({1}), {0})",
          "(unsigned long long)__match_any_sync((unsigned long long)({1}), {0})", "", "", Feature.WARP_VOTE,
          "HIP text comes from the HIP reference and is not compiled; Metal needs a portable loop of shuffles",
          numba=("match_any_sync",)),
    Entry("warp.match_all", "mask when all lanes hold the same value",
          "([&]() {{ int _pred; return (unsigned long long)__match_all_sync((unsigned)({1}), {0}, &_pred); }}())",
          "([&]() {{ int _pred; return (unsigned long long)__match_all_sync((unsigned long long)({1}), {0}, &_pred); }}())",
          "", "", Feature.WARP_VOTE,
          "HIP text comes from the HIP reference and is not compiled; the predicate comes back through a local",
          numba=("match_all_sync",)),
    *(Entry(f"warp.lanemask_{suffix}", f"lanes {summary} this one",
            formula.replace("LANE", f"({_THREAD_RANK} % {{warp}})"),
            formula.replace("LANE", f"({_THREAD_RANK} % {{warp}})"), formula.replace("LANE", "_lane"),
            formula.replace("LANE", "((unsigned)tx % {warp})"),
            numba=("lanemask_lt",) if suffix == "lt" else ())
      for suffix, summary, formula in (
          ("lt", "below", "((1ull << LANE) - 1ull)"),
          ("le", "at or below", "(0xffffffffffffffffull >> (63 - LANE))"),
          ("eq", "equal to", "(1ull << LANE)"),
          ("ge", "at or above", "({full}ull ^ ((1ull << LANE) - 1ull))"),
          ("gt", "above", "({full}ull ^ (0xffffffffffffffffull >> (63 - LANE)))"),
      )),
    Entry("popcount", "bits set", "__popc{ll}({0})", "__popc{ll}({0})", "popcount({0})", "__builtin_popcount{ll}({0})",
          numba=("popc",)),
    Entry("reverse_bits", "bit pattern reversed", "__brev{ll}({0})", "__brev{ll}({0})", "reverse_bits({0})",
          "__builtin_bitreverse{bits}({0})", numba=("brev",)),
    Entry("leading_zeros", "zero bits above the top set bit", "__clz{ll}({0})", "__clz{ll}({0})", "clz({0})",
          "({0} ? __builtin_clz{ll}({0}) : {bits})", numba=("clz",)),
    Entry("first_set", "lowest set bit counting from 1", "__ffs{ll}({0})", "__ffs{ll}({0})", "({0} ? ctz({0}) + 1 : 0)",
          "__builtin_ffs{ll}({0})", numba=("ffs",)),
    Entry("fma", "fused multiply add", "fma{f}({0}, {1}, {2})", "fma{f}({0}, {1}, {2})", "fma({0}, {1}, {2})",
          "__builtin_fma{f}({0}, {1}, {2})", numba=("fma", "fp16.hfma"), reference=partial(_math_reference, "fma")),
    Entry("cbrt", "cube root", "cbrt{f}({0})", "cbrt{f}({0})", "copysign(pow(fabs({0}), 1.0f / 3.0f), {0})",
          "__builtin_cbrt{f}({0})", numba=("cbrt",)),
    Entry("log10", "base 10 logarithm", "log10{f}({0})", "log10{f}({0})", "log10({0})", "__builtin_log10{f}({0})",
          numba=("fp16.hlog10",)),
    Entry("exp10", "10 to the power x", "exp10{f}({0})", "exp10{f}({0})", "exp10({0})", "__builtin_exp10{f}({0})",
          numba=("fp16.hexp10",)),
    Entry("trunc", "round toward zero", "trunc{f}({0})", "trunc{f}({0})", "trunc({0})", "__builtin_trunc{f}({0})",
          numba=("fp16.htrunc",)),
    Entry("sleep", "suspend the thread", "__nanosleep((unsigned)({0}))", "__builtin_amdgcn_s_sleep(({0}) / 64 + 1)",
          "(void)({0})", "(void)({0})", note="AMD sleeps in units of 64 clocks; Metal and the CPU return at once",
          numba=("nanosleep",)),
    Entry("breakpoint", "stop in the debugger", 'asm volatile("brkpt;")', "__builtin_debugtrap()", "", "__builtin_debugtrap()",
          note="Metal has no breakpoint instruction; use the Xcode shader debugger or the emulator"),
    Entry("trap", "abort the kernel", "__trap()", "__builtin_trap()", "", "__builtin_trap()",
          note="Metal has no trap; an assert there prints and returns"),
    Entry("copysign", "magnitude of x with the sign of y", "copysign{f}({0}, {1})", "copysign{f}({0}, {1})",
          "copysign({0}, {1})", "__builtin_copysign{f}({0}, {1})", reference=partial(_math_reference, "copysign")),
    Entry("ldexp", "x times 2 to the power n", "ldexp{f}({0}, (int)({1}))", "ldexp{f}({0}, (int)({1}))",
          "ldexp({0}, int({1}))", "__builtin_ldexp{f}({0}, (int)({1}))", reference=partial(_math_reference, "ldexp")),
)  # fmt: skip

_ERF_HELPERS = """@inline float _family_erfcx(float x) {
    float z = __builtin_fabsf(x), t = 1.0f / (1.0f + 0.5f*z);
    float p = -1.26551223f+t*(1.00002368f+t*(0.37409196f+t*(0.09678418f+t*(-0.18628806f+
        t*(0.27886807f+t*(-1.13520398f+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f))))))));
    float r = t * __builtin_expf(p);
    return x < 0 ? 2.0f * __builtin_expf(x*x) - r : r;
}
@inline float _family_erfc(float x) {
    float z = __builtin_fabsf(x), r = _family_erfcx(z) * __builtin_expf(-z*z);
    return x < 0 ? 2.0f - r : r;
}
@inline float _family_erf(float x) {
    float q = x*x;
    return __builtin_fabsf(x) < 0.5f ? x*1.1283791670955126f*(1.0f-q/3.0f+q*q/10.0f-
        q*q*q/42.0f+q*q*q*q/216.0f-q*q*q*q*q/1320.0f) : (x < 0 ? _family_erfc(-x)-1.0f : 1.0f-_family_erfc(x));
}
@inline float _family_erfinv(float x) {
    if (__builtin_fabsf(x) == 1.0f) return __builtin_copysignf(__builtin_inff(), x);
    float w = -__builtin_logf((1.0f-x)*(1.0f+x)), p;
    if (w < 5.0f) {
        w -= 2.5f; p = 2.81022636e-08f; p = 3.43273939e-07f+p*w; p = -3.5233877e-06f+p*w;
        p = -4.39150654e-06f+p*w; p = 0.00021858087f+p*w; p = -0.00125372503f+p*w;
        p = -0.00417768164f+p*w; p = 0.246640727f+p*w; p = 1.50140941f+p*w;
    } else {
        w = __builtin_sqrtf(w)-3.0f; p = -0.000200214257f; p = 0.000100950558f+p*w;
        p = 0.00134934322f+p*w; p = -0.00367342844f+p*w; p = 0.00573950773f+p*w;
        p = -0.0076224613f+p*w; p = 0.00943887047f+p*w; p = 1.00167406f+p*w; p = 2.83297682f+p*w;
    }
    return p*x;
}
@inline float _family_erfcinv(float x) { return _family_erfinv(1.0f-x); }
"""
_METAL_ERF = _ERF_HELPERS
for _function in ("fabs", "exp", "log", "sqrt", "copysign"):
    _METAL_ERF = _METAL_ERF.replace(f"__builtin_{_function}f", f"metal::{_function}")
_METAL_ERF = _METAL_ERF.replace("__builtin_inff()", "INFINITY")
_METAL_NEXT = """@inline float _family_nextafter(float x, float y) {
    if (isnan(x) || isnan(y)) return x+y;
    if (x == y) return y;
    unsigned u = as_type<unsigned>(x);
    if (x == 0.0f) return as_type<float>((as_type<unsigned>(y) & 0x80000000u) | 1u);
    u += ((y > x) == (x > 0.0f)) ? 1u : (unsigned)-1;
    return as_type<float>(u);
}"""
_METAL_HYPOT = """@inline float _family_hypot(float x, float y) {
    float m = max(fabs(x), fabs(y)), n = min(fabs(x), fabs(y));
    if (isinf(x) || isinf(y)) return INFINITY;
    if (isnan(x) || isnan(y)) return x+y;
    if (m == 0.0f) return m;
    float r = n/m; return m*sqrt(1.0f+r*r);
}"""
_FREXP_HELPERS = """@inline float _family_frexpf(float x) { int e; return __builtin_frexpf(x, &e); }
@inline int _family_exponentf(float x) { int e; __builtin_frexpf(x, &e); return e; }
@inline double _family_frexp(double x) { int e; return __builtin_frexp(x, &e); }
@inline int _family_exponent(double x) { int e; __builtin_frexp(x, &e); return e; }"""
_METAL_FREXP = """@inline float _family_frexpf(float x) { int e; return x == 0.0f ? x : metal::frexp(x, e); }
@inline int _family_exponentf(float x) { int e; metal::frexp(x, e); return e; }"""
_LIBM_ROWS = (
    ("erf", "x: T", "error function"),
    ("erfc", "x: T", "complementary error function"),
    ("erfinv", "x: T", "inverse error function"),
    ("erfcinv", "x: T", "inverse complementary error function"),
    ("erfcx", "x: T", "scaled complementary error function"),
    ("rsqrt", "x: T", "reciprocal square root"),
    ("tan", "x: T", "tangent"),
    ("asin", "x: T", "inverse sine"),
    ("acos", "x: T", "inverse cosine"),
    ("atan", "x: T", "inverse tangent"),
    ("atan2", "y: T, x: T", "quadrant-aware inverse tangent"),
    ("hypot", "x: T, y: T", "Euclidean magnitude"),
    ("fmod", "x: T, y: T", "truncated division remainder"),
    ("nextafter", "x: T, y: T", "next representable value toward y"),
)
# No AMD column: no HIP spelling of these has been verified against the compiler, so capture refuses fast.* on AMD.
_FAST_ROWS = (
    ("exp", "__expf({0})", "__builtin_expf({0})"),
    ("exp10", "__exp10f({0})", "__builtin_exp10f({0})"),
    ("log", "__logf({0})", "__builtin_logf({0})"),
    ("log2", "__log2f({0})", "__builtin_log2f({0})"),
    ("log10", "__log10f({0})", "__builtin_log10f({0})"),
    ("sin", "__sinf({0})", "__builtin_sinf({0})"),
    ("cos", "__cosf({0})", "__builtin_cosf({0})"),
    ("tan", "__tanf({0})", "__builtin_tanf({0})"),
    ("pow", "__powf({0}, {1})", "__builtin_powf({0}, {1})"),
    ("div", "__fdividef({0}, {1})", "(({0})/({1}))"),
    ("sqrt", "_fast_sqrt_f32({0})", "__builtin_sqrtf({0})"),
    ("rsqrt", "_fast_rsqrt_f32({0})", "(1.0f / __builtin_sqrtf({0}))"),
    ("reciprocal", "_fast_reciprocal_f32({0})", "(1.0f / ({0}))"),
    ("exp2", "_fast_exp2_f32({0})", "__builtin_exp2f({0})"),
    ("tanh", "_fast_tanh_f32({0})", "__builtin_tanhf({0})"),
)
_FAST_HELPERS: dict[str, str] = {
    "sqrt": '@inline float _fast_sqrt_f32(float x) { float y; asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }',
    "rsqrt": '@inline float _fast_rsqrt_f32(float x) { float y; asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }',
    "reciprocal": '@inline float _fast_reciprocal_f32(float x) { float y; asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }',
    "exp2": '@inline float _fast_exp2_f32(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }',
    "tanh": '@inline float _fast_tanh_f32(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }',
}
_FAST_CPU_HELPERS = {
    name: f"@inline float _fast_{name}_f32(float x"
    + (", float y" if name in {"div", "pow"} else "")
    + ") { return "
    + cpu.format("x", "y")
    + "; }"
    for name, _, cpu in _FAST_ROWS
}

_BIT_HELPERS = """@inline unsigned _family_bfe_u32(unsigned x, unsigned p, unsigned n) {
    p &= 255u; n &= 255u;
    if (!n || p >= 32u) return 0u;
    n = n < 32u - p ? n : 32u - p;
    return (x >> p) & (n == 32u ? ~0u : (1u << n) - 1u);
}
@inline int _family_bfe_s32(int x, unsigned p, unsigned n) {
    p &= 255u; n &= 255u;
    if (!n) return 0;
    p = p < 31u ? p : 31u;
    n = n < 32u - p ? n : 32u - p;
    return (int)((unsigned)x << (32u - p - n)) >> (32u - n);
}
@inline unsigned long long _family_bfe_u64(unsigned long long x, unsigned p, unsigned n) {
    p &= 255u; n &= 255u;
    if (!n || p >= 64u) return 0ull;
    n = n < 64u - p ? n : 64u - p;
    return (x >> p) & (n == 64u ? ~0ull : (1ull << n) - 1ull);
}
@inline long long _family_bfe_s64(long long x, unsigned p, unsigned n) {
    p &= 255u; n &= 255u;
    if (!n) return 0;
    p = p < 63u ? p : 63u;
    n = n < 64u - p ? n : 64u - p;
    return (long long)((unsigned long long)x << (64u - p - n)) >> (64u - n);
}
@inline unsigned _family_bfi_u32(unsigned a, unsigned b, unsigned p, unsigned n) {
    p &= 255u; n &= 255u;
    if (!n || p >= 32u) return b;
    n = n < 32u - p ? n : 32u - p;
    unsigned mask = (n == 32u ? ~0u : (1u << n) - 1u) << p;
    return (b & ~mask) | ((a << p) & mask);
}
@inline unsigned long long _family_bfi_u64(unsigned long long a, unsigned long long b, unsigned p, unsigned n) {
    p &= 255u; n &= 255u;
    if (!n || p >= 64u) return b;
    n = n < 64u - p ? n : 64u - p;
    unsigned long long mask = (n == 64u ? ~0ull : (1ull << n) - 1ull) << p;
    return (b & ~mask) | ((a << p) & mask);
}
@inline unsigned _family_perm(unsigned x, unsigned y, unsigned selector) {
    unsigned long long pair = ((unsigned long long)y<<32) | x;
    unsigned result = 0;
    for (unsigned i=0; i<4; ++i) {
        unsigned s = (selector>>(4*i))&15u, b = (unsigned)(pair>>(8*(s&7u)))&255u;
        result |= ((s&8u) ? (b&128u ? 255u : 0u) : b) << (8*i);
    }
    return result;
}
@inline unsigned _family_funnel(unsigned lo, unsigned hi, unsigned shift, int left, int clamp) {
    unsigned s = clamp ? (shift > 32 ? 32 : shift) : shift&31u;
    unsigned long long pair = ((unsigned long long)hi<<32) | lo;
    return (unsigned)(left ? (pair<<s)>>32 : pair>>s);
}
@inline unsigned _family_lop3(unsigned a, unsigned b, unsigned c, unsigned table) {
    unsigned r = 0;
    if (table & 1u) r |= ~a & ~b & ~c;
    if (table & 2u) r |= ~a & ~b & c;
    if (table & 4u) r |= ~a & b & ~c;
    if (table & 8u) r |= ~a & b & c;
    if (table & 16u) r |= a & ~b & ~c;
    if (table & 32u) r |= a & ~b & c;
    if (table & 64u) r |= a & b & ~c;
    if (table & 128u) r |= a & b & c;
    return r;
}"""
_CUDA_BITS = (
    "\n".join(
        f"@inline {ctype} _family_bfe_{sign}{width}({ctype} x, unsigned p, unsigned n) {{ {ctype} r; "
        f'asm("bfe.{sign}{width} %0, %1, %2, %3;" : "={constraint}"(r) : "{constraint}"(x), "r"(p & 255u), "r"(n & 255u)); return r; }}'
        for width, unsigned, signed, constraint in (
            (32, "unsigned", "int", "r"),
            (64, "unsigned long long", "long long", "l"),
        )
        for sign, ctype in (("u", unsigned), ("s", signed))
    )
    + "\n"
    + "\n".join(
        f"@inline {ctype} _family_bfi_u{width}({ctype} x, {ctype} b, unsigned p, unsigned n) {{ {ctype} r; "
        f'asm("bfi.b{width} %0, %1, %2, %3, %4;" : "={constraint}"(r) : "{constraint}"(x), "{constraint}"(b), "r"(p & 255u), "r"(n & 255u)); return r; }}'
        for width, ctype, constraint in ((32, "unsigned", "r"), (64, "unsigned long long", "l"))
    )
    + '\ntemplate<unsigned table> @inline unsigned _family_lop3(unsigned a, unsigned b, unsigned c) { unsigned r; asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(table)); return r; }'
    + '\n@inline unsigned _family_byte_perm(unsigned x, unsigned y, unsigned s) { unsigned r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s)); return r; }'
)
_BIT_ROWS = (
    (
        "bfe",
        "x: T, pos: u32, length: u32 -> T",
        "u32 i32 u64 i64",
        "_family_bfe_{s}{bits}({0}, {1}, {2})",
        "_family_bfe_{s}{bits}({0}, {1}, {2})",
    ),
    (
        "bfi",
        "insert: T, base: T, pos: u32, length: u32 -> T",
        "u32 u64",
        "_family_bfi_u{bits}({0}, {1}, {2}, {3})",
        "_family_bfi_u{bits}({0}, {1}, {2}, {3})",
    ),
    (
        "byte_perm",
        "x: u32, y: u32, selector: u32 -> u32",
        "",
        "_family_byte_perm({0}, {1}, {2})",
        "_family_perm({0}, {1}, {2})",
    ),
    *(
        (
            name,
            "lo: u32, hi: u32, shift: u32 -> u32",
            "",
            f"__{name}({{0}}, {{1}}, {{2}})",
            f"_family_funnel({{0}}, {{1}}, {{2}}, {int(name.startswith('funnelshift_l'))}, {int(name.endswith('c'))})",
        )
        for name in ("funnelshift_l", "funnelshift_lc", "funnelshift_r", "funnelshift_rc")
    ),
    (
        "lop3",
        "a: u32, b: u32, c: u32, table: const -> u32",
        "",
        "_family_lop3<{3}>({0}, {1}, {2})",
        "_family_lop3({0}, {1}, {2}, {3})",
    ),
)
_DOT_HELPER = """@inline unsigned _family_dot(unsigned a, unsigned b, unsigned c, int sign, int mode) {
    unsigned width = mode ? 16 : 8, count = mode ? 2 : 4, start = mode == 2 ? 2 : 0;
    for (unsigned i=0; i<count; ++i) {
        int x = (int)((a>>(width*i))&((1u<<width)-1u)), y = (int)((b>>(8*(i+start)))&255u);
        if (sign) { x = (x^(1<<(width-1)))-(1<<(width-1)); y = (y^128)-128; }
        c += (unsigned)(x*y);
    }
    return c;
}"""
_SIMD_HELPER = (
    """@inline unsigned _family_simd(unsigned a, unsigned b, unsigned lanes, unsigned op) {
    unsigned width = 32/lanes, mask = (1u<<width)-1u, result = 0;
    for (unsigned i=0; i<lanes; ++i) {
        int x = (int)((a>>(i*width))&mask), y = (int)((b>>(i*width))&mask), v = 0;
        switch (op) {
"""
    + "\n".join(
        f"        case {i}: "
        + (
            "x = (x^(1<<(width-1)))-(1<<(width-1)); y = (y^(1<<(width-1)))-(1<<(width-1)); "
            if (name.endswith("s") and not name.endswith("us")) or name in ("vabs", "vneg")
            else ""
        )
        + f"v = {expression}; "
        + ("v = v ? (int)mask : 0; " if name.startswith("vcmp") else "")
        + (
            "v = v < -(1<<(width-1)) ? -(1<<(width-1)) : v > (1<<(width-1))-1 ? (1<<(width-1))-1 : v; "
            if name.endswith("ss")
            else ""
        )
        + ("v = v < 0 ? 0 : v > (int)mask ? (int)mask : v; " if name.endswith("us") else "")
        + "break;"
        for i, (name, expression) in enumerate(_SIMD_ROWS)
    )
    + "\n        }\n        result |= ((unsigned)v&mask)<<(i*width);\n    }\n    return result;\n}"
)


def _libm_entry(name: str, params: str, summary: str) -> Entry:
    operands = ", ".join(f"{{{i}}}" for i in range(params.count(":")))
    call = f"{name}{{f}}({operands})"
    single = name in ("erfinv", "erfcinv", "erfcx")
    metal_helper = {"hypot": _METAL_HYPOT, "nextafter": _METAL_NEXT}.get(
        name, _METAL_ERF if single or name in ("erf", "erfc") else ""
    )
    cpu = (
        f"_family_{name}({operands})"
        if single
        else "(1.0 / __builtin_sqrt{f}({0}))"
        if name == "rsqrt"
        else f"__builtin_{call}"
    )
    return Entry(
        f"libm.{name}",
        summary,
        call,
        call,
        f"_family_{name}({operands})" if metal_helper else f"metal::{name}({operands})",
        cpu,
        signature=sig(f"{params} -> T", T="f32" if single else "f32 f64"),
        reference=partial(_math_reference, name),
        helper={"cpu": _ERF_HELPERS if single else "", "apple": metal_helper},
        note="Metal uses portable f32 helpers where its library has no function; no Metal double",
    )


ENTRIES += (
    *(_libm_entry(*row) for row in _LIBM_ROWS),
    *(Entry(f"libm._{name}", "component of a tuple result", f"{function}{{f}}({{0}})", f"{function}{{f}}({{0}})", f"metal::{function}({{0}})", f"__builtin_{function}{{f}}({{0}})", reference=partial(_math_reference, function)) for name, function in (("sin", "sin"), ("cos", "cos"))),
    *(Entry(f"libm._{name}", "component of frexp", f"{function}{{f}}({{0}})", f"{function}{{f}}({{0}})", f"{function}f({{0}})", f"{function}{{f}}({{0}})", reference=partial(_math_reference, name), helper={"*": _FREXP_HELPERS, "nvidia": _FREXP_HELPERS.replace("__builtin_", ""), "apple": _METAL_FREXP}) for name, function in (("frexp_mantissa", "_family_frexp"), ("frexp_exponent", "_family_exponent"))),
    Entry("libm.sincos", "sine and cosine", "", "", "", "", signature=sig("x: T -> T", T="f32 f64"), reference=partial(_math_reference, "sincos"), parts=("libm._sin", "libm._cos")),
    Entry("libm.frexp", "mantissa and signed exponent", "", "", "", "", signature=sig("x: T -> T", T="f32 f64"), reference=partial(_math_reference, "frexp"), parts=("libm._frexp_mantissa", "libm._frexp_exponent")),
    *(Entry(f"fast.{name}", f"approximate {name}", cuda, "",
            "fast::divide({0}, {1})" if name == "div" else "fast::divide(1.0f, {0})" if name == "reciprocal" else f"fast::{name}({{0}}" + (", {1})" if name == "pow" else ")"), f"_fast_{name}_f32({{0}}" + (", {1})" if name in ("pow", "div") else ")"),
            signature=sig("x: T, y: T -> T" if name in ("pow", "div") else "x: T -> T", T="f32"), reference=partial(_math_reference, name),
            helper={"nvidia": _FAST_HELPERS.get(name, ""), "cpu": _FAST_CPU_HELPERS[name]}, sm=75 if name == "tanh" else 0,
            note="explicit call-site approximation; CPU and emulator use precise math") for name, cuda, cpu in _FAST_ROWS),
    *(Entry(f"bits.{name}", "PTX bit manipulation semantics", cuda, portable, portable, portable,
            signature=sig(signature, T=types), reference=partial(_bit_reference, name), helper={"nvidia": _CUDA_BITS if name in ("bfe", "bfi", "lop3", "byte_perm") else "", "*": _BIT_HELPERS}, note="PRMT sign replication requires inline PTX; CUDA __byte_perm masks selector bits" if name == "byte_perm" else "") for name, signature, types, cuda, portable in _BIT_ROWS),
    *(Entry(f"simd.{name}", "packed integer dot product with wrapping accumulation", f"__{name}({{0}}, {{1}}, {{2}})", portable, portable, portable,
            signature=sig("a: T, b: T, c: T -> T", T="i32 u32"), reference=partial(_dot_reference, name), helper={"nvidia": "", "*": _DOT_HELPER}, sm=61,
            note="AMD uses the portable helper to avoid optional dot-product target features") for mode, name in enumerate(("dp4a", "dp2a_lo", "dp2a_hi")) for portable in (f"_family_dot({{0}}, {{1}}, {{2}}, '{{s}}' == 's', {mode})",)),
    *(Entry(f"simd.{name}{lanes}", "packed lane operation", f"__{name}{lanes}({{0}}" + (")" if name in SIMD_UNARY else ", {1})"), portable, portable, portable,
            signature=sig("a: u32 -> u32" if name in SIMD_UNARY else "a: u32, b: u32 -> u32"), reference=partial(_simd_reference, name, lanes), helper={"nvidia": "", "*": _SIMD_HELPER}, note="CUDA signed averages round ties away from zero; unsigned averages round up" if name in ("vavgs", "vavgu") else "")
      for i, (name, _) in enumerate(_SIMD_ROWS) for lanes in (2, 4) for portable in (f"_family_simd({{0}}, {'0' if name in SIMD_UNARY else '{1}'}, {lanes}, {i})",)),
)  # fmt: skip

_VIDEO_NOTE = (
    "NVIDIA video SIMD: one instruction on sm_90 and later, a software sequence on older chips; no other target"
)
_VIDEO_TERNARY = tuple(
    f"{core}_{shape}{relu}"
    for core in ("viaddmax", "viaddmin", "vimax3", "vimin3")
    for shape in ("s16x2", "u16x2", "s32", "u32")
    for relu in (("", "_relu") if shape in ("s16x2", "s32") else ("",))
)  # fmt: skip
_VIDEO_BINARY = tuple(f"{core}_{shape}_relu" for core in ("vimax", "vimin") for shape in ("s16x2", "s32"))
_VIDEO_PREDICATE = tuple(
    f"{core}_{shape}" for core in ("vibmax", "vibmin") for shape in ("s16x2", "s32", "u16x2", "u32")
)
_VIDEO_SAD = ("vsads2", "vsads4", "vsadu2", "vsadu4")
_VIDEO_LANES = re.compile(r"_(?P<sign>[su])(?P<width>16x2|32)")


def _video_shape(name: str) -> tuple[bool, int, int]:
    """Signedness, lane width and lane count of a `__vi*` name: `s16x2` is two signed halfwords, `u32` one word."""
    match = _VIDEO_LANES.search(name)
    assert match is not None
    halves = match["width"] == "16x2"
    return match["sign"] == "s", 16 if halves else 32, 2 if halves else 1


def _video_helper(name: str, part: str) -> str:
    """The device function of one result of a predicated `__vi*` call: it calls the instruction with its predicates."""
    ctype = "int" if "_s32" in name else "unsigned"
    halves = name.endswith("16x2")
    pointers, declared = ("&p, &q", "bool p, q;") if halves else ("&p", "bool p;")
    if part == "value":
        body, result = f"return __{name}(a, b, {pointers});", ctype
    else:
        body, result = f"__{name}(a, b, {pointers}); return {'q' if part == 'pred_lo' else 'p'};", "bool"
    return f"@inline {result} _family_{name}_{part}({ctype} a, {ctype} b) {{ {declared} {body} }}"


def _video_reference(name: str, *values: int, bits: int, kind: str) -> int:
    """One `__vi*` value: each lane computes the function (a sum wraps to its lane), then relu clamps at zero."""
    signed, width, lanes = _video_shape(name)
    mask = (1 << width) - 1
    result = 0
    for lane in range(lanes):
        a, b, c = (_signed_lane(v >> (lane * width), width, signed) for v in (*values, 0)[:3])
        match name.split("_", 1)[0]:
            case "viaddmax":
                value = max(_signed_lane(a + b, width, signed), c)
            case "viaddmin":
                value = min(_signed_lane(a + b, width, signed), c)
            case "vimax3":
                value = max(a, b, c)
            case "vimin3":
                value = min(a, b, c)
            case "vimax" | "vibmax":
                value = max(a, b)
            case _:
                value = min(a, b)
        if name.endswith("_relu"):
            value = max(value, 0)
        result |= (value & mask) << (lane * width)
    return result


def _signed_lane(value: int, width: int, signed: bool) -> int:
    """A lane value wrapped to its width, read as signed when the lane is signed."""
    wrapped = value & ((1 << width) - 1)
    return _signed(wrapped, width) if signed else wrapped


def _video_predicate(name: str, lane: int, a: int, b: int, *, bits: int, kind: str) -> int:
    """The predicate of one lane: a >= b for max, a <= b for min, with the lanes read as the name's signedness."""
    signed, width, _ = _video_shape(name)
    x = _signed_lane(a >> (lane * width), width, signed)
    y = _signed_lane(b >> (lane * width), width, signed)
    return int(x >= y if name.startswith("vibmax") else x <= y)


def _sad_reference(name: str, a: int, b: int, *, bits: int, kind: str) -> int:
    """The sum of the absolute lane differences, unsigned: vsads2 and vsads4 read signed lanes, vsadu2 and vsadu4 unsigned."""
    lanes = int(name[-1])
    width, signed = 32 // lanes, name.startswith("vsads")
    return sum(
        abs(_signed_lane(a >> (lane * width), width, signed) - _signed_lane(b >> (lane * width), width, signed))
        for lane in range(lanes)
    )


def _video_signature(name: str, arity: int) -> Signature:
    """`a: T, b: T` or `a: T, b: T, c: T` with T the operand type of the name: i32 for `s32`, u32 otherwise."""
    operands = ", ".join(f"{letter}: T" for letter in "abc"[:arity])
    return sig(f"{operands} -> T", T="i32" if "_s32" in name else "u32")


def _video_parts(name: str) -> tuple[str, ...]:
    """The results of a predicated call: the value, then one predicate per halfword, or the single predicate."""
    return ("value", "pred_hi", "pred_lo") if name.endswith("16x2") else ("value", "pred")


def _video_part(name: str, part: str) -> Entry:
    """One result of a predicated call. The predicate of the high halfword compares lane 1, the others lane 0."""
    reference = (
        partial(_video_reference, name) if part == "value" else partial(_video_predicate, name, int(part == "pred_hi"))
    )
    return Entry(
        f"simd.{name}.{part}",
        "component of a predicated video SIMD result",
        f"_family_{name}_{part}({{0}}, {{1}})",
        "",
        "",
        "",
        helper={"nvidia": _video_helper(name, part)},
        reference=reference,
        note=_VIDEO_NOTE,
    )


_VIDEO_ROWS = (
    *(
        Entry(
            f"simd.{name}",
            "video SIMD: max or min of the sums and a third operand, per lane",
            f"__{name}({{0}}, {{1}}, {{2}})",
            "",
            "",
            "",
            signature=_video_signature(name, 3),
            reference=partial(_video_reference, name),
            note=_VIDEO_NOTE,
        )
        for name in _VIDEO_TERNARY
    ),
    *(
        Entry(
            f"simd.{name}",
            "video SIMD: max or min of two operands, then clamped at zero, per lane",
            f"__{name}({{0}}, {{1}})",
            "",
            "",
            "",
            signature=_video_signature(name, 2),
            reference=partial(_video_reference, name),
            note=_VIDEO_NOTE,
        )
        for name in _VIDEO_BINARY
    ),
    *(
        Entry(
            f"simd.{name}",
            "video SIMD: max or min of two operands, with the comparison of each lane as predicates",
            "",
            "",
            "",
            "",
            signature=_video_signature(name, 2),
            parts=tuple(f"simd.{name}.{part}" for part in _video_parts(name)),
            note=_VIDEO_NOTE,
        )
        for name in _VIDEO_PREDICATE
    ),
    *(_video_part(name, part) for name in _VIDEO_PREDICATE for part in _video_parts(name)),
    *(
        Entry(
            f"simd.{name}",
            "sum of the absolute differences of the lanes, one unsigned result",
            f"__{name}({{0}}, {{1}})",
            "",
            "",
            "",
            signature=sig("a: u32, b: u32 -> u32"),
            reference=partial(_sad_reference, name),
            note="NVIDIA SIMD instruction; no other target has it",
        )
        for name in _VIDEO_SAD
    ),
)
ENTRIES += _VIDEO_ROWS

_PACKED_HELPERS = """@inline unsigned _family_float_word(float x) { union { float f; unsigned u; } v; v.f=x; return v.u; }
@inline float _family_word_float(unsigned x) { union { float f; unsigned u; } v; v.u=x; return v.f; }
@inline float _family_unpack(unsigned word, int bf) {
    unsigned h = word&65535u;
    if (bf) return _family_word_float(h<<16);
    unsigned s = (h&32768u)<<16, e = (h>>10)&31u, m = h&1023u;
    if (e == 31) return _family_word_float(s|0x7f800000u|(m<<13));
    if (e == 0) {
        if (!m) return _family_word_float(s);
        e=113; while (!(m&1024u)) { m<<=1; --e; } m &= 1023u;
    } else e += 112;
    return _family_word_float(s|(e<<23)|(m<<13));
}
@inline unsigned _family_round(float x, float residual, int bf) {
    unsigned word = _family_float_word(x), sign = (word>>16)&32768u, a = word&0x7fffffffu;
    if (a >= 0x7f800000u) return sign | (bf ? 32640u : 31744u) | (a > 0x7f800000u ? (bf ? 64u : 512u) : 0u);
    int exponent = (int)(a>>23)-127;
    unsigned mantissa = (a&0x7fffffu) | (exponent == -127 ? 0u : 0x800000u);
    unsigned shift = bf ? 16 : (exponent < -14 ? (unsigned)(-exponent-1) : 13u);
    if (!bf && exponent > 15) return sign|31744u;
    if (shift > 31) return sign;
    unsigned q = mantissa>>shift, remainder = mantissa&((1u<<shift)-1u), tie = 1u<<(shift-1);
    int direction = residual == 0 ? 0 : ((residual > 0) == (x > 0) ? 1 : -1);
    if (remainder > tie || (remainder == tie && (direction > 0 || (!direction && (q&1u))))) ++q;
    if (bf) return sign |
        (((a&0xffffu) > 32768u || ((a&0xffffu) == 32768u && (direction > 0 || (!direction && ((a>>16)&1u))))) ? ((a>>16)+1u)&32767u : (a>>16)&32767u);
    return sign | (exponent < -14 ? q : ((unsigned)(exponent+14)<<10)+q);
}
@inline unsigned _family_pack(float lo, float hi, int bf) {
    return _family_round(lo, 0.0f, bf) | (_family_round(hi, 0.0f, bf)<<16);
}
@inline unsigned _family_packed(unsigned a, unsigned b, unsigned c, int bf, int op) {
    unsigned result = 0;
    for (unsigned lane=0; lane<2; ++lane) {
        unsigned raw = (a>>(16*lane))&65535u, packed;
        float x = _family_unpack(raw, bf), y = _family_unpack(b>>(16*lane), bf), z = _family_unpack(c>>(16*lane), bf);
        if (op == 6) packed = raw^32768u;
        else if (op == 7) packed = raw&32767u;
        else if (op == 4 || op == 5) {
            unsigned other = (b>>(16*lane))&65535u, infinity = bf ? 32640u : 31744u;
            int nan_raw = (raw&32767u) > infinity, nan_other = (other&32767u) > infinity;
            unsigned key_raw = (raw&32768u) ? 32767u-(raw&32767u) : 32768u+(raw&32767u);
            unsigned key_other = (other&32768u) ? 32767u-(other&32767u) : 32768u+(other&32767u);
            packed = nan_raw && nan_other ? (bf ? 32704u : 32256u) : nan_raw ? other : nan_other ? raw
                : ((op == 4 ? key_raw <= key_other : key_raw >= key_other) ? raw : other);
        } else {
            double exact = op == 0 ? (double)x+y : op == 1 ? (double)x-y : op == 2 ? (double)x*y : __builtin_fma((double)x,(double)y,(double)z);
            float v = (float)exact, residual = (float)(exact-(double)v);
            packed = _family_round(v, residual, bf);
        }
        result |= packed<<(16*lane);
    }
    return result;
}"""
_METAL_PACKED = (
    _PACKED_HELPERS.replace("union { float f; unsigned u; } v; v.f=x; return v.u;", "return as_type<unsigned>(x);")
    .replace("union { float f; unsigned u; } v; v.u=x; return v.f;", "return as_type<float>(x);")
    .replace(
        "    unsigned result = 0;\n    for (unsigned lane=0",
        "    #pragma metal fp math_mode(safe)\n    unsigned result = 0;\n    for (unsigned lane=0",
    )
    .replace(
        "double exact = op == 0 ? (double)x+y : op == 1 ? (double)x-y : op == 2 ? (double)x*y : __builtin_fma((double)x,(double)y,(double)z);\n            float v = (float)exact, residual = (float)(exact-(double)v);",
        "float p = x*y, v = op == 0 ? x+y : op == 1 ? x-y : op == 2 ? p : metal::fma(x,y,z), residual = 0.0f;\n"
        "            if (op == 0 || op == 1) { float t = op == 0 ? y : -y; float q = v-x; residual = (x-(v-q))+(t-q); }\n"
        "            else if (op == 2) residual = metal::fma(x,y,-v);\n"
        "            else { float q = v-p; residual = ((p-(v-q))+(z-q))+metal::fma(x,y,-p);\n"
        "                if ((_family_float_word(p)&0x7f800000u) == 0x7f800000u) residual = 0.0f; }",
    )
)
_PACKED_ROWS = (
    ("add", "a: u32, b: u32 -> u32"),
    ("sub", "a: u32, b: u32 -> u32"),
    ("mul", "a: u32, b: u32 -> u32"),
    ("fma", "a: u32, b: u32, c: u32 -> u32"),
    ("min", "a: u32, b: u32 -> u32"),
    ("max", "a: u32, b: u32 -> u32"),
    ("neg", "a: u32 -> u32"),
    ("abs", "a: u32 -> u32"),
    ("pack", "lo: f32, hi: f32 -> u32"),
    ("low", "x: u32 -> f32"),
    ("high", "x: u32 -> f32"),
)


def _packed_cuda(bf: bool) -> str:
    """PTX helpers carry words in registers, with no CUDA half headers."""
    format_ = "bf16x2" if bf else "f16x2"
    prefix = "_family_bf" if bf else "_family_half"
    expressions = (
        (
            "add",
            "a,b,c",
            "{ .reg .b32 k; mov.b32 k, 0x3f803f80; fma.rn.bf16x2 %0, %1, k, %2; }"
            if bf
            else "add.rn.f16x2 %0, %1, %2;",
        ),
        (
            "sub",
            "b,a,c" if bf else "a,b,c",
            "{ .reg .b32 k; mov.b32 k, 0xbf80bf80; fma.rn.bf16x2 %0, %1, k, %2; }"
            if bf
            else "sub.rn.f16x2 %0, %1, %2;",
        ),
        (
            "mul",
            "a,b,c",
            "{ .reg .b32 k; mov.b32 k, 0x80008000; fma.rn.bf16x2 %0, %1, %2, k; }"
            if bf
            else "mul.rn.f16x2 %0, %1, %2;",
        ),
        ("fma", "a,b,c", f"fma.rn.{format_} %0, %1, %2, %3;"),
        ("min", "a,b,c", f"min.{format_} %0, %1, %2;"),
        ("max", "a,b,c", f"max.{format_} %0, %1, %2;"),
        ("neg", "a,b,c", f"neg.{format_} %0, %1;"),
        ("abs", "a,b,c", f"abs.{format_} %0, %1;"),
    )
    helpers: list[str] = [
        f'@inline unsigned {prefix}_{op}(unsigned a, unsigned b, unsigned c) {{ unsigned r; asm("{instruction}" : "=r"(r) : '
        + ", ".join(f'"r"({arg})' for arg in args.split(",")[: 1 if op in ("neg", "abs") else 3 if op == "fma" else 2])
        + "); return r; }"
        for op, args, instruction in expressions
    ]
    helpers.append(
        f'@inline unsigned {prefix}_pack(float lo, float hi) {{ unsigned r; asm("cvt.rn.{format_}.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r; }}'
    )
    for name, shift in (("low", 0), ("high", 16)):
        body = (
            f"return __uint_as_float((x{'>>16' if shift else ''})<<16);"
            if bf
            else (
                f'unsigned h = (x>>{shift})&65535u; float r; asm("{{ .reg .b16 h; mov.b16 h, %1; cvt.f32.f16 %0, h; }}" : "=f"(r) : "h"((unsigned short)h)); return r;'
            )
        )
        helpers.append(f"@inline float {prefix}_{name}(unsigned x) {{ {body} }}")
    return "\n".join(helpers)


ENTRIES += tuple(
    Entry(f"{module}.{name}", "packed pair with round-to-nearest-even arithmetic", cuda, portable, portable, portable,
          signature=sig(signature), reference=partial(_packed_reference, name, bf), sm=80,
          helper={"nvidia": _packed_cuda(bf), "apple": _METAL_PACKED, "*": _PACKED_HELPERS})
    for module, bf in (("half2", False), ("bfloat162", True))
    for op, (name, signature) in enumerate(_PACKED_ROWS)
    for prefix in ("_family_bf" if bf else "_family_half",)
    for cuda in (f"{prefix}_{name}({{0}}" + (", {1})" if name == "pack" else ")" if name in ("low", "high") else f", {'{1}' if op < 6 else '0'}, {'{2}' if name == 'fma' else '0'})"),)
    for portable in (f"_family_pack({{0}}, {{1}}, {int(bf)})" if name == "pack" else f"_family_unpack({{0}}{'>>16' if name == 'high' else ''}, {int(bf)})" if name in ("low", "high") else f"_family_packed({{0}}, {'{1}' if op < 6 else '0'}, {'{2}' if name == 'fma' else '0'}, {int(bf)}, {op})",)
)  # fmt: skip

ENTRIES += tuple(
    Entry(f"atomic_{scope}.{op}" + ("_" if op in (AtomicOp.AND, AtomicOp.OR) else ""),
          f"{scope}-scope atomic {op}, returning the old value",
          f"atomic{op.name if op is AtomicOp.CAS else str(op).capitalize()}_{scope}(&{{0}}, " + ("{1}, {2})" if op is AtomicOp.CAS else "{1})"),
          (f'__builtin_amdgcn_atomic_{op}32(&{{0}}, {{1}}, __ATOMIC_RELAXED, "' + ("workgroup" if scope == "block" else "") + '")') if op in (AtomicOp.INC, AtomicOp.DEC) else
          "__hip_atomic_" + ("exchange" if op is AtomicOp.EXCH else "compare_exchange_strong" if op is AtomicOp.CAS else f"fetch_{op}") + "(&{0}, " + ("&{1}, {2}, __ATOMIC_RELAXED, " if op is AtomicOp.CAS else "{1}, ") + "__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_" + ("WORKGROUP" if scope == "block" else "SYSTEM") + ")",
          "_atomic_{0}", "_atomic_{0}",
          signature=sig("target: out T, " + ("expected: T, value: T" if op is AtomicOp.CAS else "limit: T" if op in (AtomicOp.INC, AtomicOp.DEC) else "value: T") + " -> T", T=" ".join(sorted(OPERANDS[op]))),
          atomic=(op, scope),
          note="Metal uses its unscoped helper: device memory is device wide and threadgroup memory is block wide; C runs sequentially")
    for scope in ("block", "system") for op in AtomicOp
)  # fmt: skip


def _cache_reference(x: Number, *, bits: int, kind: str) -> Number:
    return x


ENTRIES += (
    *(Entry(f"cache.{name}", f"{name} load hint", f"__{name}(&{{0}})",
            "__builtin_nontemporal_load(&{0})" if name in ("ldcs", "ldlu") else "({0})", "({0})", "({0})",
            signature=sig("x: ref T -> T", T="i32 u32 i64 u64 f32 f64"), reference=_cache_reference,
            note="a cache hint never changes the value; non-NVIDIA backends may use a plain load")
      for name in ("ldg", "ldcg", "ldca", "ldcs", "ldlu", "ldcv")),
    *(Entry(f"cache.{name}", f"{name} store hint", f"__{name}(&{{0}}, {{1}})",
            "__builtin_nontemporal_store({1}, &{0})" if name in ("stcs", "stwt") else "({0} = {1})",
            "({0} = {1})", "({0} = {1})",
            signature=sig("x: out T, value: T -> None", T="i32 u32 i64 u64 f32 f64"),
            note="a cache hint never changes the value; non-NVIDIA backends may use a plain store")
      for name in ("stcg", "stcs", "stwb", "stwt")),
    *(Entry(f"cache.prefetch_l{level}", f"prefetch to L{level} cache hint", f"_prefetch_l{level}(&{{0}})",
            "(void)0", "(void)0", "__builtin_prefetch(&{0})",
            signature=sig("x: ref T -> None", T="i32 u32 i64 u64 f32 f64"),
            helper={"nvidia": f'template<class T> __device__ void _prefetch_l{level}(const T* p) {{ asm volatile("prefetch.global.L{level} [%0];" :: "l"(p)); }}'},
            note="prefetch is a hint; AMD and Metal deliberately use a no-op")
      for level in (1, 2)),
)  # fmt: skip

_ASYNC_COPY_CUDA = """template<int Bytes, class T> __device__ void _async_copy(T* dst, const T* src) {
    unsigned shared = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(shared), "l"(src), "n"(Bytes) : "memory");
}"""
_ASYNC_COPY_SYNC = """#define _async_copy_sync(d, s, n) do { \\
    for (int k = 0; k < (n); ++k) (d)[k] = (s)[k]; \\
} while (0)"""
_ASYNC_WAIT_CUDA = """template<int Pending> __device__ void _async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}"""
ENTRIES += (
    Entry("async_copy.copy", "copy 1, 2, or 4 elements from global to block-shared memory",
          "_async_copy<4 * ({2})>(&{0}, &{1})", "_async_copy_sync(&{0}, &{1}, {2})",
          "_async_copy_sync(&{0}, &{1}, {2})", "_async_copy_sync(&{0}, &{1}, {2})", sm=80,
          signature=sig("dst: out shared T, src: ref global T, count: const -> None", T="i32 u32 f32"),
          helper={"nvidia": _ASYNC_COPY_CUDA, "*": _ASYNC_COPY_SYNC},
          note="other backends copy synchronously; reading the destination requires wait"),
    Entry("async_copy.commit", "commit the calling thread's async copy group",
          'asm volatile("cp.async.commit_group;" ::: "memory")', "(void)0", "(void)0", "(void)0", sm=80,
          signature=sig(" -> None"), note="a synchronous copy needs no commit"),
    Entry("async_copy.wait", "wait until at most pending groups remain",
          "_async_wait<{0}>()", "(void)0", "(void)0", "(void)0", sm=80,
          signature=sig("pending: const -> None"), helper={"nvidia": _ASYNC_WAIT_CUDA},
          note="pending is 0 through 7; a synchronous copy needs no wait"),
)  # fmt: skip

_MBARRIER_NOTE = "NVIDIA only; compile checked on supported SMs and run on sm_120a only"
_MBARRIER_HELPERS = """__device__ void _mbarrier_init(unsigned long long* bar, unsigned count) {
    unsigned p = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(p), "r"(count) : "memory");
}
__device__ unsigned long long _mbarrier_arrive(unsigned long long* bar) {
    unsigned p = (unsigned)__cvta_generic_to_shared(bar);
    unsigned long long state;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(p) : "memory");
    return state;
}"""
_MBARRIER_TX = """__device__ unsigned long long _mbarrier_expect_tx(unsigned long long* bar, unsigned bytes) {
    unsigned p = (unsigned)__cvta_generic_to_shared(bar);
    unsigned long long state;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(state) : "r"(p), "r"(bytes) : "memory");
    return state;
}"""
_MBARRIER_WAIT = """__device__ void _mbarrier_wait(const unsigned long long* bar, unsigned parity) {
    unsigned p = (unsigned)__cvta_generic_to_shared(bar);
    for (unsigned n = 0; n < (1u << 26); ++n) {
        unsigned ready;
        asm volatile("{ .reg .pred done; mbarrier.try_wait.parity.shared::cta.b64 done, [%1], %2; "
                     "selp.u32 %0, 1, 0, done; }" : "=r"(ready) : "r"(p), "r"(parity) : "memory");
        if (ready) return;
    }
    __trap();
}"""
_BULK_COPY = """template<class T> __device__ void _bulk_copy(T* dst, const T* src, unsigned bytes,
                                                          unsigned long long* bar) {
    if ((bytes & 15u) != 0) __trap();
    unsigned d = (unsigned)__cvta_generic_to_shared(dst), b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}"""
_DSMEM_HELPERS = """__device__ unsigned _dsmem_map(const void* p, unsigned rank) {
    unsigned local = (unsigned)__cvta_generic_to_shared(p), remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank));
    return remote;
}
__device__ unsigned _dsmem_load_bits(const void* p, unsigned rank) {
    unsigned remote = _dsmem_map(p, rank), word;
    asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(word) : "r"(remote) : "memory");
    return word;
}
__device__ void _dsmem_store_bits(void* p, unsigned rank, unsigned word) {
    unsigned remote = _dsmem_map(p, rank);
    asm volatile("st.shared::cluster.b32 [%0], %1;" :: "r"(remote), "r"(word) : "memory");
}
__device__ unsigned _dsmem_load_u(const unsigned* p, unsigned r) { return _dsmem_load_bits(p, r); }
__device__ int _dsmem_load_s(const int* p, unsigned r) { return (int)_dsmem_load_bits(p, r); }
__device__ float _dsmem_load_f(const float* p, unsigned r) { return __uint_as_float(_dsmem_load_bits(p, r)); }
__device__ void _dsmem_store_u(unsigned* p, unsigned r, unsigned v) { _dsmem_store_bits(p, r, v); }
__device__ void _dsmem_store_s(int* p, unsigned r, int v) { _dsmem_store_bits(p, r, (unsigned)v); }
__device__ void _dsmem_store_f(float* p, unsigned r, float v) { _dsmem_store_bits(p, r, __float_as_uint(v)); }
"""
ENTRIES += (
    Entry("mbarrier.init", "initialize a shared transaction barrier", "_mbarrier_init(&{0}, {1})", "", "", "",
          signature=sig("bar: out shared u64, count: u32 -> None"), sm=80,
          helper={"nvidia": _MBARRIER_HELPERS}, note=_MBARRIER_NOTE),
    Entry("mbarrier.arrive", "arrive at a shared transaction barrier", "_mbarrier_arrive(&{0})", "", "", "",
          signature=sig("bar: out shared u64 -> u64"), sm=80,
          helper={"nvidia": _MBARRIER_HELPERS}, note=_MBARRIER_NOTE),
    Entry("mbarrier.arrive_expect_tx", "arrive and expect bytes of transaction completion",
          "_mbarrier_expect_tx(&{0}, {1})", "", "", "",
          signature=sig("bar: out shared u64, bytes: u32 -> u64"), sm=90,
          helper={"nvidia": _MBARRIER_TX}, note=_MBARRIER_NOTE),
    Entry("mbarrier.wait_parity", "wait for a barrier phase, trapping after a bounded number of attempts",
          "_mbarrier_wait(&{0}, {1})", "", "", "",
          signature=sig("bar: ref shared u64, parity: u32 -> None"), sm=90,
          helper={"nvidia": _MBARRIER_WAIT}, note=_MBARRIER_NOTE),
    Entry("bulk.copy", "copy a multiple of 16 bytes to shared memory with transaction completion",
          "_bulk_copy(&{0}, &{1}, {2}, &{3})", "", "", "",
          signature=sig("dst: out shared T, src: ref global T, nbytes: u32, bar: out shared u64 -> None", T="i32 u32 f32"),
          sm=90, helper={"nvidia": _BULK_COPY}, note=_MBARRIER_NOTE),
    Entry("dsmem.load", "load from the shared memory of a cluster block", "_dsmem_load_{s}(&{0}, {1})", "", "", "",
          Feature.CLUSTERS, _CLUSTER_NOTE, signature=sig("x: ref shared T, rank: u32 -> T", T="i32 u32 f32"),
          sm=90, helper={"nvidia": _DSMEM_HELPERS}),
    Entry("dsmem.store", "store to the shared memory of a cluster block", "_dsmem_store_{s}(&{0}, {1}, {2})", "", "", "",
          Feature.CLUSTERS, _CLUSTER_NOTE, signature=sig("x: out shared T, rank: u32, value: T -> None", T="i32 u32 f32"),
          sm=90, helper={"nvidia": _DSMEM_HELPERS}),
)  # fmt: skip


def _round_binary32(value: Fraction, mode: str, negative: bool, *, square_root: bool = False) -> float:
    """Round a positive rational (or its square root) on the binary32 lattice using integers."""
    sign = 0x80000000 if negative else 0
    if not value:
        return _bits_float(sign)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    if square_root:
        exponent //= 2
    quantum = Fraction(2) ** (min(127, max(-126, exponent)) - 23)
    scaled = value / (quantum * quantum if square_root else quantum)
    lower = (
        math.isqrt(scaled.numerator // scaled.denominator) if square_root else scaled.numerator // scaled.denominator
    )
    if lower >= 1 << 24:
        # The hypothetical next lattice point is 2**128; its midpoint with FLT_MAX is the RN overflow threshold.
        nearest_overflows = value >= (1 << 128) - (1 << 103)
        infinite = nearest_overflows if mode == "rn" else mode == ("rd" if negative else "ru")
        return _bits_float(sign | (0x7F800000 if infinite else 0x7F7FFFFF))
    exact = scaled == lower * lower if square_root else scaled == lower
    if mode == "rn":
        difference = (
            4 * scaled.numerator - scaled.denominator * (2 * lower + 1) ** 2
            if square_root
            else 2 * (scaled.numerator - lower * scaled.denominator) - scaled.denominator
        )
        up = difference > 0 or (difference == 0 and bool(lower & 1))
    else:
        up = not exact and mode == ("rd" if negative else "ru")
    word = _float_bits(float(lower * quantum)) + int(up)
    return _bits_float(sign | word)


def _rounded_reference(op: str, mode: str, *values: Number, bits: int, kind: str) -> float:
    operands = [float(v) for v in values]
    x = operands[0]
    signs = [math.copysign(1, v) < 0 for v in operands]
    if any(math.isnan(v) for v in operands):
        return math.nan
    if op == "sqrt":
        return (
            _round_binary32(Fraction(x), mode, signs[0], square_root=True)
            if x >= 0 and math.isfinite(x)
            else math.inf
            if x > 0
            else math.nan
        )
    if op == "rcp" and (x == 0 or math.isinf(x)):
        return math.copysign(math.inf if x == 0 else 0.0, x)
    if op == "div" and operands[1] == 0:
        return math.copysign(math.inf, -1 if signs[0] != signs[1] else 1) if x else math.nan
    if not all(math.isfinite(v) for v in operands):
        match op:
            case "add":
                return x + operands[1]
            case "mul":
                return x * operands[1]
            case "div":
                return x / operands[1]
            case "fma":
                try:
                    return math.fma(*operands)
                except ValueError:
                    return math.nan
    fractions = [Fraction(v) for v in operands]
    match op:
        case "add":
            value = fractions[0] + fractions[1]
            negative_zero = signs[0] if x == operands[1] == 0 and signs[0] == signs[1] else mode == "rd"
        case "mul":
            value = fractions[0] * fractions[1]
            negative_zero = signs[0] != signs[1]
        case "div":
            value = fractions[0] / fractions[1]
            negative_zero = signs[0] != signs[1]
        case "rcp":
            value, negative_zero = 1 / fractions[0], signs[0]
        case "fma":
            product = fractions[0] * fractions[1]
            value = product + fractions[2]
            product_sign = signs[0] != signs[1]
            negative_zero = product_sign if product == fractions[2] == 0 and product_sign == signs[2] else mode == "rd"
        case _:
            raise ValueError(f"unknown rounded operation {op}")
    return _round_binary32(abs(value), mode, value < 0 if value else negative_zero)


ENTRIES += tuple(
    Entry(f"rounded.{op}_{mode}", f"binary32 {op} rounded {description}",
          f"__{cuda}_{mode}(" + ", ".join("{" + str(i) + "}" for i in range(len(params))) + ")", "", "", "",
          signature=sig(", ".join(f"{p}: f32" for p in params) + " -> f32"),
          reference=partial(_rounded_reference, op, mode),
          note="directed rounding is an instruction on NVIDIA only; a rounding-mode switch has no per-operation form elsewhere")
    for op, cuda, params in (
        ("add", "fadd", ("x", "y")), ("mul", "fmul", ("x", "y")), ("div", "fdiv", ("x", "y")),
        ("fma", "fmaf", ("x", "y", "z")), ("sqrt", "fsqrt", ("x",)), ("rcp", "frcp", ("x",)),
    )
    for mode, description in (("rn", "to nearest even"), ("rz", "toward zero"), ("ru", "up"), ("rd", "down"))
)  # fmt: skip

_FAST_FMA_F16 = """@inline float _fma_f16(float a, float b, float c) {
    float out;
    asm("{ .reg .b16 x,y,z,r; cvt.rn.f16.f32 x,%1; cvt.rn.f16.f32 y,%2; cvt.rn.f16.f32 z,%3; "
        "fma.rn.f16 r,x,y,z; cvt.f32.f16 %0,r; }" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}"""
_FAST_FMA_BF16 = """@inline float _fma_bf16(float a, float b, float c) {
    float out;
    asm("{ .reg .b16 x,y,z,r,s; .reg .b32 xx,yy,zz,rr; "
        "cvt.rn.bf16.f32 x,%1; cvt.rn.bf16.f32 y,%2; cvt.rn.bf16.f32 z,%3; "
        "mov.b32 xx,{x,x}; mov.b32 yy,{y,y}; mov.b32 zz,{z,z}; "
        "fma.rn.bf16x2 rr,xx,yy,zz; mov.b32 {r,s},rr; cvt.f32.bf16 %0,r; }"
        : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}"""
ENTRIES += (
    Entry(
        "fma._f16",
        "native scalar half FMA",
        "_fma_f16({0}, {1}, {2})",
        "__builtin_fmaf16((_Float16)({0}), (_Float16)({1}), (_Float16)({2}))",
        "float(metal::fma(half({0}), half({1}), half({2})))",
        "",
        helper={"nvidia": _FAST_FMA_F16},
        sm=53,
        note="the CPU backend has no implemented native half FMA lowering",
    ),
    Entry(
        "fma._bf16",
        "native packed bfloat16 FMA, low lane",
        "_fma_bf16({0}, {1}, {2})",
        "",
        "",
        "",
        helper={"nvidia": _FAST_FMA_BF16},
        sm=80,
        note="this backend has no implemented native scalar bfloat16 FMA lowering",
    ),
)

_GENERAL_OPS = (  # general function, signed and unsigned SIMD rows (None: no signed row), and the Metal vector call
    ("add_sat", "vaddss", "vaddus", "addsat"),
    ("sub_sat", "vsubss", "vsubus", "subsat"),
    ("maximum", "vmaxs", "vmaxu", "max"),
    ("abs_diff", "vabsdiffs", "vabsdiffu", "absdiff"),
    ("avg", None, "vavgu", "rhadd"),
)
_GENERAL_VECTORS = {"i8x4": "char4", "u8x4": "uchar4", "i16x2": "short2", "u16x2": "ushort2"}
_AVG_SIGNED_HELPER = """@inline unsigned _family_avg_signed(unsigned a, unsigned b, unsigned lanes) {
    unsigned width = 32/lanes, mask = (1u<<width)-1u, result = 0;
    for (unsigned i=0; i<lanes; ++i) {
        int x = (int)((a>>(i*width))&mask), y = (int)((b>>(i*width))&mask);
        x = (x^(1<<(width-1)))-(1<<(width-1)); y = (y^(1<<(width-1)))-(1<<(width-1));
        result |= ((unsigned)((x+y+1)>>1)&mask)<<(i*width);
    }
    return result;
}"""


def _half_up_reference(lanes: int, a: int, b: int = 0, *, bits: int, kind: str) -> int:
    """The signed average of each lane rounded half up, (x + y + 1) >> 1 in exact integers, so it cannot overflow."""
    width = 32 // lanes
    result = 0
    for lane in range(lanes):
        x = _signed_lane(a >> (lane * width), width, True)
        y = _signed_lane(b >> (lane * width), width, True)
        result |= (((x + y + 1) >> 1) & ((1 << width) - 1)) << (lane * width)
    return result


def _general_entry(name: str, packed: str, signed: str | None, unsigned: str, call: str) -> Entry:
    """A packed lane operation: NVIDIA's SIMD instruction where one exists, a composed lane loop elsewhere, and on Apple
    a Metal vector call on the word viewed as the lane vector, since Apple has a vector function for every operation."""
    lanes, vector = packed[-1], _GENERAL_VECTORS[packed]
    apple = f"_general_{name}_{vector}({{0}}, {{1}})"
    helper = (
        f"@inline uint _general_{name}_{vector}(uint a, uint b) {{ {vector} x = as_type<{vector}>(a); "
        f"{vector} y = as_type<{vector}>(b); return as_type<uint>(metal::{call}(x, y)); }}"
    )
    if signed is None and packed.startswith("i"):
        composed = f"_family_avg_signed({{0}}, {{1}}, {lanes})"
        return Entry(
            f"general.{name}.{packed}",
            f"packed {name} on {packed}: rounds half up, a composed lane loop on NVIDIA, AMD and the C backend",
            composed,
            composed,
            apple,
            composed,
            helper={"apple": helper, "*": _AVG_SIGNED_HELPER},
            reference=partial(_half_up_reference, int(lanes)),
            note="NVIDIA's signed SIMD average rounds ties away from zero, so it cannot be the half-up average",
        )
    simd_row = BY_NAME_SIMD[f"simd.{signed if packed.startswith('i') else unsigned}{lanes}"]
    return Entry(
        f"general.{name}.{packed}",
        f"packed {name} on {packed}: a native SIMD instruction on NVIDIA, a composed sequence elsewhere",
        simd_row.nvidia,
        simd_row.amd,
        apple,
        simd_row.cpu,
        helper={"nvidia": "", "apple": helper, "*": _SIMD_HELPER},
        reference=simd_row.reference,
        note="Apple composes Metal's vector integer function: there is no SIMD instruction for it",
    )


BY_NAME_SIMD = {entry.name: entry for entry in ENTRIES}
ENTRIES += tuple(
    _general_entry(name, packed, signed, unsigned, call)
    for name, signed, unsigned, call in _GENERAL_OPS
    for packed in _GENERAL_VECTORS
)

BY_NAME = {entry.name: entry for entry in ENTRIES}
FUNCTIONS: dict[str, Entry] = {entry.name: entry for entry in ENTRIES if entry.signature is not None}

OPERATORS: dict[str, str] = {
    # the numba-cuda-mlir names that are plain Python in this language, and what to write instead
    "shared.array": "shared_array(dtype, size)",
    "local.array": "local_array(dtype, size)",
    "const.array_like": "constant(values)",
    "selp": "`a if condition else b`",
    **{f"atomic.{op.value if op.value not in ('and', 'or') else op.value + '_'}": f"atomic.{op.value}" for op in AtomicOp},
    **{
        f"fp16.h{name}": f"the operator or method on an f16 value: {how}"
        for name, how in {
            "add": "a + b", "sub": "a - b", "mul": "a * b", "div": "a / b", "neg": "-a", "abs": "abs(a)",
            "sin": "a.sin()", "cos": "a.cos()", "log": "a.log()", "log2": "a.log2()", "exp": "a.exp()",
            "exp2": "a.exp2()", "floor": "a.floor()", "ceil": "a.ceil()", "sqrt": "a.sqrt()", "rsqrt": "a.rsqrt()",
            "rcp": "a.reciprocal()", "rint": "a.round()", "eq": "a == b", "ne": "a != b", "gt": "a > b",
            "ge": "a >= b", "lt": "a < b", "le": "a <= b", "max": "maximum(a, b)", "min": "minimum(a, b)",
        }.items()
    },
    "is_float16_supported": "`targets.Dtype.F16 in target.storage`",
}  # fmt: skip

NUMBA_CUDA_MLIR: tuple[str, ...] = (
    "threadIdx", "blockIdx", "blockDim", "gridDim", "laneid", "warpsize", "grid", "gridsize",
    "shared.array", "local.array", "const.array_like",
    "atomic.add", "atomic.sub", "atomic.and_", "atomic.or_", "atomic.xor", "atomic.exch", "atomic.inc", "atomic.dec",
    "atomic.max", "atomic.cas",
    "syncthreads", "syncthreads_count", "syncthreads_and", "syncthreads_or",
    "cg.this_grid", "cg.GridGroup.sync", "threadfence", "threadfence_block", "threadfence_system",
    "syncwarp", "all_sync", "any_sync", "eq_sync", "ballot_sync", "shfl_sync", "shfl_up_sync", "shfl_down_sync",
    "shfl_xor_sync", "match_any_sync", "match_all_sync", "activemask", "lanemask_lt",
    "popc", "brev", "clz", "ffs", "fma", "cbrt", "selp", "nanosleep", "is_float16_supported",
    *(f"fp16.h{name}" for name in (
        "fma", "add", "sub", "mul", "div", "neg", "abs", "sin", "cos", "log", "log10", "log2", "exp", "exp10", "exp2",
        "floor", "ceil", "sqrt", "rsqrt", "rcp", "rint", "trunc", "eq", "ne", "gt", "ge", "lt", "le", "max", "min",
    )),
)  # fmt: skip


def covered() -> set[str]:
    """The numba-cuda-mlir names this vocabulary has an answer for: an entry of the catalog or a plain-Python form."""
    axes = ("threadIdx", "blockIdx", "blockDim", "gridDim")
    named = {name.split(".")[0] if name.startswith(axes) else name for entry in ENTRIES for name in entry.numba}
    return named | set(OPERATORS)


def missing() -> list[str]:
    """The names of numba-cuda-mlir without an answer. The tests keep this empty."""
    have = covered()
    return [name for name in NUMBA_CUDA_MLIR if name not in have]


def spell(
    name: str,
    target: Target,
    *operands: str,
    bits: int = 32,
    sign: str = "u",
    warp: int | None = None,
) -> str:
    """The source text of intrinsic `name` for `target` over operand texts. Raises where the vendor has no spelling."""
    entry = BY_NAME[name]
    if not (template := entry.lowering(target)):
        raise IntrinsicTargetError(
            name,
            target.arch,
            f"`{name}` has no lowering for {target.arch}: {entry.note or 'no instruction in this backend'}",
            "use a portable expression or a target with this intrinsic",
        )
    width = warp or target.warp_size
    try:
        return template.format(
            *operands,
            s=sign,
            ll="ll" if bits == 64 else "",
            bits=bits,
            f="f" if bits == 32 else "",
            warp=width,
            full=(1 << width) - 1,
            word="int" if sign == "s" else "unsigned",
            signed=int(sign == "s"),
            imin="(-2147483647 - 1)" if sign == "s" else "0u",
            imax="2147483647" if sign == "s" else "4294967295u",
        )
    except IndexError as error:
        raise SpellError(f"`{name}` needs more operands: {error}") from None
