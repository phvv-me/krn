"""Device number types as short lowercase generics, the way Rust names its primitives: u8 and f32 are aliases.

    uint[L[3]]        an unsigned 3-bit integer      u3 = uint[L[3]] is the alias, u8 and u32 the same way
    sint[L[8]]        a signed 8-bit integer         (`int` is taken by Python, so signed is `sint`)
    fp[L[5], L[10]]   exponent bits, mantissa bits, and a convention that defaults to ieee: f16
    cfp[f32]          a pair of floats: c64

Arithmetic is closed inside one type (`uint[W] + uint[W]` wraps modulo 2**W) and a Python literal adopts the type, so
mixing types is a static error in both ty and pyrefly. Lossless widening is confined to explicit accumulation contexts
(`sum(..., acc=T)` and mixed tile products); other conversions are explicit. Aliases are plain
assignments because a `type` alias is not callable at runtime. `uint[3]` also works at runtime but static checkers
reject the bare int, so write `uint[L[3]]` or the alias. Classic TypeVars on purpose, for invariance.

The types register with the `numbers` ABCs (`Integral`, `Real`, `Complex`) and do not inherit from `int` or `float`:
registration answers `isinstance` at run time, while a subclass would let `u8 + u32` pass both checkers and make
`2 * n` an `int` in pyrefly (notes/round9/numbers-decision.md). Hence `Idx` (any integer width), the one static
spelling that `int` cannot give.
"""

import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import (
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    Self,
    TypeVar,
    get_args,
    get_origin,
    overload,
)

from krn.facts.targets import Dtype, Feature, Target

L = Literal  # a one-letter spelling for type-level integers: uint[L[3]]

# `FloatingPoint` and `fp` keep old-style parameters on purpose: the format parameters are phantom (no member stores
# them), and ty infers loose variance for an unused PEP 695 parameter, which would let a float16 pass for a float32.
E = TypeVar("E", bound=int)
M = TypeVar("M", bound=int)
S = TypeVar("S", default="ieee")
_Convention = TypeVar("_Convention")


def todo() -> NoReturn:
    raise NotImplementedError("API sketch: signatures only")


class Scalar:
    """Base of every device scalar, used as the bound of dtype parameters.

    On the host a scalar is a boxed number, so `f32(2.0)` can be passed to a launch; inside a kernel it is a register
    value. Conversion is explicit: `u64(x)` widens, `u3(x)` keeps the low three bits, `f16(x)` rounds to nearest even.
    """

    value: int | float
    __orig_class__: type[Self]

    def __init__(self, value: Scalar | int | float) -> None:
        self.value = value.value if isinstance(value, Scalar) else value

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    @overload
    def to[U: Scalar](self, dtype: type[U]) -> U: ...
    @overload
    def to(self, dtype: Selector) -> FloatingPoint: ...
    def to[U: Scalar](self, dtype: type[U] | Selector) -> U | FloatingPoint:
        """Convert explicitly; a selector resolves when the kernel is captured. A narrow float (fp8, fp6, fp4) rounds to
        nearest even and saturates, and a target without its instruction composes it in software."""
        return todo()


class Integer(Scalar):
    """Non-generic base of the integer types, used where any integer indexes."""

    def __index__(self) -> int:
        return int(self.value)

    def tile(self, size: int, *, origin: int = 0) -> slice:
        """The slice [origin + self * size, origin + self * size + size): the part of a dimension that `self` owns."""
        return todo()


type Idx = Integer | int


class uint[W: int](Integer):
    """Unsigned integer of W bits. Operations wrap modulo 2**W, narrower than 8 bits are masked after each operation."""

    def __add__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __radd__(self, other: int) -> uint[W]:
        return todo()

    def __sub__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __rsub__(self, other: int) -> uint[W]:
        return todo()

    def __mul__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __rmul__(self, other: int) -> uint[W]:
        return todo()

    def __floordiv__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __rfloordiv__(self, other: int) -> uint[W]:
        return todo()

    def __mod__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __rmod__(self, other: int) -> uint[W]:
        return todo()

    def __divmod__(self, other: uint[W] | int) -> tuple[uint[W], uint[W]]:
        return todo()

    def __rdivmod__(self, other: int) -> tuple[uint[W], uint[W]]:
        return todo()

    def __pow__(self, exponent: int) -> uint[W]:
        """The nonnegative exponent must be known at capture."""
        return todo()

    def __abs__(self) -> uint[W]:
        """Unsigned magnitude is the identity."""
        return todo()

    def bit_count(self) -> uint[L[32]]:
        """Count the bit pattern; the result is 32-bit, unlike Python's int."""
        return todo()

    def bit_length(self) -> uint[L[32]]:
        """Count bits through the highest set bit; the result is 32-bit, unlike Python's int."""
        return todo()

    def __and__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __or__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __xor__(self, other: uint[W] | int) -> uint[W]:
        return todo()

    def __invert__(self) -> uint[W]:
        return todo()

    def __lshift__(self, other: int) -> uint[W]:
        return todo()

    def __rshift__(self, other: int) -> uint[W]:
        return todo()

    def __lt__(self, other: uint[W] | int) -> bool:
        return todo()

    def __le__(self, other: uint[W] | int) -> bool:
        return todo()

    def __gt__(self, other: uint[W] | int) -> bool:
        return todo()

    def __ge__(self, other: uint[W] | int) -> bool:
        return todo()


class sint[W: int](Integer):
    """Two's complement integer of W bits with wrapping arithmetic."""

    def __add__(self, other: sint[W] | int) -> sint[W]:
        return todo()

    def __radd__(self, other: int) -> sint[W]:
        return todo()

    def __sub__(self, other: sint[W] | int) -> sint[W]:
        return todo()

    def __rsub__(self, other: int) -> sint[W]:
        return todo()

    def __mul__(self, other: sint[W] | int) -> sint[W]:
        return todo()

    def __rmul__(self, other: int) -> sint[W]:
        return todo()

    def __neg__(self) -> sint[W]:
        return todo()

    def __abs__(self) -> sint[W]:
        """Wrapping magnitude: the minimum signed value stays the minimum."""
        return todo()

    def __pow__(self, exponent: int) -> sint[W]:
        """The nonnegative exponent must be known at capture."""
        return todo()

    def __lt__(self, other: sint[W] | int) -> bool:
        return todo()

    def __le__(self, other: sint[W] | int) -> bool:
        return todo()

    def __gt__(self, other: sint[W] | int) -> bool:
        return todo()

    def __ge__(self, other: sint[W] | int) -> bool:
        return todo()


class ieee:
    """All-ones exponent is infinity or NaN: f16, bf16, f32, f64, e5m2."""


class fn:
    """Finite: only the all-ones pattern is NaN: e4m3fn (the OCP fp8 used by NVIDIA)."""


class fnuz:
    """Finite, one NaN code, no negative zero, bias plus one: the fp8 of AMD gfx942 (Sirius)."""


class fnu:
    """Unsigned finite, all-ones is NaN: e8m0 scales, ue5m3."""


class nonan:
    """Every code is a number: e2m1, e2m3, e3m2."""


class FloatingPoint(Scalar, Generic[E, M, _Convention]):
    """Float operators shared by concrete formats and bounded dtype parameters.

    Invariant format parameters reject mixed widths in both checkers; `Self` preserves the caller's bounded type.
    A bare `FloatingPoint` bound admits any format.
    """

    def __add__(self, other: FloatingPoint[E, M, _Convention] | float) -> Self:
        return todo()

    def __radd__(self, other: float) -> Self:
        return todo()

    def __sub__(self, other: FloatingPoint[E, M, _Convention] | float) -> Self:
        return todo()

    def __rsub__(self, other: float) -> Self:
        return todo()

    def __mul__(self, other: FloatingPoint[E, M, _Convention] | float) -> Self:
        return todo()

    def __rmul__(self, other: float) -> Self:
        return todo()

    def __truediv__(self, other: FloatingPoint[E, M, _Convention] | float) -> Self:
        return todo()

    def __rtruediv__(self, other: float) -> Self:
        return todo()

    def __pow__(self, exponent: FloatingPoint[E, M, _Convention] | float) -> Self:
        return todo()

    def __rpow__(self, base: float) -> Self:
        return todo()

    def __neg__(self) -> Self:
        return todo()

    def __abs__(self) -> Self:
        return todo()

    def __lt__(self, other: FloatingPoint[E, M, _Convention] | float) -> bool:
        return todo()

    def __le__(self, other: FloatingPoint[E, M, _Convention] | float) -> bool:
        return todo()

    def __gt__(self, other: FloatingPoint[E, M, _Convention] | float) -> bool:
        return todo()

    def __ge__(self, other: FloatingPoint[E, M, _Convention] | float) -> bool:
        return todo()


class fp(FloatingPoint[E, M, S]):
    """Floating point of E exponent bits, M mantissa bits and special-value convention S.

    The class also holds the selectors that let a kernel be a template over "which float": `fp.min` and `fp.max` are the
    narrowest and the widest the target has, and `fp.emax(8)` and `fp.mmax(8)` the 8-bit float with the most exponent or
    the most mantissa bits (e5m2 and e4m3). They resolve against the target when the kernel is first launched.
    """

    min: ClassVar[Selector]
    max: ClassVar[Selector]

    @staticmethod
    def emax(bits: int) -> Selector:
        """The `bits`-wide float with the most exponent bits that the target has: e5m2 for 8, bfloat16 for 16."""
        return FloatPick(Extreme.MOST_EXPONENT, bits)

    @staticmethod
    def mmax(bits: int) -> Selector:
        """The `bits`-wide float with the most mantissa bits that the target has: e4m3 for 8, float16 for 16."""
        return FloatPick(Extreme.MOST_MANTISSA, bits)


class cfp[F: FloatingPoint](Scalar):
    """A pair of floats of one type."""

    real: F
    imag: F

    def __init__(self, real: F | float, imag: F | float = 0.0) -> None: ...

    def __add__(self, other: cfp[F]) -> cfp[F]:
        return todo()

    def __sub__(self, other: cfp[F]) -> cfp[F]:
        return todo()

    def __mul__(self, other: cfp[F] | F) -> cfp[F]:
        """cfp times complex, or complex scaled by a real of the same float type."""
        return todo()

    def conj(self) -> cfp[F]:
        return todo()

    def __abs__(self) -> F:
        return todo()


# Device types register so isinstance(x, numbers.Integral) holds and math.sqrt(x) works through __float__.
# Inheriting int instead would make mixed widths pass the checkers.
numbers.Integral.register(uint)
numbers.Integral.register(sint)
numbers.Real.register(fp)
numbers.Complex.register(cfp)

u1 = uint[L[1]]
u2 = uint[L[2]]
u3 = uint[L[3]]
u4 = uint[L[4]]
u5 = uint[L[5]]
u6 = uint[L[6]]
u7 = uint[L[7]]
u8 = uint[L[8]]
u16 = uint[L[16]]
u32 = uint[L[32]]
u64 = uint[L[64]]

i2 = sint[L[2]]
i3 = sint[L[3]]
i4 = sint[L[4]]
i8 = sint[L[8]]
i16 = sint[L[16]]
i32 = sint[L[32]]
i64 = sint[L[64]]

f64 = fp[L[11], L[52], ieee]
f32 = fp[L[8], L[23], ieee]
f16 = fp[L[5], L[10], ieee]
bf16 = fp[L[8], L[7], ieee]
f8e5m2 = fp[L[5], L[2], ieee]
f8e4m3fn = fp[L[4], L[3], fn]
f8e4m3fnuz = fp[L[4], L[3], fnuz]
f8e5m2fnuz = fp[L[5], L[2], fnuz]
f6e3m2 = fp[L[3], L[2], nonan]
f6e2m3 = fp[L[2], L[3], nonan]
f4e2m1 = fp[L[2], L[1], nonan]
e8m0 = fp[L[8], L[0], fnu]

c32 = cfp[f16]
c64 = cfp[f32]
c128 = cfp[f64]


class Packed(Scalar):
    """Lanes of one 32-bit word, low lane first. `+`, `-` and `-x` wrap inside each lane. A comparison gives a lane
    mask of the same type: all ones in a true lane, zero in a false one. `pack(...)` builds one, `unpack(x)` reads it.
    """

    def __add__(self, other: Self) -> Self:
        return todo()

    def __sub__(self, other: Self) -> Self:
        return todo()

    def __neg__(self) -> Self:
        return todo()

    def __lt__(self, other: Self) -> Self:
        return todo()

    def __le__(self, other: Self) -> Self:
        return todo()

    def __gt__(self, other: Self) -> Self:
        return todo()

    def __ge__(self, other: Self) -> Self:
        return todo()


class i8x4(Packed):
    """Four signed 8-bit lanes."""


class u8x4(Packed):
    """Four unsigned 8-bit lanes."""


class i16x2(Packed):
    """Two signed 16-bit lanes."""


class u16x2(Packed):
    """Two unsigned 16-bit lanes."""


PACKED: tuple[type[Packed], ...] = (i8x4, u8x4, i16x2, u16x2)


class Selector(ABC):
    """A type that the target chooses when a kernel is captured, such as `fp.min` or `acc(f16)`.

    A selector may stand wherever a dtype is accepted (`zeros`, `to`, `Device.empty`, a template's configuration).
    Statically its element type is the common base `FloatingPoint`, because the checker cannot know the target.
    """

    @abstractmethod
    def resolve(self, target: Target) -> type[Scalar]: ...


class Extreme(StrEnum):
    NARROWEST = auto()
    WIDEST = auto()
    MOST_EXPONENT = auto()
    MOST_MANTISSA = auto()


@dataclass(frozen=True, slots=True)
class FloatInfo:
    dtype: type[Scalar]
    bits: int
    exponent: int
    mantissa: int


FLOATS: dict[Dtype, FloatInfo] = {
    Dtype.F64: FloatInfo(f64, 64, 11, 52),
    Dtype.F32: FloatInfo(f32, 32, 8, 23),
    Dtype.BF16: FloatInfo(bf16, 16, 8, 7),
    Dtype.F16: FloatInfo(f16, 16, 5, 10),
    Dtype.F8E4M3FN: FloatInfo(f8e4m3fn, 8, 4, 3),
    Dtype.F8E4M3FNUZ: FloatInfo(f8e4m3fnuz, 8, 4, 3),
    Dtype.F8E5M2: FloatInfo(f8e5m2, 8, 5, 2),
    Dtype.F8E5M2FNUZ: FloatInfo(f8e5m2fnuz, 8, 5, 2),
    Dtype.F4E2M1: FloatInfo(f4e2m1, 4, 2, 1),
}


def product_type(left: type[FloatingPoint], right: type[FloatingPoint]) -> type[FloatingPoint]:
    """Choose an input dtype that losslessly holds the other, falling back to f32 for incomparable formats."""
    return right if widens_to(left, right) else left if widens_to(right, left) else f32


def widens_to(source: type[Scalar | bool], target: type[Scalar | bool]) -> bool:
    """Whether every source value is exactly representable in target.

    IEEE floats need at least as many exponent and mantissa bits. Integers need a containing range; a float holds
    an integer when its significand holds all of that integer's bits. Other format conventions require conversion.
    """
    if source == target:
        return True
    if source is bool or target is bool:
        return False
    floats: dict[type[Scalar | bool], FloatInfo] = {
        info.dtype: info for info in FLOATS.values() if get_args(info.dtype)[2] is ieee
    }
    source_float, target_float = floats.get(source), floats.get(target)
    if source_float is not None and target_float is not None:
        return source_float.exponent <= target_float.exponent and source_float.mantissa <= target_float.mantissa
    source_origin, target_origin = get_origin(source), get_origin(target)
    if source_origin not in (uint, sint):
        return False
    source_bits = get_args(get_args(source)[0])[0]
    if target_float is not None:
        return source_bits <= target_float.mantissa + 1
    if target_origin not in (uint, sint):
        return False
    target_bits = get_args(get_args(target)[0])[0]
    return (
        source_bits <= target_bits
        if source_origin is target_origin
        else source_origin is uint and target_origin is sint and source_bits < target_bits
    )


@dataclass(frozen=True, slots=True)
class FloatPick(Selector):
    """The float of the target that is narrowest, widest, or has the most exponent or mantissa bits at a given width.

    Candidates are the float formats the target holds and converts in hardware (`Target.storage`), plus float32, which
    every target computes in. Float64 counts only where the target has it. Ties go to more mantissa bits.
    """

    by: Extreme
    bits: int | None = None

    def resolve(self, target: Target) -> type[Scalar]:
        candidates = [info for token, info in FLOATS.items() if token in target.storage or token is Dtype.F32]
        if Feature.FLOAT64 not in target.features:
            candidates = [info for info in candidates if info.bits != 64]
        if self.bits is not None:
            candidates = [info for info in candidates if info.bits == self.bits]
        if not candidates:
            raise LookupError(f"{target.arch} has no {self.bits}-bit float")
        match self.by:
            case Extreme.NARROWEST:
                best = min(candidates, key=lambda info: (info.bits, -info.mantissa))
            case Extreme.WIDEST:
                best = max(candidates, key=lambda info: (info.bits, info.mantissa))
            case Extreme.MOST_EXPONENT:
                best = max(candidates, key=lambda info: info.exponent)
            case Extreme.MOST_MANTISSA:
                best = max(candidates, key=lambda info: info.mantissa)
        return best.dtype


@dataclass(frozen=True, slots=True)
class Accumulator(Selector):
    """The type a sum of products of `of` is kept in: float64 for float64, int32 for integers, float32 otherwise."""

    of: type[Scalar | bool] | Selector

    def resolve(self, target: Target) -> type[Scalar]:
        inner = self.of.resolve(target) if isinstance(self.of, Selector) else self.of
        if isinstance(origin := get_origin(inner), type) and issubclass(origin, Integer):
            return i32
        return f64 if inner == f64 else f32


@overload
def acc(dtype: Selector) -> Selector: ...
@overload
def acc(dtype: type[FloatingPoint]) -> type[FloatingPoint]: ...
@overload
def acc(dtype: type[Integer]) -> type[i32]: ...
@overload
def acc(dtype: type[Scalar]) -> type[Scalar]: ...
def acc(dtype: Selector | type[Scalar]) -> Selector | type[Scalar]:
    """The accumulator type of `dtype`: `acc(f16)` is `f32`, `acc(fp.min)` follows whatever `fp.min` resolves to."""
    return Accumulator(dtype) if isinstance(dtype, Selector) else Accumulator(dtype).resolve(Target.of("sm_90a"))


fp.min = FloatPick(Extreme.NARROWEST)
fp.max = FloatPick(Extreme.WIDEST)


def name_of(dtype: type[Scalar]) -> str:
    """The short name of a type: `f16`, `bf16`, `u3`, `f8e4m3fn`, `f4e2m1`, `c64`, `u8x4`."""
    if dtype in PACKED:
        return dtype.__name__
    origin, args = get_origin(dtype), get_args(dtype)
    widths = [get_args(arg)[0] for arg in args if get_origin(arg) is Literal]
    if origin is uint:
        return f"u{widths[0]}"
    if origin is sint:
        return f"i{widths[0]}"
    if origin is cfp:
        inner = [get_args(arg)[0] for arg in get_args(args[0]) if get_origin(arg) is Literal]
        return f"c{2 * (1 + sum(inner))}"
    exponent, mantissa = widths
    known = {(11, 52): "f64", (8, 23): "f32", (5, 10): "f16", (8, 7): "bf16"}
    convention = getattr(args[2], "__name__", "ieee")
    return known.get((exponent, mantissa)) or f"f{1 + exponent + mantissa}e{exponent}m{mantissa}" + (
        "" if convention in ("ieee", "nonan") else convention
    )
