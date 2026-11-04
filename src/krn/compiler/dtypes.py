"""Device number types and the arithmetic of the design: the same type stays, a literal adopts the other side's
type, and a mix is implicit only when it is lossless between standard widths."""

from dataclasses import dataclass
from enum import StrEnum, auto
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Literal, TypeGuard, get_args, get_origin

from krn.frontend import numerics
from krn.frontend.records import Auto

type Sketch = ModuleType | FunctionType | BuiltinFunctionType | numerics.Selector | type | int | float | str | Auto


class Kind(StrEnum):
    UINT = auto()
    SINT = auto()
    FLOAT = auto()
    BOOL = auto()
    PY_INT = auto()
    PY_FLOAT = auto()
    PACKED = auto()


FLOAT_NAMES = {
    (16, 10): "f16", (16, 7): "bf16", (32, 23): "f32", (64, 52): "f64",
    (8, 3): "f8e4m3fn", (8, 2): "f8e5m2", (6, 3): "f6e2m3", (6, 2): "f6e3m2", (4, 1): "f4e2m1",
}  # fmt: skip
FNUZ_NAMES = {(8, 3): "f8e4m3fnuz", (8, 2): "f8e5m2fnuz"}
STANDARD_BITS = (8, 16, 32, 64)


@dataclass(frozen=True, slots=True)
class Dtype:
    """`bits` is the total width and `man` the mantissa bits of a float. `fnuz` is the FNUZ convention of an fp8 (no
    negative zero, one NaN code, bias one higher), a float of its own beside the OCP format of the same widths.
    PY_INT and PY_FLOAT are Python numbers: literals and runtime dimensions such as `x.shape[0]`, which adopt the type
    of the other operand."""

    kind: Kind
    bits: int = 0
    man: int = 0
    lane: Dtype | None = None  # a packed word's lane type: i8x4 is four sint(8) lanes in one 32-bit word
    fnuz: bool = False

    @property
    def lanes(self) -> int:
        """The number of lanes of a packed word."""
        assert self.lane is not None, f"{self} is not packed"
        return self.bits // self.lane.bits

    @property
    def is_int(self) -> bool:
        return self.kind in (Kind.UINT, Kind.SINT)

    @property
    def is_float(self) -> bool:
        return self.kind is Kind.FLOAT

    @property
    def is_py(self) -> bool:
        return self.kind in (Kind.PY_INT, Kind.PY_FLOAT)

    @property
    def standard(self) -> bool:
        return self.bits in STANDARD_BITS

    def fits(self, value: float) -> bool:
        """Whether a literal can be this type: ints must be in range, floats accept any number."""
        if self.is_float:
            return True
        if not self.is_int or value != int(value):
            return False
        low = -(1 << (self.bits - 1)) if self.kind is Kind.SINT else 0
        return low <= value < low + (1 << self.bits)

    def __str__(self) -> str:
        match self.kind:
            case Kind.UINT | Kind.SINT:
                return f"{'u' if self.kind is Kind.UINT else 'i'}{self.bits}"
            case Kind.FLOAT:
                names = FNUZ_NAMES if self.fnuz else FLOAT_NAMES
                return names.get((self.bits, self.man), f"f{self.bits}e{self.bits - 1 - self.man}m{self.man}")
            case Kind.BOOL:
                return "bool"
            case Kind.PACKED:
                return f"{self.lane}x{self.lanes}"
            case _:
                return "int" if self.kind is Kind.PY_INT else "float"


def uint(bits: int) -> Dtype:
    return Dtype(Kind.UINT, bits)


def sint(bits: int) -> Dtype:
    return Dtype(Kind.SINT, bits)


def fp(exponent: int, mantissa: int, *, fnuz: bool = False) -> Dtype:
    return Dtype(Kind.FLOAT, 1 + exponent + mantissa, mantissa, fnuz=fnuz)


def packed(lane: Dtype) -> Dtype:
    """Lanes of `lane` in one 32-bit word, low lane first; arithmetic on it wraps inside each lane."""
    return Dtype(Kind.PACKED, 32, lane=lane)


F16, BF16, F32, F64 = fp(5, 10), fp(8, 7), fp(8, 23), fp(11, 52)
BOOL, PY_INT, PY_FLOAT = Dtype(Kind.BOOL), Dtype(Kind.PY_INT), Dtype(Kind.PY_FLOAT)
U8, U16, U32, U64, I32, I64 = uint(8), uint(16), uint(32), uint(64), sint(32), sint(64)
I8X4, U8X4, I16X2, U16X2 = packed(sint(8)), packed(U8), packed(sint(16)), packed(U16)
PACKED_DTYPES: dict[type, Dtype] = {
    numerics.i8x4: I8X4,
    numerics.u8x4: U8X4,
    numerics.i16x2: I16X2,
    numerics.u16x2: U16X2,
}


def is_sketch_type(found: Sketch | None) -> TypeGuard[type[numerics.Scalar]]:
    """Whether a `krn` value is a number type the prototype has: `T.f16`, `T.u8`, `T.uint[T.L[3]]`, `T.u8x4`."""
    return get_origin(found) in (numerics.uint, numerics.sint, numerics.fp) or found in numerics.PACKED


def from_sketch[S: numerics.Scalar | bool](sketch: type[S]) -> Dtype:
    """The dtype of a type of `krn.T`: `T.f32` is `fp[L[8], L[23], ieee]`. Of the float conventions only FNUZ is kept,
    as a dtype of its own; the other conventions are the OCP format of their widths."""
    if sketch is bool:
        return BOOL
    if sketch in numerics.PACKED:
        return PACKED_DTYPES[sketch]
    widths = tuple(get_args(arg)[0] for arg in get_args(sketch) if get_origin(arg) is Literal)
    match get_origin(sketch):
        case numerics.uint:
            return uint(*widths)
        case numerics.sint:
            return sint(*widths)
        case numerics.fp:
            return fp(*widths[:2], fnuz=numerics.fnuz in get_args(sketch))
    raise TypeError(f"{sketch} has no lowering in this slice")


SKETCH: dict[Dtype, type[numerics.Scalar | bool]] = {
    from_sketch(alias): alias for name in dir(numerics) if is_sketch_type(alias := getattr(numerics, name))
}  # the named types of krn.T by dtype
SKETCH[BOOL] = bool


def to_sketch(dtype: Dtype) -> type[numerics.Scalar | bool]:
    """The named type of `krn.T` that a dtype stands for."""
    try:
        return SKETCH[dtype]
    except KeyError:
        raise TypeError(f"{dtype} has no named type in krn.T") from None


class Arithmetic(StrEnum):
    """STRICT is the rule: the same type stays and wraps, a literal adopts the type, and any mix of two types is an
    error that names both. The owner chose it on 2026-10-31. LOSSLESS (value-preserving implicit widening) stays as a
    possible later relaxation and is what the tests use to show what STRICT rejects."""

    LOSSLESS = auto()
    STRICT = auto()

    def common(self, a: Dtype, b: Dtype) -> Dtype | None:
        """The type of `a op b`, or None when the mix needs an explicit conversion."""
        if a == b:
            return a
        if a.is_py or b.is_py:
            return self._adopt(a, b)
        return self._widen(a, b) if self is Arithmetic.LOSSLESS else None

    @staticmethod
    def _adopt(a: Dtype, b: Dtype) -> Dtype | None:
        py, other = (a, b) if a.is_py else (b, a)
        if other.is_py:
            return PY_FLOAT if PY_FLOAT in (a, b) else PY_INT
        accepts = other.is_float or (other.is_int and py.kind is Kind.PY_INT)
        return other if accepts else None

    @staticmethod
    def _widen(a: Dtype, b: Dtype) -> Dtype | None:
        if a.is_float and b.is_float and a.standard and b.standard:
            return F64 if F64 in (a, b) else F32
        if not (a.is_int and b.is_int and a.standard and b.standard):
            return None
        if a.kind is b.kind:
            return Dtype(a.kind, max(a.bits, b.bits))
        unsigned, signed = (a, b) if a.kind is Kind.UINT else (b, a)
        bits = max(signed.bits, 2 * unsigned.bits)
        return sint(bits) if bits <= 64 else None
