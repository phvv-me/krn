"""Number formats as data. Vendor names (torch dtypes, ml_dtypes, PTX types, MLX modes) belong to the lowering registry.

FloatFormat decodes every code of the torch float8 dtypes and of the ml_dtypes float4, float6, float8 and bfloat16
types (checked exhaustively in the format lane, see notes/training-formats.md).
"""

import math
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import cached_property
from typing import ClassVar


class Special(StrEnum):
    IEEE = auto()  # all-ones exponent is inf (m=0) or NaN: fp32, fp16, bf16, e5m2, e4m3
    FN = auto()  # finite, only the all-ones pattern is NaN: e4m3fn
    FNUZ = auto()  # finite, NaN is 0b1000_0000, no negative zero, bias+1: e4m3fnuz, e5m2fnuz
    FNU = auto()  # unsigned finite, only all-ones is NaN: e8m0fnu, ue5m3
    NONE = auto()  # every code is a number: e2m1fn, e2m3fn, e3m2fn


class Rounding(StrEnum):
    NEAREST_EVEN = auto()
    STOCHASTIC = auto()
    TOWARD_ZERO = auto()


class ScaleRounding(StrEnum):
    FLOOR = auto()  # OCP MX v1.0: 2^(floor(log2 amax) - emax_elem)
    CEIL = auto()  # never overflows the element format
    NEAREST = auto()  # amax / max_elem cast to a scale format with mantissa bits (NVFP4's E4M3)


class Storage(StrEnum):
    BYTE = auto()  # one code per uint8 (ml_dtypes, torch float8)
    X2 = auto()  # two 4-bit codes per uint8, low nibble first (torch float4_e2m1fn_x2, triton fp4)
    BITPACKED = auto()  # n-bit codes bit-packed along the block axis (fp6 on hardware)


@dataclass(frozen=True)
class FloatFormat:
    exp_bits: int
    man_bits: int
    signed: bool = True
    special: Special = Special.IEEE
    bias: int | None = None  # None: 2^(e-1)-1, plus 1 for FNUZ
    subnormals: bool = True  # False only for E8M0, where exponent field 0 is 2^-bias, not zero

    @property
    def bits(self) -> int:
        return int(self.signed) + self.exp_bits + self.man_bits

    @property
    def exp_bias(self) -> int:
        if self.bias is not None:
            return self.bias
        return 2 ** (self.exp_bits - 1) - 1 + (self.special is Special.FNUZ)

    def decode(self, code: int) -> float:
        man = code & ((1 << self.man_bits) - 1)
        exp = (code >> self.man_bits) & ((1 << self.exp_bits) - 1)
        neg = self.signed and bool(code >> (self.man_bits + self.exp_bits))
        top = (1 << self.exp_bits) - 1
        sign = -1.0 if neg else 1.0
        match self.special:
            case Special.IEEE if exp == top:
                return math.nan if man else sign * math.inf
            case Special.FN | Special.FNU if exp == top and man == (1 << self.man_bits) - 1:
                return math.nan
            case Special.FNUZ if code == 1 << (self.bits - 1):
                return math.nan
            case _:
                pass
        if exp == 0 and self.subnormals:
            return sign * man * 2.0 ** (1 - self.exp_bias - self.man_bits)
        return sign * (1 + man / (1 << self.man_bits)) * 2.0 ** (exp - self.exp_bias)

    @cached_property
    def table(self) -> tuple[float, ...]:
        return tuple(self.decode(code) for code in range(1 << self.bits))

    @property
    def max(self) -> float:
        return max(v for v in self.table if math.isfinite(v))


@dataclass(frozen=True)
class IntFormat:
    bits: int
    signed: bool = True


E5M2 = FloatFormat(5, 2)
E4M3 = FloatFormat(4, 3)
E4M3FN = FloatFormat(4, 3, special=Special.FN)
E4M3FNUZ = FloatFormat(4, 3, special=Special.FNUZ)  # the fp8 of AMD gfx942, the Sirius MI300A
E5M2FNUZ = FloatFormat(5, 2, special=Special.FNUZ)
E8M0 = FloatFormat(8, 0, signed=False, special=Special.FNU, bias=127, subnormals=False)
UE5M3 = FloatFormat(5, 3, signed=False, special=Special.FNU)
E2M1 = FloatFormat(2, 1, special=Special.NONE)
E2M3 = FloatFormat(2, 3, special=Special.NONE)
E3M2 = FloatFormat(3, 2, special=Special.NONE)
FP32 = FloatFormat(8, 23)
BF16 = FloatFormat(8, 7)
FP16 = FloatFormat(5, 10)


@dataclass(frozen=True, slots=True)
class BlockFormat:
    """Element format, scale format and block shape. block=(1, 32) scales 32 values along the last axis.

    A format without a scale (`scale=None`) is a plain element type, so one container and one ladder cover fp16 too.
    """

    name: ClassVar[str] = "custom"
    elem: FloatFormat | IntFormat
    scale: FloatFormat | None
    block: tuple[int, int]
    storage: Storage = Storage.BYTE
    tensor_scale: FloatFormat | None = None
    scale_rounding: ScaleRounding = ScaleRounding.FLOOR

    @property
    def bits_per_value(self) -> float:
        scale_bits = self.scale.bits if self.scale else 0
        return self.elem.bits + scale_bits / (self.block[0] * self.block[1])


@dataclass(frozen=True, slots=True)
class MXFP4(BlockFormat):
    name: ClassVar[str] = "mxfp4"
    elem: FloatFormat = E2M1
    scale: FloatFormat = E8M0
    block: tuple[int, int] = (1, 32)
    storage: Storage = Storage.X2


@dataclass(frozen=True, slots=True)
class MXFP8(BlockFormat):
    name: ClassVar[str] = "mxfp8"
    elem: FloatFormat = E4M3FN
    scale: FloatFormat = E8M0
    block: tuple[int, int] = (1, 32)


@dataclass(frozen=True, slots=True)
class NVFP4(BlockFormat):
    name: ClassVar[str] = "nvfp4"
    elem: FloatFormat = E2M1
    scale: FloatFormat = E4M3FN
    block: tuple[int, int] = (1, 16)
    storage: Storage = Storage.X2
    tensor_scale: FloatFormat | None = FP32
    scale_rounding: ScaleRounding = ScaleRounding.NEAREST


@dataclass(frozen=True, slots=True)
class FP8FN(BlockFormat):
    """OCP e4m3 with one float32 scale per 1 x 128 block (the DeepSeek recipe). NVIDIA, gfx950 and Apple dialect."""

    name: ClassVar[str] = "fp8-fn"
    elem: FloatFormat = E4M3FN
    scale: FloatFormat | None = FP32
    block: tuple[int, int] = (1, 128)


@dataclass(frozen=True, slots=True)
class FP8FNUZ(BlockFormat):
    """The same recipe in the FNUZ dialect of gfx942 (Sirius), whose largest value is 240, not 448."""

    name: ClassVar[str] = "fp8-fnuz"
    elem: FloatFormat = E4M3FNUZ
    scale: FloatFormat | None = FP32
    block: tuple[int, int] = (1, 128)


@dataclass(frozen=True, slots=True)
class Plain16(BlockFormat):
    """Unscaled float16 stored as is: the last rung of every ladder."""

    name: ClassVar[str] = "fp16"
    elem: FloatFormat = FP16
    scale: FloatFormat | None = None
    block: tuple[int, int] = (1, 1)


@dataclass(frozen=True, slots=True)
class PlainBF16(BlockFormat):
    name: ClassVar[str] = "bf16"
    elem: FloatFormat = BF16
    scale: FloatFormat | None = None
    block: tuple[int, int] = (1, 1)


@dataclass(frozen=True, slots=True)
class Plain32(BlockFormat):
    """Unscaled float32: the rung every target multiplies natively, so a ladder that ends here always resolves."""

    name: ClassVar[str] = "fp32"
    elem: FloatFormat = FP32
    scale: FloatFormat | None = None
    block: tuple[int, int] = (1, 1)
