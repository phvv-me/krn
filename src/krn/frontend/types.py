"""The number types of a kernel by their short names: `from krn.frontend.types import f32, u32`.

They are the types of `krn.frontend.dtypes`, which older code reaches as `T` (`T.f32`). `Idx` is any integer, for the
parameter that is an index: it takes the width of the argument, so a loop variable or a `1` needs no conversion."""

from .dtypes import (
    FloatingPoint,
    Idx,
    Integer,
    Scalar,
    bf16,
    e8m0,
    f4e2m1,
    f6e2m3,
    f6e3m2,
    f8e4m3fn,
    f8e5m2,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)

__all__ = [
    "FloatingPoint",
    "Idx",
    "Integer",
    "Scalar",
    "bf16",
    "e8m0",
    "f4e2m1",
    "f6e2m3",
    "f6e3m2",
    "f8e4m3fn",
    "f8e5m2",
    "f16",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
]
