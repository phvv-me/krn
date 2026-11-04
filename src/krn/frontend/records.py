"""Records passed to kernels by value, and compile-time fields."""

from dataclasses import dataclass
from typing import cast, dataclass_transform


@dataclass_transform(frozen_default=True)
def struct[C: type](cls: C) -> C:
    """A record passed to kernels by value. One class gives the host encoder and the device type, with methods.

    Fields typed `Const[X]` are compile-time values: they join the cache key and are not marshalled. The class is a
    frozen dataclass on the host, so `Tiling(rows=32)` works and values are hashable cache keys.
    """
    return cast(C, dataclass(frozen=True)(cls))  # the checker cannot see that dataclass returns the class it was given


type Const[T] = T


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Auto:
    """The type of AUTO; choices compare by identity."""

    def __repr__(self) -> str:
        return "AUTO"


AUTO = Auto()  # "let the compiler decide from the target": `pipelined(..., stages=AUTO)`
