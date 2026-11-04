"""Scalar tags and their 64-bit slot words: the one codec every krn record uses, on the host and on the device.

A signed int is its two's complement word, a uint its value, a float its IEEE bits (a float32 in the low 32 bits of
the word, zero above them), and a bool 0 or 1. Every word round trips, including UINT 2**64 - 1 and negative INT.
"""

import struct
from enum import StrEnum, auto

WORD_MASK = 2**64 - 1
SLOT_BYTES = 8
_INT_LOW = -(2**63)
_INT_HIGH = 2**63 - 1

type Scalar = int | float | bool


class Tag(StrEnum):
    """What a record field holds: a runtime scalar in one slot, or a str constant whose text its site keeps."""

    INT = auto()
    UINT = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    BOOL = auto()
    STR_CONSTANT = "str-constant"

    @property
    def is_scalar(self) -> bool:
        return self is not Tag.STR_CONSTANT


def slot_word(tag: Tag, value: Scalar) -> int:
    """The unsigned 64-bit word that holds value as tag; a value that does not fit the tag is refused."""
    match tag:
        case Tag.INT:
            integer = _integer(tag, value)
            if not _INT_LOW <= integer <= _INT_HIGH:
                raise OverflowError(f"int {integer} does not fit a 64-bit signed slot")
            return integer & WORD_MASK
        case Tag.UINT:
            integer = _integer(tag, value)
            if not 0 <= integer <= WORD_MASK:
                raise OverflowError(f"uint {integer} does not fit a 64-bit unsigned slot")
            return integer
        case Tag.FLOAT32:
            return struct.unpack("<I", struct.pack("<f", _real(tag, value)))[0]
        case Tag.FLOAT64:
            return struct.unpack("<Q", struct.pack("<d", _real(tag, value)))[0]
        case Tag.BOOL:
            if not isinstance(value, bool):
                raise TypeError(f"a bool argument was {value!r}")
            return int(value)
        case _:
            raise ValueError("a str constant takes no slot; its text lives in its site")


def slot_value(tag: Tag, word: int) -> Scalar:
    """The value a word holds as tag, the inverse of slot_word for every value slot_word accepts."""
    match tag:
        case Tag.INT:
            return word - 2**64 if word > _INT_HIGH else word
        case Tag.UINT:
            return word
        case Tag.FLOAT32:
            return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]
        case Tag.FLOAT64:
            return struct.unpack("<d", struct.pack("<Q", word))[0]
        case Tag.BOOL:
            return bool(word)
        case _:
            raise ValueError("a str constant takes no slot; its text lives in its site")


def _integer(tag: Tag, value: Scalar) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"a {tag} argument was {value!r}")
    return value


def _real(tag: Tag, value: Scalar) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"a {tag} argument was {value!r}")
    return float(value)
