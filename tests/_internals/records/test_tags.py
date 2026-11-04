"""Slot words hold every value of their tag and nothing else: the integer edges, IEEE bit patterns, and refusals."""

import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import Tag, slot_value, slot_word


@given(st.integers(-(2**63), 2**63 - 1))
def test_int_round_trips_through_its_two_complement_word(value: int) -> None:
    word = slot_word(Tag.INT, value)
    assert 0 <= word < 2**64
    assert slot_value(Tag.INT, word) == value


@given(st.integers(0, 2**64 - 1))
def test_uint_round_trips_through_its_value(value: int) -> None:
    assert slot_word(Tag.UINT, value) == value
    assert slot_value(Tag.UINT, value) == value


@given(st.floats(width=64))
def test_float64_round_trips_its_ieee_bits(value: float) -> None:
    word = slot_word(Tag.FLOAT64, value)
    assert word == struct.unpack("<Q", struct.pack("<d", value))[0]
    assert struct.pack("<d", slot_value(Tag.FLOAT64, word)) == struct.pack("<d", value)


@given(st.floats(width=32))
def test_float32_round_trips_its_bits_in_the_low_half(value: float) -> None:
    word = slot_word(Tag.FLOAT32, value)
    assert word < 2**32
    assert struct.pack("<f", slot_value(Tag.FLOAT32, word)) == struct.pack("<f", value)


def test_bit_patterns_are_the_ieee_ones() -> None:
    assert slot_word(Tag.FLOAT32, 1.0) == 0x3F800000
    assert slot_word(Tag.FLOAT64, 1.0) == 0x3FF0000000000000
    assert slot_word(Tag.FLOAT64, -2.0) == 0xC000000000000000
    assert slot_word(Tag.INT, -1) == 2**64 - 1
    assert slot_word(Tag.BOOL, True) == 1 and slot_value(Tag.BOOL, 0) is False


def test_integer_edges_are_exact() -> None:
    assert slot_word(Tag.UINT, 2**64 - 1) == 2**64 - 1
    assert slot_value(Tag.UINT, 2**64 - 1) == 2**64 - 1
    assert slot_word(Tag.INT, -(2**63)) == 2**63
    assert slot_value(Tag.INT, 2**63) == -(2**63)
    assert slot_value(Tag.INT, 2**63 - 1) == 2**63 - 1


@pytest.mark.parametrize(
    "tag,value",
    [(Tag.INT, 2**63), (Tag.INT, -(2**63) - 1), (Tag.UINT, 2**64), (Tag.UINT, -1), (Tag.FLOAT32, 1e300)],
)
def test_values_past_the_slot_are_refused(tag: Tag, value: float) -> None:
    with pytest.raises(OverflowError):
        slot_word(tag, value)


@pytest.mark.parametrize(
    "tag,value",
    [(Tag.INT, True), (Tag.INT, 1.5), (Tag.UINT, "1"), (Tag.BOOL, 1), (Tag.FLOAT64, True), (Tag.FLOAT32, "x")],
)
def test_values_of_the_wrong_type_name_the_tag(tag: Tag, value) -> None:
    with pytest.raises(TypeError, match=f"a {tag} argument was"):
        slot_word(tag, value)


def test_str_constants_take_no_slot() -> None:
    with pytest.raises(ValueError, match="no slot"):
        slot_word(Tag.STR_CONSTANT, 1)
    with pytest.raises(ValueError, match="no slot"):
        slot_value(Tag.STR_CONSTANT, 1)
    assert not Tag.STR_CONSTANT.is_scalar and Tag.FLOAT32.is_scalar
