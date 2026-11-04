"""Log records round trip through their bytes for every tag, and a record's size is set by its slot count."""

import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import (
    Envelope,
    Message,
    RecordFormatError,
    RecordKind,
    Scalar,
    Tag,
    envelope_size,
    slot_word,
)
from krn.compiler.errors import SettingError
from krn.log.record import Coordinates, Record, decode_record, encode_record
from krn.log.sites import SiteTable

MESSAGES = [
    Message.of("{}", [Tag.INT]),
    Message.of("{:x} {:c} {:.2f}", [Tag.UINT, Tag.UINT, Tag.FLOAT64]),
    Message.of("{} {:>6d} {:g}", ["saxpy", Tag.BOOL, Tag.FLOAT32]),
    Message.of("plain"),
    Message.of("{} {} {:g} {:d}", [Tag.INT, Tag.UINT, Tag.FLOAT64, Tag.BOOL]),
]


def value_strategy(tag: Tag) -> st.SearchStrategy[Scalar]:
    match tag:
        case Tag.INT:
            return st.integers(-(2**63), 2**63 - 1)
        case Tag.UINT:
            return st.integers(0, 2**64 - 1)
        case Tag.FLOAT32:
            return st.floats(allow_nan=False, width=32)
        case Tag.FLOAT64:
            return st.floats(allow_nan=False, width=64)
        case Tag.BOOL:
            return st.booleans()
        case Tag.STR_CONSTANT:
            raise AssertionError("constants take no slot")


@st.composite
def records(draw: st.DrawFn) -> tuple[SiteTable, tuple[Scalar, ...], Coordinates, int]:
    table = SiteTable()
    message = draw(st.sampled_from(MESSAGES))
    table.add(logger="saxpy", level=draw(st.integers(-5, 70)), file="a.py", line=3, kernel="saxpy", message=message)
    values = tuple(draw(value_strategy(field.tag)) for field in message.scalars)
    coordinates = Coordinates(
        seq=draw(st.integers(0, 2**32 - 1)),
        block=draw(st.integers(0, 2**32 - 1)),
        thread=draw(st.integers(0, 2**32 - 1)),
    )
    slots = draw(st.integers(len(message.scalars), len(message.scalars) + 4))
    return table, values, coordinates, slots


@given(records())
def test_every_tag_round_trips(case: tuple[SiteTable, tuple[Scalar, ...], Coordinates, int]) -> None:
    table, values, coordinates, slots = case
    data = encode_record(table[0], values, coordinates, slots)
    assert len(data) == envelope_size(slots)
    record = decode_record(data, table, slots)
    words = tuple(slot_word(field.tag, value) for field, value in zip(table[0].message.scalars, values, strict=True))
    assert record == Record(site=table[0], coordinates=coordinates, values=values, words=words)


def test_header_and_slots_are_packed_little_endian() -> None:
    table = SiteTable()
    site = table.add(logger="saxpy", level=30, file="a.py", line=3, kernel="saxpy", message=MESSAGES[1])
    data = encode_record(site, (7, 9, 0.5), Coordinates(seq=5, block=2, thread=11), slots=4)
    version, kind, flags, _, identifier, seq, block, thread = struct.unpack_from("<BBBHHIII", data)
    assert (version, kind, flags, identifier, seq, block, thread) == (1, RecordKind.LOG, 0, site.id, 5, 2, 11)
    assert struct.unpack_from("<QQdQ", data, 19) == (7, 9, 0.5, 0)


def test_constants_take_no_slot_and_slots_are_checked() -> None:
    table = SiteTable()
    site = table.add(logger="saxpy", level=20, file="a.py", line=4, kernel="saxpy", message=MESSAGES[2])
    coordinates = Coordinates(seq=0, block=0, thread=0)
    assert len(encode_record(site, (True, 0.5), coordinates, slots=2)) == envelope_size(2)
    with pytest.raises(SettingError, match=r"log\.args"):
        encode_record(site, (True, 0.5), coordinates, slots=1)
    with pytest.raises(ValueError):
        encode_record(site, (True,), coordinates, slots=2)
    with pytest.raises(ValueError):
        encode_record(site, (True, 0.5, 1), coordinates, slots=3)


def test_a_record_of_another_kind_or_version_is_refused() -> None:
    table = SiteTable()
    site = table.add(logger="saxpy", level=20, file="a.py", line=4, kernel="saxpy", message=MESSAGES[0])
    data = encode_record(site, (1,), Coordinates(seq=0, block=0, thread=0), slots=1)
    error_record = Envelope(kind=RecordKind.ERROR, site=site.id).to_row(1).tobytes()
    with pytest.raises(RecordFormatError, match="error record"):
        decode_record(error_record, table, 1)
    newer = bytearray(data)
    newer[0] = 2
    with pytest.raises(RecordFormatError, match="record version 2"):
        decode_record(bytes(newer), table, 1)
