"""Error records are krn._internals.records envelopes of kind ERROR: every tag round trips,
another kind or version is refused."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import krn
from krn._internals.records import (
    VERSION,
    Envelope,
    Message,
    RecordFormatError,
    RecordKind,
    Scalar,
    Tag,
    envelope_dtype,
)
from krn.exceptions.records import ErrorBuffer, decode_error, encode_error
from krn.exceptions.sites import ErrorSiteTable, RaiseSite
from krn.log.record import decode_record
from krn.log.sites import SiteTable as LogSiteTable

VALUES: dict[Tag, st.SearchStrategy[Scalar]] = {
    Tag.INT: st.integers(-(2**63), 2**63 - 1),
    Tag.UINT: st.integers(0, 2**64 - 1),
    Tag.FLOAT32: st.floats(width=32, allow_nan=False, allow_infinity=False),
    Tag.FLOAT64: st.floats(allow_nan=False, allow_infinity=False),
    Tag.BOOL: st.booleans(),
}


def make_site(*tags: Tag | Message) -> tuple[ErrorSiteTable, RaiseSite]:
    table = ErrorSiteTable()
    site = table.add(kernel="k", file="k.py", line=9, class_id=12, arg_tags=tags)
    return table, site


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(list(VALUES)), max_size=4), st.data())
def test_every_tag_round_trips_through_the_envelope(tags: list[Tag], data: st.DataObject) -> None:
    table, site = make_site(*tags)
    values = tuple(data.draw(VALUES[tag]) for tag in tags)
    block = data.draw(st.integers(0, 2**32 - 1))
    thread = data.draw(st.integers(0, 2**32 - 1))
    slots = len(tags) + data.draw(st.integers(0, 2))
    record = encode_error(site, block=block, thread=thread, values=values, slots=slots)
    assert record.dtype == envelope_dtype(slots) and record.shape == (1,)
    event = decode_error(record[0], table)
    assert event.site == site and event.class_id == 12
    assert (event.block, event.thread) == (block, thread)
    assert event.values == values


@pytest.mark.parametrize(
    "tags,values",
    [
        ((Tag.UINT,), (2**64 - 1,)),
        ((Tag.INT,), (-1,)),
        ((Tag.INT,), (-(2**63),)),
        ((Tag.UINT, Tag.INT, Tag.BOOL), (2**64 - 1, -(2**63), True)),
    ],
)
def test_the_integer_edges_survive_the_record(tags: tuple[Tag, ...], values: tuple[Scalar, ...]) -> None:
    table, site = make_site(*tags)
    record = encode_error(site, block=0, thread=0, values=values, slots=len(tags))
    assert decode_error(record[0], table).values == values


def test_a_uint_message_argument_renders_the_full_unsigned_value() -> None:
    table, site = make_site(Message.of("n={}", [Tag.UINT]), Tag.INT)
    record = encode_error(site, block=0, thread=0, values=(2**64 - 1, -2), slots=2)
    event = decode_error(record[0], table)
    assert site.arguments(event.values) == ("n=18446744073709551615", -2)


def test_encoding_refuses_values_that_do_not_fit_their_tag() -> None:
    _, int_site = make_site(Tag.INT)
    _, uint_site = make_site(Tag.UINT)
    _, bool_site = make_site(Tag.BOOL)
    with pytest.raises(TypeError):
        encode_error(int_site, block=0, thread=0, values=[1.5], slots=1)
    with pytest.raises(TypeError, match="bool"):
        encode_error(bool_site, block=0, thread=0, values=[1], slots=1)
    with pytest.raises(OverflowError):
        encode_error(int_site, block=0, thread=0, values=[2**63], slots=1)
    with pytest.raises(OverflowError):
        encode_error(uint_site, block=0, thread=0, values=[2**64], slots=1)
    with pytest.raises(OverflowError):
        encode_error(int_site, block=2**32, thread=0, values=[1], slots=1)
    with pytest.raises(ValueError):
        encode_error(int_site, block=0, thread=0, values=[1, 2], slots=1)


def test_fields_beyond_the_site_are_zero_and_ignored() -> None:
    table, site = make_site(Tag.INT, Tag.FLOAT64)
    record = encode_error(site, block=3, thread=5, values=[-7, 0.25], slots=4)
    assert record["args"][0, 2:].tolist() == [0, 0]
    assert decode_error(record[0], table).values == (-7, 0.25)


def test_a_log_record_is_not_read_as_an_error() -> None:
    table, _ = make_site(Tag.INT)
    log_record = Envelope(kind=RecordKind.LOG, site=0, words=(1,)).to_row(1)
    with pytest.raises(RecordFormatError, match="an error record was expected"):
        decode_error(log_record[0], table)


def test_an_error_record_is_not_read_as_a_log_record() -> None:
    _, site = make_site(Tag.INT)
    data = encode_error(site, block=0, thread=0, values=[1], slots=1).tobytes()
    with pytest.raises(RecordFormatError, match="a log record was expected"):
        decode_record(data, LogSiteTable(), 1)


def test_an_error_record_without_a_class_is_refused() -> None:
    row = Envelope(kind=RecordKind.ERROR, site=0).to_row(0)
    with pytest.raises(RecordFormatError, match="names no class"):
        decode_error(row[0], ErrorSiteTable())


def test_another_version_of_the_envelope_is_refused() -> None:
    _, site = make_site(Tag.INT)
    record = encode_error(site, block=0, thread=0, values=[1], slots=1)
    record["version"] = VERSION + 1
    with pytest.raises(RecordFormatError, match=f"record version {VERSION + 1} is not version {VERSION}"):
        decode_error(record[0], ErrorSiteTable())


def test_buffer_capacity_comes_from_the_errors_settings() -> None:
    assert ErrorBuffer.from_settings(slots=1).capacity == 256
    with krn.configure(errors=krn.Errors(capacity=5)):
        assert ErrorBuffer.from_settings(slots=1) == ErrorBuffer(capacity=5, slots=1)


@settings(max_examples=100, deadline=None)
@given(st.integers(1, 64), st.integers(0, 200))
def test_buffer_keeps_capacity_records_and_counts_the_rest(capacity: int, count: int) -> None:
    buffer = ErrorBuffer(capacity=capacity, slots=2)
    records = buffer.zeros()
    assert records.shape == (capacity,) and records.dtype == envelope_dtype(2)
    kept, dropped = buffer.split(records, count)
    assert len(kept) == min(count, capacity)
    assert dropped == max(0, count - capacity)
    assert len(kept) + dropped == count
