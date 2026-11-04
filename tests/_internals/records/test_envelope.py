"""The envelope is one versioned layout for log and error records: it round trips, and other versions are refused."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import VERSION, Envelope, RecordFormatError, RecordKind, envelope_dtype, envelope_size


def test_layout_is_the_header_then_the_slots() -> None:
    dtype = envelope_dtype(3)
    assert dtype.names == ("version", "kind", "flags", "class", "site", "seq", "block", "thread", "args")
    assert dtype.itemsize == 1 + 1 + 1 + 2 + 2 + 4 + 4 + 4 + 3 * 8
    assert dtype["args"].shape == (3,) and dtype["args"].base == np.dtype("<u8")
    assert envelope_size(0) == 19 and envelope_dtype(0)["args"].shape == (0,)


@given(
    st.sampled_from(list(RecordKind)),
    st.integers(0, 2**16 - 1),
    st.none() | st.integers(0, 2**16 - 1),
    st.integers(0, 2**32 - 1),
    st.integers(0, 2**32 - 1),
    st.integers(0, 2**32 - 1),
    st.lists(st.integers(0, 2**64 - 1), max_size=4),
    st.integers(4, 6),
)
def test_every_field_round_trips_through_the_row(
    kind: RecordKind,
    site: int,
    class_id: int | None,
    seq: int,
    block: int,
    thread: int,
    words: list[int],
    slots: int,
) -> None:
    envelope = Envelope(
        kind=kind, site=site, class_id=class_id, seq=seq, block=block, thread=thread, words=tuple(words)
    )
    row = envelope.to_row(slots)
    assert row.dtype == envelope_dtype(slots) and row.shape == (1,)
    decoded = Envelope.from_row(row[0])
    assert decoded == Envelope(
        kind=kind,
        site=site,
        class_id=class_id,
        seq=seq,
        block=block,
        thread=thread,
        words=tuple(words) + (0,) * (slots - len(words)),
    )


def test_the_version_byte_is_first_and_checked() -> None:
    row = Envelope(kind=RecordKind.LOG, site=1).to_row(1)
    assert row["version"][0] == VERSION == 1
    row["version"] = 2
    with pytest.raises(RecordFormatError, match="record version 2 is not version 1"):
        Envelope.from_row(row[0])


def test_an_unknown_kind_is_refused() -> None:
    row = Envelope(kind=RecordKind.LOG, site=1).to_row(1)
    row["kind"] = 9
    with pytest.raises(RecordFormatError, match="record kind 9 is neither"):
        Envelope.from_row(row[0])


def test_the_class_is_present_only_when_its_flag_is() -> None:
    assert Envelope.from_row(Envelope(kind=RecordKind.ERROR, site=0, class_id=0).to_row(0)[0]).class_id == 0
    assert Envelope.from_row(Envelope(kind=RecordKind.LOG, site=0).to_row(0)[0]).class_id is None


def test_more_words_than_slots_are_refused() -> None:
    with pytest.raises(ValueError, match="2 argument words do not fit a record with 1 slots"):
        Envelope(kind=RecordKind.LOG, site=0, words=(1, 2)).to_row(1)


def test_out_of_range_coordinates_are_refused_by_the_row() -> None:
    with pytest.raises(OverflowError):
        Envelope(kind=RecordKind.LOG, site=0, block=2**32).to_row(0)
