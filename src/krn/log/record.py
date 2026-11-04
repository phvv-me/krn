"""The binary record of one log call: a LOG envelope, then the call's scalar values in its slots.

The envelope is the shared record layout of krn._internals.records, so a device writes these bytes and `decode_record`
reads a device record as it reads an emulated one. A call's level is not in its record: the site it names carries it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from krn.compiler.errors import SettingError

from .._internals.records import Envelope, RecordFormatError, RecordKind, Scalar, envelope_dtype, slot_value, slot_word
from .sites import LogSite, SiteTable


@dataclass(frozen=True, kw_only=True)
class Coordinates:
    """Where a record was written: its position in the ring, and the block and thread that wrote it."""

    seq: int
    block: int
    thread: int


@dataclass(frozen=True, kw_only=True)
class Record:
    """A decoded record: its site, where it was written, and its scalar values and raw 64-bit words in field order."""

    site: LogSite
    coordinates: Coordinates
    values: tuple[Scalar, ...]
    words: tuple[int, ...]


def encode_record(site: LogSite, values: Sequence[Scalar], coordinates: Coordinates, slots: int) -> bytes:
    """The bytes of one record; `values` are the site's scalar fields in order."""
    scalars = site.message.scalars
    if len(scalars) > slots:
        expected = f"at least {len(scalars)}, the scalar arguments of the call at {site.file}:{site.line}"
        raise SettingError("log.args", str(slots), expected)
    words = tuple(slot_word(field.tag, value) for field, value in zip(scalars, values, strict=True))
    envelope = Envelope(
        kind=RecordKind.LOG,
        site=site.id,
        seq=coordinates.seq,
        block=coordinates.block,
        thread=coordinates.thread,
        words=words,
    )
    return envelope.to_row(slots).tobytes()


def decode_record(data: bytes, table: SiteTable, slots: int) -> Record:
    """The record `encode_record` wrote, its site found by the id in its envelope."""
    envelope = Envelope.from_row(np.frombuffer(data, dtype=envelope_dtype(slots), count=1)[0])
    if envelope.kind is not RecordKind.LOG:
        raise RecordFormatError("a log record was expected, but the record is an error record")
    site = table[envelope.site]
    words = envelope.words[: len(site.message.scalars)]
    values = tuple(slot_value(field.tag, word) for field, word in zip(site.message.scalars, words, strict=True))
    coordinates = Coordinates(seq=envelope.seq, block=envelope.block, thread=envelope.thread)
    return Record(site=site, coordinates=coordinates, values=values, words=words)
