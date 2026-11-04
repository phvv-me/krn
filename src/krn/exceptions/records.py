"""The device record of one raised error, a krn._internals.records envelope,
and the buffer holding a launch's records."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from krn.runtime.settings import current_settings

from .._internals.records import Envelope, RecordFormatError, RecordKind, Scalar, envelope_dtype, slot_value, slot_word
from .sites import ErrorSiteTable, RaiseSite


@dataclass(frozen=True, kw_only=True)
class ErrorEvent:
    """One decoded record: the raise site, the class id the device wrote, the thread's coordinates and its values."""

    site: RaiseSite
    class_id: int
    block: int
    thread: int
    values: tuple[Scalar, ...]


def encode_error(site: RaiseSite, *, block: int, thread: int, values: Sequence[Scalar], slots: int) -> np.ndarray:
    """A one-record array of the error site raised in block and thread with these runtime values."""
    words = tuple(slot_word(kind, value) for kind, value in zip(site.kinds, values, strict=True))
    envelope = Envelope(
        kind=RecordKind.ERROR,
        site=site.id,
        class_id=site.class_id,
        block=block,
        thread=thread,
        words=words,
    )
    return envelope.to_row(slots)


def decode_error(record: np.void, sites: ErrorSiteTable) -> ErrorEvent:
    """The event a device record stands for, its runtime values read as the site's kinds say."""
    envelope = Envelope.from_row(record)
    if envelope.kind is not RecordKind.ERROR:
        raise RecordFormatError("an error record was expected, but the record is a log record")
    if envelope.class_id is None:
        raise RecordFormatError("an error record names no class")
    site = sites[envelope.site]
    words = envelope.words[: len(site.kinds)]
    values = tuple(slot_value(kind, word) for kind, word in zip(site.kinds, words, strict=True))
    return ErrorEvent(
        site=site, class_id=envelope.class_id, block=envelope.block, thread=envelope.thread, values=values
    )


@dataclass(frozen=True, kw_only=True)
class ErrorBuffer:
    """The device ring of one launch: `capacity` envelopes of `slots` argument words.

    The device counts every error. A count past capacity is kept as the number dropped, never silently lost.
    """

    capacity: int
    slots: int

    @classmethod
    def from_settings(cls, slots: int) -> Self:
        """The buffer a launch gets under the current `errors` settings."""
        return cls(capacity=current_settings().errors.capacity, slots=slots)

    def zeros(self) -> np.ndarray:
        """An empty ring of capacity records, as a backend allocates it."""
        return np.zeros(self.capacity, dtype=envelope_dtype(self.slots))

    def split(self, records: np.ndarray, count: int) -> tuple[np.ndarray, int]:
        """The records the device kept, and how many errors it counted beyond capacity."""
        kept = records[: min(count, self.capacity)]
        return kept, count - len(kept)
