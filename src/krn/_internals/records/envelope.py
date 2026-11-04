"""The record a kernel writes for each log call or raised error: a versioned header, then 64-bit slots.

The header is `version`, `kind`, `flags` (bit 0: a class id is present), `class`, `site`, `seq`, `block` and `thread`,
all little-endian, then `slots` words. A record of another version or kind is refused rather than read.
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cache
from typing import Self

import numpy as np

VERSION = 1
_HAS_CLASS = 1


class RecordKind(IntEnum):
    """Which call wrote a record: a log call, or a raise."""

    LOG = auto()
    ERROR = auto()


class RecordFormatError(ValueError):
    """The bytes are not a record this krn reads: another version, an unknown kind, or the wrong kind for the reader."""


@cache
def envelope_dtype(slots: int) -> np.dtype:
    """The packed layout of one record with `slots` 64-bit argument words."""
    return np.dtype(
        [
            ("version", "u1"),
            ("kind", "u1"),
            ("flags", "u1"),
            ("class", "<u2"),
            ("site", "<u2"),
            ("seq", "<u4"),
            ("block", "<u4"),
            ("thread", "<u4"),
            ("args", "<u8", (slots,)),
        ],
        align=False,
    )


def envelope_size(slots: int) -> int:
    """The bytes of every record with this many slots, whatever it holds."""
    return envelope_dtype(slots).itemsize


@dataclass(frozen=True, kw_only=True)
class Envelope:
    """One record's header fields and the slot words it holds; the slots after `words` are zero."""

    kind: RecordKind
    site: int
    class_id: int | None = None
    seq: int = 0
    block: int = 0
    thread: int = 0
    words: tuple[int, ...] = ()

    def to_row(self, slots: int) -> np.ndarray:
        """A one-record array of the record, with `slots` argument words; more words than slots are refused."""
        if len(self.words) > slots:
            raise ValueError(f"{len(self.words)} argument words do not fit a record with {slots} slots")
        row = np.zeros(1, dtype=envelope_dtype(slots))
        row["version"] = VERSION
        row["kind"] = self.kind
        row["flags"] = 0 if self.class_id is None else _HAS_CLASS
        row["class"] = 0 if self.class_id is None else self.class_id
        row["site"] = self.site
        row["seq"] = self.seq
        row["block"] = self.block
        row["thread"] = self.thread
        row["args"][0, : len(self.words)] = self.words
        return row

    @classmethod
    def from_row(cls, row: np.void) -> Self:
        """The header and words of one record; another version or an unknown kind is refused."""
        version = int(row["version"])
        if version != VERSION:
            raise RecordFormatError(
                f"record version {version} is not version {VERSION}, the only version this krn reads; "
                "the record was written by another krn"
            )
        try:
            kind = RecordKind(int(row["kind"]))
        except ValueError as error:
            raise RecordFormatError(f"record kind {int(row['kind'])} is neither a log nor an error record") from error
        return cls(
            kind=kind,
            site=int(row["site"]),
            class_id=int(row["class"]) if int(row["flags"]) & _HAS_CLASS else None,
            seq=int(row["seq"]),
            block=int(row["block"]),
            thread=int(row["thread"]),
            words=tuple(int(word) for word in row["args"]),
        )
