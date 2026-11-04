"""The diagnostic records krn.log and krn.exceptions share: scalar tags and slot words, messages, sites and envelopes.

This package imports nothing from krn.log or krn.exceptions, so both of them depend on it and not on each other.
"""

from .envelope import VERSION, Envelope, RecordFormatError, RecordKind, envelope_dtype, envelope_size
from .jsonio import JSON
from .message import Argument, Field, Message, MessageFormatError, Spec
from .sites import MAX_SITE_ID, UNASSIGNED, Site, SiteTable
from .tags import SLOT_BYTES, WORD_MASK, Scalar, Tag, slot_value, slot_word

__all__ = [
    "JSON",
    "MAX_SITE_ID",
    "SLOT_BYTES",
    "UNASSIGNED",
    "VERSION",
    "WORD_MASK",
    "Argument",
    "Envelope",
    "Field",
    "Message",
    "MessageFormatError",
    "RecordFormatError",
    "RecordKind",
    "Scalar",
    "Site",
    "SiteTable",
    "Spec",
    "Tag",
    "envelope_dtype",
    "envelope_size",
    "slot_value",
    "slot_word",
]
