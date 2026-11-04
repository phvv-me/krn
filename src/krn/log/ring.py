"""The ring one launch writes: a fixed number of records, and a head counter that counts every call, kept or not."""

from dataclasses import dataclass
from functools import cached_property

from .._internals.records import envelope_size


@dataclass(frozen=True, kw_only=True)
class Ring:
    """Where a launch of `kernel` writes: `capacity` records of `slots` argument slots each, and the kernel's source.

    The device adds one to the head counter for every call, and a call past the capacity writes nothing. So the host
    reads the dropped count as the head minus the records kept, and never loses a drop silently.
    """

    kernel: str
    file: str
    line: int
    capacity: int
    slots: int

    @cached_property
    def record_bytes(self) -> int:
        return envelope_size(self.slots)
