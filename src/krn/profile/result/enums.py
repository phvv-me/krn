from enum import StrEnum, auto


class DeviceEvidence(StrEnum):
    """Whether device evidence was sought and found.

    A session that attached to no device must never read as a run that did no GPU work.

    UNSOUGHT: neither device telemetry nor GPU activity was requested.
    ABSENT: requested, but no device was visible or the profiled code never touched it.
    COLLECTED: at least one device reading, kernel, copy or activity was observed.
    """

    UNSOUGHT = auto()
    ABSENT = auto()
    COLLECTED = auto()
