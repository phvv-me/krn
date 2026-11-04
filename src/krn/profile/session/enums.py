from enum import Flag, auto


class Feature(Flag):
    """Independent collection costs that may be combined with `|`.

    DEFAULT enables every member. Collectors that require replay belong in a
    separate pass, because they change what concurrent collectors observe.
    """

    # Bit 1 named an unimplemented collector; preserve the recorded values.
    SPANS = 2
    DEVICE = auto()
    MARKERS = auto()
    ACTIVITY = auto()
    DEFAULT = SPANS | DEVICE | MARKERS | ACTIVITY
