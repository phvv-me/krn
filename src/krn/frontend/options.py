"""Internal heuristic decision modes."""

from enum import StrEnum, auto


class Mode(StrEnum):
    AUTO = auto()
    ALWAYS = auto()
    NEVER = auto()
