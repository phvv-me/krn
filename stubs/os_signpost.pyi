from collections.abc import Callable
from enum import Enum

# Surface used by krn, checked against os-signpost's shipped cy.pyx.
class Signposter:
    class Category(Enum):
        PointsOfInterest = "PointsOfInterest"
        DynamicTracing = "DynamicTracing"
        DynamicStackTracing = "DynamicStackTracing"

    def __init__(self, subsystem: str, category: Category | str) -> None: ...
    def begin_interval(self, msg: str) -> Callable[[str | None], None]: ...
