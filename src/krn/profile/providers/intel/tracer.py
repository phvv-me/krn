from typing import ClassVar

from ....compiler.errors import TargetError
from ....hw.libraries import Libraries
from ...enums import Activity
from ...tracer import Marker, Tracer, Vendor


class IntelTracer(Tracer):
    vendor: ClassVar[Vendor] = Vendor.INTEL
    label: ClassVar[str] = "level-zero"

    @classmethod
    def is_available(cls) -> bool:
        return Libraries.available("level_zero")

    def supported(self) -> Activity:
        raise TargetError(
            "Level Zero activity",
            "Intel",
            "the loader alone does not provide a tested activity collector",
            "profile on an NVIDIA or Apple target, or disable ACTIVITY to keep host spans",
        )

    def start(self, name: str) -> Marker:
        Libraries.load("level_zero")
        raise TargetError(
            "Level Zero ranges",
            "Intel",
            "a tested tracing adapter is not implemented",
            "disable MARKERS to keep host spans, or profile on an NVIDIA or Apple target",
        )
