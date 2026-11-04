from typing import ClassVar

from ....compiler.errors import TargetError
from ...collector import RuntimeCollector
from ...enums import Activity
from ...trace import TraceCollector
from ...tracer import Marker, Tracer, Vendor


class CPUTracer(Tracer):
    """CPU ranges use the host clock; Profiler obtains timings from the launch hook."""

    vendor: ClassVar[Vendor] = Vendor.UNKNOWN
    label: ClassVar[str] = "cpu"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def supported(self) -> Activity:
        return Activity.KERNEL

    def _open(self, kinds: Activity) -> TraceCollector:
        return RuntimeCollector()

    def start(self, name: str) -> Marker:
        raise TargetError(
            "native markers", "CPU", "there is no native CPU range provider", "use Feature.SPANS for host regions"
        )
