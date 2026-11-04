# macOS annotation backend: `os_signpost` intervals, shown in Instruments. The module imports only
# where `os_signpost` does, and the providers package skips it otherwise.

from typing import ClassVar

from ....compiler.errors import TargetError
from ....hw.platforms import system_key
from ...collector import RuntimeCollector
from ...enums import Activity
from ...trace import TraceCollector
from ...tracer import Marker, Tracer, Vendor

try:
    from os_signpost import Signposter
except ImportError:
    Signposter = None

_SUBSYSTEM = "org.krn.profile"


class SignpostTracer(Tracer):
    """`os_signpost` intervals."""

    vendor: ClassVar[Vendor] = Vendor.APPLE
    label: ClassVar[str] = "signpost"

    @classmethod
    def is_available(cls) -> bool:
        return system_key() == "Darwin"

    def __init__(self) -> None:
        self._signposter = None if Signposter is None else Signposter(_SUBSYSTEM, Signposter.Category.PointsOfInterest)

    def start(self, name: str) -> Marker:
        """Open a correlatable signpost interval and return its exact closer."""
        if self._signposter is None:
            raise TargetError(
                "os_signpost ranges",
                "Apple",
                "os-signpost is unavailable",
                'install "krn[perf]"; runtime Metal timings need no extra',
            )
        close = self._signposter.begin_interval(name)
        return lambda: close(None)

    def supported(self) -> Activity:
        return Activity.KERNEL

    def _open(self, kinds: Activity) -> TraceCollector:
        return RuntimeCollector()
