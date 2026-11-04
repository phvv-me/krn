# Vendor code-annotation backends: named timeline ranges.

import importlib
import time
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ..compiler.errors import TargetError
from ..hw.enums import Vendor
from .enums import Activity
from .trace import TraceCollector

if TYPE_CHECKING:
    from collections.abc import Callable

type Marker = Callable[[], None]


class Tracer(Registry):
    """No-op annotation backend and the registry root for vendor tracers.

    label: short identifier for reports.
    """

    vendor: ClassVar[Vendor] = Vendor.UNKNOWN
    label: ClassVar[str] = "none"

    @classmethod
    def detect(cls, *, present: frozenset[Vendor] = frozenset()) -> Tracer:
        """A strategy matching a present vendor, or the documented empty base strategy.

        Missing libraries are reported when their capability is requested.
        """
        importlib.import_module("krn.profile.providers")
        backends = list(cls.implementations())
        for backend in backends:
            if backend.vendor in present:
                return backend()
        return cls()

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's annotation library can be imported here."""
        return False

    def _open(self, kinds: Activity) -> TraceCollector:
        """Build the collector for already-resolved `kinds`."""
        return TraceCollector()

    def _resolve(self, kinds: Activity) -> Activity:
        """Require every requested kind, including ALL; unsupported kinds raise TargetError."""
        supported = self.supported()
        if not supported:
            raise TargetError(
                "activity collection",
                self.label,
                "this backend has no native collector",
                "use Profiler(native=False), which keeps runtime launch records, or another target",
            )
        if missing := kinds & ~supported:
            raise TargetError(
                str(missing),
                self.label,
                f"available activity kinds: {supported}",
                "request only the supported kinds, or use Profiler(native=False) for runtime launch records",
            )
        return kinds

    def collect(self, kinds: Activity = Activity.DEFAULT) -> TraceCollector:
        """A deep per-op trace collector for `kinds`, as `_resolve` reconciles them."""
        return self._open(self._resolve(kinds))

    def start(self, name: str) -> Marker:
        """Open a native range and return the exact operation that closes it; none in the base."""
        return lambda: None

    def supported(self) -> Activity:
        """The kinds this backend can collect here; none in the base, so `collect` refuses.

        Support is device- and driver-specific (consumer GPUs lack some CUPTI kinds), so a
        backend probes the hardware.
        """
        return Activity(0)

    def timestamp(self) -> int:
        """Device-clock nanoseconds for region binning; the host clock in the base."""
        return time.perf_counter_ns()

    def clock(self) -> str:
        return "host"
