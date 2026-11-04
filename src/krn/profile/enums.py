from enum import Enum, Flag, auto


class Bound(Enum):
    """Whether the dominant work is limited by memory traffic or compute throughput.

    UNKNOWN when there was nothing to classify: no kernels, no copies, no utilization signal.
    """

    MEMORY = "memory"
    COMPUTE = "compute"
    UNKNOWN = "unknown"


class Activity(Flag):
    """The activity kinds to trace, combined with `|`, each mapped to a vendor's native kind.

    KERNEL and MEMCPY become typed `KernelTrace`/`MemcpyTrace`, the rest `ActivityRecord`.
    DEFAULT is the minimal, low-impact set; ALL adds the high-volume runtime/driver kinds.
    """

    KERNEL = auto()
    MEMCPY = auto()
    MEMSET = auto()
    SYNC = auto()
    OVERHEAD = auto()
    MEMORY = auto()
    JIT = auto()
    RUNTIME = auto()
    DRIVER = auto()
    MEMORY_POOL = auto()
    DEFAULT = KERNEL | MEMCPY
    ALL = KERNEL | MEMCPY | MEMSET | SYNC | OVERHEAD | MEMORY | JIT | RUNTIME | DRIVER | MEMORY_POOL

    @property
    def label(self) -> str:
        """The lowercase name used in records and on the Perfetto timeline."""
        return (self.name or "activity").lower()
