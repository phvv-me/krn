# CUPTI's one process-wide activity subscription: it hands out buffers and takes each completed
# one back, copying its records into the collector capturing now.

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...trace import KernelTrace, MemcpyTrace
from . import bindings
from .constants import CONCURRENT_KERNEL, MEMCPY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...formats.protocols import RawActivity
    from .collector import CUPTICollector

_BUFFER_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, kw_only=True)
class RawGeneric:
    """Fields copied from one generic activity before deferred name resolution.

    It stays apart from `ActivityRecord` because it carries what only the name resolution needs:
    the activity kind and the callback id, resolved to a function name after the callback returns.
    """

    kind_id: int
    kind: str
    name: str | None
    cbid: int | None
    start_ns: int
    end_ns: int
    correlation_id: int


type Record = KernelTrace | MemcpyTrace | RawGeneric


class Delivery:
    """The process's one CUPTI buffer subscription, registered once and never unregistered.

    active: the capturing collectors, the newest receiving every completed buffer.
    labels: each enabled activity kind's friendly label, filled as kinds enable.
    domains: each API activity kind's callback domain, which names a callback id's function.
    """

    active: ClassVar[list[CUPTICollector]] = []
    labels: ClassVar[dict[int, str]] = {}
    domains: ClassVar[dict[int, int]] = {}
    registered: ClassVar[bool] = False

    @classmethod
    def completed(cls, activities: Sequence[RawActivity]) -> None:
        """Copy bounded raw records while CUPTI still owns the activity objects."""
        if not cls.active:
            return
        target = cls.active[-1]
        with target.lock:
            # Tainted until every record converts, so a failed conversion leaves the collector
            # refusing its evidence rather than handing back a silently shortened trace.
            tainted, target.callback_failed = target.callback_failed, True
            for record in [record for act in activities if (record := cls._copied(act))]:
                target.append(record)
            target.callback_failed = tainted

    @classmethod
    def register(cls) -> None:
        """Subscribe the buffer callbacks, once for the process."""
        if cls.registered:
            return
        api = bindings.cupti
        api.activity_register_callbacks(cls.requested, cls.completed)
        cls.domains[api.ActivityKind.RUNTIME] = api.CallbackDomain.RUNTIME_API
        cls.domains[api.ActivityKind.DRIVER] = api.CallbackDomain.DRIVER_API
        cls.registered = True

    @staticmethod
    def _copied(act: RawActivity) -> Record | None:
        """The record kept of one activity, None for a kind no capture enabled."""
        if act.kind == CONCURRENT_KERNEL:
            return KernelTrace.from_activity(act)
        if act.kind == MEMCPY:
            return MemcpyTrace.from_activity(act)
        if act.kind not in Delivery.labels:
            return None
        return RawGeneric(
            kind_id=act.kind,
            kind=Delivery.labels[act.kind],
            name=getattr(act, "name", None),
            cbid=getattr(act, "cbid", None),
            start_ns=act.start,
            end_ns=act.end,
            correlation_id=getattr(act, "correlation_id", 0),
        )

    @staticmethod
    def requested() -> tuple[int, int]:
        """The size of a buffer CUPTI asks for, and no record limit within it."""
        return _BUFFER_SIZE, 0
