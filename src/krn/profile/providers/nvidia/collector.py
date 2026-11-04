# The CUPTI Activity collector: one process-wide subscription whose buffer callback copies
# bounded raw records into the active collector, retrieved by window after a synchronized flush.

import sys
import threading
from collections import deque
from contextlib import ExitStack
from ctypes import addressof, c_size_t
from itertools import islice
from typing import TYPE_CHECKING

from ...enums import Activity
from ...trace import ActivityRecord, KernelTrace, MemcpyTrace, TraceCollector
from . import bindings
from .constants import CUPTI_KIND
from .delivery import Delivery, RawGeneric, Record

if TYPE_CHECKING:
    from collections.abc import Sequence

_MAX_RECORDS = 262_144


def _disable(kinds: Sequence[int]) -> None:
    """Disable exactly the native activity kinds enabled for one capture."""
    api = bindings.cupti
    cleanup = ExitStack()
    for kind in kinds:
        cleanup.callback(api.activity_disable, kind)
    # An inner ``with`` would hide the active work/start error from this stack.
    cleanup.__exit__(*sys.exc_info())


def _synchronize() -> None:
    """Synchronize the device so all kernels land in the CUPTI buffer before a flush."""
    if not bindings.has_cuda:
        raise RuntimeError("CUDA runtime binding is required to synchronize activity capture")
    (status,) = bindings.runtime.cudaDeviceSynchronize()
    if status != bindings.runtime.cudaError_t.cudaSuccess:  # asynchronous errors surface here too
        raise RuntimeError(f"cudaDeviceSynchronize failed with CUDA status {status}")


class CUPTICollector(TraceCollector):
    """Asynchronous CUPTI Activity collector — single-subscriber, low overhead.

    kinds: the :class:`Activity` flags to collect. ``KERNEL``/``MEMCPY`` become typed
    records; the rest become generic :class:`ActivityRecord`s.
    """

    # Declared here since `__enter__`, read before `__init__`, assigns them too.
    enabled_kinds: tuple[int, ...]
    scope: tuple[int, int, int] | None

    @property
    def device_index(self) -> int:
        if self.scope is None:
            raise RuntimeError("activity capture has no current CUDA context")
        return self.scope[0]

    @staticmethod
    def _enable(kinds: Activity) -> tuple[int, ...]:
        """Enable requested CUPTI kinds and return the exact native members enabled.

        ``kinds`` is already reconciled against :func:`_supported`, so every kind here is
        known to enable; an error would be a real bug, not an unsupported device.
        """
        api = bindings.cupti
        enabled: list[int] = []
        with ExitStack() as rollback:
            rollback.callback(_disable, enabled)
            for flag, enum_name in CUPTI_KIND.items():
                if flag not in kinds:
                    continue
                kind = getattr(api.ActivityKind, enum_name)
                api.activity_enable(kind)
                enabled.append(kind)
                Delivery.labels[int(kind)] = flag.label
            rollback.pop_all()
        return tuple(enabled)

    @staticmethod
    def _native_drops() -> int:
        """Drain CUPTI's reset-on-read global queue loss count, including empty flushes.

        CUDA 6 onward delivers global buffers, so context and stream are both zero.
        cupti-python takes the address of a size_t output, not an integer result.
        """
        dropped = c_size_t()
        bindings.cupti.activity_get_num_dropped_records(0, 0, addressof(dropped))
        return dropped.value

    @staticmethod
    def _scope() -> tuple[int, int, int]:
        """Actual visible ordinal, CUPTI device ID, and CUPTI context ID; no substitution."""
        if not bindings.has_cuda:
            raise RuntimeError("CUDA runtime is required to identify an activity window")
        status, device = bindings.runtime.cudaGetDevice()
        if status != 0:
            raise RuntimeError(f"cudaGetDevice failed with CUDA status {status}")
        result, context = bindings.driver.cuCtxGetCurrent()
        if result != 0 or not int(context):
            raise RuntimeError(f"cuCtxGetCurrent failed or returned no context: {result}")
        cupti = bindings.cupti
        return device, cupti.get_device_id(int(context)), cupti.get_context_id(int(context))

    @staticmethod
    def activity_name(record: RawGeneric) -> str:
        """Resolve an API function name after the CUPTI callback has returned."""
        if record.name:
            return record.name
        domain = Delivery.domains.get(record.kind_id)
        if record.cbid is not None and domain is not None and bindings.has_cupti:
            return bindings.cupti.get_callback_name(domain, record.cbid)
        return record.kind

    def __enter__(self) -> CUPTICollector:
        Delivery.register()
        if Delivery.active:
            raise RuntimeError("nested CUPTI collection unsupported (single-subscriber)")
        self.enabled_kinds = CUPTICollector._enable(self.kinds)
        self.owner_thread = threading.get_ident()
        with ExitStack() as undo:
            undo.callback(self._settle)
            _synchronize()
            self.scope = self._scope()
            bindings.cupti.activity_flush_all(1)  # drain prior records before capture starts
            CUPTICollector._native_drops()  # reset the global count from before this capture
            Delivery.active.append(self)
            self.running = True
            undo.pop_all()
        return self

    def __init__(self, kinds: Activity = Activity.DEFAULT) -> None:
        self.lock = threading.Lock()
        self.kinds = kinds
        self.records: deque[Record] = deque(maxlen=_MAX_RECORDS)
        self.dropped_records = self.native_dropped_records = 0
        self.received_records = self.lost_records = 0
        self.enabled_kinds = ()
        self.running = self.callback_failed = False
        self.scope = None
        self.owner_thread = threading.get_ident()

    def _checked(self, records: tuple[Record, ...]) -> tuple[Record, ...]:
        """`records`, refused when their GPU work is foreign to the context or incomplete.

        Global CUPTI delivery is wider than one context, so foreign or unidentifiable work is
        never filtered out silently and the remainder called complete.
        """
        for record in records:
            if not isinstance(record, (KernelTrace, MemcpyTrace)):
                continue
            if self.scope is None or (record.device_id, record.context_id) != self.scope[1:]:
                raise RuntimeError("activity window contains a foreign or unknown CUDA context")
            if record.stream_id is None or record.start_ns <= 0 or record.end_ns < record.start_ns:
                raise RuntimeError("activity window contains incomplete native timing or stream data")
        return records

    def _mark_stopped(self) -> None:
        """Forget the enables and read as not running."""
        self.enabled_kinds = ()
        self.running = False

    def _records(self, *, since: int | None, until: int | None) -> tuple[Record, ...]:
        """Copy only one delivered range; never re-materialize the entire outer trace."""
        with self.lock:
            if since is None:
                return self._checked(tuple(self.records))
            end = self.received_records if until is None else until
            first = self.received_records - len(self.records)
            if not first <= since <= end <= self.received_records:
                raise RuntimeError("activity window was reset or overwritten before retrieval")
            records = tuple(
                islice(
                    reversed(self.records),
                    self.received_records - end,
                    self.received_records - since,
                )
            )[::-1]
        return self._checked(records)

    def _retire(self) -> None:
        """Leave the active stack, tolerating a session someone else already popped."""
        if Delivery.active and Delivery.active[-1] is self:
            Delivery.active.pop()

    def _settle(self) -> None:
        """Disable what this session enabled and mark it stopped, whatever else failed."""
        with ExitStack() as rest:
            rest.callback(self._mark_stopped)
            _disable(self.enabled_kinds)

    def activities(self) -> list[ActivityRecord]:
        records = (record for record in self._records(since=None, until=None) if isinstance(record, RawGeneric))
        return [
            ActivityRecord(
                kind=record.kind,
                name=self.activity_name(record),
                start_ns=record.start_ns,
                end_ns=record.end_ns,
                correlation_id=record.correlation_id,
                clock="cupti",
            )
            for record in records
        ]

    def append(self, record: Record) -> None:
        """Append one raw record, evicting the oldest when the buffer is full.

        An eviction is not a loss of a window's own records: `_records` refuses a window whose start was evicted.
        """
        if len(self.records) == self.records.maxlen:
            self.dropped_records += 1
        self.records.append(record)
        self.received_records += 1

    def checkpoint(self, activities: Activity) -> tuple[int, int]:
        """Synchronize every stream in the captured context and freeze a raw cursor.

        The first counter is monotone across reset(); the second is CUPTI's native loss, which a window
        cannot recover, so a window whose native loss changed is refused.
        """
        if not self.running:
            raise RuntimeError("activity checkpoints require a running collector")
        enabled = Activity(0)
        for flag, name in CUPTI_KIND.items():
            if getattr(bindings.cupti.ActivityKind, name) in self.enabled_kinds:
                enabled |= flag
        if activities & ~enabled:
            raise RuntimeError("activity window requested kinds not actually enabled")
        self.flush()
        with self.lock:
            return self.received_records, self.lost_records

    def dropped(self) -> int:
        """Return native buffer loss plus deque overwrites through the latest flush."""
        with self.lock:
            return self.native_dropped_records + self.dropped_records

    def flush(self) -> None:
        if self.scope is not None and (threading.get_ident() != self.owner_thread or self._scope() != self.scope):
            raise RuntimeError("activity capture changed its issuing thread or CUDA context")
        _synchronize()
        bindings.cupti.activity_flush_all(1)
        dropped = CUPTICollector._native_drops()
        with self.lock:
            self.native_dropped_records += dropped
            self.lost_records += dropped
            if self.callback_failed:
                raise RuntimeError("activity collector tainted by callback conversion failure")

    def kernels(self, *, since: int | None = None, until: int | None = None) -> list[KernelTrace]:
        return [record for record in self._records(since=since, until=until) if isinstance(record, KernelTrace)]

    def memcpys(self, *, since: int | None = None, until: int | None = None) -> list[MemcpyTrace]:
        return [record for record in self._records(since=since, until=until) if isinstance(record, MemcpyTrace)]

    def reset(self) -> None:
        self.flush()  # drain in-flight records, then drop everything so far
        with self.lock:
            self.records.clear()
            self.dropped_records = 0
            self.native_dropped_records = 0

    def stop(self) -> None:
        if not self.running:
            return
        teardown = ExitStack()
        teardown.callback(self._settle)
        teardown.callback(self._retire)
        try:
            self.flush()
        finally:
            teardown.__exit__(*sys.exc_info())
