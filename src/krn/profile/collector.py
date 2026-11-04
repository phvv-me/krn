"""A runtime-record view, borrowing the active owner or creating one bounded session."""

from .enums import Activity
from .profiler import Profiler
from .result import Profile
from .session import Feature
from .spans import active
from .trace import KernelTrace, TraceCollector


class RuntimeCollector(TraceCollector):
    """Kernel records of the runtime hook, numbered by the owner's launch cursor.

    A window is `[since, until)` in those numbers, so retention or a reset cannot shift it onto other launches.
    """

    def __init__(self) -> None:
        self.owner: Profiler | None = None
        self.owned = False
        self.running = False
        self.origin = 0
        self.until: int | None = None

    @property
    def device_index(self) -> int:
        return self._owner().collection.device_index

    def _owner(self) -> Profiler:
        if self.owner is None:
            raise RuntimeError("enter the runtime collector before reading records")
        return self.owner

    def __enter__(self) -> RuntimeCollector:
        if self.running:
            raise RuntimeError("a runtime collector cannot be entered twice")
        self.owner = active()
        self.owned = self.owner is None
        if self.owner is None:
            self.owner = Profiler(features=Feature.ACTIVITY)
            self.owner.__enter__()
        if not self.owner.collection.features & Feature.ACTIVITY:
            raise RuntimeError("runtime collection requires a Profiler with Feature.ACTIVITY")
        self.running = True
        self.until = None
        self.origin = self.owner._cursor()
        return self

    def flush(self) -> None:
        owner = self._owner()
        if self.running and owner.active:
            owner._drain_launches()

    def checkpoint(self, activities: Activity) -> tuple[int, int]:
        if not self.running:
            raise RuntimeError("runtime checkpoints require a running collector")
        if activities & ~Activity.KERNEL:
            raise RuntimeError("runtime collector enabled only kernel activity")
        self.flush()
        owner = self._owner()
        return owner._cursor(), owner.dropped_launches

    def kernels(self, *, since: int | None = None, until: int | None = None) -> list[KernelTrace]:
        self.flush()
        owner = self._owner()
        first = max(owner.dropped_launches, self.origin) if since is None else since
        if first < self.origin:
            raise RuntimeError("runtime window was reset before retrieval")
        end = (owner._cursor() if self.until is None else self.until) if until is None else until
        return list(owner._with_launches(Profile(), owner._window(first, end)).kernels)

    def dropped(self) -> int:
        return self._owner().dropped_launches

    def reset(self) -> None:
        self.flush()
        owner = self._owner()
        self.origin = owner._cursor() if self.until is None else self.until

    def stop(self) -> None:
        if not self.running:
            return
        try:
            if self.owned:
                self._owner().__exit__()
            else:
                self.flush()
        finally:
            self.running = False
            self.until = self._owner()._cursor()
