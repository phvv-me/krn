import threading
from collections import deque
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .._constants import NANOSECONDS_PER_MILLISECOND
from ..session import Feature, SpanFrame, SpanMeasurement
from ..trace import RegionWindow

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..result.models import ProcessReading
    from ..session import Collection
    from ..tracer import Tracer


class SpanRecorder:
    """Open, nest and close the spans of one profiling session, keeping what each one measured.

    tracer: marks each span on the native timeline and stamps its device-clock window.
    snapshot: reads the device when a span closes with no reading the sampler took during it.
    """

    collection: Collection
    dropped: int
    frames: dict[int, SpanFrame]
    lock: threading.Lock
    measurements: deque[SpanMeasurement]
    next_token: int
    snapshot: Callable[[str], ProcessReading | None]
    stack: ContextVar[tuple[int, ...]]
    tracer: Tracer
    windows: deque[RegionWindow]

    def __init__(
        self,
        collection: Collection,
        *,
        tracer: Tracer,
        snapshot: Callable[[str], ProcessReading | None],
    ) -> None:
        self.collection = collection
        self.tracer = tracer
        self.snapshot = snapshot
        self.stack = ContextVar("krn_spans", default=())
        self.lock = threading.Lock()
        self.next_token = 0
        self.dropped = 0
        self.frames = {}
        self.measurements = deque(maxlen=collection.max_spans)
        self.windows = deque(maxlen=collection.max_spans)

    def _measured(self, frame: SpanFrame, *, wall_ns: int) -> None:
        """Keep the closed span's wall time and device samples.

        A boundary sample stands in when the sampler took none.
        """
        samples = list(frame.samples)
        if not samples and self.collection.features & Feature.DEVICE:
            boundary = self.snapshot(frame.path)
            samples = [boundary] if boundary is not None else []
        if not (self.collection.features & Feature.SPANS or samples):
            return
        if len(self.measurements) == self.collection.max_spans:
            self.dropped += 1
        self.measurements.append(
            SpanMeasurement(
                name=frame.path,
                wall_ms=wall_ns / NANOSECONDS_PER_MILLISECOND,
                samples=tuple(samples),
            )
        )

    def _windowed(self, frame: SpanFrame, *, wall_ns: int) -> None:
        """Keep the closed span's device activity window, when this session collects activity."""
        if not self.collection.features & Feature.ACTIVITY:
            return
        if len(self.windows) == self.collection.max_spans:
            self.dropped += 1
        self.windows.append(
            RegionWindow(
                name=frame.path,
                start_ns=frame.device_start_ns,
                end_ns=self.tracer.timestamp(),
                wall_ns=wall_ns,
                clock=self.tracer.clock(),
            )
        )

    def path(self) -> str:
        """The innermost span path open on the calling context, the empty string outside every span."""
        with self.lock:
            frames = [self.frames[token] for token in self.stack.get() if token in self.frames]
        return frames[-1].path if frames else ""

    def enter(self, name: str) -> int:
        """Open one span and return the exact token later used to close it."""
        stack = self.stack.get()
        marking = self.collection.features & Feature.MARKERS
        finish_marker = self.tracer.start(name) if marking else None
        with self.lock:
            self.next_token += 1
            token = self.next_token
            parents = [self.frames[parent].name for parent in stack if parent in self.frames]
            self.frames[token] = SpanFrame(
                name=name,
                path=".".join((*parents, name)),
                thread=threading.get_ident(),
                device_start_ns=self.tracer.timestamp() if self.collection.features & Feature.ACTIVITY else 0,
                finish_marker=finish_marker,
            )
        self.stack.set((*stack, token))
        return token

    def exit(self, token: int, *, wall_ns: int) -> None:
        """Close one span and fold its timing, device samples, and activity window."""
        stack = self.stack.get()
        if token in stack:
            self.stack.set(tuple(item for item in stack if item != token))
        with self.lock:
            frame = self.frames.pop(token, None)
        if frame is None:
            return
        if frame.finish_marker is not None:
            frame.finish_marker()
        self._measured(frame, wall_ns=wall_ns)
        self._windowed(frame, wall_ns=wall_ns)
